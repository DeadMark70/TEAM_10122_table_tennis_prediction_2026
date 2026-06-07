"""V231 action-only sequence teacher.

This script trains small causal sequence models for actionId only.  Point is
only auxiliary/context and is never exported as a submission target.  The
generated submissions keep V188 cap5 point and R121 server fixed.
"""

from __future__ import annotations

import __main__
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from analysis_v230_action_soft_teacher_factory import ACTION_FAMILY_TO_IDS, normalize_rows_safe, public_like_slice_score
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor, rebuild_r166_best_action
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v231_action_only_sequence_teacher")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/train_v231_action_only_sequence_teacher.py")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
EPOCHS = 3
PATIENCE = 1
WEAK_CLASSES = [0, 3, 5, 7, 8, 9, 12, 14]


@dataclass
class ActionBatch:
    strokes: torch.Tensor
    lengths: torch.Tensor
    static: torch.Tensor
    action: torch.Tensor
    teacher: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_family_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    out = np.zeros(len(y), dtype=np.int64)
    for fam_id, fam in enumerate(["Zero", "Attack", "Control", "Defensive", "Serve"]):
        for action_id in ACTION_FAMILY_TO_IDS[fam]:
            out[y == int(action_id)] = fam_id
    return out


def weak_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    return np.vstack([(y == cls).astype(float) for cls in WEAK_CLASSES]).T.astype(np.float32)


def action_only_loss(logits: dict, y: np.ndarray, kd_teacher: np.ndarray | None = None) -> float:
    """Numpy-friendly diagnostic loss used by tests."""
    action_logits = np.asarray(logits["action"], dtype=float)
    shifted = action_logits - action_logits.max(axis=1, keepdims=True)
    prob = normalize_rows_safe(np.exp(shifted))
    ce = -np.log(np.clip(prob[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-9, 1.0)).mean()
    fam_logits = np.asarray(logits.get("family", np.zeros((len(y), 5))), dtype=float)
    fam_prob = normalize_rows_safe(np.exp(fam_logits - fam_logits.max(axis=1, keepdims=True)))
    fam_y = action_family_targets(y)
    fam_ce = -np.log(np.clip(fam_prob[np.arange(len(y)), fam_y], 1e-9, 1.0)).mean()
    total = ce + 0.30 * fam_ce
    if kd_teacher is not None:
        total += 0.10 * float(np.mean((prob - normalize_rows_safe(kd_teacher)) ** 2))
    return float(total)


class SeqDataset(Dataset):
    def __init__(self, strokes: np.ndarray, lengths: np.ndarray, static: np.ndarray, action: np.ndarray, teacher: np.ndarray):
        self.strokes = torch.as_tensor(strokes, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.action = torch.as_tensor(action.astype(np.int64), dtype=torch.long)
        self.teacher = torch.as_tensor(teacher.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int) -> ActionBatch:
        return ActionBatch(self.strokes[idx], self.lengths[idx], self.static[idx], self.action[idx], self.teacher[idx])


def collate(batch: list[ActionBatch]) -> ActionBatch:
    return ActionBatch(
        torch.stack([b.strokes for b in batch]),
        torch.stack([b.lengths for b in batch]),
        torch.stack([b.static for b in batch]),
        torch.stack([b.action for b in batch]),
        torch.stack([b.teacher for b in batch]),
    )


class ActionSeqModel(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, mode: str, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.mode = mode
        self.embeddings = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in vocab_sizes])
        in_dim = emb_dim * len(vocab_sizes)
        if mode == "lstm":
            self.rnn = nn.LSTM(in_dim, hidden, batch_first=True)
            enc_dim = hidden
        elif mode == "transformer":
            self.proj = nn.Linear(in_dim, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=128, dropout=0.10, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            enc_dim = hidden
        else:
            self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
            enc_dim = hidden
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.10))
        self.shared = nn.Sequential(nn.Linear(enc_dim + 64, 128), nn.ReLU(), nn.Dropout(0.15))
        self.action = nn.Linear(128, 19)
        self.family = nn.Linear(128, 5)
        self.weak = nn.Linear(128, len(WEAK_CLASSES))

    def encode(self, strokes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb(strokes[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=2)
        if self.mode == "transformer":
            h = self.encoder(self.proj(x))
            idx = (lengths - 1).clamp_min(0)
            return h[torch.arange(len(h), device=h.device), idx]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        return h[-1]

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_h = self.encode(strokes, lengths)
        static_h = self.static_net(static)
        h = self.shared(torch.cat([seq_h, static_h], dim=1))
        return {"action": self.action(h), "family": self.family(h), "weak": self.weak(h)}


def torch_action_loss(out: dict[str, torch.Tensor], y: torch.Tensor, teacher: torch.Tensor, kd_weight: float) -> torch.Tensor:
    family_y = torch.as_tensor(action_family_targets(y.detach().cpu().numpy()), device=y.device, dtype=torch.long)
    weak_y = torch.as_tensor(weak_targets(y.detach().cpu().numpy()), device=y.device, dtype=torch.float32)
    loss = F.cross_entropy(out["action"], y)
    loss = loss + 0.30 * F.cross_entropy(out["family"], family_y)
    loss = loss + 0.15 * F.binary_cross_entropy_with_logits(out["weak"], weak_y)
    if kd_weight > 0:
        logp = F.log_softmax(out["action"], dim=1)
        loss = loss + float(kd_weight) * F.kl_div(logp, teacher, reduction="batchmean")
    return loss


def predict_prob(model: ActionSeqModel, dataset: SeqDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            probs.append(F.softmax(out["action"], dim=1).cpu().numpy())
    return normalize_rows_safe(np.vstack(probs))


def train_model(train_ds: SeqDataset, valid_ds: SeqDataset, vocab_sizes: list[int], static_dim: int, mode: str, kd_weight: float, seed: int) -> ActionSeqModel:
    set_seed(seed)
    model = ActionSeqModel(vocab_sizes, static_dim, mode=mode).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = torch_action_loss(out, batch.action.to(DEVICE), batch.teacher.to(DEVICE), kd_weight=kd_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        val_prob = predict_prob(model, valid_ds)
        val_loss = -np.log(np.clip(val_prob[np.arange(len(valid_ds.action)), valid_ds.action.numpy()], 1e-9, 1.0)).mean()
        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, rows: pd.DataFrame) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "public_like_action_macro_f1": public_like_slice_score(y, pred, rows),
        "public_like_delta_vs_v173": public_like_slice_score(y, pred, rows) - public_like_slice_score(y, anchor, rows),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
        "action_distribution": json.dumps(pd.Series(pred).value_counts().sort_index().to_dict(), sort_keys=True),
    }


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    y_full = data["full_pool"]["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    _r166_oof, _r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    teacher_oof = normalize_rows_safe(0.70 * v173_prob_oof + 0.30 * r166_prob_oof)
    teacher_test = normalize_rows_safe(0.70 * v173_prob_test + 0.30 * r166_prob_test)
    full_teacher = np.zeros((len(y_full), 19), dtype=float)
    full_teacher[np.arange(len(y_full)), y_full] = 1.0
    schemes = [
        ("v231_gru_raw_action", "gru", 0.0),
        ("v231_lstm_raw_action", "lstm", 0.0),
        ("v231_transformer_raw_action", "transformer", 0.0),
        ("v231_gru_v173kd_action", "gru", 0.12),
        ("v231_multistep_action", "lstm", 0.08),
    ]
    records = [
        {
            "candidate": "v173_anchor",
            "action_macro_f1": float(f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0)),
            "delta_vs_v173_anchor": 0.0,
            "public_like_action_macro_f1": public_like_slice_score(y, v173_oof, data["rows"]),
            "public_like_delta_vs_v173": 0.0,
            "action_churn_vs_v173_anchor": 0.0,
            "changed_rows": 0,
        }
    ]
    generated = []
    fold_metrics = []
    for name, mode, kd_weight in schemes:
        oof_prob = np.zeros((len(y), 19), dtype=float)
        fold_test = []
        for fold in sorted(data["rows"]["fold"].astype(int).unique()):
            valid = data["rows"]["fold"].astype(int).eq(int(fold)).to_numpy()
            train = ~valid
            train_ds = SeqDataset(data["oof_seq"][train], data["oof_len"][train], data["x_oof"][train], y[train], teacher_oof[train])
            valid_ds = SeqDataset(data["oof_seq"][valid], data["oof_len"][valid], data["x_oof"][valid], y[valid], teacher_oof[valid])
            test_ds = SeqDataset(data["test_seq"], data["test_len"], data["x_test_fullstats"], v173_test, teacher_test)
            model = train_model(train_ds, valid_ds, data["vocab_sizes"], data["x_oof"].shape[1], mode, kd_weight, 2310 + int(fold))
            oof_prob[valid] = predict_prob(model, valid_ds)
            fold_test.append(predict_prob(model, test_ds))
            fold_metrics.append({"candidate": name, "fold": int(fold), "mode": mode, "kd_weight": kd_weight})
        test_prob = normalize_rows_safe(np.mean(fold_test, axis=0))
        pred = oof_prob.argmax(axis=1).astype(int)
        test_pred = test_prob.argmax(axis=1).astype(int)
        rec = evaluate(name, y, pred, v173_oof, data["rows"])
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", oof_prob)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", test_prob)
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "public_like_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v231_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v231_fold_metrics.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": ["V231 trains action-only sequence models; point/server are fixed.", "No TTMATCH and no old-server labels are read."],
    }
    (OUTDIR / "v231_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v231_report.md").write_text(
        "# V231 Action-Only Sequence Teacher\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("train_v231_action_only_sequence_teacher.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
