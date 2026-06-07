"""V285 Windows-compatible Mamba-lite action sequence teacher.

This script trains small action-only causal sequence teachers without depending
on mamba-ssm.  Final submissions keep V261 cap1 point/server fixed and only
probe action changes against the V173 action anchor embedded in that file.
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

from analysis_v230_action_soft_teacher_factory import ACTION_FAMILY_TO_IDS, public_like_slice_score
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    GrUTuning,
    TransformerTuning,
    V3Tuning,
    distill_v173_soft_anchor,
    rebuild_r166_best_action,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v285_mambalite_sequence_teacher")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/train_v285_mambalite_sequence_teacher.py")
POINT_SERVER_ANCHOR = UPLOAD_DIR / "submission_v261_cap0p01__v173action_r121server.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
EPOCHS = 4
PATIENCE = 1
WEAK_CLASSES = [0, 3, 5, 7, 8, 9, 12, 14]


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    sums = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, sums, out=np.full_like(arr, 1.0 / arr.shape[1], dtype=float), where=sums > 0)


def geometric_logit_blend(anchor_prob: np.ndarray, teacher_prob: np.ndarray, weight: float, eps: float = 1e-8) -> np.ndarray:
    anchor = np.clip(normalize_rows_safe(anchor_prob), eps, 1.0)
    teacher = np.clip(normalize_rows_safe(teacher_prob), eps, 1.0)
    logp = (1.0 - float(weight)) * np.log(anchor) + float(weight) * np.log(teacher)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


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


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.zeros((len(labels), n_classes), dtype=float)
    out[np.arange(len(labels)), labels] = 1.0
    return out


@dataclass
class ActionBatch:
    strokes: torch.Tensor
    lengths: torch.Tensor
    static: torch.Tensor
    action: torch.Tensor
    teacher: torch.Tensor


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


class MambaLiteBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=self.kernel_size, groups=dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("MambaLiteBlock expects [batch, seq_len, dim]")
        conv_in = x.transpose(1, 2)
        conv_in = F.pad(conv_in, (self.kernel_size - 1, 0))
        conv = self.depthwise(conv_in).transpose(1, 2)
        state = torch.zeros(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
        states = []
        for t in range(x.shape[1]):
            conv_t = conv[:, t, :]
            gate_t = torch.sigmoid(self.gate_proj(conv_t))
            cand_t = torch.tanh(self.value_proj(conv_t))
            state = gate_t * state + (1.0 - gate_t) * cand_t
            states.append(state)
        rec = torch.stack(states, dim=1)
        return self.norm(x + self.out_proj(rec))


class ActionMambaLiteModel(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in vocab_sizes])
        in_dim = emb_dim * len(vocab_sizes)
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([MambaLiteBlock(hidden, kernel_size=3), MambaLiteBlock(hidden, kernel_size=5)])
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.10))
        self.shared = nn.Sequential(nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(0.15))
        self.action = nn.Linear(128, 19)
        self.family = nn.Linear(128, 5)
        self.weak = nn.Linear(128, len(WEAK_CLASSES))

    def encode(self, strokes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb(strokes[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=2)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        idx = (lengths - 1).clamp_min(0)
        return h[torch.arange(len(h), device=h.device), idx]

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


def predict_prob(model: ActionMambaLiteModel, dataset: SeqDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            probs.append(F.softmax(out["action"], dim=1).cpu().numpy())
    return normalize_rows_safe(np.vstack(probs))


def train_model(
    train_ds: SeqDataset,
    valid_ds: SeqDataset,
    vocab_sizes: list[int],
    static_dim: int,
    kd_weight: float,
    seed: int,
) -> ActionMambaLiteModel:
    set_seed(seed)
    model = ActionMambaLiteModel(vocab_sizes, static_dim, emb_dim=8, hidden=64).to(DEVICE)
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
        if val_loss + 1e-6 < best_loss:
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


def distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=19)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def write_submission(name: str, action: np.ndarray, anchor: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": anchor["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": anchor["pointId"].astype(int),
            "serverGetPoint": anchor["serverGetPoint"].astype(float),
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
    public_score = public_like_slice_score(y, pred, rows)
    public_base = public_like_slice_score(y, anchor, rows)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "public_like_action_macro_f1": float(public_score),
        "public_like_delta_vs_v173": float(public_score - public_base),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
        "action_distribution": json.dumps(distribution(pred), sort_keys=True),
    }


def anchor_record(y: np.ndarray, v173_oof: np.ndarray, rows: pd.DataFrame, v173_test: np.ndarray) -> dict:
    return {
        "candidate": "v173_anchor",
        "action_macro_f1": float(f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0)),
        "delta_vs_v173_anchor": 0.0,
        "public_like_action_macro_f1": float(public_like_slice_score(y, v173_oof, rows)),
        "public_like_delta_vs_v173": 0.0,
        "action_churn_vs_v173_anchor": 0.0,
        "changed_rows": 0,
        "test_churn_vs_v173": 0.0,
        "test_changed_rows": 0,
        "action_distribution": json.dumps(distribution(v173_oof), sort_keys=True),
        "test_action_distribution": json.dumps(distribution(v173_test), sort_keys=True),
    }


def add_test_metrics(rec: dict, test_pred: np.ndarray, v173_test: np.ndarray) -> dict:
    out = rec.copy()
    out["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
    out["test_changed_rows"] = int(np.sum(test_pred != v173_test))
    out["test_action_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
    return out


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)

    data = prepare_data()
    state = rebuild_v173_best_actions()
    anchor = pd.read_csv(POINT_SERVER_ANCHOR)
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = anchor["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    _r166_oof, _r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    teacher_oof = normalize_rows_safe(0.70 * v173_prob_oof + 0.30 * r166_prob_oof)
    teacher_test = normalize_rows_safe(0.70 * v173_prob_test + 0.30 * r166_prob_test)

    schemes = [
        ("v285_mambalite_raw_action", 0.0),
        ("v285_mambalite_v173kd_action", 0.12),
    ]
    records = [anchor_record(y, v173_oof, data["rows"], v173_test)]
    generated = []
    fold_metrics = []
    scheme_probs = {}

    for name, kd_weight in schemes:
        oof_prob = np.zeros((len(y), 19), dtype=float)
        fold_test = []
        for fold in sorted(data["rows"]["fold"].astype(int).unique()):
            valid = data["rows"]["fold"].astype(int).eq(int(fold)).to_numpy()
            train = ~valid
            train_ds = SeqDataset(data["oof_seq"][train], data["oof_len"][train], data["x_oof"][train], y[train], teacher_oof[train])
            valid_ds = SeqDataset(data["oof_seq"][valid], data["oof_len"][valid], data["x_oof"][valid], y[valid], teacher_oof[valid])
            test_ds = SeqDataset(data["test_seq"], data["test_len"], data["x_test_fullstats"], v173_test, teacher_test)
            model = train_model(train_ds, valid_ds, data["vocab_sizes"], data["x_oof"].shape[1], kd_weight, 2850 + int(fold))
            oof_prob[valid] = predict_prob(model, valid_ds)
            fold_test.append(predict_prob(model, test_ds))
            fold_metrics.append({"candidate": name, "fold": int(fold), "kd_weight": float(kd_weight)})
        test_prob = normalize_rows_safe(np.mean(fold_test, axis=0))
        pred = oof_prob.argmax(axis=1).astype(int)
        test_pred = test_prob.argmax(axis=1).astype(int)
        records.append(add_test_metrics(evaluate(name, y, pred, v173_oof, data["rows"]), test_pred, v173_test))
        generated.append(write_submission(f"submission_{name}__pv261cap1__sr121.csv", test_pred, anchor))
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", oof_prob)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", test_prob)
        scheme_probs[name] = (oof_prob, test_prob)

    blend_specs = [
        ("v285_mambalite_raw_logblend_w0p05", "v285_mambalite_raw_action", 0.05),
        ("v285_mambalite_v173kd_logblend_w0p05", "v285_mambalite_v173kd_action", 0.05),
        ("v285_mambalite_v173kd_logblend_w0p10", "v285_mambalite_v173kd_action", 0.10),
    ]
    for blend_name, source_name, weight in blend_specs:
        oof_prob, test_prob = scheme_probs[source_name]
        blended_oof = geometric_logit_blend(v173_prob_oof, oof_prob, weight)
        blended_test = geometric_logit_blend(v173_prob_test, test_prob, weight)
        pred = blended_oof.argmax(axis=1).astype(int)
        test_pred = blended_test.argmax(axis=1).astype(int)
        records.append(add_test_metrics(evaluate(blend_name, y, pred, v173_oof, data["rows"]), test_pred, v173_test))
        generated.append(write_submission(f"submission_{blend_name}__pv261cap1__sr121.csv", test_pred, anchor))

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "public_like_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v285_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v285_fold_metrics.csv", index=False)

    non_anchor = search[search["candidate"].ne("v173_anchor")]
    best = non_anchor.iloc[0].to_dict() if not non_anchor.empty else {}
    best_delta = float(best.get("delta_vs_v173_anchor", float("-inf")))
    best_public_delta = float(best.get("public_like_delta_vs_v173", float("-inf")))
    verdict = (
        "LOCAL_POSITIVE_CONSIDER_PROBE"
        if best_delta >= 0.003 and best_public_delta >= 0.001
        else "LOCAL_NEGATIVE_DO_NOT_UPLOAD"
    )
    report = {
        "verdict": verdict,
        "best": best,
        "generated": generated,
        "outdir": str(OUTDIR),
        "fixed_anchor": str(POINT_SERVER_ANCHOR),
        "notes": [
            "No mamba-ssm dependency; MambaLiteBlock uses causal depthwise conv plus gated recurrence.",
            "Final pointId and serverGetPoint are copied unchanged from the V261 cap1 anchor.",
            "No TTMATCH and no old-server labels are read by this script.",
        ],
    }
    (OUTDIR / "v285_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v285_report.md").write_text(
        "# V285 Mamba-lite Sequence Teacher\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best candidate: `{best.get('candidate', '')}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Public-like delta: `{best_public_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n"
        f"- Fixed anchor: `{POINT_SERVER_ANCHOR}`\n",
        encoding="utf-8",
    )
    shutil.copy2("train_v285_mambalite_sequence_teacher.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
