"""V189/V190 point intent sequence models.

V189 tests LSTM backbones because the official baseline uses LSTM.  V190 tests
small causal Transformer backbones.  Both reuse the V188 point-intent objective:
exact AI CUP pointId supervision for the final head, auxiliary intent heads,
optional R186 coarse teacher only for intermediate heads, and low-churn residual
submission export on top of the V173/R119/R121 no-old anchor.

TTMATCH is not read.
"""

from __future__ import annotations

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
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from analysis_v188_point_intent_gru import (
    ALPHAS,
    BATCH_SIZE,
    CHURN_CAPS,
    DEVICE,
    MAX_SEQ_LEN,
    R186_TEST,
    R186_TRAIN,
    V188Batch,
    batch_loss,
    build_padded_stroke_tensor,
    capped_residual_labels,
    load_pickle,
    raw_groups,
    row_log_blend,
    sequences_for_rows,
    static_matrix,
    teacher_matrix,
    StrokeDataset,
)
from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot, point_pred
from analysis_r187_point_intent_student import add_r186_priors
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, prepare_prefix_features
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v189_v190_point_intent_seq_models")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v189_v190_point_intent_seq_models.py")

EPOCHS = 10
PATIENCE = 2
LOSS_R186_W005 = {"terminal": 0.20, "depth": 0.20, "side": 0.10, "safety": 0.05, "width_teacher": 0.05, "r186": 0.05}


@dataclass
class ModelConfig:
    name: str
    family: str
    hidden: int = 96
    layers: int = 1
    dropout: float = 0.10
    d_model: int = 64
    heads: int = 4


CONFIGS = [
    ModelConfig("v189a_lstm1_h96", "lstm", hidden=96, layers=1, dropout=0.10),
    ModelConfig("v189b_lstm2_h96_do10", "lstm", hidden=96, layers=2, dropout=0.10),
    ModelConfig("v189c_lstm1_h128", "lstm", hidden=128, layers=1, dropout=0.10),
    ModelConfig("v190a_tf_d64_l2_h4", "transformer", d_model=64, layers=2, heads=4, dropout=0.20),
    ModelConfig("v190b_tf_d96_l2_h4", "transformer", d_model=96, layers=2, heads=4, dropout=0.20),
]


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsequent_mask(size: int) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


def causal_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    pos = torch.arange(max_len, device=lengths.device)[None, :]
    return pos >= lengths[:, None]


class SequenceHeadMixin(nn.Module):
    def make_heads(self, joined: int) -> None:
        self.shared = nn.Sequential(nn.Linear(joined, 128), nn.ReLU(), nn.Dropout(0.15))
        self.point = nn.Linear(128, 10)
        self.terminal = nn.Linear(128, 2)
        self.depth = nn.Linear(128, 3)
        self.side = nn.Linear(128, 3)
        self.width = nn.Linear(128, 2)
        self.safety = nn.Linear(128, 3)

    def heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(z)
        return {
            "point": self.point(z),
            "terminal": self.terminal(z),
            "depth": self.depth(z),
            "side": self.side(z),
            "width": self.width(z),
            "safety": self.safety(z),
        }


class LSTMBackbone(SequenceHeadMixin):
    def __init__(self, vocab_sizes: list[int], static_dim: int, hidden: int = 96, layers: int = 1, dropout: float = 0.10):
        super().__init__()
        emb_dim = 8
        self.embeddings = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in vocab_sizes])
        self.lstm = nn.LSTM(
            input_size=emb_dim * len(vocab_sizes),
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.make_heads(hidden + 32)

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([emb(strokes[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        z = torch.cat([h[-1], self.static_net(static)], dim=1)
        return self.heads(z)


class CausalTransformerBackbone(SequenceHeadMixin):
    def __init__(
        self,
        vocab_sizes: list[int],
        static_dim: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.20,
        max_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.max_len = max_len
        self.embeddings = nn.ModuleList([nn.Embedding(v, d_model, padding_idx=0) for v in vocab_sizes])
        self.proj = nn.Linear(d_model * len(vocab_sizes), d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.make_heads(d_model + 32)

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([emb(strokes[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=2)
        x = self.proj(x) + self.pos[:, : strokes.shape[1], :]
        attn_mask = subsequent_mask(strokes.shape[1]).to(strokes.device)
        pad_mask = causal_padding_mask(lengths, strokes.shape[1]).to(strokes.device)
        enc = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        idx = (lengths - 1).clamp_min(0)
        last = enc[torch.arange(len(lengths), device=strokes.device), idx]
        z = torch.cat([last, self.static_net(static)], dim=1)
        return self.heads(z)


def make_model(config: ModelConfig, vocab_sizes: list[int], static_dim: int) -> nn.Module:
    if config.family == "lstm":
        return LSTMBackbone(vocab_sizes, static_dim, hidden=config.hidden, layers=config.layers, dropout=config.dropout)
    return CausalTransformerBackbone(
        vocab_sizes,
        static_dim,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
        max_len=MAX_SEQ_LEN,
    )


def predict_proba(model: nn.Module, dataset: StrokeDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: V188Batch(
        strokes=torch.stack([x.strokes for x in b]),
        lengths=torch.stack([x.lengths for x in b]),
        static=torch.stack([x.static for x in b]),
        point=torch.stack([x.point for x in b]),
        teacher=torch.stack([x.teacher for x in b]),
    ))
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            probs.append(F.softmax(out["point"], dim=1).cpu().numpy())
    return normalize_rows_safe(np.vstack(probs))


def train_model(
    config: ModelConfig,
    train_ds: StrokeDataset,
    valid_ds: StrokeDataset,
    vocab_sizes: list[int],
    static_dim: int,
    seed: int,
) -> tuple[nn.Module, float]:
    set_seed(seed)
    model = make_model(config, vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: V188Batch(
        strokes=torch.stack([x.strokes for x in b]),
        lengths=torch.stack([x.lengths for x in b]),
        static=torch.stack([x.static for x in b]),
        point=torch.stack([x.point for x in b]),
        teacher=torch.stack([x.teacher for x in b]),
    ))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: V188Batch(
        strokes=torch.stack([x.strokes for x in b]),
        lengths=torch.stack([x.lengths for x in b]),
        static=torch.stack([x.static for x in b]),
        point=torch.stack([x.point for x in b]),
        teacher=torch.stack([x.teacher for x in b]),
    ))
    best = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), LOSS_R186_W005)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
                loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), LOSS_R186_W005)
                val += float(loss.item()) * len(batch.point)
                n += len(batch.point)
        val /= max(n, 1)
        if val + 1e-5 < best_loss:
            best_loss = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, best_loss


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, alpha: float, cap: float, config: ModelConfig) -> dict:
    point_f1 = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_f1 = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    rec = {
        "candidate": name,
        "config": config.name,
        "family": config.family,
        "alpha": float(alpha),
        "churn_cap": float(cap),
        "point_macro_f1": point_f1,
        "delta_vs_base": point_f1 - base_f1,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    for k in [0, 1, 3, 4, 7, 8, 9]:
        rec[f"point{k}_f1"] = float(rep[str(k)]["f1-score"])
    return rec


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    rows = add_r186_priors(rows, pd.read_csv(R186_TRAIN))
    test_rows = add_r186_priors(test_rows, pd.read_csv(R186_TEST))

    r111_oof = load_pickle(R111_OOF)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    v3_oof = load_pickle("oof_proba_v3.pkl")
    tuning = r111_oof["tuning"]
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    prefix_train = add_r185_columns(prefix, None, pool=True)
    v173_action_oof_prob = one_hot(state["v173_pred_oof"], 19)
    v173_action_test_prob = one_hot(state["v173_pred_test"], 19)
    r119_oof = r119_oof_prior(rows, prefix_train, v173_action_oof_prob)
    r119_test = action_conditioned_point_prior(test_rows, prefix_train, v173_action_test_prob)
    local_base_prob_oof = normalize_rows_safe(0.95 * base_point_oof + 0.05 * r119_oof)
    local_base_prob_test = normalize_rows_safe(0.95 * base_point_test + 0.05 * r119_test)
    local_base_pred_oof = point_pred(rows, local_base_prob_oof, tuning)

    train_seq, train_len = build_padded_stroke_tensor(sequences_for_rows(rows, raw_groups("train.csv")), MAX_SEQ_LEN, 0)
    test_seq, test_len = build_padded_stroke_tensor(sequences_for_rows(test_rows, raw_groups("test_new.csv")), MAX_SEQ_LEN, 0)
    vocab_sizes = [int(max(train_seq[:, :, i].max(), test_seq[:, :, i].max()) + 1) for i in range(train_seq.shape[2])]
    x_static, stats = static_matrix(rows, v173_action_oof_prob, local_base_prob_oof)
    x_test_static, _ = static_matrix(test_rows, v173_action_test_prob, local_base_prob_test, stats)
    teacher = teacher_matrix(rows)
    teacher_test = teacher_matrix(test_rows)
    y = rows["next_pointId"].astype(int).to_numpy()

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    test_base_point = base_sub["pointId"].astype(int).to_numpy()

    search_rows = [eval_candidate("local_v173_r119_base", y, local_base_pred_oof, local_base_pred_oof, 0.0, 0.0, ModelConfig("base", "base"))]
    fold_rows = []
    pred_store: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    for ci, config in enumerate(CONFIGS):
        oof_prob = np.zeros((len(rows), 10), dtype=float)
        for fold in sorted(rows["fold"].unique()):
            valid = rows["fold"].eq(fold).to_numpy()
            train = ~valid
            train_ds = StrokeDataset(train_seq[train], train_len[train], x_static[train], y[train], teacher[train])
            valid_ds = StrokeDataset(train_seq[valid], train_len[valid], x_static[valid], y[valid], teacher[valid])
            model, val_loss = train_model(config, train_ds, valid_ds, vocab_sizes, x_static.shape[1], 1890 + 10 * ci + int(fold))
            oof_prob[valid] = predict_proba(model, valid_ds)
            raw = oof_prob[valid].argmax(axis=1)
            fold_rows.append(
                {
                    "config": config.name,
                    "family": config.family,
                    "fold": int(fold),
                    "val_loss": float(val_loss),
                    "raw_point_macro_f1": float(f1_score(y[valid], raw, labels=POINT_CLASSES, average="macro", zero_division=0)),
                }
            )

        full_ds = StrokeDataset(train_seq, train_len, x_static, y, teacher)
        hold = max(1, len(train_seq) // 10)
        hold_ds = StrokeDataset(train_seq[:hold], train_len[:hold], x_static[:hold], y[:hold], teacher[:hold])
        test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
        full_model, _ = train_model(config, full_ds, hold_ds, vocab_sizes, x_static.shape[1], 2900 + ci)
        test_prob = predict_proba(full_model, test_ds)

        raw_pred = oof_prob.argmax(axis=1)
        search_rows.append(eval_candidate(f"{config.name}_raw_argmax", y, raw_pred, local_base_pred_oof, 1.0, 1.0, config))
        for alpha in ALPHAS:
            blended = row_log_blend(local_base_prob_oof, oof_prob, alpha)
            blended_test = row_log_blend(local_base_prob_test, test_prob, alpha)
            for cap in CHURN_CAPS:
                pred, _ = capped_residual_labels(local_base_pred_oof, blended, cap)
                test_pred, test_changed = capped_residual_labels(test_base_point, blended_test, cap)
                name = f"{config.name}_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                rec = eval_candidate(name, y, pred, local_base_pred_oof, alpha, cap, config)
                rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base_point))
                rec["test_changed_rows"] = int(np.sum(test_changed))
                search_rows.append(rec)
                pred_store[name] = (pred, test_pred, rec)

    search = pd.DataFrame(search_rows)
    search["tier"] = np.select(
        [search["point_churn_vs_base"].le(0.02), search["point_churn_vs_base"].le(0.05)],
        ["clean", "probe"],
        default="high_churn",
    )
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v189_v190_search.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTDIR / "v189_v190_fold_metrics.csv", index=False)

    generated = []
    emitted: set[str] = set()
    for family in ["lstm", "transformer"]:
        for tier, cap in [("clean", 0.02), ("probe", 0.03), ("probe", 0.05)]:
            part = search[
                search["family"].eq(family)
                & search["tier"].eq(tier)
                & search["delta_vs_base"].gt(0)
                & np.isclose(search["churn_cap"].astype(float), cap)
                & search["candidate"].str.contains("_a")
            ]
            if part.empty:
                continue
            rec = part.iloc[0].to_dict()
            name = str(rec["candidate"])
            if name in emitted:
                continue
            _, test_pred, stored = pred_store[name]
            sub_name = f"submission_{name}__v173action_r121server.csv"
            info = write_submission(sub_name, base_sub, test_pred)
            info.update(stored)
            info["submission"] = sub_name
            info["tier"] = tier
            generated.append(info)
            emitted.add(name)

    report = {
        "verdict": "CANDIDATES_GENERATED" if generated else "NO_POSITIVE_CANDIDATE",
        "device": DEVICE,
        "base": search[search["candidate"].eq("local_v173_r119_base")].iloc[0].to_dict(),
        "best_clean": search[search["tier"].eq("clean")].head(15).to_dict(orient="records"),
        "best_probe": search[search["tier"].eq("probe")].head(15).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "V189 uses causal LSTM point-intent backbones.",
            "V190 uses small causal Transformer point-intent backbones.",
            "R186 teacher is used only on intermediate heads, not direct pointId.",
            "Submissions are low-churn residuals on V173/R119/R121.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v189_v190_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v189_v190_report.md").write_text(
        "# V189/V190 Point Intent Sequence Models\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Device: `{DEVICE}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + ("\n".join(f"- `{g['upload_path']}` family `{g['family']}`, OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119']:.6f}`" for g in generated) or "- none")
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v189_v190_point_intent_seq_models.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated_count": len(generated), "search": str(OUTDIR / "v189_v190_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
