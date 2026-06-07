"""V203/V206 AICUP-only TT-ShuttleNet smoke experiment.

This is a ShuttleNet-inspired sequence experiment for the table-tennis task.
It keeps the first implementation deliberately small and clean:

  - AICUP-only; no ShuttleSet/CoachAI/TTMATCH external rows are read.
  - Type and area stroke streams are encoded separately.
  - A player/style-like fusion gate combines type, area, and static context.
  - Raw neural argmax is diagnostic only.
  - Exported submissions are capped point residuals over the current
    V173 action + V188 cap5 point + R121 server no-old anchor.
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_v188_point_intent_gru import batch_loss, row_log_blend, set_seed
from analysis_v195_distribution_matched_point_gru import MATCH_COLS, distribution, distribution_match_weights, prepare_data
from analysis_v196_point0_calibrated_gru import CalibrationSetting, calibrated_batch_loss
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v203_tt_shuttlenet")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/train_v203_tt_shuttlenet.py")

TYPE_IDXS = [0, 2, 3, 4, 6]  # action, spin, strength, hand, strike
AREA_IDXS = [1, 5]  # point, position
BATCH_SIZE = 512
EPOCHS = 8
PATIENCE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIBRATION = CalibrationSetting("v203_p0t026_conf075", 0.26, 0.75, 0.80, 0.30, 0.05)
AUX_WEIGHTS = {"terminal": 0.25, "depth": 0.25, "side": 0.15, "safety": 0.05, "width_teacher": 0.03, "r186": 0.03}


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D")
    out = arr.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = 0.0
    out = np.maximum(out, 0.0)
    row_sum = out.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0
    if zero.any():
        out[zero] = 1.0 / out.shape[1]
        row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


def depth_from_point(point_id: int) -> int:
    p = int(point_id)
    if p == 0:
        return 0
    return ((p - 1) // 3) + 1


def side_from_point(point_id: int) -> int:
    p = int(point_id)
    if p == 0:
        return 0
    return ((p - 1) % 3) + 1


def split_type_area_static(static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(static, dtype=float)
    if arr.ndim != 2:
        raise ValueError("static must be 2D")
    return arr[:, 0::2], arr[:, 1::2]


def apply_point_residual(
    base_labels: np.ndarray,
    prob: np.ndarray,
    max_churn: float,
    gate: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=np.int64)
    pred = np.asarray(prob).argmax(axis=1).astype(np.int64)
    changed = pred != base
    if gate is not None:
        changed &= np.asarray(gate, dtype=bool)
    max_rows = int(np.floor(len(base) * float(max_churn)))
    if changed.sum() > max_rows:
        base_score = prob[np.arange(len(prob)), base]
        pred_score = prob[np.arange(len(prob)), pred]
        gain = pred_score - base_score
        cand = np.where(changed)[0]
        keep = cand[np.argsort(gain[cand])[::-1][:max_rows]]
        capped = np.zeros(len(base), dtype=bool)
        capped[keep] = True
        changed = capped
    out = base.copy()
    out[changed] = pred[changed]
    return out, changed


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


@dataclass
class TTBatch:
    type_strokes: torch.Tensor
    area_strokes: torch.Tensor
    lengths: torch.Tensor
    static: torch.Tensor
    point: torch.Tensor
    teacher: torch.Tensor


class TTShuttleDataset(Dataset):
    def __init__(self, strokes: np.ndarray, lengths: np.ndarray, static: np.ndarray, point: np.ndarray, teacher: np.ndarray):
        self.type_strokes = torch.as_tensor(strokes[:, :, TYPE_IDXS], dtype=torch.long)
        self.area_strokes = torch.as_tensor(strokes[:, :, AREA_IDXS], dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.point = torch.as_tensor(np.asarray(point, dtype=np.int64).copy(), dtype=torch.long)
        self.teacher = torch.as_tensor(teacher, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.point)

    def __getitem__(self, idx: int) -> TTBatch:
        return TTBatch(
            self.type_strokes[idx],
            self.area_strokes[idx],
            self.lengths[idx],
            self.static[idx],
            self.point[idx],
            self.teacher[idx],
        )


def collate(batch: list[TTBatch]) -> TTBatch:
    return TTBatch(
        type_strokes=torch.stack([b.type_strokes for b in batch]),
        area_strokes=torch.stack([b.area_strokes for b in batch]),
        lengths=torch.stack([b.lengths for b in batch]),
        static=torch.stack([b.static for b in batch]),
        point=torch.stack([b.point for b in batch]),
        teacher=torch.stack([b.teacher for b in batch]),
    )


class TTShuttleNetLite(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        type_vocab = [vocab_sizes[i] for i in TYPE_IDXS]
        area_vocab = [vocab_sizes[i] for i in AREA_IDXS]
        self.type_emb = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in type_vocab])
        self.area_emb = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in area_vocab])
        self.type_gru = nn.GRU(emb_dim * len(type_vocab), hidden, batch_first=True)
        self.area_gru = nn.GRU(emb_dim * len(area_vocab), hidden, batch_first=True)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 32), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2 + 32, 3), nn.Softmax(dim=1))
        self.shared = nn.Sequential(nn.Linear(hidden * 2 + 32, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 96), nn.ReLU())
        self.point = nn.Linear(96, 10)
        self.terminal = nn.Linear(96, 2)
        self.depth = nn.Linear(96, 3)
        self.side = nn.Linear(96, 3)
        self.width = nn.Linear(96, 2)
        self.safety = nn.Linear(96, 3)

    @staticmethod
    def encode_stream(gru: nn.GRU, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = gru(packed)
        return h[-1]

    def forward(self, type_strokes: torch.Tensor, area_strokes: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        type_x = torch.cat([emb(type_strokes[:, :, i]) for i, emb in enumerate(self.type_emb)], dim=2)
        area_x = torch.cat([emb(area_strokes[:, :, i]) for i, emb in enumerate(self.area_emb)], dim=2)
        type_h = self.encode_stream(self.type_gru, type_x, lengths)
        area_h = self.encode_stream(self.area_gru, area_x, lengths)
        static_h = self.static_net(static)
        weights = self.gate(torch.cat([type_h, area_h, static_h], dim=1))
        fused = torch.cat([type_h * weights[:, [0]], area_h * weights[:, [1]], static_h * weights[:, [2]]], dim=1)
        z = self.shared(fused)
        return {
            "point": self.point(z),
            "terminal": self.terminal(z),
            "depth": self.depth(z),
            "side": self.side(z),
            "width": self.width(z),
            "safety": self.safety(z),
        }


def train_model(
    train_ds: TTShuttleDataset,
    valid_ds: TTShuttleDataset,
    vocab_sizes: list[int],
    static_dim: int,
    seed: int,
    sample_weights: np.ndarray | None,
) -> tuple[TTShuttleNetLite, float]:
    set_seed(seed)
    model = TTShuttleNetLite(vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    if sample_weights is None:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    else:
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.type_strokes.to(DEVICE), batch.area_strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = calibrated_batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), AUX_WEIGHTS, CALIBRATION)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val_loss = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch.type_strokes.to(DEVICE), batch.area_strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
                loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), AUX_WEIGHTS)
                val_loss += float(loss.item()) * len(batch.point)
                n += len(batch.point)
        val_loss /= max(n, 1)
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_proba(model: TTShuttleNetLite, dataset: TTShuttleDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.type_strokes.to(DEVICE), batch.area_strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))["point"]
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return normalize_rows(np.vstack(out))


def make_dataset(data: dict, source: str, idx: np.ndarray | slice, static: np.ndarray | None = None) -> TTShuttleDataset:
    if source == "oof":
        return TTShuttleDataset(
            data["oof_seq"][idx],
            data["oof_len"][idx],
            data["x_oof"][idx] if static is None else static,
            data["y_oof"][idx],
            data["teacher_oof"][idx],
        )
    if source == "full":
        return TTShuttleDataset(
            data["full_seq"][idx],
            data["full_len"][idx],
            data["x_full"][idx] if static is None else static,
            data["y_full"][idx],
            data["teacher_full"][idx],
        )
    if source == "test":
        return TTShuttleDataset(
            data["test_seq"][idx],
            data["test_len"][idx],
            data["x_test_fullstats"][idx] if static is None else static,
            np.zeros(len(data["test_seq"][idx]), dtype=np.int64),
            data["teacher_test"][idx],
        )
    raise ValueError(source)


def run_v203(data: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = data["rows"]
    y = data["y_oof"]
    train_rows_all = data["full_pool"]
    train_weights_all = distribution_match_weights(train_rows_all, data["test_rows"], MATCH_COLS)
    test_ds = make_dataset(data, "test", slice(None))
    oof_prob = np.zeros((len(rows), 10), dtype=float)
    fold_test = []
    fold_records = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_mask = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        valid_ds = make_dataset(data, "oof", valid_mask)
        train_mask = ~train_rows_all["fold"].astype(int).eq(int(fold)).to_numpy()
        train_idx = np.where(train_mask)[0]
        train_ds = make_dataset(data, "full", train_idx)
        model, val_loss = train_model(
            train_ds,
            valid_ds,
            data["vocab_sizes"],
            data["x_full"].shape[1],
            seed=2030 + int(fold),
            sample_weights=train_weights_all[train_idx],
        )
        oof_prob[valid_mask] = predict_proba(model, valid_ds)
        fold_test.append(predict_proba(model, test_ds))
        raw = oof_prob[valid_mask].argmax(axis=1).astype(int)
        fold_records.append(
            {
                "scheme": "v203_ttshuttle_fullprefix",
                "fold": int(fold),
                "train_rows": int(len(train_idx)),
                "val_loss": float(val_loss),
                "raw_point_macro_f1": float(f1_score(y[valid_mask], raw, labels=POINT_CLASSES, average="macro", zero_division=0)),
                "raw_point0_rate": float(np.mean(raw == 0)),
            }
        )
    return normalize_rows(oof_prob), normalize_rows(np.mean(fold_test, axis=0)), fold_records


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    score = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_score = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "point_macro_f1": score,
        "delta_vs_base": score - base_score,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "pred_point0_rate": float(np.mean(pred == 0)),
    }
    rec.update(meta)
    return rec


def long_attack_gate(rows: pd.DataFrame) -> np.ndarray:
    phase = rows.get("r184_phase", rows.get("audit_phase", "")).astype(str)
    depth = rows.get("r184_lag0_depth", rows.get("audit_lag0_depth", "")).astype(str)
    family = rows.get("r184_lag0_family", rows.get("audit_lag0_action_family", "")).astype(str)
    prefix = pd.to_numeric(rows.get("prefix_len", pd.Series([0] * len(rows))), errors="coerce").fillna(0)
    return (phase.eq("rally") | depth.eq("long") | family.eq("Attack") | prefix.ge(3)).to_numpy()


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
    random.seed(203)
    np.random.seed(203)
    set_seed(203)
    data = prepare_data()
    y = data["y_oof"]
    base = data["base_pred_oof"]
    test_base = data["test_base_point"]
    oof_prob, test_prob, folds = run_v203(data)
    raw_oof = oof_prob.argmax(axis=1).astype(int)
    raw_test = test_prob.argmax(axis=1).astype(int)

    records = [
        eval_candidate(
            "v203_ttshuttle_raw_diagnostic",
            y,
            raw_oof,
            base,
            {
                "alpha": 1.0,
                "cap": 1.0,
                "gate": "none",
                "test_raw_point0_rate": float(np.mean(raw_test == 0)),
                "test_raw_distribution": json.dumps(distribution(raw_test), sort_keys=True),
            },
        )
    ]
    pred_store = {}
    blended_store = {}
    for alpha, cap in [(0.05, 0.02), (0.075, 0.05)]:
        blend = row_log_blend(data["base_prob_oof"], normalize_rows_safe(oof_prob), alpha)
        blend_test = row_log_blend(data["base_prob_test"], normalize_rows_safe(test_prob), alpha)
        pred, changed = apply_point_residual(base, blend, cap)
        test_pred, test_changed = apply_point_residual(test_base, blend_test, cap)
        name = f"v203_ttshuttle_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
        rec = eval_candidate(name, y, pred, base, {"alpha": alpha, "cap": cap, "gate": "all"})
        rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base))
        rec["test_changed_rows"] = int(np.sum(test_changed))
        rec["test_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
        records.append(rec)
        pred_store[name] = test_pred
        blended_store[name] = (blend, blend_test)

    alpha, cap = 0.075, 0.05
    blend, blend_test = blended_store["v203_ttshuttle_a0p075_cap0p05"]
    gate = long_attack_gate(data["rows"])
    test_gate = long_attack_gate(data["test_rows"])
    pred, changed = apply_point_residual(base, blend, cap, gate=gate)
    test_pred, test_changed = apply_point_residual(test_base, blend_test, cap, gate=test_gate)
    name = "v206_ttshuttle_long_attack_a0p075_cap0p05"
    rec = eval_candidate(name, y, pred, base, {"alpha": alpha, "cap": cap, "gate": "long_attack"})
    rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base))
    rec["test_changed_rows"] = int(np.sum(test_changed))
    rec["test_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
    records.append(rec)
    pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v203_search.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTDIR / "v203_fold_metrics.csv", index=False)

    generated = []
    for name in ["v203_ttshuttle_a0p05_cap0p02", "v203_ttshuttle_a0p075_cap0p05", "v206_ttshuttle_long_attack_a0p075_cap0p05"]:
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        rec = search[search["candidate"].eq(name)].iloc[0].to_dict()
        info.update(rec)
        generated.append(info)

    report = {
        "verdict": "GENERATED",
        "device": DEVICE,
        "raw_test_point0_rate": float(np.mean(raw_test == 0)),
        "raw_test_distribution": distribution(raw_test),
        "generated": generated,
        "best": search.head(8).to_dict(orient="records"),
        "notes": [
            "V203 is an AICUP-only ShuttleNet-inspired smoke test.",
            "The model separates type and area stroke streams and fuses them with static context.",
            "Raw argmax is diagnostic only and is not exported.",
            "Generated submissions keep action=V173 and server=R121.",
            "TTMATCH and ShuttleSet external rows are not read.",
        ],
    }
    (OUTDIR / "v203_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v203_report.md").write_text(
        "# V203 TT-ShuttleNet Smoke\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Device: `{DEVICE}`\n"
        f"- Raw test point0 rate: `{report['raw_test_point0_rate']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + "\n".join(
            f"- `{g['submission']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119']:.6f}`"
            for g in generated
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("train_v203_tt_shuttlenet.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
