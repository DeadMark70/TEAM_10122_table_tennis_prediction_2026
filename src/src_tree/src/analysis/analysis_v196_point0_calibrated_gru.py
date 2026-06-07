"""V196 point0-calibrated V188-style GRU.

V195 showed that matching train sampling alone does not stop raw neural point
collapse on test.  V196 keeps the V188 architecture and V195 sampling schemes,
but adds explicit point0/terminal calibration terms to the training objective.

Submissions keep action=V173 and server=R121.  TTMATCH is not read.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_v188_point_intent_gru import (
    BATCH_SIZE,
    DEVICE,
    LOSS_SETTINGS,
    PointIntentGRU,
    StrokeDataset,
    batch_loss,
    capped_residual_labels,
    collate,
    predict_proba,
    row_log_blend,
    set_seed,
)
from analysis_v195_distribution_matched_point_gru import (
    MATCH_COLS,
    distribution,
    distribution_match_weights,
    prepare_data,
)
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v196_point0_calibrated_gru")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v196_point0_calibrated_gru.py")

ALPHAS = [0.05, 0.075]
CHURN_CAPS = [0.02, 0.03, 0.05]
EPOCHS = 10
PATIENCE = 2


@dataclass(frozen=True)
class CalibrationSetting:
    name: str
    target: float
    rate_weight: float
    confidence_threshold: float
    confidence_weight: float
    consistency_weight: float


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


CALIBRATION_SETTINGS = [
    CalibrationSetting("p0t029_rw1_conf085_cw025_tc005", 0.29, 1.0, 0.85, 0.25, 0.05),
    CalibrationSetting("p0t026_rw1_conf075_cw05_tc005", 0.26, 1.0, 0.75, 0.50, 0.05),
]


def point0_rate_penalty(point_logits: torch.Tensor, target: float, weight: float) -> torch.Tensor:
    if weight <= 0:
        return point_logits.sum() * 0.0
    p0 = F.softmax(point_logits, dim=1)[:, 0].mean()
    return float(weight) * (p0 - float(target)).pow(2)


def point0_confidence_penalty(point_logits: torch.Tensor, threshold: float, weight: float) -> torch.Tensor:
    if weight <= 0:
        return point_logits.sum() * 0.0
    p0 = F.softmax(point_logits, dim=1)[:, 0]
    return float(weight) * F.relu(p0 - float(threshold)).pow(2).mean()


def terminal_consistency_penalty(point_logits: torch.Tensor, terminal_logits: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0:
        return point_logits.sum() * 0.0
    point_p0 = F.softmax(point_logits, dim=1)[:, 0]
    terminal_p = F.softmax(terminal_logits, dim=1)[:, 1]
    return float(weight) * F.mse_loss(point_p0, terminal_p)


def calibrated_batch_loss(
    outputs: dict[str, torch.Tensor],
    point: torch.Tensor,
    teacher: torch.Tensor,
    aux_weights: dict[str, float],
    calibration: CalibrationSetting,
) -> torch.Tensor:
    loss = batch_loss(outputs, point, teacher, aux_weights)
    loss = loss + point0_rate_penalty(outputs["point"], calibration.target, calibration.rate_weight)
    loss = loss + point0_confidence_penalty(outputs["point"], calibration.confidence_threshold, calibration.confidence_weight)
    loss = loss + terminal_consistency_penalty(outputs["point"], outputs["terminal"], calibration.consistency_weight)
    return loss


def train_model_calibrated(
    train_ds: StrokeDataset,
    valid_ds: StrokeDataset,
    vocab_sizes: list[int],
    static_dim: int,
    aux_weights: dict[str, float],
    calibration: CalibrationSetting,
    seed: int,
    sample_weights: np.ndarray | None = None,
) -> tuple[PointIntentGRU, float]:
    set_seed(seed)
    model = PointIntentGRU(vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = calibrated_batch_loss(
                out,
                batch.point.to(DEVICE),
                batch.teacher.to(DEVICE),
                aux_weights,
                calibration,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
                loss = calibrated_batch_loss(
                    out,
                    batch.point.to(DEVICE),
                    batch.teacher.to(DEVICE),
                    aux_weights,
                    calibration,
                )
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


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
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
    for k in [0, 1, 3, 4, 7, 8, 9]:
        rec[f"point{k}_f1"] = float(rep[str(k)]["f1-score"])
    rec.update(meta)
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


def make_dataset(data: dict, source: str, idx: np.ndarray | slice, *, valid: bool = False) -> StrokeDataset:
    if source == "oof":
        return StrokeDataset(
            data["oof_seq"][idx],
            data["oof_len"][idx],
            data["x_oof"][idx],
            data["y_oof"][idx],
            data["teacher_oof"][idx],
        )
    if source == "full":
        return StrokeDataset(
            data["full_seq"][idx],
            data["full_len"][idx],
            data["x_full"][idx],
            data["y_full"][idx],
            data["teacher_full"][idx],
        )
    raise ValueError(source)


def run_scheme(name: str, data: dict, source: str, calibration: CalibrationSetting) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = data["rows"]
    y = data["y_oof"]
    aux_weights = dict(LOSS_SETTINGS)["r186_w005"]
    if source == "oof":
        train_rows_all = rows
        train_source = "oof"
        train_static_dim = data["x_oof"].shape[1]
        train_weights_all = distribution_match_weights(train_rows_all, data["test_rows"], MATCH_COLS)
        test_static = data["x_test_oofstats"]
    elif source == "full":
        train_rows_all = data["full_pool"]
        train_source = "full"
        train_static_dim = data["x_full"].shape[1]
        train_weights_all = distribution_match_weights(train_rows_all, data["test_rows"], MATCH_COLS)
        test_static = data["x_test_fullstats"]
    else:
        raise ValueError(source)
    test_ds = StrokeDataset(
        data["test_seq"],
        data["test_len"],
        test_static,
        np.zeros(len(data["test_seq"]), dtype=np.int64),
        data["teacher_test"],
    )
    oof_prob = np.zeros((len(rows), 10), dtype=float)
    fold_test_probs = []
    fold_records = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_mask = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        valid_ds = make_dataset(data, "oof", valid_mask, valid=True)
        train_mask = ~train_rows_all["fold"].astype(int).eq(int(fold)).to_numpy()
        train_idx = np.where(train_mask)[0]
        train_ds = make_dataset(data, train_source, train_idx)
        model, val_loss = train_model_calibrated(
            train_ds,
            valid_ds,
            data["vocab_sizes"],
            train_static_dim,
            aux_weights,
            calibration,
            1960 + int(fold),
            sample_weights=train_weights_all[train_idx],
        )
        oof_prob[valid_mask] = predict_proba(model, valid_ds)
        fold_test_probs.append(predict_proba(model, test_ds))
        pred = oof_prob[valid_mask].argmax(axis=1).astype(int)
        fold_records.append(
            {
                "scheme": name,
                "fold": int(fold),
                "source": source,
                "calibration": calibration.name,
                "train_rows": int(len(train_idx)),
                "val_loss": float(val_loss),
                "raw_point_macro_f1": float(f1_score(y[valid_mask], pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
                "raw_point0_rate": float(np.mean(pred == 0)),
            }
        )
    return normalize_rows_safe(oof_prob), normalize_rows_safe(np.mean(fold_test_probs, axis=0)), fold_records


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    random.seed(196)
    np.random.seed(196)
    set_seed(196)
    data = prepare_data()
    y = data["y_oof"]
    base = data["base_pred_oof"]
    test_base = data["test_base_point"]
    search_records = [eval_candidate("local_v173_r119_base", y, base, base, {"scheme": "base"})]
    fold_records = []
    pred_store: dict[str, np.ndarray] = {}

    schemes = [
        ("v196a_r111_importance", "oof"),
        ("v196c_fullpool_importance", "full"),
    ]
    for calibration in CALIBRATION_SETTINGS:
        for scheme, source in schemes:
            tag = f"{scheme}_{calibration.name}"
            oof_prob, test_prob, folds = run_scheme(tag, data, source, calibration)
            fold_records.extend(folds)
            raw_oof = oof_prob.argmax(axis=1).astype(int)
            raw_test = test_prob.argmax(axis=1).astype(int)
            search_records.append(
                eval_candidate(
                    f"{tag}_raw_argmax",
                    y,
                    raw_oof,
                    base,
                    {
                        "scheme": scheme,
                        "source": source,
                        "calibration": calibration.name,
                        "alpha": 1.0,
                        "cap": 1.0,
                        "test_raw_point0_rate": float(np.mean(raw_test == 0)),
                        "test_raw_distribution": json.dumps(distribution(raw_test), sort_keys=True),
                        "test_raw_p0_mean": float(test_prob[:, 0].mean()),
                    },
                )
            )
            for alpha in ALPHAS:
                blended = row_log_blend(data["base_prob_oof"], oof_prob, alpha)
                blended_test = row_log_blend(data["base_prob_test"], test_prob, alpha)
                for cap in CHURN_CAPS:
                    pred, _ = capped_residual_labels(base, blended, cap)
                    test_pred, test_changed = capped_residual_labels(test_base, blended_test, cap)
                    name = f"{tag}_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                    rec = eval_candidate(
                        name,
                        y,
                        pred,
                        base,
                        {"scheme": scheme, "source": source, "calibration": calibration.name, "alpha": alpha, "cap": cap},
                    )
                    rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base))
                    rec["test_changed_rows"] = int(np.sum(test_changed))
                    rec["test_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
                    search_records.append(rec)
                    pred_store[name] = test_pred

    search = pd.DataFrame(search_records)
    search["tier"] = np.select(
        [search["point_churn_vs_base"].le(0.02), search["point_churn_vs_base"].le(0.05)],
        ["clean", "probe"],
        default="high_churn",
    )
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v196_search.csv", index=False)
    pd.DataFrame(fold_records).to_csv(OUTDIR / "v196_fold_metrics.csv", index=False)

    generated = []
    emitted = set()
    positive = search[(search["candidate"].str.startswith("v196")) & search["delta_vs_base"].gt(0) & search["tier"].isin(["clean", "probe"])]
    for tier in ["clean", "probe"]:
        part = positive[positive["tier"].eq(tier)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        if name in emitted or name not in pred_store:
            continue
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        info.update(rec)
        info["submission"] = sub_name
        generated.append(info)
        emitted.add(name)

    raw_rows = search[search["candidate"].str.endswith("_raw_argmax")].to_dict(orient="records")
    report = {
        "verdict": "CANDIDATES_GENERATED" if generated else "NO_POSITIVE_CANDIDATE",
        "device": DEVICE,
        "base": search[search["candidate"].eq("local_v173_r119_base")].iloc[0].to_dict(),
        "raw_rows": raw_rows,
        "best_clean": search[search["tier"].eq("clean")].head(12).to_dict(orient="records"),
        "best_probe": search[search["tier"].eq("probe")].head(12).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "V196 keeps V188 architecture but adds point0 rate, point0 overconfidence, and terminal consistency losses.",
            "A uses the R111 aligned pool with importance sampling; C uses the full prefix pool with importance sampling.",
            "Submissions keep action=V173 and server=R121.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v196_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v196_report.md").write_text(
        "# V196 Point0-Calibrated GRU\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Device: `{DEVICE}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Raw Test Stability\n\n"
        + "\n".join(
            f"- `{r['candidate']}` OOF `{r['point_macro_f1']:.6f}`, test raw p0 `{r.get('test_raw_point0_rate', float('nan')):.6f}`, dist `{r.get('test_raw_distribution', '{}')}`"
            for r in raw_rows
        )
        + "\n\n## Generated\n\n"
        + ("\n".join(f"- `{g['upload_path']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119']:.6f}`" for g in generated) or "- none")
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v196_point0_calibrated_gru.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated_count": len(generated), "search": str(OUTDIR / "v196_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
