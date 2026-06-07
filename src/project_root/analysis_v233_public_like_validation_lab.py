"""V233 public-like action validation lab.

This script turns the recent public failures into a local gate.  It does not
train models.  It checks whether ordinary OOF, public-like weighted OOF,
worst-group slices, and no-label test sanity agree with known public outcomes.

No TTMATCH and no old-server labels are read.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r179_action_physics_hierarchy import action_family
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v233_public_like_validation_lab")
SRC_DEST = Path("src/analysis/analysis_v233_public_like_validation_lab.py")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"

HISTORICAL_PUBLIC = {
    "current_anchor": 0.3573932,
    "submission_v220_backoff_balanced_weakonly__pv188cap5__sr121.csv": 0.3542440,
    "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv": 0.3509562,
}


def bin_prefix_len(value: int | float) -> str:
    v = int(value)
    if v <= 1:
        return "1"
    if v == 2:
        return "2"
    if v == 3:
        return "3"
    if v <= 6:
        return "4_6"
    return "7_plus"


def _context_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    prefix = pd.to_numeric(rows.get("prefix_len", 0), errors="coerce").fillna(0)
    out["prefix_bin"] = [bin_prefix_len(v) for v in prefix]
    out["phase"] = rows["audit_phase"].astype(str) if "audit_phase" in rows.columns else "unknown"
    out["lag0_family"] = rows["audit_lag0_action_family"].astype(str) if "audit_lag0_action_family" in rows.columns else "unknown"
    out["lag0_depth"] = rows["audit_lag0_depth"].astype(str) if "audit_lag0_depth" in rows.columns else "unknown"
    return out


def density_ratio_weights(train_context: pd.DataFrame, test_context: pd.DataFrame, cols: list[str], clip: tuple[float, float] = (0.25, 4.0)) -> np.ndarray:
    train_key = train_context[cols].astype(str).agg("||".join, axis=1)
    test_key = test_context[cols].astype(str).agg("||".join, axis=1)
    train_freq = train_key.value_counts(normalize=True)
    test_freq = test_key.value_counts(normalize=True)
    ratios = train_key.map(lambda k: float(test_freq.get(k, 0.0) / max(train_freq.get(k, 0.0), 1e-12))).to_numpy(dtype=float)
    ratios = np.where(np.isfinite(ratios), ratios, 1.0)
    ratios = np.clip(ratios, float(clip[0]), float(clip[1]))
    ratios = ratios / max(float(np.mean(ratios)), 1e-12)
    return np.clip(ratios, float(clip[0]), float(clip[1]))


def weighted_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, labels=ACTION_CLASSES) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", sample_weight=np.asarray(weights, dtype=float), zero_division=0))


def worst_group_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, labels=ACTION_CLASSES, min_rows: int = 50) -> float:
    vals = []
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    for value, idx in g.groupby(g).groups.items():
        mask = np.zeros(len(g), dtype=bool)
        mask[list(idx)] = True
        if int(mask.sum()) < int(min_rows):
            continue
        vals.append(float(f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)))
    if not vals:
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    return float(np.min(vals))


def family_tv_distance(action_a: np.ndarray, action_b: np.ndarray) -> float:
    fams = ["Zero", "Attack", "Control", "Defensive", "Serve"]
    def dist(actions: np.ndarray) -> np.ndarray:
        counts = pd.Series([action_family(int(a)) for a in actions]).value_counts(normalize=True)
        return np.array([float(counts.get(f, 0.0)) for f in fams], dtype=float)
    return float(0.5 * np.abs(dist(action_a) - dist(action_b)).sum())


def load_candidate_oof() -> dict[str, tuple[np.ndarray, str]]:
    """Return known OOF action predictions keyed by submission-style name."""
    out: dict[str, tuple[np.ndarray, str]] = {}
    # V191 only has test submissions, but its local metrics table records action
    # source quality.  Keep V220 from search table where candidate labels exist.
    search_tables = [
        Path("v220_action_backoff_support_filter/v220_action_search.csv"),
        Path("v230_action_soft_teacher_factory/v230_action_search.csv"),
        Path("v232_v173_curriculum_deepening/v232_action_search.csv"),
    ]
    for path in search_tables:
        if path.exists():
            out[path.stem] = (np.array([], dtype=int), str(path))
    return out


def evaluate_oof_candidate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, rows: pd.DataFrame, weights: np.ndarray) -> dict:
    ctx = _context_frame(rows)
    ordinary = float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))
    anchor_ordinary = float(f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0))
    weighted = weighted_macro_f1(y, pred, weights)
    anchor_weighted = weighted_macro_f1(y, anchor, weights)
    group_scores = {
        "worst_phase": worst_group_macro_f1(y, pred, ctx["phase"]),
        "worst_prefix": worst_group_macro_f1(y, pred, ctx["prefix_bin"]),
        "worst_lag0_family": worst_group_macro_f1(y, pred, ctx["lag0_family"]),
        "worst_lag0_depth": worst_group_macro_f1(y, pred, ctx["lag0_depth"]),
    }
    anchor_groups = {
        "worst_phase": worst_group_macro_f1(y, anchor, ctx["phase"]),
        "worst_prefix": worst_group_macro_f1(y, anchor, ctx["prefix_bin"]),
        "worst_lag0_family": worst_group_macro_f1(y, anchor, ctx["lag0_family"]),
        "worst_lag0_depth": worst_group_macro_f1(y, anchor, ctx["lag0_depth"]),
    }
    rec = {
        "candidate": name,
        "ordinary_action_macro_f1": ordinary,
        "ordinary_delta_vs_v173": ordinary - anchor_ordinary,
        "iw_action_macro_f1": weighted,
        "iw_delta_vs_v173": weighted - anchor_weighted,
        "action_churn_vs_v173": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
        "family_tv_vs_v173": family_tv_distance(anchor, pred),
    }
    for key, val in group_scores.items():
        rec[key] = val
        rec[f"{key}_delta_vs_v173"] = val - anchor_groups[key]
    rec["gate_pass"] = bool(
        rec["ordinary_delta_vs_v173"] >= 0
        and rec["iw_delta_vs_v173"] >= 0.0015
        and min(rec["worst_phase_delta_vs_v173"], rec["worst_prefix_delta_vs_v173"], rec["worst_lag0_family_delta_vs_v173"]) >= -0.0005
    )
    return rec


def test_sanity_for_submission(path: Path, anchor: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    merged = anchor[["rally_uid", "actionId"]].merge(sub[["rally_uid", "actionId"]], on="rally_uid", suffixes=("_anchor", "_cand"), validate="one_to_one")
    anchor_action = merged["actionId_anchor"].astype(int).to_numpy()
    cand_action = merged["actionId_cand"].astype(int).to_numpy()
    return {
        "candidate": path.name,
        "test_action_churn_vs_anchor": float(np.mean(anchor_action != cand_action)),
        "test_changed_rows": int(np.sum(anchor_action != cand_action)),
        "test_family_tv_vs_anchor": family_tv_distance(anchor_action, cand_action),
        "serve_count": int(np.isin(cand_action, [15, 16, 17, 18]).sum()),
        "action_distribution": json.dumps(pd.Series(cand_action).value_counts().sort_index().to_dict(), sort_keys=True),
    }


def historical_public_backtest() -> pd.DataFrame:
    rows = []
    for name, public in HISTORICAL_PUBLIC.items():
        rows.append({"candidate": name, "public_pl": public, "delta_vs_anchor_public": public - HISTORICAL_PUBLIC["current_anchor"]})
    return pd.DataFrame(rows).sort_values("public_pl", ascending=False)


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    rows = add_audit_columns(data["rows"].copy())
    test_rows = add_audit_columns(state["test_rows"].copy())
    y = rows["next_actionId"].astype(int).to_numpy()
    v173 = state["v173_pred_oof"].astype(int)
    train_ctx = _context_frame(rows)
    test_ctx = _context_frame(test_rows)
    weights = density_ratio_weights(train_ctx, test_ctx, ["prefix_bin", "phase", "lag0_family", "lag0_depth"])

    records = [evaluate_oof_candidate("v173_anchor", y, v173, v173, rows, weights)]

    # Fold in available OOF predictions from recent suites where full OOF action
    # labels were saved or can be reconstructed from search artifacts.
    for name, path in [
        ("v220_backoff_balanced_weakonly", Path("v220_action_backoff_support_filter/v220_action_search.csv")),
        ("v230_aggressive_teacher_mix", Path("v230_action_soft_teacher_factory/v230_aggressive_teacher_mix_oof_action_prob.npy")),
        ("v232_external_plus_v173_kd", Path("v232_v173_curriculum_deepening/v232_external_plus_v173_kd_oof_action_prob.npy")),
    ]:
        if path.suffix == ".npy" and path.exists():
            pred = np.load(path).argmax(axis=1).astype(int)
            records.append(evaluate_oof_candidate(name, y, pred, v173, rows, weights))

    pd.DataFrame(records).to_csv(OUTDIR / "v233_oof_validation_scores.csv", index=False)
    pd.DataFrame({"weight": weights}).describe().to_csv(OUTDIR / "v233_importance_weight_summary.csv")
    historical_public_backtest().to_csv(OUTDIR / "v233_historical_public_backtest.csv", index=False)

    anchor = pd.read_csv(ANCHOR)
    sanity_rows = []
    for path in [
        ANCHOR,
        UPLOAD_DIR / "submission_v220_backoff_balanced_weakonly__pv188cap5__sr121.csv",
        UPLOAD_DIR / "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv",
        UPLOAD_DIR / "submission_v230_aggressive_mix__pv188cap5__sr121.csv",
        UPLOAD_DIR / "submission_v232_v173kd__pv188cap5__sr121.csv",
    ]:
        if path.exists():
            sanity_rows.append(test_sanity_for_submission(path, anchor))
    pd.DataFrame(sanity_rows).to_csv(OUTDIR / "v233_test_sanity.csv", index=False)

    report = {
        "verdict": "VALIDATION_LAB_READY",
        "records": records,
        "historical_public": historical_public_backtest().to_dict(orient="records"),
        "notes": [
            "This is a validation lab, not a submission generator.",
            "Importance weights use test_new observed prefix/context distribution only.",
            "No TTMATCH and no old-server labels are read.",
        ],
    }
    (OUTDIR / "v233_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v233_report.md").write_text(
        "# V233 Public-Like Validation Lab\n\n"
        "- Verdict: `VALIDATION_LAB_READY`\n"
        f"- OOF candidates scored: `{len(records)}`\n"
        "- Outputs: `v233_oof_validation_scores.csv`, `v233_test_sanity.csv`, `v233_historical_public_backtest.csv`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v233_public_like_validation_lab.py", SRC_DEST)
    print(json.dumps({"outdir": str(OUTDIR), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
