"""R182 audit for the R180 point churn mismatch.

This script does not create new submissions.  It reconstructs the R180 test
base and selected point probabilities, then separates base mismatch from
calibration/transform churn and packaging drift.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r1_oof_ensemble import compose_v3
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from analysis_r108_r110_r109_transductive import foldsafe_priors, test_priors
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r120_r123_sequence_meta import apply_motif_prior, r120_motif_oof
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r180_point_physics_calibration import (
    ARTIFACT_PATH,
    OUTDIR as R180_OUTDIR,
    R67_ANCHOR,
    add_point_physics_columns,
    apply_long_side_redistribution,
    apply_point_hierarchy_calibration,
    full_structured_priors,
    point_pred,
)


OUTDIR = Path("r182_r180_churn_audit")
UPLOAD_DIR = Path("upload_candidates_20260519")
R180_REPORT = R180_OUTDIR / "r180_report.json"
R181_R180_NO_OLD = UPLOAD_DIR / "submission_r181_no_old_r67action_r180point_r121server.csv"


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_report() -> dict:
    return json.loads(R180_REPORT.read_text(encoding="utf-8"))


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    aligned = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if aligned[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align with R67")
    return aligned


def margin_bins(prob: np.ndarray) -> np.ndarray:
    sorted_prob = np.sort(prob, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]
    return pd.cut(
        margin,
        bins=[-np.inf, 0.005, 0.01, 0.02, 0.05, 0.10, np.inf],
        labels=["<=0.005", "0.005-0.01", "0.01-0.02", "0.02-0.05", "0.05-0.10", ">0.10"],
    )


def churn(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def changed(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def build_r180_test_state() -> dict:
    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, _ = prepare_prefix_features()
    prefix = add_point_physics_columns(prefix)
    test_prefix = add_point_physics_columns(test_prefix)

    r101_oof = load_pickle("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
    r101_test = load_pickle("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
    r111_oof = load_pickle("r111_remaining_moe_gru/oof_proba_r111.pkl")
    r111_test = load_pickle("r111_remaining_moe_gru/test_proba_r111.pkl")
    v3_oof = load_pickle("oof_proba_v3.pkl")

    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    meta = art["valid_meta"].copy().reset_index(drop=True)
    rows = add_point_physics_columns(align_prefix_meta(meta, prefix).reset_index(drop=True))
    test_meta = r101_test["test_meta"].copy().reset_index(drop=True)
    tuning = r111_oof["tuning"]
    current_action_oof = build_current_oof_action()
    teacher_action_oof = normalize_rows_safe(0.80 * current_action_oof + 0.20 * art["experts_oof"]["v47_v64_oof_soft"])
    teacher_action_test = normalize_rows_safe(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    base_action_oof = normalize_rows_safe(0.925 * r111_oof["gru_action"] + 0.075 * teacher_action_oof)
    base_action_test = normalize_rows_safe(0.925 * r111_test["gru_action"] + 0.075 * teacher_action_test)

    r101_base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    r101_base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    _, tlp_oof = foldsafe_priors(rows, prefix, base_action_oof, r101_base_point_oof, mode="tlp", k=100, train_weight=0.50)
    _, tlp_test = test_priors(test_prefix, prefix, base_action_test, r101_base_point_test, mode="tlp", k=100, train_weight=0.50)

    ent_oof = -np.sum(np.clip(r101_base_point_oof, 1e-12, 1.0) * np.log(np.clip(r101_base_point_oof, 1e-12, 1.0)), axis=1)
    ent_test = -np.sum(np.clip(r101_base_point_test, 1e-12, 1.0) * np.log(np.clip(r101_base_point_test, 1e-12, 1.0)), axis=1)
    cut = np.quantile(ent_oof, 0.70)
    base_point_oof = r101_base_point_oof.copy()
    base_point_test = r101_base_point_test.copy()
    high_oof = ent_oof > cut
    high_test = ent_test > cut
    base_point_oof[high_oof] = normalize_rows_safe(0.98 * base_point_oof[high_oof] + 0.02 * tlp_oof[high_oof])
    base_point_test[high_test] = normalize_rows_safe(0.98 * base_point_test[high_test] + 0.02 * tlp_test[high_test])

    terminal_test, depth_test, side_test = full_structured_priors(prefix, test_prefix)
    r119_oof = r119_oof_prior(rows, prefix, base_action_oof)
    r119_test = action_conditioned_point_prior(test_prefix, prefix, base_action_test)
    _, r120_oof = r120_motif_oof(rows, prefix)
    r120_test = apply_motif_prior(test_prefix, prefix, "next_pointId", 10)
    q_oof = normalize_rows_safe((r119_oof[:, 7:10] + r120_oof[:, 7:10]) / 2.0 + 1e-9)
    q_test = normalize_rows_safe((r119_test[:, 7:10] + r120_test[:, 7:10]) / 2.0 + 1e-9)

    report = load_report()
    best = report["best"]
    calibrated_test = apply_point_hierarchy_calibration(
        base_point_test,
        terminal_prior=terminal_test,
        depth_prior=depth_test,
        side_prior=side_test,
        terminal_weight=float(best["terminal_weight"]),
        depth_weight=float(best["depth_weight"]),
        side_weight=float(best["side_weight"]),
    )
    selected_test = apply_long_side_redistribution(calibrated_test, q_test, alpha=float(best["long_alpha"]), long_thr=0.35)

    calibrated_oof = apply_point_hierarchy_calibration(
        base_point_oof,
        terminal_prior=np.zeros(len(base_point_oof)),
        depth_prior=np.ones((len(base_point_oof), 3)) / 3.0,
        side_prior=np.ones((len(base_point_oof), 3)) / 3.0,
        terminal_weight=0.0,
        depth_weight=0.0,
        side_weight=0.0,
    )
    _ = apply_long_side_redistribution(calibrated_oof, q_oof, alpha=0.0, long_thr=0.35)

    return {
        "meta": meta,
        "test_prefix": test_prefix,
        "test_meta": test_meta,
        "tuning": tuning,
        "base_point_test": base_point_test,
        "calibrated_point_test": calibrated_test,
        "selected_point_test": selected_test,
        "saved_selected_point_test": np.load(R180_OUTDIR / "r180_best_point_test.npy"),
        "best": best,
    }


def summarize_by_group(group: pd.Series, changed_mask: np.ndarray) -> list[dict]:
    tmp = pd.DataFrame({"group": group.astype(str), "changed": changed_mask.astype(int)})
    rows = []
    for key, part in tmp.groupby("group", dropna=False):
        rows.append({"group": key, "rows": int(len(part)), "changed_rows": int(part["changed"].sum()), "churn": float(part["changed"].mean())})
    return rows


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    state = build_r180_test_state()
    test_meta = state["test_meta"]
    tuning = state["tuning"]

    r67 = load_sub(R67_ANCHOR)
    rally_uids = r67["rally_uid"].astype(int).to_numpy()
    r180_raw = load_sub(Path(load_report()["generated"][0]["upload_path"]), rally_uids)
    r181_pkg = load_sub(R181_R180_NO_OLD, rally_uids)

    base_pred = point_pred(test_meta, state["base_point_test"], tuning)
    selected_pred = point_pred(test_meta, state["selected_point_test"], tuning)
    saved_selected_pred = point_pred(test_meta, state["saved_selected_point_test"], tuning)
    calibrated_pred = point_pred(test_meta, state["calibrated_point_test"], tuning)
    r67_point = r67["pointId"].astype(int).to_numpy()
    r180_raw_point = r180_raw["pointId"].astype(int).to_numpy()
    r181_point = r181_pkg["pointId"].astype(int).to_numpy()

    terminal_delta = np.abs(state["calibrated_point_test"][:, 0] - state["base_point_test"][:, 0])
    long_mass_calibrated = state["calibrated_point_test"][:, 7:10].sum(axis=1)
    long_mass_selected = state["selected_point_test"][:, 7:10].sum(axis=1)
    long_mass_base = state["base_point_test"][:, 7:10].sum(axis=1)

    metrics = {
        "r180_best_candidate": state["best"]["candidate"],
        "r180_selected_vs_r180_base_churn": churn(selected_pred, base_pred),
        "r180_selected_vs_r180_base_changed_rows": changed(selected_pred, base_pred),
        "r180_calibrated_vs_r180_base_churn": churn(calibrated_pred, base_pred),
        "r180_selected_vs_r180_calibrated_churn": churn(selected_pred, calibrated_pred),
        "r180_test_base_vs_r67_point_churn": churn(base_pred, r67_point),
        "r180_selected_vs_r67_point_churn": churn(selected_pred, r67_point),
        "r180_raw_submission_vs_recomputed_selected_churn": churn(r180_raw_point, selected_pred),
        "r181_package_vs_r180_raw_point_churn": churn(r181_point, r180_raw_point),
        "r181_package_vs_recomputed_selected_churn": churn(r181_point, selected_pred),
        "saved_npy_vs_recomputed_selected_argmax_churn": churn(saved_selected_pred, selected_pred),
        "saved_npy_max_abs_prob_delta": float(np.max(np.abs(state["saved_selected_point_test"] - state["selected_point_test"]))),
        "long_mass_selected_vs_calibrated_max_abs_delta": float(np.max(np.abs(long_mass_selected - long_mass_calibrated))),
        "long_mass_selected_vs_base_max_abs_delta": float(np.max(np.abs(long_mass_selected - long_mass_base))),
        "point0_abs_delta_mean": float(np.mean(terminal_delta)),
        "point0_abs_delta_p95": float(np.quantile(terminal_delta, 0.95)),
        "point0_abs_delta_max": float(np.max(terminal_delta)),
        "pass_transform_churn_le_5pct": bool(churn(selected_pred, base_pred) <= 0.05),
        "pass_package_matches_r180": bool(churn(r181_point, r180_raw_point) == 0.0 and churn(r180_raw_point, selected_pred) == 0.0),
    }
    metrics["r182_pass"] = bool(metrics["pass_transform_churn_le_5pct"] and metrics["pass_package_matches_r180"])

    pd.DataFrame([metrics]).to_csv(OUTDIR / "r182_summary_metrics.csv", index=False)
    pd.DataFrame(summarize_by_group(test_meta["prefix_len"], selected_pred != base_pred)).to_csv(OUTDIR / "r182_transform_churn_by_prefix_len.csv", index=False)
    pd.DataFrame(summarize_by_group(margin_bins(state["base_point_test"]), selected_pred != base_pred)).to_csv(OUTDIR / "r182_transform_churn_by_base_margin.csv", index=False)

    rows = pd.DataFrame(
        {
            "rally_uid": rally_uids,
            "r67_point": r67_point,
            "r180_base_point": base_pred,
            "r180_selected_point": selected_pred,
            "r180_raw_submission_point": r180_raw_point,
            "r181_package_point": r181_point,
            "selected_vs_base_changed": selected_pred != base_pred,
            "base_vs_r67_changed": base_pred != r67_point,
            "selected_vs_r67_changed": selected_pred != r67_point,
            "raw_vs_recomputed_selected_changed": r180_raw_point != selected_pred,
            "package_vs_raw_changed": r181_point != r180_raw_point,
        }
    )
    rows.to_csv(OUTDIR / "r182_row_level_audit.csv", index=False)

    report = {
        "metrics": metrics,
        "notes": [
            "R180 OOF churn in r180_report is against R180 internal base, not R67 point.",
            "R182 pass means selected-vs-base test churn is <= 5% and R181 packaged point exactly matches R180 selected output.",
            "No submission files are generated by this audit.",
        ],
    }
    (OUTDIR / "r182_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r182_report.md").write_text(
        "# R182 R180 Churn Audit\n\n"
        "## Summary\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in metrics.items())
        + "\n\n## Files\n\n"
        "- `r182_summary_metrics.csv`\n"
        "- `r182_transform_churn_by_prefix_len.csv`\n"
        "- `r182_transform_churn_by_base_margin.csv`\n"
        "- `r182_row_level_audit.csv`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r182_r180_churn_audit.py", "src/analysis/analysis_r182_r180_churn_audit.py")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
