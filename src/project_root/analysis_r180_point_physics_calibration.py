"""R180 point physics calibration experiment.

Point hierarchy is used as an auxiliary calibration layer over a direct
10-class point decoder.  The script never replaces the decoder with an
independent depth x side model.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from analysis_r1_oof_ensemble import compose_v3
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from analysis_r108_r110_r109_transductive import foldsafe_priors, test_priors
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r120_r123_sequence_meta import apply_motif_prior, r120_motif_oof
from analysis_r179_action_physics_hierarchy import normalize_rows_safe, point_depth, point_side
from baseline_lgbm import POINT_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR


OUTDIR = Path("r180_point_physics_calibration")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"

TERMINAL_WEIGHTS = [0.0025, 0.005, 0.01]
DEPTH_WEIGHTS = [0.0025, 0.005]
SIDE_WEIGHTS = [0.0, 0.0025]
LONG_ALPHAS = [0.03, 0.05, 0.075, 0.10]
POINT_CHURN_LIMIT = 0.08

DEPTH_GROUPS = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
}
SIDE_GROUPS = {
    1: [1, 4, 7],
    2: [2, 5, 8],
    3: [3, 6, 9],
}


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


def point_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode).astype(int)


def apply_group_mass(base: np.ndarray, target_mass: np.ndarray, groups: dict[int, list[int]], weight: float) -> np.ndarray:
    out = normalize_rows_safe(base)
    if weight <= 0:
        return out
    for j, cls_list in groups.items():
        cols = np.array(cls_list, dtype=int)
        current = out[:, cols].sum(axis=1)
        target = (1.0 - weight) * current + weight * target_mass[:, j - 1]
        ratio = np.divide(target, np.clip(current, 1e-12, None))
        out[:, cols] *= ratio[:, None]
    return normalize_rows_safe(out)


def apply_point0_calibration(base: np.ndarray, terminal_prior: np.ndarray, weight: float) -> np.ndarray:
    out = normalize_rows_safe(base)
    if weight <= 0:
        return out
    p0 = np.clip((1.0 - weight) * out[:, 0] + weight * terminal_prior, 1e-8, 1.0 - 1e-8)
    non = out[:, 1:].sum(axis=1)
    scale = (1.0 - p0) / np.clip(non, 1e-12, None)
    out[:, 1:] *= scale[:, None]
    out[:, 0] = p0
    return normalize_rows_safe(out)


def apply_point_hierarchy_calibration(
    base: np.ndarray,
    *,
    terminal_prior: np.ndarray,
    depth_prior: np.ndarray,
    side_prior: np.ndarray,
    terminal_weight: float,
    depth_weight: float,
    side_weight: float,
) -> np.ndarray:
    out = apply_point0_calibration(base, terminal_prior, terminal_weight)
    nonterminal = np.clip(1.0 - out[:, 0], 1e-12, None)
    depth_target = normalize_rows_safe(depth_prior) * nonterminal[:, None]
    out = apply_group_mass(out, depth_target, DEPTH_GROUPS, depth_weight)
    side_target = normalize_rows_safe(side_prior) * nonterminal[:, None]
    out = apply_group_mass(out, side_target, SIDE_GROUPS, side_weight)
    return normalize_rows_safe(out)


def apply_long_side_redistribution(base: np.ndarray, q_long: np.ndarray, *, alpha: float, long_thr: float = 0.35) -> np.ndarray:
    out = normalize_rows_safe(base)
    q = normalize_rows_safe(q_long)
    long_mass = out[:, 7:10].sum(axis=1)
    current = normalize_rows_safe(out[:, 7:10] + 1e-12)
    target = normalize_rows_safe((1.0 - alpha) * current + alpha * q)
    use = long_mass >= long_thr
    out[use, 7:10] = long_mass[use, None] * target[use]
    return normalize_rows_safe(out)


def add_point_physics_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r180_lag0_depth"] = [point_depth(v) for v in out["lag0_pointId"]]
    out["r180_lag0_side"] = [point_side(v) for v in out["lag0_pointId"]]
    out["r180_incoming_state"] = (
        out["phase_id"].astype(str)
        + "|a"
        + out["lag0_actionId"].astype(str)
        + "|d"
        + out["r180_lag0_depth"].astype(str)
        + "|s"
        + out["lag0_spinId"].astype(str)
        + "|t"
        + out["lag0_strengthId"].astype(str)
    )
    return out


def target_depth_matrix(values: pd.Series) -> np.ndarray:
    out = np.zeros((len(values), 3), dtype=float)
    for i, value in enumerate(values.astype(int)):
        d = point_depth(value)
        if d > 0:
            out[i, d - 1] = 1.0
    return out


def target_side_matrix(values: pd.Series) -> np.ndarray:
    out = np.zeros((len(values), 3), dtype=float)
    for i, value in enumerate(values.astype(int)):
        s = point_side(value)
        if s > 0:
            out[i, s - 1] = 1.0
    return out


def make_lookup(pool: pd.DataFrame, key_cols: list[str], target: np.ndarray, alpha: float, global_prior: np.ndarray) -> dict[tuple, np.ndarray]:
    tmp = pool.reset_index(drop=True)
    out: dict[tuple, np.ndarray] = {}
    for key, idx in tmp.groupby(key_cols, dropna=False).groups.items():
        vals = target[list(idx)]
        dist = normalize_rows_safe((vals.sum(axis=0) + alpha * global_prior)[None, :])[0]
        out[key if isinstance(key, tuple) else (key,)] = dist
    return out


def apply_lookup(rows: pd.DataFrame, lookups: list[tuple[list[str], dict[tuple, np.ndarray]]], global_prior: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), len(global_prior)), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        dist = None
        for cols, lookup in lookups:
            key = tuple(getattr(row, c) for c in cols)
            dist = lookup.get(key)
            if dist is not None:
                break
        out[i] = global_prior if dist is None else dist
    return normalize_rows_safe(out)


def foldsafe_structured_priors(rows: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    terminal = np.zeros(len(rows), dtype=float)
    depth = np.zeros((len(rows), 3), dtype=float)
    side = np.zeros((len(rows), 3), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].reset_index(drop=True)
        term_prior = float(pool["next_pointId"].eq(0).mean())
        terminal_lookup = make_lookup(pool, ["r180_incoming_state"], pool["next_pointId"].eq(0).astype(float).to_numpy()[:, None], 20.0, np.array([term_prior]))
        terminal[idx] = apply_lookup(rows.loc[idx], [(["r180_incoming_state"], terminal_lookup)], np.array([term_prior]))[:, 0]

        d_target = target_depth_matrix(pool["next_pointId"])
        d_global = normalize_rows_safe((d_target.sum(axis=0) + 1.0)[None, :])[0]
        d_lookups = [
            (["r180_incoming_state"], make_lookup(pool, ["r180_incoming_state"], d_target, 25.0, d_global)),
            (["phase_id", "lag0_actionId"], make_lookup(pool, ["phase_id", "lag0_actionId"], d_target, 50.0, d_global)),
        ]
        depth[idx] = apply_lookup(rows.loc[idx], d_lookups, d_global)

        s_target = target_side_matrix(pool["next_pointId"])
        s_global = normalize_rows_safe((s_target.sum(axis=0) + 1.0)[None, :])[0]
        s_lookups = [
            (["r180_incoming_state"], make_lookup(pool, ["r180_incoming_state"], s_target, 35.0, s_global)),
            (["phase_id", "lag0_pointId"], make_lookup(pool, ["phase_id", "lag0_pointId"], s_target, 60.0, s_global)),
        ]
        side[idx] = apply_lookup(rows.loc[idx], s_lookups, s_global)
    return terminal, depth, side


def full_structured_priors(prefix: pd.DataFrame, test_prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term_prior = float(prefix["next_pointId"].eq(0).mean())
    t_lookup = make_lookup(prefix, ["r180_incoming_state"], prefix["next_pointId"].eq(0).astype(float).to_numpy()[:, None], 20.0, np.array([term_prior]))
    terminal = apply_lookup(test_prefix, [(["r180_incoming_state"], t_lookup)], np.array([term_prior]))[:, 0]

    d_target = target_depth_matrix(prefix["next_pointId"])
    d_global = normalize_rows_safe((d_target.sum(axis=0) + 1.0)[None, :])[0]
    d_lookups = [
        (["r180_incoming_state"], make_lookup(prefix, ["r180_incoming_state"], d_target, 25.0, d_global)),
        (["phase_id", "lag0_actionId"], make_lookup(prefix, ["phase_id", "lag0_actionId"], d_target, 50.0, d_global)),
    ]
    depth = apply_lookup(test_prefix, d_lookups, d_global)

    s_target = target_side_matrix(prefix["next_pointId"])
    s_global = normalize_rows_safe((s_target.sum(axis=0) + 1.0)[None, :])[0]
    s_lookups = [
        (["r180_incoming_state"], make_lookup(prefix, ["r180_incoming_state"], s_target, 35.0, s_global)),
        (["phase_id", "lag0_pointId"], make_lookup(prefix, ["phase_id", "lag0_pointId"], s_target, 60.0, s_global)),
    ]
    side = apply_lookup(test_prefix, s_lookups, s_global)
    return terminal, depth, side


def per_class_f1(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    return {f"point{c}_f1": float(report[str(c)]["f1-score"]) for c in POINT_CLASSES}


def prefix_bin_report(meta: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    y = meta["next_pointId"].astype(int).to_numpy()
    bins = {
        "prefix_le2": meta["prefix_len"].astype(int).le(2).to_numpy(),
        "prefix_ge3": meta["prefix_len"].astype(int).ge(3).to_numpy(),
    }
    rows = []
    for name, mask in bins.items():
        rows.append({"slice": name, "rows": int(mask.sum()), "point_macro_f1": float(f1_score(y[mask], pred[mask], average="macro", labels=POINT_CLASSES, zero_division=0))})
    return rows


def eval_candidate(meta: pd.DataFrame, prob: np.ndarray, tuning, name: str, base_prob: np.ndarray, base_pred: np.ndarray) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base_pred)),
        "changed_rows": int(np.sum(pred != base_pred)),
        "long_mass_abs_delta_max": float(np.max(np.abs(prob[:, 7:10].sum(axis=1) - base_prob[:, 7:10].sum(axis=1)))),
    }
    rec.update(per_class_f1(y, pred))
    return rec


def write_submission(test_meta: pd.DataFrame, point_prob: np.ndarray, tuning, anchor: pd.DataFrame, name: str, extra: dict) -> dict:
    sub = anchor.copy()
    sub["pointId"] = point_pred(test_meta, point_prob, tuning)
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    shutil.copy2(path, upload_path)
    shutil.copy2(path, selected_path)
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    info.update(extra)
    return info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

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
    base_point_oof[ent_oof > cut] = normalize_rows_safe(0.98 * base_point_oof[ent_oof > cut] + 0.02 * tlp_oof[ent_oof > cut])
    base_point_test[ent_test > cut] = normalize_rows_safe(0.98 * base_point_test[ent_test > cut] + 0.02 * tlp_test[ent_test > cut])
    base_pred = point_pred(meta, base_point_oof, tuning)

    terminal_oof, depth_oof, side_oof = foldsafe_structured_priors(rows, prefix)
    terminal_test, depth_test, side_test = full_structured_priors(prefix, test_prefix)

    r119_oof = r119_oof_prior(rows, prefix, base_action_oof)
    r119_test = action_conditioned_point_prior(test_prefix, prefix, base_action_test)
    _, r120_oof = r120_motif_oof(rows, prefix)
    r120_test = apply_motif_prior(test_prefix, prefix, "next_pointId", 10)
    q_oof = normalize_rows_safe((r119_oof[:, 7:10] + r120_oof[:, 7:10]) / 2.0 + 1e-9)
    q_test = normalize_rows_safe((r119_test[:, 7:10] + r120_test[:, 7:10]) / 2.0 + 1e-9)

    search_rows = [eval_candidate(meta, base_point_oof, tuning, "r108_tlp_selective_base", base_point_oof, base_pred)]
    probs_oof: dict[str, np.ndarray] = {"r108_tlp_selective_base": base_point_oof}
    probs_test: dict[str, np.ndarray] = {"r108_tlp_selective_base": base_point_test}
    for tw in TERMINAL_WEIGHTS:
        for dw in DEPTH_WEIGHTS:
            for sw in SIDE_WEIGHTS:
                calibrated_oof = apply_point_hierarchy_calibration(
                    base_point_oof,
                    terminal_prior=terminal_oof,
                    depth_prior=depth_oof,
                    side_prior=side_oof,
                    terminal_weight=tw,
                    depth_weight=dw,
                    side_weight=sw,
                )
                calibrated_test = apply_point_hierarchy_calibration(
                    base_point_test,
                    terminal_prior=terminal_test,
                    depth_prior=depth_test,
                    side_prior=side_test,
                    terminal_weight=tw,
                    depth_weight=dw,
                    side_weight=sw,
                )
                for alpha in LONG_ALPHAS:
                    name = f"r180_t{str(tw).replace('.', 'p')}_d{str(dw).replace('.', 'p')}_s{str(sw).replace('.', 'p')}_l{str(alpha).replace('.', 'p')}"
                    prob_oof = apply_long_side_redistribution(calibrated_oof, q_oof, alpha=alpha, long_thr=0.35)
                    prob_test = apply_long_side_redistribution(calibrated_test, q_test, alpha=alpha, long_thr=0.35)
                    rec = eval_candidate(meta, prob_oof, tuning, name, base_point_oof, base_pred)
                    rec.update({"terminal_weight": tw, "depth_weight": dw, "side_weight": sw, "long_alpha": alpha, "tier": "clean" if rec["point_churn_vs_base"] <= POINT_CHURN_LIMIT else "probe_rejected"})
                    search_rows.append(rec)
                    probs_oof[name] = prob_oof
                    probs_test[name] = prob_test

    search = pd.DataFrame(search_rows).sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r180_point_calibration_search.csv", index=False)
    rejected = search[search["tier"].eq("probe_rejected")]
    if not rejected.empty:
        rejected.to_csv(OUTDIR / "r180_rejected_high_churn.csv", index=False)

    clean = search[search["tier"].ne("probe_rejected") & search["candidate"].ne("r108_tlp_selective_base")]
    if clean.empty:
        clean = search[search["candidate"].ne("r108_tlp_selective_base")]
    anchor = pd.read_csv(R67_ANCHOR)
    generated = []
    for _, rec in clean.head(6).iterrows():
        name = str(rec["candidate"])
        sub_name = f"submission_{name}_r67_anchor.csv"
        generated.append(write_submission(test_meta, probs_test[name], tuning, anchor, sub_name, rec.to_dict()))

    best_name = str(clean.iloc[0]["candidate"]) if not clean.empty else "r108_tlp_selective_base"
    np.save(OUTDIR / "r180_best_point_oof.npy", probs_oof[best_name])
    np.save(OUTDIR / "r180_best_point_test.npy", probs_test[best_name])
    best_pred = point_pred(meta, probs_oof[best_name], tuning)
    pd.DataFrame(prefix_bin_report(meta, best_pred)).to_csv(OUTDIR / "r180_prefix_bin_report.csv", index=False)
    report = {
        "base": search[search["candidate"].eq("r108_tlp_selective_base")].iloc[0].to_dict(),
        "best": clean.iloc[0].to_dict() if not clean.empty else search.iloc[0].to_dict(),
        "generated": generated,
        "rejected_high_churn_count": int(len(rejected)),
        "notes": [
            "R180 keeps the direct 10-class point decoder as anchor.",
            "Terminal/depth/side hierarchy is used only as calibration.",
            "Long-side redistribution preserves total pointId 7/8/9 mass for eligible rows.",
        ],
    }
    (OUTDIR / "r180_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r180_report.md").write_text(
        "# R180 Point Physics Calibration\n\n"
        f"- Best: `{best_name}`\n"
        f"- Rejected high-churn candidates: `{len(rejected)}`\n\n"
        "## Generated\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r180_point_physics_calibration.py", "src/analysis/analysis_r180_point_physics_calibration.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
