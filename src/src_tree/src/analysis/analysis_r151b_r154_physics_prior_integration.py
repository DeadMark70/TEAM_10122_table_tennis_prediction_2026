"""R151B/R154 safe physics-prior integration.

R151B:
  Map R151 external physics/canonical priors to AI CUP row-level diagnostic
  features. This uses only DeepMind robot table tennis and TT-MatchDynamics.

R154:
  Test extremely low-weight point-only prior blends. pointId=0 is preserved;
  external priors only redistribute non-terminal point classes 1..9.

No TTMATCH data is used here.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ANALYSIS_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from analysis_r1_oof_ensemble import compose_v3
from analysis_r108_r110_r109_transductive import foldsafe_priors, test_priors
from analysis_r120_r123_sequence_meta import apply_motif_prior, r120_motif_oof
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from baseline_lgbm import POINT_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR, normalize_rows


ROOT = Path(".")
OUTDIR = Path("r151b_r154_physics_prior_integration")
SELECTED_DIR = Path("submissions/selected")
R151_DIR = Path("r151_safe_physics_priors")
R67_ANCHOR = Path("upload_candidates_20260519/submission_r67_r63_blend_w0p2_current_point_server.csv")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def point_depth(point: int) -> str:
    if point in {1, 2, 3}:
        return "near"
    if point in {4, 5, 6}:
        return "mid"
    if point in {7, 8, 9}:
        return "far"
    return "terminal"


def point_side(point: int, orientation: str) -> str:
    if point == 0:
        return "terminal"
    idx = (point - 1) % 3
    if orientation == "direct":
        return ["left", "middle", "right"][idx]
    if orientation == "mirror":
        return ["right", "middle", "left"][idx]
    raise ValueError(f"unknown orientation: {orientation}")


def point_grid(point: int, orientation: str) -> str:
    if point == 0:
        return "terminal"
    return f"{point_side(point, orientation)}_{point_depth(point)}"


def normalize_nonzero_prior(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, 1e-9, None)
    return arr / arr.sum(axis=1, keepdims=True)


def static_grid_prior_from_r151(source_filter: str, orientation: str, depth_only: bool = False) -> np.ndarray:
    combined = pd.read_csv(R151_DIR / "combined_canonical_grid_prior.csv")
    if source_filter == "deepmind":
        sub = combined[combined["source"].eq("deepmind_robot")].copy()
    elif source_filter == "matchdynamics":
        sub = combined[combined["source"].eq("tt_matchdynamics")].copy()
    elif source_filter == "combined":
        sub = combined.copy()
    else:
        raise ValueError(source_filter)

    if depth_only:
        sub["depth"] = sub["canonical_grid"].astype(str).str.split("_").str[-1]
        rates = sub.groupby("depth")["rate"].sum().to_dict()
        vals = np.array([rates.get(point_depth(p), 0.0) for p in range(1, 10)], dtype=float)
        # Preserve only depth preference; side is uniform inside each depth.
        for depth in ["near", "mid", "far"]:
            idx = [i for i, p in enumerate(range(1, 10)) if point_depth(p) == depth]
            total = vals[idx].sum()
            if total > 0:
                vals[idx] = total / len(idx)
    else:
        rates = sub.groupby("canonical_grid")["rate"].sum().to_dict()
        vals = np.array([rates.get(point_grid(p, orientation), 0.0) for p in range(1, 10)], dtype=float)
    vals = np.clip(vals, 1e-6, None)
    vals = vals / vals.sum()
    return vals


def md_conditioned_prior(rows: pd.DataFrame, orientation: str, fallback: np.ndarray) -> np.ndarray:
    md = pd.read_csv(R151_DIR / "tt_matchdynamics_grid_spin_hand_prior.csv")
    md_rates: dict[tuple[str, str, str], float] = {}
    for _, row in md.iterrows():
        md_rates[(str(row["md_spin_family"]), str(row["md_hand_family"]), str(row["md_grid_3x3"]))] = float(row["rate"])

    def spin_family(spin: int) -> str | None:
        # Conservative common mapping used only for weak priors.
        if spin == 1:
            return "topspin"
        if spin == 2:
            return "backspin"
        return None

    def hand_family(hand: int) -> str | None:
        if hand == 1:
            return "forehand"
        if hand == 2:
            return "backhand"
        return None

    priors = np.tile(fallback.reshape(1, -1), (len(rows), 1))
    spin_col = "lag0_spinId" if "lag0_spinId" in rows.columns else "spinId"
    hand_col = "lag0_handId" if "lag0_handId" in rows.columns else "handId"
    for i, row in enumerate(rows.itertuples(index=False)):
        spin = spin_family(int(getattr(row, spin_col, 0)))
        hand = hand_family(int(getattr(row, hand_col, 0)))
        if spin is None or hand is None:
            continue
        vals = np.array([md_rates.get((spin, hand, point_grid(p, orientation)), 0.0) for p in range(1, 10)], dtype=float)
        if vals.sum() > 0:
            priors[i] = vals / vals.sum()
    return normalize_nonzero_prior(priors)


def blend_nonterminal_point(base: np.ndarray, prior9: np.ndarray, alpha: float) -> np.ndarray:
    base = normalize_rows(base)
    prior9 = normalize_nonzero_prior(prior9)
    out = base.copy()
    nonzero_mass = np.clip(1.0 - out[:, 0], 1e-9, 1.0)
    base9 = out[:, 1:10] / nonzero_mass[:, None]
    q = normalize_nonzero_prior((1.0 - alpha) * base9 + alpha * prior9)
    out[:, 1:10] = nonzero_mass[:, None] * q
    out[:, 0] = base[:, 0]
    return normalize_rows(out)


def point_pred(meta: pd.DataFrame, point_prob: np.ndarray, tuning: GrUTuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode).astype(int)


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict | None = None) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    base = point_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    for k in POINT_CLASSES:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def make_row_features(rows: pd.DataFrame, priors: dict[str, np.ndarray], split: str) -> pd.DataFrame:
    out = rows[["rally_uid", "match", "prefix_len"]].copy()
    out["split"] = split
    for name, prior in priors.items():
        entropy = -np.sum(np.clip(prior, 1e-12, 1.0) * np.log(np.clip(prior, 1e-12, 1.0)), axis=1)
        out[f"{name}_top_point_nonzero"] = np.argmax(prior, axis=1) + 1
        out[f"{name}_top_prob"] = np.max(prior, axis=1)
        out[f"{name}_entropy"] = entropy
        out[f"{name}_near_mass"] = prior[:, 0:3].sum(axis=1)
        out[f"{name}_mid_mass"] = prior[:, 3:6].sum(axis=1)
        out[f"{name}_far_mass"] = prior[:, 6:9].sum(axis=1)
    return out


def write_submission(test_meta: pd.DataFrame, point_pred_values: np.ndarray, name: str) -> dict:
    anchor = pd.read_csv(R67_ANCHOR)
    if not anchor["rally_uid"].equals(test_meta["rally_uid"]):
        anchor = anchor.set_index("rally_uid").loc[test_meta["rally_uid"].to_numpy()].reset_index()
    sub = anchor.copy()
    sub["pointId"] = point_pred_values.astype(int)
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    upload.write_bytes(path.read_bytes())
    selected.write_bytes(path.read_bytes())
    return {"path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, _features = prepare_prefix_features()
    r101_oof = load_pickle("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
    r101_test = load_pickle("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
    r111_oof = load_pickle("r111_remaining_moe_gru/oof_proba_r111.pkl")
    v3_oof = load_pickle("oof_proba_v3.pkl")

    meta = art["valid_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix).reset_index(drop=True)
    test_meta = r101_test["test_meta"].copy().reset_index(drop=True)
    tuning = r111_oof["tuning"]

    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])

    r101_base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    r101_base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    _, tlp_p_oof = foldsafe_priors(rows, prefix, np.zeros((len(rows), 19)), r101_base_point_oof, mode="tlp", k=100, train_weight=0.50)
    _, tlp_p_test = test_priors(test_prefix, prefix, np.zeros((len(test_prefix), 19)), r101_base_point_test, mode="tlp", k=100, train_weight=0.50)
    oof_entropy = -np.sum(np.clip(r101_base_point_oof, 1e-12, 1.0) * np.log(np.clip(r101_base_point_oof, 1e-12, 1.0)), axis=1)
    test_entropy = -np.sum(np.clip(r101_base_point_test, 1e-12, 1.0) * np.log(np.clip(r101_base_point_test, 1e-12, 1.0)), axis=1)
    high_oof = oof_entropy > np.quantile(oof_entropy, 0.70)
    high_test = test_entropy > np.quantile(oof_entropy, 0.70)
    base_point_oof = r101_base_point_oof.copy()
    base_point_test = r101_base_point_test.copy()
    base_point_oof[high_oof] = normalize_rows(0.98 * base_point_oof[high_oof] + 0.02 * tlp_p_oof[high_oof])
    base_point_test[high_test] = normalize_rows(0.98 * base_point_test[high_test] + 0.02 * tlp_p_test[high_test])

    static_priors = {
        "dm_depth": static_grid_prior_from_r151("deepmind", "direct", depth_only=True),
        "dm_direct": static_grid_prior_from_r151("deepmind", "direct", depth_only=False),
        "dm_mirror": static_grid_prior_from_r151("deepmind", "mirror", depth_only=False),
        "md_depth": static_grid_prior_from_r151("matchdynamics", "direct", depth_only=True),
        "md_direct": static_grid_prior_from_r151("matchdynamics", "direct", depth_only=False),
        "md_mirror": static_grid_prior_from_r151("matchdynamics", "mirror", depth_only=False),
        "combined_depth": static_grid_prior_from_r151("combined", "direct", depth_only=True),
        "combined_direct": static_grid_prior_from_r151("combined", "direct", depth_only=False),
        "combined_mirror": static_grid_prior_from_r151("combined", "mirror", depth_only=False),
    }

    priors_oof: dict[str, np.ndarray] = {}
    priors_test: dict[str, np.ndarray] = {}
    for name, vec in static_priors.items():
        priors_oof[name] = np.tile(vec.reshape(1, -1), (len(rows), 1))
        priors_test[name] = np.tile(vec.reshape(1, -1), (len(test_prefix), 1))
    priors_oof["md_cond_direct"] = md_conditioned_prior(rows, "direct", static_priors["md_direct"])
    priors_test["md_cond_direct"] = md_conditioned_prior(test_prefix, "direct", static_priors["md_direct"])
    priors_oof["md_cond_mirror"] = md_conditioned_prior(rows, "mirror", static_priors["md_mirror"])
    priors_test["md_cond_mirror"] = md_conditioned_prior(test_prefix, "mirror", static_priors["md_mirror"])

    make_row_features(rows, priors_oof, "oof").to_csv(OUTDIR / "r151b_oof_external_prior_features.csv", index=False)
    make_row_features(test_prefix, priors_test, "test").to_csv(OUTDIR / "r151b_test_external_prior_features.csv", index=False)

    search_rows: list[dict] = []
    base_rec = eval_point(meta, base_point_oof, base_point_oof, tuning, "r108_tlp_base")
    search_rows.append({**base_rec, "family": "base", "alpha": 0.0, "prior": "none"})
    generated: list[dict] = []

    for prior_name in [
        "dm_depth",
        "dm_direct",
        "dm_mirror",
        "md_depth",
        "md_direct",
        "md_mirror",
        "md_cond_direct",
        "md_cond_mirror",
        "combined_depth",
    ]:
        for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03]:
            prob_oof = blend_nonterminal_point(base_point_oof, priors_oof[prior_name], alpha)
            prob_test = blend_nonterminal_point(base_point_test, priors_test[prior_name], alpha)
            name = f"r154_{prior_name}_a{clean_float(alpha)}"
            rec = eval_point(meta, prob_oof, base_point_oof, tuning, name, {"family": "r154", "prior": prior_name, "alpha": alpha})
            search_rows.append(rec)
            if alpha in {0.005, 0.01, 0.02} and prior_name in {
                "dm_depth",
                "dm_direct",
                "dm_mirror",
                "md_mirror",
                "md_cond_direct",
                "md_cond_mirror",
                "combined_depth",
            }:
                pred_test = point_pred(test_meta, prob_test, tuning)
                sub_name = f"submission_{name}_r67_anchor.csv"
                info = write_submission(test_meta, pred_test, sub_name)
                generated.append({**rec, "candidate": sub_name, **info})

    search = pd.DataFrame(search_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True])
    search.to_csv(OUTDIR / "r154_physics_prior_search.csv", index=False)
    pd.DataFrame(generated).to_csv(OUTDIR / "r154_generated_candidates.csv", index=False)

    best = search.head(10).to_dict(orient="records")
    for cand in ["r154_dm_depth_a0p01", "r154_md_cond_direct_a0p01", "r154_md_cond_mirror_a0p01", "r154_combined_depth_a0p01"]:
        row = search[search["candidate"].eq(cand)]
        if not row.empty:
            pred = point_pred(meta, blend_nonterminal_point(base_point_oof, priors_oof[row.iloc[0]["prior"]], float(row.iloc[0]["alpha"])), tuning)
            report = classification_report(meta["next_pointId"].astype(int), pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
            pd.DataFrame(report).T.to_csv(OUTDIR / f"class_report_{cand}.csv")

    report = {
        "base_point_macro_f1": float(base_rec["point_macro_f1"]),
        "best": best,
        "generated_count": len(generated),
        "safety": "No TTMATCH, no AI CUP test labels, point0 preserved, non-terminal point-only prior.",
        "outputs": {
            "search": str(OUTDIR / "r154_physics_prior_search.csv"),
            "generated": str(OUTDIR / "r154_generated_candidates.csv"),
            "oof_features": str(OUTDIR / "r151b_oof_external_prior_features.csv"),
            "test_features": str(OUTDIR / "r151b_test_external_prior_features.csv"),
        },
    }
    (OUTDIR / "r151b_r154_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r151b_r154_report.md").write_text(
        "# R151B/R154 Physics Prior Integration\n\n"
        f"- Base point Macro-F1: `{report['base_point_macro_f1']:.6f}`\n"
        f"- Generated candidates: `{len(generated)}`\n"
        f"- Safety: {report['safety']}\n\n"
        "## Top Search Rows\n\n"
        + search.head(12)[["candidate", "point_macro_f1", "point_churn_vs_base", "changed_rows", "prior", "alpha"]].to_csv(index=False)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
