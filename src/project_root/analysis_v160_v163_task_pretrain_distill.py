"""V160-V163 private-safe task-aligned pretraining/distillation probes.

V160:
  AICUP task-adaptive transition pretraining proxy.  Build fold-safe
  empirical priors for next action/point from AICUP train prefixes, and use
  train + test_new observed internal transitions for test-time priors.  This
  uses no hidden target and no old-test server label.

V161:
  Coarse external pretraining proxy.  Combine V160 internal priors with the
  OpenTTGames action-family prior from R155.

V162:
  Hierarchical point fine-tune proxy.  Preserve point0 mass, then refine
  non-terminal point distribution using V160 and safe physics priors.

V163:
  No-old final distillation ensemble candidates.  Produce upload-ready
  submissions using action/point outputs from V160-V162 and robust no-old
  server choices.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT_DIR = Path(__file__).resolve().parent
SRC_ANALYSIS = ROOT_DIR / "src" / "analysis"
for p in [ROOT_DIR, SRC_ANALYSIS]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features  # noqa: E402
from analysis_r155_r156_external_pretrain_priors import (  # noqa: E402
    OPEN_EVENTS,
    blend_action,
    estimate_opentt_aux_priors,
    opentt_action_prior,
)
from analysis_r151b_r154_physics_prior_integration import (  # noqa: E402
    blend_nonterminal_point,
    point_pred,
)
from analysis_r157_r158_physics_pretraining_smoke import (  # noqa: E402
    build_external_prototypes,
    build_mpm_tables,
    load_external_states,
    mpm_prior_for_rows,
    prototype_prior_for_rows,
)
from analysis_r116_r119_point_server import apply_predictions  # noqa: E402
from baseline_lgbm import (  # noqa: E402
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    validate_raw_data,
)
from analysis_r7_phase_features import add_phase_features  # noqa: E402
from analysis_r57_player_style_clustering import add_player_id_features  # noqa: E402
from baseline_v3 import apply_segmented_multipliers  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


OUTDIR = Path("v160_v163_task_pretrain_distill")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")

R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R121_MIN = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R119_POINT = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R154_BEST = UPLOAD_DIR / "submission_r154_md_mirror_a0p01_r67_anchor.csv"

R101_OOF = Path("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
R101_TEST = Path("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
R111_OOF = Path("r111_remaining_moe_gru/oof_proba_r111.pkl")
R111_TEST = Path("r111_remaining_moe_gru/test_proba_r111.pkl")


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def log_blend(base: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    base = normalize_rows(np.clip(base, 1e-9, None))
    prior = normalize_rows(np.clip(prior, 1e-9, None))
    out = np.exp((1.0 - alpha) * np.log(base) + alpha * np.log(prior))
    return normalize_rows(out)


def action_pred(meta: pd.DataFrame, prob: np.ndarray, tuning: GrUTuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode).astype(int)


def eval_action(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict | None = None) -> dict:
    y = meta["next_actionId"].astype(int).to_numpy()
    pred = action_pred(meta, prob, tuning)
    base = action_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "action_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=ACTION_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 5, 8, 9, 12, 14]:
        rec[f"f1_action_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_action_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict | None = None) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    base = point_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 1, 3, 5, 7, 8, 9]:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_point_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def fold_matches(meta: pd.DataFrame, fold: int) -> set:
    return set(meta.loc[meta["fold"].eq(fold), "match"].astype(int).unique())


def ensure_fold(meta: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Older neural OOF files did not persist fold ids; reconstruct by match."""
    out = meta.copy()
    if "fold" in out.columns:
        return out
    out["fold"] = -1
    groups = out["match"].astype(int).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(out), 1), dtype=float)
    dummy_y = np.zeros(len(out), dtype=int)
    for fold, (_, valid_idx) in enumerate(splitter.split(dummy_x, dummy_y, groups=groups)):
        out.iloc[valid_idx, out.columns.get_loc("fold")] = fold
    if (out["fold"] < 0).any():
        raise ValueError("Failed to reconstruct fold ids")
    return out


KEYS = [
    ["phase_id", "prefix_len", "lag0_actionId", "lag0_pointId", "lag0_spinId"],
    ["phase_id", "lag0_actionId", "lag0_pointId", "lag0_spinId"],
    ["phase_id", "lag0_actionId", "lag0_pointId"],
    ["phase_id", "lag0_actionId", "lag0_spinId"],
    ["phase_id", "lag0_actionId"],
    ["phase_id", "prefix_len"],
    ["phase_id"],
]


def row_key(row, cols: list[str]) -> tuple:
    return tuple(int(getattr(row, c)) for c in cols)


def fit_prior_tables(df: pd.DataFrame, target_col: str, classes: list[int], alpha: float = 35.0) -> dict:
    y = df[target_col].astype(int).to_numpy()
    global_counts = np.array([np.sum(y == c) for c in classes], dtype=float) + 1.0
    global_prior = global_counts / global_counts.sum()
    tables = []
    for cols in KEYS:
        counter: dict[tuple, np.ndarray] = {}
        support: Counter = Counter()
        for row in df[list(cols) + [target_col]].itertuples(index=False):
            key = row_key(row, cols)
            if key not in counter:
                counter[key] = np.zeros(len(classes), dtype=float)
            val = int(getattr(row, target_col))
            if val in classes:
                counter[key][classes.index(val)] += 1.0
                support[key] += 1
        prob = {k: (v + alpha * global_prior) / (float(v.sum()) + alpha) for k, v in counter.items()}
        tables.append({"cols": cols, "prob": prob, "support": support})
    return {"global": global_prior, "tables": tables, "classes": classes}


def predict_prior(rows: pd.DataFrame, fit: dict, min_support: int = 8) -> np.ndarray:
    out = np.zeros((len(rows), len(fit["classes"])), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        found = None
        for table in fit["tables"]:
            key = row_key(row, table["cols"])
            if table["support"].get(key, 0) >= min_support:
                found = table["prob"][key]
                break
        out[i] = fit["global"] if found is None else found
    return normalize_rows(out)


def foldsafe_internal_priors(prefix: pd.DataFrame, valid_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    action_prior = np.zeros((len(valid_rows), len(ACTION_CLASSES)), dtype=float)
    point_prior = np.zeros((len(valid_rows), len(POINT_CLASSES)), dtype=float)
    for fold in sorted(valid_rows["fold"].astype(int).unique()):
        matches = fold_matches(valid_rows, fold)
        train_part = prefix[~prefix["match"].astype(int).isin(matches)].copy()
        valid_part = valid_rows[valid_rows["fold"].astype(int).eq(fold)].copy()
        fit_a = fit_prior_tables(train_part, "next_actionId", ACTION_CLASSES)
        fit_p = fit_prior_tables(train_part, "next_pointId", POINT_CLASSES)
        idx = valid_part.index.to_numpy()
        action_prior[idx] = predict_prior(valid_part, fit_a)
        point_prior[idx] = predict_prior(valid_part, fit_p)
    return normalize_rows(action_prior), normalize_rows(point_prior)


def build_test_internal_prefixes(test_raw: pd.DataFrame) -> pd.DataFrame:
    test_tmp = test_raw.copy()
    test_tmp["serverGetPoint"] = 0
    test_tmp = add_role_and_score_features(test_tmp)
    internal = build_train_prefix_table(test_tmp, 6)
    internal = add_phase_features(internal, test_tmp)
    internal = add_player_id_features(internal, test_tmp)
    return internal


def full_internal_priors(prefix: pd.DataFrame, test_prefix: pd.DataFrame, test_internal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fit_src = pd.concat([prefix, test_internal], axis=0, ignore_index=True)
    fit_a = fit_prior_tables(fit_src, "next_actionId", ACTION_CLASSES)
    fit_p = fit_prior_tables(fit_src, "next_pointId", POINT_CLASSES)
    return predict_prior(test_prefix, fit_a), predict_prior(test_prefix, fit_p)


def load_submission(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, rally_uids: np.ndarray, action: np.ndarray, point: np.ndarray, server: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(
        {
            "rally_uid": rally_uids.astype(int),
            "actionId": action.astype(int),
            "pointId": point.astype(int),
            "serverGetPoint": np.round(np.clip(server.astype(float), 1e-6, 1.0 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"candidate": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)

    train, test, prefix, test_prefix, _ = prepare_prefix_features()

    r111_oof = load_pickle(R111_OOF)
    r111_test = load_pickle(R111_TEST)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)

    tuning: GrUTuning = r111_oof["tuning"]
    valid_meta = ensure_fold(r111_oof["valid_meta"])
    rows = align_prefix_meta(valid_meta, prefix).reset_index(drop=True)
    test_rows = test_prefix.reset_index(drop=True)
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])
    base_point_oof = normalize_rows(0.50 * r111_oof["gru_point"] + 0.50 * r101_oof["gru_point"])
    base_point_test = normalize_rows(0.50 * r111_test["gru_point"] + 0.50 * r101_test["gru_point"])

    base_action_pred = action_pred(rows, base_action_oof, tuning)
    base_point_pred = point_pred(rows, base_point_oof, tuning)
    base_action_metric = float(f1_score(rows["next_actionId"].astype(int), base_action_pred, labels=ACTION_CLASSES, average="macro", zero_division=0))
    base_point_metric = float(f1_score(rows["next_pointId"].astype(int), base_point_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_server_auc = float(roc_auc_score(rows["serverGetPoint"].astype(int), 0.5 * r111_oof["gru_server"] + 0.5 * r101_oof["gru_server"]))

    # V160: internal task-adaptive transition priors.
    v160_action_prior_oof, v160_point_prior_oof = foldsafe_internal_priors(prefix, rows)
    test_internal = build_test_internal_prefixes(test_raw)
    v160_action_prior_test, v160_point_prior_test = full_internal_priors(prefix, test_rows, test_internal)
    test_internal_summary = {
        "test_internal_transition_rows": int(len(test_internal)),
        "unique_test_internal_rallies": int(test_internal["rally_uid"].nunique()) if len(test_internal) else 0,
    }

    v160_action_rows = []
    v160_action_probs = {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075]:
        prob = log_blend(base_action_oof, v160_action_prior_oof, alpha)
        name = f"v160_internal_action_a{clean_float(alpha)}"
        rec = eval_action(rows, prob, base_action_oof, tuning, name, {"alpha": alpha})
        v160_action_rows.append(rec)
        v160_action_probs[name] = {
            "oof": prob,
            "test": log_blend(base_action_test, v160_action_prior_test, alpha),
        }
    v160_action_search = pd.DataFrame(v160_action_rows).sort_values("action_macro_f1", ascending=False).reset_index(drop=True)
    v160_action_search.to_csv(OUTDIR / "v160_internal_action_search.csv", index=False)

    v160_point_rows = []
    v160_point_probs = {}
    for mode in ["full", "nonterminal"]:
        for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]:
            if mode == "full":
                prob = log_blend(base_point_oof, v160_point_prior_oof, alpha)
                test_prob = log_blend(base_point_test, v160_point_prior_test, alpha)
            else:
                prob = blend_nonterminal_point(base_point_oof, v160_point_prior_oof[:, 1:10], alpha)
                test_prob = blend_nonterminal_point(base_point_test, v160_point_prior_test[:, 1:10], alpha)
            name = f"v160_internal_point_{mode}_a{clean_float(alpha)}"
            rec = eval_point(rows, prob, base_point_oof, tuning, name, {"alpha": alpha, "mode": mode})
            v160_point_rows.append(rec)
            v160_point_probs[name] = {"oof": prob, "test": test_prob}
    v160_point_search = pd.DataFrame(v160_point_rows).sort_values("point_macro_f1", ascending=False).reset_index(drop=True)
    v160_point_search.to_csv(OUTDIR / "v160_internal_point_search.csv", index=False)

    # V161: add external coarse OpenTT action-family prior.
    if OPEN_EVENTS.exists():
        opentt_events = pd.read_csv(OPEN_EVENTS)
        opentt_priors, opentt_table = estimate_opentt_aux_priors(opentt_events)
        opentt_table.to_csv(OUTDIR / "v161_opentt_prior_table.csv", index=False)
        ext_action_oof = opentt_action_prior(rows, base_action_oof, opentt_priors)
        ext_action_test = opentt_action_prior(test_rows, base_action_test, opentt_priors)
    else:
        opentt_priors = {"segments_count": 0, "transition_rows": 0}
        ext_action_oof = base_action_oof.copy()
        ext_action_test = base_action_test.copy()

    v161_rows = []
    v161_action_probs = {}
    for ai in [0.005, 0.01, 0.02, 0.03]:
        internal_oof = log_blend(base_action_oof, v160_action_prior_oof, ai)
        internal_test = log_blend(base_action_test, v160_action_prior_test, ai)
        for ae in [0.005, 0.01, 0.02, 0.03, 0.05]:
            prob = blend_action(internal_oof, ext_action_oof, ae)
            test_prob = blend_action(internal_test, ext_action_test, ae)
            name = f"v161_internal_a{clean_float(ai)}_opentt_a{clean_float(ae)}"
            rec = eval_action(rows, prob, base_action_oof, tuning, name, {"alpha_internal": ai, "alpha_external": ae})
            v161_rows.append(rec)
            v161_action_probs[name] = {"oof": prob, "test": test_prob}
    v161_search = pd.DataFrame(v161_rows).sort_values("action_macro_f1", ascending=False).reset_index(drop=True)
    v161_search.to_csv(OUTDIR / "v161_external_coarse_action_search.csv", index=False)

    # V162: hierarchical point fine-tune proxy with safe physics priors.
    dm, md = load_external_states()
    mpm_tables, mpm_global, _ = build_mpm_tables(dm, mirror=True)
    mpm_oof = mpm_prior_for_rows(rows, mpm_tables, mpm_global)
    mpm_test = mpm_prior_for_rows(test_rows, mpm_tables, mpm_global)
    proto = build_external_prototypes(dm, md, mirror=False)
    proto_oof = prototype_prior_for_rows(rows, proto, tau=0.35, k=250)
    proto_test = prototype_prior_for_rows(test_rows, proto, tau=0.35, k=250)

    v162_rows = []
    v162_point_probs = {}
    for a_internal in [0.0, 0.0025, 0.005, 0.01, 0.02]:
        p_oof = base_point_oof.copy()
        p_test = base_point_test.copy()
        if a_internal > 0:
            p_oof = blend_nonterminal_point(p_oof, v160_point_prior_oof[:, 1:10], a_internal)
            p_test = blend_nonterminal_point(p_test, v160_point_prior_test[:, 1:10], a_internal)
        for a_mpm in [0.0, 0.0025, 0.005, 0.01, 0.02]:
            p2_oof = p_oof if a_mpm == 0 else blend_nonterminal_point(p_oof, mpm_oof, a_mpm)
            p2_test = p_test if a_mpm == 0 else blend_nonterminal_point(p_test, mpm_test, a_mpm)
            for a_proto in [0.0, 0.001, 0.0025, 0.005]:
                if a_internal == 0 and a_mpm == 0 and a_proto == 0:
                    continue
                prob = p2_oof if a_proto == 0 else blend_nonterminal_point(p2_oof, proto_oof, a_proto)
                test_prob = p2_test if a_proto == 0 else blend_nonterminal_point(p2_test, proto_test, a_proto)
                name = f"v162_hpoint_i{clean_float(a_internal)}_m{clean_float(a_mpm)}_p{clean_float(a_proto)}"
                rec = eval_point(
                    rows,
                    prob,
                    base_point_oof,
                    tuning,
                    name,
                    {"alpha_internal": a_internal, "alpha_mpm": a_mpm, "alpha_proto": a_proto},
                )
                v162_rows.append(rec)
                v162_point_probs[name] = {"oof": prob, "test": test_prob}
    v162_search = pd.DataFrame(v162_rows).sort_values("point_macro_f1", ascending=False).reset_index(drop=True)
    v162_search.to_csv(OUTDIR / "v162_hierarchical_point_search.csv", index=False)

    # V163: no-old ensemble candidates.
    r67_sub = load_submission(R67_ANCHOR, rally_uids)
    r121_sub = load_submission(R121_MIN, rally_uids) if R121_MIN.exists() else r67_sub
    r119_sub = load_submission(R119_POINT, rally_uids) if R119_POINT.exists() else r67_sub
    r154_sub = load_submission(R154_BEST, rally_uids) if R154_BEST.exists() else r67_sub

    best_v160_action = v160_action_search.iloc[0]["candidate"]
    best_v161_action = v161_search.iloc[0]["candidate"]
    best_v160_point = v160_point_search.iloc[0]["candidate"]
    best_v162_point = v162_search.iloc[0]["candidate"]

    action_sources = {
        "r67_public": r67_sub["actionId"].astype(int).to_numpy(),
        "v160_internal": action_pred(test_rows, v160_action_probs[best_v160_action]["test"], tuning),
        "v161_internal_opentt": action_pred(test_rows, v161_action_probs[best_v161_action]["test"], tuning),
    }
    point_sources = {
        "r67_v3": r67_sub["pointId"].astype(int).to_numpy(),
        "r119_public_point": r119_sub["pointId"].astype(int).to_numpy(),
        "r154_safe_physics": r154_sub["pointId"].astype(int).to_numpy(),
        "v160_internal_point": point_pred(test_rows, v160_point_probs[best_v160_point]["test"], tuning),
        "v162_hier_point": point_pred(test_rows, v162_point_probs[best_v162_point]["test"], tuning),
    }
    server_sources = {
        "r67_current_server": r67_sub["serverGetPoint"].astype(float).to_numpy(),
        "r121_min_w0p2": r121_sub["serverGetPoint"].astype(float).to_numpy(),
    }

    known_metrics = {
        "r67_public_action": 0.29700342743872726,
        "r67_v3_point": 0.20465533648195663,
        "r67_current_server": 0.6188096352755093,
        "r119_point_w0p05": 0.21324624407399356,
        "r154_md_mirror_a0p01_point": 0.214022,
        "r121_min_w0p2_server": 0.6225400681884332,
    }

    generated = []
    combos = [
        ("r67_public", "v162_hier_point", "r121_min_w0p2"),
        ("r67_public", "v162_hier_point", "r67_current_server"),
        ("r67_public", "r154_safe_physics", "r121_min_w0p2"),
        ("r67_public", "r119_public_point", "r121_min_w0p2"),
        ("v161_internal_opentt", "v162_hier_point", "r121_min_w0p2"),
        ("v160_internal", "v162_hier_point", "r121_min_w0p2"),
    ]
    for a_key, p_key, s_key in combos:
        name = f"submission_v163_no_old__a{a_key}__p{p_key}__s{s_key}.csv"
        info = write_submission(name, rally_uids, action_sources[a_key], point_sources[p_key], server_sources[s_key])
        info.update({"action_source": a_key, "point_source": p_key, "server_source": s_key})
        generated.append(info)

    # Search table with local model metrics where available.  For public R67
    # action/source rows use known metrics, because R67 is a public-validated
    # class-only branch without OOF probabilities.
    source_metric_rows = []
    source_metric_rows.append({"source": "v160_internal", "task": "action", **v160_action_search.iloc[0].to_dict()})
    source_metric_rows.append({"source": "v161_internal_opentt", "task": "action", **v161_search.iloc[0].to_dict()})
    source_metric_rows.append({"source": "v160_internal_point", "task": "point", **v160_point_search.iloc[0].to_dict()})
    source_metric_rows.append({"source": "v162_hier_point", "task": "point", **v162_search.iloc[0].to_dict()})
    source_metric_rows.append({"source": "base_r111_r101_action", "task": "action", "action_macro_f1": base_action_metric})
    source_metric_rows.append({"source": "base_r111_r101_point", "task": "point", "point_macro_f1": base_point_metric})
    pd.DataFrame(source_metric_rows).to_csv(OUTDIR / "v160_v163_source_metric_summary.csv", index=False)

    report = {
        "safety": {
            "uses_old_test_server_labels": False,
            "uses_test_new_hidden_targets": False,
            "uses_test_new_observed_internal_transitions": True,
            "test_internal_summary": test_internal_summary,
        },
        "base_metrics": {
            "base_action_macro_f1": base_action_metric,
            "base_point_macro_f1": base_point_metric,
            "base_server_auc": base_server_auc,
        },
        "best_v160_action": v160_action_search.head(10).to_dict(orient="records"),
        "best_v160_point": v160_point_search.head(10).to_dict(orient="records"),
        "best_v161_action": v161_search.head(10).to_dict(orient="records"),
        "best_v162_point": v162_search.head(10).to_dict(orient="records"),
        "generated": generated,
        "known_public_safe_components": known_metrics,
        "opentt": {
            "segments": int(opentt_priors.get("segments_count", 0)),
            "transition_rows": int(opentt_priors.get("transition_rows", 0)),
        },
        "notes": [
            "V160/V161/V162 are proxy implementations of task-aligned pretraining/fine-tuning using priors and low-DoF residuals.",
            "V163 candidates deliberately use no old-test server replacement.",
            "R67 action remains included as a public-validated anchor because local action F1 alone has not tracked public LB reliably.",
        ],
    }
    (OUTDIR / "v160_v163_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v160_v163_report.md").write_text(
        "# V160-V163 Task-Aligned Pretraining / Distillation\n\n"
        "## Safety\n\n"
        "- No old-test server label replacement.\n"
        "- No hidden test target usage.\n"
        "- Uses only public `test_new` observed prefix-internal transitions for V160 test priors.\n\n"
        "## Base Metrics\n\n"
        f"- Base action Macro-F1: `{base_action_metric:.6f}`\n"
        f"- Base point Macro-F1: `{base_point_metric:.6f}`\n"
        f"- Base server AUC: `{base_server_auc:.6f}`\n\n"
        "## Best Local Results\n\n"
        f"- V160 action: `{v160_action_search.iloc[0]['candidate']}` = `{v160_action_search.iloc[0]['action_macro_f1']:.6f}`\n"
        f"- V161 action: `{v161_search.iloc[0]['candidate']}` = `{v161_search.iloc[0]['action_macro_f1']:.6f}`\n"
        f"- V160 point: `{v160_point_search.iloc[0]['candidate']}` = `{v160_point_search.iloc[0]['point_macro_f1']:.6f}`\n"
        f"- V162 point: `{v162_search.iloc[0]['candidate']}` = `{v162_search.iloc[0]['point_macro_f1']:.6f}`\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )

    # Keep a copy under src/analysis for the organized workspace mirror.
    shutil.copy2(Path(__file__), SRC_ANALYSIS / Path(__file__).name)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
