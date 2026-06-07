"""R179 action physics hierarchy experiment.

This experiment keeps R177 untouched and builds action-only candidates from
low-DoF table-tennis priors:

- phase / legality,
- incoming-ball action/point/spin state,
- a conditional-style prior when the R63 OOF signal is available.

Point and server columns are copied from the public-validated R67 anchor.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import UPLOAD_DIR


OUTDIR = Path("r179_action_physics_hierarchy")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R63_OOF_PATH = Path("r63_r64_conditional_momentum/r63_transductive_k8_oof_action.npy")
R63_TEST_SUB = Path("upload_candidates_20260519/submission_r63_transductive_k8_cls_low_action_w0p3_current_point_server.csv")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"

LOW_ACTION_CLASSES = [0, 3, 4, 7, 8, 9, 11, 12, 14]
RARE_ACTION_CLASSES = [8, 9, 12, 14]
STYLE_CLASSES = [4, 7, 8, 9, 11, 12, 14]
WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.075]


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
class ActionCandidate:
    name: str
    prob_oof: np.ndarray
    prob_test: np.ndarray
    family: str
    weight: float


def action_family(action_id: int) -> str:
    action_id = int(action_id)
    if action_id == 0:
        return "Zero"
    if 1 <= action_id <= 7:
        return "Attack"
    if 8 <= action_id <= 11:
        return "Control"
    if 12 <= action_id <= 14:
        return "Defensive"
    if 15 <= action_id <= 18:
        return "Serve"
    raise ValueError(f"unknown actionId: {action_id}")


def point_depth(point_id: int) -> int:
    point_id = int(point_id)
    if point_id == 0:
        return 0
    if 1 <= point_id <= 3:
        return 1
    if 4 <= point_id <= 6:
        return 2
    if 7 <= point_id <= 9:
        return 3
    raise ValueError(f"unknown pointId: {point_id}")


def point_side(point_id: int) -> int:
    point_id = int(point_id)
    if point_id == 0:
        return 0
    if point_id in {1, 4, 7}:
        return 1
    if point_id in {2, 5, 8}:
        return 2
    if point_id in {3, 6, 9}:
        return 3
    raise ValueError(f"unknown pointId: {point_id}")


def phase_name(phase_id: int, prefix_len: int) -> str:
    phase_id = int(phase_id)
    prefix_len = int(prefix_len)
    if phase_id == 1 or prefix_len == 1:
        return "receive"
    if phase_id == 2 or prefix_len == 2:
        return "third_ball"
    if phase_id == 3 or prefix_len == 3:
        return "fourth_ball"
    return "rally"


def normalize_rows_safe(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= eps
    if np.any(bad):
        arr[bad] = 1.0 / arr.shape[1]
        denom = arr.sum(axis=1, keepdims=True)
    return arr / denom


def apply_logit_bias(
    base_prob: np.ndarray,
    prior_prob: np.ndarray,
    weight: float,
    class_caps: dict[int, float] | None = None,
) -> np.ndarray:
    base = normalize_rows_safe(base_prob)
    prior = normalize_rows_safe(prior_prob)
    if base.shape != prior.shape:
        raise ValueError(f"shape mismatch: {base.shape} != {prior.shape}")
    logit = np.log(np.clip(base, 1e-12, 1.0)) + float(weight) * np.log(np.clip(prior, 1e-12, 1.0))
    logit -= logit.max(axis=1, keepdims=True)
    out = np.exp(logit)
    out = normalize_rows_safe(out)
    if class_caps:
        capped = np.zeros(out.shape[1], dtype=bool)
        for cls, cap in class_caps.items():
            cls_i = int(cls)
            out[:, cls_i] = np.minimum(out[:, cls_i], float(cap))
            capped[cls_i] = True
        uncapped = ~capped
        capped_sum = out[:, capped].sum(axis=1)
        uncapped_sum = out[:, uncapped].sum(axis=1)
        target_uncapped = np.clip(1.0 - capped_sum, 0.0, 1.0)
        out[:, uncapped] *= (target_uncapped / np.clip(uncapped_sum, 1e-12, None))[:, None]
    return out


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def onehot_action(labels: np.ndarray, smooth: float = 0.01) -> np.ndarray:
    out = np.full((len(labels), len(ACTION_CLASSES)), smooth / len(ACTION_CLASSES), dtype=float)
    for i, label in enumerate(labels.astype(int)):
        if int(label) in ACTION_CLASSES:
            out[i, ACTION_CLASSES.index(int(label))] += 1.0 - smooth
    return normalize_rows_safe(out)


def add_physics_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r179_phase_name"] = [phase_name(p, l) for p, l in zip(out["phase_id"], out["prefix_len"])]
    out["r179_lag0_depth"] = [point_depth(v) for v in out["lag0_pointId"]]
    out["r179_lag0_side"] = [point_side(v) for v in out["lag0_pointId"]]
    out["r179_lag0_family"] = [action_family(v) for v in out["lag0_actionId"]]
    out["r179_incoming_state"] = (
        out["phase_id"].astype(str)
        + "|a"
        + out["lag0_actionId"].astype(str)
        + "|d"
        + out["r179_lag0_depth"].astype(str)
        + "|s"
        + out["lag0_spinId"].astype(str)
        + "|t"
        + out["lag0_strengthId"].astype(str)
    )
    out["r179_style_state"] = (
        out["phase_id"].astype(str)
        + "|fam"
        + out["r179_lag0_family"].astype(str)
        + "|p"
        + out["lag0_positionId"].astype(str)
        + "|h"
        + out["lag0_handId"].astype(str)
    )
    return out


def make_action_lookup(pool: pd.DataFrame, key_cols: list[str], alpha: float, global_prior: np.ndarray) -> dict[tuple, np.ndarray]:
    lookup: dict[tuple, np.ndarray] = {}
    for key, group in pool.groupby(key_cols, dropna=False):
        counts = group["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
        dist = (counts + alpha * global_prior) / (counts.sum() + alpha)
        lookup[key if isinstance(key, tuple) else (key,)] = dist
    return lookup


def apply_lookup_prior(rows: pd.DataFrame, lookups: list[tuple[list[str], dict[tuple, np.ndarray]]], global_prior: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        dist = None
        for cols, lookup in lookups:
            key = tuple(getattr(row, c) for c in cols)
            dist = lookup.get(key)
            if dist is not None:
                break
        out[i] = global_prior if dist is None else dist
    return normalize_rows_safe(out)


def foldsafe_lookup_prior(rows: pd.DataFrame, prefix: pd.DataFrame, mode: str) -> np.ndarray:
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].copy()
        counts = pool["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
        global_prior = (counts + 1.0) / (counts.sum() + len(ACTION_CLASSES))
        if mode == "phase":
            specs = [
                (["phase_id", "r179_lag0_family"], 30.0),
                (["phase_id"], 50.0),
            ]
        elif mode == "incoming":
            specs = [
                (["r179_incoming_state"], 20.0),
                (["phase_id", "lag0_actionId", "r179_lag0_depth"], 35.0),
                (["phase_id", "lag0_actionId"], 50.0),
            ]
        elif mode == "style":
            specs = [
                (["r179_style_state"], 25.0),
                (["phase_id", "lag0_positionId", "lag0_handId"], 50.0),
            ]
        else:
            raise ValueError(mode)
        lookups = [(cols, make_action_lookup(pool, cols, alpha, global_prior)) for cols, alpha in specs]
        out[idx] = apply_lookup_prior(rows.loc[idx], lookups, global_prior)
    return out


def full_lookup_prior(prefix: pd.DataFrame, test_prefix: pd.DataFrame, mode: str) -> np.ndarray:
    counts = prefix["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (counts + 1.0) / (counts.sum() + len(ACTION_CLASSES))
    if mode == "phase":
        specs = [(["phase_id", "r179_lag0_family"], 30.0), (["phase_id"], 50.0)]
    elif mode == "incoming":
        specs = [
            (["r179_incoming_state"], 20.0),
            (["phase_id", "lag0_actionId", "r179_lag0_depth"], 35.0),
            (["phase_id", "lag0_actionId"], 50.0),
        ]
    elif mode == "style":
        specs = [(["r179_style_state"], 25.0), (["phase_id", "lag0_positionId", "lag0_handId"], 50.0)]
    else:
        raise ValueError(mode)
    lookups = [(cols, make_action_lookup(prefix, cols, alpha, global_prior)) for cols, alpha in specs]
    return apply_lookup_prior(test_prefix, lookups, global_prior)


def family_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = ["Zero", "Attack", "Control", "Defensive", "Serve"]
    yt = np.array([action_family(v) for v in y_true])
    yp = np.array([action_family(v) for v in y_pred])
    return float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0))


def selected_action_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode).astype(int)


def candidate_metrics(meta: pd.DataFrame, prob: np.ndarray, tuning, name: str, base_pred: np.ndarray, family: str, weight: float) -> dict:
    y = meta["next_actionId"].astype(int).to_numpy()
    pred = selected_action_pred(meta, prob, tuning)
    per = f1_score(y, pred, average=None, labels=ACTION_CLASSES, zero_division=0)
    return {
        "candidate": name,
        "family": family,
        "weight": float(weight),
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "family_macro_f1": family_macro_f1(y, pred),
        "low_class_mean_f1": float(np.mean([per[ACTION_CLASSES.index(c)] for c in LOW_ACTION_CLASSES])),
        "rare_class_mean_f1": float(np.mean([per[ACTION_CLASSES.index(c)] for c in RARE_ACTION_CLASSES])),
        "action_churn_vs_base": float(np.mean(pred != base_pred)),
        "changed_rows": int(np.sum(pred != base_pred)),
    }


def write_submission(test_meta: pd.DataFrame, action_prob: np.ndarray, tuning, anchor: pd.DataFrame, name: str, extra: dict) -> dict:
    pred = selected_action_pred(test_meta, action_prob, tuning)
    sub = anchor.copy()
    sub["actionId"] = pred
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
    prefix = add_physics_columns(prefix)
    test_prefix = add_physics_columns(test_prefix)
    meta = art["valid_meta"].copy().reset_index(drop=True)
    rows = add_physics_columns(align_prefix_meta(meta, prefix).reset_index(drop=True))
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    tuning = load_pickle("r111_remaining_moe_gru/oof_proba_r111.pkl")["tuning"]

    current_oof = build_current_oof_action()
    current_test = normalize_rows_safe(art["current_test_action"])
    if R63_OOF_PATH.exists():
        r63 = normalize_rows_safe(np.load(R63_OOF_PATH))
        base_oof = normalize_rows_safe(0.80 * current_oof + 0.20 * r63)
    else:
        base_oof = normalize_rows_safe(current_oof)
    base_test = current_test
    base_pred = selected_action_pred(meta, base_oof, tuning)

    prior_oof = {
        "phase": foldsafe_lookup_prior(rows, prefix, "phase"),
        "incoming": foldsafe_lookup_prior(rows, prefix, "incoming"),
        "style": foldsafe_lookup_prior(rows, prefix, "style"),
    }
    prior_test = {
        "phase": full_lookup_prior(prefix, test_prefix, "phase"),
        "incoming": full_lookup_prior(prefix, test_prefix, "incoming"),
        "style": full_lookup_prior(prefix, test_prefix, "style"),
    }
    if R63_OOF_PATH.exists():
        prior_oof["r63_style"] = normalize_rows_safe(np.load(R63_OOF_PATH))
        if R63_TEST_SUB.exists():
            r63_sub = pd.DataFrame({"rally_uid": test_meta["rally_uid"].astype(int)}).merge(
                pd.read_csv(R63_TEST_SUB)[["rally_uid", "actionId"]],
                on="rally_uid",
                how="left",
                validate="one_to_one",
            )
            if r63_sub["actionId"].isna().any():
                raise ValueError("R63 test submission does not align.")
            prior_test["r63_style"] = onehot_action(r63_sub["actionId"].to_numpy())
        else:
            prior_test["r63_style"] = base_test

    class_caps = {15: 1e-4, 16: 1e-4, 17: 1e-4, 18: 1e-4}
    rows_out: list[dict] = []
    generated: list[dict] = []
    candidates: list[ActionCandidate] = []
    for family, poof in prior_oof.items():
        ptest = prior_test[family]
        for weight in WEIGHTS:
            prob_oof = apply_logit_bias(base_oof, poof, weight, class_caps=class_caps)
            prob_test = apply_logit_bias(base_test, ptest, weight, class_caps=class_caps)
            name = f"r179_{family}_w{str(weight).replace('.', 'p')}"
            rec = candidate_metrics(meta, prob_oof, tuning, name, base_pred, family, weight)
            rows_out.append(rec)
            candidates.append(ActionCandidate(name, prob_oof, prob_test, family, weight))

    search = pd.DataFrame(rows_out).sort_values(
        ["action_macro_f1", "action_churn_vs_base", "rare_class_mean_f1"],
        ascending=[False, True, False],
    )
    search.to_csv(OUTDIR / "r179_action_hierarchy_search.csv", index=False)

    anchor = pd.read_csv(R67_ANCHOR)
    candidate_map = {c.name: c for c in candidates}
    for _, rec in search.head(6).iterrows():
        cand = candidate_map[str(rec["candidate"])]
        sub_name = f"submission_{cand.name}_r67point_current_server.csv"
        generated.append(write_submission(test_meta, cand.prob_test, tuning, anchor, sub_name, rec.to_dict()))

    best = search.iloc[0].to_dict()
    best_name = str(best["candidate"])
    best_cand = candidate_map[best_name]
    np.save(OUTDIR / "r179_best_action_oof.npy", best_cand.prob_oof)
    np.save(OUTDIR / "r179_best_action_test.npy", best_cand.prob_test)
    report = {
        "base": candidate_metrics(meta, base_oof, tuning, "r67_like_base", base_pred, "base", 0.0),
        "best": best,
        "generated": generated,
        "notes": [
            "R179 changes action only; point/server are copied from the R67 anchor submission.",
            "Serve classes 15-18 are capped by the phase-legality prior because they are nearly impossible as next labels.",
            "R63 is used as an OOF conditional-style prior when available; test-side default remains conservative.",
        ],
    }
    (OUTDIR / "r179_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r179_report.md").write_text(
        "# R179 Action Physics Hierarchy\n\n"
        "## Best Candidate\n\n"
        f"- `{best_name}`\n"
        f"- action Macro-F1: `{best['action_macro_f1']}`\n"
        f"- churn vs base: `{best['action_churn_vs_base']}`\n\n"
        "## Generated\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r179_action_physics_hierarchy.py", "src/analysis/analysis_r179_action_physics_hierarchy.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
