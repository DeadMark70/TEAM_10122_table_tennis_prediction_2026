"""R91-R95 phase/style/mask action experiments.

R91:
  Phase-gated conditional-style blend. Blend R42 action and R63 conditional
  style action with a different style weight for receive / third-ball / rally
  phases instead of one global weight.

R92:
  Style-injected sequence proxy. A fold-safe action meta LGBM that receives
  prefix/lag features, conditional-style dense features, and R42/R63 action
  probabilities. This is a lightweight diagnostic before retraining a full GRU.

R93:
  Data-driven kinematic/legality soft mask. Estimate fold-safe
  P(action | phase, incoming action/point/spin) and softly suppress classes
  that are very unlikely in the current context. This intentionally avoids
  brittle hand-written impossibility rules.

R95:
  Compact candidate sweep combining the best R91/R92/R93 action branches with
  V3 point and the validated R83 point-style probe.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r82_r86_point_style import (
    ConditionalStyleEncoder,
    add_conditional_style_features,
    align_prefix_meta,
    aligned_multiclass_proba,
    blend_point,
    clean_float,
    compose_v3_full_point,
    prepare_prefix_features,
    train_point_style_oof,
    train_point_style_test,
)
from analysis_r67_r70_meta_priors import R63_OOF_PATH, apply_action
from analysis_r87_r90_point_action_meta import style_gated_multiplier
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r91_r95_phase_mask_style")
SELECTED_DIR = Path("submissions/selected")
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


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(prob: np.ndarray) -> np.ndarray:
    order = np.sort(prob, axis=1)
    return order[:, -1] - order[:, -2]


def phase_gated_blend(base: np.ndarray, style: np.ndarray, phases: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    w = np.array([float(weights.get(int(p), weights.get(0, 0.20))) for p in phases], dtype=float)
    return normalize_rows(base * (1.0 - w[:, None]) + style * w[:, None])


def make_action_model(seed: int, n_estimators: int = 220) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=39,
        min_child_samples=26,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.25,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def build_r92_frame(
    rows: pd.DataFrame,
    cond_rows: pd.DataFrame,
    r42: np.ndarray,
    r63: np.ndarray,
    safe_feature_cols: list[str],
) -> pd.DataFrame:
    cond_cols = [c for c in cond_rows.columns if c.startswith("cond_")]
    # Keep this proxy leak-safe: only use feature_columns returned by
    # prepare_prefix_features, not numeric columns from aligned OOF rows
    # such as next labels, remaining length, or fold metadata.
    base_cols = [c for c in safe_feature_cols if c in rows.columns and pd.api.types.is_numeric_dtype(rows[c])]
    x = rows[base_cols].copy()
    for c in cond_cols:
        x[c] = cond_rows[c].to_numpy()
    for i, cls in enumerate(ACTION_CLASSES):
        x[f"r42_prob_{cls:02d}"] = r42[:, i]
        x[f"r63_prob_{cls:02d}"] = r63[:, i]
        x[f"style_delta_{cls:02d}"] = r63[:, i] - r42[:, i]
    x["r42_entropy"] = entropy(r42)
    x["r63_entropy"] = entropy(r63)
    x["r42_margin"] = margin(r42)
    x["r63_margin"] = margin(r63)
    x["r42_top"] = r42.argmax(axis=1)
    x["r63_top"] = r63.argmax(axis=1)
    x["r42_r63_agree"] = (x["r42_top"].to_numpy() == x["r63_top"].to_numpy()).astype(int)
    return x.replace([np.inf, -np.inf], 0).fillna(0)


def train_r92_oof(x: pd.DataFrame, y: pd.Series, rows: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        train_idx = rows.index[~rows["fold"].eq(fold)].to_numpy()
        model = make_action_model(9200 + int(fold), n_estimators=240)
        model.fit(x.iloc[train_idx], y.iloc[train_idx], sample_weight=class_weight_sample(y.iloc[train_idx]))
        out[idx] = aligned_multiclass_proba(model, x.iloc[idx], ACTION_CLASSES)
    return normalize_rows(out)


def train_r92_test(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    model = make_action_model(9299, n_estimators=260)
    model.fit(x, y, sample_weight=class_weight_sample(y))
    return aligned_multiclass_proba(model, x_test, ACTION_CLASSES)


def _global_action_prior(train_df: pd.DataFrame) -> np.ndarray:
    counts = train_df["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    return (counts + 1.0) / (counts.sum() + len(ACTION_CLASSES))


def _action_lookup(train_df: pd.DataFrame, cols: list[str], alpha: float, global_prior: np.ndarray) -> dict[tuple[int, ...], tuple[np.ndarray, int]]:
    lookup: dict[tuple[int, ...], tuple[np.ndarray, int]] = {}
    for key, sub in train_df.groupby(cols, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        counts = sub["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
        prior = (counts + alpha * global_prior) / (counts.sum() + alpha)
        lookup[tuple(int(x) for x in key_tuple)] = (prior, int(len(sub)))
    return lookup


def build_legality_lookup(train_df: pd.DataFrame, alpha: float = 25.0) -> dict:
    global_prior = _global_action_prior(train_df)
    return {
        "global": global_prior,
        "k4": _action_lookup(train_df, ["phase_id", "lag0_actionId", "lag0_pointId", "lag0_spinId"], alpha, global_prior),
        "k3": _action_lookup(train_df, ["phase_id", "lag0_actionId", "lag0_spinId"], alpha, global_prior),
        "k2": _action_lookup(train_df, ["phase_id", "lag0_actionId"], alpha, global_prior),
        "k1": _action_lookup(train_df, ["phase_id"], alpha, global_prior),
    }


def legality_prior_for_rows(rows: pd.DataFrame, lookup: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    level = np.zeros(len(rows), dtype=int)
    for i, row in enumerate(rows.itertuples(index=False)):
        keys = [
            ("k4", (int(row.phase_id), int(row.lag0_actionId), int(row.lag0_pointId), int(row.lag0_spinId))),
            ("k3", (int(row.phase_id), int(row.lag0_actionId), int(row.lag0_spinId))),
            ("k2", (int(row.phase_id), int(row.lag0_actionId))),
            ("k1", (int(row.phase_id),)),
        ]
        found = None
        for lev, key in keys:
            if key in lookup[lev]:
                found = (lev, lookup[lev][key])
                break
        if found is None:
            out[i] = lookup["global"]
            support[i] = 0
            level[i] = 0
        else:
            lev, (prior, n) = found
            out[i] = prior
            support[i] = n
            level[i] = int(lev[1:])
    return normalize_rows(out), support, level


def legality_prior_oof(rows: pd.DataFrame, train_prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    level = np.zeros(len(rows), dtype=int)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        tr = train_prefix[~train_prefix["match"].isin(valid_matches)].copy()
        lookup = build_legality_lookup(tr)
        p, s, l = legality_prior_for_rows(rows.loc[idx], lookup)
        out[idx] = p
        support[idx] = s
        level[idx] = l
    return normalize_rows(out), support, level


def apply_soft_legality(
    base: np.ndarray,
    prior: np.ndarray,
    support: np.ndarray,
    gamma: float,
    floor: float,
    cap: float,
    min_support: int,
) -> np.ndarray:
    global_prior = np.clip(base.mean(axis=0), 1e-5, 1.0)
    ratio = np.clip(prior / global_prior[None, :], 1e-3, 1e3)
    factor = np.power(ratio, gamma)
    factor = np.clip(factor, floor, cap)
    factor[support < min_support, :] = 1.0
    return normalize_rows(base * factor)


def write_submission(
    test_meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
    name: str,
    extra: dict | None = None,
) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path = SELECTED_DIR / name
    selected_path.write_bytes(path.read_bytes())
    info = {
        "candidate": name,
        "path": str(path),
        "upload_path": str(upload_path),
        "selected_path": str(selected_path),
    }
    if extra:
        info.update(extra)
    return info


def evaluate_action(name: str, prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, mult: dict, kind: str, extra: dict | None = None) -> dict:
    pred = apply_action(prob, meta, mult)
    row = {
        "variant": name,
        "kind": kind,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "action_churn_vs_r67_w0p2": float(np.mean(pred != base_pred)),
    }
    if extra:
        row.update(extra)
    return row


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, features = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)

    v3_oof = load_pickle("oof_proba_v3.pkl")
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]):
        raise ValueError("V3 OOF does not align.")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    test_prefix_v3, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    if not test_prefix_v3["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 test point rows do not align.")

    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    r63_oof = np.load(R63_OOF_PATH)
    r67_oof = normalize_rows(0.80 * r42_oof + 0.20 * r63_oof)

    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=7350).fit(pd.concat([train_raw, test_raw], ignore_index=True), train_raw)
    train_cond = add_conditional_style_features(prefix, encoder)
    rows_cond = align_prefix_meta(meta, train_cond)
    test_cond = add_conditional_style_features(test_prefix, encoder)
    cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
    cond_features = [c for c in features if c in train_cond.columns] + cond_cols
    r63_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=7350,
        n_jobs=-1,
        verbosity=-1,
    )
    r63_model.fit(train_cond[cond_features], train_cond["next_actionId"], sample_weight=class_weight_sample(train_cond["next_actionId"]))
    r63_test = aligned_multiclass_proba(r63_model, test_cond[cond_features], ACTION_CLASSES)
    r67_test = normalize_rows(0.80 * r42_test + 0.20 * r63_test)

    y_action = meta["next_actionId"].to_numpy(dtype=int)
    y_point = meta["next_pointId"].to_numpy(dtype=int)
    base_action_pred = apply_action(r67_oof, meta, art["selected"]["action_multipliers"])
    base_action_f1 = f1_score(y_action, base_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    base_point_pred = apply_segmented_multipliers(meta, v3_point_oof, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
    base_point_f1 = f1_score(y_point, base_point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)

    rows_out: list[dict] = [
        {
            "variant": "r67_w0p2",
            "kind": "base",
            "action_macro_f1": float(base_action_f1),
            "action_churn_vs_r67_w0p2": 0.0,
        }
    ]
    action_probs: dict[str, np.ndarray] = {"r67_w0p2": r67_oof}
    action_test_probs: dict[str, np.ndarray] = {"r67_w0p2": r67_test}

    # R91: phase-gated style blend.
    r91_candidates: list[dict] = []
    for w1 in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for w2 in [0.10, 0.15, 0.20, 0.25]:
            for w3 in [0.15, 0.25, 0.35, 0.45]:
                for w4 in [0.02, 0.05, 0.08, 0.10, 0.15]:
                    weights = {1: w1, 2: w2, 3: w3, 4: w4}
                    name = f"r91_phase_w{clean_float(w1)}_{clean_float(w2)}_{clean_float(w3)}_{clean_float(w4)}"
                    prob = phase_gated_blend(r42_oof, r63_oof, rows["phase_id"].to_numpy(dtype=int), weights)
                    row = evaluate_action(name, prob, meta, y_action, base_action_pred, art["selected"]["action_multipliers"], "r91_phase_gate", weights)
                    rows_out.append(row)
                    r91_candidates.append(row)
    best_r91 = sorted([r for r in r91_candidates if r["action_churn_vs_r67_w0p2"] <= 0.12], key=lambda r: r["action_macro_f1"], reverse=True)[:8]
    for row in best_r91:
        weights = {1: row[1], 2: row[2], 3: row[3], 4: row[4]}
        action_probs[row["variant"]] = phase_gated_blend(r42_oof, r63_oof, rows["phase_id"].to_numpy(dtype=int), weights)
        action_test_probs[row["variant"]] = phase_gated_blend(r42_test, r63_test, test_prefix["phase_id"].to_numpy(dtype=int), weights)

    # R92: style-injected sequence proxy meta model.
    x92 = build_r92_frame(rows, rows_cond, r42_oof, r63_oof, features)
    x92_test = build_r92_frame(test_prefix, test_cond, r42_test, r63_test, features)
    x92_test = x92_test.reindex(columns=x92.columns, fill_value=0)
    r92_oof = train_r92_oof(x92, meta["next_actionId"].astype(int), rows)
    r92_test = train_r92_test(x92, meta["next_actionId"].astype(int), x92_test)
    np.save(OUTDIR / "r92_style_injected_oof_action.npy", r92_oof)
    np.save(OUTDIR / "r92_style_injected_test_action.npy", r92_test)
    rows_out.append(evaluate_action("r92_direct", r92_oof, meta, y_action, base_action_pred, art["selected"]["action_multipliers"], "r92_style_injected_proxy"))
    for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
        name = f"r92_blend_w{clean_float(w)}"
        prob = normalize_rows((1.0 - w) * r67_oof + w * r92_oof)
        row = evaluate_action(name, prob, meta, y_action, base_action_pred, art["selected"]["action_multipliers"], "r92_style_injected_proxy", {"w92": w})
        rows_out.append(row)
        if row["action_macro_f1"] > base_action_f1 and row["action_churn_vs_r67_w0p2"] <= 0.12:
            action_probs[name] = prob
            action_test_probs[name] = normalize_rows((1.0 - w) * r67_test + w * r92_test)

    # R93: fold-safe legality prior and soft mask.
    legality_oof, support_oof, level_oof = legality_prior_oof(rows, prefix)
    lookup_full = build_legality_lookup(prefix)
    legality_test, support_test, level_test = legality_prior_for_rows(test_prefix, lookup_full)
    np.save(OUTDIR / "r93_legality_prior_oof.npy", legality_oof)
    np.save(OUTDIR / "r93_legality_prior_test.npy", legality_test)
    pd.DataFrame({"support": support_oof, "level": level_oof}).to_csv(OUTDIR / "r93_oof_support_levels.csv", index=False)

    # Include the best-known R88 branch as a mask source because R93 is mainly
    # designed to clean multiplier/style false positives.
    target_classes = [0, 3, 7, 8, 9, 11, 12, 14]
    r88_oof = style_gated_multiplier(r67_oof, r63_oof, alpha=0.10, beta=0.10, cap=3.0, target_classes=target_classes)
    r88_test = style_gated_multiplier(r67_test, r63_test, alpha=0.10, beta=0.10, cap=3.0, target_classes=target_classes)
    rows_out.append(evaluate_action("r88_a0p1_b0p1_cap3p0", r88_oof, meta, y_action, base_action_pred, art["selected"]["action_multipliers"], "r88_anchor"))
    action_probs["r88_a0p1_b0p1_cap3p0"] = r88_oof
    action_test_probs["r88_a0p1_b0p1_cap3p0"] = r88_test
    for source_name, source_oof, source_test in [("r67", r67_oof, r67_test), ("r88", r88_oof, r88_test)]:
        for gamma in [0.03, 0.05, 0.08, 0.10, 0.15]:
            for floor in [0.35, 0.50, 0.65]:
                for cap in [1.25, 1.5, 2.0]:
                    for min_support in [20, 50, 100]:
                        name = f"r93_{source_name}_g{clean_float(gamma)}_fl{clean_float(floor)}_cap{clean_float(cap)}_s{min_support}"
                        prob = apply_soft_legality(source_oof, legality_oof, support_oof, gamma, floor, cap, min_support)
                        row = evaluate_action(
                            name,
                            prob,
                            meta,
                            y_action,
                            base_action_pred,
                            art["selected"]["action_multipliers"],
                            "r93_soft_legality",
                            {"source": source_name, "gamma": gamma, "floor": floor, "cap": cap, "min_support": min_support},
                        )
                        rows_out.append(row)
                        if row["action_macro_f1"] > base_action_f1 and row["action_churn_vs_r67_w0p2"] <= 0.12:
                            action_probs[name] = prob
                            action_test_probs[name] = apply_soft_legality(source_test, legality_test, support_test, gamma, floor, cap, min_support)

    search = pd.DataFrame(rows_out).sort_values("action_macro_f1", ascending=False)
    search.to_csv(OUTDIR / "r91_r95_action_search.csv", index=False)

    # Point branch: keep R83 as the only point-positive public-safe line.
    r83_oof = train_point_style_oof(train_raw, prefix, rows, features)
    r83_test = train_point_style_test(train_raw, test_raw, prefix, test_prefix, features)
    point_variants = {
        "v3point": (
            apply_segmented_multipliers(test_meta, v3_point_test, art["selected"]["point_multipliers"], POINT_CLASSES, "two"),
            {"point_variant": "v3point", "oof_point_f1": float(base_point_f1), "point_churn": 0.0},
        )
    }
    for w in [0.05, 0.075, 0.10, 0.15]:
        prob = blend_point(v3_point_oof, r83_oof, w)
        pred = apply_segmented_multipliers(meta, prob, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
        f1 = f1_score(y_point, pred, average="macro", labels=POINT_CLASSES, zero_division=0)
        churn = float(np.mean(pred != base_point_pred))
        test_prob = blend_point(v3_point_test, r83_test, w)
        test_pred = apply_segmented_multipliers(test_meta, test_prob, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
        point_variants[f"r83point_w{clean_float(w)}"] = (
            test_pred,
            {"point_variant": f"r83point_w{clean_float(w)}", "oof_point_f1": float(f1), "point_churn": churn},
        )

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current submission did not align.")
    server_test = current_sub["serverGetPoint"].to_numpy(dtype=float)

    # R95: compact candidate sweep. Use top variants by OOF action with a churn
    # cap, plus one conservative R92/R93 if present.
    top_action_rows = search[(search["action_macro_f1"] > base_action_f1) & (search["action_churn_vs_r67_w0p2"] <= 0.12)].head(10)
    generated: list[dict] = []
    seen = set()
    for variant in ["r67_w0p2"] + top_action_rows["variant"].tolist():
        if variant not in action_probs or variant in seen:
            continue
        seen.add(variant)
        action_pred = apply_action(action_test_probs[variant], test_meta, art["selected"]["action_multipliers"])
        row = search[search["variant"].eq(variant)].head(1)
        action_info = {
            "action_variant": variant,
            "oof_action_f1": float(row["action_macro_f1"].iloc[0]) if len(row) else float(base_action_f1),
            "action_churn": float(row["action_churn_vs_r67_w0p2"].iloc[0]) if len(row) else 0.0,
        }
        # Generate V3-point version for every action candidate, and R83 w0.075
        # for the best five. This keeps upload folder readable.
        for point_key, (point_pred, point_info) in point_variants.items():
            if point_key != "v3point" and len(seen) > 6:
                continue
            if point_key not in {"v3point", "r83point_w0p075", "r83point_w0p15"}:
                continue
            name = f"submission_r95_{variant}_{point_key}_current_server.csv"
            generated.append(write_submission(test_meta, action_pred, point_pred, server_test, name, {**action_info, **point_info}))

    pd.DataFrame(generated).to_csv(OUTDIR / "r95_generated_candidates.csv", index=False)

    report = {
        "base": {
            "r67_w0p2_action_f1": float(base_action_f1),
            "v3_point_f1": float(base_point_f1),
        },
        "best_action": search.head(25).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "r91_r95_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(35).to_string(index=False))
    print(pd.DataFrame(generated).head(50).to_string(index=False))


if __name__ == "__main__":
    main()
