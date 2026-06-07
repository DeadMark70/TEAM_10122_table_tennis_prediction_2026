"""R52 point feature influence audit.

This is an interpretability/diagnostic experiment, not a submission branch.
It answers which feature groups matter for pointId under a leakage-safe
match-group CV setup.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    POINT_CLASSES,
    POINT_NONTERMINAL_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    validate_raw_data,
)


OUTDIR = Path("r52_point_feature_influence")
POINT_LABELS = list(range(10))


@dataclass(frozen=True)
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_ngram_weight: float
    server_blend_weight: float
    server_parity_weight: float
    server_remaining_weight: float
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    bins_mode: str


def make_lgbm(objective: str, seed: int, num_class: int | None = None) -> lgb.LGBMClassifier:
    params: dict[str, int | float | str] = {
        "objective": objective,
        "n_estimators": 160,
        "learning_rate": 0.045,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if num_class is not None:
        params["num_class"] = num_class
    return lgb.LGBMClassifier(**params)


def aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: Iterable[int]) -> np.ndarray:
    classes = list(classes)
    proba = model.predict_proba(x)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    out = np.zeros((len(x), len(classes)), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    for src_idx, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = proba[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero_rows = row_sum[:, 0] <= 0
    if zero_rows.any():
        out[zero_rows, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def predict_point(terminal_model: lgb.LGBMClassifier, nonterminal_model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    terminal_raw = terminal_model.predict_proba(x)
    terminal_prob = terminal_raw[:, 1] if terminal_raw.ndim == 2 else terminal_raw
    terminal_prob = np.clip(terminal_prob.astype(float), 1e-6, 1.0 - 1e-6)
    nonterm = aligned_proba(nonterminal_model, x, POINT_NONTERMINAL_CLASSES)
    out = np.zeros((len(x), len(POINT_CLASSES)), dtype=float)
    out[:, 0] = terminal_prob
    out[:, 1:] = (1.0 - terminal_prob[:, None]) * nonterm
    return out / out.sum(axis=1, keepdims=True)


def point_depth(v: int) -> int:
    if v == 0:
        return 0
    return (int(v) - 1) // 3 + 1


def point_side(v: int) -> int:
    if v == 0:
        return 0
    return (int(v) - 1) % 3 + 1


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "role_score_phase": [
            c
            for c in features
            if c
            in {
                "sex",
                "numberGame",
                "rally_id",
                "prefix_len",
                "prefix_len_is_odd",
                "next_hitter_is_server",
                "next_strikeId_rule",
                "is_server_hitter",
                "serverScore",
                "receiverScore",
                "serverScoreDiff",
                "scoreTotal",
            }
        ],
        "lag0_last_stroke": [c for c in features if c.startswith("lag0_")],
        "lag1_prev_stroke": [c for c in features if c.startswith("lag1_")],
        "lag2_prev_stroke": [c for c in features if c.startswith("lag2_")],
        "lag3plus_history": [c for c in features if c.startswith(("lag3_", "lag4_", "lag5_"))],
        "action_counts": [c for c in features if c.startswith("count_actionId_") or c == "nunique_actionId"],
        "point_counts": [c for c in features if c.startswith("count_pointId_") or c == "nunique_pointId"],
        "spin_counts": [c for c in features if c.startswith("count_spinId_") or c == "nunique_spinId"],
        "hand_counts": [c for c in features if c.startswith("count_handId_") or c == "nunique_handId"],
        "position_counts": [c for c in features if c.startswith("count_positionId_") or c == "nunique_positionId"],
        "repeat_flags": [c for c in features if c.startswith("last_") and c.endswith("_same_as_prev")],
    }
    return {k: v for k, v in groups.items() if v}


def grouped_permutation_importance(prefix_df: pd.DataFrame, features: list[str], n_splits: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    groups = feature_groups(features)
    oof_prob = np.zeros((len(prefix_df), len(POINT_CLASSES)), dtype=float)
    rows: list[dict[str, float | str | int]] = []
    rng = np.random.default_rng(520)

    for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=n_splits).split(prefix_df, groups=prefix_df["match"]), start=1):
        train_df = prefix_df.iloc[train_idx].copy()
        valid_df = prefix_df.iloc[valid_idx].copy()
        x_train = train_df[features]
        x_valid = valid_df[features]

        terminal = make_lgbm("binary", seed=1000 + fold)
        terminal.fit(x_train, train_df["next_is_terminal"])

        nt_train = train_df[train_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
        nonterminal = make_lgbm("multiclass", seed=2000 + fold, num_class=len(POINT_NONTERMINAL_CLASSES))
        nonterminal.fit(
            nt_train[features],
            nt_train["next_pointId"],
            sample_weight=class_weight_sample(nt_train["next_pointId"]),
        )

        base_prob = predict_point(terminal, nonterminal, x_valid)
        oof_prob[valid_idx] = base_prob
        base_pred = np.asarray(POINT_CLASSES)[np.argmax(base_prob, axis=1)]
        base_f1 = f1_score(valid_df["next_pointId"], base_pred, average="macro", labels=POINT_LABELS, zero_division=0)

        for group_name, cols in groups.items():
            x_perm = x_valid.copy()
            for col in cols:
                x_perm[col] = rng.permutation(x_perm[col].to_numpy())
            perm_prob = predict_point(terminal, nonterminal, x_perm)
            perm_pred = np.asarray(POINT_CLASSES)[np.argmax(perm_prob, axis=1)]
            perm_f1 = f1_score(valid_df["next_pointId"], perm_pred, average="macro", labels=POINT_LABELS, zero_division=0)
            rows.append(
                {
                    "fold": fold,
                    "group": group_name,
                    "feature_count": len(cols),
                    "base_point_f1": float(base_f1),
                    "permuted_point_f1": float(perm_f1),
                    "f1_drop": float(base_f1 - perm_f1),
                }
            )

    imp = pd.DataFrame(rows)
    summary = (
        imp.groupby("group", as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
            mean_f1_drop=("f1_drop", "mean"),
            std_f1_drop=("f1_drop", "std"),
            min_f1_drop=("f1_drop", "min"),
            max_f1_drop=("f1_drop", "max"),
        )
        .sort_values("mean_f1_drop", ascending=False)
    )
    return summary, oof_prob


def full_model_importance(prefix_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = prefix_df[features]
    terminal = make_lgbm("binary", seed=777)
    terminal.fit(x, prefix_df["next_is_terminal"])

    nt_df = prefix_df[prefix_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
    nonterminal = make_lgbm("multiclass", seed=778, num_class=len(POINT_NONTERMINAL_CLASSES))
    nonterminal.fit(
        nt_df[features],
        nt_df["next_pointId"],
        sample_weight=class_weight_sample(nt_df["next_pointId"]),
    )

    rows: list[dict[str, float | str]] = []
    for model_name, model in [("terminal_point0", terminal), ("nonterminal_1to9", nonterminal)]:
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        total_gain = float(np.sum(gain)) or 1.0
        total_split = float(np.sum(split)) or 1.0
        for feat, g, s in zip(features, gain, split):
            rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "gain": float(g),
                    "gain_share": float(g) / total_gain,
                    "split": float(s),
                    "split_share": float(s) / total_split,
                    "group": next((name for name, cols in feature_groups(features).items() if feat in cols), "other"),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "gain_share"], ascending=[True, False])


def error_slices(meta: pd.DataFrame, prob: np.ndarray) -> dict[str, pd.DataFrame]:
    pred = np.asarray(POINT_CLASSES)[np.argmax(prob, axis=1)]
    df = meta.copy()
    df["pred_pointId"] = pred
    df["correct"] = df["pred_pointId"].eq(df["next_pointId"]).astype(int)
    df["true_depth"] = df["next_pointId"].map(point_depth)
    df["pred_depth"] = df["pred_pointId"].map(point_depth)
    df["true_side"] = df["next_pointId"].map(point_side)
    df["pred_side"] = df["pred_pointId"].map(point_side)
    df["lag0_actionId"] = meta.get("lag0_actionId", -1)
    df["lag0_spinId"] = meta.get("lag0_spinId", -1)
    df["lag0_pointId"] = meta.get("lag0_pointId", -1)
    df["prefix_bin"] = np.where(df["prefix_len"] <= 2, "le2", "ge3")

    def table(col: str) -> pd.DataFrame:
        out = (
            df.groupby(col, dropna=False)
            .agg(
                rows=("next_pointId", "size"),
                point_accuracy=("correct", "mean"),
                point0_true_rate=("next_pointId", lambda s: float(np.mean(np.asarray(s) == 0))),
                point0_pred_rate=("pred_pointId", lambda s: float(np.mean(np.asarray(s) == 0))),
                depth_match_rate=("true_depth", lambda s: float(np.mean(df.loc[s.index, "true_depth"].to_numpy() == df.loc[s.index, "pred_depth"].to_numpy()))),
                side_match_rate=("true_side", lambda s: float(np.mean(df.loc[s.index, "true_side"].to_numpy() == df.loc[s.index, "pred_side"].to_numpy()))),
            )
            .reset_index()
        )
        return out.sort_values("rows", ascending=False)

    return {
        "by_prefix_bin": table("prefix_bin"),
        "by_prefix_len": table("prefix_len"),
        "by_lag0_action": table("lag0_actionId"),
        "by_lag0_spin": table("lag0_spinId"),
        "by_lag0_point": table("lag0_pointId"),
    }


def maybe_shap_sample(prefix_df: pd.DataFrame, features: list[str]) -> pd.DataFrame | None:
    try:
        import shap  # type: ignore
    except Exception:
        return None
    sample = prefix_df.sample(n=min(1500, len(prefix_df)), random_state=52)
    nt_df = prefix_df[prefix_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
    model = make_lgbm("multiclass", seed=5252, num_class=len(POINT_NONTERMINAL_CLASSES))
    model.fit(
        nt_df[features],
        nt_df["next_pointId"],
        sample_weight=class_weight_sample(nt_df["next_pointId"]),
    )
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample[features])
    if isinstance(values, list):
        arr = np.mean([np.abs(v) for v in values], axis=0)
    else:
        arr = np.abs(values)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    mean_abs = arr.mean(axis=0)
    out = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_df = build_train_prefix_table(train, max_lag=6)
    _ = build_test_prefix_table(test, max_lag=6)
    features = feature_columns(prefix_df)

    summary, oof_prob = grouped_permutation_importance(prefix_df, features)
    summary.to_csv(OUTDIR / "r52_grouped_permutation_importance.csv", index=False)

    full_imp = full_model_importance(prefix_df, features)
    full_imp.to_csv(OUTDIR / "r52_lgbm_feature_importance.csv", index=False)
    full_imp.groupby(["model", "group"], as_index=False).agg(
        gain_share=("gain_share", "sum"),
        split_share=("split_share", "sum"),
        feature_count=("feature", "count"),
    ).sort_values(["model", "gain_share"], ascending=[True, False]).to_csv(
        OUTDIR / "r52_lgbm_group_importance.csv", index=False
    )

    slices = error_slices(prefix_df[["rally_uid", "match", "prefix_len", "next_pointId", "lag0_actionId", "lag0_spinId", "lag0_pointId"]], oof_prob)
    for name, frame in slices.items():
        frame.to_csv(OUTDIR / f"r52_error_slice_{name}.csv", index=False)

    shap_df = maybe_shap_sample(prefix_df, features)
    shap_available = shap_df is not None
    if shap_df is not None:
        shap_df.to_csv(OUTDIR / "r52_shap_nonterminal_sample.csv", index=False)

    pred = np.asarray(POINT_CLASSES)[np.argmax(oof_prob, axis=1)]
    oof_point_f1 = f1_score(prefix_df["next_pointId"], pred, average="macro", labels=POINT_LABELS, zero_division=0)
    report = {
        "rows": int(len(prefix_df)),
        "feature_count": int(len(features)),
        "audit_oof_point_macro_f1": float(oof_point_f1),
        "top_grouped_permutation": summary.head(10).to_dict(orient="records"),
        "top_terminal_features": full_imp[full_imp["model"].eq("terminal_point0")].head(15).to_dict(orient="records"),
        "top_nonterminal_features": full_imp[full_imp["model"].eq("nonterminal_1to9")].head(15).to_dict(orient="records"),
        "shap_available": bool(shap_available),
    }
    (OUTDIR / "r52_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
