"""R50 point/action/spin dependency analysis.

Diagnoses whether pointId can be improved through action/spin-conditioned
specialists rather than another flat point model.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r12_rare_action_rescue import assign_folds_from_report
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, build_train_prefix_table, validate_raw_data
from baseline_v3 import apply_segmented_multipliers


OUT_DIR = Path("r50_point_action_spin_dependency")


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


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def action_family(a: int) -> str:
    if a == 0:
        return "zero"
    if 1 <= a <= 7:
        return "attack"
    if a in {8, 9, 10, 11}:
        return "control"
    if a in {12, 13, 14}:
        return "defense"
    return "serve"


def point_depth(p: int) -> str:
    if p == 0:
        return "terminal"
    if p in {1, 2, 3}:
        return "short"
    if p in {4, 5, 6}:
        return "half"
    return "long"


def point_side(p: int) -> str:
    if p == 0:
        return "terminal"
    return {1: "fh", 2: "mid", 3: "bh", 4: "fh", 5: "mid", 6: "bh", 7: "fh", 8: "mid", 9: "bh"}[p]


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(df: pd.DataFrame, key_cols: list[str], target: str = "next_pointId") -> dict:
    base_counts = df[target].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    base_h = entropy_from_counts(base_counts)
    h = 0.0
    rows = []
    for key, g in df.groupby(key_cols, dropna=False):
        counts = g[target].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
        hg = entropy_from_counts(counts)
        h += len(g) / len(df) * hg
        rows.append((key, len(g), hg))
    return {
        "key": "+".join(key_cols),
        "groups": len(rows),
        "base_entropy": base_h,
        "conditional_entropy": float(h),
        "entropy_reduction": float(base_h - h),
    }


def class_report_by_group(df: pd.DataFrame, pred_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col, dropna=False):
        rows.append(
            {
                "group": group,
                "rows": len(g),
                "point_macro_f1": float(
                    f1_score(g["next_pointId"], g[pred_col], average="macro", labels=POINT_CLASSES, zero_division=0)
                ),
                "point0_true_rate": float(g["next_pointId"].eq(0).mean()),
                "point0_pred_rate": float(g[pred_col].eq(0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("point_macro_f1")


def smoothed_prior_predict(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    key_cols: list[str],
    alpha: float = 20.0,
) -> np.ndarray:
    global_counts = train_df["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(POINT_CLASSES))
    table: dict[tuple, np.ndarray] = {}
    for key, g in train_df.groupby(key_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        counts = g["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
        table[key] = (counts + alpha * global_prior) / (counts.sum() + alpha)
    out = np.zeros((len(valid_df), len(POINT_CLASSES)), dtype=float)
    for i, row in enumerate(valid_df[key_cols].itertuples(index=False, name=None)):
        out[i] = table.get(tuple(row), global_prior)
    return out.argmax(axis=1)


def fold_safe_prior_scores(df: pd.DataFrame, key_sets: list[list[str]]) -> pd.DataFrame:
    rows = []
    for key_cols in key_sets:
        pred = np.zeros(len(df), dtype=int)
        for fold in sorted(df["fold"].unique()):
            tr = df[df["fold"].ne(fold)]
            va = df[df["fold"].eq(fold)]
            pred[va.index.to_numpy()] = smoothed_prior_predict(tr, va, key_cols)
        rows.append(
            {
                "key": "+".join(key_cols),
                "point_macro_f1": float(
                    f1_score(df["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
                ),
                "pred0_rate": float(np.mean(pred == 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("point_macro_f1", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    v3 = load_pickle("oof_proba_v3.pkl")
    v5 = load_pickle("oof_proba_v5.pkl")
    v7 = load_pickle("oof_proba_v7.pkl")
    meta = assign_folds_from_report(normalize_meta(v3["valid_meta"]), v3["fold_report"]).reset_index(drop=True)
    _, v3_point, _ = compose_v3(v3)
    point_pred = apply_segmented_multipliers(
        meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode
    )
    r1_action_prob = 0.4 * v5["gru_action"] + 0.6 * v7["tr_action"]
    r1_action_prob = r1_action_prob / r1_action_prob.sum(axis=1, keepdims=True)
    r1_action_pred = r1_action_prob.argmax(axis=1)

    df = meta.copy()
    df["point_pred"] = point_pred
    df["r1_action_pred"] = r1_action_pred
    df["true_action_family"] = df["next_actionId"].map(action_family)
    df["pred_action_family"] = df["r1_action_pred"].map(action_family)
    df["next_point_depth"] = df["next_pointId"].map(point_depth)
    df["next_point_side"] = df["next_pointId"].map(point_side)
    if "lag0_actionId" not in df.columns:
        train_raw = pd.read_csv("train.csv")
        validate_raw_data(train_raw, train_raw.iloc[0:0].copy())
        train = add_role_and_score_features(train_raw)
        prefix_df = build_train_prefix_table(train, 6)
        lag_cols = ["rally_uid", "prefix_len", "lag0_actionId", "lag0_spinId", "lag0_pointId"]
        df = df.merge(prefix_df[lag_cols], on=["rally_uid", "prefix_len"], how="left", validate="one_to_one")
        if df[["lag0_actionId", "lag0_spinId", "lag0_pointId"]].isna().any().any():
            raise ValueError("Failed to align lag0 features for R50.")
    df["last_action"] = df["lag0_actionId"].astype(int)
    df["last_spin"] = df["lag0_spinId"].astype(int)
    df["last_point"] = df["lag0_pointId"].astype(int)
    df["last_point_depth"] = df["last_point"].map(point_depth)
    df["last_point_side"] = df["last_point"].map(point_side)
    df["prefix_bin"] = np.where(df["prefix_len"].eq(1), "1", np.where(df["prefix_len"].eq(2), "2", "3+"))

    group_reports = {
        "by_true_action": class_report_by_group(df, "point_pred", "next_actionId"),
        "by_true_action_family": class_report_by_group(df, "point_pred", "true_action_family"),
        "by_pred_action": class_report_by_group(df, "point_pred", "r1_action_pred"),
        "by_prefix_bin": class_report_by_group(df, "point_pred", "prefix_bin"),
    }
    for name, rep in group_reports.items():
        rep.to_csv(OUT_DIR / f"r50_{name}.csv", index=False)

    entropy_rows = []
    for keys in [
        ["prefix_bin"],
        ["next_actionId"],
        ["true_action_family"],
        ["last_action"],
        ["last_spin"],
        ["last_point_depth"],
        ["last_action", "last_spin"],
        ["next_actionId", "last_spin"],
        ["next_actionId", "prefix_bin"],
        ["next_actionId", "last_point_depth"],
        ["next_actionId", "last_action", "last_spin"],
    ]:
        entropy_rows.append(conditional_entropy(df, keys))
    entropy_df = pd.DataFrame(entropy_rows).sort_values("entropy_reduction", ascending=False)
    entropy_df.to_csv(OUT_DIR / "r50_point_conditional_entropy.csv", index=False)

    prior_scores = fold_safe_prior_scores(
        df,
        [
            ["prefix_bin"],
            ["next_actionId"],
            ["true_action_family"],
            ["last_action"],
            ["last_spin"],
            ["last_point_depth"],
            ["last_action", "last_spin"],
            ["next_actionId", "last_spin"],
            ["next_actionId", "prefix_bin"],
            ["next_actionId", "last_point_depth"],
            ["next_actionId", "last_action", "last_spin"],
        ],
    )
    prior_scores.to_csv(OUT_DIR / "r50_foldsafe_point_prior_scores.csv", index=False)

    # Which true action classes have poor point F1 and enough support?
    action_point = group_reports["by_true_action"].copy()
    action_point["action_name_hint"] = action_point["group"].map(lambda x: action_family(int(x)) if str(x).isdigit() else "")
    action_point.to_csv(OUT_DIR / "r50_point_f1_by_true_action.csv", index=False)

    report = {
        "base_v3_point_macro_f1": float(
            f1_score(df["next_pointId"], df["point_pred"], average="macro", labels=POINT_CLASSES, zero_division=0)
        ),
        "top_entropy_reductions": entropy_df.head(8).to_dict(orient="records"),
        "prior_scores": prior_scores.head(8).to_dict(orient="records"),
        "weak_point_by_true_action": action_point[action_point["rows"].ge(100)].head(10).to_dict(orient="records"),
        "outputs": [str(p) for p in OUT_DIR.glob("*.csv")],
    }
    (OUT_DIR / "r50_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
