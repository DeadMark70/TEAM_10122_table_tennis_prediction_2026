"""R54 structured point diagnostic.

Train separate terminal/depth/side heads and combine them into a point
distribution. This is diagnostic only: it checks whether point mistakes are
more recoverable as terminal/depth/side than as direct 10-class point.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_r53_incoming_interaction_features import add_r53_features
from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    validate_raw_data,
)


OUTDIR = Path("r54_structured_point_diagnostic")
DEPTH_CLASSES = [1, 2, 3]
SIDE_CLASSES = [1, 2, 3]


def point_depth(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) // 3 + 1).astype(np.int16)
    return out


def point_side(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) % 3 + 1).astype(np.int16)
    return out


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


def aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    out = np.zeros((len(x), len(classes)), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    for src_idx, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = proba[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0
    if zero.any():
        out[zero, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def combine_structured(p0: np.ndarray, depth_prob: np.ndarray, side_prob: np.ndarray) -> np.ndarray:
    out = np.zeros((len(p0), len(POINT_CLASSES)), dtype=float)
    out[:, 0] = p0
    idx = 1
    for depth_i in range(3):
        for side_i in range(3):
            out[:, idx] = (1.0 - p0) * depth_prob[:, depth_i] * side_prob[:, side_i]
            idx += 1
    return out / out.sum(axis=1, keepdims=True)


def metrics_by_slice(meta: pd.DataFrame, pred: np.ndarray, label: str) -> pd.DataFrame:
    df = meta.copy()
    df["pred"] = pred
    df["prefix_bin"] = np.where(df["prefix_len"] <= 2, "le2", "ge3")
    rows = []
    for name, part in [("all", df), *list(df.groupby("prefix_bin")), *[(f"lag0_action_{k}", v) for k, v in df.groupby("lag0_actionId")]]:
        if len(part) < 30:
            continue
        rows.append(
            {
                "model": label,
                "slice": str(name),
                "rows": int(len(part)),
                "point_macro_f1": float(
                    f1_score(part["next_pointId"], part["pred"], average="macro", labels=POINT_CLASSES, zero_division=0)
                ),
                "accuracy": float(np.mean(part["next_pointId"].to_numpy() == part["pred"].to_numpy())),
                "point0_true_rate": float(np.mean(part["next_pointId"].to_numpy() == 0)),
                "point0_pred_rate": float(np.mean(part["pred"].to_numpy() == 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_df = add_r53_features(build_train_prefix_table(train, max_lag=6))
    prefix_df["point_depth_target"] = point_depth(prefix_df["next_pointId"])
    prefix_df["point_side_target"] = point_side(prefix_df["next_pointId"])
    features = feature_columns(prefix_df)
    features = [c for c in features if c not in {"point_depth_target", "point_side_target"}]

    oof_struct = np.zeros((len(prefix_df), len(POINT_CLASSES)), dtype=float)
    oof_p0 = np.zeros(len(prefix_df), dtype=float)
    oof_depth = np.zeros((len(prefix_df), 3), dtype=float)
    oof_side = np.zeros((len(prefix_df), 3), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=5).split(prefix_df, groups=prefix_df["match"]), start=1):
        tr = prefix_df.iloc[train_idx].copy()
        va = prefix_df.iloc[valid_idx].copy()
        x_tr = tr[features]
        x_va = va[features]

        terminal = make_lgbm("binary", seed=5400 + fold)
        terminal.fit(x_tr, tr["next_pointId"].eq(0).astype(int))
        p0_raw = terminal.predict_proba(x_va)
        p0 = p0_raw[:, 1] if p0_raw.ndim == 2 else p0_raw

        nt = tr[tr["next_pointId"].gt(0)].copy()
        depth_model = make_lgbm("multiclass", seed=5500 + fold, num_class=3)
        depth_model.fit(
            nt[features],
            nt["point_depth_target"],
            sample_weight=class_weight_sample(nt["point_depth_target"]),
        )
        side_model = make_lgbm("multiclass", seed=5600 + fold, num_class=3)
        side_model.fit(
            nt[features],
            nt["point_side_target"],
            sample_weight=class_weight_sample(nt["point_side_target"]),
        )

        depth_prob = aligned_proba(depth_model, x_va, DEPTH_CLASSES)
        side_prob = aligned_proba(side_model, x_va, SIDE_CLASSES)
        oof_p0[valid_idx] = p0
        oof_depth[valid_idx] = depth_prob
        oof_side[valid_idx] = side_prob
        oof_struct[valid_idx] = combine_structured(p0, depth_prob, side_prob)
        print(f"fold {fold} done")

    pred_struct = np.asarray(POINT_CLASSES)[np.argmax(oof_struct, axis=1)]
    p0_auc = roc_auc_score(prefix_df["next_pointId"].eq(0).astype(int), oof_p0)
    nt_mask = prefix_df["next_pointId"].gt(0).to_numpy()
    depth_pred = np.asarray(DEPTH_CLASSES)[np.argmax(oof_depth, axis=1)]
    side_pred = np.asarray(SIDE_CLASSES)[np.argmax(oof_side, axis=1)]
    depth_f1 = f1_score(
        prefix_df.loc[nt_mask, "point_depth_target"],
        depth_pred[nt_mask],
        average="macro",
        labels=DEPTH_CLASSES,
        zero_division=0,
    )
    side_f1 = f1_score(
        prefix_df.loc[nt_mask, "point_side_target"],
        side_pred[nt_mask],
        average="macro",
        labels=SIDE_CLASSES,
        zero_division=0,
    )
    point_f1 = f1_score(prefix_df["next_pointId"], pred_struct, average="macro", labels=POINT_CLASSES, zero_division=0)

    meta = prefix_df[["rally_uid", "match", "prefix_len", "next_pointId", "lag0_actionId", "lag0_spinId", "lag0_pointId"]].copy()
    slices = metrics_by_slice(meta, pred_struct, "r54_structured")
    slices.to_csv(OUTDIR / "r54_structured_slice_report.csv", index=False)

    report = {
        "feature_count": int(len(features)),
        "structured_point_macro_f1": float(point_f1),
        "terminal_point0_auc": float(p0_auc),
        "nonterminal_depth_macro_f1": float(depth_f1),
        "nonterminal_side_macro_f1": float(side_f1),
        "point0_pred_rate": float(np.mean(pred_struct == 0)),
        "point0_true_rate": float(np.mean(prefix_df["next_pointId"].to_numpy() == 0)),
        "slice_top": slices.sort_values("point_macro_f1", ascending=False).head(20).to_dict(orient="records"),
        "slice_bottom": slices.sort_values("point_macro_f1", ascending=True).head(20).to_dict(orient="records"),
    }
    (OUTDIR / "r54_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(OUTDIR / "r54_oof_structured_point.npy", oof_struct)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
