"""V263B simple balanced point residual.

Local-only point residual experiment for the questionnaire baseline.  The
test baseline is the public-positive V261 cap1 point anchor with action/server
kept fixed from that anchor.  OOF deltas use a fold-safe lag-based point proxy.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)
from analysis_v263_questionnaire_baseline_helpers import (
    OUTDIR,
    add_questionnaire_columns,
    cap_by_score,
    class_f1_table,
    load_v261_cap1_anchor,
    normalize_rows,
    numeric_features,
    point_depth,
    point_side,
    safe_predict_proba,
    write_local_submission,
)


CAPS = [0.005, 0.010, 0.015]
MAX_LAG = 6
EXPECTED_ROWS = 1845
BLOCKED_FEATURES = {
    "rally_uid",
    "match",
    "next_actionId",
    "next_pointId",
    "next_is_terminal",
    "serverGetPoint",
}


def cap_name(cap: float) -> str:
    return f"{cap:.3f}".replace(".", "p")


def add_point_context(train_df: pd.DataFrame, test_df: pd.DataFrame, anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.merge(
        anchor[["rally_uid", "actionId", "pointId"]],
        on="rally_uid",
        how="left",
        validate="one_to_one",
    )
    if test_out[["actionId", "pointId"]].isna().any().any():
        raise ValueError("V261 cap1 anchor did not align one-to-one with test rows.")

    # Train cannot use V173 OOF here, so use only already-observed lag context.
    train_out["v263_action_context"] = train_out["lag0_actionId"].astype(int)
    train_out["v263_action_family_context"] = train_out["lag0_action_family"].astype(int)
    train_out["v263_anchor_point_context"] = train_out["lag0_pointId"].astype(int)
    train_out["v263_anchor_point_depth"] = train_out["lag0_point_depth"].astype(int)
    train_out["v263_anchor_point_side"] = train_out["lag0_point_side"].astype(int)
    train_out["v263_point0_proxy"] = train_out["lag0_pointId"].astype(int).eq(0).astype(int)

    test_out = test_out.rename(columns={"actionId": "v263_action_context", "pointId": "v263_anchor_point_context"})
    test_out["v263_action_context"] = test_out["v263_action_context"].astype(int)
    test_out["v263_action_family_context"] = test_out["v263_action_context"].map(
        lambda value: 0
        if int(value) == 0
        else 1
        if 1 <= int(value) <= 7
        else 2
        if 8 <= int(value) <= 11
        else 3
        if 12 <= int(value) <= 14
        else 4
        if 15 <= int(value) <= 18
        else 0
    )
    test_out["v263_anchor_point_context"] = test_out["v263_anchor_point_context"].astype(int)
    test_out["v263_anchor_point_depth"] = test_out["v263_anchor_point_context"].map(point_depth)
    test_out["v263_anchor_point_side"] = test_out["v263_anchor_point_context"].map(point_side)
    test_out["v263_point0_proxy"] = test_out["v263_anchor_point_context"].eq(0).astype(int)
    return train_out, test_out


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = add_questionnaire_columns(build_train_prefix_table(train_raw, MAX_LAG))
    test_df = add_questionnaire_columns(build_test_prefix_table(test_raw, MAX_LAG))
    anchor = load_v261_cap1_anchor()
    if len(test_df) != EXPECTED_ROWS:
        raise ValueError(f"test prefix rows={len(test_df)}, expected {EXPECTED_ROWS}")
    train_df, test_df = add_point_context(train_df, test_df, anchor)
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), anchor


def fit_models(fold: int) -> list[object]:
    return [
        ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=4,
            class_weight="balanced",
            max_features="sqrt",
            random_state=2630 + fold,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=2640 + fold,
            n_jobs=1,
        ),
    ]


def train_point_oof_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    y = train_df["next_pointId"].astype(int).to_numpy()
    oof = np.zeros((len(train_df), len(POINT_CLASSES)), dtype=float)
    test_sum = np.zeros((len(test_df), len(POINT_CLASSES)), dtype=float)
    folds: list[dict[str, int]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(train_df, y, groups=train_df["match"].astype(int))):
        x_fit = train_df.iloc[fit_idx][features].replace([np.inf, -np.inf], 0).fillna(0)
        x_valid = train_df.iloc[valid_idx][features].replace([np.inf, -np.inf], 0).fillna(0)
        x_test = test_df[features].replace([np.inf, -np.inf], 0).fillna(0)
        valid_prob = np.zeros((len(valid_idx), len(POINT_CLASSES)), dtype=float)
        test_prob = np.zeros((len(test_df), len(POINT_CLASSES)), dtype=float)
        for model in fit_models(fold):
            model.fit(x_fit, y[fit_idx])
            valid_prob += safe_predict_proba(model, x_valid, POINT_CLASSES)
            test_prob += safe_predict_proba(model, x_test, POINT_CLASSES)
        oof[valid_idx] = normalize_rows(valid_prob / 2.0)
        test_sum += normalize_rows(test_prob / 2.0)
        folds.append({"fold": int(fold), "train_rows": int(len(fit_idx)), "valid_rows": int(len(valid_idx))})
        print(f"fold {fold}: train={len(fit_idx)} valid={len(valid_idx)}")
    return normalize_rows(oof), normalize_rows(test_sum / len(folds)), folds


def lag_proxy_point(train_df: pd.DataFrame) -> np.ndarray:
    proxy = train_df["lag0_pointId"].astype(int).clip(0, 9).to_numpy()
    return proxy


def capped_labels(base_labels: np.ndarray, prob: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    p = normalize_rows(prob)
    top = np.asarray(POINT_CLASSES, dtype=int)[p.argmax(axis=1)]
    base_pos = np.clip(base, 0, len(POINT_CLASSES) - 1)
    gain = p[np.arange(len(p)), top] - p[np.arange(len(p)), base_pos]
    scores = np.where((top != base) & np.isfinite(gain) & (gain > 0), gain, -np.inf)
    changed = cap_by_score(scores, cap)
    out = base.copy()
    out[changed] = top[changed]
    return out, changed, gain


def mean_class_delta(y_true: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray, classes: list[int]) -> float:
    table = class_f1_table(y_true, base_pred, cand_pred, classes)
    return float(table["delta_f1"].mean())


def candidate_verdict(delta: float, point0_shift: float, churn: float, cap: float, changed: np.ndarray, base: np.ndarray, cand: np.ndarray) -> str:
    if abs(point0_shift) > 0.02:
        return "REJECT_POINT0_RATE_SHIFT"
    if churn > cap + 0.002:
        return "REJECT_CHURN_TOO_HIGH"
    if changed.any() and np.all((base[changed] != 0) & (cand[changed] == 0)):
        return "REJECT_POINT0_ONLY_INFLATION"
    if delta > 0:
        return "CANDIDATE_FOR_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, anchor = build_frames()

    features = numeric_features(train_df, test_df, BLOCKED_FEATURES)
    y = train_df["next_pointId"].astype(int).to_numpy()
    print(f"train rows={len(train_df)} test rows={len(test_df)} features={len(features)}")
    model_oof, model_test, folds = train_point_oof_test(train_df, test_df, features)

    base_oof = lag_proxy_point(train_df)
    base_f1 = float(f1_score(y, base_oof, labels=POINT_CLASSES, average="macro", zero_division=0))
    raw_pred = np.asarray(POINT_CLASSES, dtype=int)[model_oof.argmax(axis=1)]
    raw_f1 = float(f1_score(y, raw_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    test_base = anchor["pointId"].astype(int).to_numpy()

    records: list[dict[str, object]] = []
    submissions: list[str] = []
    for cap in CAPS:
        oof_pred, oof_changed, _ = capped_labels(base_oof, model_oof, cap)
        test_pred, test_changed, _ = capped_labels(test_base, model_test, cap)
        score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
        delta = score - base_f1
        point0_rate = float(np.mean(test_pred == 0))
        point0_shift = point0_rate - float(np.mean(test_base == 0))
        churn = float(np.mean(test_changed))
        verdict = candidate_verdict(delta, point0_shift, churn, cap, test_changed, test_base, test_pred)
        name = f"submission_v263b_point_cap{cap_name(cap)}__v173action_r121server.csv"
        path = OUTDIR / name
        sub = anchor.copy()
        sub["pointId"] = test_pred.astype(int)
        write_local_submission(path, sub)
        submissions.append(name)
        records.append(
            {
                "candidate": f"v263b_point_cap{cap_name(cap)}",
                "point_macro_f1": score,
                "delta_vs_proxy_base": delta,
                "point_churn_vs_v261cap1": churn,
                "changed_rows": int(test_changed.sum()),
                "point0_rate_test": point0_rate,
                "rare_134_delta": mean_class_delta(y, base_oof, oof_pred, [1, 3, 4]),
                "long_789_delta": mean_class_delta(y, base_oof, oof_pred, [7, 8, 9]),
                "verdict": verdict,
                "path": str(path),
            }
        )

    search = pd.DataFrame(records)
    search.to_csv(OUTDIR / "v263b_point_search.csv", index=False)
    best = search.sort_values(["delta_vs_proxy_base", "point_churn_vs_v261cap1"], ascending=[False, True]).iloc[0].to_dict()
    summary = {
        "branch": "v263b_simple_balanced_point",
        "outdir": str(OUTDIR),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(features)),
        "proxy_base_point_macro_f1": base_f1,
        "raw_model_point_macro_f1": raw_f1,
        "raw_model_delta_vs_proxy_base": raw_f1 - base_f1,
        "best_candidate": best,
        "generated_submissions": submissions,
        "folds": folds,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
