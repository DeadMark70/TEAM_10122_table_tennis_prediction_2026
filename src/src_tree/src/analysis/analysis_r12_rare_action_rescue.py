"""R12 targeted rare-action rescue for actionId 8/9.

This experiment keeps point/server fixed and tests whether a narrow binary
detector can safely override the base action prediction to rare actionId 8/9.

Leakage controls:
  - Detector OOF is trained by outer match fold.
  - Fold-valid rallies are excluded from the detector training rows.
  - Point and server branches are not changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r7_phase_features import add_phase_features
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R12 rare action rescue audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r1-feature-report", default="feature_report_r1.json")
    parser.add_argument("--topk-report", default="r12_rare_action_topk_report.csv")
    parser.add_argument("--detector-report", default="r12_detector_report.csv")
    parser.add_argument("--override-search", default="r12_override_search.csv")
    parser.add_argument("--class-report", default="r12_action_class_report.csv")
    parser.add_argument("--selected", default="r12_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r12.json")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def assign_folds_from_report(meta: pd.DataFrame, fold_report: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    folds: list[int] = []
    rows = fold_report[fold_report["valid_rows"].gt(0)][["fold", "valid_rows"]]
    for _, row in rows.iterrows():
        folds.extend([int(row["fold"])] * int(row["valid_rows"]))
    if len(folds) != len(out):
        raise ValueError(f"Cannot assign folds: fold rows={len(folds)} meta rows={len(out)}")
    out["fold"] = folds
    return out


def build_action_components(v3, v5, v7, v10, r7, selected_v10, selected_r8):
    v3_action, v3_point, v3_server = compose_v3(v3)
    r7_action, _, _ = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    safe_action = blend_probs(r1_action, v10["v10_action"], float(selected_v10["action_v10_weight"]))
    safe_server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]
    r8_action = blend_probs(safe_action, r7_action, float(selected_r8["r7_weight"]))
    return {
        "v3": v3_action,
        "v5_gru": v5["gru_action"],
        "v7_transformer": v7["tr_action"],
        "v10b": v10["v10_action"],
        "r7_phase": r7_action,
        "r1": r1_action,
        "v10b_safe": safe_action,
        "r8": r8_action,
    }, v3_point, safe_server


def topk_report(meta: pd.DataFrame, components: dict[str, np.ndarray], classes=(8, 9)) -> pd.DataFrame:
    rows = []
    y = meta["next_actionId"].to_numpy(dtype=int)
    for model, prob in components.items():
        order = np.argsort(-prob, axis=1)
        pred = np.asarray(ACTION_CLASSES)[order[:, 0]]
        for cls in classes:
            idx = np.where(y == cls)[0]
            if len(idx) == 0:
                continue
            confusions = pd.Series(pred[idx]).value_counts().head(8).to_dict()
            rows.append(
                {
                    "model": model,
                    "actionId": cls,
                    "support": int(len(idx)),
                    "top1_rate": float((order[idx, :1] == cls).any(axis=1).mean()),
                    "top3_rate": float((order[idx, :3] == cls).any(axis=1).mean()),
                    "top5_rate": float((order[idx, :5] == cls).any(axis=1).mean()),
                    "mean_prob": float(prob[idx, cls].mean()),
                    "median_prob": float(np.median(prob[idx, cls])),
                    "top_confusions": json.dumps({int(k): int(v) for k, v in confusions.items()}),
                }
            )
    return pd.DataFrame(rows)


def align_validation_features(prefix_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keys = ["rally_uid", "prefix_len"]
    cols = list(dict.fromkeys(keys + list(prefix_df.columns)))
    merged = meta[keys + ["fold"]].merge(prefix_df[cols], on=keys, how="left", validate="one_to_one")
    if merged.isna().any().any():
        bad = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Validation feature alignment produced NaN in {bad[:10]}")
    return merged


def binary_weights(y: pd.Series, hard_negative: pd.Series | None = None) -> np.ndarray:
    y_arr = y.to_numpy(dtype=int)
    pos = max(int(y_arr.sum()), 1)
    neg = max(len(y_arr) - pos, 1)
    weights = np.where(y_arr == 1, 0.5 / pos, 0.5 / neg).astype(float)
    weights *= len(y_arr) / weights.sum()
    if hard_negative is not None:
        weights *= np.where((y_arr == 0) & hard_negative.to_numpy(dtype=bool), 1.5, 1.0)
        weights *= len(y_arr) / weights.sum()
    return weights


def make_detector(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_detector_oof(
    prefix_df: pd.DataFrame,
    valid_features: pd.DataFrame,
    meta: pd.DataFrame,
    features: list[str],
    target_action: int,
    base_pred_for_hard_neg: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    out = np.zeros(len(valid_features), dtype=float)
    valid_rallies_by_fold = {
        fold: set(meta.loc[meta["fold"].eq(fold), "rally_uid"].astype(int).tolist())
        for fold in sorted(meta["fold"].unique())
    }
    hard_neg_actions = {
        8: {10, 11, 13, 4, 6},
        9: {10, 11, 13, 6, 12},
    }.get(target_action, set())
    for fold in sorted(meta["fold"].unique()):
        valid_mask = meta["fold"].eq(fold).to_numpy()
        train_mask = ~prefix_df["rally_uid"].astype(int).isin(valid_rallies_by_fold[fold])
        train_part = prefix_df.loc[train_mask].copy()
        y = train_part["next_actionId"].eq(target_action).astype(int)
        hard_neg = train_part["next_actionId"].isin(hard_neg_actions)
        model = make_detector(seed + int(fold) * 17 + target_action, n_estimators)
        model.fit(train_part[features], y, sample_weight=binary_weights(y, hard_neg))
        out[valid_mask] = model.predict_proba(valid_features.loc[valid_mask, features])[:, 1]
    return out


def detector_report(meta: pd.DataFrame, det8: np.ndarray, det9: np.ndarray) -> pd.DataFrame:
    rows = []
    y = meta["next_actionId"].to_numpy(dtype=int)
    for cls, det in [(8, det8), (9, det9)]:
        yy = (y == cls).astype(int)
        try:
            auc = roc_auc_score(yy, det)
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "actionId": cls,
                "support": int(yy.sum()),
                "auc": float(auc),
                "average_precision": float(average_precision_score(yy, det)),
                "positive_mean_score": float(det[yy == 1].mean()) if yy.sum() else float("nan"),
                "negative_mean_score": float(det[yy == 0].mean()),
                "positive_p95_score": float(np.quantile(det[yy == 1], 0.95)) if yy.sum() else float("nan"),
                "negative_p99_score": float(np.quantile(det[yy == 0], 0.99)),
            }
        )
    return pd.DataFrame(rows)


def action_class_report(meta: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for model, pred in preds.items():
        for cls in ACTION_CLASSES:
            support = int((y == cls).sum())
            pred_count = int((pred == cls).sum())
            tp = int(((y == cls) & (pred == cls)).sum())
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rows.append(
                {
                    "model": model,
                    "actionId": cls,
                    "support": support,
                    "pred_count": pred_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def override_predictions(
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    det8: np.ndarray,
    det9: np.ndarray,
    t8: float,
    t9: float,
    minp8: float,
    minp9: float,
    margin: float,
) -> np.ndarray:
    pred = base_pred.copy()
    base_top_prob = base_prob[np.arange(len(base_prob)), base_pred]
    score8 = det8 * np.clip(base_prob[:, 8], 1e-6, 1.0)
    score9 = det9 * np.clip(base_prob[:, 9], 1e-6, 1.0)
    cand8 = (det8 >= t8) & (base_prob[:, 8] >= minp8) & (base_prob[:, 8] >= base_top_prob * margin)
    cand9 = (det9 >= t9) & (base_prob[:, 9] >= minp9) & (base_prob[:, 9] >= base_top_prob * margin)
    both = cand8 & cand9
    pred[cand8 & ~cand9] = 8
    pred[cand9 & ~cand8] = 9
    pred[both & (score8 >= score9)] = 8
    pred[both & (score9 > score8)] = 9
    return pred


def search_overrides(
    meta: pd.DataFrame,
    base_probs: dict[str, np.ndarray],
    base_preds: dict[str, np.ndarray],
    det8: np.ndarray,
    det9: np.ndarray,
    point_f1: float,
    server_auc: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    y = meta["next_actionId"].to_numpy(dtype=int)
    base_f1 = {name: f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0) for name, pred in base_preds.items()}
    best = None
    for base_name, prob in base_probs.items():
        base_pred = base_preds[base_name]
        for t8 in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            for t9 in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
                for minp8 in [0.005, 0.01, 0.02, 0.04]:
                    for minp9 in [0.005, 0.01, 0.02, 0.04]:
                        for margin in [0.10, 0.15, 0.20, 0.30]:
                            pred = override_predictions(base_pred, prob, det8, det9, t8, t9, minp8, minp9, margin)
                            churn = float((pred != base_pred).mean())
                            if churn > 0.06:
                                continue
                            action_f1 = f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
                            rows.append(
                                {
                                    "base": base_name,
                                    "threshold8": t8,
                                    "threshold9": t9,
                                    "min_base_prob8": minp8,
                                    "min_base_prob9": minp9,
                                    "margin": margin,
                                    "action_macro_f1": float(action_f1),
                                    "gain_vs_base": float(action_f1 - base_f1[base_name]),
                                    "point_macro_f1": point_f1,
                                    "server_auc": server_auc,
                                    "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
                                    "churn_vs_base": churn,
                                    "pred8_count": int((pred == 8).sum()),
                                    "pred9_count": int((pred == 9).sum()),
                                }
                            )
                            row = rows[-1]
                            eligible = (
                                row["gain_vs_base"] >= 0.003
                                and row["churn_vs_base"] <= 0.06
                                and row["overall"] >= 0.316
                            )
                            objective = row["action_macro_f1"] - 0.02 * row["churn_vs_base"]
                            if eligible and (best is None or objective > best["objective"]):
                                best = {"objective": objective, "row": row, "pred": pred}
    report = pd.DataFrame(rows).sort_values(["overall", "action_macro_f1"], ascending=False)
    if best is None:
        # Still select the best diagnostic row, but mark it unusable.
        if len(report):
            selected = report.iloc[0].to_dict()
        else:
            selected = {"base": "none", "action_macro_f1": float("nan"), "overall": float("nan")}
        selected["selected_policy"] = "no_override"
        selected["submit_recommendation"] = False
        return report, {"row": selected, "pred": None}
    best["row"]["selected_policy"] = "rare_override"
    best["row"]["submit_recommendation"] = True
    return report, best


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r7 = load_pickle(args.r7_oof)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    selected_r8 = json.loads(Path(args.r8_selected).read_text(encoding="utf-8"))
    r1_report = json.loads(Path(args.r1_feature_report).read_text(encoding="utf-8"))

    meta = normalize_meta(v3["valid_meta"])
    meta = assign_folds_from_report(meta, v3["fold_report"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7)]:
        assert_aligned(meta.drop(columns=["fold"]), oof["valid_meta"], name)

    components, v3_point, safe_server = build_action_components(v3, v5, v7, v10, r7, selected_v10, selected_r8)
    topk = topk_report(meta, components)
    topk.to_csv(args.topk_report, index=False)

    train_raw = pd.read_csv(args.train)
    validate_raw_data(train_raw, train_raw.iloc[0:0].copy())
    train = add_role_and_score_features(train_raw)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    prefix_df = add_phase_features(prefix_df, train)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    valid_features = align_validation_features(prefix_df, meta)

    base_for_hard_neg = np.asarray(ACTION_CLASSES)[np.argmax(components["r1"], axis=1)]
    det8 = train_detector_oof(
        prefix_df, valid_features, meta, features, 8, base_for_hard_neg, args.seed, args.n_estimators
    )
    det9 = train_detector_oof(
        prefix_df, valid_features, meta, features, 9, base_for_hard_neg, args.seed, args.n_estimators
    )
    det_report = detector_report(meta, det8, det9)
    det_report.to_csv(args.detector_report, index=False)

    point_pred = apply_segmented_multipliers(
        meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode
    )
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], safe_server)
    base_probs = {
        "r1": components["r1"],
        "v10b_safe": components["v10b_safe"],
        "r8": components["r8"],
    }
    base_preds = {
        "r1": apply_segmented_multipliers(meta, components["r1"], r1_report["action_multipliers"], ACTION_CLASSES, "two"),
        "v10b_safe": apply_segmented_multipliers(
            meta, components["v10b_safe"], selected_v10["action_multipliers"], ACTION_CLASSES, "two"
        ),
        "r8": apply_segmented_multipliers(
            meta, components["r8"], selected_r8["action_multipliers"], ACTION_CLASSES, "two"
        ),
    }
    override_report, selected = search_overrides(meta, base_probs, base_preds, det8, det9, point_f1, server_auc)
    override_report.to_csv(args.override_search, index=False)

    class_preds = dict(base_preds)
    if selected["pred"] is not None:
        class_preds["r12_selected"] = selected["pred"]
    action_class_report(meta, class_preds).to_csv(args.class_report, index=False)

    selected_out = {
        "selected": selected["row"],
        "point_policy": "fixed_v3_point",
        "server_policy": "fixed_v10b_safe_server",
        "detector_protocol": "fold_safe_binary_lgbm_for_action8_action9",
    }
    Path(args.selected).write_text(json.dumps(selected_out, indent=2), encoding="utf-8")
    feature_report = {
        "topk_report": args.topk_report,
        "detector_report": args.detector_report,
        "override_search": args.override_search,
        "class_report": args.class_report,
        "selected": selected_out,
        "features": features,
    }
    Path(args.feature_report).write_text(json.dumps(feature_report, indent=2), encoding="utf-8")

    print("R12 selected:")
    print(json.dumps(selected_out, indent=2))
    print(f"wrote {args.topk_report}, {args.detector_report}, {args.override_search}, {args.class_report}")


if __name__ == "__main__":
    main()
