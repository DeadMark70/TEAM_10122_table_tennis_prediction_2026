"""R13 family-aware control action reranker.

Targets actionId 8/9 through the control-like family instead of one-vs-rest
rare overrides.

Control-like classes:
  8  pimple's long push / control
  9  pimple's fast push / control
  10 long push / control
  11 drop shot / control
  13 block / defensive but often confused with control

The reranker is fold-safe:
  - A multiclass control specialist is trained on fold-train rallies only.
  - It predicts OOF probabilities for all fold-valid rows.
  - Final labels are only changed inside control-like gates.
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
from sklearn.metrics import f1_score, roc_auc_score

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


CONTROL_CLASSES = [8, 9, 10, 11, 13]
CONTROL_INDEX = {cls: i for i, cls in enumerate(CONTROL_CLASSES)}
AUDIT_ACTIONS = [8, 9, 10, 11, 13]


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
    parser = argparse.ArgumentParser(description="Run R13 control-family action reranker.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r1-feature-report", default="feature_report_r1.json")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--confusion-report", default="r13_control_confusion_report.csv")
    parser.add_argument("--feature-distribution-report", default="r13_control_feature_distribution.csv")
    parser.add_argument("--specialist-report", default="r13_control_specialist_report.csv")
    parser.add_argument("--rerank-search", default="r13_rerank_search.csv")
    parser.add_argument("--class-report", default="r13_action_class_report.csv")
    parser.add_argument("--selected", default="r13_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r13.json")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=160)
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
    v10b_safe = blend_probs(r1_action, v10["v10_action"], float(selected_v10["action_v10_weight"]))
    safe_server = (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]
    r8_action = blend_probs(v10b_safe, r7_action, float(selected_r8["r7_weight"]))
    return {
        "v3": v3_action,
        "v5_gru": v5["gru_action"],
        "v7_transformer": v7["tr_action"],
        "v10b": v10["v10_action"],
        "r7_phase": r7_action,
        "r1": r1_action,
        "v10b_safe": v10b_safe,
        "r8": r8_action,
    }, v3_point, safe_server


def align_validation_features(prefix_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keys = ["rally_uid", "prefix_len"]
    cols = list(dict.fromkeys(keys + list(prefix_df.columns)))
    merged = meta[keys + ["fold"]].merge(prefix_df[cols], on=keys, how="left", validate="one_to_one")
    if merged.isna().any().any():
        bad = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Validation feature alignment produced NaN in {bad[:10]}")
    return merged


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


def confusion_report(meta: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for model, pred in preds.items():
        for true_cls in AUDIT_ACTIONS:
            mask = y == true_cls
            support = int(mask.sum())
            counts = pd.Series(pred[mask]).value_counts().reindex(ACTION_CLASSES, fill_value=0)
            for pred_cls, count in counts[counts.gt(0)].sort_values(ascending=False).items():
                rows.append(
                    {
                        "model": model,
                        "true_actionId": true_cls,
                        "pred_actionId": int(pred_cls),
                        "count": int(count),
                        "rate": float(count / support) if support else 0.0,
                        "support": support,
                    }
                )
    return pd.DataFrame(rows)


def feature_distribution(valid_features: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    audit_features = [
        "prefix_len",
        "phase_id",
        "next_strikeId_rule",
        "lag0_actionId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_handId",
        "lag0_pointId",
        "lag0_positionId",
        "serve_actionId",
        "serve_spinId",
        "serve_pointId",
        "receive_actionId",
        "receive_pointId",
    ]
    data = valid_features.copy()
    data["next_actionId"] = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for action_id in AUDIT_ACTIONS:
        part = data[data["next_actionId"].eq(action_id)]
        support = len(part)
        for feature in audit_features:
            if feature not in part.columns:
                continue
            counts = part[feature].value_counts().sort_values(ascending=False).head(12)
            for value, count in counts.items():
                rows.append(
                    {
                        "actionId": action_id,
                        "support": support,
                        "feature": feature,
                        "value": int(value),
                        "count": int(count),
                        "rate": float(count / support) if support else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def make_control_model(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CONTROL_CLASSES),
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=12,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_control_specialist_oof(
    prefix_df: pd.DataFrame,
    valid_features: pd.DataFrame,
    meta: pd.DataFrame,
    features: list[str],
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    out = np.zeros((len(valid_features), len(CONTROL_CLASSES)), dtype=float)
    valid_rallies_by_fold = {
        fold: set(meta.loc[meta["fold"].eq(fold), "rally_uid"].astype(int).tolist())
        for fold in sorted(meta["fold"].unique())
    }
    for fold in sorted(meta["fold"].unique()):
        valid_mask = meta["fold"].eq(fold).to_numpy()
        train_mask = ~prefix_df["rally_uid"].astype(int).isin(valid_rallies_by_fold[fold])
        train_part = prefix_df.loc[train_mask & prefix_df["next_actionId"].isin(CONTROL_CLASSES)].copy()
        y = train_part["next_actionId"].map(CONTROL_INDEX).astype(int)
        model = make_control_model(seed + int(fold) * 23, n_estimators)
        model.fit(train_part[features], y, sample_weight=class_weight_sample(y, beta=0.5))
        out[valid_mask] = model.predict_proba(valid_features.loc[valid_mask, features])
    return normalize_rows(out)


def specialist_report(meta: pd.DataFrame, control_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    y = meta["next_actionId"].to_numpy(dtype=int)
    pred = np.asarray(CONTROL_CLASSES)[np.argmax(control_prob, axis=1)]
    mask = np.isin(y, CONTROL_CLASSES)
    for cls in CONTROL_CLASSES:
        cls_mask = y == cls
        rows.append(
            {
                "actionId": cls,
                "support": int(cls_mask.sum()),
                "within_control_recall": float((pred[cls_mask] == cls).mean()) if cls_mask.any() else 0.0,
                "within_control_pred_count_all_rows": int((pred == cls).sum()),
                "mean_prob_true_rows": float(control_prob[cls_mask, CONTROL_INDEX[cls]].mean()) if cls_mask.any() else 0.0,
            }
        )
    rows.append(
        {
            "actionId": -1,
            "support": int(mask.sum()),
            "within_control_macro_f1_on_true_control_rows": float(
                f1_score(y[mask], pred[mask], labels=CONTROL_CLASSES, average="macro", zero_division=0)
            ),
            "within_control_pred_count_all_rows": int(len(pred)),
            "mean_prob_true_rows": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def adjusted_log_scores(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict, mode: str) -> np.ndarray:
    score = np.log(np.clip(prob, 1e-12, 1.0))
    prefix = meta["prefix_len"]
    if mode == "two":
        bins = [("le2", prefix.le(2).to_numpy()), ("ge3", prefix.ge(3).to_numpy())]
    elif mode == "global":
        bins = [("global", np.ones(len(meta), dtype=bool))]
    else:
        raise ValueError(f"Unsupported multiplier mode for R13: {mode}")
    for label, mask in bins:
        mult = np.asarray(multipliers[label], dtype=float)
        score[mask] += np.log(np.clip(mult, 1e-6, None))[None, :]
    return score


def topk_intersects(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-scores, axis=1)[:, :k]
    return np.isin(order, CONTROL_CLASSES).any(axis=1)


def rerank_predictions(
    meta: pd.DataFrame,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    base_scores: np.ndarray,
    control_prob: np.ndarray,
    gate: str,
    weight: float,
    rare_bias: float,
    margin: float,
) -> np.ndarray:
    pred = base_pred.copy()
    if gate == "base_control":
        gate_mask = np.isin(base_pred, CONTROL_CLASSES)
    elif gate == "top2_control":
        gate_mask = topk_intersects(base_scores, 2)
    elif gate == "top3_control":
        gate_mask = topk_intersects(base_scores, 3)
    else:
        raise ValueError(gate)
    idx = np.where(gate_mask)[0]
    if len(idx) == 0:
        return pred
    base_control_score = base_scores[idx][:, CONTROL_CLASSES]
    spec_score = np.log(np.clip(control_prob[idx], 1e-12, 1.0))
    combined = (1.0 - weight) * base_control_score + weight * spec_score
    combined[:, CONTROL_INDEX[8]] += rare_bias
    combined[:, CONTROL_INDEX[9]] += rare_bias
    new_cls = np.asarray(CONTROL_CLASSES)[np.argmax(combined, axis=1)]

    old_scores = base_scores[idx, base_pred[idx]]
    new_scores = base_scores[idx, new_cls]
    allow = new_scores >= old_scores + margin
    # If already inside control family, allow replacement even with a small
    # negative margin; the specialist is only reordering similar actions.
    allow |= np.isin(base_pred[idx], CONTROL_CLASSES) & (new_scores >= old_scores - abs(margin))
    pred[idx[allow]] = new_cls[allow]
    return pred


def search_rerank(
    meta: pd.DataFrame,
    base_probs: dict[str, np.ndarray],
    base_preds: dict[str, np.ndarray],
    base_scores: dict[str, np.ndarray],
    control_prob: np.ndarray,
    point_f1: float,
    server_auc: float,
) -> tuple[pd.DataFrame, dict]:
    y = meta["next_actionId"].to_numpy(dtype=int)
    base_f1 = {
        name: f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        for name, pred in base_preds.items()
    }
    rows = []
    best = None
    for name in base_probs:
        for gate in ["base_control", "top2_control", "top3_control"]:
            for weight in [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]:
                for rare_bias in [0.0, 0.1, 0.2, 0.35, 0.5]:
                    for margin in [-0.10, -0.05, 0.0, 0.05, 0.10]:
                        pred = rerank_predictions(
                            meta,
                            base_probs[name],
                            base_preds[name],
                            base_scores[name],
                            control_prob,
                            gate,
                            weight,
                            rare_bias,
                            margin,
                        )
                        churn = float((pred != base_preds[name]).mean())
                        if churn > 0.05:
                            continue
                        action_f1 = f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
                        class_df = action_class_report(meta, {"tmp": pred})
                        f1_by_class = class_df.set_index("actionId")["f1"].to_dict()
                        min_drop_core = min(
                            f1_by_class.get(cls, 0.0)
                            - action_class_report(meta, {name: base_preds[name]}).set_index("actionId")["f1"].to_dict().get(cls, 0.0)
                            for cls in [10, 11, 13]
                        )
                        row = {
                            "base": name,
                            "gate": gate,
                            "weight": weight,
                            "rare_bias": rare_bias,
                            "margin": margin,
                            "action_macro_f1": float(action_f1),
                            "gain_vs_base": float(action_f1 - base_f1[name]),
                            "point_macro_f1": point_f1,
                            "server_auc": server_auc,
                            "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
                            "churn_vs_base": churn,
                            "pred8_count": int((pred == 8).sum()),
                            "pred9_count": int((pred == 9).sum()),
                            "min_f1_delta_10_11_13": float(min_drop_core),
                            "action8_f1": float(f1_by_class.get(8, 0.0)),
                            "action9_f1": float(f1_by_class.get(9, 0.0)),
                        }
                        rows.append(row)
                        eligible = (
                            row["gain_vs_base"] >= 0.003
                            and row["churn_vs_base"] <= 0.05
                            and row["min_f1_delta_10_11_13"] >= -0.015
                            and row["overall"] >= 0.316
                        )
                        objective = row["action_macro_f1"] - 0.02 * churn
                        if eligible and (best is None or objective > best["objective"]):
                            best = {"objective": objective, "row": row, "pred": pred}
    report = pd.DataFrame(rows).sort_values(["overall", "action_macro_f1"], ascending=False)
    if best is None:
        selected = report.iloc[0].to_dict() if len(report) else {"base": "none"}
        selected["selected_policy"] = "no_rerank"
        selected["submit_recommendation"] = False
        return report, {"row": selected, "pred": None}
    best["row"]["selected_policy"] = "control_rerank"
    best["row"]["submit_recommendation"] = True
    return report, best


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r7 = load_pickle(args.r7_oof)
    r1_report = json.loads(Path(args.r1_feature_report).read_text(encoding="utf-8"))
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    selected_r8 = json.loads(Path(args.r8_selected).read_text(encoding="utf-8"))

    meta = normalize_meta(v3["valid_meta"])
    meta = assign_folds_from_report(meta, v3["fold_report"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7)]:
        assert_aligned(meta.drop(columns=["fold"]), oof["valid_meta"], name)

    components, v3_point, safe_server = build_action_components(v3, v5, v7, v10, r7, selected_v10, selected_r8)
    base_probs = {"r1": components["r1"], "v10b_safe": components["v10b_safe"], "r8": components["r8"]}
    base_mults = {
        "r1": (r1_report["action_multipliers"], "two"),
        "v10b_safe": (selected_v10["action_multipliers"], "two"),
        "r8": (selected_r8["action_multipliers"], "two"),
    }
    base_scores = {
        name: adjusted_log_scores(meta, prob, base_mults[name][0], base_mults[name][1])
        for name, prob in base_probs.items()
    }
    base_preds = {name: np.asarray(ACTION_CLASSES)[np.argmax(score, axis=1)] for name, score in base_scores.items()}

    confusion_report(meta, base_preds).to_csv(args.confusion_report, index=False)

    train_raw = pd.read_csv(args.train)
    validate_raw_data(train_raw, train_raw.iloc[0:0].copy())
    train = add_role_and_score_features(train_raw)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    prefix_df = add_phase_features(prefix_df, train)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    valid_features = align_validation_features(prefix_df, meta)
    feature_distribution(valid_features, meta).to_csv(args.feature_distribution_report, index=False)

    control_prob = train_control_specialist_oof(
        prefix_df, valid_features, meta, features, args.seed, args.n_estimators
    )
    specialist_report(meta, control_prob).to_csv(args.specialist_report, index=False)

    point_pred = apply_segmented_multipliers(
        meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode
    )
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], safe_server)
    rerank_report, selected = search_rerank(
        meta, base_probs, base_preds, base_scores, control_prob, point_f1, server_auc
    )
    rerank_report.to_csv(args.rerank_search, index=False)

    class_preds = dict(base_preds)
    if selected["pred"] is not None:
        class_preds["r13_selected"] = selected["pred"]
    action_class_report(meta, class_preds).to_csv(args.class_report, index=False)

    selected_out = {
        "selected": selected["row"],
        "point_policy": "fixed_v3_point",
        "server_policy": "fixed_v10b_safe_server",
        "control_classes": CONTROL_CLASSES,
        "specialist_protocol": "fold_safe_multiclass_control_family_lgbm",
    }
    Path(args.selected).write_text(json.dumps(selected_out, indent=2), encoding="utf-8")
    metadata = {
        "confusion_report": args.confusion_report,
        "feature_distribution_report": args.feature_distribution_report,
        "specialist_report": args.specialist_report,
        "rerank_search": args.rerank_search,
        "class_report": args.class_report,
        "selected": selected_out,
        "features": features,
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("R13 selected:")
    print(json.dumps(selected_out, indent=2))
    print(f"wrote {args.confusion_report}, {args.rerank_search}, {args.class_report}")


if __name__ == "__main__":
    main()
