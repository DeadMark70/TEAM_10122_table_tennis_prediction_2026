"""R5/R6 nested point decoder and loss-simulation diagnostics.

R5: nested OOF decoder tuning for pointId without training a new point model.
R6: post-hoc simulations of class-prior/logit-adjustment style losses using
    only OOF probabilities. This is a cheap filter before retraining any point
    head with a new loss.

The script tunes decoder parameters on inner folds and evaluates on held-out
inner folds grouped by match, so point gains are judged more strictly than a
single OOF-wide multiplier search.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers
from generate_r1_submission import compose_v3, compose_v3_full
from analysis_r1_oof_ensemble import normalize_meta


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
    parser = argparse.ArgumentParser(description="Run R5/R6 nested point decoder diagnostics.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--v10b-r1-selected", default="v10b_r1_selected.json")
    parser.add_argument("--nested-folds", type=int, default=5)
    parser.add_argument("--churn-penalty", type=float, default=0.02)
    parser.add_argument("--max-test-churn", type=float, default=0.05)
    parser.add_argument("--summary", default="r5_r6_decoder_summary.csv")
    parser.add_argument("--nested-report", default="r5_r6_nested_fold_report.csv")
    parser.add_argument("--class-report", default="r5_r6_point_class_report.csv")
    parser.add_argument("--selected", default="r5_r6_selected.json")
    parser.add_argument("--recommendation", default="r5_r6_recommendation.md")
    parser.add_argument("--submission", default="submission_r5.csv")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def point_pred(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict[str, list[float]]) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, multipliers, POINT_CLASSES, "two")


def relative_multipliers(base: dict[str, list[float]], rel_by_bin: dict[str, dict[int, float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for bin_name, values in base.items():
        arr = np.asarray(values, dtype=float).copy()
        for cls, rel in rel_by_bin.get(bin_name, {}).items():
            arr[POINT_CLASSES.index(cls)] *= float(rel)
        out[bin_name] = arr.tolist()
    return out


def class_balanced_rel(y: np.ndarray, beta: float) -> dict[int, float]:
    counts = pd.Series(y).value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    counts = np.maximum(counts, 1.0)
    weights = counts ** (-beta)
    weights = weights / np.exp(np.mean(np.log(weights)))
    return {cls: float(weights[i]) for i, cls in enumerate(POINT_CLASSES)}


def effective_number_rel(y: np.ndarray, beta: float) -> dict[int, float]:
    counts = pd.Series(y).value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    counts = np.maximum(counts, 1.0)
    effective = (1.0 - np.power(beta, counts)) / max(1e-12, 1.0 - beta)
    weights = 1.0 / np.maximum(effective, 1e-12)
    weights = weights / np.exp(np.mean(np.log(weights)))
    return {cls: float(weights[i]) for i, cls in enumerate(POINT_CLASSES)}


def logit_adjust_rel(y: np.ndarray, tau: float) -> dict[int, float]:
    counts = pd.Series(y).value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    prior = (counts + 1.0) / (counts.sum() + len(POINT_CLASSES))
    weights = np.power(prior, -tau)
    weights = weights / np.exp(np.mean(np.log(weights)))
    return {cls: float(weights[i]) for i, cls in enumerate(POINT_CLASSES)}


def candidate_builders(base_mult: dict[str, list[float]]) -> dict[str, Callable[[pd.DataFrame, np.ndarray], list[dict[str, object]]]]:
    def base_candidates(meta: pd.DataFrame, train_idx: np.ndarray) -> list[dict[str, object]]:
        del meta, train_idx
        return [{"params": {"kind": "base"}, "multipliers": base_mult}]

    def r5_targeted(meta: pd.DataFrame, train_idx: np.ndarray) -> list[dict[str, object]]:
        del meta, train_idx
        out = []
        m0_grid = [1.0, 1.05, 1.10, 1.20]
        m2_grid = [1.0, 0.9, 0.8]
        m3_grid = [1.0, 2.0, 3.0, 4.0]
        m6_grid = [1.0, 1.10, 1.20]
        m8_grid = [1.0, 1.10, 1.20]
        for m0 in m0_grid:
            for m2 in m2_grid:
                for m3 in m3_grid:
                    for m6 in m6_grid:
                        for m8 in m8_grid:
                            rel = {b: {0: m0, 2: m2, 3: m3, 6: m6, 8: m8} for b in ["le2", "ge3"]}
                            out.append({"params": {"rel": rel}, "multipliers": relative_multipliers(base_mult, rel)})
        return out

    def r5_two_bin(meta: pd.DataFrame, train_idx: np.ndarray) -> list[dict[str, object]]:
        del meta, train_idx
        out = []
        for sm0 in [1.0, 1.05, 1.10]:
            for sm2 in [1.0, 0.9, 0.8]:
                for lm0 in [1.0, 1.05, 1.10]:
                    for lm2 in [1.0, 0.9, 0.8]:
                        rel = {"le2": {0: sm0, 2: sm2}, "ge3": {0: lm0, 2: lm2}}
                        out.append({"params": {"rel": rel}, "multipliers": relative_multipliers(base_mult, rel)})
        return out

    def r6_prior(meta: pd.DataFrame, train_idx: np.ndarray) -> list[dict[str, object]]:
        y = meta.iloc[train_idx]["next_pointId"].to_numpy(dtype=int)
        out = []
        for beta in [0.1, 0.25, 0.35, 0.5, 0.75]:
            rel = class_balanced_rel(y, beta)
            rel_bins = {b: rel for b in ["le2", "ge3"]}
            out.append(
                {
                    "params": {"method": "count_power", "beta": beta, "rel": rel_bins},
                    "multipliers": relative_multipliers(base_mult, rel_bins),
                }
            )
        for beta in [0.9, 0.95, 0.99, 0.995]:
            rel = effective_number_rel(y, beta)
            rel_bins = {b: rel for b in ["le2", "ge3"]}
            out.append(
                {
                    "params": {"method": "effective_number", "beta": beta, "rel": rel_bins},
                    "multipliers": relative_multipliers(base_mult, rel_bins),
                }
            )
        for tau in [0.05, 0.10, 0.20, 0.30, 0.50]:
            rel = logit_adjust_rel(y, tau)
            rel_bins = {b: rel for b in ["le2", "ge3"]}
            out.append(
                {
                    "params": {"method": "inverse_prior", "tau": tau, "rel": rel_bins},
                    "multipliers": relative_multipliers(base_mult, rel_bins),
                }
            )
        return out

    return {
        "base_v3": base_candidates,
        "r5_targeted_bias": r5_targeted,
        "r5_m0_m2_two_bin": r5_two_bin,
        "r6_prior_simulation": r6_prior,
    }


def score_candidate(
    meta: pd.DataFrame,
    prob: np.ndarray,
    idx: np.ndarray,
    multipliers: dict[str, list[float]],
    base_pred: np.ndarray,
    churn_penalty: float,
) -> tuple[float, float, float, np.ndarray]:
    pred = point_pred(meta.iloc[idx].reset_index(drop=True), prob[idx], multipliers)
    y = meta.iloc[idx]["next_pointId"].to_numpy(dtype=int)
    f1 = f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    churn = float((pred != base_pred[idx]).mean())
    objective = float(f1 - churn_penalty * churn)
    return objective, float(f1), churn, pred


def nested_evaluate(
    meta: pd.DataFrame,
    point_prob: np.ndarray,
    base_mult: dict[str, list[float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]], str, np.ndarray]:
    base_pred_all = point_pred(meta, point_prob, base_mult)
    builders = candidate_builders(base_mult)
    splitter = GroupKFold(n_splits=args.nested_folds)
    fold_rows = []
    all_pred_by_variant: dict[str, np.ndarray] = {name: np.full(len(meta), -1, dtype=int) for name in builders}
    selected_pred = np.full(len(meta), -1, dtype=int)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(meta, groups=meta["match"]), start=1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        for name, builder in builders.items():
            best = None
            for cand in builder(meta, train_idx):
                obj, f1, churn, _ = score_candidate(
                    meta, point_prob, train_idx, cand["multipliers"], base_pred_all, args.churn_penalty
                )
                if best is None or obj > best["objective"]:
                    best = {
                        "objective": obj,
                        "train_point_f1": f1,
                        "train_churn": churn,
                        "multipliers": cand["multipliers"],
                        "params": cand["params"],
                    }
            assert best is not None
            _, valid_f1, valid_churn, pred = score_candidate(
                meta, point_prob, valid_idx, best["multipliers"], base_pred_all, args.churn_penalty
            )
            all_pred_by_variant[name][valid_idx] = pred
            fold_rows.append(
                {
                    "fold": fold,
                    "variant": name,
                    "train_point_f1": float(best["train_point_f1"]),
                    "train_churn": float(best["train_churn"]),
                    "valid_point_f1": valid_f1,
                    "valid_churn": valid_churn,
                    "params_json": json.dumps(best["params"], sort_keys=True),
                }
            )

    summary_rows = []
    y = meta["next_pointId"].to_numpy(dtype=int)
    base_f1 = f1_score(y, base_pred_all, average="macro", labels=POINT_CLASSES, zero_division=0)
    for name, pred in all_pred_by_variant.items():
        if (pred < 0).any():
            raise RuntimeError(f"Missing nested predictions for {name}.")
        f1 = f1_score(y, pred, average="macro", labels=POINT_CLASSES, zero_division=0)
        churn = float((pred != base_pred_all).mean())
        fold_scores = [r["valid_point_f1"] for r in fold_rows if r["variant"] == name]
        summary_rows.append(
            {
                "variant": name,
                "nested_point_macro_f1": float(f1),
                "gain_vs_base": float(f1 - base_f1),
                "nested_churn_vs_base": churn,
                "fold_mean": float(np.mean(fold_scores)),
                "fold_std": float(np.std(fold_scores, ddof=1)),
                "selection_score": float(f1 - args.churn_penalty * churn - 0.25 * np.std(fold_scores, ddof=1)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    selected_variant = str(summary.iloc[0]["variant"])
    selected_pred[:] = all_pred_by_variant[selected_variant]

    final_best = None
    all_idx = np.arange(len(meta))
    for cand in builders[selected_variant](meta, all_idx):
        obj, f1, churn, _ = score_candidate(meta, point_prob, all_idx, cand["multipliers"], base_pred_all, args.churn_penalty)
        if final_best is None or obj > final_best["objective"]:
            final_best = {
                "objective": obj,
                "point_f1_on_all_oof": f1,
                "churn_on_all_oof": churn,
                "multipliers": cand["multipliers"],
                "params": cand["params"],
            }
    assert final_best is not None
    return summary, pd.DataFrame(fold_rows), final_best["multipliers"], selected_variant, selected_pred


def class_report(meta: pd.DataFrame, pred: np.ndarray, variant: str) -> pd.DataFrame:
    y = meta["next_pointId"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=POINT_CLASSES, zero_division=0
    )
    rows = []
    for idx, cls in enumerate(POINT_CLASSES):
        rows.append(
            {
                "variant": variant,
                "pointId": cls,
                "support": int(support[idx]),
                "pred_count": int((pred == cls).sum()),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )
    return pd.DataFrame(rows)


def write_recommendation(
    path: Path,
    summary: pd.DataFrame,
    safe_overall: float,
    test_churn: float,
    selected_variant: str,
) -> None:
    top = summary.iloc[0]
    text = f"""# R5/R6 point decoder diagnostics

## Selected nested variant

- Variant: `{selected_variant}`
- Nested point Macro-F1: `{top['nested_point_macro_f1']:.6f}`
- Gain vs base nested V3 point: `{top['gain_vs_base']:.6f}`
- Nested churn vs base: `{top['nested_churn_vs_base']:.4%}`
- Fold std: `{top['fold_std']:.6f}`
- Safe-action/server overall with final all-OOF decoder reference: `{safe_overall:.6f}`
- Test point churn vs R1/V3 point: `{test_churn:.4%}`

## Interpretation

R5/R6 is stricter than R4 because decoder parameters are selected on inner
training folds and evaluated on held-out match groups. Treat candidates with
tiny nested gain or high churn as research-only, even if all-OOF tuning looks
better.
"""
    path.write_text(text, encoding="utf-8")


def make_submission(
    path: Path,
    test_meta: pd.DataFrame,
    action_prob: np.ndarray,
    action_mult: dict[str, list[float]],
    point_prob: np.ndarray,
    point_mult: dict[str, list[float]],
    server_prob: np.ndarray,
    expected_rows: int,
) -> pd.DataFrame:
    action_pred = apply_segmented_multipliers(test_meta, action_prob, action_mult, ACTION_CLASSES, "two")
    point_prediction = point_pred(test_meta, point_prob, point_mult)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_prediction.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != expected_rows:
        raise ValueError("Submission row count mismatch.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    sub.to_csv(path, index=False, float_format="%.8f")
    return sub


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10b = load_pickle(args.v10b_oof)
    selected_v10 = json.loads(Path(args.v10b_r1_selected).read_text(encoding="utf-8"))
    meta = normalize_meta(v3["valid_meta"])
    for name, other in [("v5", v5), ("v7", v7), ("v10b", v10b)]:
        other_meta = normalize_meta(other["valid_meta"])
        cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
        if not meta[cols].equals(other_meta[cols]):
            raise ValueError(f"{name} OOF is not aligned.")

    _, point_prob, v3_server = compose_v3(v3)
    r1_action = 0.4 * v5["gru_action"] + 0.6 * v7["tr_action"]
    r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    safe_action = blend_probs(r1_action, v10b["v10_action"], float(selected_v10["action_v10_weight"]))
    safe_server = (
        (1.0 - float(selected_v10["server_v10_weight"])) * r1_server
        + float(selected_v10["server_v10_weight"]) * v10b["v10_server"]
    )

    base_pred = point_pred(meta, point_prob, v3["tuning"].point_multipliers)
    base_point_f1 = f1_score(meta["next_pointId"], base_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    summary, nested_report, final_mult, selected_variant, nested_selected_pred = nested_evaluate(
        meta, point_prob, v3["tuning"].point_multipliers, args
    )
    summary.to_csv(args.summary, index=False)
    nested_report.to_csv(args.nested_report, index=False)

    final_pred = point_pred(meta, point_prob, final_mult)
    final_point_f1 = f1_score(meta["next_pointId"], final_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    safe_action_pred = apply_segmented_multipliers(
        meta, safe_action, selected_v10["action_multipliers"], ACTION_CLASSES, "two"
    )
    safe_action_f1 = f1_score(meta["next_actionId"], safe_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    safe_auc = roc_auc_score(meta["serverGetPoint"], safe_server)
    safe_overall = 0.4 * safe_action_f1 + 0.4 * final_point_f1 + 0.2 * safe_auc

    class_df = pd.concat(
        [
            class_report(meta, base_pred, "base_v3_point"),
            class_report(meta, nested_selected_pred, f"nested_{selected_variant}"),
            class_report(meta, final_pred, f"all_oof_{selected_variant}"),
        ],
        ignore_index=True,
    )
    class_df.to_csv(args.class_report, index=False)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    with open(args.r1_sequence_proba, "rb") as f:
        r1_full = pickle.load(f)
    with open(args.v10b_full_proba, "rb") as f:
        v10b_full = pickle.load(f)
    test_prefix, _, test_point, test_v3_server = compose_v3_full(train, test, v3["tuning"])
    test_meta = v10b_full["test_meta"].reset_index(drop=True)
    if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("Test rows are not aligned.")
    full_r1_action = 0.4 * r1_full["gru_action"] + 0.6 * r1_full["tr_action"]
    full_r1_action = full_r1_action / full_r1_action.sum(axis=1, keepdims=True)
    full_r1_server = 0.8 * test_v3_server + 0.1 * r1_full["gru_server"] + 0.1 * r1_full["tr_server"]
    full_safe_action = blend_probs(full_r1_action, v10b_full["v10_action"], float(selected_v10["action_v10_weight"]))
    full_safe_server = (
        (1.0 - float(selected_v10["server_v10_weight"])) * full_r1_server
        + float(selected_v10["server_v10_weight"]) * v10b_full["v10_server"]
    )
    sub = make_submission(
        Path(args.submission),
        test_meta,
        full_safe_action,
        selected_v10["action_multipliers"],
        test_point,
        final_mult,
        full_safe_server,
        test["rally_uid"].nunique(),
    )
    base_test_pred = point_pred(test_meta, test_point, v3["tuning"].point_multipliers)
    test_churn = float((sub["pointId"].to_numpy(dtype=int) != base_test_pred).mean())

    selected_payload = {
        "selected_variant": selected_variant,
        "base_point_macro_f1": float(base_point_f1),
        "nested_summary": summary.to_dict(orient="records"),
        "final_point_macro_f1_all_oof": float(final_point_f1),
        "safe_action_macro_f1": float(safe_action_f1),
        "safe_server_auc": float(safe_auc),
        "safe_overall_with_final_decoder": float(safe_overall),
        "test_point_churn_vs_v3": test_churn,
        "point_multipliers": final_mult,
        "churn_penalty": args.churn_penalty,
    }
    Path(args.selected).write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    write_recommendation(Path(args.recommendation), summary, safe_overall, test_churn, selected_variant)
    print(f"base nested point={base_point_f1:.6f}")
    print(f"selected nested variant={selected_variant}")
    print(f"top nested point={float(summary.iloc[0]['nested_point_macro_f1']):.6f}")
    print(f"final all-OOF point={final_point_f1:.6f}")
    print(f"safe overall={safe_overall:.6f}")
    print(f"test point churn={test_churn:.4%}")
    print(f"wrote {args.submission}, {args.summary}, {args.selected}")


if __name__ == "__main__":
    main()
