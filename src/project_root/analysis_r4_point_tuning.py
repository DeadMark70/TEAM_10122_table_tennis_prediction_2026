"""R4 conservative pointId decision tuning.

This script does not train a new point model. It uses the current strongest
V3/R1 point probabilities and searches small relative multipliers for classes
identified by R3:

- increase pointId=0
- decrease pointId=2
- cautiously probe pointId=3
- mildly increase pointId=6/8

The output includes OOF diagnostics and a full-test submission using the
selected point multipliers with the V10B-safe action/server branch.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers
from generate_r1_submission import compose_v3, compose_v3_full, normalize_meta


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


TARGETED_CLASSES = [0, 2, 3, 6, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R4 conservative point tuning.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--v10b-r1-selected", default="v10b_r1_selected.json")
    parser.add_argument("--submission", default="submission_r4.csv")
    parser.add_argument("--submission-r1", default="submission_r4_r1.csv")
    parser.add_argument("--summary", default="r4_point_tuning_summary.csv")
    parser.add_argument("--class-report", default="r4_point_class_report.csv")
    parser.add_argument("--prefix-report", default="r4_prefix_report.csv")
    parser.add_argument("--selected", default="r4_selected.json")
    parser.add_argument("--recommendation", default="r4_recommendation.md")
    parser.add_argument("--max-diff", type=float, default=0.08)
    parser.add_argument("--max-point3-pred", type=int, default=80)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def aligned_meta(v3: dict, *others: dict) -> pd.DataFrame:
    meta = normalize_meta(v3["valid_meta"])
    check_cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    for idx, other in enumerate(others, start=1):
        other_meta = normalize_meta(other["valid_meta"])
        if not meta[check_cols].equals(other_meta[check_cols]):
            raise ValueError(f"OOF meta alignment failed for component {idx}.")
    return meta


def r1_oof_probs(v3: dict, v5: dict, v7: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, point_prob, v3_server = compose_v3(v3)
    action_prob = 0.4 * v5["gru_action"] + 0.6 * v7["tr_action"]
    action_prob = action_prob / action_prob.sum(axis=1, keepdims=True)
    server_prob = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    return action_prob, point_prob, np.clip(server_prob, 1e-6, 1.0 - 1e-6)


def safe_oof_probs(v3: dict, v5: dict, v7: dict, v10b: dict, selected: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1_action, point_prob, r1_server = r1_oof_probs(v3, v5, v7)
    action_prob = blend_probs(r1_action, v10b["v10_action"], float(selected["action_v10_weight"]))
    server_prob = (
        (1.0 - float(selected["server_v10_weight"])) * r1_server
        + float(selected["server_v10_weight"]) * v10b["v10_server"]
    )
    return action_prob, point_prob, np.clip(server_prob, 1e-6, 1.0 - 1e-6)


def relative_multipliers(base: dict[str, list[float]], rel_by_bin: dict[str, dict[int, float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for bin_name, values in base.items():
        arr = np.asarray(values, dtype=float).copy()
        rel = rel_by_bin.get(bin_name, {})
        for cls, multiplier in rel.items():
            arr[POINT_CLASSES.index(cls)] *= float(multiplier)
        out[bin_name] = arr.tolist()
    return out


def score_point(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict[str, list[float]]) -> tuple[float, np.ndarray]:
    pred = apply_segmented_multipliers(meta, prob, multipliers, POINT_CLASSES, "two")
    score = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    return float(score), pred


def class_report(meta: pd.DataFrame, pred: np.ndarray, variant: str) -> pd.DataFrame:
    y = meta["next_pointId"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=POINT_CLASSES, zero_division=0
    )
    rows = []
    for i, cls in enumerate(POINT_CLASSES):
        rows.append(
            {
                "variant": variant,
                "pointId": cls,
                "support": int(support[i]),
                "pred_count": int((pred == cls).sum()),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        )
    return pd.DataFrame(rows)


def prefix_report(
    meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
) -> pd.DataFrame:
    bins = [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
        ("le2", meta["prefix_len"].le(2).to_numpy()),
        ("ge3", meta["prefix_len"].ge(3).to_numpy()),
    ]
    rows = []
    for name, mask in bins:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        action_f1 = f1_score(
            meta.iloc[idx]["next_actionId"], action_pred[idx], average="macro", labels=ACTION_CLASSES, zero_division=0
        )
        point_f1 = f1_score(
            meta.iloc[idx]["next_pointId"], point_pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0
        )
        try:
            server_auc = roc_auc_score(meta.iloc[idx]["serverGetPoint"], server_prob[idx])
        except ValueError:
            server_auc = np.nan
        rows.append(
            {
                "prefix_len_bin": name,
                "count": int(len(idx)),
                "action_macro_f1": float(action_f1),
                "point_macro_f1": float(point_f1),
                "server_auc": float(server_auc),
                "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
            }
        )
    return pd.DataFrame(rows)


def prediction_distribution(meta: pd.DataFrame, pred: np.ndarray) -> dict[str, int]:
    counts = pd.Series(pred).value_counts().reindex(POINT_CLASSES, fill_value=0)
    return {str(cls): int(counts.loc[cls]) for cls in POINT_CLASSES}


def evaluate_candidate(
    name: str,
    meta: pd.DataFrame,
    point_prob: np.ndarray,
    base_pred: np.ndarray,
    multipliers: dict[str, list[float]],
    action_f1: float,
    server_auc: float,
    rel: dict[str, dict[int, float]],
) -> tuple[dict[str, object], np.ndarray]:
    point_f1, pred = score_point(meta, point_prob, multipliers)
    pred_count = prediction_distribution(meta, pred)
    row = {
        "variant": name,
        "point_macro_f1": point_f1,
        "action_macro_f1": action_f1,
        "server_auc": server_auc,
        "overall": 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc,
        "diff_vs_v3_point": float((pred != base_pred).mean()),
        "point3_pred_count": int(pred_count["3"]),
        "pred_counts_json": json.dumps(pred_count, sort_keys=True),
        "relative_multipliers_json": json.dumps(
            {bin_name: {str(k): v for k, v in cls_map.items()} for bin_name, cls_map in rel.items()},
            sort_keys=True,
        ),
    }
    return row, pred


def search_r4(
    meta: pd.DataFrame,
    point_prob: np.ndarray,
    base_mult: dict[str, list[float]],
    base_pred: np.ndarray,
    action_f1: float,
    server_auc: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, list[float]], np.ndarray]:
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_mult: dict[str, list[float]] | None = None
    best_pred: np.ndarray | None = None

    def consider(name: str, rel: dict[str, dict[int, float]]) -> None:
        nonlocal best_row, best_mult, best_pred
        mult = relative_multipliers(base_mult, rel)
        row, pred = evaluate_candidate(name, meta, point_prob, base_pred, mult, action_f1, server_auc, rel)
        row["eligible"] = bool(
            row["diff_vs_v3_point"] <= args.max_diff and row["point3_pred_count"] <= args.max_point3_pred
        )
        rows.append(row)
        if row["eligible"]:
            if best_row is None or float(row["point_macro_f1"]) > float(best_row["point_macro_f1"]):
                best_row = row
                best_mult = mult
                best_pred = pred

    consider("base_v3_point", {"le2": {}, "ge3": {}})

    m0_grid = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30]
    m2_grid = [1.0, 0.9, 0.8, 0.7, 0.6]
    m3_grid = [1.0, 1.25, 1.5, 2.0, 3.0]
    m6_grid = [1.0, 1.10, 1.20, 1.35, 1.50]
    m8_grid = [1.0, 1.10, 1.20, 1.35, 1.50]

    for m0 in m0_grid:
        for m2 in m2_grid:
            for m3 in m3_grid:
                for m6 in m6_grid:
                    for m8 in m8_grid:
                        rel = {bin_name: {0: m0, 2: m2, 3: m3, 6: m6, 8: m8} for bin_name in ["le2", "ge3"]}
                        consider("r4a_global", rel)

    short_m0_grid = [1.0, 1.05, 1.10]
    short_m2_grid = [1.0, 0.9, 0.8]
    long_m0_grid = [1.0, 1.05, 1.10]
    long_m2_grid = [1.0, 0.9, 0.8]
    r4a_rows = [r for r in rows if r["variant"] == "r4a_global" and r["eligible"]]
    r4a_rows = sorted(r4a_rows, key=lambda r: float(r["point_macro_f1"]), reverse=True)[:25]
    for base in r4a_rows:
        base_rel_raw = json.loads(str(base["relative_multipliers_json"]))
        base_rel = {
            bin_name: {int(k): float(v) for k, v in cls_map.items()} for bin_name, cls_map in base_rel_raw.items()
        }
        for sm0 in short_m0_grid:
            for sm2 in short_m2_grid:
                for lm0 in long_m0_grid:
                    for lm2 in long_m2_grid:
                        rel = {
                            "le2": dict(base_rel["le2"]),
                            "ge3": dict(base_rel["ge3"]),
                        }
                        rel["le2"][0] *= sm0
                        rel["le2"][2] *= sm2
                        rel["ge3"][0] *= lm0
                        rel["ge3"][2] *= lm2
                        consider("r4b_two_bin", rel)

    if best_row is None or best_mult is None or best_pred is None:
        raise RuntimeError("No eligible R4 candidate found.")
    summary = pd.DataFrame(rows).sort_values(["eligible", "point_macro_f1"], ascending=[False, False])
    selected = dict(best_row)
    selected["variant"] = "selected_" + str(selected["variant"])
    summary = pd.concat([pd.DataFrame([selected]), summary], ignore_index=True)
    return summary, best_mult, best_pred


def write_recommendation(
    path: Path,
    base_point: float,
    selected_row: pd.Series,
    safe_overall: float,
    r1_overall: float,
) -> None:
    text = f"""# R4 conservative point tuning

## Selected

- Variant: `{selected_row['variant']}`
- Point Macro-F1: `{selected_row['point_macro_f1']:.6f}` vs V3/R1 point `{base_point:.6f}`
- Safe overall reference: `{safe_overall:.6f}`
- R1 overall reference: `{r1_overall:.6f}`
- Point prediction diff vs V3/R1: `{selected_row['diff_vs_v3_point']:.4%}`
- pointId=3 pred count: `{int(selected_row['point3_pred_count'])}`

## Decision

R4 only changes point decision multipliers. It does not train or blend a new
point model. Submit only if the point gain is meaningful and the prediction
diff remains conservative. If the selected gain is tiny, prefer the existing
`submission_v10b_r1_safe.csv` or the submitted `submission_r1.csv`.
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
    point_pred = apply_segmented_multipliers(test_meta, point_prob, point_mult, POINT_CLASSES, "two")
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != expected_rows:
        raise ValueError("Submission row count mismatch.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    if not sub["actionId"].between(0, 18).all() or not sub["pointId"].between(0, 9).all():
        raise ValueError("Submission contains invalid classes.")
    if not sub["serverGetPoint"].between(0, 1).all():
        raise ValueError("Submission contains invalid server probabilities.")
    sub.to_csv(path, index=False, float_format="%.8f")
    return sub


def main() -> None:
    args = parse_args()
    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10b = load_pickle(args.v10b_oof)
    selected_v10 = json.loads(Path(args.v10b_r1_selected).read_text(encoding="utf-8"))
    meta = aligned_meta(v3, v5, v7, v10b)

    r1_action_prob, point_prob, r1_server_prob = r1_oof_probs(v3, v5, v7)
    safe_action_prob, _, safe_server_prob = safe_oof_probs(v3, v5, v7, v10b, selected_v10)
    safe_action_pred = apply_segmented_multipliers(
        meta, safe_action_prob, selected_v10["action_multipliers"], ACTION_CLASSES, "two"
    )
    r1_action_pred = apply_segmented_multipliers(
        meta, r1_action_prob, selected_v10["action_multipliers"], ACTION_CLASSES, "two"
    )
    safe_action_f1 = f1_score(meta["next_actionId"], safe_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    r1_action_f1 = f1_score(meta["next_actionId"], r1_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    safe_server_auc = roc_auc_score(meta["serverGetPoint"], safe_server_prob)
    r1_server_auc = roc_auc_score(meta["serverGetPoint"], r1_server_prob)

    base_point_f1, base_pred = score_point(meta, point_prob, v3["tuning"].point_multipliers)
    summary, selected_mult, selected_pred = search_r4(
        meta,
        point_prob,
        v3["tuning"].point_multipliers,
        base_pred,
        float(safe_action_f1),
        float(safe_server_auc),
        args,
    )
    summary.to_csv(args.summary, index=False)
    class_report_df = pd.concat(
        [
            class_report(meta, base_pred, "base_v3_point"),
            class_report(meta, selected_pred, "r4_selected_point"),
        ],
        ignore_index=True,
    )
    class_report_df.to_csv(args.class_report, index=False)
    safe_prefix = prefix_report(meta, safe_action_pred, selected_pred, safe_server_prob)
    safe_prefix.to_csv(args.prefix_report, index=False)

    selected_row = summary.iloc[0]
    safe_overall = 0.4 * safe_action_f1 + 0.4 * float(selected_row["point_macro_f1"]) + 0.2 * safe_server_auc
    r1_overall = 0.4 * r1_action_f1 + 0.4 * float(selected_row["point_macro_f1"]) + 0.2 * r1_server_auc
    selected_payload = {
        "base_point_macro_f1": base_point_f1,
        "safe_action_macro_f1": float(safe_action_f1),
        "safe_server_auc": float(safe_server_auc),
        "safe_overall": float(safe_overall),
        "r1_action_macro_f1_with_v10_multiplier": float(r1_action_f1),
        "r1_server_auc": float(r1_server_auc),
        "r1_overall_with_r4_point": float(r1_overall),
        "point_multipliers": selected_mult,
        "selected_row": selected_row.to_dict(),
        "limits": {"max_diff": args.max_diff, "max_point3_pred": args.max_point3_pred},
    }
    Path(args.selected).write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    write_recommendation(Path(args.recommendation), base_point_f1, selected_row, safe_overall, r1_overall)

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

    make_submission(
        Path(args.submission),
        test_meta,
        full_safe_action,
        selected_v10["action_multipliers"],
        test_point,
        selected_mult,
        full_safe_server,
        test["rally_uid"].nunique(),
    )
    make_submission(
        Path(args.submission_r1),
        test_meta,
        full_r1_action,
        selected_v10["action_multipliers"],
        test_point,
        selected_mult,
        full_r1_server,
        test["rally_uid"].nunique(),
    )
    print(f"base point={base_point_f1:.6f}")
    print(f"selected point={float(selected_row['point_macro_f1']):.6f}")
    print(f"safe overall={safe_overall:.6f}")
    print(f"wrote {args.submission}, {args.submission_r1}, {args.summary}, {args.selected}")


if __name__ == "__main__":
    main()
