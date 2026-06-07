"""R3 pointId class-wise error and confusion analysis.

This script focuses on the current strongest point policy: V3 hierarchical
point probabilities with V3 segmented multipliers. R1 keeps the same point
policy, so this is also the point analysis for the submitted R1.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from baseline_lgbm import POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


PREFIX_BINS = [
    ("1", lambda s: s.eq(1)),
    ("2", lambda s: s.eq(2)),
    ("3", lambda s: s.eq(3)),
    ("4-6", lambda s: s.between(4, 6)),
    ("7+", lambda s: s.ge(7)),
    ("le2", lambda s: s.le(2)),
    ("ge3", lambda s: s.ge(3)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R3 pointId error analysis.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--out-dir", default=".")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_point_prob(oof: dict) -> tuple[pd.DataFrame, np.ndarray]:
    tuning = oof["tuning"]
    meta = oof["valid_meta"].reset_index(drop=True).copy()
    point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    return meta, point


def point_pred(meta: pd.DataFrame, prob: np.ndarray, multipliers: dict[str, list[float]], bins_mode: str) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, multipliers, POINT_CLASSES, bins_mode)


def class_report(meta: pd.DataFrame, pred: np.ndarray, prob: np.ndarray, variant: str) -> pd.DataFrame:
    y = meta["next_pointId"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=POINT_CLASSES, zero_division=0
    )
    rows = []
    for idx, cls in enumerate(POINT_CLASSES):
        cls_mask = y == cls
        pred_mask = pred == cls
        rows.append(
            {
                "variant": variant,
                "pointId": cls,
                "support": int(support[idx]),
                "pred_count": int(pred_mask.sum()),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "mean_prob_on_true": float(prob[cls_mask, idx].mean()) if cls_mask.any() else np.nan,
                "mean_prob_all": float(prob[:, idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def prefix_class_report(meta: pd.DataFrame, pred: np.ndarray, variant: str) -> pd.DataFrame:
    y = meta["next_pointId"].to_numpy(dtype=int)
    rows = []
    for label, fn in PREFIX_BINS:
        mask = fn(meta["prefix_len"]).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        precision, recall, f1, support = precision_recall_fscore_support(
            y[idx], pred[idx], labels=POINT_CLASSES, zero_division=0
        )
        macro = f1_score(y[idx], pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0)
        for cls_idx, cls in enumerate(POINT_CLASSES):
            rows.append(
                {
                    "variant": variant,
                    "prefix_len_bin": label,
                    "pointId": cls,
                    "bin_count": int(len(idx)),
                    "bin_macro_f1": float(macro),
                    "support": int(support[cls_idx]),
                    "pred_count": int((pred[idx] == cls).sum()),
                    "precision": float(precision[cls_idx]),
                    "recall": float(recall[cls_idx]),
                    "f1": float(f1[cls_idx]),
                }
            )
    return pd.DataFrame(rows)


def write_confusions(meta: pd.DataFrame, pred: np.ndarray, variant: str, out_dir: Path) -> pd.DataFrame:
    y = meta["next_pointId"].to_numpy(dtype=int)
    all_rows = []
    for label, mask in [("all", np.ones(len(meta), dtype=bool))] + [(name, fn(meta["prefix_len"]).to_numpy()) for name, fn in PREFIX_BINS]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        cm = confusion_matrix(y[idx], pred[idx], labels=POINT_CLASSES)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in POINT_CLASSES], columns=[f"pred_{c}" for c in POINT_CLASSES])
        cm_df.to_csv(out_dir / f"r3_confusion_{variant}_{label}.csv")
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
        cm_norm_df = pd.DataFrame(cm_norm, index=[f"true_{c}" for c in POINT_CLASSES], columns=[f"pred_{c}" for c in POINT_CLASSES])
        cm_norm_df.to_csv(out_dir / f"r3_confusion_{variant}_{label}_rownorm.csv")
        for true_idx, true_cls in enumerate(POINT_CLASSES):
            for pred_idx, pred_cls in enumerate(POINT_CLASSES):
                count = int(cm[true_idx, pred_idx])
                if count <= 0 or true_cls == pred_cls:
                    continue
                all_rows.append(
                    {
                        "variant": variant,
                        "prefix_len_bin": label,
                        "true_pointId": true_cls,
                        "pred_pointId": pred_cls,
                        "count": count,
                        "row_rate": float(cm_norm[true_idx, pred_idx]),
                    }
                )
    return pd.DataFrame(all_rows).sort_values(["prefix_len_bin", "count"], ascending=[True, False])


def prediction_distribution(meta: pd.DataFrame, pred: np.ndarray, variant: str) -> pd.DataFrame:
    rows = []
    for label, fn in [("all", lambda s: pd.Series(np.ones(len(s), dtype=bool), index=s.index))] + PREFIX_BINS:
        mask = fn(meta["prefix_len"]).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        y_counts = meta.iloc[idx]["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0)
        p_counts = pd.Series(pred[idx]).value_counts().reindex(POINT_CLASSES, fill_value=0)
        for cls in POINT_CLASSES:
            rows.append(
                {
                    "variant": variant,
                    "prefix_len_bin": label,
                    "pointId": cls,
                    "true_count": int(y_counts.loc[cls]),
                    "pred_count": int(p_counts.loc[cls]),
                    "true_rate": float(y_counts.loc[cls] / len(idx)),
                    "pred_rate": float(p_counts.loc[cls] / len(idx)),
                    "pred_minus_true_rate": float((p_counts.loc[cls] - y_counts.loc[cls]) / len(idx)),
                }
            )
    return pd.DataFrame(rows)


def compare_v10b_point(v3_meta: pd.DataFrame, v3_pred: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.v10b_oof)
    if not path.exists():
        return pd.DataFrame()
    v10 = load_pickle(str(path))
    meta = v10["valid_meta"].reset_index(drop=True)
    cols = ["rally_uid", "prefix_len", "next_pointId"]
    if not meta[cols].equals(v3_meta[cols].reset_index(drop=True)):
        return pd.DataFrame([{"comparison": "v10b_alignment_failed"}])
    v10_point = v10["v10_point"]
    rows = []
    for w in [0.0, 0.05, 0.1, 0.2]:
        mixed = blend_probs(v3_prob_global, v10_point, w)  # filled by main before call
        mult = tune_segmented_multipliers(meta, mixed, POINT_CLASSES, "point", "two")
        pred = apply_segmented_multipliers(meta, mixed, mult, POINT_CLASSES, "two")
        rows.append(
            {
                "point_v10_weight": w,
                "point_macro_f1": float(
                    f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
                ),
                "diff_vs_v3_pred": float((pred != v3_pred).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_recommendation(
    report: pd.DataFrame,
    prefix_report_df: pd.DataFrame,
    top_confusions: pd.DataFrame,
    dist: pd.DataFrame,
    v10_compare: pd.DataFrame,
    out_dir: Path,
) -> None:
    weak = report[(report["variant"].eq("v3_point")) & (report["f1"].lt(0.05))]
    low_recall = report[(report["variant"].eq("v3_point")) & (report["recall"].lt(0.05))]
    summary = report[report["variant"].eq("v3_point")][
        ["pointId", "support", "pred_count", "precision", "recall", "f1"]
    ]
    le2 = prefix_report_df[
        (prefix_report_df["variant"].eq("v3_point"))
        & (prefix_report_df["prefix_len_bin"].eq("le2"))
    ][["pointId", "support", "pred_count", "precision", "recall", "f1"]]
    ge3 = prefix_report_df[
        (prefix_report_df["variant"].eq("v3_point"))
        & (prefix_report_df["prefix_len_bin"].eq("ge3"))
    ][["pointId", "support", "pred_count", "precision", "recall", "f1"]]

    def md(df: pd.DataFrame, n: int | None = None) -> str:
        if n is not None:
            df = df.head(n)
        if df.empty:
            return "_empty_"
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.6f}" if isinstance(v, float) else str(v))
        header = "| " + " | ".join(out.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in out.astype(str).to_numpy()]
        return "\n".join([header, sep] + body)

    lines = [
        "# R3 PointId Error Analysis",
        "",
        "## Overall Class Report",
        md(summary),
        "",
        "## Low-Recall Classes",
        md(low_recall[["pointId", "support", "pred_count", "precision", "recall", "f1"]]),
        "",
        "## Prefix <= 2 Class Report",
        md(le2),
        "",
        "## Prefix >= 3 Class Report",
        md(ge3),
        "",
        "## Top Confusions",
        md(top_confusions[["prefix_len_bin", "true_pointId", "pred_pointId", "count", "row_rate"]], 20),
        "",
        "## V10B Point Probe",
        md(v10_compare),
        "",
        "## Interpretation",
        "- If classes have high support but near-zero recall, R4 should focus on constrained class multiplier / decision tuning.",
        "- If predicted distribution heavily under-predicts true classes, avoid adding new point models before testing conservative multiplier changes.",
        "- If V10B point weight improves OOF by only a tiny amount while changing many predictions, keep point fixed to V3 in submissions.",
        "",
    ]
    (out_dir / "r3_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


v3_prob_global: np.ndarray


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = load_pickle(args.v3_oof)
    meta, prob = normalize_point_prob(oof)
    global v3_prob_global
    v3_prob_global = prob
    pred = point_pred(meta, prob, oof["tuning"].point_multipliers, oof["tuning"].bins_mode)

    report = class_report(meta, pred, prob, "v3_point")
    report.to_csv(out_dir / "r3_point_class_report.csv", index=False)
    prefix_rep = prefix_class_report(meta, pred, "v3_point")
    prefix_rep.to_csv(out_dir / "r3_point_prefix_class_report.csv", index=False)
    top_confusions = write_confusions(meta, pred, "v3_point", out_dir)
    top_confusions.to_csv(out_dir / "r3_point_top_confusions.csv", index=False)
    dist = prediction_distribution(meta, pred, "v3_point")
    dist.to_csv(out_dir / "r3_point_prediction_distribution.csv", index=False)
    v10_compare = compare_v10b_point(meta, pred, args)
    v10_compare.to_csv(out_dir / "r3_v10b_point_probe.csv", index=False)
    write_recommendation(report, prefix_rep, top_confusions, dist, v10_compare, out_dir)

    print("overall point macro f1", f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0))
    print(report[["pointId", "support", "pred_count", "precision", "recall", "f1"]].to_string(index=False))
    print("wrote R3 reports")


if __name__ == "__main__":
    main()
