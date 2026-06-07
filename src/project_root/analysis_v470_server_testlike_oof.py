"""V470 labeled test-like OOF metrics for clean server models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis_v465_clean_server_line import (
    ANCHOR_RELATIVE,
    EXPECTED_ROWS,
    ROOT,
    TEST_NEW_RELATIVE,
    TRAIN_RELATIVE,
    build_feature_matrices,
    no_banned_input_guard,
)
from analysis_v467_server_exhaustive_clean_sweep import fit_tabular_zoo
from analysis_v468_server_full_run import build_full_model_configs
from analysis_v469_server_public_like_validation import fit_density_weights, make_public_like_bins

OUT_DIR = ROOT / "v470_server_testlike_oof"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def weighted_auc(y: np.ndarray | pd.Series, pred: np.ndarray | pd.Series, weights: np.ndarray | pd.Series) -> float:
    y_arr = np.asarray(y, dtype=int)
    pred_arr = np.asarray(pred, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    if len(np.unique(y_arr)) < 2 or len(y_arr) == 0:
        return float("nan")
    try:
        auc = float(roc_auc_score(y_arr, pred_arr, sample_weight=weight_arr))
    except Exception:
        return float("nan")
    return auc if math.isfinite(auc) else float("nan")


def compute_model_metrics(y: np.ndarray, weights: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in predictions.items():
        pred_arr = np.asarray(pred, dtype=float)
        rows.append(
            {
                "model": name,
                "ordinary_auc": weighted_auc(y, pred_arr, np.ones_like(weights, dtype=float)),
                "testlike_weighted_auc": weighted_auc(y, pred_arr, weights),
                "prediction_mean": float(np.mean(pred_arr)),
                "prediction_std": float(np.std(pred_arr)),
            }
        )
    return pd.DataFrame(rows).sort_values(["testlike_weighted_auc", "ordinary_auc"], ascending=False).reset_index(drop=True)


def compute_slice_metrics(
    y: np.ndarray,
    weights: np.ndarray,
    predictions: dict[str, np.ndarray],
    slices: pd.DataFrame,
    slice_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for model, pred in predictions.items():
        for col in slice_cols:
            if col not in slices.columns:
                continue
            for value, idx in slices.groupby(col, sort=False).groups.items():
                index = np.asarray(list(idx), dtype=int)
                if len(index) < 2 or len(np.unique(y[index])) < 2:
                    continue
                rows.append(
                    {
                        "model": model,
                        "slice_name": col,
                        "slice_value": str(value),
                        "rows": int(len(index)),
                        "ordinary_auc": weighted_auc(y[index], pred[index], np.ones(len(index))),
                        "weighted_auc": weighted_auc(y[index], pred[index], weights[index]),
                        "weight_mean": float(np.mean(weights[index])),
                    }
                )
    return pd.DataFrame(rows)


def _write_report_md(path: Path, report: dict[str, Any], metrics: pd.DataFrame) -> None:
    top = metrics.head(15)
    lines = [
        "# V470 server test-like OOF",
        "",
        f"Runtime: {report['runtime']}",
        f"Models: {report['model_count']}",
        f"Weight min/mean/max: {report['weight_min']:.4f} / {report['weight_mean']:.4f} / {report['weight_max']:.4f}",
        "",
        "| model | ordinary_auc | testlike_weighted_auc |",
        "| --- | --- | --- |",
    ]
    for _, row in top.iterrows():
        lines.append(f"| {row['model']} | {row['ordinary_auc']:.6f} | {row['testlike_weighted_auc']:.6f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path | None = None,
    expected_rows: int = EXPECTED_ROWS,
    runtime: str = "fast",
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / OUT_DIR.name
    no_banned_input_guard([root / TRAIN_RELATIVE, root / TEST_NEW_RELATIVE, root / ANCHOR_RELATIVE, outdir])
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / TRAIN_RELATIVE)
    test_new = pd.read_csv(root / TEST_NEW_RELATIVE)
    anchor = pd.read_csv(root / ANCHOR_RELATIVE)
    if expected_rows is not None and len(anchor) != expected_rows:
        raise ValueError(f"anchor row count mismatch: {len(anchor)} != {expected_rows}")
    if "serverGetPoint" not in train.columns:
        raise ValueError("train.csv missing serverGetPoint")

    y = (pd.to_numeric(train["serverGetPoint"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 0.5).astype(int)
    train_bins = make_public_like_bins(train)
    test_bins = make_public_like_bins(test_new)
    weights = fit_density_weights(train_bins, test_bins)
    weights_path = outdir / "v470_train_public_like_weights.csv"
    train_bins.assign(weight=weights, serverGetPoint=y).to_csv(weights_path, index=False)

    train_x, test_x = build_feature_matrices(train, test_new, anchor)
    configs = [config for config in build_full_model_configs(runtime=runtime, seed=470) if "hist_gradient" not in config.name]
    optional_skips: dict[str, str] = {}
    signals = fit_tabular_zoo(
        train_x,
        y,
        test_x,
        groups=train["rally_uid"] if "rally_uid" in train.columns else None,
        configs=configs,
        skip_report=optional_skips,
    )
    predictions = {signal.name: signal.oof for signal in signals}
    metrics = compute_model_metrics(y, weights, predictions)
    metrics.to_csv(outdir / "v470_model_oof_metrics.csv", index=False)

    slices = train_bins.copy()
    slice_metrics = compute_slice_metrics(
        y,
        weights,
        predictions,
        slices,
        ["prefix_bin", "phase_bin", "score_pressure", "score_total_bin", "lag_action_family", "lag_point_depth"],
    )
    slice_metrics.to_csv(outdir / "v470_slice_oof_metrics.csv", index=False)

    best = metrics.iloc[0].to_dict() if not metrics.empty else {}
    report = {
        "pipeline": "v470_server_testlike_oof",
        "runtime": runtime,
        "model_count": int(len(signals)),
        "optional_model_skips": optional_skips,
        "weight_min": float(np.min(weights)),
        "weight_mean": float(np.mean(weights)),
        "weight_max": float(np.max(weights)),
        "best_model": best.get("model"),
        "best_ordinary_auc": best.get("ordinary_auc"),
        "best_testlike_weighted_auc": best.get("testlike_weighted_auc"),
        "outputs": {
            "model_metrics": str((outdir / "v470_model_oof_metrics.csv").resolve()),
            "slice_metrics": str((outdir / "v470_slice_oof_metrics.csv").resolve()),
            "weights": str(weights_path.resolve()),
        },
    }
    (outdir / "v470_report.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(outdir / "v470_report.md", report, metrics)
    print(json.dumps(_json_safe({"model_count": len(signals), "best_model": report["best_model"], "best_testlike_weighted_auc": report["best_testlike_weighted_auc"]}), sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
