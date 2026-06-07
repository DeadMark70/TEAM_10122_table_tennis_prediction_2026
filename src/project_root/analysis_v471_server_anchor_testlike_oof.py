"""V471: recompute old public-anchor server family with V470 test-like OOF.

This script compares the V300/V263/V269 clean server family against the V470
server models using the same train/test_new density weighting.  It does not
read old-server labels, TTMATCH, or produce new submissions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v263c_simple_server_probability import (
    build_feature_tables as build_v263_feature_tables,
    run_oof as run_v263_oof,
)
from analysis_v469_server_public_like_validation import fit_density_weights, make_public_like_bins
from analysis_v470_server_testlike_oof import weighted_auc
from src.analysis.analysis_v269_clean_server_value_ranker import run_oof as run_v269_oof


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v471_server_anchor_testlike_oof"
V470_METRICS = ROOT / "v470_server_testlike_oof" / "v470_model_oof_metrics.csv"
V263_WEIGHTS = (0.005, 0.010, 0.020)
V269_WEIGHTS = (0.005, 0.010, 0.020, 0.030, 0.050)
V266_REPACK_WEIGHT = 0.030
V263_TEACHER_SOURCE_WEIGHT = 0.020


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


def clip_prob(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def blend_prob(anchor: np.ndarray, teacher: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    anchor_arr = clip_prob(anchor)
    teacher_arr = clip_prob(teacher)
    if len(anchor_arr) != len(teacher_arr):
        raise ValueError("anchor and teacher lengths differ")
    return clip_prob((1.0 - float(weight)) * anchor_arr + float(weight) * teacher_arr)


def weight_name(weight: float) -> str:
    return f"{float(weight):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def derive_teacher_from_blend(proxy: np.ndarray, blend: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 < float(weight) <= 1.0:
        raise ValueError(f"weight must be in (0, 1], got {weight}")
    proxy_arr = np.asarray(proxy, dtype=float)
    blend_arr = np.asarray(blend, dtype=float)
    if len(proxy_arr) != len(blend_arr):
        raise ValueError("proxy and blend lengths differ")
    teacher = (blend_arr - (1.0 - float(weight)) * proxy_arr) / float(weight)
    return clip_prob(teacher)


def candidate_auc_rows(
    y: np.ndarray,
    weights: np.ndarray,
    candidates: dict[str, np.ndarray],
    *,
    compare_auc: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, pred in candidates.items():
        pred_arr = clip_prob(pred)
        rows.append(
            {
                "candidate": name,
                "ordinary_auc": weighted_auc(y, pred_arr, np.ones_like(weights, dtype=float)),
                "testlike_weighted_auc": weighted_auc(y, pred_arr, weights),
                "delta_vs_v470_best_testlike": weighted_auc(y, pred_arr, weights) - float(compare_auc),
                "prediction_mean": float(np.mean(pred_arr)),
                "prediction_std": float(np.std(pred_arr)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["testlike_weighted_auc", "ordinary_auc"],
        ascending=False,
    ).reset_index(drop=True)


def load_v470_best_metric(root: Path = ROOT) -> tuple[str, float]:
    path = root / "v470_server_testlike_oof" / "v470_model_oof_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path)
    required = {"model", "testlike_weighted_auc"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    metrics["testlike_weighted_auc"] = pd.to_numeric(metrics["testlike_weighted_auc"], errors="coerce")
    metrics = metrics.dropna(subset=["testlike_weighted_auc"]).sort_values("testlike_weighted_auc", ascending=False)
    if metrics.empty:
        raise ValueError(f"{path} has no finite testlike_weighted_auc")
    top = metrics.iloc[0]
    return str(top["model"]), float(top["testlike_weighted_auc"])


def prefix_frame_for_public_like_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Map V263/V269 prefix-table fields into V470's public-like bin schema."""
    out = pd.DataFrame(index=frame.index)
    out["strikeNumber"] = pd.to_numeric(frame.get("prefix_len", 1), errors="coerce").fillna(1)
    out["scoreSelf"] = pd.to_numeric(frame.get("serverScore", frame.get("scoreSelf", 0)), errors="coerce").fillna(0)
    out["scoreOther"] = pd.to_numeric(frame.get("receiverScore", frame.get("scoreOther", 0)), errors="coerce").fillna(0)
    out["actionId"] = pd.to_numeric(frame.get("lag0_actionId", frame.get("actionId", -1)), errors="coerce").fillna(-1)
    out["pointId"] = pd.to_numeric(frame.get("lag0_pointId", frame.get("pointId", -1)), errors="coerce").fillna(-1)
    return out


def build_public_like_weights(train_prefix: pd.DataFrame, test_prefix: pd.DataFrame) -> np.ndarray:
    train_bins = make_public_like_bins(prefix_frame_for_public_like_bins(train_prefix))
    test_bins = make_public_like_bins(prefix_frame_for_public_like_bins(test_prefix))
    return fit_density_weights(train_bins, test_bins)


def build_anchor_family_candidates(v263_oof: dict[str, Any], v269_oof: dict[str, Any]) -> dict[str, np.ndarray]:
    v263_proxy = clip_prob(np.asarray(v263_oof["oof_proxy"], dtype=float))
    v263_model = clip_prob(np.asarray(v263_oof["oof_model"], dtype=float))
    v269_proxy = clip_prob(np.asarray(v269_oof["oof_proxy"], dtype=float))
    v269_model = clip_prob(np.asarray(v269_oof["oof_model"], dtype=float))

    candidates: dict[str, np.ndarray] = {
        "v263_proxy_anchor": v263_proxy,
        "v263_model_raw": v263_model,
        "v269_proxy_anchor": v269_proxy,
        "v269_model_raw": v269_model,
    }
    for weight in V263_WEIGHTS:
        candidates[f"v263_blend_w{weight_name(weight)}"] = blend_prob(v263_proxy, v263_model, weight)
    for weight in V269_WEIGHTS:
        candidates[f"v269_blend_w{weight_name(weight)}"] = blend_prob(v269_proxy, v269_model, weight)

    v263_w02 = candidates["v263_blend_w0p02"]
    v266_teacher = derive_teacher_from_blend(v263_proxy, v263_w02, V263_TEACHER_SOURCE_WEIGHT)
    v266_repack = blend_prob(v263_proxy, v266_teacher, V266_REPACK_WEIGHT)
    candidates["v266_teacher_w0p03_approx"] = v266_repack
    candidates["v300_best_safe_repack_approx"] = v266_repack
    return candidates


def write_report_files(
    outdir: Path,
    metrics: pd.DataFrame,
    *,
    v470_best_model: str,
    v470_best_auc: float,
    weight_summary: dict[str, float] | None = None,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "v471_anchor_family_oof_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    anchor_only = metrics[~metrics["candidate"].astype(str).str.startswith("v470_")].copy()
    best_anchor = anchor_only.iloc[0].to_dict() if not anchor_only.empty else {}
    report = {
        "pipeline": "v471_server_anchor_testlike_oof",
        "best_anchor_family_candidate": best_anchor.get("candidate"),
        "best_anchor_family_ordinary_auc": best_anchor.get("ordinary_auc"),
        "best_anchor_family_testlike_weighted_auc": best_anchor.get("testlike_weighted_auc"),
        "v470_best_model": v470_best_model,
        "v470_best_testlike_weighted_auc": float(v470_best_auc),
        "best_anchor_family_delta_vs_v470": best_anchor.get("delta_vs_v470_best_testlike"),
        "weight_summary": weight_summary or {},
        "outputs": {"metrics": str(metrics_path.resolve())},
    }
    (outdir / "v471_report.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# V471 server anchor test-like OOF",
        "",
        "Recomputes the old V263/V269/V266/V300 clean server family under V470-style density-weighted OOF.",
        "",
        f"V470 best model: `{v470_best_model}`",
        f"V470 best test-like weighted AUC: `{v470_best_auc:.6f}`",
        "",
    ]
    if weight_summary:
        lines.extend(
            [
                f"Weight min/mean/max: `{weight_summary['min']:.4f}` / `{weight_summary['mean']:.4f}` / `{weight_summary['max']:.4f}`",
                f"Unique rounded weights: `{int(weight_summary['unique_rounded'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "| candidate | ordinary_auc | testlike_weighted_auc | delta_vs_v470 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in metrics.head(20).iterrows():
        lines.append(
            f"| `{row['candidate']}` | {row['ordinary_auc']:.6f} | "
            f"{row['testlike_weighted_auc']:.6f} | {row['delta_vs_v470_best_testlike']:.6f} |"
        )
    (outdir / "v471_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_pipeline(*, outdir: Path | None = None) -> dict[str, Any]:
    outdir = OUT_DIR if outdir is None else Path(outdir)
    v470_best_model, v470_best_auc = load_v470_best_metric(ROOT)

    train_prefix, test_prefix, features = build_v263_feature_tables()
    weights = build_public_like_weights(train_prefix, test_prefix)
    weight_summary = {
        "min": float(np.min(weights)),
        "mean": float(np.mean(weights)),
        "max": float(np.max(weights)),
        "unique_rounded": float(len(set(np.round(weights, 6)))),
    }
    v263_oof = run_v263_oof(train_prefix, features)
    v269_oof = run_v269_oof(train_prefix, features)

    y263 = np.asarray(v263_oof["y"], dtype=int)
    y269 = np.asarray(v269_oof["y"], dtype=int)
    if not np.array_equal(y263, y269):
        raise ValueError("V263 and V269 OOF labels are not aligned")
    if len(weights) != len(y263):
        raise ValueError(f"weight length {len(weights)} does not match OOF rows {len(y263)}")

    candidates = build_anchor_family_candidates(v263_oof, v269_oof)
    metrics = candidate_auc_rows(y263, weights, candidates, compare_auc=v470_best_auc)
    v470_row = pd.DataFrame(
        [
            {
                "candidate": f"v470_best__{v470_best_model}",
                "ordinary_auc": float("nan"),
                "testlike_weighted_auc": v470_best_auc,
                "delta_vs_v470_best_testlike": 0.0,
                "prediction_mean": float("nan"),
                "prediction_std": float("nan"),
            }
        ]
    )
    metrics = pd.concat([metrics, v470_row], ignore_index=True)
    report = write_report_files(
        outdir,
        metrics,
        v470_best_model=v470_best_model,
        v470_best_auc=v470_best_auc,
        weight_summary=weight_summary,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "outdir": str(outdir),
                    "best_anchor_family_candidate": report["best_anchor_family_candidate"],
                    "best_anchor_family_testlike_weighted_auc": report["best_anchor_family_testlike_weighted_auc"],
                    "v470_best_model": v470_best_model,
                    "v470_best_testlike_weighted_auc": v470_best_auc,
                    "delta": report["best_anchor_family_delta_vs_v470"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
