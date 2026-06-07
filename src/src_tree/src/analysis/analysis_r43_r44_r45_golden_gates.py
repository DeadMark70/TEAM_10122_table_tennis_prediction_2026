"""R43/R44/R45 golden-current action gates.

R42 proved that a light golden action soft-prob blend improves public LB.
This script keeps the same safety boundary:
  - action: current R33/R34 action probabilities gated with golden soft probs
  - point: current R34 hard point labels
  - server: current R34 server probabilities

It generates three families:
  R43 class-aware: trust golden more on classes where R41 showed V64 wins.
  R44 prefix-aware: use stronger golden weights on short prefixes.
  R45 confidence-aware: use golden only when its margin/confidence is favorable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_lgbm import ACTION_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import (
    CURRENT_SUB_PATH,
    OUT_DIR as R42_OUT_DIR,
    UPLOAD_DIR,
    build_current_r33_action_prob,
    normalize_rows,
    read_golden,
)


OUT_DIR = Path("r43_r44_r45_golden_gates")
STRONG_GOLDEN_CLASSES = {0, 3, 4, 7, 8, 9, 11, 12, 14}
CURRENT_STRONG_CLASSES = {1, 2, 5, 10, 13}


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


def margins(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-prob, axis=1)
    top = order[:, 0]
    top1 = prob[np.arange(len(prob)), order[:, 0]]
    top2 = prob[np.arange(len(prob)), order[:, 1]]
    return top, top1, top1 - top2


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def class_aware_blend(current: np.ndarray, golden: np.ndarray, high_w: float, low_w: float) -> np.ndarray:
    weights = np.full(current.shape[1], low_w, dtype=float)
    for c in STRONG_GOLDEN_CLASSES:
        weights[c] = high_w
    for c in CURRENT_STRONG_CLASSES:
        weights[c] = min(weights[c], low_w)
    blended = (1.0 - weights[None, :]) * current + weights[None, :] * golden
    return normalize_rows(blended)


def prefix_aware_blend(meta: pd.DataFrame, current: np.ndarray, golden: np.ndarray, w1: float, w2: float, w3p: float) -> np.ndarray:
    plen = meta["prefix_len"].to_numpy(dtype=int)
    w = np.where(plen == 1, w1, np.where(plen == 2, w2, w3p)).astype(float)
    return normalize_rows((1.0 - w[:, None]) * current + w[:, None] * golden)


def confidence_aware_blend(
    current: np.ndarray,
    golden: np.ndarray,
    base_w: float,
    strong_w: float,
    min_golden_prob: float,
    margin_delta: float,
    require_strong_class: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cur_top, cur_p, cur_m = margins(current)
    gold_top, gold_p, gold_m = margins(golden)
    mask = (gold_p >= min_golden_prob) & (gold_m >= cur_m + margin_delta)
    if require_strong_class:
        mask &= np.isin(gold_top, list(STRONG_GOLDEN_CLASSES))
    w = np.full(len(current), base_w, dtype=float)
    w[mask] = strong_w
    return normalize_rows((1.0 - w[:, None]) * current + w[:, None] * golden), mask


def write_submission(
    test_meta: pd.DataFrame,
    action_prob: np.ndarray,
    current_sub: pd.DataFrame,
    action_mult: dict,
    name: str,
) -> dict:
    action_pred = apply_segmented_multipliers(test_meta, action_prob, action_mult, ACTION_CLASSES, "two")
    current_action = current_sub["actionId"].to_numpy(dtype=int)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "path": str(path),
        "upload_path": str(UPLOAD_DIR / name),
        "action_diff_vs_current_r34": float(np.mean(action_pred != current_action)),
        "action0_count": int((action_pred == 0).sum()),
        "action8_count": int((action_pred == 8).sum()),
        "action9_count": int((action_pred == 9).sum()),
        "action12_count": int((action_pred == 12).sum()),
        "action14_count": int((action_pred == 14).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    test_meta, current_action_prob, selected = build_current_r33_action_prob()
    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")
    golden_action, _, _, _ = read_golden(test_meta)

    rows: list[dict] = []

    # R43: class-aware per-action probability blending.
    for high_w, low_w in [(0.35, 0.05), (0.50, 0.05), (0.65, 0.10), (0.80, 0.10)]:
        prob = class_aware_blend(current_action_prob, golden_action, high_w=high_w, low_w=low_w)
        rows.append(
            write_submission(
                test_meta,
                prob,
                current_sub,
                selected["action_multipliers"],
                f"submission_r43_classaware_high{str(high_w).replace('.', 'p')}_low{str(low_w).replace('.', 'p')}.csv",
            )
        )
        rows[-1].update({"family": "R43_class_aware", "high_w": high_w, "low_w": low_w})

    # R44: prefix-aware row-wise blending.
    for w1, w2, w3p in [(0.50, 0.25, 0.10), (0.65, 0.35, 0.10), (1.00, 0.50, 0.10), (0.75, 0.50, 0.20)]:
        prob = prefix_aware_blend(test_meta, current_action_prob, golden_action, w1=w1, w2=w2, w3p=w3p)
        rows.append(
            write_submission(
                test_meta,
                prob,
                current_sub,
                selected["action_multipliers"],
                f"submission_r44_prefix_w1_{str(w1).replace('.', 'p')}_w2_{str(w2).replace('.', 'p')}_w3p_{str(w3p).replace('.', 'p')}.csv",
            )
        )
        rows[-1].update({"family": "R44_prefix_aware", "w1": w1, "w2": w2, "w3p": w3p})

    # R45: confidence-aware row gates. Base w=0.20 preserves the public-positive R42 behavior.
    for strong_w, minp, md, strong_only in [
        (0.50, 0.25, 0.00, True),
        (0.65, 0.25, 0.02, True),
        (0.50, 0.35, -0.02, False),
        (0.80, 0.35, 0.00, True),
        (0.65, 0.45, -0.05, False),
    ]:
        prob, mask = confidence_aware_blend(
            current_action_prob,
            golden_action,
            base_w=0.20,
            strong_w=strong_w,
            min_golden_prob=minp,
            margin_delta=md,
            require_strong_class=strong_only,
        )
        name = (
            "submission_r45_conf_base0p2"
            f"_strong{str(strong_w).replace('.', 'p')}"
            f"_minp{str(minp).replace('.', 'p')}"
            f"_md{str(md).replace('-', 'm').replace('.', 'p')}"
            f"_{'strongonly' if strong_only else 'anyclass'}.csv"
        )
        rows.append(write_submission(test_meta, prob, current_sub, selected["action_multipliers"], name))
        rows[-1].update(
            {
                "family": "R45_confidence_aware",
                "base_w": 0.20,
                "strong_w": strong_w,
                "min_golden_prob": minp,
                "margin_delta": md,
                "require_strong_class": strong_only,
                "gate_rows": int(mask.sum()),
                "gate_rate": float(mask.mean()),
            }
        )

    # Baseline reference row for comparison.
    r42w02 = R42_OUT_DIR / "submission_r42_golden_action_w0p2_current_point_server.csv"
    if r42w02.exists():
        (UPLOAD_DIR / r42w02.name).write_bytes(r42w02.read_bytes())

    summary = pd.DataFrame(rows).sort_values(["action_diff_vs_current_r34", "candidate"])
    summary.to_csv(OUT_DIR / "r43_r44_r45_summary.csv", index=False)
    report = {
        "public_anchor": {
            "submission": "submission_r42_golden_action_w0p2_current_point_server.csv",
            "pl": 0.3342886,
            "time": "2026-05-19 14:12:20",
        },
        "strong_golden_classes": sorted(STRONG_GOLDEN_CLASSES),
        "current_strong_classes": sorted(CURRENT_STRONG_CLASSES),
        "generated": rows,
        "recommendation": [
            "First try a moderate R44 prefix-aware candidate or R43 class-aware candidate.",
            "Keep point/server fixed unless action-only candidates stop improving.",
        ],
    }
    (OUT_DIR / "r43_r44_r45_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
