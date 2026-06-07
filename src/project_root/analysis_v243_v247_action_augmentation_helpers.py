"""Shared utilities for V243-V247 action augmentation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import normalize_probability_rows


def build_context_key_frame(rows: pd.DataFrame) -> pd.DataFrame:
    prefix = pd.to_numeric(rows.get("prefix_len", 0), errors="coerce").fillna(0).astype(int)
    prefix_bin = prefix.map(lambda v: "1" if v <= 1 else "2" if v == 2 else "3" if v == 3 else "4_6" if v <= 6 else "7_plus")
    return pd.DataFrame(
        {
            "prefix_bin": prefix_bin.astype(str),
            "phase": rows.get("audit_phase", pd.Series("unknown", index=rows.index)).astype(str).fillna("unknown"),
            "lag0_family": rows.get("audit_lag0_action_family", pd.Series("unknown", index=rows.index)).astype(str).fillna("unknown"),
            "lag0_depth": rows.get("audit_lag0_depth", pd.Series("unknown", index=rows.index)).astype(str).fillna("unknown"),
        },
        index=rows.index,
    )


def clip_density_weights(weights: np.ndarray, low: float = 0.25, high: float = 4.0) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 1.0)
    if arr.size == 0:
        return arr
    if not np.isfinite(arr.mean()) or arr.mean() <= 0:
        return np.ones_like(arr)
    arr = arr / arr.mean()
    lo = float(low)
    hi = float(high)
    for _ in range(20):
        arr = np.clip(arr, lo, hi)
        delta = float(arr.size) - float(arr.sum())
        if abs(delta) < 1e-12:
            break
        if delta > 0:
            free = arr < hi - 1e-12
            if not free.any():
                break
            arr[free] += delta / free.sum()
        else:
            free = arr > lo + 1e-12
            if not free.any():
                break
            arr[free] += delta / free.sum()
    return np.clip(arr, lo, hi)


def balanced_softmax_adjustment(prob: np.ndarray, counts: np.ndarray, strength: float = 0.5, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(normalize_probability_rows(prob), eps, 1.0)
    c = np.asarray(counts, dtype=float)
    c = np.where(np.isfinite(c) & (c > 0), c, 1.0)
    prior = c / c.sum()
    target = 1.0 / len(c)
    factor = np.power(target / np.clip(prior, eps, 1.0), float(strength))
    return normalize_probability_rows(p * factor[None, :])


def label_smoothed_targets(y: np.ndarray, n_classes: int, smoothing: float = 0.05) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    smooth = float(smoothing)
    out = np.full((len(labels), int(n_classes)), smooth / max(int(n_classes) - 1, 1), dtype=float)
    out[np.arange(len(labels)), labels] = 1.0 - smooth
    return normalize_probability_rows(out)


def mix_probabilities(anchor: np.ndarray, teacher: np.ndarray, weight: float) -> np.ndarray:
    a = normalize_probability_rows(anchor)
    t = normalize_probability_rows(teacher)
    return normalize_probability_rows((1.0 - float(weight)) * a + float(weight) * t)


def js_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pa = np.clip(normalize_probability_rows(a), eps, 1.0)
    pb = np.clip(normalize_probability_rows(b), eps, 1.0)
    m = 0.5 * (pa + pb)
    kl_a = np.sum(pa * (np.log(pa) - np.log(m)), axis=1)
    kl_b = np.sum(pb * (np.log(pb) - np.log(m)), axis=1)
    return np.sqrt(np.maximum(0.5 * (kl_a + kl_b), 0.0))
