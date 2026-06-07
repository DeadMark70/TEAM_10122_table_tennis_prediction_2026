"""Shared helpers for V250-V253 action source rebuild experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import normalize_probability_rows


def weighted_neighbor_action_prob(labels: np.ndarray, distances: np.ndarray, n_classes: int = 19, temperature: float = 1.0) -> np.ndarray:
    lab = np.asarray(labels, dtype=int)
    dist = np.asarray(distances, dtype=float)
    weights = np.exp(-np.maximum(dist, 0.0) / max(float(temperature), 1e-6))
    out = np.zeros((lab.shape[0], int(n_classes)), dtype=float)
    for i in range(lab.shape[0]):
        for j, cls in enumerate(lab[i]):
            if 0 <= int(cls) < int(n_classes):
                out[i, int(cls)] += weights[i, j]
    return normalize_probability_rows(out)


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train, dtype=float)
    z = np.asarray(test, dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return np.nan_to_num((x - mean) / std), np.nan_to_num((z - mean) / std)


def phase_family_one_hot(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for prefix, col in [
        ("phase", "audit_phase"),
        ("family", "audit_lag0_action_family"),
        ("depth", "audit_lag0_depth"),
    ]:
        vals = rows.get(col, pd.Series("missing", index=rows.index)).astype(str).fillna("missing")
        dummies = pd.get_dummies(vals, prefix=prefix, prefix_sep="=", dtype=float)
        parts.append(dummies)
    if not parts:
        return pd.DataFrame(index=rows.index)
    return pd.concat(parts, axis=1).astype(float)


def logit_adjust_probability(prob: np.ndarray, counts: np.ndarray, tau: float = 0.3, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(normalize_probability_rows(prob), eps, 1.0)
    c = np.asarray(counts, dtype=float)
    c = np.where(np.isfinite(c) & (c > 0), c, 1.0)
    prior = c / c.sum()
    logits = np.log(p) - float(tau) * np.log(np.clip(prior, eps, 1.0))[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probability_rows(np.exp(logits))


def confidence_margin(prob: np.ndarray) -> np.ndarray:
    p = normalize_probability_rows(prob)
    part = np.sort(p, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]
