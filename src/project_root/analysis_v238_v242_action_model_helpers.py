"""Shared helpers for V238-V242 action research scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_probability_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    sums = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, sums, out=np.full_like(arr, 1.0 / arr.shape[1]), where=sums > 0)


def blend_probabilities(anchor: np.ndarray, teacher: np.ndarray, weight: float, eps: float = 1e-8) -> np.ndarray:
    a = np.clip(normalize_probability_rows(anchor), eps, 1.0)
    t = np.clip(normalize_probability_rows(teacher), eps, 1.0)
    logp = (1.0 - float(weight)) * np.log(a) + float(weight) * np.log(t)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_probability_rows(np.exp(logp))


def topk_candidate_frame(anchor_action: np.ndarray, sources: dict[str, np.ndarray], top_k: int = 3) -> pd.DataFrame:
    anchor = np.asarray(anchor_action, dtype=int)
    rows = []
    for i, action in enumerate(anchor):
        rows.append({"row_id": i, "candidate_action": int(action), "source": "anchor", "source_prob": 1.0, "source_rank": 0, "is_anchor": 1})
        for source, prob in sources.items():
            p = normalize_probability_rows(prob)
            order = np.argsort(-p[i])[: int(top_k)]
            for rank, cand in enumerate(order, start=1):
                rows.append(
                    {
                        "row_id": i,
                        "candidate_action": int(cand),
                        "source": str(source),
                        "source_prob": float(p[i, cand]),
                        "source_rank": int(rank),
                        "is_anchor": 0,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["row_id", "candidate_action", "source"]).reset_index(drop=True)


def precision_constrained_threshold(scores: np.ndarray, labels: np.ndarray, min_precision: float = 0.80) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores)
    tp = 0
    fp = 0
    best = float(np.inf)
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        if precision >= float(min_precision):
            best = float(scores[idx])
    if not np.isfinite(best):
        return float(np.max(scores) + 1e-6)
    return best


def select_top_changes(anchor: np.ndarray, candidate: np.ndarray, score: np.ndarray, cap: float, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(anchor, dtype=int).copy()
    cand = np.asarray(candidate, dtype=int)
    score = np.asarray(score, dtype=float)
    diff = cand != out
    budget = int(np.floor(len(out) * float(cap)))
    chosen = np.zeros(len(out), dtype=bool)
    if budget <= 0:
        return out, chosen
    eligible = np.where(diff & np.isfinite(score) & (score >= float(min_score)))[0]
    if len(eligible) == 0:
        return out, chosen
    order = eligible[np.argsort(-score[eligible])[:budget]]
    out[order] = cand[order]
    chosen[order] = True
    return out, chosen
