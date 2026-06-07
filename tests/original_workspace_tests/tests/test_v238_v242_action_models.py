import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import (
    blend_probabilities,
    normalize_probability_rows,
    precision_constrained_threshold,
    topk_candidate_frame,
)


def test_normalize_probability_rows_handles_nan_and_zero_rows():
    x = np.array([[np.nan, 1.0, 1.0], [0.0, 0.0, 0.0]])
    out = normalize_probability_rows(x)

    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
    assert np.allclose(out[1], np.full(3, 1 / 3))


def test_blend_probabilities_moves_toward_teacher():
    anchor = np.array([[0.9, 0.1], [0.2, 0.8]])
    teacher = np.array([[0.1, 0.9], [0.8, 0.2]])
    out = blend_probabilities(anchor, teacher, 0.25)

    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 1] > anchor[0, 1]
    assert out[1, 0] > anchor[1, 0]


def test_topk_candidate_frame_contains_anchor_and_topk_sources():
    anchor = np.array([1, 2])
    sources = {"a": np.array([[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]])}
    frame = topk_candidate_frame(anchor, sources, top_k=2)

    assert set(["row_id", "candidate_action", "source", "is_anchor"]).issubset(frame.columns)
    assert frame.groupby("row_id")["is_anchor"].sum().tolist() == [1, 1]
    assert len(frame) >= 4


def test_precision_constrained_threshold_returns_high_threshold_for_precision():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    threshold = precision_constrained_threshold(scores, labels, min_precision=0.75)

    assert 0.8 < threshold <= 0.9
