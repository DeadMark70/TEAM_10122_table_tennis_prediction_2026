import numpy as np
import pandas as pd
import pytest

from analysis_v300_clean_server_blend_recycler import (
    blend_server_only,
    cap_prob,
    rank_normalize_to_anchor,
    validate_submission_frame,
)


def test_rank_normalize_to_anchor_maps_source_order_to_anchor_distribution():
    source = np.array([30.0, 10.0, 20.0])
    anchor = np.array([0.1, 0.2, 0.3])

    normalized = rank_normalize_to_anchor(source, anchor)

    assert normalized.tolist() == pytest.approx([0.3, 0.1, 0.2])


def test_rank_normalize_to_anchor_ties_use_midrank_quantile():
    source = np.array([1.0, 1.0, 3.0])
    anchor = np.array([0.1, 0.2, 0.3])

    normalized = rank_normalize_to_anchor(source, anchor)

    assert normalized.tolist() == pytest.approx([0.15, 0.15, 0.3])


def test_blending_preserves_rows_action_and_point():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [4, 10, 13],
            "pointId": [8, 5, 0],
            "serverGetPoint": [0.2, 0.5, 0.8],
        }
    )
    target_server = np.array([0.4, 0.4, 0.4])

    blended = blend_server_only(anchor, target_server, weight=0.25)

    assert blended[["rally_uid", "actionId", "pointId"]].equals(
        anchor[["rally_uid", "actionId", "pointId"]]
    )
    assert blended["serverGetPoint"].tolist() == pytest.approx([0.25, 0.475, 0.7])


def test_server_range_is_finite_after_capping_and_validation():
    values = cap_prob(np.array([-np.inf, 0.25, np.nan, 2.0]))

    assert np.isfinite(values).all()
    assert ((values > 0.0) & (values < 1.0)).all()

    df = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3, 4],
            "actionId": [1, 1, 1, 1],
            "pointId": [8, 8, 8, 8],
            "serverGetPoint": values,
        }
    )
    validate_submission_frame(df, expected_rows=4)
