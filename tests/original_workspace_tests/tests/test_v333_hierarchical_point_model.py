from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis_v333_hierarchical_point_model import (
    build_export_frame,
    compose_point_probabilities,
    depth_side_to_point,
    evidence_passes,
    point_to_depth_side,
    protected_output_path,
    select_variant_predictions,
)


def test_point_depth_side_mapping_roundtrips():
    seen = set()
    for point_id in range(1, 10):
        depth, side = point_to_depth_side(point_id)
        assert depth in (0, 1, 2)
        assert side in (0, 1, 2)
        assert depth_side_to_point(depth, side) == point_id
        seen.add((depth, side))

    assert len(seen) == 9
    assert point_to_depth_side(0) == (-1, -1)
    with pytest.raises(ValueError):
        point_to_depth_side(10)
    with pytest.raises(ValueError):
        depth_side_to_point(3, 0)


def test_point_probabilities_normalize():
    terminal = np.array([[0.8, 0.2], [0.25, 0.75]])
    depth = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    sides = {
        0: np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]),
        1: np.array([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]]),
        2: np.array([[0.6, 0.2, 0.2], [0.2, 0.5, 0.3]]),
    }

    prob = compose_point_probabilities(terminal, depth, sides)

    assert prob.shape == (2, 10)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert prob[0, 0] == pytest.approx(0.2)
    assert prob[0, 1] == pytest.approx(0.8 * 0.5 * 0.2)
    assert prob[1, 9] == pytest.approx(0.25 * 0.7 * 0.3)


def test_no_point0_add_variant_blocks_p0_additions():
    anchor = np.array([1, 2, 0, 7])
    prob = np.zeros((4, 10), dtype=float)
    prob[0, 0] = 0.9
    prob[1, 5] = 0.8
    prob[2, 3] = 0.7
    prob[3, 0] = 0.95
    prob += 0.01

    pred, selected, _ = select_variant_predictions(anchor, prob, budget=4, selector="no_p0_add")

    assert pred.tolist() == [1, 5, 3, 7]
    assert selected.tolist() == [False, True, True, False]
    assert not np.any((anchor != 0) & (pred == 0))


def test_export_preserves_action_and_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102],
            "actionId": [4, 15],
            "pointId": [0, 9],
            "serverGetPoint": [0.25, 0.75],
        }
    )

    out = build_export_frame(anchor, np.array([3, 8]))

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [4, 15]
    assert out["serverGetPoint"].tolist() == [0.25, 0.75]
    assert out["pointId"].tolist() == [3, 8]


def test_no_export_when_anchor_is_fallback():
    row = {
        "point_oof_delta_vs_v306": 0.25,
        "test_changed_rows": 12,
        "test_point0_total": 10,
        "anchor_point0_total": 10,
    }

    assert evidence_passes(row, anchor_is_fallback=False)
    assert not evidence_passes(row, anchor_is_fallback=True)


def test_protected_output_path_blocks_banned_locations():
    outdir = Path("v333_hierarchical_point_model")
    path = protected_output_path(outdir, "submission_v333_ok.csv")

    assert path.parent == outdir

    with pytest.raises(ValueError, match="refusing non-local V333 export path"):
        protected_output_path(outdir, "../upload_candidates/bad.csv")
