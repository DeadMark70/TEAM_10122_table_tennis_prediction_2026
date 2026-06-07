import numpy as np

from analysis_v259_v262_breakthrough_helpers import (
    ACTION_FAMILY,
    action_family,
    normalize_rows_safe,
    point_depth,
    point_side,
    verdict_from_deltas,
)


def test_action_family_mapping():
    assert action_family(0) == ACTION_FAMILY["Zero"]
    for value in range(1, 8):
        assert action_family(value) == ACTION_FAMILY["Attack"]
    for value in range(8, 12):
        assert action_family(value) == ACTION_FAMILY["Control"]
    for value in range(12, 15):
        assert action_family(value) == ACTION_FAMILY["Defensive"]
    for value in range(15, 19):
        assert action_family(value) == ACTION_FAMILY["Serve"]


def test_point_geometry_mapping():
    assert point_depth(0) == 0
    assert [point_depth(x) for x in [1, 2, 3]] == [1, 1, 1]
    assert [point_depth(x) for x in [4, 5, 6]] == [2, 2, 2]
    assert [point_depth(x) for x in [7, 8, 9]] == [3, 3, 3]
    assert [point_side(x) for x in [1, 4, 7]] == [1, 1, 1]
    assert [point_side(x) for x in [2, 5, 8]] == [2, 2, 2]
    assert [point_side(x) for x in [3, 6, 9]] == [3, 3, 3]


def test_normalize_rows_safe():
    matrix = np.array([[1.0, 1.0], [0.0, 0.0], [np.nan, 5.0]])
    out = normalize_rows_safe(matrix)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert not np.isnan(out).any()


def test_verdict_from_deltas():
    assert verdict_from_deltas(0.004, 0.002) == "CANDIDATE_FOR_PUBLIC_PROBE"
    assert verdict_from_deltas(0.001, 0.0) == "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    assert verdict_from_deltas(-0.001, 0.0) == "LOCAL_NEGATIVE_DO_NOT_SUBMIT"
