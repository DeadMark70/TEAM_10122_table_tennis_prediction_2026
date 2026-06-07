import pandas as pd

from analysis_v383_synthetic_adjusted_packager import (
    block_unsupported_point0,
    package_candidate,
    select_supported_point_updates,
)


def test_packager_blocks_point0_without_support():
    rows = pd.DataFrame(
        {"base_point": [8, 0], "candidate_point": [0, 0], "synthetic_score": [0.2, 1.0]}
    )

    out = block_unsupported_point0(rows, min_score=0.8)

    assert out["allowed"].tolist() == [False, True]


def test_submission_preserves_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [10, 11],
            "pointId": [8, 9],
            "serverGetPoint": [0.2, 0.8],
        }
    )

    out = package_candidate(anchor, point_updates={1: 7}, action_updates={})

    assert out["serverGetPoint"].tolist() == [0.2, 0.8]


def test_select_supported_point_updates_prefers_nonterminal_supported_rows():
    scores = pd.DataFrame(
        {
            "rally_uid": [10, 11, 12, 13],
            "candidate_point": [7, 0, 6, 8],
            "is_point0_addition": [False, True, False, False],
            "same_depth": [True, False, False, True],
            "source_family_count": [7, 9, 7, 2],
            "support_count": [60, 80, 60, 60],
            "synthetic_teacher_score": [0.19, 0.99, 0.15, 0.20],
            "synthetic_adjusted_score": [10.0, 99.0, 9.0, 50.0],
        }
    )

    updates = select_supported_point_updates(scores, budget=2)

    assert updates == {10: 7, 12: 6}
