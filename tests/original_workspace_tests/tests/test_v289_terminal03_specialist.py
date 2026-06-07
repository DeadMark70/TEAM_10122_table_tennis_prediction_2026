import pandas as pd

from analysis_v289_terminal03_specialist import build_terminal_feature_frame, filter_terminal_candidates


def test_terminal_feature_frame_adds_terminal_context():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 4],
            "lag0_pointId": [0, 7],
            "lag0_actionId": [13, 1],
            "scoreTotal": [20, 3],
            "serverScoreDiff": [0, 2],
        }
    )
    out = build_terminal_feature_frame(rows)
    assert out["lag0_point_is_zero"].tolist() == [1, 0]
    assert out["is_late_pressure"].tolist() == [1, 0]
    assert out["terminal_context_score"].iloc[0] > out["terminal_context_score"].iloc[1]


def test_terminal_candidate_filter_only_allows_0_3():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "candidate_action": [0, 3, 5],
            "terminal_score": [0.9, 0.8, 0.99],
            "support_count": [10, 10, 10],
        }
    )
    out = filter_terminal_candidates(frame, min_score=0.5, min_support=5)
    assert out["candidate_action"].tolist() == [0, 3]
