import pandas as pd

from analysis_v290_shortcontrol411_specialist import (
    build_shortcontrol_feature_frame,
    filter_shortcontrol_candidates,
)


def test_shortcontrol_feature_frame_marks_receive_short_context():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 4],
            "lag0_pointId": [1, 9],
            "lag0_actionId": [15, 1],
            "lag0_spinId": [2, 0],
            "lag0_strengthId": [1, 3],
        }
    )
    out = build_shortcontrol_feature_frame(rows)
    assert out["is_receive_short"].tolist() == [1, 0]
    assert out["shortcontrol_context_score"].iloc[0] > out["shortcontrol_context_score"].iloc[1]


def test_shortcontrol_candidate_filter_only_allows_4_11_and_blocks_protected_anchor():
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "anchor_action": [1, 13, 10],
            "candidate_action": [4, 11, 5],
            "shortcontrol_score": [0.9, 0.9, 0.99],
            "support_count": [10, 10, 10],
        }
    )
    out = filter_shortcontrol_candidates(frame, min_score=0.5, min_support=5)
    assert out["candidate_action"].tolist() == [4]
