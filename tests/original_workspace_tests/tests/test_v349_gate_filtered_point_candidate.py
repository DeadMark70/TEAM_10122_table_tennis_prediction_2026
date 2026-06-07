import pandas as pd

from analysis_v349_gate_filtered_point_candidate import (
    build_fallback_gate_scores,
    build_submission_from_rows,
    extra_nonp0_pool,
    point0_pool,
)


def test_fallback_gate_scores_penalizes_v341_extra_risk():
    bank = pd.DataFrame(
        {
            "row_id": [1, 2],
            "rally_uid": ["a", "b"],
            "task": ["point", "point"],
            "anchor_value": [8, 8],
            "candidate_value": [9, 7],
            "source": ["submission_v338_safe", "submission_v341_extra"],
            "source_dir": ["v338_joint_moe_pack", "v341_no_p0_point_pack"],
            "source_public_tag": ["v338_public_positive", "no_p0_expansion"],
            "changed_in_v338": [True, False],
            "source_local_delta_if_known": [0.003, 0.0],
        }
    )
    scores = build_fallback_gate_scores(bank)
    safe = scores[scores["row_id"].eq(1)].iloc[0]
    risky = scores[scores["row_id"].eq(2)].iloc[0]
    assert safe["trust_score"] > risky["trust_score"]
    assert bool(risky["v341_extra_risk"])


def test_extra_nonp0_pool_excludes_v338_and_point0_rows():
    scores = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "anchor_value": [8, 8, 8],
            "candidate_value": [9, 7, 0],
            "changed_in_v338": [False, True, False],
            "nonterminal_swap": [True, True, False],
            "point0_addition": [False, False, True],
            "trust_score": [1.0, 5.0, 5.0],
            "risk_score": [0.0, 0.0, 0.0],
            "agreement_count": [2, 2, 2],
        }
    )
    assert extra_nonp0_pool(scores)["row_id"].tolist() == [1]
    assert point0_pool(scores)["row_id"].tolist() == [3]


def test_build_submission_from_rows_preserves_non_point_columns():
    base = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [1, 2],
            "pointId": [8, 7],
            "serverGetPoint": [0.1, 0.2],
        }
    )
    selected = pd.DataFrame({"row_id": [1], "candidate_value": [9]})
    out = build_submission_from_rows(base, selected)
    assert out["pointId"].tolist() == [8, 9]
    assert out["actionId"].tolist() == [1, 2]
    assert out["serverGetPoint"].tolist() == [0.1, 0.2]
