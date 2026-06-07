import pandas as pd

from analysis_v284_public_response_ranker import (
    classify_family,
    family_public_delta,
    risk_penalty,
    score_candidate,
)


def test_classify_family_uses_known_patterns():
    assert classify_family("submission_v261_cap0p01__v173action_r121server.csv") == "v261_cap1_point"
    assert classify_family("submission_v273_action_style_cap0p010__pv261cap1__sr121.csv") == "v273_action_style"
    assert classify_family("submission_v263a_action_cap0p010__pv261cap1__sr121.csv") == "v263_questionnaire_action"
    assert classify_family("submission_v267_action_logadj_tau0p20_cap0p010__pv261cap1__sr121.csv") == "v267_longtail_action"
    assert classify_family("submission_v264_clean_v261cap1_r121.csv") == "clean_anchor_copy"
    assert classify_family("submission_v270_anchor_copy__clean.csv") == "clean_anchor_copy"
    assert classify_family("submission_v277_nonterminal_cap0p010__v173action_r121server.csv") == "v277_point_refine"
    assert classify_family("submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv") == "v191_v166_action"
    assert classify_family("submission_v282_support_both_churn0p010__sr121.csv") == "v282_joint_support"


def test_family_public_delta_uses_best_known_result_by_family():
    public = pd.DataFrame(
        [
            {"candidate": "submission_v261_cap0p01__v173action_r121server.csv", "public_pl": 0.3576720},
            {"candidate": "submission_v277_nonterminal_cap0p010__v173action_r121server.csv", "public_pl": 0.3574825},
            {"candidate": "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv", "public_pl": 0.3509562},
        ]
    )
    deltas = family_public_delta(public, anchor_public=0.3576720)
    assert deltas["v261_cap1_point"] == 0.0
    assert round(deltas["v277_point_refine"], 7) == round(0.3574825 - 0.3576720, 7)
    assert round(deltas["v191_v166_action"], 7) == round(0.3509562 - 0.3576720, 7)


def test_risk_penalty_penalizes_public_negative_and_churn():
    low = risk_penalty(
        action_churn=0.0,
        point_churn=0.0,
        server_mad=0.0,
        family_delta=0.0,
        decision="KEEP",
    )
    high = risk_penalty(
        action_churn=0.08,
        point_churn=0.02,
        server_mad=0.0,
        family_delta=-0.006,
        decision="KEEP",
    )
    assert high > low


def test_score_candidate_prefers_public_positive_low_risk_candidate():
    candidate = pd.Series(
        {
            "candidate": "submission_v261_cap0p01__v173action_r121server.csv",
            "action_churn_vs_anchor": "0.0",
            "point_churn_vs_anchor": "0.0",
            "server_mad_vs_anchor": "0.0",
            "decision": "KEEP",
        }
    )
    risky = pd.Series(
        {
            "candidate": "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv",
            "action_churn_vs_anchor": "0.08",
            "point_churn_vs_anchor": "0.0",
            "server_mad_vs_anchor": "0.0",
            "decision": "KEEP",
        }
    )
    deltas = {"v261_cap1_point": 0.0, "v191_v166_action": -0.006437}
    assert score_candidate(candidate, deltas) > score_candidate(risky, deltas)
