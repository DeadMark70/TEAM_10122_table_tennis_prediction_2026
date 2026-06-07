import pandas as pd

from analysis_r200_local_validation_dashboard import churn_tier, decision_label, slice_masks


def test_churn_tier_uses_task_specific_thresholds():
    assert churn_tier("action", 0.04) == "low"
    assert churn_tier("action", 0.08) == "medium"
    assert churn_tier("action", 0.12) == "high"
    assert churn_tier("point", 0.015) == "safe_probe"
    assert churn_tier("point", 0.04) == "normal"
    assert churn_tier("point", 0.07) == "high"


def test_decision_label_blocks_bad_point_distribution():
    assert decision_label(point_churn=0.04, action_churn=0.02, server_mad=0.01, point0_rate=0.3) == "KEEP"
    assert decision_label(point_churn=0.08, action_churn=0.02, server_mad=0.01, point0_rate=0.3) == "REJECT_POINT_CHURN"
    assert decision_label(point_churn=0.04, action_churn=0.02, server_mad=0.01, point0_rate=0.95) == "REJECT_POINT0_COLLAPSE"


def test_slice_masks_builds_expected_named_slices():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 3, 5],
            "audit_phase": ["receive", "fourth_ball", "rally"],
            "audit_lag0_depth": ["short", "long", "long"],
            "audit_lag0_action_family": ["Attack", "Control", "Attack"],
        }
    )
    masks = slice_masks(rows)
    assert masks["all"].sum() == 3
    assert masks["prefix_1"].sum() == 1
    assert masks["phase_rally"].sum() == 1
    assert masks["lag0_long"].sum() == 2
    assert masks["lag0_attack"].sum() == 2
