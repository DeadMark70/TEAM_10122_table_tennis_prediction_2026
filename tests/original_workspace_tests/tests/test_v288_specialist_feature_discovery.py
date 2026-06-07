from analysis_v288_specialist_feature_discovery import (
    SPECIALIST_GROUPS,
    feature_family_audit,
    feature_family_columns,
    group_for_action,
    select_group_candidates,
)
import numpy as np
import pandas as pd

from analysis_v288_specialist_feature_discovery import build_basic_feature_frame, family_average_precision


def test_specialist_groups_are_disjoint_for_first_pass():
    seen = []
    for actions in SPECIALIST_GROUPS.values():
        seen.extend(actions)
    assert len(seen) == len(set(seen))
    assert SPECIALIST_GROUPS["fast_attack_57"] == [5, 7]
    assert SPECIALIST_GROUPS["terminal_03"] == [0, 3]


def test_group_for_action_maps_known_weak_actions():
    assert group_for_action(5) == "fast_attack_57"
    assert group_for_action(7) == "fast_attack_57"
    assert group_for_action(0) == "terminal_03"
    assert group_for_action(3) == "terminal_03"
    assert group_for_action(1) == ""


def test_feature_family_columns_have_no_duplicate_column_names():
    families = feature_family_columns()
    all_cols = []
    for cols in families.values():
        all_cols.extend(cols)
    assert len(all_cols) == len(set(all_cols))
    assert "phase_bin" in families["phase_prefix"]
    assert "lag0_actionId" in families["incoming_ball"]


def test_build_basic_feature_frame_adds_phase_and_pair_features():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 2, 4],
            "lag0_actionId": [1, 10, 13],
            "lag0_pointId": [7, 2, 0],
            "lag0_spinId": [1, 2, 0],
            "lag0_strengthId": [3, 1, 0],
            "lag0_positionId": [2, 3, 0],
            "scoreSelf": [1, 9, 10],
            "scoreOther": [0, 9, 11],
            "scoreTotal": [1, 18, 21],
            "serverScoreDiff": [1, 0, -1],
        }
    )
    out = build_basic_feature_frame(rows)
    assert out["phase_bin"].tolist() == ["receive", "third", "rally"]
    assert out["is_receive"].tolist() == [1, 0, 0]
    assert out["lag0_action_point_pair"].tolist() == ["1_7", "10_2", "13_0"]
    assert out["is_deuce_like"].tolist() == [0, 1, 1]


def test_family_average_precision_ranks_useful_score_higher_than_constant_score():
    y = np.array([0, 1, 1, 0])
    useful = np.array([0.1, 0.9, 0.8, 0.2])
    constant = np.array([0.5, 0.5, 0.5, 0.5])
    assert family_average_precision(y, useful) > family_average_precision(y, constant)


def test_feature_family_audit_reports_best_available_column_per_group():
    frame = pd.DataFrame(
        {
            "phase_bin": ["receive", "third", "third", "rally"],
            "lag0_actionId": [1, 5, 5, 10],
            "scoreTotal": [1, 18, 19, 2],
        }
    )
    y = np.array([1, 5, 7, 3])
    out = feature_family_audit(frame, y)
    fast_phase = out[(out["group"] == "fast_attack_57") & (out["family"] == "phase_prefix")].iloc[0]
    assert fast_phase["available_cols"] == 1
    assert fast_phase["best_col"] == "phase_bin"
    assert float(fast_phase["best_ap"]) > 0.0


def test_select_group_candidates_keeps_one_candidate_per_row():
    frame = pd.DataFrame(
        {
            "row_id": [0, 0, 1],
            "candidate_action": [5, 7, 3],
            "group_score": [0.7, 0.9, 0.8],
            "support_count": [10, 5, 20],
        }
    )
    out = select_group_candidates(frame, min_score=0.6, min_support=1)
    assert out.sort_values("row_id")["candidate_action"].tolist() == [7, 3]
