import numpy as np
import pandas as pd

from analysis_v291_weak_class_training_upgrade import (
    HARD_NEGATIVES,
    SPECIALIST_GROUPS,
    build_complete_feature_frame,
    choose_shortcontrol_action_or_keep,
    feature_family_columns,
    group_for_action,
    hard_negative_mask,
    normalize_score01,
    train_model_bank_scores,
)


def test_specialist_groups_and_hard_negatives_are_defined():
    assert SPECIALIST_GROUPS["fast_attack_57"] == [5, 7]
    assert SPECIALIST_GROUPS["terminal_03"] == [0, 3]
    assert HARD_NEGATIVES["fast_attack_57"] == [1, 2, 4, 6, 10, 13]
    assert 10 in HARD_NEGATIVES["short_control_411"]


def test_group_for_action_maps_specialist_actions():
    assert group_for_action(5) == "fast_attack_57"
    assert group_for_action(0) == "terminal_03"
    assert group_for_action(11) == "short_control_411"
    assert group_for_action(1) == ""


def test_feature_family_columns_include_missing_v288_families():
    families = feature_family_columns()
    assert "teacher_specialist" in families
    assert "support_backoff" in families
    assert "style_response" in families
    assert "specialist_p_5" in families["teacher_specialist"]
    assert "support_family_depth_5" in families["support_backoff"]
    assert "style_cond_family_match" in families["style_response"]


def test_build_complete_feature_frame_includes_all_families():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 2],
            "lag0_actionId": [1, 10],
            "lag0_pointId": [7, 2],
            "lag0_spinId": [1, 2],
            "lag0_strengthId": [3, 1],
            "scoreSelf": [1, 9],
            "scoreOther": [0, 9],
            "scoreTotal": [1, 18],
            "serverScoreDiff": [1, 0],
        }
    )
    v286 = pd.DataFrame(
        {
            "specialist_p_0": [0.1, 0.2],
            "specialist_p_3": [0.2, 0.1],
            "specialist_p_5": [0.3, 0.4],
            "specialist_p_7": [0.4, 0.3],
            "specialist_p_8": [0.0, 0.1],
            "specialist_p_9": [0.0, 0.1],
            "specialist_p_14": [0.0, 0.1],
            "support_0": [10, 20],
            "support_3": [10, 20],
            "support_5": [10, 20],
            "support_7": [10, 20],
            "support_8": [10, 20],
            "support_9": [10, 20],
            "support_14": [10, 20],
        }
    )
    out = build_complete_feature_frame(rows, v286)
    assert "specialist_p_5" in out
    assert "support_family_depth_5" in out
    assert "style_cond_family_match" in out
    assert len(out) == 2


def test_hard_negative_mask_keeps_positive_group_and_hard_negatives():
    y = np.array([5, 7, 1, 10, 14])
    mask = hard_negative_mask(y, "fast_attack_57")
    assert mask.tolist() == [True, True, True, True, False]


def test_normalize_score01_clips_and_handles_nan():
    score = np.array([-1.0, 0.5, 2.0, np.nan])
    out = normalize_score01(score)
    assert out.tolist() == [0.0, 0.5, 1.0, 0.0]


def test_choose_shortcontrol_action_or_keep_returns_keep_when_anchor_protected():
    row = pd.Series(
        {
            "anchor_action": 10,
            "shortcontrol_context_score": 0.9,
            "lag0_point_depth": "short",
            "lag0_spin": 2,
        }
    )
    assert choose_shortcontrol_action_or_keep(row) == 10


def test_choose_shortcontrol_action_or_keep_prefers_11_for_receive_short():
    row = pd.Series(
        {
            "anchor_action": 1,
            "shortcontrol_context_score": 0.9,
            "lag0_point_depth": "short",
            "lag0_spin": 2,
        }
    )
    assert choose_shortcontrol_action_or_keep(row) == 11


def test_train_model_bank_scores_returns_nonconstant_test_scores():
    y = np.array([5, 7, 1, 10, 0, 3, 8, 9, 4, 11, 12, 14, 13, 2, 6] * 2)
    n = len(y)
    rows = pd.DataFrame({"fold": np.arange(n) % 3})
    train_frame = pd.DataFrame(
        {
            "prefix_len": np.arange(n) % 7 + 1,
            "anchor_action": (np.arange(n) * 3) % 15,
            "lag0_actionId": (np.arange(n) * 5) % 15,
            "lag0_pointId": np.arange(n) % 10,
            "scoreTotal": np.arange(n) % 21,
            "specialist_p_5": np.linspace(0.05, 0.95, n),
            "support_family_depth_5": np.arange(n) + 1,
        }
    )
    test_frame = pd.DataFrame(
        {
            "prefix_len": np.arange(8) % 7 + 1,
            "anchor_action": (np.arange(8) * 2) % 15,
            "lag0_actionId": (np.arange(8) * 4) % 15,
            "lag0_pointId": np.arange(8) % 10,
            "scoreTotal": np.arange(8) % 21,
            "specialist_p_5": np.linspace(0.1, 0.9, 8),
            "support_family_depth_5": np.arange(8) + 2,
        }
    )

    _oof_scores, test_scores, comparison = train_model_bank_scores(train_frame, rows, y, test_frame=test_frame)

    score_col = "fast_attack_57__extratrees_balanced"
    assert score_col in test_scores
    assert len(test_scores) == len(test_frame)
    assert test_scores[score_col].between(0.0, 1.0).all()
    assert test_scores[score_col].nunique() > 1
    assert comparison.loc[
        (comparison["group"] == "fast_attack_57") & (comparison["model"] == "extratrees_balanced"),
        "test_score_mean",
    ].iloc[0] > 0.0
