import pandas as pd

from analysis_v433_weak_class_expert_bank import (
    WEAK_ACTION_CLASSES,
    WEAK_POINT_CLASSES,
    apply_train_only_oversampling,
    train_expert_bank,
    weak_group_definitions,
    write_expert_outputs,
)


def tiny_v433_frames():
    train = pd.DataFrame(
        {
            "rally_uid": [f"r{i // 2}" for i in range(24)],
            "strikeNumber": [i % 6 + 1 for i in range(24)],
            "actionId": [0, 3, 4, 5, 7, 8, 9, 12, 14, 1, 2, 6] * 2,
            "pointId": [1, 3, 4, 7, 8, 9, 2, 5, 6, 0, 1, 7] * 2,
            "spinId": [i % 4 for i in range(24)],
            "strengthId": [i % 3 for i in range(24)],
            "scoreSelf": [i % 11 for i in range(24)],
            "scoreOther": [(i + 3) % 11 for i in range(24)],
            "anchor_actionId": [1, 3, 4, 2, 7, 8, 1, 12, 6, 1, 2, 6] * 2,
            "anchor_pointId": [2, 3, 5, 7, 8, 1, 2, 5, 6, 0, 1, 7] * 2,
            "v432_action_prob_3": [0.1 + (i % 5) * 0.1 for i in range(24)],
            "v432_point_prob_7": [0.2 + (i % 4) * 0.1 for i in range(24)],
        }
    )
    test = train.drop(columns=["actionId", "pointId"]).head(8).copy()
    test["rally_uid"] = [f"t{i}" for i in range(len(test))]
    return train, test


def test_v433_weak_groups_cover_action_and_point_specialists():
    groups = weak_group_definitions()
    assert "action_terminal_zero" in groups
    assert "action_control_8_9_11" in groups
    assert "action_long_rally_transition" in groups
    assert "point_short_side_1_3" in groups
    assert "point_long_side_7_8_9" in groups
    assert WEAK_ACTION_CLASSES == {0, 3, 4, 5, 7, 8, 9, 12, 14}
    assert WEAK_POINT_CLASSES == {1, 3, 4, 7, 8, 9}


def test_v433_train_only_oversampling_does_not_alter_validation_or_test_size():
    train, test = tiny_v433_frames()
    validation = train.iloc[:5].copy()
    sampled_train, unchanged_validation, unchanged_test, report = apply_train_only_oversampling(
        train,
        validation,
        test,
        target_col="actionId",
        positive_labels={0, 3},
        multiplier=3,
    )
    assert len(sampled_train) > len(train)
    assert len(unchanged_validation) == len(validation)
    assert len(unchanged_test) == len(test)
    assert report["train_only"] is True


def test_v433_expert_scores_are_bounded_and_no_submission_export(tmp_path):
    train, test = tiny_v433_frames()
    result = train_expert_bank(
        train,
        test,
        groups=weak_group_definitions(),
        n_splits=3,
        oversample_multiplier=2,
        random_state=433,
    )
    assert set(result["action_groups"]) >= {"action_terminal_zero", "action_attack_3_4_5_7"}
    assert set(result["point_groups"]) >= {"point_short_side_1_3", "point_long_side_7_8_9"}

    score_columns = [c for c in result["expert_test_scores"].columns if c.endswith("_score")]
    assert score_columns
    assert result["expert_test_scores"][score_columns].min().min() >= 0.0
    assert result["expert_test_scores"][score_columns].max().max() <= 1.0
    assert len(result["expert_test_scores"]) == len(test)

    report_paths = write_expert_outputs(result, tmp_path)
    assert "expert_scores_test" in report_paths
    assert not any(path.name.lower().startswith("submission") for path in tmp_path.iterdir())
