import pandas as pd

from analysis_v420_rare_class_augmented_exact_models import (
    augment_minority_rows,
    build_ranked_changes,
    build_submission,
)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2", "r3"],
            "actionId": [1, 2, 3, 15],
            "pointId": [1, 5, 7, 0],
            "serverGetPoint": [0.10, 0.20, 0.30, 0.40],
        }
    )


def test_augment_minority_rows_increases_only_train_rows_and_keeps_labels_sane():
    x_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y_train = pd.Series([1, 1, 8, 8], name="target")
    groups = pd.Series(["g1", "g2", "g3", "g4"], name="rally_uid")

    aug_x, aug_y, aug_groups, report = augment_minority_rows(
        x_train,
        y_train,
        groups,
        rare_classes={8},
        multiplier=2,
        seed=420,
    )

    assert len(aug_x) > len(x_train)
    assert len(aug_x) == len(aug_y) == len(aug_groups)
    assert aug_x.iloc[: len(x_train)].reset_index(drop=True).equals(x_train)
    assert set(aug_y.unique()) == {1, 8}
    assert set(aug_y.iloc[len(y_train) :].unique()) == {8}
    assert aug_groups.iloc[len(groups) :].astype(str).str.startswith("synthetic_8_").all()
    assert report["original_rows"] == 4
    assert report["synthetic_rows"] == len(aug_x) - len(x_train)


def test_build_submission_preserves_anchor_server_and_schema():
    anchor = _anchor()
    pred = pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2", "r3"],
            "pred_actionId": [4, 6, 8, 16],
            "pred_pointId": [2, 6, 8, 9],
            "action_confidence": [0.5, 0.9, 0.2, 0.8],
            "point_confidence": [0.5, 0.8, 0.9, 0.7],
        }
    )

    packed, report = build_submission(anchor, pred, mode="joint", max_changes=2, expected_rows=len(anchor))

    assert list(packed.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert packed["serverGetPoint"].equals(anchor["serverGetPoint"])
    assert report["selected_row_count"] == 2
    assert report["server_changed"] == 0


def test_point0_addition_gate_and_serve_action_gate_work_on_tiny_fixtures():
    anchor = _anchor()
    pred = pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2", "r3"],
            "pred_actionId": [15, 6, 17, 16],
            "pred_pointId": [0, 0, 8, 9],
            "action_confidence": [0.99, 0.10, 0.98, 0.50],
            "point_confidence": [0.99, 0.98, 0.20, 0.50],
            "joint_confidence": [0.99, 0.98, 0.97, 0.50],
        }
    )

    changes = build_ranked_changes(anchor, pred)
    packed, report = build_submission(anchor, pred, mode="joint", max_changes=4, expected_rows=len(anchor))

    assert changes.loc[0, "action_eligible"] is False
    assert changes.loc[0, "point_eligible"] is False
    assert changes.loc[2, "action_eligible"] is False
    assert changes.loc[3, "action_eligible"] is True
    assert packed.loc[0, "actionId"] == anchor.loc[0, "actionId"]
    assert packed.loc[0, "pointId"] == anchor.loc[0, "pointId"]
    assert packed.loc[2, "actionId"] == anchor.loc[2, "actionId"]
    assert packed.loc[3, "actionId"] == 16
    assert report["blocked_point0_additions"] == 2
    assert report["blocked_serve_15_18_additions"] == 2
    assert report["serve_15_18_additions"] == 0
