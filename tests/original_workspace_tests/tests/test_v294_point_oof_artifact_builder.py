import numpy as np
import pandas as pd
import pytest

from analysis_v294_point_oof_artifact_builder import (
    OOF_BASE_COLUMNS,
    POINT_CLASSES,
    TEST_BASE_COLUMNS,
    build_feature_frame,
    point_depth,
    point_side,
    normalize_rows_safe,
    train_extratrees_point_base,
    validate_artifact_schema,
)


def test_normalize_rows_safe_repairs_bad_rows():
    matrix = np.array(
        [
            [1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [np.nan, np.inf, -5.0],
            [-1.0, 5.0, 0.0],
        ]
    )

    out = normalize_rows_safe(matrix)

    assert out.shape == matrix.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
    assert np.allclose(out[1], [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(out[2], [1 / 3, 1 / 3, 1 / 3])
    assert np.all(out >= 0.0)


def test_point_depth_side_mapping():
    assert point_depth(0) == 0
    assert [point_depth(i) for i in [1, 2, 3]] == [1, 1, 1]
    assert [point_depth(i) for i in [4, 5, 6]] == [2, 2, 2]
    assert [point_depth(i) for i in [7, 8, 9]] == [3, 3, 3]
    assert [point_side(i) for i in [1, 4, 7]] == [1, 1, 1]
    assert [point_side(i) for i in [2, 5, 8]] == [2, 2, 2]
    assert [point_side(i) for i in [3, 6, 9]] == [3, 3, 3]
    with pytest.raises(ValueError):
        point_depth(10)
    with pytest.raises(ValueError):
        point_side(-1)


def test_build_feature_frame_has_required_columns():
    train_rows = pd.DataFrame(
        {
            "rally_uid": [101, 102],
            "match": [1, 2],
            "prefix_len": [1, 4],
            "lag0_actionId": [8, 12],
            "lag0_pointId": [7, 0],
            "lag0_spinId": [2, 1],
            "lag0_strengthId": [3, 0],
            "lag0_positionId": [4, 5],
            "serverScore": [5, 8],
            "receiverScore": [4, 9],
            "next_pointId": [8, 0],
        }
    )
    test_rows = train_rows.drop(columns=["next_pointId"]).copy()
    test_rows["rally_uid"] = [201, 202]
    anchor_action = np.array([8, 12])
    anchor_point = np.array([9, 0])

    train_feat, test_feat, features = build_feature_frame(
        train_rows,
        test_rows,
        train_anchor_action=train_rows["lag0_actionId"].to_numpy(),
        train_base_point=train_rows["lag0_pointId"].to_numpy(),
        test_anchor_action=anchor_action,
        test_base_point=anchor_point,
    )

    required = {
        "prefix_len",
        "phase",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_point_depth",
        "lag0_point_side",
        "lag0_action_family",
        "score_total",
        "score_diff",
        "anchor_action",
        "anchor_action_family",
        "base_point",
        "base_point_depth",
        "base_point_side",
    }
    assert required.issubset(set(train_feat.columns))
    assert required.issubset(set(test_feat.columns))
    assert required.issubset(set(features))
    assert train_feat.loc[0, "lag0_point_depth"] == 3
    assert train_feat.loc[0, "lag0_point_side"] == 1
    assert test_feat.loc[0, "base_point"] == 9


def test_artifact_schema_validator_rejects_bad_columns():
    good_oof = pd.DataFrame(columns=OOF_BASE_COLUMNS)
    good_test = pd.DataFrame(columns=TEST_BASE_COLUMNS)
    validate_artifact_schema(good_oof, "oof_base")
    validate_artifact_schema(good_test, "test_base")

    bad = good_oof.drop(columns=["anchor_action"])
    with pytest.raises(ValueError, match="oof_base"):
        validate_artifact_schema(bad, "oof_base")


def test_extra_trees_oof_shapes_on_synthetic_data():
    train_rows = pd.DataFrame(
        {
            "rally_uid": np.arange(60),
            "match": np.repeat(np.arange(6), 10),
            "prefix_len": np.tile([1, 2, 3, 4, 5, 6], 10),
            "lag0_actionId": np.arange(60) % 19,
            "lag0_pointId": np.arange(60) % 10,
            "lag0_spinId": np.arange(60) % 4,
            "lag0_strengthId": np.arange(60) % 5,
            "lag0_positionId": np.arange(60) % 6,
            "serverScore": np.arange(60) % 11,
            "receiverScore": (np.arange(60) * 2) % 11,
            "fold": np.arange(60) % 5,
        }
    )
    test_rows = train_rows.head(7).drop(columns=["fold"]).copy()
    y = np.arange(60) % 10
    train_feat, test_feat, features = build_feature_frame(
        train_rows,
        test_rows,
        train_anchor_action=train_rows["lag0_actionId"].to_numpy(),
        train_base_point=train_rows["lag0_pointId"].to_numpy(),
        test_anchor_action=test_rows["lag0_actionId"].to_numpy(),
        test_base_point=test_rows["lag0_pointId"].to_numpy(),
    )

    oof_proba, test_proba, fold_report = train_extratrees_point_base(
        train_feat,
        test_feat,
        y,
        features,
        n_estimators=12,
        min_samples_leaf=1,
    )

    assert oof_proba.shape == (60, len(POINT_CLASSES))
    assert test_proba.shape == (7, len(POINT_CLASSES))
    assert np.allclose(oof_proba.sum(axis=1), 1.0)
    assert np.allclose(test_proba.sum(axis=1), 1.0)
    assert np.isfinite(oof_proba).all()
    assert np.isfinite(test_proba).all()
    assert len(fold_report) == 5
