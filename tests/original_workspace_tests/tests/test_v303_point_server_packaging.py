import numpy as np
import pandas as pd
import pytest

from analysis_v303_point_server_packaging import (
    SUBMISSION_COLUMNS,
    CandidateSpec,
    build_package_submission,
    point0_rate_delta,
    recommendation_for,
    summarize_package,
)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [10, 11, 12, 13],
            "actionId": [4, 8, 13, 2],
            "pointId": [7, 8, 0, 5],
            "serverGetPoint": [0.2, 0.8, 0.5, 0.1],
        }
    )


def _point_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [10, 11, 12, 13],
            "actionId": [1, 1, 1, 1],
            "pointId": [8, 8, 0, 0],
            "serverGetPoint": [0.9, 0.9, 0.9, 0.9],
        }
    )


def test_build_package_submission_preserves_v300_schema_action_and_server():
    packaged = build_package_submission(_anchor(), _point_source(), expected_rows=4)

    assert list(packaged.columns) == SUBMISSION_COLUMNS
    assert packaged["rally_uid"].tolist() == [10, 11, 12, 13]
    assert packaged["pointId"].tolist() == [8, 8, 0, 0]
    assert packaged["actionId"].tolist() == [4, 8, 13, 2]
    assert np.allclose(packaged["serverGetPoint"], [0.2, 0.8, 0.5, 0.1])


def test_build_package_submission_rejects_misaligned_rows():
    bad_source = _point_source().iloc[[1, 0, 2, 3]].reset_index(drop=True)

    with pytest.raises(ValueError, match="rally_uid"):
        build_package_submission(_anchor(), bad_source, expected_rows=4)


def test_summarize_package_reports_fixed_action_server_metrics():
    anchor = _anchor()
    packaged = build_package_submission(anchor, _point_source(), expected_rows=4)
    spec = CandidateSpec(
        package_name="v303_unit",
        source_candidate="unit_source",
        source_path="unit.csv",
        source_search_path="unit_search.csv",
        source_local_delta=0.0005,
        source_local_delta_column="delta_vs_unit",
    )

    row = summarize_package(spec, anchor, packaged, "v303_point_server_packaging/submission_v303_unit.csv")

    assert row["point_changed_rows_vs_v300"] == 2
    assert row["action_changed_rows_vs_v300"] == 0
    assert row["server_mad_vs_v300"] == pytest.approx(0.0)
    assert row["point0_rate_delta_vs_v300"] == pytest.approx(0.25)
    assert row["recommendation"] == "DO_NOT_UPLOAD"


def test_recommendation_requires_point_source_delta_at_least_0p001():
    assert recommendation_for(None) == "DO_NOT_UPLOAD"
    assert recommendation_for(0.000999) == "DO_NOT_UPLOAD"
    assert recommendation_for(0.001) == "REVIEW_UPLOAD"


def test_point0_rate_delta_is_candidate_minus_anchor_rate():
    assert point0_rate_delta(np.array([8, 8, 0, 7]), np.array([0, 8, 0, 7])) == pytest.approx(0.25)
