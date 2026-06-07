import numpy as np
import pandas as pd
import pytest

from analysis_v309_v306_server_packaging import (
    SUBMISSION_COLUMNS,
    ServerSpec,
    build_server_packaged_submission,
    decision_for_server,
    summarize_server_variant,
    validate_submission_frame,
)


def _v306_anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [101, 102, 103, 104],
            "actionId": [3, 7, 12, 18],
            "pointId": [0, 8, 4, 0],
            "serverGetPoint": [0.20, 0.80, 0.50, 0.30],
        }
    )


def _server_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [101, 102, 103, 104],
            "actionId": [1, 1, 1, 1],
            "pointId": [9, 9, 9, 9],
            "serverGetPoint": [0.21, 0.81, 0.51, 0.31],
        }
    )


def test_validate_submission_frame_requires_expected_schema_and_probabilities():
    validate_submission_frame(_v306_anchor(), expected_rows=4)

    bad = _v306_anchor().copy()
    bad["serverGetPoint"] = [0.2, 1.1, 0.5, 0.3]

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_submission_frame(bad, expected_rows=4)


def test_build_server_packaged_submission_preserves_v306_action_and_point():
    packaged = build_server_packaged_submission(_v306_anchor(), _server_source(), expected_rows=4)

    assert list(packaged.columns) == SUBMISSION_COLUMNS
    assert packaged["rally_uid"].tolist() == [101, 102, 103, 104]
    assert packaged["actionId"].tolist() == [3, 7, 12, 18]
    assert packaged["pointId"].tolist() == [0, 8, 4, 0]
    assert np.allclose(packaged["serverGetPoint"], [0.21, 0.81, 0.51, 0.31])


def test_build_server_packaged_submission_rejects_misaligned_rally_uid():
    bad_source = _server_source().iloc[[1, 0, 2, 3]].reset_index(drop=True)

    with pytest.raises(ValueError, match="rally_uid"):
        build_server_packaged_submission(_v306_anchor(), bad_source, expected_rows=4)


def test_summarize_server_variant_reports_server_metrics_and_preserved_columns():
    anchor = _v306_anchor()
    packaged = build_server_packaged_submission(anchor, _server_source(), expected_rows=4)
    spec = ServerSpec(
        source_key="unit_server",
        source_path="v300_clean_server_blend_recycler/unit.csv",
        source_family="v300",
        source_is_clean=True,
    )

    row = summarize_server_variant(
        spec,
        anchor,
        packaged,
        "v309_v306_server_packaging/submission_unit.csv",
    )

    assert row["server_source"] == "unit_server"
    assert row["server_mad_vs_v306_best_server"] == pytest.approx(0.01)
    assert row["server_corr_vs_v306_best_server"] == pytest.approx(1.0)
    assert row["server_min"] == pytest.approx(0.21)
    assert row["server_max"] == pytest.approx(0.81)
    assert row["row_count"] == 4
    assert row["action_changed_rows_vs_v306_anchor"] == 0
    assert row["point_changed_rows_vs_v306_anchor"] == 0
    assert row["decision"] == "REVIEW_SERVER"


def test_decision_requires_clean_source_and_mad_at_most_0p02():
    assert decision_for_server(0.02, True) == "REVIEW_SERVER"
    assert decision_for_server(0.020001, True) == "DIAGNOSTIC"
    assert decision_for_server(0.001, False) == "DIAGNOSTIC"
