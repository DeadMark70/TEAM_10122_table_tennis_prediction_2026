import numpy as np
import pandas as pd
import pytest

from analysis_v314_clean_server_value_research import (
    ServerVariant,
    apply_temperature,
    build_packaged_submission,
    decision_for_variant,
    rank_normalize_to_anchor,
    shrink_to_anchor,
    summarize_combination,
)


def _point_anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [11, 12, 13, 14],
            "actionId": [3, 7, 12, 18],
            "pointId": [0, 8, 4, 0],
            "serverGetPoint": [0.20, 0.80, 0.50, 0.30],
        }
    )


def test_build_packaged_submission_preserves_action_and_point_anchor_columns():
    anchor = _point_anchor()
    variant = ServerVariant(
        key="unit_server",
        family="unit",
        source_path="v300_clean_server_blend_recycler/unit.csv",
        server=np.array([0.21, 0.81, 0.51, 0.31]),
        source_is_clean=True,
        server_auc_oof=0.61,
        risk_hint="safe",
    )

    packaged = build_packaged_submission(anchor, variant, expected_rows=4)

    assert packaged["rally_uid"].tolist() == anchor["rally_uid"].tolist()
    assert packaged["actionId"].tolist() == anchor["actionId"].tolist()
    assert packaged["pointId"].tolist() == anchor["pointId"].tolist()
    assert packaged["serverGetPoint"].tolist() == pytest.approx([0.21, 0.81, 0.51, 0.31])


def test_build_packaged_submission_rejects_length_mismatch():
    variant = ServerVariant(
        key="bad_length",
        family="unit",
        source_path="unit.csv",
        server=np.array([0.1, 0.2, 0.3]),
        source_is_clean=True,
        server_auc_oof=float("nan"),
        risk_hint="diagnostic",
    )

    with pytest.raises(ValueError, match="length"):
        build_packaged_submission(_point_anchor(), variant, expected_rows=4)


def test_calibration_math_for_shrink_temperature_and_rank_normalization():
    base = np.array([0.2, 0.8, 0.5])
    anchor = np.array([0.4, 0.4, 0.1])

    assert shrink_to_anchor(base, anchor, strength=0.25).tolist() == pytest.approx([0.25, 0.7, 0.4])

    sharpened = apply_temperature(np.array([0.25, 0.5, 0.75]), temperature=0.9)
    softened = apply_temperature(np.array([0.25, 0.5, 0.75]), temperature=1.1)
    assert sharpened[0] < 0.25
    assert sharpened[2] > 0.75
    assert softened[0] > 0.25
    assert softened[2] < 0.75

    ranked = rank_normalize_to_anchor(np.array([10.0, 30.0, 20.0]), np.array([0.2, 0.8, 0.5]))
    assert ranked.tolist() == pytest.approx([0.2, 0.8, 0.5])


def test_summary_and_decision_use_clean_source_mad_threshold_and_distribution():
    anchor = _point_anchor()
    variant = ServerVariant(
        key="unit_server",
        family="unit",
        source_path="v300_clean_server_blend_recycler/unit.csv",
        server=np.array([0.21, 0.81, 0.51, 0.31]),
        source_is_clean=True,
        server_auc_oof=0.61,
        risk_hint="safe",
    )
    packaged = build_packaged_submission(anchor, variant, expected_rows=4)

    row = summarize_combination(
        point_key="v306_p0_cap0p01",
        point_path="v306_point0_addition_probe/submission.csv",
        v306_anchor=anchor,
        point_anchor=anchor,
        variant=variant,
        packaged=packaged,
        output_path="v314_clean_server_value_research/submission.csv",
    )

    assert row["server_mad_vs_current_v306"] == pytest.approx(0.01)
    assert row["server_corr_vs_current_v306"] == pytest.approx(1.0)
    assert row["action_changed_rows_vs_point_anchor"] == 0
    assert row["point_changed_rows_vs_point_anchor"] == 0
    assert row["server_auc_oof"] == pytest.approx(0.61)
    assert row["decision"] == "REVIEW_SERVER"

    assert decision_for_variant(source_is_clean=True, mad=0.01, server_min=0.01, server_max=0.99) == "REVIEW_SERVER"
    assert decision_for_variant(source_is_clean=True, mad=0.010001, server_min=0.01, server_max=0.99) == "DIAGNOSTIC"
    assert decision_for_variant(source_is_clean=False, mad=0.001, server_min=0.01, server_max=0.99) == "DIAGNOSTIC"
    assert decision_for_variant(source_is_clean=True, mad=0.001, server_min=0.5, server_max=0.5) == "DIAGNOSTIC"
