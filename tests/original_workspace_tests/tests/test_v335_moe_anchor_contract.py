from pathlib import Path

import pandas as pd
import pytest

from analysis_v335_moe_anchor_contract import (
    action_distribution_report,
    drop_leaky_columns,
    safe_output_path,
    validate_submission_schema,
)


def test_drop_leaky_columns_removes_targets():
    frame = pd.DataFrame(
        {
            "safe_phase": [1],
            "next_actionId": [2],
            "y_point": [3],
            "true_action": [4],
            "actionId": [5],
        }
    )
    clean = drop_leaky_columns(frame)
    assert list(clean.columns) == ["safe_phase"]


def test_safe_output_path_rejects_escape():
    outdir = Path("v335_moe_anchor_contract")
    with pytest.raises(ValueError):
        safe_output_path(outdir, "../bad.csv")


def test_submission_schema_exact():
    frame = pd.DataFrame(
        {
            "rally_uid": ["a"],
            "actionId": [1],
            "pointId": [8],
            "serverGetPoint": [0.5],
        }
    )
    validate_submission_schema(frame, expected_rows=1)


def test_serve_explosion_detector_flags_growth():
    base = pd.Series([1, 1, 10, 13, 0])
    cand = pd.Series([15, 16, 17, 18, 1])
    report = action_distribution_report(base, cand)
    assert report["serve_15_18_delta"] == 4
    assert report["serve_15_18_explosion"] is True
