from pathlib import Path

import pandas as pd
import pytest

from analysis_v472_old_overlap_ceiling import (
    build_old_overlap_submission,
    make_old_server_map,
)


def test_make_old_server_map_requires_consistent_duplicate_labels():
    old = pd.DataFrame(
        {
            "rally_uid": [1, 1, 2, 2],
            "serverGetPoint": [1, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="conflicting serverGetPoint"):
        make_old_server_map(old)


def test_old_overlap_submission_only_replaces_server_for_overlap(tmp_path: Path):
    anchor = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3],
            "actionId": [4, 5, 6],
            "pointId": [7, 8, 9],
            "serverGetPoint": [0.2, 0.4, 0.6],
        }
    )
    old = pd.DataFrame(
        {
            "rally_uid": [1, 1, 3],
            "serverGetPoint": [1, 1, 0],
        }
    )
    out_path = tmp_path / "submission.csv"

    report = build_old_overlap_submission(anchor, old, out_path)
    result = pd.read_csv(out_path)

    assert list(result.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert result["actionId"].tolist() == [4, 5, 6]
    assert result["pointId"].tolist() == [7, 8, 9]
    assert result["serverGetPoint"].tolist() == [1.0, 0.4, 0.0]
    assert report["overlap_rows"] == 2
    assert report["server_changed_rows"] == 2


def test_old_overlap_submission_rejects_invalid_schema(tmp_path: Path):
    anchor = pd.DataFrame(
        {
            "rally_uid": [1],
            "actionId": [4],
            "serverGetPoint": [0.2],
        }
    )
    old = pd.DataFrame({"rally_uid": [1], "serverGetPoint": [1]})

    with pytest.raises(ValueError, match="missing required columns"):
        build_old_overlap_submission(anchor, old, tmp_path / "submission.csv")
