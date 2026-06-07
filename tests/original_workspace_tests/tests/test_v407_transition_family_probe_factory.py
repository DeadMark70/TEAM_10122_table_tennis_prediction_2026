from pathlib import Path

import pandas as pd

from analysis_v407_transition_family_probe_factory import (
    SUBMISSION_COLUMNS,
    build_transition_family_rows,
    classify_transition,
    package_candidate,
    run_pipeline,
)


def _anchor(points: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [f"r{i}" for i in range(len(points))],
            "actionId": [1 + (i % 3) for i in range(len(points))],
            "pointId": points,
            "serverGetPoint": [0.25 + (i * 0.01) for i in range(len(points))],
        }
    )


def _selected(rows: list[tuple[int, int, int]], ranks: list[int] | None = None) -> pd.DataFrame:
    ranks = ranks or list(range(1, len(rows) + 1))
    return pd.DataFrame(
        {
            "rank": ranks,
            "row_id": [row_id for row_id, _, _ in rows],
            "rally_uid": [f"r{row_id}" for row_id, _, _ in rows],
            "anchor_point": [old for _, old, _ in rows],
            "new_point": [new for _, _, new in rows],
            "agreement_count": [10] * len(rows),
        }
    )


def _write_inputs(tmp_path: Path, rows_by_tier: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths = {}
    for tier in ("top9", "top15", "top24"):
        path = tmp_path / f"selected_rows_v400_public_agree_{tier}.csv"
        rows_by_tier.get(tier, pd.DataFrame(columns=["rank", "row_id", "rally_uid", "anchor_point", "new_point"])).to_csv(
            path,
            index=False,
        )
        paths[tier] = path
    return paths


def test_groups_transition_strings_correctly():
    assert classify_transition(7, 8) == "longside_centering"
    assert classify_transition(8, 9) == "longside_corner"
    assert classify_transition(9, 6) == "long_to_half"
    assert classify_transition(2, 5) == "short_to_middle"
    assert classify_transition(4, 8) is None


def test_empty_groups_are_skipped(tmp_path):
    anchor_path = tmp_path / "anchor.csv"
    _anchor([7, 1, 4]).to_csv(anchor_path, index=False)
    selected_paths = _write_inputs(
        tmp_path,
        {
            "top9": _selected([(0, 7, 8)]),
            "top15": _selected([(0, 7, 8)]),
            "top24": _selected([(0, 7, 8)]),
        },
    )

    report = run_pipeline(outdir=tmp_path / "out", expected_rows=3, anchor_path=anchor_path, selected_paths=selected_paths)

    assert [item["candidate"] for item in report["generated_submissions"]] == [
        "v407_longside_centering",
        "v407_mixed_high_agreement",
    ]
    assert report["skipped_families"]["longside_corner"] == "zero_selected_rows"
    assert report["skipped_families"]["long_to_half"] == "zero_selected_rows"
    assert report["skipped_families"]["short_to_middle"] == "zero_selected_rows"


def test_output_preserves_action_and_server():
    anchor = _anchor([7, 8, 9])
    selected = _selected([(0, 7, 8), (1, 8, 9)])
    selected["transition"] = ["7->8", "8->9"]
    selected["transition_group"] = ["longside_centering", "longside_corner"]
    selected["in_top9"] = True
    selected["in_top15"] = True
    selected["in_top24"] = True

    out = package_candidate(anchor, selected)

    assert out["actionId"].astype(int).tolist() == anchor["actionId"].astype(int).tolist()
    assert out["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
    assert out["pointId"].astype(int).tolist() == [8, 9, 9]


def test_point0_additions_are_blocked_from_family_rows():
    rows = _selected([(0, 7, 0), (1, 7, 8), (2, 2, 5)])
    rows["transition"] = ["7->0", "7->8", "2->5"]
    rows["transition_group"] = [None, "longside_centering", "short_to_middle"]
    rows["in_top9"] = True
    rows["in_top15"] = True
    rows["in_top24"] = True

    families = build_transition_family_rows(rows)

    assert families["longside_centering"]["row_id"].tolist() == [1]
    assert families["short_to_middle"]["row_id"].tolist() == [2]
    assert 0 not in set(families["mixed_high_agreement"]["row_id"].astype(int))


def test_ranked_metadata_includes_transition_group_and_point_churn(tmp_path):
    anchor_path = tmp_path / "anchor.csv"
    _anchor([7, 8, 9, 2]).to_csv(anchor_path, index=False)
    selected_paths = _write_inputs(
        tmp_path,
        {
            "top9": _selected([(0, 7, 8), (1, 8, 9), (2, 9, 6), (3, 2, 5)]),
            "top15": _selected([(0, 7, 8), (1, 8, 9), (2, 9, 6), (3, 2, 5)]),
            "top24": _selected([(0, 7, 8), (1, 8, 9), (2, 9, 6), (3, 2, 5)]),
        },
    )

    report = run_pipeline(outdir=tmp_path / "out", expected_rows=4, anchor_path=anchor_path, selected_paths=selected_paths)
    ranked = pd.read_csv(report["ranked_candidates"])

    assert {"transition_group", "point_churn", "selected_row_count"}.issubset(ranked.columns)
    assert set(ranked["transition_group"]) == {
        "longside_centering",
        "longside_corner",
        "long_to_half",
        "short_to_middle",
        "mixed_high_agreement",
    }
    assert ranked.set_index("transition_group").loc["mixed_high_agreement", "point_churn"] == 4
    for item in report["generated_submissions"]:
        frame = pd.read_csv(item["path"])
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert item["action_churn"] == 0
        assert item["server_changed"] == 0
        assert item["point0_additions"] == 0


def test_real_anchor_pipeline_writes_nonempty_candidates(tmp_path):
    report = run_pipeline(outdir=tmp_path)

    assert report["anchor_rows"] == 1845
    assert report["generated_submission_count"] >= 4
    assert report["inputs"]["loaded_selected_inputs"] == ["top9", "top15", "top24"]
    for item in report["generated_submissions"]:
        assert item["selected_row_count"] > 0
        assert item["point_churn"] == item["selected_row_count"]
        assert item["action_churn"] == 0
        assert item["server_changed"] == 0
        assert item["point0_additions"] == 0
