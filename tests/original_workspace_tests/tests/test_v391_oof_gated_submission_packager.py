from pathlib import Path

import pandas as pd

import analysis_v391_oof_gated_submission_packager as v391


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": ["r1", "r2", "r3", "r4"],
            "actionId": [1, 2, 3, 4],
            "pointId": [1, 2, 3, 4],
            "serverGetPoint": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_point_selection_respects_pass_gate_and_blocks_point0_additions() -> None:
    ranked = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2", "r3", "r4"],
            "candidate_point": [5, 0, 6, 7],
            "pass_gate": [True, True, False, True],
            "is_point0_addition": [False, True, False, False],
            "proxy_score": [0.90, 1.00, 0.99, 0.70],
        }
    )
    augmented = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2", "r3", "r4"],
            "candidate_point": [5, 0, 6, 7],
            "augmented_score": [0.92, 0.99, 0.98, 0.75],
        }
    )

    selected = v391.select_point_rows(ranked, augmented, budget=3)

    assert selected["rally_uid"].tolist() == ["r1", "r4"]
    assert selected["candidate_point"].tolist() == [5, 7]
    assert not selected["is_point0_addition"].any()
    assert selected["pass_gate"].all()


def test_point_candidate_preserves_action_server_and_schema() -> None:
    anchor = _anchor()
    selected = pd.DataFrame({"rally_uid": ["r1", "r4"], "candidate_point": [5, 7]})

    candidate = v391.package_candidate(anchor, selected, pd.DataFrame())

    assert list(candidate.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert candidate["actionId"].tolist() == anchor["actionId"].tolist()
    assert candidate["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
    assert candidate["pointId"].tolist() == [5, 2, 3, 7]


def test_mixed_action_selection_blocks_serve_15_18_additions() -> None:
    ranked = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "candidate_action": [15, 8],
            "pass_gate": [True, True],
            "is_serve_15_18_addition": [True, False],
            "proxy_score": [0.99, 0.84],
        }
    )
    augmented = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "candidate_action": [15, 8],
            "augmented_score": [0.99, 0.88],
        }
    )

    selected = v391.select_action_rows(ranked, augmented, budget=5)

    assert selected["rally_uid"].tolist() == ["r2"]
    assert selected["candidate_action"].tolist() == [8]
    assert not selected["is_serve_15_18_addition"].any()


def test_pipeline_reports_missing_inputs_and_generates_zero_submissions(tmp_path: Path) -> None:
    anchor_path = tmp_path / "anchor.csv"
    _anchor().to_csv(anchor_path, index=False)

    report = v391.run_pipeline(
        outdir=tmp_path / "out",
        anchor_path=anchor_path,
        ranked_point_path=tmp_path / "missing_point_ranked.csv",
        ranked_action_path=tmp_path / "missing_action_ranked.csv",
        augmented_point_path=tmp_path / "missing_point_augmented.csv",
        augmented_action_path=tmp_path / "missing_action_augmented.csv",
    )

    ranked_candidates = pd.read_csv(tmp_path / "out" / "ranked_candidates.csv")
    assert report["generated_submission_count"] == 0
    assert len(report["missing_inputs"]) == 4
    assert ranked_candidates.empty
    assert not list((tmp_path / "out").glob("submission_*.csv"))
