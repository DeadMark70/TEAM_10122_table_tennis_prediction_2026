import json

import pandas as pd

from analysis_v386_synthetic_contrastive_scorer import (
    output_filenames,
    run_pipeline,
    score_action_point_compatibility,
    score_candidate_frame,
)


def test_compatible_pair_scores_above_incompatible_pair():
    good = score_action_point_compatibility(action_id=3, point_id=9, phase="rally")
    bad = score_action_point_compatibility(action_id=11, point_id=9, phase="rally")
    assert good > bad


def test_candidate_frame_blocks_point0_addition_without_terminal_support():
    frame = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "base_point": [8, 8],
            "candidate_point": [0, 9],
            "base_action": [10, 3],
            "candidate_action": [10, 3],
            "phase": ["rally", "rally"],
            "support_count": [50, 50],
            "source_family_count": [7, 7],
        }
    )
    scored = score_candidate_frame(frame)
    assert scored.loc[scored["rally_uid"] == 1, "synthetic_allowed"].item() is False
    assert scored.loc[scored["rally_uid"] == 2, "synthetic_allowed"].item() is True


def test_pipeline_uses_fallback_without_v385_and_emits_no_submissions(tmp_path):
    (tmp_path / "v382_synthetic_teacher_evaluator").mkdir()
    out_dir = tmp_path / "v386_synthetic_contrastive_scorer"

    pd.DataFrame(
        {
            "rally_uid": [101],
            "base_point": [8],
            "candidate_point": [9],
            "base_action": [3],
            "candidate_action": [3],
            "phase": ["rally"],
            "support_count": [12],
            "source_family_count": [3],
            "score": [4.0],
        }
    ).to_csv(
        tmp_path
        / "v382_synthetic_teacher_evaluator"
        / "point_candidate_synthetic_scores.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "rally_uid": [102],
            "base_point": [9],
            "candidate_point": [9],
            "base_action": [11],
            "candidate_action": [3],
            "phase": ["rally"],
            "support_count": [8],
            "source_family_count": [2],
            "score": [3.0],
        }
    ).to_csv(
        tmp_path
        / "v382_synthetic_teacher_evaluator"
        / "action_candidate_synthetic_scores.csv",
        index=False,
    )

    report = run_pipeline(root=tmp_path, outdir=out_dir)

    assert report["missing_v385"] is True
    assert report["point_candidates_scored"] == 1
    assert report["action_candidates_scored"] == 1
    assert all(not name.startswith("submission_") for name in output_filenames())
    assert not list(out_dir.glob("submission_*.csv"))
    stored = json.loads((out_dir / "search_report.json").read_text())
    assert stored["emitted_submission_csvs"] == []
