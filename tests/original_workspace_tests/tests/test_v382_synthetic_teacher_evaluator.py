import json

import pandas as pd


def test_teacher_score_rewards_matching_depth_family():
    from analysis_v382_synthetic_teacher_evaluator import synthetic_teacher_score

    good = synthetic_teacher_score(action_family="attack", point_depth="long", terminal=False)
    bad = synthetic_teacher_score(action_family="attack", point_depth="short", terminal=False)
    assert good > bad


def test_teacher_does_not_emit_submission_rows():
    from analysis_v382_synthetic_teacher_evaluator import output_filenames

    assert all(not name.startswith("submission_") for name in output_filenames())


def test_missing_v381_uses_deterministic_fallback_and_reports_it(tmp_path):
    from analysis_v382_synthetic_teacher_evaluator import evaluate_synthetic_teacher

    (tmp_path / "v370_point_breakthrough_pool").mkdir()
    (tmp_path / "v371_joint_causal_consistency_lab").mkdir()
    (tmp_path / "v372_action_weakness_redux").mkdir()
    out_dir = tmp_path / "v382_synthetic_teacher_evaluator"

    pd.DataFrame(
        [
            {
                "row_id": 1,
                "rally_uid": 1001,
                "base_point": 9,
                "candidate_point": 7,
                "candidate_depth": "long",
                "candidate_side": "left",
                "is_point0_addition": False,
                "score": 5.0,
            }
        ]
    ).to_csv(tmp_path / "v370_point_breakthrough_pool" / "row_candidate_bank.csv", index=False)
    pd.DataFrame(
        [
            {
                "row_index": 1,
                "rally_uid": 1001,
                "proposed_family": "attack",
                "proposed_depth": "long",
                "candidate_score": 1.5,
            }
        ]
    ).to_csv(
        tmp_path / "v371_joint_causal_consistency_lab" / "consistency_evidence.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "row_index": 2,
                "rally_uid": 1002,
                "base_action": 3,
                "candidate_action": 1,
                "candidate_family": "attack",
                "score": 4.0,
            }
        ]
    ).to_csv(tmp_path / "v372_action_weakness_redux" / "action_candidate_bank.csv", index=False)

    result = evaluate_synthetic_teacher(root=tmp_path, output_dir=out_dir)

    assert result["synthetic_source"] == "deterministic_fallback"
    assert (out_dir / "point_candidate_synthetic_scores.csv").exists()
    assert (out_dir / "action_candidate_synthetic_scores.csv").exists()
    report = json.loads((out_dir / "search_report.json").read_text())
    assert report["missing_v381"] is True
    assert report["emitted_submission_csvs"] == []
