import json
from pathlib import Path

import pandas as pd

from analysis_v392_submission_budget_recommender import (
    BEST_PUBLIC_PROVEN_PATH,
    build_recommended_queue,
    run_pipeline,
)


def _write_ranked(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_queue_reserves_final_slot_for_best_known_public_candidate(tmp_path):
    v391 = _write_ranked(
        tmp_path / "v391_oof_gated_submission_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "v391_oof_point_top36",
                "path": "v391_oof_gated_submission_packager/submission_v391_oof_point_top36__v173action_v300server.csv",
                "action_churn": 0,
                "point_churn": 36,
                "point0_additions": 0,
                "server_changed": 0,
            }
        ],
    )
    v387 = _write_ranked(tmp_path / "v387_expanded_synthetic_packager" / "ranked_candidates.csv", [])

    queue, report = build_recommended_queue(v387_path=v387, v391_path=v391)

    assert len(queue) <= 7
    assert queue.iloc[-1]["purpose"] == "final_resubmit_best_public_proven"
    assert queue.iloc[-1]["candidate_path"] == BEST_PUBLIC_PROVEN_PATH
    assert report["reserved_final_slot"] is True
    assert report["missing_inputs"]["missing_v391"] is False


def test_does_not_recommend_high_risk_action_before_low_risk_point_without_public_evidence(tmp_path):
    action_submission = tmp_path / "action.csv"
    point_submission = tmp_path / "point.csv"
    action_submission.write_text("rally_uid,actionId,pointId,serverGetPoint\n1,1,1,0.1\n", encoding="utf-8")
    point_submission.write_text("rally_uid,actionId,pointId,serverGetPoint\n1,1,2,0.1\n", encoding="utf-8")
    v391 = _write_ranked(
        tmp_path / "v391_oof_gated_submission_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "v391_action_probe",
                "path": str(action_submission),
                "action_churn": 5,
                "point_churn": 72,
                "point0_additions": 0,
                "server_changed": 0,
            },
            {
                "rank": 2,
                "candidate": "v391_oof_point_top36",
                "path": str(point_submission),
                "action_churn": 0,
                "point_churn": 36,
                "point0_additions": 0,
                "server_changed": 0,
            },
        ],
    )
    v387 = _write_ranked(tmp_path / "v387_expanded_synthetic_packager" / "ranked_candidates.csv", [])

    queue, _ = build_recommended_queue(v387_path=v387, v391_path=v391)
    candidate_paths = queue["candidate_path"].tolist()

    point_index = candidate_paths.index(str(point_submission))
    action_index = candidate_paths.index(str(action_submission))
    assert point_index < action_index


def test_missing_v391_uses_v387_then_v383_fallback_and_records_missing(tmp_path):
    v387 = _write_ranked(
        tmp_path / "v387_expanded_synthetic_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "v387_contrastive_point_top9",
                "path": "v387_expanded_synthetic_packager/submission_v387_contrastive_point_top9__v173action_v300server.csv",
                "action_churn": 0,
                "point_churn": 9,
                "point0_additions": 0,
                "server_changed": 0,
            }
        ],
    )
    v383 = _write_ranked(
        tmp_path / "v383_synthetic_adjusted_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "v383_synth_scored_top9",
                "path": "v383_synthetic_adjusted_packager/submission_v383_synth_scored_top9__v173action_v300server.csv",
                "action_churn": 0,
                "point_churn": 9,
                "point0_additions": 0,
                "server_changed": 0,
            }
        ],
    )

    queue, report = build_recommended_queue(
        v387_path=v387,
        v391_path=tmp_path / "missing_v391.csv",
        v383_path=v383,
    )

    assert report["missing_inputs"]["missing_v391"] is True
    assert queue.iloc[0]["candidate_path"] == (
        "v387_expanded_synthetic_packager/submission_v387_contrastive_point_top9__v173action_v300server.csv"
    )
    assert queue.iloc[1]["candidate_path"] == (
        "v383_synthetic_adjusted_packager/submission_v383_synth_scored_top9__v173action_v300server.csv"
    )


def test_empty_v391_is_treated_as_missing_for_fallback_report(tmp_path):
    v391 = _write_ranked(tmp_path / "v391_oof_gated_submission_packager" / "ranked_candidates.csv", [])
    v387 = _write_ranked(
        tmp_path / "v387_expanded_synthetic_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "v387_contrastive_point_top9",
                "path": "v387_expanded_synthetic_packager/submission_v387_contrastive_point_top9__v173action_v300server.csv",
                "action_churn": 0,
                "point_churn": 9,
                "point0_additions": 0,
                "server_changed": 0,
            }
        ],
    )

    queue, report = build_recommended_queue(v387_path=v387, v391_path=v391)

    assert report["missing_inputs"]["missing_v391"] is True
    assert queue.iloc[0]["candidate_path"].startswith("v387_expanded_synthetic_packager/")


def test_dedupes_candidates_with_identical_submission_contents(tmp_path):
    submission_a = tmp_path / "same_a.csv"
    submission_b = tmp_path / "same_b.csv"
    identical_csv = "rally_uid,actionId,pointId,serverGetPoint\n1,2,3,0.4\n"
    submission_a.write_text(identical_csv, encoding="utf-8")
    submission_b.write_text(identical_csv, encoding="utf-8")
    v391 = _write_ranked(
        tmp_path / "v391_oof_gated_submission_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "same_top36",
                "path": str(submission_a),
                "action_churn": 0,
                "point_churn": 32,
                "point0_additions": 0,
                "server_changed": 0,
            },
            {
                "rank": 2,
                "candidate": "same_top72",
                "path": str(submission_b),
                "action_churn": 0,
                "point_churn": 32,
                "point0_additions": 0,
                "server_changed": 0,
            },
        ],
    )

    queue, _ = build_recommended_queue(v391_path=v391)

    assert queue["candidate_path"].tolist().count(str(submission_a)) == 1
    assert str(submission_b) not in queue["candidate_path"].tolist()


def test_run_pipeline_writes_queue_and_search_report(tmp_path):
    v387 = _write_ranked(
        tmp_path / "v387_expanded_synthetic_packager" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "v387_contrastive_point_top9",
                "path": "v387_expanded_synthetic_packager/submission_v387_contrastive_point_top9__v173action_v300server.csv",
                "action_churn": 0,
                "point_churn": 9,
                "point0_additions": 0,
                "server_changed": 0,
            }
        ],
    )

    report = run_pipeline(
        outdir=tmp_path / "v392_submission_budget_recommender",
        v387_path=v387,
        v391_path=tmp_path / "missing_v391.csv",
        v383_path=tmp_path / "missing_v383.csv",
        experiments_log_path=tmp_path / "missing_experiments_log.md",
    )

    queue_path = tmp_path / "v392_submission_budget_recommender" / "recommended_upload_queue.csv"
    report_path = tmp_path / "v392_submission_budget_recommender" / "search_report.json"
    assert queue_path.exists()
    assert report_path.exists()
    assert pd.read_csv(queue_path)["slot"].max() <= 7
    assert json.loads(report_path.read_text())["missing_inputs"]["missing_v391"] is True
    assert report["outputs"]["recommended_upload_queue"].endswith("recommended_upload_queue.csv")
