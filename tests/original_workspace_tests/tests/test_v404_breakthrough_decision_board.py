from pathlib import Path

import pandas as pd

from analysis_v404_breakthrough_decision_board import BEST_PUBLIC_PROVEN_PATH, build_queue, run_pipeline


def _ranked(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_final_slot_is_always_public_proven_candidate(tmp_path):
    queue, report = build_queue(ranked_inputs=[])

    assert queue.iloc[-1]["purpose"] == "final_resubmit_best_public_proven"
    assert queue.iloc[-1]["candidate_path"] == BEST_PUBLIC_PROVEN_PATH
    assert report["reserved_final_slot"] is True


def test_blocks_v391_and_point0_additions(tmp_path):
    ranked = _ranked(
        tmp_path / "v400" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "bad_v391",
                "path": "v391_oof_gated_submission_packager/submission_v391_oof_point_top36__v173action_v300server.csv",
                "point_churn": 12,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            },
            {
                "rank": 2,
                "candidate": "bad_p0",
                "path": "some_submission.csv",
                "point_churn": 8,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 1,
            },
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v400", ranked)])

    assert report["candidate_count_after_gates"] == 0
    assert len(queue) == 1
    assert queue.iloc[0]["candidate_path"] == BEST_PUBLIC_PROVEN_PATH


def test_dedupes_byte_identical_candidates(tmp_path):
    sub_a = tmp_path / "a.csv"
    sub_b = tmp_path / "b.csv"
    content = "rally_uid,actionId,pointId,serverGetPoint\n1,2,3,0.1\n"
    sub_a.write_text(content, encoding="utf-8")
    sub_b.write_text(content, encoding="utf-8")
    ranked = _ranked(
        tmp_path / "v400" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "a",
                "path": str(sub_a),
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            },
            {
                "rank": 2,
                "candidate": "b",
                "path": str(sub_b),
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            },
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v400", ranked)], max_new_probes=2)

    assert report["candidate_count_after_gates"] == 1
    assert queue["candidate_path"].tolist().count(str(sub_a)) == 1
    assert str(sub_b) not in queue["candidate_path"].tolist()


def test_limits_new_probes_to_two_by_default(tmp_path):
    ranked = _ranked(
        tmp_path / "v400" / "ranked_candidates.csv",
        [
            {
                "rank": idx,
                "candidate": f"c{idx}",
                "path": f"submission_{idx}.csv",
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            }
            for idx in range(1, 5)
        ],
    )

    queue, _ = build_queue(ranked_inputs=[("v400", ranked)])

    assert len(queue) == 3
    assert queue.iloc[-1]["purpose"] == "final_resubmit_best_public_proven"


def test_resolves_candidate_path_relative_to_ranked_file_parent(tmp_path):
    submission = tmp_path / "v402_rare_point_specialist_lab" / "submission.csv"
    submission.parent.mkdir(parents=True, exist_ok=True)
    submission.write_text("rally_uid,actionId,pointId,serverGetPoint\n1,2,3,0.1\n", encoding="utf-8")
    ranked = _ranked(
        tmp_path / "v402_rare_point_specialist_lab" / "ranked_candidates.csv",
        [
            {
                "rank": 1,
                "candidate": "relative_path_candidate",
                "path": "submission.csv",
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            }
        ],
    )

    queue, _ = build_queue(ranked_inputs=[("v402", ranked)])

    assert queue.iloc[0]["candidate_path"].endswith("v402_rare_point_specialist_lab/submission.csv")


def test_run_pipeline_writes_outputs(tmp_path):
    report = run_pipeline(outdir=tmp_path / "v404")

    assert (tmp_path / "v404" / "recommended_upload_queue.csv").exists()
    assert (tmp_path / "v404" / "search_report.json").exists()
    assert report["outputs"]["recommended_upload_queue"].endswith("recommended_upload_queue.csv")
