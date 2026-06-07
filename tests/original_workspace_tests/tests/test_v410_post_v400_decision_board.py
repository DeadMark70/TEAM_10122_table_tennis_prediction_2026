from pathlib import Path

import pandas as pd

from analysis_v410_post_v400_decision_board import BEST_PUBLIC_PROVEN_PATH, build_queue, run_pipeline


def _ranked(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_final_slot_is_always_v362():
    queue, report = build_queue(ranked_inputs=[])

    assert queue.iloc[-1]["candidate_path"] == BEST_PUBLIC_PROVEN_PATH
    assert queue.iloc[-1]["purpose"] == "final_resubmit_best_public_proven"
    assert report["reserved_final_slot"] is True


def test_blocks_v391_and_point0_additions(tmp_path):
    ranked = _ranked(
        tmp_path / "ranked.csv",
        [
            {
                "candidate": "v391_bad",
                "path": "v391_oof_gated_submission_packager/submission.csv",
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            },
            {
                "candidate": "p0_bad",
                "path": "some.csv",
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 1,
            },
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v400", ranked)])

    assert report["candidate_count_after_gates"] == 0
    assert len(queue) == 1


def test_dedupes_byte_identical_candidates(tmp_path):
    sub_a = tmp_path / "a.csv"
    sub_b = tmp_path / "b.csv"
    content = "rally_uid,actionId,pointId,serverGetPoint\n1,2,3,0.1\n"
    sub_a.write_text(content, encoding="utf-8")
    sub_b.write_text(content, encoding="utf-8")
    ranked = _ranked(
        tmp_path / "ranked.csv",
        [
            {"candidate": "a", "path": str(sub_a), "point_churn": 9, "action_churn": 0, "server_changed": 0, "point0_additions": 0},
            {"candidate": "b", "path": str(sub_b), "point_churn": 9, "action_churn": 0, "server_changed": 0, "point0_additions": 0},
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v400", ranked)])

    assert report["candidate_count_after_gates"] == 1
    assert queue["candidate_path"].tolist().count(str(sub_a)) == 1
    assert str(sub_b) not in queue["candidate_path"].tolist()


def test_v408_server_only_can_pass_but_other_server_changes_block(tmp_path):
    v408 = _ranked(
        tmp_path / "v408.csv",
        [
            {
                "candidate": "server_ok",
                "path": "server_ok.csv",
                "point_churn": 0,
                "action_churn": 0,
                "server_changed": 1845,
                "server_mad": 0.005,
                "point0_additions": 0,
            }
        ],
    )
    v400 = _ranked(
        tmp_path / "v400.csv",
        [
            {
                "candidate": "server_bad",
                "path": "server_bad.csv",
                "point_churn": 0,
                "action_churn": 0,
                "server_changed": 1845,
                "server_mad": 0.005,
                "point0_additions": 0,
            }
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v408", v408), ("v400", v400)])

    assert report["candidate_count_after_gates"] == 1
    assert queue.iloc[0]["candidate_path"] == "server_ok.csv"


def test_blocks_non_server_noop_candidates(tmp_path):
    ranked = _ranked(
        tmp_path / "ranked.csv",
        [
            {
                "candidate": "noop",
                "path": "noop.csv",
                "point_churn": 0,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            }
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v405", ranked)])

    assert report["candidate_count_after_gates"] == 0
    assert len(queue) == 1


def test_blocks_tiny_point_only_diagnostic_candidates(tmp_path):
    ranked = _ranked(
        tmp_path / "ranked.csv",
        [
            {
                "candidate": "tiny",
                "path": "tiny.csv",
                "point_churn": 2,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            }
        ],
    )

    queue, report = build_queue(ranked_inputs=[("v407", ranked)])

    assert report["candidate_count_after_gates"] == 0
    assert len(queue) == 1


def test_prefers_lower_risk_v400_probe_before_higher_response_medium_candidate(tmp_path):
    ranked = _ranked(
        tmp_path / "ranked.csv",
        [
            {
                "candidate": "top15",
                "path": "top15.csv",
                "point_churn": 15,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            },
            {
                "candidate": "top9",
                "path": "top9.csv",
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
            },
        ],
    )
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {"candidate_path": "top15.csv", "response_score": 0.9},
            {"candidate_path": "top9.csv", "response_score": 0.1},
        ]
    ).to_csv(scores, index=False)

    queue, _ = build_queue(ranked_inputs=[("v400", ranked)], v406_scores_path=scores)

    assert queue.iloc[0]["candidate_path"] == "top9.csv"


def test_queue_contains_at_most_two_new_probes(tmp_path):
    ranked = _ranked(
        tmp_path / "ranked.csv",
        [
            {"candidate": f"c{i}", "path": f"c{i}.csv", "point_churn": 9, "action_churn": 0, "server_changed": 0, "point0_additions": 0}
            for i in range(5)
        ],
    )

    queue, _ = build_queue(ranked_inputs=[("v400", ranked)])

    assert len(queue) == 3
    assert queue.iloc[-1]["candidate_path"] == BEST_PUBLIC_PROVEN_PATH


def test_run_pipeline_writes_outputs(tmp_path):
    report = run_pipeline(outdir=tmp_path / "v410")

    assert (tmp_path / "v410" / "recommended_upload_queue.csv").exists()
    assert (tmp_path / "v410" / "search_report.json").exists()
    assert report["outputs"]["recommended_upload_queue"].endswith("recommended_upload_queue.csv")
