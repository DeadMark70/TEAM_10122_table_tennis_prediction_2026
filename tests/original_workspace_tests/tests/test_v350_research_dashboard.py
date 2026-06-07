from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from analysis_v350_research_dashboard import collect_report_status, rank_candidates, run_pipeline


def write_submission(path: Path, points: list[int], actions: list[int] | None = None) -> None:
    actions = actions or [1] * len(points)
    frame = pd.DataFrame(
        {
            "rally_uid": [f"r{i}" for i in range(len(points))],
            "actionId": actions,
            "pointId": points,
            "serverGetPoint": [0.0, 1.0, 0.0, 1.0][: len(points)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def case_root(name: str) -> Path:
    root = Path("v350_research_dashboard") / "test_cases" / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def make_root(name: str) -> Path:
    root = case_root(name)
    write_submission(
        root / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv",
        [1, 2, 3, 4],
    )
    write_submission(
        root
        / "v338_joint_moe_pack"
        / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv",
        [1, 5, 6, 4],
    )
    return root


def test_missing_v347_v348_v349_reports_are_rerunnable() -> None:
    root = make_root("missing_reports")
    write_json(
        root / "v345_nonpoint0_utility_optimizer" / "search_report.json",
        {
            "version": "V345",
            "decision": "HAS_EXPORT",
            "candidate_summary": "v345_nonpoint0_utility_optimizer/candidate_summary.csv",
            "generated_submissions": [],
        },
    )
    write_submission(
        root / "v345_nonpoint0_utility_optimizer" / "submission_v345_nonp0_util_b18__v173action_v300server.csv",
        [1, 5, 7, 4],
    )
    pd.DataFrame(
        [
            {
                "candidate": "v345_nonp0_util_b18",
                "path": "v345_nonpoint0_utility_optimizer/submission_v345_nonp0_util_b18__v173action_v300server.csv",
                "selected_rows": 2,
            }
        ]
    ).to_csv(root / "v345_nonpoint0_utility_optimizer" / "candidate_summary.csv", index=False)

    outdir = root / "v350_research_dashboard"
    report = run_pipeline(root=root, outdir=outdir, expected_rows=4)

    assert report["decision"] == "PARTIAL_RERUN_LATER"
    assert {"V344", "V346", "V347", "V348", "V349"}.issubset(set(report["missing_or_unreadable_versions"]))
    assert (outdir / "candidate_priority.csv").exists()
    assert (outdir / "recommendation.md").exists()
    assert (outdir / "search_report.json").exists()


def test_priority_prefers_limited_nonpoint0_novelty_over_point0() -> None:
    root = make_root("priority")
    write_json(
        root / "v347_v338_v341_diff_audit" / "search_report.json",
        {"version": "V347", "decision": "HAS_EXPORT", "summary": "v347_v338_v341_diff_audit/row_diff.csv"},
    )
    pd.DataFrame(
        [
            {"row_id": 3, "in_v338": False, "in_v341": True},
        ]
    ).to_csv(root / "v347_v338_v341_diff_audit" / "row_diff.csv", index=False)
    write_json(
        root / "v345_nonpoint0_utility_optimizer" / "search_report.json",
        {
            "version": "V345",
            "decision": "HAS_EXPORT",
            "candidate_summary": "v345_nonpoint0_utility_optimizer/candidate_summary.csv",
            "generated_submissions": [],
        },
    )
    write_json(
        root / "v344_point0_swap_optimizer" / "search_report.json",
        {
            "version": "V344",
            "decision": "HAS_EXPORT",
            "summary": "v344_point0_swap_optimizer/candidate_summary.csv",
            "generated_submissions": [],
        },
    )
    write_submission(
        root / "v345_nonpoint0_utility_optimizer" / "submission_v345_nonp0_util_b18__v173action_v300server.csv",
        [1, 5, 7, 4],
    )
    write_submission(
        root / "v344_point0_swap_optimizer" / "submission_v344_point0_swap_k08__v173action_v300server.csv",
        [0, 5, 6, 4],
    )
    pd.DataFrame(
        [
            {
                "candidate": "v345_nonp0_util_b18",
                "path": "v345_nonpoint0_utility_optimizer/submission_v345_nonp0_util_b18__v173action_v300server.csv",
            }
        ]
    ).to_csv(root / "v345_nonpoint0_utility_optimizer" / "candidate_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": "k08",
                "path": "v344_point0_swap_optimizer/submission_v344_point0_swap_k08__v173action_v300server.csv",
            }
        ]
    ).to_csv(root / "v344_point0_swap_optimizer" / "candidate_summary.csv", index=False)

    priority = rank_candidates(root=root, expected_rows=4)

    assert priority.iloc[0]["candidate"] == "v345_nonp0_util_b18"
    assert priority.iloc[0]["recommendation_tier"] == "top_next_upload_priority"
    assert priority.loc[priority["candidate"].eq("k08"), "recommendation_tier"].iloc[0] == "hold_point0_addition"


def test_collect_report_status_marks_present_and_missing() -> None:
    root = make_root("status")
    write_json(
        root / "v346_row_utility_pack" / "search_report.json",
        {"version": "V346", "decision": "HAS_EXPORT", "summary": "v346_row_utility_pack/joint_summary.csv"},
    )
    pd.DataFrame([{"candidate": "c"}]).to_csv(root / "v346_row_utility_pack" / "joint_summary.csv", index=False)

    status, rows = collect_report_status(root)

    assert status["V346"]["status"] == "present"
    assert status["V346"]["summary_rows"] == 1
    assert status["V349"]["status"] == "missing"
    assert {row["version"] for row in rows} == {"V344", "V345", "V346", "V347", "V348", "V349"}
