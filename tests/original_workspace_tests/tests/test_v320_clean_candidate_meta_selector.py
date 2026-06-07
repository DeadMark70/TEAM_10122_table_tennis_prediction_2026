import csv
import json
import shutil
import uuid
from pathlib import Path

from analysis_v320_clean_candidate_meta_selector import (
    CURRENT_PUBLIC_BEST_FILE,
    CandidateEvidence,
    build_candidate_evidence,
    candidate_from_row,
    rank_candidates,
    select_upload_queue,
    write_outputs,
)


def _candidate(
    name: str,
    branch: str,
    *,
    public_pl: float | None = None,
    local_delta: float = 0.0,
    changed_rows: int = 0,
    point0_additions: int = 0,
    action_server_changed_rows: int = 0,
    risk_tier: str = "low",
    already_uploaded: bool = False,
    recommendation: str = "REVIEW",
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_file=name,
        branch=branch,
        public_pl=public_pl,
        local_delta=local_delta,
        changed_rows=changed_rows,
        point0_additions=point0_additions,
        action_server_changed_rows=action_server_changed_rows,
        risk_tier=risk_tier,
        source_report="unit",
        already_uploaded=already_uploaded,
        recommendation=recommendation,
    )


def _workspace_tmp() -> Path:
    path = Path.cwd() / f"v320_test_tmp_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_v307_public_negative_is_uploaded_baseline_not_new_upload():
    row = {
        "submission": "submission_v307_p0_budget24__v173action_v300server.csv",
        "literal_oof_delta": 0.004692,
        "test_changed_rows": 24,
        "point0_additions": 24,
        "decision": "REVIEW_EXPLORE",
    }

    candidate = candidate_from_row(
        row,
        "v307_point0_dose_extension/v307_report.json",
        {"submission_v307_p0_budget24__v173action_v300server.csv": 0.3577789},
    )

    assert candidate is not None
    assert candidate.already_uploaded is True
    assert candidate.branch == "V307 point0 dose extension"
    assert candidate.recommendation == "DO_NOT_UPLOAD_SATURATED"
    assert candidate.risk_tier == "high"


def test_rank_penalizes_saturated_point0_below_novel_nonterminal_candidate():
    saturated = _candidate(
        "submission_v311_v188_margin_budget36__v173action_v300server.csv",
        "V311 saturated point0 expansion",
        local_delta=0.0074,
        changed_rows=36,
        point0_additions=36,
        risk_tier="high",
        recommendation="DO_NOT_UPLOAD_SATURATED",
    )
    novel = _candidate(
        "submission_v316_longside_side_budget12__v173action_v300server.csv",
        "V316 nonterminal point correction",
        local_delta=0.0012,
        changed_rows=12,
        point0_additions=0,
        risk_tier="medium",
        recommendation="REVIEW",
    )

    ranked = rank_candidates([saturated, novel])

    assert ranked[0].candidate_file == novel.candidate_file
    assert ranked[-1].recommendation == "DO_NOT_UPLOAD_SATURATED"


def test_select_queue_excludes_uploaded_except_baselines_and_marks_first_second_uploads():
    baseline = _candidate(
        CURRENT_PUBLIC_BEST_FILE,
        "V306 public best baseline",
        public_pl=0.3577905,
        local_delta=0.003578,
        changed_rows=18,
        point0_additions=18,
        risk_tier="low",
        already_uploaded=True,
        recommendation="BASELINE",
    )
    uploaded_negative = _candidate(
        "submission_v307_p0_budget24__v173action_v300server.csv",
        "V307 point0 dose extension",
        public_pl=0.3577789,
        local_delta=0.004692,
        changed_rows=24,
        point0_additions=24,
        risk_tier="high",
        already_uploaded=True,
        recommendation="DO_NOT_UPLOAD_SATURATED",
    )
    first = _candidate(
        "submission_v316_actioncond_nonterminal_budget24__v173action_v300server.csv",
        "V316 nonterminal point correction",
        local_delta=0.0015,
        changed_rows=20,
        risk_tier="medium",
    )
    second = _candidate(
        "submission_v319_server_valueblend_mad0p002__v173action_v306point.csv",
        "V319 clean server value state",
        local_delta=0.0002,
        changed_rows=0,
        action_server_changed_rows=1,
        risk_tier="medium",
        recommendation="CONDITIONAL_QUOTA",
    )

    queue = select_upload_queue([uploaded_negative, second, first, baseline], limit=5)
    roles = {row.candidate_file: row.upload_role for row in queue}

    assert roles[CURRENT_PUBLIC_BEST_FILE] == "BASELINE_PUBLIC_BEST"
    assert roles[first.candidate_file] == "FIRST_NEW_UPLOAD"
    assert roles[second.candidate_file] == "SECOND_UPLOAD"
    assert roles[uploaded_negative.candidate_file] == "DO_NOT_UPLOAD"


def test_select_queue_does_not_promote_superseded_v306_or_v308_point0_variants():
    baseline = _candidate(
        CURRENT_PUBLIC_BEST_FILE,
        "V306 public best baseline",
        public_pl=0.3577905,
        already_uploaded=True,
        recommendation="BASELINE",
    )
    superseded_v306 = _candidate(
        "submission_v306_p0_budget18__v173action_v300server.csv",
        "V306 superseded point0 variant",
        local_delta=0.0035,
        changed_rows=18,
        point0_additions=18,
        risk_tier="high",
        recommendation="DO_NOT_UPLOAD_SUPERSEDED",
    )
    superseded_v308 = _candidate(
        "submission_v308_high_margin_top18__v173action_v300server.csv",
        "V308 point0 row ablation",
        local_delta=0.0035,
        changed_rows=18,
        risk_tier="high",
        recommendation="DO_NOT_UPLOAD_SUPERSEDED",
    )
    server = _candidate(
        "submission_v319_server_valueblend_mad0p002__v173action_v306point.csv",
        "V319 clean server value state",
        action_server_changed_rows=1,
        risk_tier="medium",
        recommendation="CONDITIONAL_QUOTA",
    )

    queue = select_upload_queue([superseded_v308, superseded_v306, server, baseline], limit=5)
    roles = {row.candidate_file: row.upload_role for row in queue}

    assert roles[server.candidate_file] == "FIRST_NEW_UPLOAD"
    assert roles[superseded_v306.candidate_file] == "DO_NOT_UPLOAD"
    assert roles[superseded_v308.candidate_file] == "DO_NOT_UPLOAD"


def test_build_candidate_evidence_tolerates_missing_future_reports():
    tmp_path = _workspace_tmp()
    try:
        (tmp_path / "experiments_log.md").write_text(
            "submission_v307_p0_budget24__v173action_v300server.csv\nPL = 0.3577789\n",
            encoding="utf-8",
        )
        report_dir = tmp_path / "v316_nonterminal_point_correction"
        report_dir.mkdir()
        (report_dir / "v316_report.json").write_text(
            json.dumps(
                {
                    "top_review_candidates": [
                        {
                            "submission": "submission_v316_longside_side_budget12__v173action_v300server.csv",
                            "literal_oof_delta": 0.001,
                            "test_changed_rows": 12,
                            "point0_additions": 0,
                            "decision": "REVIEW",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        candidates = build_candidate_evidence(tmp_path)

        assert any(row.candidate_file.startswith("submission_v316") for row in candidates)
        assert any(row.candidate_file == CURRENT_PUBLIC_BEST_FILE for row in candidates)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_write_outputs_creates_v320_queue_and_reports():
    tmp_path = _workspace_tmp()
    try:
        outdir = tmp_path / "v320_clean_candidate_meta_selector"
        baseline = _candidate(
            CURRENT_PUBLIC_BEST_FILE,
            "V306 public best baseline",
            public_pl=0.3577905,
            already_uploaded=True,
            recommendation="BASELINE",
        )
        first = _candidate(
            "submission_v316_longside_side_budget12__v173action_v300server.csv",
            "V316 nonterminal point correction",
            local_delta=0.001,
            changed_rows=12,
        )
        queue = select_upload_queue([baseline, first], limit=5)

        write_outputs([baseline, first], queue, outdir)

        queue_path = outdir / "v320_upload_queue.csv"
        rows = list(csv.DictReader(queue_path.open(encoding="utf-8")))
        assert rows[0]["upload_role"] == "BASELINE_PUBLIC_BEST"
        assert rows[1]["upload_role"] == "FIRST_NEW_UPLOAD"
        assert (outdir / "v320_report.md").exists()
        assert (outdir / "v320_report.json").exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
