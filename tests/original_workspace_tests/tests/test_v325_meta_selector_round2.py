import csv
import json
import shutil
import uuid
from pathlib import Path

from analysis_v325_meta_selector_round2 import (
    CURRENT_PUBLIC_BEST_FILE,
    CandidateEvidence,
    build_candidate_evidence,
    candidate_from_row,
    select_upload_queue,
    write_outputs,
)


def _workspace_tmp() -> Path:
    path = Path.cwd() / "v325_meta_selector_round2" / f"test_tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


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
    evidence_margin_vs_v319: float = 0.0,
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
        evidence_margin_vs_v319=evidence_margin_vs_v319,
    )


def test_missing_round2_reports_still_builds_baseline_and_v320_candidates():
    tmp_path = _workspace_tmp()
    try:
        (tmp_path / "experiments_log.md").write_text("", encoding="utf-8")
        v320_dir = tmp_path / "v320_clean_candidate_meta_selector"
        v320_dir.mkdir()
        (v320_dir / "v320_report.json").write_text(
            json.dumps(
                {
                    "candidate_evidence": [
                        {
                            "candidate_file": "submission_v316_actioncond_nonterminal_budget24__v173action_v300server.csv",
                            "branch": "V316 nonterminal point correction",
                            "local_delta": 0.0002659787,
                            "changed_rows": 24,
                            "recommendation": "REVIEW",
                            "risk_tier": "medium",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        candidates = build_candidate_evidence(tmp_path)

        assert any(row.candidate_file == CURRENT_PUBLIC_BEST_FILE for row in candidates)
        assert any(row.source_report == "v320_clean_candidate_meta_selector/v320_report.json" for row in candidates)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_noop_and_local_negative_candidates_are_not_uploadable():
    noop = candidate_from_row(
        {
            "submission": "submission_v318_joint_actionpoint_agree_safe__v300server.csv",
            "literal_oof_delta": 0.0,
            "test_changed_rows": 0,
            "decision": "REVIEW",
        },
        "v318_joint_nonterminal_consistency/v318_report.json",
        {},
    )
    local_negative = candidate_from_row(
        {
            "submission": "submission_v322_modelbank_agree12__v173action_v300server.csv",
            "local_delta_vs_v306_point_anchor": -0.0002,
            "test_changed_rows": 12,
            "decision": "REVIEW",
        },
        "v322_nonterminal_point_modelbank/v322_report.json",
        {},
    )

    assert noop is not None
    assert local_negative is not None
    assert noop.recommendation == "DO_NOT_UPLOAD_NOOP"
    assert local_negative.recommendation == "DO_NOT_UPLOAD_LOCAL_NEGATIVE"

    baseline = _candidate(
        CURRENT_PUBLIC_BEST_FILE,
        "V306 public best baseline",
        public_pl=0.3577905,
        already_uploaded=True,
        recommendation="BASELINE",
    )
    queue = select_upload_queue([baseline, noop, local_negative], limit=5)
    roles = {row.candidate_file: row.upload_role for row in queue}

    assert roles[noop.candidate_file] == "DO_NOT_UPLOAD"
    assert roles[local_negative.candidate_file] == "DO_NOT_UPLOAD"
    assert "FIRST_NEW_UPLOAD" not in roles.values()


def test_server_only_stays_quota_only_unless_it_beats_v319_evidence():
    baseline = _candidate(
        CURRENT_PUBLIC_BEST_FILE,
        "V306 public best baseline",
        public_pl=0.3577905,
        already_uploaded=True,
        recommendation="BASELINE",
    )
    v319_like = _candidate(
        "submission_v321_server_value_consensus_mad0p002__v173action_v306point.csv",
        "V321 server robust rankblend",
        local_delta=0.010,
        action_server_changed_rows=1,
        risk_tier="medium",
        recommendation="CONDITIONAL_QUOTA",
        evidence_margin_vs_v319=-0.001,
    )
    beats_v319 = _candidate(
        "submission_v321_server_rankblend_mad0p001__v173action_v306point.csv",
        "V321 server robust rankblend",
        local_delta=0.031,
        action_server_changed_rows=1,
        risk_tier="medium",
        recommendation="REVIEW_SERVER_STRONGER_THAN_V319",
        evidence_margin_vs_v319=0.0014,
    )

    queue = select_upload_queue([baseline, v319_like, beats_v319], limit=5)
    roles = {row.candidate_file: row.upload_role for row in queue}

    assert roles[beats_v319.candidate_file] == "FIRST_NEW_UPLOAD"
    assert roles[v319_like.candidate_file] == "QUOTA_SCARCE_ONLY"


def test_superseded_v306_point0_from_historical_selector_is_not_uploadable():
    candidate = candidate_from_row(
        {
            "candidate_file": "submission_v306_p0_budget18__v173action_v300server.csv",
            "branch": "V306 low-churn point0",
            "local_delta": 0.0035,
            "changed_rows": 18,
            "point0_additions": 18,
            "recommendation": "REVIEW",
        },
        "v315_public_response_meta_selector/v315_report.json",
        {},
    )

    assert candidate is not None
    assert candidate.branch == "V306 superseded point0 variant"
    assert candidate.recommendation == "DO_NOT_UPLOAD_SUPERSEDED"


def test_queue_names_do_not_upload_before_quota_rows_fill_limit():
    baseline = _candidate(
        CURRENT_PUBLIC_BEST_FILE,
        "V306 public best baseline",
        public_pl=0.3577905,
        already_uploaded=True,
        recommendation="BASELINE",
    )
    server_a = _candidate(
        "submission_v319_server_ranktiny__v173action_v306point.csv",
        "V319 clean server value state",
        local_delta=0.029,
        action_server_changed_rows=1,
        risk_tier="medium",
        recommendation="CONDITIONAL_QUOTA",
    )
    server_b = _candidate(
        "submission_v321_server_rankblend_mad0p001__v173action_v306point.csv",
        "V321 server robust rankblend",
        action_server_changed_rows=1,
        risk_tier="medium",
        recommendation="CONDITIONAL_QUOTA",
    )
    saturated = _candidate(
        "submission_v311_auto_calibrated__v173action_v300server.csv",
        "V311 saturated point0 expansion",
        local_delta=0.007,
        changed_rows=36,
        point0_additions=36,
        risk_tier="high",
        recommendation="DO_NOT_UPLOAD_SATURATED",
    )

    queue = select_upload_queue([baseline, server_a, server_b, saturated], limit=3)
    roles = {row.candidate_file: row.upload_role for row in queue}

    assert roles[saturated.candidate_file] == "DO_NOT_UPLOAD"
    assert "QUOTA_SCARCE_ONLY" in roles.values()


def test_write_outputs_exports_v325_queue_report_and_json_only():
    tmp_path = _workspace_tmp()
    outdir = tmp_path / "v325_meta_selector_round2"
    try:
        baseline = _candidate(
            CURRENT_PUBLIC_BEST_FILE,
            "V306 public best baseline",
            public_pl=0.3577905,
            already_uploaded=True,
            recommendation="BASELINE",
        )
        first = _candidate(
            "submission_v322_modelbank_agree12__v173action_v300server.csv",
            "V322 nonterminal point model bank",
            local_delta=0.0011,
            changed_rows=12,
            risk_tier="medium",
        )
        research = _candidate(
            "v324_clean_external_corpus_audit",
            "V324 clean external corpus audit",
            recommendation="RESEARCH_CONTINUE",
            risk_tier="low",
        )
        saturated = _candidate(
            "submission_v311_auto_calibrated__v173action_v300server.csv",
            "V311 saturated point0 expansion",
            local_delta=0.007,
            changed_rows=36,
            point0_additions=36,
            risk_tier="high",
            recommendation="DO_NOT_UPLOAD_SATURATED",
        )
        queue = select_upload_queue([baseline, first, research, saturated], limit=6)

        write_outputs([baseline, first, research, saturated], queue, outdir)

        assert sorted(path.name for path in outdir.iterdir()) == [
            "v325_report.json",
            "v325_report.md",
            "v325_upload_queue.csv",
        ]
        rows = list(csv.DictReader((outdir / "v325_upload_queue.csv").open(encoding="utf-8")))
        roles = {row["upload_role"] for row in rows}
        assert {"BASELINE_PUBLIC_BEST", "FIRST_NEW_UPLOAD", "RESEARCH_CONTINUE", "DO_NOT_UPLOAD"} <= roles
        report = json.loads((outdir / "v325_report.json").read_text(encoding="utf-8"))
        assert report["policy"]["no_upload_copy"] is True
        assert report["next_actions"]["current_clean_best_baseline"] == CURRENT_PUBLIC_BEST_FILE
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
