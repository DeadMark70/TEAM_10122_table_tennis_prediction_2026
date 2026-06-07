from analysis_v315_public_response_meta_selector import (
    CandidateEvidence,
    candidate_from_row,
    rank_candidates,
)


def _candidate(
    name: str,
    branch: str,
    *,
    public_pl: float | None = None,
    local_delta: float = 0.0,
    changed_rows: int = 0,
    point0_additions: int = 0,
    server_changed_rows: int = 0,
    risk_tier: str = "low",
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_file=f"{name}.csv",
        branch=branch,
        public_pl=public_pl,
        local_delta=local_delta,
        changed_rows=changed_rows,
        point0_additions=point0_additions,
        action_server_changed_rows=server_changed_rows,
        risk_tier=risk_tier,
        source_report="unit",
    )


def test_public_positive_v306_baseline_ranks_before_unproven_server_only():
    v306 = _candidate(
        "submission_v306_p0_cap0p01__v173action_v300server",
        "V306 low-churn point0",
        public_pl=0.3577905,
        local_delta=0.00358,
        changed_rows=18,
        point0_additions=18,
        risk_tier="low",
    )
    server_only = _candidate(
        "submission_v309_v302_meanmix_w_0p25__v306p0cap0p01_v173action",
        "V309 server-only",
        local_delta=0.0,
        changed_rows=0,
        server_changed_rows=1,
        risk_tier="medium",
    )

    ranked = rank_candidates([server_only, v306], mode="conservative")

    assert [row.candidate_file for row in ranked][:2] == [
        v306.candidate_file,
        server_only.candidate_file,
    ]


def test_current_v306_row_keeps_public_low_churn_branch_not_server_only():
    row = {
        "candidate": "v306_p0_cap0p01",
        "submission": "submission_v306_p0_cap0p01__v173action_v300server.csv",
        "literal_oof_delta": 0.00358,
        "test_changed_rows": 18,
        "point0_additions": 18,
        "server_source": "v300",
    }

    candidate = candidate_from_row(
        row,
        "v306_point0_addition_probe/v306_report.json",
        {"submission_v306_p0_cap0p01__v173action_v300server.csv": 0.3577905},
    )

    assert candidate is not None
    assert candidate.branch == "V306 low-churn point0"
    assert candidate.public_pl == 0.3577905
    assert candidate.risk_tier == "low"


def test_v306_budget_variant_is_still_low_churn_point0():
    row = {
        "submission": "submission_v306_p0_budget18__v173action_v300server.csv",
        "literal_oof_delta": 0.00351,
        "test_changed_rows": 18,
        "point0_additions": 18,
        "server_source": "v300",
    }

    candidate = candidate_from_row(row, "v306_point0_addition_probe/v306_report.json", {})

    assert candidate is not None
    assert candidate.branch == "V306 low-churn point0"
    assert candidate.risk_tier == "low"


def test_conservative_mode_prefers_v307_budget24_before_cap0p02():
    budget24 = _candidate(
        "submission_v307_p0_budget24__v173action_v300server",
        "V307 point0 dose",
        local_delta=0.00469,
        changed_rows=24,
        point0_additions=24,
        risk_tier="low",
    )
    cap0p02 = _candidate(
        "submission_v307_p0_cap0p02__v173action_v300server",
        "V307 point0 dose",
        local_delta=0.00713,
        changed_rows=36,
        point0_additions=36,
        risk_tier="medium",
    )

    ranked = rank_candidates([cap0p02, budget24], mode="conservative")

    assert ranked[0].candidate_file == budget24.candidate_file


def test_aggressive_mode_prefers_v307_cap0p02_before_budget24():
    budget24 = _candidate(
        "submission_v307_p0_budget24__v173action_v300server",
        "V307 point0 dose",
        local_delta=0.00469,
        changed_rows=24,
        point0_additions=24,
        risk_tier="low",
    )
    cap0p02 = _candidate(
        "submission_v307_p0_cap0p02__v173action_v300server",
        "V307 point0 dose",
        local_delta=0.00713,
        changed_rows=36,
        point0_additions=36,
        risk_tier="medium",
    )

    ranked = rank_candidates([budget24, cap0p02], mode="aggressive")

    assert ranked[0].candidate_file == cap0p02.candidate_file
