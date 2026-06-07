from pathlib import Path

import pytest

from analysis_v331_anchor_contract_validation_lab import (
    OUTDIR,
    PublicRecord,
    ensure_output_path,
    evaluate_anchor_contract,
    historical_sanity_checks,
    known_public_frame,
)


def test_fallback_anchor_is_unsafe():
    report_path = OUTDIR / "unit_v331_report.json"
    report = {
        "frame_meta": {"anchor_oof_source": "fallback_lag0_actionId"},
        "point_fixed_to": "V306 p0 cap0p01",
        "server_fixed_to": "V300 serverGetPoint",
        "generated_submission_count": 0,
        "decision": "DO_NOT_UPLOAD",
    }

    contract = evaluate_anchor_contract(report_path, report)

    assert contract.unsafe
    assert "fallback_lag0_actionId" in contract.unsafe_reasons
    assert contract.action_anchor_source == "fallback_lag0_actionId"


def test_generated_csv_is_not_safe_when_evidence_false():
    report_path = OUTDIR / "unit_v331_report.json"
    report = {
        "action_anchor": "V173 action from V306 submission",
        "point_fixed_to": "V306 p0 cap0p01",
        "server_fixed_to": "V300 serverGetPoint",
        "generated_submissions": ["submission_unit.csv"],
        "best_candidate": {"evidence_pass": 0, "changed_action_rows": 4},
        "decision": "DO_NOT_UPLOAD",
    }

    contract = evaluate_anchor_contract(report_path, report)

    assert contract.generated_submission_count == 1
    assert not contract.evidence_pass
    assert contract.unsafe
    assert "generated_without_evidence" in contract.unsafe_reasons


def test_known_public_ranking_flags_v191_v220_v291_as_negative():
    public = known_public_frame()
    negative = set(public.loc[public["is_negative_public"], "version"])

    assert {"V191", "V220", "V291"}.issubset(negative)

    sanity = historical_sanity_checks()
    assert sanity["passed"]
    assert sanity["checks"]["v306_above_v307"]
    assert sanity["checks"]["v300_above_v307"]
    assert sanity["rank_scores"]["V191"] < 0
    assert sanity["rank_scores"]["V220"] < 0


def test_historical_sanity_fails_if_v220_is_not_negative():
    records = [
        PublicRecord("V306", "v306.csv", 0.3577905, "positive_current_best", "BASELINE_PUBLIC_BEST", ""),
        PublicRecord("V300", "v300.csv", 0.3576975, "positive_clean_best", "CLEAN_BASELINE", ""),
        PublicRecord("V307", "v307.csv", 0.3577789, "negative_saturated", "DO_NOT_UPLOAD_SATURATED", ""),
        PublicRecord("V322", "v322.csv", None, "not_public_small", "REVIEW_SMALL_NONTERMINAL", ""),
        PublicRecord("V328", "v328.csv", None, "local_do_not_upload", "DO_NOT_UPLOAD", ""),
        PublicRecord("V291", "v291.csv", 0.3559391, "negative_action_microedit", "NEGATIVE_PUBLIC", ""),
        PublicRecord("V220", "v220.csv", 0.3542440, "wrong", "CLEAN_BASELINE", ""),
        PublicRecord("V191", "v191.csv", 0.3509562, "negative_full_action", "NEGATIVE_PUBLIC", ""),
    ]

    sanity = historical_sanity_checks(records)

    assert not sanity["passed"]
    assert not sanity["checks"]["v191_v220_v291_negative"]


def test_no_upload_path_writes():
    outdir = OUTDIR
    good = ensure_output_path(outdir / "v331_report.json", outdir=outdir)

    assert good == outdir / "v331_report.json"

    with pytest.raises(ValueError, match="refuses upload"):
        ensure_output_path(outdir / "upload_candidates_20260519" / "bad.csv", outdir=outdir)

    with pytest.raises(ValueError, match="outputs must stay"):
        ensure_output_path(outdir.parent / "elsewhere.csv", outdir=outdir)
