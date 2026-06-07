import pandas as pd


def test_each_specialist_only_changes_allowed_point_group():
    from analysis_v402_rare_point_specialist_lab import apply_specialist_gate

    candidates = pd.DataFrame(
        [
            {"row_id": 1, "base_point": 8, "candidate_point": 9, "phase": "rally", "lag0_point_depth": "long", "agreement_count": 3, "source_diversity_count": 2, "nonterminal_safety_score": 1.0},
            {"row_id": 2, "base_point": 8, "candidate_point": 2, "phase": "receive", "lag0_point_depth": "short", "agreement_count": 3, "source_diversity_count": 2, "nonterminal_safety_score": 1.0},
            {"row_id": 3, "base_point": 8, "candidate_point": 5, "phase": "rally", "lag0_point_depth": "half", "agreement_count": 3, "source_diversity_count": 2, "nonterminal_safety_score": 1.0},
            {"row_id": 4, "base_point": 0, "candidate_point": 7, "phase": "rally", "lag0_point_depth": "terminal", "agreement_count": 3, "source_diversity_count": 2, "nonterminal_safety_score": 1.0},
        ]
    )

    assert set(apply_specialist_gate(candidates, "long_side")["candidate_point"]) == {9}
    assert set(apply_specialist_gate(candidates, "short_control")["candidate_point"]) == {2}
    assert set(apply_specialist_gate(candidates, "half_long_boundary")["candidate_point"]) == {5}
    assert set(apply_specialist_gate(candidates, "terminal_removal")["candidate_point"]) == {7}


def test_mixed_candidate_dedupes_overlapping_rows_by_highest_score():
    from analysis_v402_rare_point_specialist_lab import dedupe_mixed_candidates

    rows = pd.DataFrame(
        [
            {"row_id": 10, "candidate_point": 8, "specialist": "long_side", "specialist_score": 0.51},
            {"row_id": 10, "candidate_point": 9, "specialist": "long_side", "specialist_score": 0.73},
            {"row_id": 11, "candidate_point": 2, "specialist": "short_control", "specialist_score": 0.61},
        ]
    )

    mixed = dedupe_mixed_candidates(rows)

    assert list(mixed["row_id"]) == [10, 11]
    assert int(mixed.loc[mixed["row_id"].eq(10), "candidate_point"].iloc[0]) == 9


def test_terminal_removal_never_creates_point0_additions():
    from analysis_v402_rare_point_specialist_lab import apply_specialist_gate

    candidates = pd.DataFrame(
        [
            {"row_id": 1, "base_point": 8, "candidate_point": 0, "phase": "rally", "lag0_point_depth": "long", "agreement_count": 9, "source_diversity_count": 3, "nonterminal_safety_score": 1.0},
            {"row_id": 2, "base_point": 0, "candidate_point": 8, "phase": "rally", "lag0_point_depth": "terminal", "agreement_count": 2, "source_diversity_count": 2, "nonterminal_safety_score": 1.0},
        ]
    )

    selected = apply_specialist_gate(candidates, "terminal_removal")

    assert not ((selected["base_point"] != 0) & (selected["candidate_point"] == 0)).any()
    assert set(selected["candidate_point"]) == {8}


def test_missing_v401_gracefully_falls_back_to_zero_compatibility(tmp_path):
    from analysis_v402_rare_point_specialist_lab import load_v401_compatibility

    compat, report = load_v401_compatibility(tmp_path / "missing_v401")

    assert compat == {}
    assert report["available"] is False
    assert report["fallback"] == "zero_compatibility"


def test_run_pipeline_writes_schema_1845_rows_and_reports():
    from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS
    from analysis_v402_rare_point_specialist_lab import OUTDIR, run_pipeline

    report = run_pipeline(outdir=OUTDIR)

    assert report["anchor_rows"] == 1845
    assert report["v401"]["compatibility_rows"] >= 0
    for key in ["long_side", "short_control", "half_long_boundary", "mixed_specialists"]:
        path = OUTDIR / report["submissions"][key]["path"]
        frame = pd.read_csv(path)
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert len(frame) == 1845
        assert int(report["submissions"][key]["point0_additions"]) == 0
