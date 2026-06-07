import json

import pandas as pd

from analysis_v390_synthetic_augmented_scorer import (
    output_filenames,
    run_pipeline,
    score_feature_frame,
    synthetic_features,
    fit_augmented_model,
)


def _sample_grammar() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "synthetic_id": "synthetic_test_0001",
                "rally_uid": "synthetic_test_0001",
                "phase": "rally",
                "prefix_len_bin": "mid_prefix",
                "last_action_family": "attack",
                "last_spin": "topspin",
                "last_strength": "medium",
                "terminal_context": False,
                "target_action_family": "attack",
                "target_action_id_optional": 3,
                "target_point_depth": "long",
                "target_point_side": "right",
                "target_point_id_optional": 9,
                "compatibility_label": "compatible",
                "weight": 1.0,
            },
            {
                "synthetic_id": "synthetic_test_0002",
                "rally_uid": "synthetic_test_0002",
                "phase": "rally",
                "prefix_len_bin": "short_prefix",
                "last_action_family": "control",
                "last_spin": "backspin",
                "last_strength": "soft",
                "terminal_context": False,
                "target_action_family": "control",
                "target_action_id_optional": 11,
                "target_point_depth": "terminal",
                "target_point_side": "terminal",
                "target_point_id_optional": 0,
                "compatibility_label": "incompatible",
                "weight": 0.35,
            },
            {
                "synthetic_id": "synthetic_test_0003",
                "rally_uid": "synthetic_test_0003",
                "phase": "receive",
                "prefix_len_bin": "short_prefix",
                "last_action_family": "receive",
                "last_spin": "topspin",
                "last_strength": "medium",
                "terminal_context": False,
                "target_action_family": "serve",
                "target_action_id_optional": 15,
                "target_point_depth": "short",
                "target_point_side": "left",
                "target_point_id_optional": 1,
                "compatibility_label": "incompatible",
                "weight": 0.35,
            },
        ]
    )


def test_compatible_synthetic_rows_score_above_incompatible_rows():
    rows = synthetic_features(_sample_grammar())
    model, model_used, error = fit_augmented_model(rows, prefer_sklearn=False)
    scored = score_feature_frame(rows, model)

    compatible = scored.loc[scored["row_id"] == "synthetic_test_0001", "risk_adjusted_score"].item()
    incompatible = scored.loc[scored["row_id"] == "synthetic_test_0002", "risk_adjusted_score"].item()

    assert model_used == "deterministic_linear_fallback"
    assert error is None
    assert compatible > incompatible


def test_point0_additions_are_penalized():
    rows = synthetic_features(_sample_grammar())
    model, _, _ = fit_augmented_model(rows, prefer_sklearn=False)
    base = rows.loc[rows["row_id"] == "synthetic_test_0001"].copy()
    risky = base.copy()
    risky["row_id"] = "synthetic_test_point0"
    risky["target_point_id"] = 0
    risky["target_point_depth"] = "terminal"
    risky["target_point_side"] = "terminal"
    risky["is_point0_addition"] = 1
    frame = pd.concat([base, risky], ignore_index=True)

    scored = score_feature_frame(frame, model)
    safe_score = scored.loc[scored["row_id"] == "synthetic_test_0001", "risk_adjusted_score"].item()
    risky_score = scored.loc[scored["row_id"] == "synthetic_test_point0", "risk_adjusted_score"].item()

    assert safe_score > risky_score
    assert scored.loc[scored["row_id"] == "synthetic_test_point0", "pass_augmented_gate"].item() is False


def test_serve_15_18_additions_are_penalized():
    rows = synthetic_features(_sample_grammar())
    model, _, _ = fit_augmented_model(rows, prefer_sklearn=False)
    base = rows.loc[rows["row_id"] == "synthetic_test_0001"].copy()
    risky = base.copy()
    risky["row_id"] = "synthetic_test_serve15"
    risky["target_action_id"] = 15
    risky["target_action_family"] = "serve"
    risky["phase"] = "rally"
    risky["is_serve_15_18_addition"] = 1
    frame = pd.concat([base, risky], ignore_index=True)

    scored = score_feature_frame(frame, model)
    safe_score = scored.loc[scored["row_id"] == "synthetic_test_0001", "risk_adjusted_score"].item()
    risky_score = scored.loc[scored["row_id"] == "synthetic_test_serve15", "risk_adjusted_score"].item()

    assert safe_score > risky_score
    assert scored.loc[scored["row_id"] == "synthetic_test_serve15", "pass_augmented_gate"].item() is False


def test_pipeline_fallback_records_missing_inputs_and_emits_no_submissions(tmp_path):
    out_dir = tmp_path / "v390_synthetic_augmented_scorer"
    (tmp_path / "v385_expanded_synthetic_grammar").mkdir()
    _sample_grammar().to_csv(tmp_path / "v385_expanded_synthetic_grammar" / "expanded_synthetic_grammar.csv", index=False)

    report = run_pipeline(root=tmp_path, outdir=out_dir, prefer_sklearn=False)

    assert report["missing_v385"] is False
    assert report["missing_v388"] is True
    assert report["candidate_source"] == "deterministic_synthetic_fallback"
    assert report["point_candidates_scored"] == 3
    assert report["action_candidates_scored"] == 3
    assert all(not name.startswith("submission_") for name in output_filenames())
    assert not list(out_dir.glob("submission_*.csv"))

    stored = json.loads((out_dir / "search_report.json").read_text())
    assert stored["emitted_submission_csvs"] == []
    assert (out_dir / "point_augmented_scores.csv").exists()
    assert (out_dir / "action_augmented_scores.csv").exists()


def test_pipeline_scores_v388_pools_even_when_action_pool_lacks_point_columns(tmp_path):
    out_dir = tmp_path / "v390_synthetic_augmented_scorer"
    (tmp_path / "v385_expanded_synthetic_grammar").mkdir()
    (tmp_path / "v388_large_synthetic_candidate_pool").mkdir()
    _sample_grammar().to_csv(tmp_path / "v385_expanded_synthetic_grammar" / "expanded_synthetic_grammar.csv", index=False)
    pd.DataFrame(
        {
            "rally_uid": [101],
            "base_point": [8],
            "candidate_point": [9],
            "support_count": [12],
            "source_family_count": [3],
            "is_point0_addition": [False],
            "same_depth": [True],
            "same_side": [False],
        }
    ).to_csv(tmp_path / "v388_large_synthetic_candidate_pool" / "point_change_pool.csv", index=False)
    pd.DataFrame(
        {
            "rally_uid": [102],
            "base_action": [3],
            "candidate_action": [1],
            "support_count": [4],
            "source_family_count": [2],
            "is_serve_15_18_addition": [False],
            "same_family": [True],
        }
    ).to_csv(tmp_path / "v388_large_synthetic_candidate_pool" / "action_change_pool.csv", index=False)

    report = run_pipeline(root=tmp_path, outdir=out_dir, prefer_sklearn=False)

    assert report["missing_v388"] is False
    assert report["point_candidates_scored"] == 1
    assert report["action_candidates_scored"] == 1
