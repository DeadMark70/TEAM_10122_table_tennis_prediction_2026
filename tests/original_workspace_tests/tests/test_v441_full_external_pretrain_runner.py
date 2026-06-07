import json

import pandas as pd

from analysis_v441_full_external_pretrain_runner import (
    FORBIDDEN_COLUMNS,
    PretrainRunConfig,
    _cap_quick_sequences,
    build_professor_model_grid,
    load_professor_pretrain_sequences,
    select_configs_for_mode,
    write_planned_run_outputs,
)


def tiny_external_sequence_frame(source_dataset: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_dataset": [source_dataset, source_dataset, source_dataset],
            "sequence_id": ["s1", "s1", "s1"],
            "event_index": [0, 1, 2],
            "coarse_family": ["serve", "attack", "finish"],
            "phase": ["serve", "rally", "rally"],
            "terminal_label": ["nonterminal", "nonterminal", "terminal"],
            "landing_depth_bin": ["short", "half", "long"],
            "landing_side_bin": ["left", "middle", "right"],
            "speed_bin": ["low", "medium", "high"],
            "spin_bin": ["flat", "top", "back"],
            "actionId": [1, 2, 3],
            "pointId": [0, 1, 2],
            "serverGetPoint": [0, 1, 0],
        }
    )


def test_v441_grid_has_gru_lstm_transformer_small_medium_and_dropout_masking():
    grid = build_professor_model_grid()
    names = set(grid)
    assert {
        "gru_small_full",
        "lstm_small_full",
        "transformer_small_full",
        "gru_medium_full",
        "lstm_medium_full",
        "transformer_medium_full",
    }.issubset(names)
    assert isinstance(grid["gru_small_full"], PretrainRunConfig)
    assert grid["gru_small_full"].dropout > 0
    assert grid["gru_small_full"].mask_probability > 0


def test_v441_quick_mode_limits_windows_and_full_mode_does_not():
    grid = build_professor_model_grid()
    quick = select_configs_for_mode(grid, mode="quick")
    full = select_configs_for_mode(grid, mode="full")
    assert all(cfg.max_windows <= 5000 for cfg in quick)
    assert any(cfg.max_windows >= 50000 for cfg in full)


def test_v441_prefers_v440_weighted_external_events_and_strips_exact_labels(tmp_path):
    v440 = tmp_path / "v440_professor_corpus_weighting"
    v430 = tmp_path / "v430_external_audit_canonical_expander"
    v440.mkdir()
    v430.mkdir()
    tiny_external_sequence_frame("weighted_v440").to_csv(v440 / "v440_weighted_external_events.csv", index=False)
    tiny_external_sequence_frame("fallback_v430").to_csv(v430 / "canonical_expanded_events.csv", index=False)

    loaded, report = load_professor_pretrain_sequences(root=tmp_path)

    assert report["input_kind"] == "v440_weighted_external_events"
    assert set(loaded["source_dataset"]) == {"weighted_v440"}
    assert FORBIDDEN_COLUMNS.isdisjoint(loaded.columns)


def test_v441_planned_full_mode_writes_reports_without_training(tmp_path):
    grid = build_professor_model_grid()
    selected = select_configs_for_mode(grid, mode="full", models=["gru_small_full"])

    summary = write_planned_run_outputs(
        selected,
        outdir=tmp_path,
        mode="full",
        input_report={"status": "loaded", "input_kind": "v430_canonical_expanded_events"},
        reason="dry-run full mode",
    )

    reports = pd.read_csv(tmp_path / "model_reports.csv")
    saved_summary = json.loads((tmp_path / "pretrain_run_summary.json").read_text(encoding="utf-8"))
    assert reports.loc[0, "status"] == "planned"
    assert reports.loc[0, "model_name"] == "gru_small_full"
    assert summary["status"] == "planned"
    assert saved_summary["launch_policy"] == "non_quick_modes_are_planned_by_default"


def test_v441_quick_cap_keeps_multiple_sources_when_available():
    dominant = pd.concat([tiny_external_sequence_frame("source_a").assign(sequence_id=f"a_{idx}") for idx in range(4)])
    seq = pd.concat(
        [
            dominant,
            tiny_external_sequence_frame("source_b"),
            tiny_external_sequence_frame("source_c"),
        ],
        ignore_index=True,
    )

    capped = _cap_quick_sequences(seq, row_cap=6)

    assert len(capped) <= 6
    assert capped.groupby(["source_dataset", "sequence_id"]).size().min() >= 2
    assert capped["source_dataset"].nunique() >= 2
