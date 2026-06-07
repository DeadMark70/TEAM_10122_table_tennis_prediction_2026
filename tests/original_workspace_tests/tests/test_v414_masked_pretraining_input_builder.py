import pandas as pd

from analysis_v414_masked_pretraining_input_builder import (
    EXPECTED_BINS,
    build_pretraining_inputs,
    run_pipeline,
)


def _canonical_fixture():
    return pd.DataFrame(
        [
            {
                "source_dataset": "TT3D",
                "sequence_id": "s1",
                "event_index": 0,
                "coarse_family": "table_tennis_trajectory",
                "phase": "trajectory",
                "terminal_label": "",
                "landing_x": 0.0,
                "landing_y": 0.1,
                "landing_z": 0.2,
                "landing_depth_bin": "",
                "landing_side_bin": "",
                "speed_norm": 1.0,
                "spin_norm": "",
            },
            {
                "source_dataset": "TT3D",
                "sequence_id": "s1",
                "event_index": 1,
                "coarse_family": "table_tennis_trajectory",
                "phase": "trajectory",
                "terminal_label": "",
                "landing_x": 0.5,
                "landing_y": 0.8,
                "landing_z": 0.2,
                "landing_depth_bin": "",
                "landing_side_bin": "",
                "speed_norm": 5.0,
                "spin_norm": "",
            },
            {
                "source_dataset": "CoachAI-Projects-main",
                "sequence_id": "r1",
                "event_index": 0,
                "coarse_family": "badminton_drop",
                "phase": "serve",
                "terminal_label": "",
                "landing_x": 1.0,
                "landing_y": 1.0,
                "landing_z": "",
                "landing_depth_bin": "short",
                "landing_side_bin": "left",
                "speed_norm": "",
                "spin_norm": "",
            },
        ]
    )


def test_outputs_do_not_include_exact_aicup_labels():
    outputs, _ = build_pretraining_inputs(_canonical_fixture())

    forbidden = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
    for frame in outputs.values():
        assert forbidden.isdisjoint(frame.columns)


def test_masked_examples_contain_context_and_targets():
    outputs, _ = build_pretraining_inputs(_canonical_fixture())
    masked = outputs["masked_event_examples"]

    assert {"context_indices", "masked_index", "target_family", "target_phase", "target_terminal"}.issubset(masked.columns)
    assert len(masked) >= 1


def test_bins_are_in_expected_vocabularies():
    outputs, _ = build_pretraining_inputs(_canonical_fixture())
    sequences = outputs["pretrain_sequences"]

    assert set(sequences["landing_depth_bin"]).issubset(EXPECTED_BINS["landing_depth_bin"])
    assert set(sequences["landing_side_bin"]).issubset(EXPECTED_BINS["landing_side_bin"])
    assert set(sequences["speed_bin"]).issubset(EXPECTED_BINS["speed_bin"])


def test_ttmatch_and_sony_rows_do_not_appear():
    canonical = _canonical_fixture()
    canonical.loc[len(canonical)] = {
        "source_dataset": "TTMATCH",
        "sequence_id": "bad",
        "event_index": 0,
        "coarse_family": "bad",
        "phase": "rally",
        "terminal_label": "",
        "landing_x": "",
        "landing_y": "",
        "landing_z": "",
        "landing_depth_bin": "",
        "landing_side_bin": "",
        "speed_norm": "",
        "spin_norm": "",
    }

    outputs, _ = build_pretraining_inputs(canonical)

    assert "TTMATCH" not in set(outputs["pretrain_sequences"]["source_dataset"])


def test_run_pipeline_writes_report_and_outputs(tmp_path):
    canonical_path = tmp_path / "clean.csv"
    _canonical_fixture().to_csv(canonical_path, index=False)

    report = run_pipeline(canonical_path=canonical_path, outdir=tmp_path / "out")

    assert report["objective_counts"]["masked_event_examples"] >= 1
    assert (tmp_path / "out" / "pretraining_input_report.json").exists()
    assert (tmp_path / "out" / "physics_reconstruction_examples.csv").exists()
