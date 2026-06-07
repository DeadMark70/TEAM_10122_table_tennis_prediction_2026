import json

import pandas as pd

from analysis_v415_clean_external_representation import (
    FORBIDDEN_COLUMNS,
    build_clean_representations,
    run_pipeline,
)


def _pretrain_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "CleanCoach",
                "sequence_id": "s1",
                "event_index": 0,
                "token_family": "rally_drive",
                "phase": "serve",
                "landing_depth_bin": "short",
                "landing_side_bin": "left",
                "speed_bin": "low",
                "spin_bin": "medium",
            },
            {
                "source_dataset": "CleanCoach",
                "sequence_id": "s1",
                "event_index": 1,
                "token_family": "rally_loop",
                "phase": "rally",
                "landing_depth_bin": "long",
                "landing_side_bin": "right",
                "speed_bin": "high",
                "spin_bin": "high",
            },
            {
                "source_dataset": "CleanLab",
                "sequence_id": "s2",
                "event_index": 0,
                "token_family": "block",
                "phase": "rally",
                "landing_depth_bin": "half",
                "landing_side_bin": "middle",
                "speed_bin": "medium",
                "spin_bin": "low",
            },
            {
                "source_dataset": "TTMATCH",
                "sequence_id": "bad1",
                "event_index": 0,
                "token_family": "should_not_escape",
                "phase": "rally",
                "landing_depth_bin": "short",
                "landing_side_bin": "left",
                "speed_bin": "low",
                "spin_bin": "low",
            },
            {
                "source_dataset": "sonytabletennis",
                "sequence_id": "bad2",
                "event_index": 0,
                "token_family": "also_bad",
                "phase": "rally",
                "landing_depth_bin": "short",
                "landing_side_bin": "left",
                "speed_bin": "low",
                "spin_bin": "low",
            },
        ]
    )


def _objective_fixture(name: str) -> pd.DataFrame:
    if name == "masked":
        return pd.DataFrame(
            [
                {"source_dataset": "CleanCoach", "sequence_id": "s1", "target_family": "rally_drive"},
                {"source_dataset": "TTMATCH", "sequence_id": "bad1", "target_family": "bad"},
            ]
        )
    if name == "landing":
        return pd.DataFrame(
            [
                {"source_dataset": "CleanCoach", "sequence_id": "s1", "target_depth_bin": "short"},
                {"source_dataset": "sonytabletennis", "sequence_id": "bad2", "target_depth_bin": "short"},
            ]
        )
    return pd.DataFrame(
        [
            {"source_dataset": "CleanLab", "sequence_id": "s2", "speed_norm": 3.0},
            {"source_dataset": "TT-MatchDynamics", "sequence_id": "bad3", "speed_norm": 9.0},
        ]
    )


def test_token_embeddings_have_deterministic_numeric_columns():
    outputs_a, report_a = build_clean_representations(
        _pretrain_fixture(),
        _objective_fixture("masked"),
        _objective_fixture("landing"),
        _objective_fixture("physics"),
    )
    outputs_b, report_b = build_clean_representations(
        _pretrain_fixture(),
        _objective_fixture("masked"),
        _objective_fixture("landing"),
        _objective_fixture("physics"),
    )

    token_embeddings = outputs_a["token_embeddings"]
    embedding_cols = [col for col in token_embeddings.columns if col.startswith("svd_")]

    assert embedding_cols
    assert token_embeddings[embedding_cols].apply(pd.api.types.is_numeric_dtype).all()
    pd.testing.assert_frame_equal(token_embeddings, outputs_b["token_embeddings"])
    assert report_a["embedding_columns"] == report_b["embedding_columns"] == embedding_cols


def test_outputs_do_not_contain_forbidden_exact_aicup_labels():
    outputs, report = build_clean_representations(
        _pretrain_fixture(),
        _objective_fixture("masked"),
        _objective_fixture("landing"),
        _objective_fixture("physics"),
    )

    for frame in outputs.values():
        assert FORBIDDEN_COLUMNS.isdisjoint(frame.columns)
        text = frame.astype(str).to_string()
        for forbidden in FORBIDDEN_COLUMNS:
            assert forbidden not in text
    assert set(report["forbidden_columns"]) == FORBIDDEN_COLUMNS


def test_ttmatch_and_sony_sources_are_excluded_from_outputs():
    outputs, report = build_clean_representations(
        _pretrain_fixture(),
        _objective_fixture("masked"),
        _objective_fixture("landing"),
        _objective_fixture("physics"),
    )

    for frame in outputs.values():
        text = frame.astype(str).to_string()
        assert "TTMATCH" not in text
        assert "TT-MatchDynamics" not in text
        assert "sonytabletennis" not in text
    assert report["excluded_rows"]["pretrain_sequences"] == 2


def test_tiny_fixture_with_fewer_than_32_features_runs(tmp_path):
    tiny = _pretrain_fixture().head(2)
    outputs, report = build_clean_representations(
        tiny,
        pd.DataFrame(columns=["source_dataset", "sequence_id"]),
        pd.DataFrame(columns=["source_dataset", "sequence_id"]),
        pd.DataFrame(columns=["source_dataset", "sequence_id"]),
    )

    embedding_cols = [col for col in outputs["sequence_embeddings"].columns if col.startswith("svd_")]
    assert 1 <= len(embedding_cols) < 32
    assert report["svd_components"] == len(embedding_cols)


def test_run_pipeline_writes_artifacts_and_report_counts(tmp_path):
    indir = tmp_path / "v414"
    outdir = tmp_path / "v415"
    indir.mkdir()
    _pretrain_fixture().to_csv(indir / "pretrain_sequences.csv", index=False)
    _objective_fixture("masked").to_csv(indir / "masked_event_examples.csv", index=False)
    _objective_fixture("landing").to_csv(indir / "landing_intent_examples.csv", index=False)
    _objective_fixture("physics").to_csv(indir / "physics_reconstruction_examples.csv", index=False)

    report = run_pipeline(input_dir=indir, outdir=outdir)

    token_embeddings = pd.read_csv(outdir / "token_embeddings.csv")
    sequence_embeddings = pd.read_csv(outdir / "sequence_embeddings.csv")
    source_summary = pd.read_csv(outdir / "source_embedding_summary.csv")
    saved_report = json.loads((outdir / "pretraining_report.json").read_text(encoding="utf-8"))

    assert len(token_embeddings) == report["output_rows"]["token_embeddings"]
    assert len(sequence_embeddings) == report["output_rows"]["sequence_embeddings"]
    assert len(source_summary) == report["output_rows"]["source_embedding_summary"]
    assert saved_report["output_rows"] == report["output_rows"]
