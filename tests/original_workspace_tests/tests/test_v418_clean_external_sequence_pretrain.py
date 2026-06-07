import json

import pandas as pd

from analysis_v418_clean_external_sequence_pretrain import (
    FORBIDDEN_COLUMNS,
    TrainConfig,
    build_vocabulary,
    run_pipeline,
    train_sequence_model,
)


def _tiny_sequence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "OpenTTGames",
                "sequence_id": "r1",
                "event_index": 0,
                "token_family": "attack",
                "phase": "receive",
                "terminal_label": "nonterminal",
                "landing_depth_bin": "short",
                "landing_side_bin": "left",
                "speed_bin": "high",
                "spin_bin": "low",
                "actionId": 9,
                "pointId": 7,
            },
            {
                "source_dataset": "OpenTTGames",
                "sequence_id": "r1",
                "event_index": 1,
                "token_family": "control",
                "phase": "rally",
                "terminal_label": "nonterminal",
                "landing_depth_bin": "long",
                "landing_side_bin": "right",
                "speed_bin": "medium",
                "spin_bin": "high",
                "actionId": 4,
                "pointId": 5,
            },
            {
                "source_dataset": "OpenTTGames",
                "sequence_id": "r1",
                "event_index": 2,
                "token_family": "attack",
                "phase": "rally",
                "terminal_label": "terminal",
                "landing_depth_bin": "half",
                "landing_side_bin": "middle",
                "speed_bin": "low",
                "spin_bin": "medium",
                "actionId": 1,
                "pointId": 2,
            },
            {
                "source_dataset": "CleanLab",
                "sequence_id": "r2",
                "event_index": 0,
                "token_family": "control",
                "phase": "serve",
                "terminal_label": "nonterminal",
                "landing_depth_bin": "short",
                "landing_side_bin": "left",
                "speed_bin": "low",
                "spin_bin": "low",
                "actionId": 3,
                "pointId": 1,
            },
            {
                "source_dataset": "CleanLab",
                "sequence_id": "r2",
                "event_index": 1,
                "token_family": "finish",
                "phase": "rally",
                "terminal_label": "terminal",
                "landing_depth_bin": "long",
                "landing_side_bin": "right",
                "speed_bin": "high",
                "spin_bin": "high",
                "actionId": 2,
                "pointId": 4,
            },
        ]
    )


def test_build_vocabulary_excludes_aicup_exact_labels_and_includes_coarse_tokens():
    vocab = build_vocabulary(_tiny_sequence_frame(), min_count=1)

    assert all("actionId" not in token and "pointId" not in token for token in vocab)
    assert "fam=attack" in vocab
    assert "depth=short" in vocab
    assert "terminal=terminal" in vocab


def test_deterministic_tiny_training_exports_requested_embedding_dimension(tmp_path):
    seq = _tiny_sequence_frame()
    config = TrainConfig(epochs=1, embedding_dim=8, hidden_dim=8, batch_size=2, seed=418, max_windows=20)

    result_a = train_sequence_model(seq, config=config, outdir=tmp_path / "run_a")
    result_b = train_sequence_model(seq, config=config, outdir=tmp_path / "run_b")

    assert result_a.token_embeddings.shape[0] >= 4
    assert result_a.token_embeddings.filter(like="emb_").shape[1] == 8
    assert result_a.sequence_embeddings.filter(like="emb_").shape[1] == 8
    assert result_a.report["epochs"] == 1
    pd.testing.assert_frame_equal(result_a.token_embeddings, result_b.token_embeddings)


def test_run_pipeline_writes_only_v418_outputs_without_forbidden_exact_label_columns(tmp_path):
    input_dir = tmp_path / "v414_masked_pretraining_inputs"
    outdir = tmp_path / "v418_clean_external_sequence_pretrain"
    input_dir.mkdir()
    _tiny_sequence_frame().to_csv(input_dir / "pretrain_sequences.csv", index=False)

    report = run_pipeline(
        input_path=input_dir / "pretrain_sequences.csv",
        outdir=outdir,
        config=TrainConfig(epochs=1, embedding_dim=8, hidden_dim=8, batch_size=2, seed=418, max_windows=20),
    )

    expected_files = {"token_embeddings.csv", "sequence_embeddings.csv", "pretraining_report.json"}
    assert {path.name for path in outdir.iterdir()} == expected_files
    assert report["outdir"] == str(outdir)

    token_embeddings = pd.read_csv(outdir / "token_embeddings.csv")
    sequence_embeddings = pd.read_csv(outdir / "sequence_embeddings.csv")
    saved_report = json.loads((outdir / "pretraining_report.json").read_text(encoding="utf-8"))

    for frame in [token_embeddings, sequence_embeddings]:
        assert FORBIDDEN_COLUMNS.isdisjoint(frame.columns)
        assert all(str(outdir) in str(path) for path in saved_report["outputs"].values())
    assert saved_report["output_rows"]["token_embeddings"] == len(token_embeddings)
    assert saved_report["output_rows"]["sequence_embeddings"] == len(sequence_embeddings)
