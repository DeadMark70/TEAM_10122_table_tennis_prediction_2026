import json

import pandas as pd

from analysis_v431_external_sequence_model_zoo import (
    FORBIDDEN_COLUMNS,
    ModelConfig,
    build_model_registry,
    load_pretrain_sequences,
    run_pipeline,
    train_one_pretrain_model,
)


def tiny_external_sequence_frame() -> pd.DataFrame:
    rows = []
    events = [
        ("s1", 0, "attack", "serve", "nonterminal", "short", "left", "high", "low"),
        ("s1", 1, "control", "rally", "nonterminal", "half", "middle", "medium", "medium"),
        ("s1", 2, "finish", "rally", "terminal", "long", "right", "low", "high"),
        ("s2", 0, "control", "serve", "nonterminal", "short", "right", "low", "low"),
        ("s2", 1, "attack", "rally", "nonterminal", "half", "left", "medium", "high"),
        ("s2", 2, "finish", "rally", "terminal", "long", "middle", "high", "medium"),
    ]
    for sequence_id, event_index, family, phase, terminal, depth, side, speed, spin in events:
        rows.append(
            {
                "source_dataset": "OpenTTGames",
                "sequence_id": sequence_id,
                "event_index": event_index,
                "token_family": family,
                "coarse_family": family,
                "phase": phase,
                "terminal_label": terminal,
                "landing_depth_bin": depth,
                "landing_side_bin": side,
                "speed_bin": speed,
                "spin_bin": spin,
                "actionId": 7,
                "pointId": 9,
                "serverGetPoint": 1,
            }
        )
    return pd.DataFrame(rows)


def test_v431_model_registry_contains_small_medium_large_gru_lstm_transformer():
    registry = build_model_registry()
    names = set(registry)
    assert {"gru_small", "gru_medium", "lstm_small", "lstm_medium", "transformer_small", "transformer_medium"}.issubset(names)
    assert registry["gru_small"].embedding_dim < registry["gru_medium"].embedding_dim
    assert registry["transformer_small"].model_type == "transformer"
    assert registry["gru_small"].mask_probability > 0
    assert "gru_large" not in names

    with_large = build_model_registry(include_large=True)
    assert {"gru_large", "lstm_large", "transformer_large"}.issubset(with_large)


def test_v431_train_one_tiny_model_exports_probability_checkpoint_and_embeddings(tmp_path):
    seq = tiny_external_sequence_frame()
    config = ModelConfig("gru_tiny", "gru", 8, 8, 1, 0.1, 1, 20, batch_size=2, seed=431)

    result = train_one_pretrain_model(seq, config=config, outdir=tmp_path)

    assert result["token_embeddings"].filter(like="emb_").shape[1] == 8
    assert result["sequence_embeddings"].filter(like="emb_").shape[1] == 8
    assert result["report"]["model_name"] == "gru_tiny"
    assert result["report"]["objectives"] == ["family", "depth", "side", "speed", "spin", "terminal"]
    assert (tmp_path / "token_embeddings_gru_tiny.csv").exists()
    assert (tmp_path / "sequence_embeddings_gru_tiny.csv").exists()
    assert (tmp_path / "probabilities_gru_tiny.csv").exists()
    assert (tmp_path / "gru_tiny" / "checkpoint.pt").exists()

    for frame in [result["token_embeddings"], result["sequence_embeddings"], result["probabilities"]]:
        assert FORBIDDEN_COLUMNS.isdisjoint(frame.columns)


def test_v431_load_pretrain_sequences_prefers_v430_and_strips_exact_labels(tmp_path):
    v430 = tmp_path / "v430_external_audit_canonical_expander"
    v414 = tmp_path / "v414_masked_pretraining_inputs"
    v430.mkdir()
    v414.mkdir()

    preferred = tiny_external_sequence_frame()
    preferred["source_dataset"] = "preferred_v430"
    fallback = tiny_external_sequence_frame()
    fallback["source_dataset"] = "fallback_v414"
    preferred.to_csv(v430 / "canonical_expanded_events.csv", index=False)
    fallback.to_csv(v414 / "pretrain_sequences.csv", index=False)

    loaded, report = load_pretrain_sequences(root=tmp_path)

    assert report["input_kind"] == "v430_canonical_expanded_events"
    assert set(loaded["source_dataset"]) == {"preferred_v430"}
    assert FORBIDDEN_COLUMNS.isdisjoint(loaded.columns)
    assert {"token_family", "terminal_label", "landing_depth_bin", "landing_side_bin", "speed_bin", "spin_bin"}.issubset(loaded.columns)


def test_v431_run_pipeline_without_inputs_writes_clear_report_and_does_not_crash(tmp_path):
    report = run_pipeline(root=tmp_path, outdir=tmp_path / "v431_external_sequence_model_zoo", models=["gru_small"], quick=True)

    assert report["status"] == "no_input"
    summary_path = tmp_path / "v431_external_sequence_model_zoo" / "model_zoo_summary.json"
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["status"] == "no_input"
