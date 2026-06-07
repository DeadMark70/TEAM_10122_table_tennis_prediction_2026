from analysis_v447_professor_full_pretrain_execute import (
    FullPretrainConfig,
    build_v447_execution_grid,
    select_v447_configs,
)


def test_v447_grid_has_small_and_medium_real_train_configs():
    grid = build_v447_execution_grid()
    names = set(grid)
    assert {"gru_small_exec", "lstm_small_exec", "transformer_small_exec"}.issubset(names)
    assert {"gru_medium_exec", "lstm_medium_exec"}.issubset(names)
    assert grid["gru_small_exec"].max_windows > 256
    assert grid["gru_small_exec"].dropout > 0
    assert grid["gru_small_exec"].mask_probability > 0


def test_v447_smoke_selection_caps_runtime_but_uses_real_models():
    selected = select_v447_configs(build_v447_execution_grid(), mode="smoke")
    assert {cfg.name for cfg in selected} == {"gru_small_exec", "lstm_small_exec"}
    assert all(isinstance(cfg, FullPretrainConfig) for cfg in selected)
    assert all(cfg.max_windows <= 2048 for cfg in selected)
    assert all(cfg.epochs == 1 for cfg in selected)
