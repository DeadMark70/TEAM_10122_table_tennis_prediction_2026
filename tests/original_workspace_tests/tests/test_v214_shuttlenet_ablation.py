from analysis_v214_shuttlenet_component_ablation import (
    AblationConfig,
    config_slug,
    should_use_component,
    summarize_ablation_winner,
)


def test_config_slug_names_disabled_components():
    cfg = AblationConfig(use_tpe=False, use_pgfn_beta=False, use_type_area=False, use_point_aux=False)
    assert config_slug(cfg) == "no_tpe_no_beta_no_taa_no_point_aux"


def test_should_use_component_respects_config_flags():
    cfg = AblationConfig(use_tpe=False, use_pgfn_alpha=True, use_pgfn_beta=False, use_type_area=True, use_point_aux=False)
    assert should_use_component(cfg, "tpe") is False
    assert should_use_component(cfg, "alpha") is True
    assert should_use_component(cfg, "beta") is False
    assert should_use_component(cfg, "type_area") is True
    assert should_use_component(cfg, "point_aux") is False


def test_summarize_ablation_winner_prefers_selector_delta_then_churn():
    rows = [
        {"ablation": "a", "best_selector_delta": 0.001, "best_selector_churn": 0.01},
        {"ablation": "b", "best_selector_delta": 0.001, "best_selector_churn": 0.005},
        {"ablation": "c", "best_selector_delta": -0.1, "best_selector_churn": 0.001},
    ]
    winner = summarize_ablation_winner(rows)
    assert winner["ablation"] == "b"
