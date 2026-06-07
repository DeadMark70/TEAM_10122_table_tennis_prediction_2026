from analysis_v450_deep_rare_expert_suite import build_expert_specs, group_expert_specs


def test_v450_has_action_and_point_expert_families():
    specs = build_expert_specs()
    grouped = group_expert_specs(specs)
    assert "action" in grouped
    assert "point" in grouped
    assert any("rare_action" in spec.name for spec in grouped["action"])
    assert any("long_point" in spec.name for spec in grouped["point"])
