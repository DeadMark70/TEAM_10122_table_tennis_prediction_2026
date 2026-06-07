def test_registry_requires_provenance_fields():
    from analysis_v380_synthetic_authenticity_registry import validate_rule_registry

    rules = [
        {
            "rule_id": "attack_long_side",
            "motivation": "rare long-side attack",
            "allowed_targets": ["point_depth", "point_side"],
            "provenance": "self_made_table_tennis_grammar",
            "not_test_specific": True,
        }
    ]
    result = validate_rule_registry(rules)
    assert result["ok"] is True


def test_registry_rejects_test_specific_rule():
    from analysis_v380_synthetic_authenticity_registry import validate_rule_registry

    rules = [
        {
            "rule_id": "bad",
            "motivation": "uses test row",
            "allowed_targets": ["pointId"],
            "provenance": "manual_test_inspection",
            "not_test_specific": False,
        }
    ]
    result = validate_rule_registry(rules)
    assert result["ok"] is False
    assert "test-specific" in " ".join(result["errors"]).lower()


def test_default_registry_covers_required_rare_classes_and_targets():
    from analysis_v380_synthetic_authenticity_registry import (
        COARSE_TARGETS,
        RARE_ACTION_IDS,
        RARE_POINT_IDS,
        build_rule_registry,
        validate_rule_registry,
    )

    rules = build_rule_registry()
    result = validate_rule_registry(rules)
    assert result["ok"] is True

    covered_actions = {action for rule in rules for action in rule.get("rare_action_ids", [])}
    covered_points = {point for rule in rules for point in rule.get("rare_point_ids", [])}
    covered_targets = {target for rule in rules for target in rule["allowed_targets"]}

    assert set(RARE_ACTION_IDS).issubset(covered_actions)
    assert set(RARE_POINT_IDS).issubset(covered_points)
    assert set(COARSE_TARGETS).issubset(covered_targets)
    assert all(rule["provenance"] for rule in rules)
    assert all(rule["not_test_specific"] is True for rule in rules)
