import pandas as pd

from analysis_v385_expanded_synthetic_grammar import (
    compatible_action_point,
    generate_expanded_synthetic_examples,
)


def test_expanded_rows_have_required_authenticity_fields():
    rows = generate_expanded_synthetic_examples(n_per_combo=1, seed=11)
    assert len(rows) >= 500
    first = rows[0]
    required = {
        "synthetic_id",
        "rally_uid",
        "rule_id",
        "provenance",
        "source_type",
        "phase",
        "prefix_len_bin",
        "last_action_family",
        "target_action_family",
        "target_point_depth",
        "target_point_side",
        "compatibility_label",
        "weight",
    }
    assert required.issubset(first)
    assert all(str(row["rally_uid"]).startswith("synthetic_") for row in rows)


def test_expanded_rows_include_negative_contrastive_examples():
    rows = generate_expanded_synthetic_examples(n_per_combo=1, seed=11)
    labels = {row["compatibility_label"] for row in rows}
    assert {"compatible", "incompatible"}.issubset(labels)


def test_compatibility_rules_remain_physical():
    assert compatible_action_point(action_id=11, point_id=1) is True
    assert compatible_action_point(action_id=11, point_id=9) is False
    assert compatible_action_point(action_id=3, point_id=9) is True
    assert compatible_action_point(action_id=12, point_id=1) is False
