import json
from pathlib import Path


def test_synthetic_rows_have_required_fields():
    from analysis_v381_rare_synthetic_grammar_generator import generate_synthetic_examples

    rows = generate_synthetic_examples(n_per_rule=2, seed=7)
    row = rows[0]
    assert "synthetic_id" in row
    assert "rule_id" in row
    assert "provenance" in row
    assert "target_action_family" in row
    assert "target_point_depth" in row


def test_synthetic_rows_do_not_use_real_test_rally_uid():
    from analysis_v381_rare_synthetic_grammar_generator import generate_synthetic_examples

    rows = generate_synthetic_examples(n_per_rule=3, seed=7)
    assert all(str(row.get("rally_uid", "")).startswith("synthetic_") for row in rows)


def test_action_point_compatibility_is_reasonable():
    from analysis_v381_rare_synthetic_grammar_generator import compatible_action_point

    assert compatible_action_point(action_id=11, point_id=1) is True
    assert compatible_action_point(action_id=11, point_id=9) is False
    assert compatible_action_point(action_id=3, point_id=9) is True


def test_generate_synthetic_examples_cover_rare_actions_and_points():
    from analysis_v381_rare_synthetic_grammar_generator import generate_synthetic_examples

    rows = generate_synthetic_examples(n_per_rule=1, seed=11)
    action_ids = {row["target_action_id_optional"] for row in rows if row["target_action_id_optional"] is not None}
    point_ids = {row["target_point_id_optional"] for row in rows if row["target_point_id_optional"] is not None}
    assert {3, 4, 5, 7, 8, 9, 12, 14}.issubset(action_ids)
    assert {0, 1, 3, 4, 6, 7, 9}.issubset(point_ids)


def test_write_outputs_records_v380_fallback(tmp_path):
    from analysis_v381_rare_synthetic_grammar_generator import write_outputs

    out_dir = tmp_path / "v381"
    result = write_outputs(out_dir=out_dir, n_per_rule=1, seed=5)

    assert result["row_count"] > 0
    assert (out_dir / "synthetic_rare_grammar.csv").exists()
    assert (out_dir / "synthetic_coverage_summary.csv").exists()
    report = json.loads((out_dir / "search_report.json").read_text(encoding="utf-8"))
    assert report["v380_registry_source"] in {"v380_registry", "fallback_rules"}
    if report["v380_registry_source"] == "fallback_rules":
        note = report["v380_registry_note"].lower()
        assert "absent" in note or "no usable" in note
    assert Path(result["grammar_csv"]).name == "synthetic_rare_grammar.csv"
