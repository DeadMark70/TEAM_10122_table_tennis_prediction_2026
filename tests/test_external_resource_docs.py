from pathlib import Path

import yaml


def test_external_docs_and_audit_tables_exist():
    assert Path("docs/external_resources.md").exists()
    assert Path("docs/ai_usage_disclosure.md").exists()
    assert Path("configs/external_sources.yaml").exists()
    assert Path("scripts/check_external_sources.py").exists()
    assert Path("artifacts/external_audit/license_summary.csv").exists()
    assert Path("artifacts/external_audit/allowed_sources.csv").exists()
    assert Path("artifacts/external_audit/external_source_audit.csv").exists()


def test_old_overlap_note_separates_diagnostic_from_final():
    text = Path("docs/old_overlap_diagnostic_note.md").read_text(encoding="utf-8")
    assert "not selected as the final clean submission" in text
    assert "submission_v362_depth_agree_only__v173action_v300server.csv" in text


def test_external_source_manifest_has_urls_and_no_raw_redistribution():
    manifest = yaml.safe_load(Path("configs/external_sources.yaml").read_text(encoding="utf-8"))
    sources = manifest["external_sources"]
    assert {
        "openttgames",
        "coachai_projects",
        "deepmind_robot_table_tennis",
        "tt3d",
        "spindoe",
        "tt_matchdynamics",
        "aimy",
        "sony_table_tennis",
    }.issubset(sources)

    for key, item in sources.items():
        assert item["source_url"], key
        assert item["license"], key
        assert item["local_path"].startswith("external_data/"), key
        assert item["redistribution_in_repo"] is False, key
        assert item["exact_label_mapping_to_aicup"] is False, key
