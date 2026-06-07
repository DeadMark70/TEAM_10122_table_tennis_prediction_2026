from pathlib import Path


def test_external_docs_and_audit_tables_exist():
    assert Path("docs/external_resources.md").exists()
    assert Path("docs/ai_usage_disclosure.md").exists()
    assert Path("artifacts/external_audit/license_summary.csv").exists()
    assert Path("artifacts/external_audit/allowed_sources.csv").exists()
    assert Path("artifacts/external_audit/external_source_audit.csv").exists()


def test_old_overlap_note_separates_diagnostic_from_final():
    text = Path("docs/old_overlap_diagnostic_note.md").read_text(encoding="utf-8")
    assert "not selected as the final clean submission" in text
    assert "submission_v362_depth_agree_only__v173action_v300server.csv" in text
