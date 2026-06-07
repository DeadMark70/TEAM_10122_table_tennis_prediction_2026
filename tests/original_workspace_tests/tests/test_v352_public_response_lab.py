import json
from pathlib import Path

import pandas as pd

from analysis_v352_public_response_lab import (
    build_public_response_table,
    classify_family,
    is_clean_recommendation,
    parse_public_results,
    write_reports,
)


CASE_ROOT = Path("v352_public_response_lab/test_cases")


def case_dir(name: str) -> Path:
    path = CASE_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_public_result_parser_extracts_pl_values_from_markdown_snippets():
    markdown = """
| ID | File | Public LB / PL | Notes |
| --- | --- | ---: | --- |
| V338 | submission_v338.csv | 0.3590041 | public-positive anchor |
| v341_expansion | submission_v341_extra.csv | 0.3581101 | expansion probe |

- V220 public PL: 0.3542440
- `submission_v191_pack.csv` scored PL=0.3509562.
"""

    parsed = parse_public_results(markdown)

    by_name = {row["candidate"]: row["public_pl"] for row in parsed}
    assert by_name["V338"] == 0.3590041
    assert by_name["v341_expansion"] == 0.3581101
    assert by_name["V220"] == 0.3542440
    assert by_name["submission_v191_pack.csv"] == 0.3509562


def test_family_classifier_maps_known_candidate_versions():
    assert classify_family("submission_v338_positive.csv") == "v338_positive"
    assert classify_family("v341_expansion_extra") == "v341_expansion_negative"
    assert classify_family("submission_v191_v166_cap5_action_packager") == "v191_v166_action_negative"
    assert classify_family("v220_action_backoff_support_filter") == "v220_action_repair_negative"
    assert classify_family("submission_v291_weakclass_probe") == "v291_weakclass_negative"
    assert classify_family("V307") == "v307_p0_saturated"
    assert classify_family("submission_v272_point_micro") == "v272_v277_point_micro_negative"
    assert classify_family("submission_v300_anchor.csv") == "anchor_v300"
    assert classify_family("submission_v306_anchor.csv") == "anchor_v306"


def test_v338_ranks_above_known_negative_public_families():
    log_text = """
| ID | File | Public LB / PL | Rank | Notes |
| --- | --- | ---: | --- | --- |
| V338 | submission_v338_positive.csv | 0.3590041 | 1 | clean anchor |
| V341 | submission_v341_expansion.csv | 0.3581101 | 2 | expansion |
| V220 | submission_v220_action_repair.csv | 0.3542440 | 3 | action repair |
| V191 | submission_v191_v166_action.csv | 0.3509562 | 4 | action pack |
"""
    table = build_public_response_table(root=case_dir("rank"), log_text=log_text, include_fallback=False)

    rank_by_family = {
        row.family: row.public_rank_desc for row in table.itertuples(index=False)
    }
    assert rank_by_family["v338_positive"] < rank_by_family["v341_expansion_negative"]
    assert rank_by_family["v338_positive"] < rank_by_family["v220_action_repair_negative"]
    assert rank_by_family["v338_positive"] < rank_by_family["v191_v166_action_negative"]
    assert table.loc[table["family"].eq("v338_positive"), "public_delta_vs_v306"].iloc[0] > 0


def test_policy_blocks_old_server_and_ttmatch_from_clean_recommendation():
    assert is_clean_recommendation("v338_positive", "submission_v338.csv")
    assert not is_clean_recommendation("old_server_direct", "submission_r28_old_server_direct_diagnostic.csv")
    assert not is_clean_recommendation("unknown", "ttmatch_candidate.csv")


def test_write_reports_exports_response_summary_and_search_report():
    root = case_dir("write_reports")
    meta_dir = root / "v350_research_dashboard"
    meta_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "candidate": "submission_v338_positive",
                "point_churn_vs_v306": 24,
                "point0_additions": 0,
                "v338_changed_overlap": 24,
                "v341_extra_overlap_rate": 0.0,
            }
        ]
    ).to_csv(meta_dir / "candidate_priority.csv", index=False)

    log_text = "V338 submission_v338_positive PL 0.3590041\nV341 submission_v341_expansion PL 0.3581101\n"

    outputs = write_reports(root=root, log_text=log_text, include_fallback=False)

    assert outputs["public_response_table"].exists()
    assert outputs["family_response_summary"].exists()
    assert outputs["search_report"].exists()
    table = pd.read_csv(outputs["public_response_table"])
    summary = pd.read_csv(outputs["family_response_summary"])
    report = json.loads(outputs["search_report"].read_text())
    assert {"public_delta_vs_v338", "point0_addition_family", "clean_recommendation"}.issubset(table.columns)
    assert "v338_positive" in set(summary["family"])
    assert report["policy"]["wrote_upload_candidates"] is False
    assert report["policy"]["used_ttmatch"] is False
    assert report["policy"]["used_old_server_branch"] is False
