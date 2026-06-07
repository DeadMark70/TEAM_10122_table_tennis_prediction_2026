import pandas as pd

from analysis_v343_row_candidate_bank import extract_point_edits, filter_point0_policy


def test_extract_point_edits_skips_unchanged_rows():
    base = pd.DataFrame({"rally_uid": ["a", "b"], "pointId": [8, 9]})
    cand = pd.DataFrame({"rally_uid": ["a", "b"], "pointId": [7, 9]})
    edits = extract_point_edits(base, cand, "source_a")
    assert edits[["rally_uid", "task", "anchor_value", "candidate_value", "source"]].to_dict("records") == [
        {"rally_uid": "a", "task": "point", "anchor_value": 8, "candidate_value": 7, "source": "source_a"}
    ]


def test_candidate_bank_blocks_point0_add_if_policy_false():
    rows = pd.DataFrame({"anchor_value": [8], "candidate_value": [0]})
    assert filter_point0_policy(rows, allow_p0_add=False).empty
