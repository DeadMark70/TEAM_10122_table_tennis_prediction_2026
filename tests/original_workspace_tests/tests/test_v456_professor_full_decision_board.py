import pandas as pd

from analysis_v456_professor_full_decision_board import rank_full_professor_queue


def test_v456_ranks_existing_clean_exploratory_before_private_probe():
    rows = pd.DataFrame(
        {
            "candidate": ["safe_top5", "private_top20"],
            "path": ["safe.csv", "private.csv"],
            "clean_eligible": [True, True],
            "changed_rows": [5, 40],
            "public_evidence": ["none", "none"],
            "risk_penalty": [0.0, 50.0],
            "path_exists": [1, 1],
        }
    )
    ranked = rank_full_professor_queue(rows)
    assert ranked.iloc[0]["candidate"] == "safe_top5"
    assert ranked.loc[ranked["candidate"].eq("private_top20"), "recommendation"].iloc[0] == "exploratory_private"


def test_v456_marks_risky_candidate_never_upload():
    rows = pd.DataFrame(
        {
            "candidate": ["safe", "risky"],
            "path": ["safe.csv", "risk.csv"],
            "clean_eligible": [True, False],
            "changed_rows": [5, 1],
            "public_evidence": ["none", "none"],
            "risk_penalty": [0.0, 0.0],
            "path_exists": [1, 1],
        }
    )
    ranked = rank_full_professor_queue(rows)
    assert ranked.loc[ranked["candidate"].eq("risky"), "recommendation"].iloc[0] == "never_upload"
