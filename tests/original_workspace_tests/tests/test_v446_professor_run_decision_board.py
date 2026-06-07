import pandas as pd

from analysis_v446_professor_run_decision_board import rank_professor_upload_queue


def test_v446_keeps_v362_final_fallback_and_marks_new_candidates_exploratory():
    rows = pd.DataFrame(
        {
            "candidate": ["new_v445", "v362_final"],
            "path": ["new.csv", "v362.csv"],
            "clean_eligible": [True, True],
            "public_evidence": ["none", "positive"],
            "changed_rows": [10, 0],
            "risk_penalty": [0.0, 0.0],
        }
    )

    ranked = rank_professor_upload_queue(rows)

    assert "v362_final" in ranked["candidate"].tolist()
    assert ranked.loc[ranked["candidate"].eq("new_v445"), "recommendation"].iloc[0] == "exploratory"


def test_v446_risky_candidate_never_beats_clean_candidate():
    rows = pd.DataFrame(
        {
            "candidate": ["clean_small", "risky_big"],
            "path": ["clean.csv", "risky.csv"],
            "clean_eligible": [True, False],
            "public_evidence": ["none", "none"],
            "changed_rows": [20, 1],
            "risk_penalty": [0.0, 0.0],
        }
    )

    ranked = rank_professor_upload_queue(rows)

    assert ranked.iloc[0]["candidate"] == "clean_small"
    assert ranked.loc[ranked["candidate"].eq("risky_big"), "recommendation"].iloc[0] == "never_upload"
