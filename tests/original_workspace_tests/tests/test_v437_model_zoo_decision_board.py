import pandas as pd

from analysis_v437_model_zoo_decision_board import (
    candidate_risk_penalty,
    rank_upload_candidates,
)


def test_v437_ranks_clean_safe_candidates_above_risky_outputs():
    rows = pd.DataFrame(
        {
            "candidate": ["clean_good", "risky_high"],
            "clean_eligible": [True, False],
            "local_delta": [0.01, 1.0],
            "point0_additions": [0, 0],
            "serve_additions": [0, 0],
            "target_changed": [9, 9],
        }
    )

    ranked = rank_upload_candidates(rows)

    assert ranked.iloc[0]["candidate"] == "clean_good"


def test_v437_penalizes_point0_and_serve_additions():
    safe = {"clean_eligible": True, "point0_additions": 0, "serve_additions": 0, "target_changed": 10}
    unsafe = {"clean_eligible": True, "point0_additions": 2, "serve_additions": 1, "target_changed": 10}

    assert candidate_risk_penalty(unsafe) > candidate_risk_penalty(safe)


def test_v437_keeps_known_v362_as_fallback_final():
    rows = pd.DataFrame(
        {
            "candidate": ["new_probe", "v362_final_resubmit"],
            "clean_eligible": [True, True],
            "local_delta": [0.001, 0.0],
            "point0_additions": [0, 0],
            "serve_additions": [0, 0],
            "target_changed": [20, 0],
            "is_fallback_final": [False, True],
        }
    )

    ranked = rank_upload_candidates(rows)

    assert "v362_final_resubmit" in ranked["candidate"].tolist()
    assert ranked.loc[ranked["candidate"].eq("v362_final_resubmit"), "recommendation"].iloc[0] == "fallback_final"
