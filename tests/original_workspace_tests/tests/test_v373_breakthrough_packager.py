import pandas as pd

from analysis_v373_breakthrough_packager import (
    candidate_allowed,
    top_usable_candidates,
    prediction_signature,
)


def test_blocks_banned_candidates_by_name():
    assert candidate_allowed("submission_clean.csv")
    assert not candidate_allowed("submission_ttmatch.csv")
    assert not candidate_allowed("submission_oldserver.csv")


def test_signature_dedupes_identical_submission():
    a = pd.DataFrame(
        {
            "rally_uid": [1],
            "actionId": [10],
            "pointId": [8],
            "serverGetPoint": [0.5],
        }
    )

    assert prediction_signature(a) == prediction_signature(a.copy())


def test_top_usable_filters_noop_and_prefers_policy_score():
    ranked = pd.DataFrame(
        {
            "name": ["noop", "medium_point", "safe_point"],
            "path": ["noop.csv", "medium.csv", "safe.csv"],
            "policy_blocked": [False, False, False],
            "duplicate_prediction": [False, False, False],
            "risk_tier": ["safe", "normal", "safe"],
            "changed_rows": [0, 36, 12],
            "score": [99.0, 10.0, 8.0],
        }
    )

    top = top_usable_candidates(ranked, limit=5)

    assert top["name"].tolist() == ["medium_point", "safe_point"]
