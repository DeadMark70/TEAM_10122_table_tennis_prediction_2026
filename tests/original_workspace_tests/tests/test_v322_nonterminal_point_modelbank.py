import numpy as np
import pandas as pd
import pytest

from analysis_v322_nonterminal_point_modelbank import (
    EXPECTED_COLUMNS,
    build_best_nonterminal_candidates,
    count_point0_changes,
    ensure_local_output_path,
    select_modelbank_replacements,
    specialist_vote_bank,
    validate_submission_frame,
)


def _prob(rows):
    return np.array(rows, dtype=float)


def test_specialist_vote_bank_counts_agreement_and_supports_high_margin():
    base = np.array([7, 8, 4, 5])
    long_prob = _prob(
        [
            [0, 0, 0, 0, 0, 0, 0, 0.20, 0.70, 0.10],
            [0, 0, 0, 0, 0, 0, 0, 0.45, 0.44, 0.11],
            [0, 0, 0, 0, 0.30, 0.20, 0.10, 0.10, 0.10, 0.10],
            [0, 0, 0, 0, 0.20, 0.30, 0.10, 0.10, 0.10, 0.10],
        ]
    )
    half_prob = _prob(
        [
            [0, 0, 0, 0, 0.10, 0.10, 0.10, 0.40, 0.39, 0.01],
            [0, 0, 0, 0, 0.10, 0.10, 0.10, 0.20, 0.19, 0.01],
            [0, 0, 0, 0, 0.10, 0.55, 0.35, 0, 0, 0],
            [0, 0, 0, 0, 0.20, 0.21, 0.60, 0, 0, 0],
        ]
    )
    action_prob = _prob(
        [
            [0, 0, 0, 0, 0, 0, 0, 0.10, 0.65, 0.25],
            [0, 0, 0, 0, 0, 0, 0, 0.80, 0.10, 0.10],
            [0, 0, 0, 0, 0.20, 0.50, 0.30, 0, 0, 0],
            [0, 0, 0, 0, 0.20, 0.30, 0.60, 0, 0, 0],
        ]
    )

    bank = specialist_vote_bank(
        base,
        {
            "long_side": (long_prob, {7, 8, 9}),
            "half_depth": (half_prob, {4, 5, 6}),
            "action_conditioned": (action_prob, {4, 5, 6, 7, 8, 9}),
        },
    )

    assert bank.candidate.tolist() == [8, 7, 5, 6]
    assert bank.agree_count.tolist() == [2, 1, 2, 2]
    assert bank.best_family.tolist() == ["long_side", "action_conditioned", "half_depth", "half_depth"]
    assert np.all(bank.margin >= 0)


def test_select_modelbank_replacements_uses_agreement_or_high_margin_with_support():
    base = np.array([7, 8, 4, 5, 0, 9])
    candidate = np.array([8, 7, 5, 6, 8, 0])
    score = np.array([0.20, 0.35, 0.10, 0.30, 0.99, 0.99])
    agree = np.array([2, 1, 2, 1, 2, 2])
    margin = np.array([0.20, 0.36, 0.10, 0.29, 0.99, 0.99])
    support = np.array([5, 5, 5, 1, 5, 5])

    selected = select_modelbank_replacements(
        base,
        candidate,
        score,
        agree_count=agree,
        margin=margin,
        support=support,
        budget=4,
        high_margin=0.30,
        min_support=3,
        allowed_pairs={(7, 8), (8, 7), (4, 5), (5, 6)},
    )

    assert selected.tolist() == [True, True, True, False, False, False]


def test_select_modelbank_replacements_is_stable_for_score_ties():
    base = np.array([7, 8, 4])
    candidate = np.array([8, 7, 5])
    score = np.array([0.2, 0.2, 0.2])
    agree = np.array([2, 2, 2])
    margin = np.array([0.2, 0.2, 0.2])
    support = np.array([10, 10, 10])

    selected = select_modelbank_replacements(
        base,
        candidate,
        score,
        agree_count=agree,
        margin=margin,
        support=support,
        budget=2,
    )

    assert selected.tolist() == [True, True, False]


def test_point0_and_submission_guards_match_v322_policy():
    counts = count_point0_changes(np.array([0, 7, 8]), np.array([0, 8, 0]))

    assert counts == {"point0_additions": 1, "point0_removals": 0}

    frame = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [0, 18],
            "pointId": [0, 9],
            "serverGetPoint": [0.0, 1.0],
        }
    )
    out = validate_submission_frame(frame, expected_rows=2)
    assert list(out.columns) == EXPECTED_COLUMNS

    with pytest.raises(ValueError, match="local-only"):
        ensure_local_output_path("upload_candidates_20260519/submission.csv")
    with pytest.raises(ValueError, match="local-only"):
        ensure_local_output_path("submissions/selected/submission.csv")


def test_build_best_nonterminal_candidates_never_selects_zero_target():
    base = np.array([7, 4])
    prob = _prob(
        [
            [0.95, 0, 0, 0, 0, 0, 0, 0.10, 0.20, 0.30],
            [0.90, 0, 0, 0, 0.10, 0.40, 0.30, 0, 0, 0],
        ]
    )

    cand, margin = build_best_nonterminal_candidates(base, prob, allowed_targets={4, 5, 6, 7, 8, 9})

    assert cand.tolist() == [9, 5]
    assert np.allclose(margin, [0.20, 0.30])
