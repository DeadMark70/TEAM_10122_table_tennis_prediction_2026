import pandas as pd

from analysis_v221_action_backoff_candidate_generator import (
    build_support_tables,
    choose_supported_candidate,
    choose_exact_gated_candidate,
)


def test_choose_supported_candidate_finds_weak_class_candidate():
    examples = pd.DataFrame(
        {
            "phase": ["rally"] * 30,
            "lag0_action": [1] * 30,
            "lag0_point": [9] * 30,
            "lag0_depth": [3] * 30,
            "lag0_spin": [1] * 30,
            "lag0_strength": [2] * 30,
            "next_action": [12] * 12 + [13] * 10 + [9] * 2 + [1] * 6,
        }
    )
    tables = build_support_tables(examples)
    cand = choose_supported_candidate(
        tables,
        context={
            "phase": "rally",
            "lag0_action": 1,
            "lag0_point": 9,
            "lag0_depth": 3,
            "lag0_spin": 1,
            "lag0_strength": 2,
        },
        base_action=9,
        allowed_actions={12},
        min_support=20,
        min_score=1,
        min_margin=0.05,
    )
    assert cand["candidate_action"] == 12
    assert cand["support_score"] > 0
    assert cand["support_margin"] > 0.05


def test_choose_supported_candidate_returns_none_when_base_is_better():
    examples = pd.DataFrame(
        {
            "phase": ["receive"] * 25,
            "lag0_action": [15] * 25,
            "lag0_point": [5] * 25,
            "lag0_depth": [2] * 25,
            "lag0_spin": [5] * 25,
            "lag0_strength": [2] * 25,
            "next_action": [10] * 20 + [12] * 5,
        }
    )
    tables = build_support_tables(examples)
    cand = choose_supported_candidate(
        tables,
        context={
            "phase": "receive",
            "lag0_action": 15,
            "lag0_point": 5,
            "lag0_depth": 2,
            "lag0_spin": 5,
            "lag0_strength": 2,
        },
        base_action=10,
        allowed_actions={12},
        min_support=20,
        min_score=1,
        min_margin=0.01,
    )
    assert cand is None


def test_choose_exact_gated_candidate_requires_exact_top1():
    examples = pd.DataFrame(
        {
            "phase": ["rally"] * 120,
            "lag0_action": [5] * 120,
            "lag0_point": [7] * 120,
            "lag0_depth": [3] * 120,
            "lag0_spin": [1] * 120,
            "lag0_strength": [2] * 120,
            "next_action": [5] * 53 + [6] * 26 + [13] * 18 + [1] * 10 + [2] * 13,
        }
    )
    tables = build_support_tables(examples)
    cand = choose_exact_gated_candidate(
        tables,
        context={
            "phase": "rally",
            "lag0_action": 5,
            "lag0_point": 7,
            "lag0_depth": 3,
            "lag0_spin": 1,
            "lag0_strength": 2,
        },
        base_action=3,
        gates={5: {"exact_n": 100, "exact_count": 30, "exact_rate": 0.25, "top_gap": 0.045, "support_score": 5, "support_margin": 0.05}},
    )
    assert cand["candidate_action"] == 5
    assert cand["exact_is_top1"] == 1
