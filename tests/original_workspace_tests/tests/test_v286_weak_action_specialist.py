import numpy as np
import pandas as pd

from analysis_v286_weak_action_specialist_pretraining import (
    WEAK_ACTIONS,
    add_support_counts,
    build_external_response_prior,
    action_family,
    build_candidate_gate,
    filter_clean_external_corpus,
    normalize_rows_safe,
    weak_action_targets,
)


def test_weak_actions_are_the_v173_weak_targets():
    assert list(WEAK_ACTIONS) == [0, 3, 5, 7, 8, 9, 14]


def test_action_family_mapping():
    assert action_family(0) == "Zero"
    assert action_family(1) == "Attack"
    assert action_family(7) == "Attack"
    assert action_family(8) == "Control"
    assert action_family(11) == "Control"
    assert action_family(12) == "Defensive"
    assert action_family(14) == "Defensive"
    assert action_family(15) == "Serve"


def test_weak_action_targets_are_multilabel():
    y = np.array([0, 1, 3, 8, 14])
    out = weak_action_targets(y)
    assert out.shape == (5, 7)
    assert out[0, 0] == 1
    assert out[1].sum() == 0
    assert out[2, 1] == 1
    assert out[3, 4] == 1
    assert out[4, 6] == 1


def test_normalize_rows_safe_handles_bad_rows():
    arr = np.array([[0.0, 0.0], [np.nan, 2.0]])
    out = normalize_rows_safe(arr)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
    assert np.allclose(out[0], [0.5, 0.5])


def test_build_candidate_gate_requires_confidence_and_support():
    frame = pd.DataFrame(
        {
            "specialist_score": [0.9, 0.9, 0.4],
            "support_count": [30, 2, 40],
            "candidate_action": [8, 8, 8],
            "anchor_action": [10, 10, 10],
        }
    )
    gate = build_candidate_gate(frame, min_score=0.7, min_support=10)
    assert gate.tolist() == [True, False, False]


def test_filter_clean_external_corpus_excludes_banned_sources_and_non_yellow_ttmatchdynamics():
    corpus = pd.DataFrame(
        {
            "source_dataset": ["openttgames", "TTMATCH", "TT-MatchDynamics", "TT-MatchDynamics", "CoachAI"],
            "risk_tier": ["GREEN", "GREEN", "GREEN", "YELLOW", "GREEN"],
            "coarse_family": ["Attack", "Control", "Defensive", "Zero", "Attack"],
            "phase": ["receive", "third", "rally", "rally", "receive"],
            "terminal_like": [False, False, False, True, False],
        }
    )
    clean, audit = filter_clean_external_corpus(corpus)
    assert clean["source_dataset"].tolist() == ["openttgames", "TT-MatchDynamics"]
    assert audit["ttmatch_rows_used"] == 0
    assert audit["coachai_rows_used"] == 0


def test_external_prior_is_coarse_family_only():
    corpus = pd.DataFrame(
        {
            "phase": ["receive", "receive", "rally"],
            "phase_bin": ["receive", "receive", "rally"],
            "prev_family": ["Attack", "Attack", "Control"],
            "depth_bin": ["short", "short", "long"],
            "coarse_family": ["Attack", "Control", "Defensive"],
            "terminal_like": [False, False, True],
            "raw_label": ["drive", "push", "lob"],
        }
    )
    prior = build_external_response_prior(corpus)
    assert "actionId" not in prior.columns
    assert {"v286_ext_family_Attack", "v286_ext_family_Control", "v286_ext_family_Defensive", "v286_ext_family_Zero"}.issubset(
        prior.columns
    )
    assert np.allclose(prior[[c for c in prior.columns if c.startswith("v286_ext_family_")]].sum(axis=1), 1.0)


def test_add_support_counts_uses_max_backoff_support():
    frame = pd.DataFrame(
        {
            "phase_bin": ["receive", "receive"],
            "lag0_actionId": [1, 1],
            "lag0_pointId": [3, 4],
            "lag0_action_family": ["Attack", "Attack"],
            "lag0_point_depth": ["long", "long"],
            "candidate_action": [8, 14],
        }
    )
    support_tables = {
        "exact": pd.DataFrame(
            {"phase_bin": ["receive"], "lag0_actionId": [1], "lag0_pointId": [3], "candidate_action": [8], "support_count": [5]}
        ),
        "family_depth": pd.DataFrame(
            {
                "phase_bin": ["receive"],
                "lag0_action_family": ["Attack"],
                "lag0_point_depth": ["long"],
                "candidate_action": [8],
                "support_count": [11],
            }
        ),
        "phase": pd.DataFrame({"phase_bin": ["receive"], "candidate_action": [14], "support_count": [17]}),
        "global": pd.DataFrame({"candidate_action": [8], "support_count": [23]}),
    }
    out = add_support_counts(frame, support_tables)
    assert out["support_count"].tolist() == [23, 17]
