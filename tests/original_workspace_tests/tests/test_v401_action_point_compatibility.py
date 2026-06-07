from pathlib import Path

import math

import pandas as pd


def test_family_and_depth_mapping_is_deterministic():
    from analysis_v401_action_point_compatibility import action_family, point_to_depth

    first = [action_family(v) for v in range(19)]
    second = [action_family(v) for v in range(19)]
    assert first == second
    assert action_family(15) == "serve"
    assert point_to_depth(0) == "terminal"
    assert point_to_depth(1) == "short"
    assert point_to_depth(4) == "half"
    assert point_to_depth(9) == "long"


def test_smoothing_returns_finite_log_probabilities_for_unseen_keys():
    from analysis_v401_action_point_compatibility import build_compatibility_scorer

    train = pd.DataFrame(
        {
            "rally_uid": [1, 1, 2],
            "strikeNumber": [1, 2, 1],
            "actionId": [10, 11, 10],
            "pointId": [8, 7, 8],
            "spinId": [1, 2, 1],
            "strengthId": [2, 2, 3],
        }
    )
    scorer = build_compatibility_scorer(train, alpha=0.5)
    context = {
        "phase": "rally",
        "lag0_depth": "unseen_depth",
        "lag0_action_family": "unseen_family",
    }

    value = scorer.log_point_probability(99, 3, context)

    assert math.isfinite(value)


def test_point0_additions_are_blocked():
    from analysis_v401_action_point_compatibility import build_candidate_pool

    anchor = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [10, 10],
            "pointId": [8, 0],
            "serverGetPoint": [0.4, 0.5],
        }
    )
    source = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [10, 10],
            "pointId": [0, 7],
            "serverGetPoint": [0.4, 0.5],
        }
    )

    pool = build_candidate_pool(anchor, [(Path("source.csv"), source)])

    assert list(pool["rally_uid"]) == [2]
    assert list(pool["candidate_point"]) == [7]


def test_compatibility_ranking_prefers_higher_train_supported_pairs():
    from analysis_v401_action_point_compatibility import (
        build_compatibility_scorer,
        score_candidate_pool,
    )

    train = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3, 4, 5, 6],
            "strikeNumber": [1, 1, 1, 1, 1, 1],
            "actionId": [10, 10, 10, 10, 10, 10],
            "pointId": [8, 8, 8, 8, 7, 7],
            "spinId": [1, 1, 1, 1, 1, 1],
            "strengthId": [2, 2, 2, 2, 2, 2],
        }
    )
    scorer = build_compatibility_scorer(train, alpha=0.5)
    pool = pd.DataFrame(
        [
            {
                "rally_uid": 1,
                "row_id": 0,
                "actionId": 10,
                "anchor_point": 7,
                "candidate_point": 8,
                "source_agreement": 1,
                "source_count": 1,
                "sources": "a",
            },
            {
                "rally_uid": 2,
                "row_id": 1,
                "actionId": 10,
                "anchor_point": 8,
                "candidate_point": 7,
                "source_agreement": 1,
                "source_count": 1,
                "sources": "b",
            },
        ]
    )
    contexts = {
        1: {"phase": "receive", "lag0_depth": "long", "lag0_action_family": "drive"},
        2: {"phase": "receive", "lag0_depth": "long", "lag0_action_family": "drive"},
    }

    scored = score_candidate_pool(pool, scorer, contexts, threshold=-999)

    assert int(scored.iloc[0]["candidate_point"]) == 8
    assert scored.iloc[0]["compat_delta"] > scored.iloc[1]["compat_delta"]


def test_run_pipeline_writes_schema_and_1845_rows(tmp_path):
    from analysis_v401_action_point_compatibility import SUBMISSION_COLUMNS, run_pipeline

    report = run_pipeline(outdir=tmp_path)

    expected = {
        "submission_v401_compat_top9__v173action_v300server.csv",
        "submission_v401_compat_top15__v173action_v300server.csv",
        "submission_v401_compat_nonterminal_top24__v173action_v300server.csv",
    }
    assert expected.issubset(set(report["generated_candidates"]))
    for name in expected:
        frame = pd.read_csv(tmp_path / name)
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert len(frame) == 1845
