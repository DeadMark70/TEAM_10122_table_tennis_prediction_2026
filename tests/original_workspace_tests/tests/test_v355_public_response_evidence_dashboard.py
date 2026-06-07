import pandas as pd

from analysis_v355_public_response_evidence_dashboard import (
    block_policy_violations,
    rank_candidates,
    top_recommendations,
)


def test_rank_candidates_prefers_v338_subset_over_point0_addition():
    candidates = pd.DataFrame(
        {
            "name": ["safe_prune", "point0_add"],
            "path": ["v353/safe.csv", "v344/p0.csv"],
            "new_rows_beyond_v338": [0, 0],
            "point0_additions_vs_v306": [0, 12],
            "point_churn_vs_v338": [2, 12],
            "action_preserved": [True, True],
            "server_preserved": [True, True],
        }
    )
    family = pd.DataFrame(
        {
            "family": ["v338_subset", "point0_addition"],
            "best_public_delta_vs_v338": [0.0, -0.001],
        }
    )
    evidence = pd.DataFrame(
        {
            "row_id": [1, 2],
            "candidate_key": ["safe_prune", "point0_add"],
            "independent_evidence_score": [4.0, 2.0],
        }
    )

    ranked = rank_candidates(candidates, family, evidence)

    assert ranked.iloc[0]["name"] == "safe_prune"
    assert ranked.iloc[0]["recommendation_tier"] == "top_review"


def test_block_policy_violations_blocks_old_server_and_ttmatch():
    candidates = pd.DataFrame(
        {
            "name": ["clean", "oldhard", "ttmatch_probe"],
            "path": ["clean.csv", "oldhard.csv", "ttmatch.csv"],
        }
    )

    out = block_policy_violations(candidates)

    blocked = dict(zip(out["name"], out["policy_blocked"]))
    assert blocked["clean"] is False
    assert blocked["oldhard"] is True
    assert blocked["ttmatch_probe"] is True


def test_top_recommendations_limits_to_five_rows():
    ranked = pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(8)],
            "score": list(range(8, 0, -1)),
            "policy_blocked": [False] * 8,
        }
    )

    top = top_recommendations(ranked, limit=5)

    assert len(top) == 5
    assert top["name"].tolist() == ["c0", "c1", "c2", "c3", "c4"]


def test_top_recommendations_deduplicates_same_reverted_row_set():
    ranked = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "path": ["a.csv", "b.csv", "c.csv"],
            "score": [10.0, 9.0, 8.0],
            "policy_blocked": [False, False, False],
            "reverted_row_ids": ["1 2", "1 2", "3 4"],
        }
    )

    top = top_recommendations(ranked, limit=5)

    assert top["name"].tolist() == ["a", "c"]


def test_rank_candidates_uses_best_delta_vs_v338_and_ignores_empty_paths_for_top():
    candidates = pd.DataFrame(
        {
            "name": ["with_path", "empty_path"],
            "path": ["local/submission.csv", ""],
            "family": ["v338_subset", "v338_subset"],
            "new_rows_beyond_v338": [0, 0],
            "point0_additions_vs_v306": [0, 0],
            "point_churn_vs_v338": [2, 1],
        }
    )
    family = pd.DataFrame(
        {
            "family": ["v338_subset"],
            "best_delta_vs_v338": [0.0],
            "clean_recommendation_count": [6],
        }
    )
    evidence = pd.DataFrame()

    ranked = rank_candidates(candidates, family, evidence)
    top = top_recommendations(ranked, limit=5)

    assert ranked["family_public_delta"].max() == 0.0
    assert top["name"].tolist() == ["with_path"]


def test_rank_candidates_uses_low_reverted_row_evidence_for_pruning():
    candidates = pd.DataFrame(
        {
            "name": ["revert_weak", "revert_strong"],
            "path": ["weak.csv", "strong.csv"],
            "new_rows_beyond_v338": [0, 0],
            "point0_additions_vs_v306": [0, 0],
            "point_churn_vs_v338": [1, 1],
            "reverted_row_ids": ["1", "2"],
        }
    )
    evidence = pd.DataFrame(
        {
            "row_id": [1, 2],
            "independent_evidence_score": [1.0, 8.0],
        }
    )

    ranked = rank_candidates(candidates, pd.DataFrame(), evidence)

    assert ranked.iloc[0]["name"] == "revert_weak"
    assert ranked.iloc[0]["reverted_row_evidence_mean"] == 1.0
