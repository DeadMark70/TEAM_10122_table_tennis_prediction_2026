import pandas as pd

from analysis_v348_public_risk_row_gate import aggregate_candidate_rows, score_rows


def test_score_rows_prefers_public_positive_over_unlabeled_candidate():
    rows = pd.DataFrame(
        [
            {
                "row_id": 10,
                "rally_uid": 100,
                "anchor_value": 8,
                "candidate_value": 9,
                "transition": "8->9",
                "same_depth": True,
                "point0_addition": False,
                "point0_removal": False,
                "nonterminal_point_swap": True,
                "changed_in_v338": True,
                "source_count": 3,
                "agreement_count": 3,
                "source_family_count": 2,
                "mean_source_local_delta": 0.003,
                "prefix_len": 4,
                "phase": "mid",
            },
            {
                "row_id": 11,
                "rally_uid": 101,
                "anchor_value": 8,
                "candidate_value": 7,
                "transition": "8->7",
                "same_depth": True,
                "point0_addition": False,
                "point0_removal": False,
                "nonterminal_point_swap": True,
                "changed_in_v338": False,
                "source_count": 1,
                "agreement_count": 1,
                "source_family_count": 1,
                "mean_source_local_delta": 0.0,
                "prefix_len": 6,
                "phase": "late",
            },
        ]
    )

    scored = score_rows(rows, positive_keys={(10, 9)}, v341_keys=set(), v307_keys=set())

    by_key = {(int(row.row_id), int(row.candidate_value)): row for row in scored.itertuples()}
    assert by_key[(10, 9)].positive_label is True
    assert by_key[(10, 9)].trust_score > by_key[(11, 7)].trust_score
    assert by_key[(10, 9)].gate_decision == "ALLOW_HIGH_TRUST"


def test_score_rows_blocks_v341_and_v307_public_risk_rows():
    rows = pd.DataFrame(
        [
            {
                "row_id": 20,
                "rally_uid": 200,
                "anchor_value": 8,
                "candidate_value": 7,
                "transition": "8->7",
                "same_depth": True,
                "point0_addition": False,
                "point0_removal": False,
                "nonterminal_point_swap": True,
                "changed_in_v338": False,
                "source_count": 4,
                "agreement_count": 4,
                "source_family_count": 3,
                "mean_source_local_delta": 0.005,
                "prefix_len": 3,
                "phase": "mid",
            },
            {
                "row_id": 21,
                "rally_uid": 201,
                "anchor_value": 9,
                "candidate_value": 0,
                "transition": "9->0",
                "same_depth": False,
                "point0_addition": True,
                "point0_removal": False,
                "nonterminal_point_swap": False,
                "changed_in_v338": False,
                "source_count": 4,
                "agreement_count": 4,
                "source_family_count": 3,
                "mean_source_local_delta": 0.005,
                "prefix_len": 5,
                "phase": "mid",
            },
        ]
    )

    scored = score_rows(rows, positive_keys=set(), v341_keys={(20, 7)}, v307_keys={(21, 0)})

    assert set(scored["gate_decision"]) == {"BLOCK_PUBLIC_RISK"}
    row20 = scored.loc[scored["row_id"] == 20].iloc[0]
    row21 = scored.loc[scored["row_id"] == 21].iloc[0]
    assert bool(row20["v341_extra_risk"])
    assert bool(row21["v307_extra_p0_risk"])
    assert row21["risk_score"] > row20["risk_score"]


def test_aggregate_candidate_rows_counts_source_families_and_agreement():
    bank = pd.DataFrame(
        [
            {
                "row_id": 1,
                "rally_uid": 99,
                "anchor_value": 8,
                "candidate_value": 9,
                "transition": "8->9",
                "source": "a",
                "source_dir": "dir_a",
                "source_public_tag": "v338_public_positive",
                "source_local_delta_if_known": 0.1,
                "is_point0_addition": False,
                "is_point0_removal": False,
                "is_nonterminal_point_swap": True,
                "is_same_depth_swap": True,
                "changed_in_v338": True,
            },
            {
                "row_id": 1,
                "rally_uid": 99,
                "anchor_value": 8,
                "candidate_value": 9,
                "transition": "8->9",
                "source": "b",
                "source_dir": "dir_b",
                "source_public_tag": "historical_point_model",
                "source_local_delta_if_known": 0.3,
                "is_point0_addition": False,
                "is_point0_removal": False,
                "is_nonterminal_point_swap": True,
                "is_same_depth_swap": True,
                "changed_in_v338": True,
            },
        ]
    )
    context = pd.DataFrame({"row_id": [1], "prefix_len": [7], "phase": ["late"]})

    rows = aggregate_candidate_rows(bank, context)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["agreement_count"] == 2
    assert row["source_count"] == 2
    assert row["source_family_count"] == 2
    assert row["family_count_v338_public_positive"] == 1
    assert row["family_count_historical_point_model"] == 1
    assert row["mean_source_local_delta"] == 0.2
