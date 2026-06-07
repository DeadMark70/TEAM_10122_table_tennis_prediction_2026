import pandas as pd

from analysis_v345_nonpoint0_utility_optimizer import filter_nonpoint0_swaps, select_budget


def test_filter_nonpoint0_only():
    bank = pd.DataFrame(
        {
            "anchor_value": [8, 8, 0],
            "candidate_value": [7, 0, 8],
        }
    )
    out = filter_nonpoint0_swaps(bank)
    assert out[["anchor_value", "candidate_value"]].values.tolist() == [[8, 7]]


def test_select_utility_budget_prefers_high_utility():
    bank = pd.DataFrame({"row_id": [0, 1], "utility": [0.1, 0.9]})
    out = select_budget(bank, budget=1)
    assert out["row_id"].tolist() == [1]
