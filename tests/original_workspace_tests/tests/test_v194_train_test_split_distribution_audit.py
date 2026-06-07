import numpy as np
import pandas as pd

from analysis_v194_train_test_split_distribution_audit import (
    add_audit_columns,
    categorical_shift_rows,
    phase_from_prefix_len,
    total_variation_distance,
)


def test_phase_from_prefix_len_names_next_stroke_phase():
    assert phase_from_prefix_len(0) == "serve"
    assert phase_from_prefix_len(1) == "receive"
    assert phase_from_prefix_len(2) == "third_ball"
    assert phase_from_prefix_len(3) == "fourth_ball"
    assert phase_from_prefix_len(4) == "rally"
    assert phase_from_prefix_len(12) == "rally"


def test_add_audit_columns_maps_incoming_state():
    df = pd.DataFrame(
        {
            "prefix_len": [1, 2, 4],
            "lag0_actionId": [4, 12, 16],
            "lag0_pointId": [3, 8, 0],
        }
    )
    out = add_audit_columns(df)
    assert out["audit_phase"].tolist() == ["receive", "third_ball", "rally"]
    assert out["audit_lag0_action_family"].tolist() == ["Attack", "Defensive", "Serve"]
    assert out["audit_lag0_depth"].tolist() == ["short", "long", "zero"]
    assert out["audit_lag0_side"].tolist() == ["backhand", "middle", "zero"]


def test_total_variation_distance_and_categorical_shift_are_stable():
    left = pd.Series(["a", "a", "b", "b"])
    right = pd.Series(["a", "b", "b", "c"])
    assert np.isclose(total_variation_distance(left, right), 0.25)
    rows = categorical_shift_rows("feature", left, right, "train", "test")
    row_c = rows[rows["value"].eq("c")].iloc[0]
    assert row_c["train_share"] == 0.0
    assert row_c["test_share"] == 0.25
