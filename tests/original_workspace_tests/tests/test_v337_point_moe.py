import numpy as np
import pandas as pd

from analysis_v337_point_moe import (
    _point0_totals,
    apply_point0_policy,
    build_export_frame,
    build_point_experts,
    select_by_budget,
)


def test_no_p0_add_filter_blocks_nonzero_to_zero():
    base = np.array([7, 8, 9, 4])
    cand = np.array([0, 7, 0, 6])
    filtered = apply_point0_policy(base, cand, allow_p0_add=False)
    assert filtered.tolist() == [7, 7, 9, 6]


def test_point_budget_selector_limits_rows():
    base = np.array([8, 8, 8, 8])
    cand = np.array([7, 9, 0, 4])
    utility = np.array([0.1, 0.8, 0.7, 0.6])
    selected = select_by_budget(base, cand, utility, budget=2)
    assert selected.tolist() == [8, 9, 0, 8]


def test_point_experts_include_required_routes_and_preserve_length():
    frame = pd.DataFrame(
        {
            "lag0_pointId": [7, 4, 8],
            "lag0_actionId": [10, 3, 15],
            "prefix_len": [2, 4, 6],
        }
    )
    base = np.array([8, 5, 0])
    experts = build_point_experts(frame, base, action_anchor=np.array([10, 3, 15]))

    assert set(experts) == {
        "terminal_p0",
        "depth_short_half_long",
        "side_fh_mid_bh",
        "long_side_789",
        "no_p0_add_depthside",
        "action_conditioned_table",
    }
    assert all(len(pred) == len(base) for pred in experts.values())
    assert not np.any((base != 0) & (experts["no_p0_add_depthside"] == 0))


def test_export_frame_preserves_action_and_server_schema():
    anchor = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [4, 15],
            "pointId": [0, 9],
            "serverGetPoint": [0.25, 0.75],
        }
    )

    out = build_export_frame(anchor, np.array([3, 8]))

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [4, 15]
    assert out["serverGetPoint"].tolist() == [0.25, 0.75]
    assert out["pointId"].tolist() == [3, 8]


def test_point0_totals_accept_shared_v335_report_shape():
    report = {
        "point0_base": 2,
        "point0_candidate": 3,
        "point0_additions": 1,
        "point0_removals": 0,
    }

    totals = _point0_totals(report)

    assert totals == {
        "anchor_point0_total": 2,
        "test_point0_total": 3,
        "test_point0_additions": 1,
        "test_point0_removals": 0,
    }
