import numpy as np
import pandas as pd

from analysis_v308_point0_row_ablation import build_subgroup_masks, build_subgroup_specs, detect_point0_changed_rows


def test_detect_point0_changed_rows_keeps_only_nonzero_to_zero_changes():
    base = pd.DataFrame(
        {
            "rally_uid": [10, 11, 12, 13],
            "actionId": [1, 1, 1, 1],
            "pointId": [8, 0, 5, 7],
            "serverGetPoint": [0.1, 0.2, 0.3, 0.4],
        }
    )
    candidate = base.copy()
    candidate["pointId"] = [0, 0, 4, 0]

    rows = detect_point0_changed_rows(base, candidate)

    assert rows["row_id"].tolist() == [0, 3]
    assert rows["rally_uid"].tolist() == [10, 13]
    assert rows["source_point"].tolist() == [8, 7]
    assert rows["candidate_point"].tolist() == [0, 0]


def test_build_subgroup_masks_constructs_requested_groups_and_leave_one_out():
    changed = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "source_point": [7, 8, 9, 4, 5],
            "model_p0_margin": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )

    masks = build_subgroup_masks(changed)

    assert masks["former_7_8_9_to_0"].tolist() == [True, True, True, False, False]
    assert masks["former_8_9_to_0"].tolist() == [False, True, True, False, False]
    assert masks["former_4_5_6_to_0"].tolist() == [False, False, False, True, True]
    assert masks["high_margin_top9"].tolist() == [True, True, True, True, True]
    assert masks["high_margin_top14"].tolist() == [True, True, True, True, True]
    assert masks["high_margin_top18"].tolist() == [True, True, True, True, True]
    assert masks["leave_source_7_out"].tolist() == [False, True, True, True, True]
    assert masks["leave_source_8_out"].tolist() == [True, False, True, True, True]
    assert masks["leave_source_9_out"].tolist() == [True, True, False, True, True]
    assert isinstance(masks["former_7_8_9_to_0"], np.ndarray)


def test_subgroup_oof_selectors_stay_inside_v306_oof_mask():
    changed = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "source_point": [7, 8, 9],
            "model_p0_margin": [0.9, 0.8, 0.7],
        }
    )
    oof_base = np.array([7, 8, 9, 8, 9])
    oof_margin = np.array([0.5, 0.4, 0.3, 0.9, 0.8])
    v306_oof_mask = np.array([True, True, True, False, False])

    specs = {
        spec.name: spec
        for spec in build_subgroup_specs(changed, oof_rows=5, test_rows=100, v306_oof_mask=v306_oof_mask)
    }

    leave7 = specs["leave_source_7_out"].oof_selector(oof_base, oof_margin)
    former89 = specs["former_8_9_to_0"].oof_selector(oof_base, oof_margin)

    assert leave7.tolist() == [False, True, True, False, False]
    assert former89.tolist() == [False, True, True, False, False]
