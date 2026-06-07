import numpy as np
import pandas as pd

from analysis_v199_point_calibration_sweep import apply_point_changes, point_gate


def test_apply_point_changes_keeps_anchor_outside_mask():
    anchor = np.array([0, 8, 9])
    source = np.array([0, 0, 7])
    mask = np.array([False, True, False])
    assert apply_point_changes(anchor, source, mask).tolist() == [0, 0, 9]


def test_point_gate_selects_domain_shift_rows():
    rows = pd.DataFrame(
        {
            "prefix_len": [1, 4, 2],
            "audit_phase": ["receive", "rally", "third_ball"],
            "audit_lag0_depth": ["short", "long", "half"],
        }
    )
    assert point_gate(rows, "domain_shift").tolist() == [False, True, False]
    assert point_gate(rows, "not_receive").tolist() == [False, True, True]
