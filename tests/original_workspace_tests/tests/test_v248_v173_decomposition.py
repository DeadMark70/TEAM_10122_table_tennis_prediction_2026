import numpy as np
import pandas as pd

from analysis_v248_v173_decomposition_helpers import (
    acceptance_mask_by_phase,
    component_weight_grid,
    transition_counts,
)


def test_component_weight_grid_normalizes():
    grid = component_weight_grid()

    assert grid
    for rec in grid:
        assert abs(rec["external"] + rec["internal"] + rec["teacher"] - 1.0) < 1e-12
        assert rec["name"]


def test_acceptance_mask_by_phase_selects_only_requested_phase():
    rows = pd.DataFrame({"phase": ["receive", "third_ball", "rally"]})
    changed = np.array([True, True, True])

    mask = acceptance_mask_by_phase(rows, changed, ["receive", "rally"], phase_col="phase")

    assert mask.tolist() == [True, False, True]


def test_transition_counts_sorts_descending():
    frame = pd.DataFrame({"phase": ["a", "a", "b"], "base": [1, 1, 2], "teacher": [3, 3, 4]})

    out = transition_counts(frame, "phase", "base", "teacher")

    assert out.iloc[0]["rows"] == 2
    assert list(out.columns) == ["phase", "base", "teacher", "rows"]
