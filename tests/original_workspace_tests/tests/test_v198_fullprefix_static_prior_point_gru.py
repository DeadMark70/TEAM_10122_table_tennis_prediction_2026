import numpy as np
import pandas as pd

from train_v198_fullprefix_static_prior_point_gru import slice_macro_f1


def test_slice_macro_f1_returns_nan_for_empty_slice_and_score_for_nonempty():
    rows = pd.DataFrame({"audit_phase": ["rally", "receive"]})
    y = np.array([0, 8])
    pred = np.array([0, 9])
    scores = slice_macro_f1(rows, y, pred)
    assert "phase_rally" in scores
    assert scores["phase_rally"] == 0.1
    assert np.isnan(scores["phase_fourth_ball"])
