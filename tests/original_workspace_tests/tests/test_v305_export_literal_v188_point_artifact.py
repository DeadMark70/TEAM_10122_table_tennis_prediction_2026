import numpy as np

from analysis_v305_export_literal_v188_point_artifact import cap_residual_pred, normalize_rows_safe


def test_normalize_rows_safe_returns_finite_row_normalized_probabilities():
    x = np.array(
        [
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [np.nan, np.inf, -np.inf],
        ],
        dtype=float,
    )
    y = normalize_rows_safe(x)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert np.all(y >= 0.0)
    assert np.allclose(y.sum(axis=1), 1.0)


def test_cap_residual_pred_respects_cap_by_selecting_highest_margin_changed_rows():
    base = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    prob = np.array(
        [
            [0.90, 0.05, 0.05],  # change, margin 0.85
            [0.05, 0.90, 0.05],  # no change
            [0.05, 0.20, 0.75],  # change, margin 0.55
            [0.20, 0.35, 0.45],  # change, margin 0.10
            [0.05, 0.80, 0.15],  # no change
        ],
        dtype=float,
    )
    pred, changed = cap_residual_pred(base, prob, cap=0.4)
    assert changed.dtype == bool
    assert changed.tolist() == [True, False, True, False, False]
    assert pred.tolist() == [0, 1, 2, 1, 1]
