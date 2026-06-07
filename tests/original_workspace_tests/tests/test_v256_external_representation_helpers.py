import numpy as np
import pandas as pd

from analysis_v256_external_representation_helpers import (
    BIN_CLASSES,
    FAMILY_CLASSES,
    PHASE_CLASSES,
    ensure_probability_columns,
    external_target_frame,
    normalize_rows_safe,
    numeric_bin,
)


def test_numeric_bin_handles_missing_and_edges():
    values = pd.Series([0.0, 0.2, 0.8, np.nan])
    result = numeric_bin(values, bins=[0.25, 0.75], labels=["low", "mid", "high"])
    assert result.tolist() == ["low", "low", "high", "missing"]
    assert "missing" in BIN_CLASSES


def test_normalize_rows_safe_handles_zero_and_nan_rows():
    matrix = np.array([[1.0, 1.0], [0.0, 0.0], [np.nan, 3.0]])
    normalized = normalize_rows_safe(matrix)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert not np.isnan(normalized).any()
    assert np.allclose(normalized[1], [0.5, 0.5])


def test_external_target_frame_creates_supported_targets():
    corpus = pd.DataFrame(
        {
            "action_family": ["Attack", None, "Control"],
            "phase": ["receive_like", "rally_like", None],
            "terminal_like": [0, 1, 0],
            "speed": [0.1, 2.0, np.nan],
            "spin": [0.0, 8.0, np.nan],
            "landing_y": [0.1, 0.7, np.nan],
        }
    )
    targets = external_target_frame(corpus)
    assert set(targets["family"]).issubset(set(FAMILY_CLASSES))
    assert set(targets["phase"]).issubset(set(PHASE_CLASSES))
    assert targets["terminal"].tolist() == [0, 1, 0]
    assert targets["speed_bin"].iloc[2] == "missing"
    assert targets["spin_bin"].iloc[2] == "missing"
    assert targets["depth_bin"].iloc[2] == "missing"


def test_ensure_probability_columns_fills_missing_classes():
    frame = pd.DataFrame({"Attack": [0.8], "Control": [0.2]})
    result = ensure_probability_columns(frame, FAMILY_CLASSES)
    assert list(result.columns) == FAMILY_CLASSES
    assert np.allclose(result.sum(axis=1).to_numpy(), 1.0)
    assert result.loc[0, "Zero"] == 0.0
