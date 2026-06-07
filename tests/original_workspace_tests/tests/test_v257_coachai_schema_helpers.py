import numpy as np
import pandas as pd

from analysis_v257_coachai_schema_helpers import (
    BADMINTON_FAMILY_MAP,
    build_padding_mask,
    canonicalize_phase,
    forbid_ttmatch_path,
    normalize_xy,
    sequence_pad,
)


def test_forbid_ttmatch_path_rejects_overlap_source():
    try:
        forbid_ttmatch_path("external_data/TTMATCH/train.csv")
    except RuntimeError as exc:
        assert "TTMATCH" in str(exc)
    else:
        raise AssertionError("TTMATCH path should be rejected")


def test_canonicalize_phase_from_index():
    assert canonicalize_phase(0) == "serve_like"
    assert canonicalize_phase(1) == "receive_like"
    assert canonicalize_phase(2) == "third_ball_like"
    assert canonicalize_phase(3) == "fourth_ball_like"
    assert canonicalize_phase(7) == "rally_like"


def test_sequence_pad_and_mask():
    values = [3, 4, 5]
    padded = sequence_pad(values, max_len=5, pad_value=0)
    assert padded.tolist() == [3, 4, 5, 0, 0]
    mask = build_padding_mask(padded, pad_value=0)
    assert mask.tolist() == [1, 1, 1, 0, 0]


def test_normalize_xy_is_finite_and_bounded():
    x = pd.Series([0.0, 177.5, 355.0, np.nan])
    y = pd.Series([0.0, 240.0, 480.0, np.nan])
    nx, ny = normalize_xy(x, y)
    assert np.isfinite(nx).all()
    assert np.isfinite(ny).all()
    assert nx.min() >= -1.0 and nx.max() <= 1.0
    assert ny.min() >= -1.0 and ny.max() <= 1.0


def test_badminton_family_map_has_fallbacks():
    assert BADMINTON_FAMILY_MAP.get("clear") == "Defensive"
    assert BADMINTON_FAMILY_MAP.get("smash") == "Attack"
    assert BADMINTON_FAMILY_MAP.get("net shot") == "Control"
