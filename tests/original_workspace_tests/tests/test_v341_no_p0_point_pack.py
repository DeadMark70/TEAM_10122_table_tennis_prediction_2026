import numpy as np
import pandas as pd

from analysis_v341_no_p0_point_pack import no_p0_additions, pack_point_only, transition_counts


def test_pack_preserves_action_and_server():
    base = pd.DataFrame(
        {
            "rally_uid": ["a"],
            "actionId": [4],
            "pointId": [8],
            "serverGetPoint": [0.3],
        }
    )
    packed = pack_point_only(base, np.array([7]))
    assert packed["actionId"].tolist() == [4]
    assert packed["serverGetPoint"].tolist() == [0.3]
    assert packed["pointId"].tolist() == [7]


def test_no_p0_additions_detects_nonzero_to_zero():
    base = pd.Series([8, 0, 7])
    good = pd.Series([9, 0, 8])
    bad = pd.Series([0, 0, 8])
    assert no_p0_additions(base, good) is True
    assert no_p0_additions(base, bad) is False


def test_transition_counts_only_changed_rows():
    counts = transition_counts(pd.Series([8, 8, 7]), pd.Series([7, 8, 9]))
    assert counts == {"7->9": 1, "8->7": 1}
