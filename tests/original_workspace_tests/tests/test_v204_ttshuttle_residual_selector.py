import numpy as np
import pandas as pd

from analysis_v204_ttshuttle_residual_selector import (
    build_candidate_change_frame,
    select_changes_by_trust,
    point0_transition_kind,
)


def test_point0_transition_kind_labels_direction():
    assert point0_transition_kind(8, 0) == "to_point0"
    assert point0_transition_kind(0, 8) == "from_point0"
    assert point0_transition_kind(7, 8) == "nonterminal"
    assert point0_transition_kind(0, 0) == "unchanged"


def test_build_candidate_change_frame_marks_correct_residual_changes():
    rows = pd.DataFrame({"prefix_len": [1, 3, 4]})
    base = np.array([8, 8, 0])
    neural = np.array([0, 7, 8])
    truth = np.array([0, 8, 8])
    base_prob = np.array(
        [
            [0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.0],
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.70, 0.0],
            [0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.0],
        ]
    )
    neural_prob = np.array(
        [
            [0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.0],
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.10, 0.0],
            [0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70, 0.0],
        ]
    )
    frame = build_candidate_change_frame(rows, base, neural, truth, base_prob, neural_prob)
    assert frame["is_correct_change"].tolist() == [1, 0, 1]
    assert frame["transition_kind"].tolist() == ["to_point0", "nonterminal", "from_point0"]
    assert np.allclose(frame["neural_margin"].to_numpy(), [0.8, 0.7, 0.4])


def test_select_changes_by_trust_respects_gate_and_cap():
    base = np.array([8, 8, 8, 8])
    neural = np.array([0, 7, 0, 7])
    trust = np.array([0.9, 0.8, 0.7, 0.6])
    allow = np.array([True, False, True, True])
    out, changed = select_changes_by_trust(base, neural, trust, max_churn=0.5, allow_mask=allow)
    assert changed.tolist() == [True, False, True, False]
    assert out.tolist() == [0, 8, 0, 8]
