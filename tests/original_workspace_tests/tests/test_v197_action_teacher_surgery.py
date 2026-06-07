import numpy as np
import pandas as pd

from analysis_v197_action_teacher_surgery import apply_action_gate, transition_gate


def test_apply_action_gate_only_changes_masked_rows():
    base = np.array([1, 2, 3, 4])
    source = np.array([9, 9, 9, 9])
    mask = np.array([True, False, True, False])
    out = apply_action_gate(base, source, mask)
    assert out.tolist() == [9, 2, 9, 4]


def test_transition_gate_filters_phase_and_transition_pairs():
    rows = pd.DataFrame({"audit_phase": ["receive", "receive", "third_ball"], "lag0_pointId": [1, 8, 8]})
    base = np.array([4, 7, 1])
    source = np.array([10, 10, 3])
    mask = transition_gate(rows, base, source, phases={"receive"}, pairs={(4, 10), (7, 11)}, short_only=True)
    assert mask.tolist() == [True, False, False]
