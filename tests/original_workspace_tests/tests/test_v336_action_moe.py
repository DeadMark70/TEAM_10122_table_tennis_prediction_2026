import numpy as np
import pandas as pd

from analysis_v336_action_moe import build_rule_action_experts, select_by_budget


def test_budget_selector_limits_changed_rows():
    base = np.array([1, 1, 1, 1, 1])
    cand = np.array([2, 2, 1, 3, 4])
    utility = np.array([0.9, 0.1, 0.8, 0.7, 0.6])
    selected = select_by_budget(base, cand, utility, budget=2)
    assert (selected != base).sum() == 2
    assert selected.tolist() == [2, 1, 1, 3, 1]


def test_action_experts_do_not_output_serve_classes_by_default():
    frame = pd.DataFrame({"phase_id": ["receive", "third"], "lag0_actionId": [4, 1]})
    experts = build_rule_action_experts(frame, base_action=np.array([10, 3]))
    for pred in experts.values():
        assert not set(pred).intersection({15, 16, 17, 18})
