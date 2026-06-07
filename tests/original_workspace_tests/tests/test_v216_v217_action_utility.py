import numpy as np

from analysis_v216_terminal_action_tuner import (
    build_terminal_action_candidate,
    terminal_action_scores,
)
from analysis_v217_macro_f1_utility_reranker import (
    class_f1_delta_for_change,
    expected_macro_f1_delta,
    row_delta_tables,
)


def test_terminal_action_scores_penalizes_action0_on_nonterminal_point():
    anchor = np.array([0, 10, 15, 1])
    point = np.array([8, 0, 8, 0])
    scores = terminal_action_scores(anchor, point)
    assert scores[0, 0] < 0
    assert scores[1, 0] > 0
    assert scores[2, 15] < 0
    assert scores[3, 0] > 0


def test_build_terminal_action_candidate_uses_best_allowed_action():
    anchor = np.array([0, 10, 15])
    point = np.array([8, 0, 8])
    prior = np.full((3, 19), 0.01)
    prior[0, 10] = 0.6
    prior[1, 0] = 0.7
    prior[2, 1] = 0.5
    candidate, gain = build_terminal_action_candidate(anchor, point, prior)
    assert candidate.tolist() == [10, 0, 1]
    assert gain[0] > 0
    assert gain[1] > 0
    assert gain[2] > 0


def test_class_f1_delta_for_change_rewards_rare_true_positive():
    y = np.array([1, 1, 2, 2, 2])
    pred = np.array([1, 2, 2, 1, 1])
    delta = class_f1_delta_for_change(y, pred, row=3, new_label=2, labels=[1, 2])
    assert delta > 0


def test_expected_macro_f1_delta_combines_correct_gain_and_wrong_loss():
    gain_correct = np.array([0.4, 0.2])
    loss_wrong = np.array([0.1, 0.5])
    p = np.array([0.75, 0.25])
    out = expected_macro_f1_delta(p, gain_correct, loss_wrong)
    assert out[0] > 0
    assert out[1] < 0


def test_row_delta_tables_uses_candidate_row_ids():
    y = np.array([1, 2, 2])
    anchor = np.array([1, 1, 1])
    candidate_row_ids = np.array([1, 2])
    candidate_actions = np.array([2, 2])
    gain, loss = row_delta_tables(y, anchor, candidate_row_ids, candidate_actions)
    assert gain.shape == (2,)
    assert loss.shape == (2,)
    assert gain[0] > 0
