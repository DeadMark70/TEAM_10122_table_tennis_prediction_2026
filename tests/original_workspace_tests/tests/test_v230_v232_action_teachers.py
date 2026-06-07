import numpy as np
import pandas as pd

from analysis_v230_action_soft_teacher_factory import (
    ACTION_FAMILY_TO_IDS,
    apply_family_calibration,
    geometric_log_blend,
    normalize_rows_safe,
    public_like_slice_score,
)
from train_v231_action_only_sequence_teacher import (
    action_family_targets,
    action_only_loss,
)
from analysis_v232_v173_curriculum_deepening import (
    build_family_prior_matrix,
    apply_curriculum_family_prior,
)


def test_geometric_log_blend_normalizes_and_moves_probability():
    anchor = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=float)
    teacher = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=float)
    out = geometric_log_blend(anchor, teacher, 0.25)

    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 1] > anchor[0, 1]
    assert np.isfinite(out).all()


def test_family_calibration_preserves_rows_and_boosts_family():
    prob = np.full((1, 19), 1 / 19, dtype=float)
    family_prior = {"Attack": 0.80, "Control": 0.05, "Defensive": 0.05, "Zero": 0.05, "Serve": 0.05}
    out = apply_family_calibration(prob, [family_prior], weight=0.5)
    attack_mass = out[:, ACTION_FAMILY_TO_IDS["Attack"]].sum()
    control_mass = out[:, ACTION_FAMILY_TO_IDS["Control"]].sum()

    assert np.allclose(out.sum(axis=1), 1.0)
    assert attack_mass > control_mass


def test_public_like_slice_score_weights_test_like_rows():
    y = np.array([1, 1, 5, 5])
    good = np.array([1, 1, 5, 5])
    bad = np.array([5, 5, 1, 1])
    rows = pd.DataFrame({"prefix_len": [1, 2, 4, 5], "audit_phase": ["receive", "third_ball", "rally", "rally"]})

    assert public_like_slice_score(y, good, rows) > public_like_slice_score(y, bad, rows)


def test_action_family_targets_maps_all_classes():
    y = np.array([0, 1, 8, 12, 15])
    assert action_family_targets(y).tolist() == [0, 1, 2, 3, 4]


def test_action_only_loss_accepts_auxiliary_logits():
    logits = {
        "action": np.zeros((3, 19), dtype=float),
        "family": np.zeros((3, 5), dtype=float),
        "weak": np.zeros((3, 8), dtype=float),
    }
    y = np.array([0, 5, 12])
    loss = action_only_loss(logits, y, kd_teacher=None)
    assert loss > 0
    assert np.isfinite(loss)


def test_curriculum_family_prior_matrix_and_application():
    rows = pd.DataFrame(
        {
            "phase": ["receive", "rally"],
            "current_family": ["serve", "attack"],
            "next_family_attack": [0.2, 0.7],
            "next_family_control": [0.7, 0.1],
            "next_family_defensive": [0.05, 0.15],
            "next_family_serve": [0.0, 0.0],
            "next_family_unknown": [0.05, 0.05],
        }
    )
    contexts = pd.DataFrame({"phase": ["receive"], "current_family": ["serve"]})
    prior = build_family_prior_matrix(rows, contexts)
    prob = np.full((1, 19), 1 / 19, dtype=float)
    out = apply_curriculum_family_prior(prob, prior, weight=0.5)

    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[:, ACTION_FAMILY_TO_IDS["Control"]].sum() > out[:, ACTION_FAMILY_TO_IDS["Defensive"]].sum()
