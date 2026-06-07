import numpy as np
import pandas as pd

from analysis_v292_weak_class_pretraining_action_teacher import (
    HARD_NEGATIVES,
    WEAK_CLASS_WEIGHTS,
    WEAK_GROUPS,
    apply_action_caps,
    blend_with_anchor_probs,
    hard_negative_mask,
    normalize_rows_safe,
    numeric_matrix,
    sample_weight_for_actions,
    train_weak_auxiliary_heads,
)


def test_weak_groups_and_hard_negatives_are_defined():
    assert WEAK_GROUPS["fast_attack_57"] == [5, 7]
    assert WEAK_GROUPS["terminal_03"] == [0, 3]
    assert 10 in HARD_NEGATIVES["short_control_411"]
    assert WEAK_CLASS_WEIGHTS[5] > 1.0


def test_sample_weight_for_actions_boosts_weak_classes():
    y = np.array([1, 5, 7, 10, 14])
    w = sample_weight_for_actions(y)
    assert w[1] > w[0]
    assert w[2] > w[0]
    assert w[4] > w[0]
    assert w[3] == 1.0


def test_normalize_rows_safe_repairs_bad_rows():
    matrix = np.array([[1.0, np.nan, -3.0], [0.0, 0.0, 0.0]])
    out = normalize_rows_safe(matrix)
    assert out.shape == matrix.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()
    assert np.allclose(out[1], np.ones(3) / 3)


def test_blend_with_anchor_probs_normalizes_and_preserves_shape():
    anchor = np.zeros((2, 19))
    anchor[:, 1] = 1.0
    teacher = np.ones((2, 19))
    out = blend_with_anchor_probs(anchor, teacher, weight=0.1)
    assert out.shape == (2, 19)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()


def test_apply_action_caps_blocks_serves_and_limits_to_allowed():
    proba = np.ones((3, 19)) / 19
    proba[:, 15:19] = 10.0
    capped = apply_action_caps(proba, allowed_override_actions=[5, 7])
    assert capped.shape == (3, 19)
    assert np.allclose(capped[:, 15:19], 0.0)
    assert np.allclose(capped.sum(axis=1), 1.0)


def test_numeric_matrix_aligns_categorical_columns():
    train = pd.DataFrame(
        {
            "prefix_len": [1, 2],
            "phase_bin": ["receive", "third"],
            "lag0_action_family": ["Attack", "Control"],
            "v292_aux_fast_attack_57_logistic": [0.1, 0.9],
        }
    )
    test = pd.DataFrame(
        {
            "prefix_len": [3],
            "phase_bin": ["rally"],
            "lag0_action_family": ["Defensive"],
            "v292_aux_fast_attack_57_logistic": [0.4],
        }
    )
    x_train, x_test = numeric_matrix(train, test)
    assert list(x_train.columns) == list(x_test.columns)
    assert x_train.shape[0] == 2
    assert x_test.shape[0] == 1
    assert "phase_bin_rally" in x_train.columns
    assert "lag0_action_family_Defensive" in x_train.columns


def test_hard_negative_mask_keeps_group_positives_and_hard_negatives():
    y = np.array([5, 7, 1, 2, 8, 14])
    mask = hard_negative_mask(y, "fast_attack_57")
    assert mask.tolist() == [True, True, True, True, False, False]


def test_train_weak_auxiliary_heads_returns_finite_aligned_scores():
    train_frame = pd.DataFrame(
        {
            "prefix_len": [1, 2, 3, 4, 5, 6, 7, 8],
            "phase_bin": ["receive", "third", "rally", "rally", "receive", "third", "rally", "rally"],
            "lag0_actionId": [1, 2, 4, 6, 10, 13, 5, 7],
            "scoreTotal": [0, 2, 4, 6, 8, 10, 12, 14],
        }
    )
    test_frame = pd.DataFrame(
        {
            "prefix_len": [2, 7],
            "phase_bin": ["third", "rally"],
            "lag0_actionId": [4, 9],
            "scoreTotal": [3, 11],
        }
    )
    rows = pd.DataFrame({"fold": [0, 1, 0, 1, 0, 1, 0, 1]})
    y = np.array([5, 1, 7, 2, 0, 3, 14, 8])
    aux_oof, aux_test, report = train_weak_auxiliary_heads(train_frame, rows, y, test_frame)
    assert "v292_aux_fast_attack_57_logistic" in aux_oof.columns
    assert "v292_aux_fast_attack_57_extratrees" in aux_oof.columns
    assert list(aux_oof.columns) == list(aux_test.columns)
    assert aux_oof.shape[0] == len(train_frame)
    assert aux_test.shape[0] == len(test_frame)
    assert np.isfinite(aux_oof.to_numpy()).all()
    assert np.isfinite(aux_test.to_numpy()).all()
    assert ((aux_oof >= 0.0) & (aux_oof <= 1.0)).all().all()
    assert ((aux_test >= 0.0) & (aux_test <= 1.0)).all().all()
    assert {"group", "model", "ap", "auc", "positive_rows", "hard_negative_rows", "test_mean"}.issubset(
        report.columns
    )
