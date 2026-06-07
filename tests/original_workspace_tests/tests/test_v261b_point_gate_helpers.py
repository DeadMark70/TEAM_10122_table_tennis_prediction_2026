import numpy as np

from analysis_v261b_direct_v188_point_gate import (
    cap_changed_rows,
    is_direct_v188_oof_artifact,
    per_class_f1_delta,
    point0_rate,
)


def test_cap_changed_rows_selects_top_fraction():
    scores = np.array([0.1, 0.9, 0.4, 0.8, 0.2])
    mask = cap_changed_rows(scores, cap=0.4)
    assert mask.tolist() == [False, True, False, True, False]


def test_point0_rate():
    assert point0_rate(np.array([0, 0, 1, 8])) == 0.5


def test_per_class_f1_delta_has_all_classes():
    y = np.array([0, 1, 1, 2, 2, 2])
    base = np.array([0, 1, 2, 2, 2, 1])
    cand = np.array([0, 1, 1, 2, 2, 1])
    out = per_class_f1_delta(y, base, cand, classes=[0, 1, 2])
    assert list(out["class_id"]) == [0, 1, 2]
    assert "delta_f1" in out.columns


def test_direct_v188_oof_artifact_rejects_bias_grid_csv():
    assert not is_direct_v188_oof_artifact("v192_v188_generalization_audit/v192_oof_point0_bias_grid.csv")
    assert is_direct_v188_oof_artifact("v188_point_intent_gru/v188_r186_w005_cap5_point_oof.npy")
