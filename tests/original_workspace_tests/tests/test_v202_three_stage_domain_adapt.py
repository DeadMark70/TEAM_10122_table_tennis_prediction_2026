import numpy as np
import pandas as pd

from train_v202_three_stage_domain_adapt import adapter_sample_weights, build_adapter_features


def test_adapter_sample_weights_mix_testlike_and_replay():
    rows = pd.DataFrame({"bucket": ["a", "a", "b", "b"]})
    test = pd.DataFrame({"bucket": ["b", "b", "b", "a"]})
    w = adapter_sample_weights(rows, test, ["bucket"], testlike_share=0.7)
    assert np.isclose(w.mean(), 1.0)
    assert w[rows["bucket"].eq("b")].mean() > w[rows["bucket"].eq("a")].mean()


def test_build_adapter_features_keeps_train_pred_columns_aligned():
    rows = pd.DataFrame({"prefix_len": [1, 3], "audit_phase": ["receive", "rally"]})
    pred_rows = pd.DataFrame({"prefix_len": [2], "audit_phase": ["third_ball"]})
    base = np.ones((2, 2)) / 2
    stage = np.array([[0.8, 0.2], [0.3, 0.7]])
    base_pred = np.ones((1, 2)) / 2
    stage_pred = np.array([[0.4, 0.6]])
    x_train, x_pred = build_adapter_features(rows, pred_rows, base, stage, base_pred, stage_pred)
    assert list(x_train.columns) == list(x_pred.columns)
    assert x_train.shape[0] == 2
    assert x_pred.shape[0] == 1
