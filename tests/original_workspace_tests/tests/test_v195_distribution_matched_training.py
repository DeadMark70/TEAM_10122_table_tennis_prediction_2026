import numpy as np
import pandas as pd

from analysis_v195_distribution_matched_point_gru import (
    distribution_match_weights,
    stratified_resample_indices,
)


def test_distribution_match_weights_upweights_test_heavy_bucket():
    train = pd.DataFrame({"bucket": ["short", "short", "long", "long", "long", "long"]})
    test = pd.DataFrame({"bucket": ["short", "long", "long", "long"]})
    w = distribution_match_weights(train, test, ["bucket"], clip=(0.1, 10.0))
    assert np.isclose(w.mean(), 1.0)
    assert w[train["bucket"].eq("long")].mean() > w[train["bucket"].eq("short")].mean()


def test_stratified_resample_indices_follows_test_bucket_distribution():
    train = pd.DataFrame({"bucket": ["a"] * 80 + ["b"] * 20})
    test = pd.DataFrame({"bucket": ["a"] * 10 + ["b"] * 90})
    idx = stratified_resample_indices(train, test, ["bucket"], n=200, seed=7)
    sampled = train.iloc[idx]["bucket"].value_counts(normalize=True).to_dict()
    assert sampled["b"] > 0.70
    assert sampled["a"] < 0.30
