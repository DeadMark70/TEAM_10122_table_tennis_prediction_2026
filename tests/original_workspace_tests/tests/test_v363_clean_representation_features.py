def test_drop_label_like_external_columns():
    import pandas as pd
    from analysis_v363_clean_representation_features import drop_label_like_columns

    df = pd.DataFrame({
        "safe_feature": [1, 2],
        "actionId": [3, 4],
        "pointId": [8, 9],
        "serverGetPoint": [0, 1],
        "family_prior": [0.2, 0.8],
    })
    out = drop_label_like_columns(df)
    assert "safe_feature" in out.columns
    assert "family_prior" in out.columns
    assert "actionId" not in out.columns
    assert "pointId" not in out.columns
    assert "serverGetPoint" not in out.columns


def test_prefix_feature_bins_are_stable():
    from analysis_v363_clean_representation_features import prefix_len_bin

    assert prefix_len_bin(1) == "1"
    assert prefix_len_bin(2) == "2"
    assert prefix_len_bin(3) == "3"
    assert prefix_len_bin(5) == "4_6"
    assert prefix_len_bin(8) == "7p"


def test_high_cardinality_object_columns_are_frequency_encoded():
    import pandas as pd
    from analysis_v363_clean_representation_features import control_high_cardinality_objects

    df = pd.DataFrame({
        "small_category": ["a", "b", "a"],
        "big_category": [f"id_{i}" for i in range(3)],
        "numeric_feature": [1, 2, 3],
    })

    out, report = control_high_cardinality_objects(df, max_unique=2)

    assert "small_category" in out.columns
    assert "big_category" not in out.columns
    assert "big_category_freq" in out.columns
    assert list(out["big_category_freq"]) == [1, 1, 1]
    assert report[0]["feature"] == "big_category"
    assert report[0]["action"] == "frequency_encoded"


def test_context_features_drop_exact_labels_but_keep_coarse_features():
    import pandas as pd
    from analysis_v363_clean_representation_features import build_context_features, fit_aicup_priors

    train = pd.DataFrame({
        "rally_uid": [1, 1, 2],
        "match": [10, 10, 11],
        "strikeNumber": [1, 2, 1],
        "scoreSelf": [0, 1, 3],
        "scoreOther": [0, 0, 3],
        "strengthId": [2, 1, 2],
        "spinId": [3, 2, 5],
        "positionId": [1, 2, 3],
        "pointId": [9, 5, 4],
        "actionId": [15, 12, 10],
        "serverGetPoint": [0, 1, 0],
    })
    priors = fit_aicup_priors(train)

    out = build_context_features(train, "train", priors)

    assert not {"actionId", "pointId", "serverGetPoint"}.intersection(out.columns)
    assert {"lag0_action_family", "lag0_point_depth", "lag0_point_side"}.issubset(out.columns)
    assert "aicup_family_point_depth_prior" in out.columns
