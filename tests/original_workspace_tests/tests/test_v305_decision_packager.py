from analysis_v305_decision_packager import rank_key


def test_rank_key_prefers_review_then_delta_then_low_churn():
    a = {"decision": "REVIEW", "literal_delta": 0.002, "test_churn": 0.02}
    b = {"decision": "DO_NOT_UPLOAD", "literal_delta": 0.010, "test_churn": 0.01}
    c = {"decision": "REVIEW", "literal_delta": 0.0016, "test_churn": 0.005}
    assert rank_key(a) > rank_key(b)
    assert rank_key(a) > rank_key(c)


def test_rank_key_accepts_v261_column_names():
    a = {"decision": "REVIEW", "delta_vs_v188_cap5": 0.002, "test_churn_vs_v188_cap5": 0.02}
    b = {"decision": "REVIEW", "delta_vs_v188_cap5": 0.001, "test_churn_vs_v188_cap5": 0.01}
    assert rank_key(a) > rank_key(b)
