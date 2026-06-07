from analysis_v305_point_residual_retest_suite import classify_candidate


def test_classify_candidate_requires_material_oof_gain():
    assert classify_candidate(delta=0.0004, churn=0.01, point0_add=0) == "DO_NOT_UPLOAD"
    assert classify_candidate(delta=0.0017, churn=0.02, point0_add=1) == "REVIEW"
    assert classify_candidate(delta=0.0020, churn=0.08, point0_add=0) == "DO_NOT_UPLOAD"


def test_classify_candidate_rejects_excess_point0_additions():
    assert classify_candidate(delta=0.002, churn=0.01, point0_add=5) == "DO_NOT_UPLOAD"
