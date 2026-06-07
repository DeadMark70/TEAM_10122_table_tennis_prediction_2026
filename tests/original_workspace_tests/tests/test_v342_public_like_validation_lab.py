import pandas as pd

from analysis_v342_public_like_validation_lab import (
    density_ratio_from_counts,
    historical_sanity,
    point_candidate_audit,
    transition_counts,
)


def test_density_weight_is_clipped():
    train_counts = {"a": 10, "b": 1}
    test_counts = {"a": 5, "b": 20}
    weights = density_ratio_from_counts(["a", "b"], train_counts, test_counts, clip=(0.5, 3.0))
    assert weights.tolist() == [0.5, 3.0]


def test_transition_counts_are_stable():
    base = pd.Series([8, 8, 7, 0])
    cand = pd.Series([7, 8, 9, 0])
    assert transition_counts(base, cand) == {"7->9": 1, "8->7": 1}


def test_historical_sanity_orders_known_public_results():
    records = [
        {"version": "V338", "public_delta": 0.0012136},
        {"version": "V341", "public_delta": 0.0003196},
        {"version": "V191", "public_delta": -0.0064370},
    ]
    report = historical_sanity(records)
    assert report["v338_above_v341"] is True
    assert report["positive_above_v191"] is True


def test_point_candidate_audit_reports_v338_overlap_and_family():
    base = pd.DataFrame(
        {
            "rally_uid": ["a", "b", "c", "d"],
            "actionId": [1, 1, 1, 1],
            "pointId": [8, 8, 7, 0],
            "serverGetPoint": [0.1, 0.2, 0.3, 0.4],
        }
    )
    public_anchor = base.copy()
    public_anchor["pointId"] = [7, 8, 7, 0]
    cand = base.copy()
    cand["pointId"] = [7, 0, 9, 8]

    audit = point_candidate_audit(base, public_anchor, cand)

    assert audit["point_churn_vs_v306"] == 4
    assert audit["point_churn_vs_v338"] == 3
    assert audit["point0_additions"] == 1
    assert audit["point0_removals"] == 1
    assert audit["overlap_with_v338_changed_rows"] == 1
    assert audit["new_rows_beyond_v338"] == 3
    assert audit["family"] == "mixed"
