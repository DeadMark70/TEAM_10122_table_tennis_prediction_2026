import numpy as np
import pandas as pd
import pytest

from analysis_v321_server_robust_rankblend import (
    ServerSource,
    apply_temperature,
    blend_to_target_mad,
    build_candidate,
    build_packaged_submission,
    decision_for_candidate,
    direction_agreement,
    rank_normalize_to_anchor,
    validate_output_path,
)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [101, 102, 103, 104],
            "actionId": [3, 7, 12, 18],
            "pointId": [0, 8, 4, 0],
            "serverGetPoint": [0.20, 0.80, 0.50, 0.30],
        }
    )


def _sources() -> list[ServerSource]:
    return [
        ServerSource("s1", "v319", np.array([30.0, 20.0, 40.0, 10.0]), 0.03, "p1"),
        ServerSource("s2", "v300", np.array([35.0, 25.0, 45.0, 15.0]), 0.01, "p2"),
        ServerSource("s3", "v302", np.array([10.0, 40.0, 30.0, 20.0]), -0.01, "p3"),
    ]


def test_rank_and_temperature_transforms_are_finite_ordered_and_target_mad_limited():
    anchor = np.array([0.2, 0.8, 0.5, 0.3])
    source = np.array([40.0, 10.0, 30.0, 20.0])

    ranked = rank_normalize_to_anchor(source, anchor)
    blended = blend_to_target_mad(anchor, ranked, target_mad=0.05)
    softened = apply_temperature(anchor, temperature=1.1)

    assert ranked.tolist() == pytest.approx([0.8, 0.2, 0.5, 0.3])
    assert float(np.mean(np.abs(blended - anchor))) == pytest.approx(0.05)
    assert np.isfinite(softened).all()
    assert ((softened > 0.0) & (softened < 1.0)).all()
    assert softened[0] > anchor[0]
    assert softened[1] < anchor[1]


def test_direction_agreement_counts_sources_that_move_same_way_as_target():
    anchor = np.array([0.2, 0.8, 0.5, 0.3])
    matrix = np.column_stack([rank_normalize_to_anchor(s.server, anchor) for s in _sources()])
    target = np.array([0.7, 0.25, 0.6, 0.28])

    agreed = direction_agreement(anchor, target, matrix)

    assert agreed.tolist() == [2, 2, 2, 2]


def test_build_candidate_preserves_server_only_anchor_and_hits_target_mad():
    anchor = _anchor()
    candidate = build_candidate(
        "unit_rankblend",
        anchor["serverGetPoint"].to_numpy(dtype=float),
        _sources(),
        target_mad=0.002,
        kind="rankblend",
        filename="submission_unit.csv",
    )
    packaged = build_packaged_submission(anchor, candidate, expected_rows=4)

    assert packaged[["rally_uid", "actionId", "pointId"]].equals(
        anchor[["rally_uid", "actionId", "pointId"]]
    )
    assert float(
        np.mean(
            np.abs(
                packaged["serverGetPoint"].to_numpy(dtype=float)
                - anchor["serverGetPoint"].to_numpy(dtype=float)
            )
        )
    ) == pytest.approx(0.002)
    assert candidate.min_agree_sources >= 2
    assert candidate.source_count == 3


def test_decision_requires_server_only_mad_limit_sane_distribution_and_two_source_agreement():
    assert (
        decision_for_candidate(
            action_changed_rows=0,
            point_changed_rows=0,
            mad=0.005,
            server_min=0.1,
            server_max=0.9,
            min_agree_sources=2,
            source_family_count=2,
            evidence_delta=0.01,
        )
        == "REVIEW_SERVER"
    )
    assert (
        decision_for_candidate(
            action_changed_rows=1,
            point_changed_rows=0,
            mad=0.001,
            server_min=0.1,
            server_max=0.9,
            min_agree_sources=2,
            source_family_count=2,
            evidence_delta=0.01,
        )
        == "DIAGNOSTIC"
    )
    assert (
        decision_for_candidate(
            action_changed_rows=0,
            point_changed_rows=0,
            mad=0.006,
            server_min=0.1,
            server_max=0.9,
            min_agree_sources=2,
            source_family_count=2,
            evidence_delta=0.01,
        )
        == "DIAGNOSTIC"
    )
    assert (
        decision_for_candidate(
            action_changed_rows=0,
            point_changed_rows=0,
            mad=0.001,
            server_min=0.5,
            server_max=0.5,
            min_agree_sources=2,
            source_family_count=2,
            evidence_delta=0.01,
        )
        == "DIAGNOSTIC"
    )
    assert (
        decision_for_candidate(
            action_changed_rows=0,
            point_changed_rows=0,
            mad=0.001,
            server_min=0.1,
            server_max=0.9,
            min_agree_sources=1,
            source_family_count=2,
            evidence_delta=0.01,
        )
        == "DIAGNOSTIC"
    )


def test_validate_output_path_rejects_upload_selected_ttmatch_and_old_server_targets():
    validate_output_path("v321_server_robust_rankblend/submission.csv")

    for path in [
        "upload_candidates_20260519/submission.csv",
        "submissions/selected/submission.csv",
        "external_data/TTMATCH/train.csv",
        "v321_server_robust_rankblend/old-server.csv",
    ]:
        with pytest.raises(ValueError):
            validate_output_path(path)
