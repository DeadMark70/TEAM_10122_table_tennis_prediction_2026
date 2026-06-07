from pathlib import Path

import pandas as pd

from analysis_v354_independent_row_evidence import (
    build_row_evidence,
    point_depth,
    point_side,
    run_pipeline,
)


def test_geometry_features_for_point_grid_are_correct():
    assert point_depth(0) == -1
    assert point_depth(1) == 0
    assert point_depth(6) == 1
    assert point_depth(9) == 2
    assert point_side(0) == -1
    assert point_side(1) == 0
    assert point_side(5) == 1
    assert point_side(9) == 2

    features = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "strikeNumber": [1, 3],
            "actionId": [15, 4],
            "pointId": [7, 8],
        }
    )
    bank = pd.DataFrame(
        {
            "row_id": [0, 1],
            "rally_uid": ["a", "b"],
            "anchor_value": [7, 8],
            "candidate_value": [9, 0],
            "source": ["submission_a", "submission_b"],
            "source_dir": ["dir_a", "dir_b"],
            "source_public_tag": ["v338_family_support", "historical_point_model"],
            "changed_in_v338": [True, False],
        }
    )

    evidence = build_row_evidence(features, bank, train=None, v341_extra_transitions=set())

    first = evidence.loc[evidence["row_id"].eq(0)].iloc[0]
    assert first["same_depth"] is True
    assert first["depth_delta"] == 0
    assert first["side_delta"] == 2
    assert first["terminal_transition"] is True
    assert "independent_evidence_score" in evidence.columns

    second = evidence.loc[evidence["row_id"].eq(1)].iloc[0]
    assert second["point0_addition"] is True
    assert second["no_p0_swap"] is False


def test_source_agreement_counts_unique_source_directories():
    features = pd.DataFrame({"rally_uid": ["a"], "strikeNumber": [2], "actionId": [3], "pointId": [8]})
    bank = pd.DataFrame(
        {
            "row_id": [0, 0, 0],
            "rally_uid": ["a", "a", "a"],
            "anchor_value": [8, 8, 8],
            "candidate_value": [9, 9, 9],
            "source": ["s1", "s2", "s3"],
            "source_dir": ["dir_a", "dir_a", "dir_b"],
            "source_public_tag": ["v338_family_support", "public_risk_probe", "historical_point_model"],
            "changed_in_v338": [False, True, False],
        }
    )

    row = build_row_evidence(features, bank, train=None, v341_extra_transitions={"8->9"}).iloc[0]

    assert row["source_dir_count"] == 2
    assert row["source_count"] == 3
    assert row["source_public_tag_safe_count"] == 2
    assert row["source_public_tag_risk_count"] == 1
    assert row["changed_in_v338"] is True
    assert row["v341_extra_like_transition"] is True


def test_rows_can_be_produced_without_train_using_source_evidence_only(tmp_path: Path):
    root = tmp_path
    outdir = root / "v354_independent_row_evidence"
    (root / "v306_point0_addition_probe").mkdir()
    (root / "v338_joint_moe_pack").mkdir()
    (root / "v343_row_candidate_bank").mkdir()

    test_new = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "strikeNumber": [1, 2],
            "actionId": [15, 4],
            "pointId": [7, 8],
        }
    )
    submission = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [15, 4],
            "pointId": [7, 9],
            "serverGetPoint": [0.1, 0.2],
        }
    )
    bank = pd.DataFrame(
        {
            "row_id": [1],
            "rally_uid": ["b"],
            "anchor_value": [8],
            "candidate_value": [9],
            "source": ["submission_x"],
            "source_dir": ["source_a"],
            "source_public_tag": ["v338_family_support"],
            "changed_in_v338": [True],
        }
    )

    test_new.to_csv(root / "test_new.csv", index=False)
    submission.to_csv(root / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv", index=False)
    submission.to_csv(
        root / "v338_joint_moe_pack" / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv",
        index=False,
    )
    bank.to_csv(root / "v343_row_candidate_bank" / "candidate_bank.csv", index=False)

    report = run_pipeline(root=root, outdir=outdir, expected_rows=None)
    evidence = pd.read_csv(outdir / "row_evidence.csv")
    summary = pd.read_csv(outdir / "evidence_summary.csv")

    assert report["decision"] == "REPORTS_EXPORTED"
    assert report["train_available"] is False
    assert len(evidence) == 1
    assert len(summary) >= 1
    assert evidence.loc[0, "support_count_phase_action_old_new"] == 0
    assert evidence.loc[0, "source_dir_count"] == 1
    assert bool(evidence.loc[0, "changed_in_v338"]) is False
