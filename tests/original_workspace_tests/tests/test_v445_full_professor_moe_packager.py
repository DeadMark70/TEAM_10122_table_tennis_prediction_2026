import pandas as pd

from analysis_v445_full_professor_moe_packager import (
    package_professor_candidates,
    rank_professor_sources,
)


def test_v445_ranks_sources_with_oof_and_safety_above_raw_confidence():
    rows = pd.DataFrame(
        {
            "source": ["safe_oof", "raw_confident"],
            "oof_delta": [0.002, -0.01],
            "changed_rows": [10, 10],
            "point0_additions": [0, 0],
            "serve_additions": [0, 0],
            "confidence": [0.6, 0.99],
        }
    )

    ranked = rank_professor_sources(rows)

    assert ranked.iloc[0]["source"] == "safe_oof"


def test_v445_packager_preserves_server_and_blocks_unsafe_changes():
    anchor = pd.DataFrame(
        {"rally_uid": ["r1"], "actionId": [4], "pointId": [5], "serverGetPoint": [0.8]}
    )
    candidates = pd.DataFrame({"rally_uid": ["r1"], "candidate_pointId": [0], "utility": [9.0]})

    submission, report = package_professor_candidates(anchor, point_candidates=candidates, point_top=1)

    assert submission.loc[0, "pointId"] == 5
    assert submission.loc[0, "serverGetPoint"] == 0.8
    assert report["point"]["blocked_point0_additions"] == 1


def test_v445_packager_blocks_serve_15_18_additions_and_risky_sources():
    anchor = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "actionId": [4, 10],
            "pointId": [5, 7],
            "serverGetPoint": [0.8, 0.2],
        }
    )
    action_candidates = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "candidate_actionId": [16, 3],
            "utility": [10.0, 9.0],
            "source": ["v444_safe", "v438_sony_nd_audit_only/bad"],
        }
    )

    submission, report = package_professor_candidates(anchor, action_candidates=action_candidates, action_top=2)

    assert submission["actionId"].tolist() == [4, 10]
    assert report["action"]["blocked_serve_additions"] == 1
    assert report["action"]["blocked_risky_sources"] == 1
    assert report["risky_source_rows"] == 1


def test_v445_packager_reports_source_families_and_changed_rows():
    anchor = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "actionId": [4, 10],
            "pointId": [5, 7],
            "serverGetPoint": [0.8, 0.2],
        }
    )
    point_candidates = pd.DataFrame(
        {
            "rally_uid": ["r2"],
            "candidate_pointId": [8],
            "utility": [2.0],
            "source": ["v442_intent_first_sequence_point"],
        }
    )

    submission, report = package_professor_candidates(anchor, point_candidates=point_candidates, point_top=1)

    assert submission["pointId"].tolist() == [5, 8]
    assert report["total_changed_rows"] == 1
    assert report["source_families"] == {"v442_intent_first_sequence_point": 1}
