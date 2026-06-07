import json
from pathlib import Path

import pandas as pd

from analysis_v412_build_clean_canonical_external import CANONICAL_COLUMNS, build_canonical, run_pipeline


def _manifest(rows):
    return pd.DataFrame(rows)


def test_build_canonical_keeps_only_allowed_sources(tmp_path):
    root = tmp_path / "external_data"
    allowed = root / "TT3D" / "evaluation" / "3D_gt"
    blocked = root / "TTMATCH"
    allowed.mkdir(parents=True)
    blocked.mkdir(parents=True)
    (allowed / "001.csv").write_text("Timestamp,X,Y,Z\n0.1,1,2,3\n", encoding="utf-8")
    (blocked / "train.csv").write_text("rally_uid,actionId,pointId\n1,2,3\n", encoding="utf-8")
    manifest = _manifest(
        [
            {
                "source_dataset": "TT3D",
                "path": str(allowed / "001.csv"),
                "relative_path": "external_data/TT3D/evaluation/3D_gt/001.csv",
                "license_tag": "CC-BY-4.0",
                "risk_tier": "clean_physics",
                "allowed_first_version": True,
                "extension": ".csv",
                "columns_json": json.dumps(["Timestamp", "X", "Y", "Z"]),
            },
            {
                "source_dataset": "TTMATCH",
                "path": str(blocked / "train.csv"),
                "relative_path": "external_data/TTMATCH/train.csv",
                "license_tag": "excluded_overlap_risk",
                "risk_tier": "excluded_overlap_risk",
                "allowed_first_version": False,
                "extension": ".csv",
                "columns_json": json.dumps(["rally_uid", "actionId", "pointId"]),
            },
        ]
    )

    canonical, report = build_canonical(manifest)

    assert set(canonical["source_dataset"]) == {"TT3D"}
    assert report["blocked_manifest_rows"] == 1


def test_coachai_labels_are_prefixed_coarse_not_exact_action_ids(tmp_path):
    csv_path = tmp_path / "coach.csv"
    csv_path.write_text(
        "rally,ball_round,player,type,landing_area,landing_x,landing_y,lose_reason,getpoint_player\n"
        "7,1,A,發短球,2,1.5,3.2,,\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        [
            {
                "source_dataset": "CoachAI-Projects-main",
                "path": str(csv_path),
                "relative_path": "external_data/CoachAI-Projects-main/coach.csv",
                "license_tag": "MIT_repo_citation_required",
                "risk_tier": "clean_cross_sport_coarse",
                "allowed_first_version": True,
                "extension": ".csv",
                "columns_json": json.dumps(["rally", "ball_round", "type", "landing_x", "landing_y"]),
            }
        ]
    )

    canonical, _ = build_canonical(manifest)

    assert canonical.iloc[0]["coarse_family"].startswith("badminton_")
    assert canonical.iloc[0]["coarse_family"] != "15"


def test_deepmind_norms_are_finite_when_components_exist(tmp_path):
    json_path = tmp_path / "rallies.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "id": 10,
                    "pos_x": 0.1,
                    "pos_y": 0.2,
                    "pos_z": 0.3,
                    "vel_x": 3.0,
                    "vel_y": 4.0,
                    "vel_z": 0.0,
                    "w_vel_x": 0.0,
                    "w_vel_y": 0.0,
                    "w_vel_z": 5.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = _manifest(
        [
            {
                "source_dataset": "DeepMindrobottabletennis",
                "path": str(json_path),
                "relative_path": "external_data/DeepMindrobottabletennis/rallies.json",
                "license_tag": "CC-BY-4.0-data_Apache-2.0-code",
                "risk_tier": "clean_physics",
                "allowed_first_version": True,
                "extension": ".json",
                "columns_json": json.dumps(["pos_x", "pos_y", "pos_z"]),
            }
        ]
    )

    canonical, _ = build_canonical(manifest)

    assert float(canonical.iloc[0]["speed_norm"]) == 5.0
    assert float(canonical.iloc[0]["spin_norm"]) == 5.0


def test_run_pipeline_writes_required_canonical_columns(tmp_path):
    outdir = tmp_path / "out"
    manifest_path = tmp_path / "manifest.csv"
    tt3d = tmp_path / "001.csv"
    tt3d.write_text("Timestamp,X,Y,Z\n0.1,1,2,3\n", encoding="utf-8")
    _manifest(
        [
            {
                "source_dataset": "TT3D",
                "path": str(tt3d),
                "relative_path": "external_data/TT3D/evaluation/3D_gt/001.csv",
                "license_tag": "CC-BY-4.0",
                "risk_tier": "clean_physics",
                "allowed_first_version": True,
                "extension": ".csv",
                "columns_json": json.dumps(["Timestamp", "X", "Y", "Z"]),
            }
        ]
    ).to_csv(manifest_path, index=False)

    report = run_pipeline(manifest_path=manifest_path, outdir=outdir)
    canonical = pd.read_csv(outdir / "canonical_external_events.csv")

    assert report["canonical_rows"] == 1
    assert list(canonical.columns) == CANONICAL_COLUMNS
