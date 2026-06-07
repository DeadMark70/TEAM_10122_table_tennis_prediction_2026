from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import write_json
from analysis_v338_joint_moe_pack import (
    apply_compatibility_veto,
    pack_joint_submission,
    run_pipeline,
)


def test_compatibility_veto_blocks_bad_action_point_pair():
    rows = pd.DataFrame(
        {
            "base_action": [10, 3],
            "cand_action": [3, 3],
            "base_point": [2, 8],
            "cand_point": [2, 8],
            "compat_score": [0.01, 0.8],
        }
    )
    keep = apply_compatibility_veto(rows, threshold=0.05)
    assert keep.tolist() == [False, True]


def test_joint_pack_preserves_server():
    base = pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [1, 2],
            "pointId": [8, 9],
            "serverGetPoint": [0.2, 0.8],
        }
    )
    packed = pack_joint_submission(base, action=np.array([3, 2]), point=np.array([8, 0]))
    assert packed["serverGetPoint"].tolist() == [0.2, 0.8]


def _case_dir(name: str) -> Path:
    path = Path("v338_joint_moe_pack") / "test_runs" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_pipeline_waits_when_v337_report_absent():
    root = _case_dir("waits_v337_absent")
    outdir = root / "v338"
    v336_dir = root / "v336"
    v337_dir = root / "v337"
    v336_dir.mkdir()
    write_json(
        v336_dir / "search_report.json",
        {
            "version": "V336",
            "decision": "DO_NOT_UPLOAD",
            "generated_submissions": [],
        },
    )

    report = run_pipeline(outdir=outdir, v336_dir=v336_dir, v337_dir=v337_dir)

    assert report["decision"] == "WAITING_FOR_V337"
    assert report["generated_submission_count"] == 0
    assert (outdir / "search_report.json").exists()


def test_pipeline_packages_point_only_when_v337_passes_and_action_does_not():
    root = _case_dir("point_only")
    outdir = root / "v338"
    v336_dir = root / "v336"
    v337_dir = root / "v337"
    v336_dir.mkdir()
    v337_dir.mkdir()
    point_submission = v337_dir / "submission_v337_point.csv"
    pd.DataFrame(
        {
            "rally_uid": ["a", "b"],
            "actionId": [1, 2],
            "pointId": [7, 0],
            "serverGetPoint": [0.2, 0.8],
        }
    ).to_csv(point_submission, index=False)
    write_json(
        v336_dir / "search_report.json",
        {
            "version": "V336",
            "decision": "DO_NOT_UPLOAD",
            "generated_submissions": [],
        },
    )
    write_json(
        v337_dir / "search_report.json",
        {
            "version": "V337",
            "decision": "HAS_EXPORT",
            "best_candidate": {
                "candidate": "point_moe_no_p0_add_b12",
                "point_oof_delta_vs_v306": 0.004,
            },
            "generated_submissions": [
                {
                    "candidate": "point_moe_no_p0_add_b12",
                    "path": str(point_submission),
                }
            ],
        },
    )

    report = run_pipeline(outdir=outdir, v336_dir=v336_dir, v337_dir=v337_dir, expected_rows=2)

    assert report["decision"] == "POINT_ONLY"
    assert report["generated_submission_count"] == 1
    assert Path(report["generated_submissions"][0]["path"]).name.startswith("submission_v338_point_only_")
