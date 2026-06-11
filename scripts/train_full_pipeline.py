from __future__ import annotations

import argparse
import json

from pipeline_utils import PROJECT_ROOT, ROOT, add_common_args, prepare_official_data_links, run_python


STAGES = [
    {
        "name": "action_teacher_v173",
        "script": PROJECT_ROOT / "analysis_v173_external_curriculum_pretrain.py",
        "purpose": "external curriculum action teacher and action candidate sweep",
    },
    {
        "name": "server_v300",
        "script": PROJECT_ROOT / "analysis_v300_clean_server_blend_recycler.py",
        "purpose": "clean conservative server blend",
    },
    {
        "name": "joint_moe_anchor_v338",
        "script": PROJECT_ROOT / "analysis_v338_joint_moe_pack.py",
        "purpose": "public-positive action/point/server anchor used by V362 point refinement",
    },
    {
        "name": "point_residual_v362",
        "script": PROJECT_ROOT / "analysis_v362_point_hierarchical_specialists.py",
        "purpose": "depth-agreement point residual specialist preserving V173 action and V300 server",
    },
]


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(
            description=(
                "Run the documented component-level training pipeline. This expects official data "
                "in data/raw and optional external resources in external_data."
            )
        )
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=[stage["name"] for stage in STAGES],
        help="Subset of stages to run in order. Defaults to all documented stages.",
    )
    args = parser.parse_args()

    data_status = prepare_official_data_links(copy=not args.skip_data_copy)
    selected = set(args.stages)
    print(
        json.dumps(
            {
                "stage": "prepare_data",
                "status": data_status,
                "selected_stages": args.stages,
                "note": "Exact leaderboard reproduction also depends on optional external_data and generated intermediate anchors.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    unknown = selected.difference(stage["name"] for stage in STAGES)
    if unknown:
        raise SystemExit(f"Unknown stage(s): {sorted(unknown)}")

    for stage in STAGES:
        if stage["name"] not in selected:
            continue
        print(json.dumps({"stage": stage["name"], "purpose": stage["purpose"]}, ensure_ascii=False))
        run_python(stage["script"], cwd=ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
