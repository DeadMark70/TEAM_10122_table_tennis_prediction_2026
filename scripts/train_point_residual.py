from __future__ import annotations

import argparse
import json

from pipeline_utils import PROJECT_ROOT, ROOT, add_common_args, prepare_official_data_links, run_python


POINT_SCRIPT = PROJECT_ROOT / "analysis_v362_point_hierarchical_specialists.py"


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description="Train/build the V362 depth-agreement point residual specialist.")
    )
    args = parser.parse_args()

    data_status = prepare_official_data_links(copy=not args.skip_data_copy)
    print(json.dumps({"stage": "prepare_data", "status": data_status}, indent=2, ensure_ascii=False))
    run_python(POINT_SCRIPT, cwd=ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
