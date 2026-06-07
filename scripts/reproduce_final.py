from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_submission import verify_submission


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "final_submission" / "submission_v362_depth_agree_only__v173action_v300server.csv"
DST = ROOT / "outputs" / "final_submission.csv"


def main() -> None:
    verify_submission(SRC)
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    verify_submission(DST)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
