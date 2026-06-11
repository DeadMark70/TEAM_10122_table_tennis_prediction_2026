from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "scripts/reproduce_final.py"])
    run([sys.executable, "scripts/verify_submission.py", "outputs/final_submission.csv"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_release_submission.py",
            "tests/test_external_resource_docs.py",
            "tests/test_release_code_completeness.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
    )


if __name__ == "__main__":
    main()
