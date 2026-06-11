from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "src" / "project_root"
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "external_data"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def existing_path(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def require_files(paths: Iterable[Path], *, label: str) -> list[Path]:
    missing = [path for path in paths if not path.exists()]
    if missing:
        details = "\n".join(f"  - {rel(path)}" for path in missing)
        raise FileNotFoundError(f"Missing required {label} file(s):\n{details}")
    return list(paths)


def prepare_official_data_links(*, copy: bool = True) -> dict[str, str]:
    """Expose official data at both documented and legacy script locations.

    Most original experiment scripts were written from the workspace root and
    read train.csv/test_new.csv directly. The release layout documents
    data/raw/*.csv, so this helper copies those files to the legacy locations
    when needed. Raw data is ignored by git and is never redistributed.
    """

    required = {
        "train.csv": RAW_DIR / "train.csv",
        "test_new.csv": RAW_DIR / "test_new.csv",
        "sample_submission.csv": RAW_DIR / "sample_submission.csv",
    }
    require_files(required.values(), label="official competition data")

    status: dict[str, str] = {}
    for name, source in required.items():
        target = ROOT / name
        if target.exists():
            status[name] = f"exists:{rel(target)}"
            continue
        if not copy:
            status[name] = f"missing_legacy_copy:{rel(target)}"
            continue
        shutil.copy2(source, target)
        status[name] = f"copied:{rel(source)}->{rel(target)}"
    return status


def run_python(script: Path, *, cwd: Path = ROOT, dry_run: bool = False) -> int:
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [sys.executable, str(script)]
    print(json.dumps({"cmd": cmd, "cwd": str(cwd), "dry_run": dry_run}, ensure_ascii=False))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=cwd, check=True)
    return completed.returncode


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and validate inputs without running training scripts.",
    )
    parser.add_argument(
        "--skip-data-copy",
        action="store_true",
        help="Do not copy data/raw/*.csv to legacy root-level train.csv/test_new.csv/sample_submission.csv.",
    )
    return parser
