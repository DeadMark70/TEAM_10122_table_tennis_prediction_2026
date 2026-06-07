from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def verify_submission(path: Path) -> None:
    df = pd.read_csv(path)
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != 1845:
        raise ValueError(f"Expected 1845 rows, got {len(df)}")
    if df.isna().sum().sum() != 0:
        raise ValueError("Submission contains NaN values")
    if not df["serverGetPoint"].between(0, 1).all():
        raise ValueError("serverGetPoint must be in [0, 1]")
    print(f"OK: {path}")


if __name__ == "__main__":
    verify_submission(Path(sys.argv[1]))
