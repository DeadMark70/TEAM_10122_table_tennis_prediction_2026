from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from analysis_v257_coachai_schema_helpers import (
    badminton_family,
    canonicalize_phase,
    forbid_ttmatch_path,
    normalize_xy,
)


ROOT = Path(".")
COACHAI_ROOT = ROOT / "external_data" / "CoachAI-Projects-main"
OUTDIR = ROOT / "v257_shuttlenet_corpus"
MAX_ROWS_PER_CSV = int(os.environ.get("V257_MAX_ROWS_PER_CSV", "0"))


def discover_csvs() -> list[Path]:
    roots = [
        COACHAI_ROOT / "Stroke Forecasting",
        COACHAI_ROOT / "ShuttleSet",
        COACHAI_ROOT / "CoachAI-Challenge-IJCAI2023",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            forbid_ttmatch_path(path)
            paths.append(path)
    return sorted(paths)


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).lower().strip(): str(col) for col in df.columns}


def _first_existing(lower: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in lower:
            return lower[name]
    return None


def standardize_one_csv(path: Path) -> pd.DataFrame:
    forbid_ttmatch_path(path)
    try:
        df = pd.read_csv(path, nrows=MAX_ROWS_PER_CSV or None)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig", nrows=MAX_ROWS_PER_CSV or None)
    lower = _column_lookup(df)
    shot_col = _first_existing(lower, ("type", "shot_type", "ball_type", "action"))
    rally_col = _first_existing(lower, ("rally_id", "rally", "rallyid"))
    order_col = _first_existing(lower, ("ball_round", "stroke_index", "round", "frame_num"))
    player_col = _first_existing(lower, ("player", "player_id", "hitter"))
    x_col = _first_existing(lower, ("landing_x", "x", "pos_x"))
    y_col = _first_existing(lower, ("landing_y", "y", "pos_y"))
    if shot_col is None or rally_col is None:
        return pd.DataFrame()

    work = df.copy()
    if order_col is not None:
        work = work.sort_values([rally_col, order_col], kind="mergesort")

    out = pd.DataFrame(index=work.index)
    out["source_file"] = str(path)
    if "CoachAI-Projects-main" in path.parts:
        out["source_dataset"] = path.parts[path.parts.index("CoachAI-Projects-main") + 1]
    else:
        out["source_dataset"] = "CoachAI"
    out["rally_id"] = work[rally_col].astype(str)
    out["stroke_index"] = work.groupby(rally_col, sort=False).cumcount()
    out["shot_type_raw"] = work[shot_col].astype(str).str.lower().str.strip()
    out["action_family"] = out["shot_type_raw"].map(badminton_family)
    out["phase"] = out["stroke_index"].map(canonicalize_phase)
    out["player_id"] = work[player_col].astype(str) if player_col else (out["stroke_index"] % 2).astype(str)
    if x_col and y_col:
        out["x_norm"], out["y_norm"] = normalize_xy(work[x_col], work[y_col])
    else:
        out["x_norm"] = 0.0
        out["y_norm"] = 0.0
    out["terminal_like"] = 0
    last_idx = out.groupby("rally_id")["stroke_index"].transform("max")
    out.loc[out["stroke_index"] == last_idx, "terminal_like"] = 1
    return out.reset_index(drop=True)


def write_corpus(corpus: pd.DataFrame) -> tuple[str, str | None]:
    parquet_path = OUTDIR / "v257_canonical_sequences.parquet"
    csv_path = OUTDIR / "v257_canonical_sequences.csv"
    try:
        corpus.to_parquet(parquet_path, index=False)
        return str(parquet_path), None
    except Exception as exc:
        corpus.to_csv(csv_path, index=False)
        return str(csv_path), f"parquet_unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    if not COACHAI_ROOT.exists():
        raise FileNotFoundError(f"Missing CoachAI root: {COACHAI_ROOT}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    skipped = []
    for path in discover_csvs():
        frame = standardize_one_csv(path)
        if len(frame) == 0:
            skipped.append(str(path))
        else:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No usable CoachAI/ShuttleSet CSV files found for V257 corpus.")

    corpus = pd.concat(frames, ignore_index=True)
    corpus["global_rally_uid"] = corpus["source_file"].astype(str) + "::" + corpus["rally_id"].astype(str)
    corpus_path, fallback_reason = write_corpus(corpus)
    report = {
        "rows": int(len(corpus)),
        "rallies": int(corpus["global_rally_uid"].nunique()),
        "sources": corpus["source_dataset"].value_counts().to_dict(),
        "family_counts": corpus["action_family"].value_counts().to_dict(),
        "corpus_path": corpus_path,
        "fallback_reason": fallback_reason,
        "skipped_csvs": skipped[:100],
        "skipped_count": len(skipped),
    }
    (OUTDIR / "v257_corpus_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(OUTDIR), "rows": report["rows"], "rallies": report["rallies"], "corpus_path": corpus_path}))


if __name__ == "__main__":
    main()
