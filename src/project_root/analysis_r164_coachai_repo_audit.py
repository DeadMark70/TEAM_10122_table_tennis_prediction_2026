from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("external_data") / "CoachAI-Projects-main"
OUTDIR = Path("r164_coachai_audit")


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def safe_nunique(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    return int(df[col].nunique(dropna=True))


def csv_summary(name: str, path: Path, df: pd.DataFrame) -> dict:
    type_col = "type" if "type" in df.columns else None
    landing_col = "landing_area" if "landing_area" in df.columns else None
    return {
        "dataset": name,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "rallies": safe_nunique(df, "rally_id") or safe_nunique(df, "rally"),
        "matches": safe_nunique(df, "match_id"),
        "sets": safe_nunique(df, "set"),
        "players": safe_nunique(df, "player"),
        "shot_types": safe_nunique(df, type_col) if type_col else None,
        "landing_areas": safe_nunique(df, landing_col) if landing_col else None,
        "has_landing_xy": {"landing_x", "landing_y"}.issubset(df.columns),
        "has_player_xy": {"player_location_x", "player_location_y"}.issubset(df.columns),
        "has_opponent_xy": {"opponent_location_x", "opponent_location_y"}.issubset(df.columns),
        "has_terminal_signal": any(c in df.columns for c in ["getpoint_player", "lose_reason", "win_reason", "flaw"]),
        "columns_list": "|".join(df.columns.astype(str)),
    }


def collect_set_files(base: Path) -> list[Path]:
    return sorted(
        p
        for p in base.rglob("*.csv")
        if p.name.lower() not in {"match.csv", "homography.csv"}
    )


def summarize_many_csvs(name: str, files: list[Path]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    col_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    landing_counter: Counter[str] = Counter()

    total_rows = 0
    rally_keys = set()
    match_ids = set()
    set_ids = set()
    players = set()
    sample_path = files[0] if files else None

    for idx, path in enumerate(files):
        df = read_csv(path)
        total_rows += len(df)
        col_counter.update(map(str, df.columns))
        if "type" in df.columns:
            type_counter.update(df["type"].dropna().astype(str))
        if "landing_area" in df.columns:
            landing_counter.update(df["landing_area"].dropna().astype(str))
        if "rally" in df.columns:
            rally_keys.update((str(path.parent.name), str(v)) for v in df["rally"].dropna().unique())
        if "match_id" in df.columns:
            match_ids.update(df["match_id"].dropna().astype(str).unique())
        if "set" in df.columns:
            set_ids.update((str(path.parent.name), str(v)) for v in df["set"].dropna().unique())
        if "player" in df.columns:
            players.update(df["player"].dropna().astype(str).unique())
        if idx < 10:
            rows.append(
                {
                    "dataset": name,
                    "path": str(path),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_list": "|".join(df.columns.astype(str)),
                }
            )

    all_cols = sorted(col_counter)
    summary = {
        "dataset": name,
        "path": str(sample_path.parent.parent if sample_path else ""),
        "rows": total_rows,
        "columns": len(all_cols),
        "rallies": len(rally_keys) if rally_keys else None,
        "matches": len(match_ids) if match_ids else None,
        "sets": len(set_ids) if set_ids else None,
        "players": len(players) if players else None,
        "shot_types": len(type_counter) if type_counter else None,
        "landing_areas": len(landing_counter) if landing_counter else None,
        "has_landing_xy": {"landing_x", "landing_y"}.issubset(all_cols),
        "has_player_xy": {"player_location_x", "player_location_y"}.issubset(all_cols),
        "has_opponent_xy": {"opponent_location_x", "opponent_location_y"}.issubset(all_cols),
        "has_terminal_signal": any(c in all_cols for c in ["getpoint_player", "lose_reason", "win_reason", "flaw"]),
        "columns_list": "|".join(all_cols),
        "files": len(files),
    }
    type_df = pd.DataFrame(
        [{"dataset": name, "field": "type", "value": k, "count": v} for k, v in type_counter.most_common()]
        + [{"dataset": name, "field": "landing_area", "value": k, "count": v} for k, v in landing_counter.most_common()]
    )
    return summary, pd.DataFrame(rows), type_df


def write_recommendations(inventory: pd.DataFrame) -> None:
    md = """# R164 CoachAI Repository Audit

## Decision

Use CoachAI as external badminton sequence pretraining data only. Do not map badminton shot labels directly to AICUP table-tennis actionId. The safe target is coarse sequence knowledge: phase, action family, landing depth/side, coordinates, and terminal/outcome structure.

## Files To Keep For Our Work

1. `external_data/CoachAI-Projects-main/CoachAI-Challenge-IJCAI2023/Track 2_ Stroke Forecasting/data/train.csv`
   - Best immediate source. It is already formatted for prefix-to-future stroke forecasting.
   - Use for coarse action-family, landing-area, landing-xy, terminal/remaining pretraining.

2. `external_data/CoachAI-Projects-main/CoachAI-Challenge-IJCAI2023/Track 2_ Stroke Forecasting/data/val_given.csv`, `val_gt.csv`, `test_given.csv`, `test_gt.csv`
   - Useful for reproducing ShuttleNet-style validation and checking prefix/future split logic.
   - Do not treat its leaderboard logic as AICUP logic.

3. `external_data/CoachAI-Projects-main/CoachAI-Challenge-IJCAI2023/ShuttleSet22/set/**/set*.csv`
   - Best larger raw stroke-level corpus from 2022.
   - Use for canonical conversion and masked/causal sequence pretraining.

4. `external_data/CoachAI-Projects-main/ShuttleSet/set/**/set*.csv`
   - Older raw corpus. Useful, but lower priority than ShuttleSet22 because Track 2 data is already built from ShuttleSet22.

5. Code references:
   - `external_data/CoachAI-Projects-main/Stroke Forecasting/ShuttleNet/`
   - `external_data/CoachAI-Projects-main/Stroke Forecasting/badmintondataset.py`
   - `external_data/CoachAI-Projects-main/Stroke Forecasting/evaluate.py`
   Use these for architecture and preprocessing reference, not as a direct dependency in the AICUP pipeline.

6. Documentation/license:
   - `external_data/CoachAI-Projects-main/LICENSE`
   - `external_data/CoachAI-Projects-main/CITATIONS.bib`
   - `external_data/CoachAI-Projects-main/README.md`
   Keep for final report attribution.

## Files To Defer Or Ignore

- `Visualization Platform/`: not needed for sequence pretraining.
- `Movement Forecasting/`: useful only if we later build player-movement auxiliary objectives.
- `RallyNet/`, `Shot Influence/`, `Strategic Environment/`, `CoachAI Badminton Environment/`: code/research reference only for now.
- Video/skeleton/crawling utilities: avoid for this competition unless we make a separate CV pipeline and license audit.

## Canonical Mapping

| CoachAI field | Canonical use |
| --- | --- |
| `ball_round` | phase / prefix position |
| `type` | technique then coarse action family |
| `landing_area`, `landing_x`, `landing_y` | landing depth/side/coordinate auxiliary labels |
| `player`, `server` | hitter role / style context |
| `backhand`, `aroundhead` | hand/body auxiliary labels |
| `getpoint_player`, `lose_reason`, `win_reason`, `flaw`, `rally_length` | terminal / remaining / outcome auxiliary labels |
| `player_location_*`, `opponent_location_*` | optional spatial pressure auxiliary labels |

## Recommended Next Experiment

V165 CoachAI coarse pretraining:

1. Convert Track 2 train plus ShuttleSet22 set CSVs into a canonical sequence table.
2. Pretrain small sequence encoder on:
   - next coarse shot family
   - next landing depth/side
   - next terminal / remaining bucket
   - masked `type` / `landing_area`
3. Transfer only as low-weight distillation/prior into AICUP action/point, with R67 action anchor and stable point branch kept unchanged.

Risk level: low to medium. It is badminton, not table tennis, so it should regularize sequence geometry but must not be used as direct label mapping.
"""
    (OUTDIR / "coachai_usage_recommendations.md").write_text(md, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    summaries = []
    samples = []
    type_dfs = []

    track2_dir = ROOT / "CoachAI-Challenge-IJCAI2023" / "Track 2_ Stroke Forecasting" / "data"
    for name in ["train.csv", "val_given.csv", "val_gt.csv", "test_given.csv", "test_gt.csv"]:
        path = track2_dir / name
        if path.exists():
            df = read_csv(path)
            summaries.append(csv_summary(f"Track2/{name}", path, df))
            samples.append(df.head(5).assign(_source=f"Track2/{name}"))
            if "type" in df.columns:
                vc = df["type"].dropna().astype(str).value_counts().reset_index()
                vc.columns = ["value", "count"]
                vc["dataset"] = f"Track2/{name}"
                vc["field"] = "type"
                type_dfs.append(vc[["dataset", "field", "value", "count"]])
            if "landing_area" in df.columns:
                vc = df["landing_area"].dropna().astype(str).value_counts().reset_index()
                vc.columns = ["value", "count"]
                vc["dataset"] = f"Track2/{name}"
                vc["field"] = "landing_area"
                type_dfs.append(vc[["dataset", "field", "value", "count"]])

    for name, base in [
        ("ShuttleSet22/raw_sets", ROOT / "CoachAI-Challenge-IJCAI2023" / "ShuttleSet22" / "set"),
        ("ShuttleSet/raw_sets", ROOT / "ShuttleSet" / "set"),
    ]:
        files = collect_set_files(base)
        if files:
            summary, sample_df, type_df = summarize_many_csvs(name, files)
            summaries.append(summary)
            samples.append(sample_df)
            if not type_df.empty:
                type_dfs.append(type_df)

    sample_path = ROOT / "Stroke Forecasting" / "data" / "dataset_sample.csv"
    if sample_path.exists():
        df = read_csv(sample_path)
        summaries.append(csv_summary("StrokeForecasting/dataset_sample.csv", sample_path, df))
        samples.append(df.head(5).assign(_source="StrokeForecasting/dataset_sample.csv"))

    inventory = pd.DataFrame(summaries)
    inventory.to_csv(OUTDIR / "coachai_dataset_inventory.csv", index=False, encoding="utf-8-sig")

    if samples:
        pd.concat(samples, ignore_index=True, sort=False).to_csv(
            OUTDIR / "coachai_samples.csv", index=False, encoding="utf-8-sig"
        )
    if type_dfs:
        pd.concat(type_dfs, ignore_index=True, sort=False).to_csv(
            OUTDIR / "coachai_type_distributions.csv", index=False, encoding="utf-8-sig"
        )

    columns = []
    for row in summaries:
        for col in str(row["columns_list"]).split("|"):
            columns.append({"dataset": row["dataset"], "column": col})
    pd.DataFrame(columns).to_csv(OUTDIR / "coachai_column_inventory.csv", index=False, encoding="utf-8-sig")

    write_recommendations(inventory)
    print(inventory[["dataset", "rows", "rallies", "matches", "players", "shot_types", "landing_areas"]])
    print(f"Wrote {OUTDIR}")


if __name__ == "__main__":
    main()
