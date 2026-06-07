from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "external_data"
OUTDIR = ROOT / "r150_external_data_audit"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_ttmatch_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "strickNumber": "strikeNumber",
        "strickId": "strikeId",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def file_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(EXTERNAL.rglob("*")):
        if p.is_file():
            rows.append(
                {
                    "dataset": p.relative_to(EXTERNAL).parts[0],
                    "relative_path": str(p.relative_to(ROOT)),
                    "suffix": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                }
            )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    rows = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = df[col]
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "column": col,
                "count": int(s.notna().sum()),
                "nunique": int(s.nunique(dropna=True)),
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def value_counts_summary(df: pd.DataFrame, dataset: str, split: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).sort_index()
        total = len(df)
        for value, count in vc.items():
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "rate": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def fingerprint_overlap(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    cols = [c for c in cols if c in a.columns and c in b.columns]
    if not cols:
        return {"common_columns": [], "left_unique": 0, "right_unique": 0, "intersection": 0, "left_rate": 0.0, "right_rate": 0.0}

    left = pd.util.hash_pandas_object(a[cols].astype("string"), index=False)
    right = pd.util.hash_pandas_object(b[cols].astype("string"), index=False)
    left_set = set(left.astype("uint64").tolist())
    right_set = set(right.astype("uint64").tolist())
    inter = len(left_set & right_set)
    return {
        "common_columns": cols,
        "left_unique": len(left_set),
        "right_unique": len(right_set),
        "intersection": inter,
        "left_rate": inter / max(1, len(left_set)),
        "right_rate": inter / max(1, len(right_set)),
    }


def key_overlap(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    cols = [c for c in cols if c in a.columns and c in b.columns]
    if not cols:
        return {"key_columns": [], "left_unique": 0, "right_unique": 0, "intersection": 0, "left_rate": 0.0, "right_rate": 0.0}
    left = set(map(tuple, a[cols].astype("string").itertuples(index=False, name=None)))
    right = set(map(tuple, b[cols].astype("string").itertuples(index=False, name=None)))
    inter = len(left & right)
    return {
        "key_columns": cols,
        "left_unique": len(left),
        "right_unique": len(right),
        "intersection": inter,
        "left_rate": inter / max(1, len(left)),
        "right_rate": inter / max(1, len(right)),
    }


def audit_deepmind() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for split in ["rallies", "serves"]:
        path = EXTERNAL / "DeepMindrobottabletennis" / f"{split}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["speed_mps"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2 + df["vel_z"] ** 2)
        df["spin_radps"] = np.sqrt(df["w_vel_x"] ** 2 + df["w_vel_y"] ** 2 + df["w_vel_z"] ** 2)
        df["split"] = split
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    summary = numeric_summary(all_df, "DeepMindrobottabletennis", "all")
    by_split = (
        all_df.groupby("split")
        .agg(
            rows=("id", "count"),
            speed_mean=("speed_mps", "mean"),
            speed_p95=("speed_mps", lambda x: float(np.percentile(x, 95))),
            spin_mean=("spin_radps", "mean"),
            spin_p95=("spin_radps", lambda x: float(np.percentile(x, 95))),
            pos_x_min=("pos_x", "min"),
            pos_x_max=("pos_x", "max"),
            pos_y_min=("pos_y", "min"),
            pos_y_max=("pos_y", "max"),
            pos_z_min=("pos_z", "min"),
            pos_z_max=("pos_z", "max"),
        )
        .reset_index()
    )
    meta = {
        "rows": int(len(all_df)),
        "rallies_rows": int((all_df["split"] == "rallies").sum()),
        "serves_rows": int((all_df["split"] == "serves").sum()),
        "license_status": "Apache-2.0 software / CC-BY-4.0 materials from README",
        "risk": "low",
        "recommended_use": "physics auxiliary pretraining only: velocity/spin/landing-grid priors; no direct AI CUP label mapping",
    }
    return summary, by_split, meta


def audit_matchdynamics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = EXTERNAL / "TT-MatchDynamics" / "table_tennis_data.csv"
    df = _read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Coarse 3x3 grid from centimeter-like X/Y ranges; used for audit only.
    x_bins = pd.qcut(df["X"], q=3, labels=["x_low", "x_mid", "x_high"], duplicates="drop")
    y_bins = pd.qcut(df["Y"], q=3, labels=["y_low", "y_mid", "y_high"], duplicates="drop")
    df["xy_grid_bin"] = x_bins.astype(str) + "_" + y_bins.astype(str)
    summary = numeric_summary(df.drop(columns=["date"]), "TT-MatchDynamics", "all")
    counts = value_counts_summary(
        df,
        "TT-MatchDynamics",
        "all",
        ["Topspin/Backspin Indicator", "Forehand/Backhand Indicator", "Winning in First Three Strokes", "xy_grid_bin"],
    )
    meta = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "license_status": "no license/readme found in local folder; provenance must be documented before use",
        "risk": "low-to-medium until license is verified",
        "recommended_use": "small auxiliary landing/spin/hand prior; not useful for 19-class action labels",
    }
    return summary, counts, meta


def audit_ttmatch() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ttm_train = normalize_ttmatch_columns(_read_csv(EXTERNAL / "TTMATCH" / "train.csv"))
    ttm_test = normalize_ttmatch_columns(_read_csv(EXTERNAL / "TTMATCH" / "test.csv"))
    ttm_sample = _read_csv(EXTERNAL / "TTMATCH" / "sample_submission.csv")

    raw_train = normalize_ttmatch_columns(_read_csv(DATA_RAW / "train.csv"))
    raw_new = normalize_ttmatch_columns(_read_csv(DATA_RAW / "test_new.csv"))
    raw_old = normalize_ttmatch_columns(_read_csv(DATA_RAW / "test_old.csv"))

    datasets = {
        "ttmatch_train": ttm_train,
        "ttmatch_test": ttm_test,
        "aicup_train": raw_train,
        "aicup_test_new": raw_new,
        "aicup_test_old": raw_old,
    }

    schema_rows = []
    for name, df in datasets.items():
        schema_rows.append(
            {
                "dataset": name,
                "rows": int(len(df)),
                "rallies": int(df["rally_uid"].nunique()) if "rally_uid" in df.columns else None,
                "matches": int(df["match"].nunique()) if "match" in df.columns else None,
                "players": int(pd.concat([df.get("gamePlayerId", pd.Series(dtype=int)), df.get("gamePlayerOtherId", pd.Series(dtype=int))]).nunique()),
                "columns": ",".join(df.columns),
                "has_serverGetPoint": "serverGetPoint" in df.columns,
                "rally_uid_min": int(df["rally_uid"].min()) if "rally_uid" in df.columns and len(df) else None,
                "rally_uid_max": int(df["rally_uid"].max()) if "rally_uid" in df.columns and len(df) else None,
            }
        )
    schema = pd.DataFrame(schema_rows)

    overlap_rows = []
    public_cols = [
        "sex",
        "match",
        "numberGame",
        "rally_id",
        "strikeNumber",
        "scoreSelf",
        "scoreOther",
        "gamePlayerId",
        "gamePlayerOtherId",
        "strikeId",
        "handId",
        "strengthId",
        "spinId",
        "pointId",
        "actionId",
        "positionId",
    ]
    public_no_ids = [
        "sex",
        "numberGame",
        "strikeNumber",
        "scoreSelf",
        "scoreOther",
        "strikeId",
        "handId",
        "strengthId",
        "spinId",
        "pointId",
        "actionId",
        "positionId",
    ]
    key_sets = {
        "rally_uid": ["rally_uid"],
        "rally_uid_strike": ["rally_uid", "strikeNumber"],
        "match_game_rally_strike": ["match", "numberGame", "rally_id", "strikeNumber"],
        "player_match_game_rally_strike": ["match", "numberGame", "rally_id", "strikeNumber", "gamePlayerId", "gamePlayerOtherId"],
    }
    pairs = [
        ("ttmatch_train", ttm_train, "aicup_train", raw_train),
        ("ttmatch_train", ttm_train, "aicup_test_old", raw_old),
        ("ttmatch_train", ttm_train, "aicup_test_new", raw_new),
        ("ttmatch_test", ttm_test, "aicup_train", raw_train),
        ("ttmatch_test", ttm_test, "aicup_test_old", raw_old),
        ("ttmatch_test", ttm_test, "aicup_test_new", raw_new),
    ]
    for left_name, left, right_name, right in pairs:
        for key_name, cols in key_sets.items():
            res = key_overlap(left, right, cols)
            overlap_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "overlap_type": key_name,
                    **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in res.items()},
                }
            )
        for fp_name, cols in [("public_full_fingerprint", public_cols), ("public_no_ids_fingerprint", public_no_ids)]:
            res = fingerprint_overlap(left, right, cols)
            overlap_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "overlap_type": fp_name,
                    **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in res.items()},
                }
            )
    overlap = pd.DataFrame(overlap_rows)

    counts = pd.concat(
        [
            value_counts_summary(ttm_train, "TTMATCH", "train", ["serverGetPoint", "actionId", "pointId", "spinId", "strengthId", "handId", "positionId"]),
            value_counts_summary(ttm_test, "TTMATCH", "test", ["actionId", "pointId", "spinId", "strengthId", "handId", "positionId"]),
        ],
        ignore_index=True,
    )

    meta = {
        "train_rows": int(len(ttm_train)),
        "test_rows": int(len(ttm_test)),
        "sample_rows": int(len(ttm_sample)),
        "normalized_columns_train": list(ttm_train.columns),
        "normalized_columns_test": list(ttm_test.columns),
        "license_status": "no local license/readme found; deepresearch notes Kaggle competition dataset, verify Kaggle license before report/use",
        "risk": "high until provenance and overlap are reviewed",
        "recommended_use": "if no exact overlap: supervised augmentation/transition prior with low weight and full report disclosure; never direct lookup into AI CUP test/private",
    }
    return schema, overlap, counts, meta


def build_markdown_report(meta: dict[str, Any], overlap: pd.DataFrame) -> str:
    lines = [
        "# R150 External Data Audit",
        "",
        "Purpose: audit the three new datasets under `external_data/` before any training use.",
        "",
        "## Summary",
        "",
        "| Dataset | Local folder | Risk | Recommended handling |",
        "|---|---|---|---|",
        f"| DeepMind Robot Table Tennis | `external_data/DeepMindrobottabletennis` | {meta['deepmind']['risk']} | {meta['deepmind']['recommended_use']} |",
        f"| TT-MatchDynamics | `external_data/TT-MatchDynamics` | {meta['matchdynamics']['risk']} | {meta['matchdynamics']['recommended_use']} |",
        f"| TTMATCH | `external_data/TTMATCH` | {meta['ttmatch']['risk']} | {meta['ttmatch']['recommended_use']} |",
        "",
        "## TTMATCH Overlap Notes",
        "",
        "The `rally_uid` overlap alone is not treated as evidence of identical samples because competition IDs may be re-used or randomly assigned. The safer signal is exact fingerprint overlap over public stroke columns.",
        "",
    ]
    view = overlap[
        overlap["overlap_type"].isin(["rally_uid", "rally_uid_strike", "public_full_fingerprint", "public_no_ids_fingerprint"])
    ][["left", "right", "overlap_type", "intersection", "left_rate", "right_rate"]]
    table_cols = view.columns.tolist()
    lines.append("| " + " | ".join(table_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(table_cols)) + "|")
    for _, row in view.iterrows():
        values = []
        for col in table_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Safety Decision",
            "",
            "- Use DeepMind only for physics priors or auxiliary pretraining; it has no AI CUP label-equivalent fields.",
            "- Use TT-MatchDynamics only after documenting source/license; it is small and should only contribute coarse landing/spin/hand priors.",
            "- Keep TTMATCH isolated until the exact fingerprint overlap and provenance are reviewed. If used, prefer fold-safe transition priors or low-weight supervised augmentation, not direct test replacement.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    inv = file_inventory()
    inv.to_csv(OUTDIR / "external_dataset_inventory.csv", index=False)

    dm_summary, dm_by_split, dm_meta = audit_deepmind()
    dm_summary.to_csv(OUTDIR / "deepmind_numeric_summary.csv", index=False)
    dm_by_split.to_csv(OUTDIR / "deepmind_split_summary.csv", index=False)

    md_summary, md_counts, md_meta = audit_matchdynamics()
    md_summary.to_csv(OUTDIR / "tt_matchdynamics_numeric_summary.csv", index=False)
    md_counts.to_csv(OUTDIR / "tt_matchdynamics_value_counts.csv", index=False)

    ttm_schema, ttm_overlap, ttm_counts, ttm_meta = audit_ttmatch()
    ttm_schema.to_csv(OUTDIR / "ttmatch_schema_report.csv", index=False)
    ttm_overlap.to_csv(OUTDIR / "ttmatch_overlap_report.csv", index=False)
    ttm_counts.to_csv(OUTDIR / "ttmatch_value_counts.csv", index=False)

    meta = {
        "deepmind": dm_meta,
        "matchdynamics": md_meta,
        "ttmatch": ttm_meta,
        "outputs": {
            "inventory": str(OUTDIR / "external_dataset_inventory.csv"),
            "ttmatch_overlap": str(OUTDIR / "ttmatch_overlap_report.csv"),
            "ttmatch_schema": str(OUTDIR / "ttmatch_schema_report.csv"),
            "ttmatch_counts": str(OUTDIR / "ttmatch_value_counts.csv"),
            "deepmind_summary": str(OUTDIR / "deepmind_split_summary.csv"),
            "matchdynamics_summary": str(OUTDIR / "tt_matchdynamics_numeric_summary.csv"),
        },
    }
    (OUTDIR / "r150_external_data_audit_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r150_external_data_audit_report.md").write_text(build_markdown_report(meta, ttm_overlap), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
