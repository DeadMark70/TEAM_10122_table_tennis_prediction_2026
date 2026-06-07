from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external_data"
OUTDIR = ROOT / "r151_safe_physics_priors"


def load_deepmind() -> pd.DataFrame:
    frames = []
    for split in ["rallies", "serves"]:
        path = EXTERNAL / "DeepMindrobottabletennis" / f"{split}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["source_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["speed_mps"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2 + df["vel_z"] ** 2)
    df["spin_radps"] = np.sqrt(df["w_vel_x"] ** 2 + df["w_vel_y"] ** 2 + df["w_vel_z"] ** 2)
    df["horizontal_speed_mps"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2)
    df["vertical_speed_abs_mps"] = df["vel_z"].abs()
    df["spin_top_axis_abs"] = df["w_vel_x"].abs()
    df["spin_side_axis_abs"] = df["w_vel_z"].abs()
    return df


def qbin(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels).astype(str)


def side_from_x(x: pd.Series, left: float, right: float) -> pd.Series:
    return pd.cut(x, bins=[-np.inf, left, right, np.inf], labels=["left", "middle", "right"]).astype(str)


def depth_from_y(y: pd.Series, short: float, long: float) -> pd.Series:
    return pd.cut(y, bins=[-np.inf, short, long, np.inf], labels=["near", "mid", "far"]).astype(str)


def canonicalize_deepmind(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["physics_speed_bin"] = qbin(out["speed_mps"], ["very_slow", "slow", "medium", "fast", "very_fast"])
    out["physics_spin_bin"] = qbin(out["spin_radps"], ["very_low", "low", "medium", "high", "very_high"])
    out["physics_x_side"] = side_from_x(out["pos_x"], -1.525 / 6.0, 1.525 / 6.0)
    out["physics_y_depth"] = depth_from_y(out["pos_y"], -2.74 / 6.0, 2.74 / 6.0)
    out["physics_z_height_bin"] = pd.cut(
        out["pos_z"],
        bins=[-np.inf, 0.76, 0.95, 1.2, np.inf],
        labels=["table_or_lower", "net_heightish", "medium", "high"],
    ).astype(str)
    out["velocity_direction_y"] = np.where(out["vel_y"] >= 0, "positive_y", "negative_y")
    out["canonical_grid_3x3"] = out["physics_x_side"] + "_" + out["physics_y_depth"]

    canonical_cols = [
        "source_split",
        "id",
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "w_vel_x",
        "w_vel_y",
        "w_vel_z",
        "speed_mps",
        "spin_radps",
        "horizontal_speed_mps",
        "vertical_speed_abs_mps",
        "physics_speed_bin",
        "physics_spin_bin",
        "physics_x_side",
        "physics_y_depth",
        "physics_z_height_bin",
        "velocity_direction_y",
        "canonical_grid_3x3",
    ]
    canonical = out[canonical_cols].copy()

    profile = (
        canonical.groupby(["source_split", "physics_speed_bin", "physics_spin_bin"], dropna=False)
        .agg(
            rows=("id", "count"),
            speed_mean=("speed_mps", "mean"),
            spin_mean=("spin_radps", "mean"),
            horizontal_speed_mean=("horizontal_speed_mps", "mean"),
            vertical_speed_abs_mean=("vertical_speed_abs_mps", "mean"),
        )
        .reset_index()
    )
    profile["rate_within_split"] = profile["rows"] / profile.groupby("source_split")["rows"].transform("sum")

    grid = (
        canonical.groupby(["source_split", "canonical_grid_3x3"], dropna=False)
        .agg(
            rows=("id", "count"),
            speed_mean=("speed_mps", "mean"),
            spin_mean=("spin_radps", "mean"),
        )
        .reset_index()
    )
    grid["rate_within_split"] = grid["rows"] / grid.groupby("source_split")["rows"].transform("sum")
    return canonical, profile, grid


def load_matchdynamics() -> pd.DataFrame:
    path = EXTERNAL / "TT-MatchDynamics" / "table_tennis_data.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["source_split"] = "all"
    return df


def canonicalize_matchdynamics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    # Use fixed ranges from observed centimeters. This is a prior only, not AI CUP label mapping.
    x_min, x_max = float(out["X"].min()), float(out["X"].max())
    y_min, y_max = float(out["Y"].min()), float(out["Y"].max())
    x1 = x_min + (x_max - x_min) / 3.0
    x2 = x_min + 2.0 * (x_max - x_min) / 3.0
    y1 = y_min + (y_max - y_min) / 3.0
    y2 = y_min + 2.0 * (y_max - y_min) / 3.0

    out["md_x_side"] = side_from_x(out["X"], x1, x2)
    out["md_y_depth"] = depth_from_y(out["Y"], y1, y2)
    out["md_grid_3x3"] = out["md_x_side"] + "_" + out["md_y_depth"]
    out["md_spin_family"] = np.where(out["Topspin/Backspin Indicator"] == 1, "topspin", "backspin")
    out["md_hand_family"] = np.where(out["Forehand/Backhand Indicator"] == 1, "forehand", "backhand")
    out["md_first3_win"] = out["Winning in First Three Strokes"].astype(int)

    canonical_cols = [
        "source_split",
        "date",
        "Topspin/Backspin Indicator",
        "Forehand/Backhand Indicator",
        "Winning in First Three Strokes",
        "X",
        "Y",
        "md_x_side",
        "md_y_depth",
        "md_grid_3x3",
        "md_spin_family",
        "md_hand_family",
        "md_first3_win",
    ]
    canonical = out[canonical_cols].copy()

    grid_prior = (
        canonical.groupby(["md_grid_3x3", "md_spin_family", "md_hand_family"], dropna=False)
        .agg(
            rows=("X", "count"),
            x_mean=("X", "mean"),
            y_mean=("Y", "mean"),
            first3_win_rate=("md_first3_win", "mean"),
        )
        .reset_index()
    )
    grid_prior["rate"] = grid_prior["rows"] / len(canonical)

    first3_prior = (
        canonical.groupby(["md_spin_family", "md_hand_family", "md_y_depth"], dropna=False)
        .agg(
            rows=("X", "count"),
            first3_win_rate=("md_first3_win", "mean"),
            x_mean=("X", "mean"),
            y_mean=("Y", "mean"),
        )
        .reset_index()
    )
    first3_prior["rate"] = first3_prior["rows"] / len(canonical)
    return canonical, grid_prior, first3_prior


def build_combined_prior(dm_grid: pd.DataFrame, md_grid: pd.DataFrame) -> pd.DataFrame:
    dm = dm_grid.copy()
    dm["source"] = "deepmind_robot"
    dm = dm.rename(columns={"canonical_grid_3x3": "canonical_grid", "rate_within_split": "rate"})
    dm_prior = dm[["source", "source_split", "canonical_grid", "rows", "rate", "speed_mean", "spin_mean"]]
    md = (
        md_grid.groupby("md_grid_3x3", dropna=False)
        .agg(rows=("rows", "sum"), first3_win_rate=("first3_win_rate", "mean"))
        .reset_index()
    )
    md["rate"] = md["rows"] / md["rows"].sum()
    md["source"] = "tt_matchdynamics"
    md["source_split"] = "all"
    md["speed_mean"] = np.nan
    md["spin_mean"] = np.nan
    md_prior = md.rename(columns={"md_grid_3x3": "canonical_grid"})[
        ["source", "source_split", "canonical_grid", "rows", "rate", "speed_mean", "spin_mean", "first3_win_rate"]
    ]
    dm_prior["first3_win_rate"] = np.nan
    return pd.concat([dm_prior, md_prior], ignore_index=True)


def write_report(meta: dict[str, Any], combined: pd.DataFrame) -> str:
    top = combined.sort_values(["source", "rate"], ascending=[True, False]).groupby("source").head(8)
    lines = [
        "# R151 Safe Physics Priors",
        "",
        "R151 intentionally uses only low-risk external data: DeepMind robot ball states and TT-MatchDynamics. It does not use `TTMATCH`.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in meta["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- No AI CUP test labels are used.",
            "- No `rally_uid` alignment or lookup is used.",
            "- Priors are physical/canonical only: speed, spin, coarse x/y grid, first-three-strokes summary.",
            "- Use these outputs only as auxiliary pretraining inputs or very low-weight priors.",
            "",
            "## Top Coarse Grid Priors",
            "",
            "| source | split | grid | rows | rate | speed_mean | spin_mean | first3_win_rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            "| {source} | {source_split} | {canonical_grid} | {rows} | {rate:.4f} | {speed_mean} | {spin_mean} | {first3} |".format(
                source=row["source"],
                source_split=row["source_split"],
                canonical_grid=row["canonical_grid"],
                rows=int(row["rows"]),
                rate=float(row["rate"]),
                speed_mean="" if pd.isna(row["speed_mean"]) else f"{float(row['speed_mean']):.3f}",
                spin_mean="" if pd.isna(row["spin_mean"]) else f"{float(row['spin_mean']):.3f}",
                first3="" if pd.isna(row["first3_win_rate"]) else f"{float(row['first3_win_rate']):.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            "Use `combined_canonical_grid_prior.csv` as a reportable prior table first. If connected to training, use it as an auxiliary feature or label-smoothing prior, not as direct action/point/server supervision.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    dm = load_deepmind()
    dm_canonical, dm_profile, dm_grid = canonicalize_deepmind(dm)
    dm_canonical.to_csv(OUTDIR / "deepmind_canonical_physics_states.csv", index=False)
    dm_profile.to_csv(OUTDIR / "deepmind_speed_spin_profile.csv", index=False)
    dm_grid.to_csv(OUTDIR / "deepmind_grid_prior.csv", index=False)

    md = load_matchdynamics()
    md_canonical, md_grid, md_first3 = canonicalize_matchdynamics(md)
    md_canonical.to_csv(OUTDIR / "tt_matchdynamics_canonical_states.csv", index=False)
    md_grid.to_csv(OUTDIR / "tt_matchdynamics_grid_spin_hand_prior.csv", index=False)
    md_first3.to_csv(OUTDIR / "tt_matchdynamics_first3_prior.csv", index=False)

    combined = build_combined_prior(dm_grid, md_grid)
    combined.to_csv(OUTDIR / "combined_canonical_grid_prior.csv", index=False)

    feature_spec = {
        "deepmind": {
            "rows": int(len(dm_canonical)),
            "features": [
                "speed_mps",
                "spin_radps",
                "horizontal_speed_mps",
                "vertical_speed_abs_mps",
                "physics_speed_bin",
                "physics_spin_bin",
                "physics_x_side",
                "physics_y_depth",
                "physics_z_height_bin",
                "canonical_grid_3x3",
            ],
            "license": "Apache-2.0 software / CC-BY-4.0 materials",
            "risk": "low",
        },
        "tt_matchdynamics": {
            "rows": int(len(md_canonical)),
            "features": [
                "md_x_side",
                "md_y_depth",
                "md_grid_3x3",
                "md_spin_family",
                "md_hand_family",
                "md_first3_win",
            ],
            "license": "not present in local folder; verify before final report/use",
            "risk": "low-to-medium",
        },
        "ai_cup_mapping": {
            "allowed": "auxiliary/pretraining/low-weight canonical prior",
            "not_allowed": "direct actionId/pointId/serverGetPoint labels",
            "canonical_grid": "left/middle/right x near/mid/far, not receiver-relative AI CUP pointId",
        },
    }
    (OUTDIR / "r151_feature_spec.json").write_text(json.dumps(feature_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "outputs": {
            "deepmind_canonical": str(OUTDIR / "deepmind_canonical_physics_states.csv"),
            "deepmind_speed_spin_profile": str(OUTDIR / "deepmind_speed_spin_profile.csv"),
            "deepmind_grid_prior": str(OUTDIR / "deepmind_grid_prior.csv"),
            "tt_matchdynamics_canonical": str(OUTDIR / "tt_matchdynamics_canonical_states.csv"),
            "tt_matchdynamics_grid_spin_hand_prior": str(OUTDIR / "tt_matchdynamics_grid_spin_hand_prior.csv"),
            "tt_matchdynamics_first3_prior": str(OUTDIR / "tt_matchdynamics_first3_prior.csv"),
            "combined_canonical_grid_prior": str(OUTDIR / "combined_canonical_grid_prior.csv"),
            "feature_spec": str(OUTDIR / "r151_feature_spec.json"),
        },
        "deepmind_rows": int(len(dm_canonical)),
        "tt_matchdynamics_rows": int(len(md_canonical)),
        "safety": "No TTMATCH, no AI CUP test labels, no rally_uid lookup.",
    }
    (OUTDIR / "r151_safe_physics_priors_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r151_safe_physics_priors_report.md").write_text(write_report(meta, combined), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
