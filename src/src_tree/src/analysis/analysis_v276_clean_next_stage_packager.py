"""V276 clean next-stage packager.

Packages accepted clean components from V271/V272/V273 using strict local gates.
This script does not use TTMATCH or old-server signals and never uploads
automatically.  It writes review candidates only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v263_questionnaire_baseline_helpers import EXPECTED_COLUMNS, load_v261_cap1_anchor, write_local_submission


OUTDIR = Path("v276_clean_next_stage_packager")
UPLOAD_DIR = Path("upload_candidates_20260519")
ACTION_SEARCH = Path("v273_player_conditional_action_response/v273_action_search.csv")
POINT_SEARCH = Path("v272_action_conditioned_point_residual/v272_point_search.csv")
SERVER_SEARCH = Path("v271_server_microblend_probe/v271_server_probe_search.csv")
VALIDATION_SEARCH = Path("v275_public_like_validation_lab/v275_candidate_sanity.csv")
PACKAGE_SEARCH = OUTDIR / "v276_package_search.csv"


def load_search(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def validate_component(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != 1845:
        raise ValueError(f"{path} rows={len(df)} expected 1845")
    return df


def clean_path_ok(path: str) -> bool:
    upper = str(path).upper()
    return "TTMATCH" not in upper and "OLDHARD" not in upper and "OLDSHARPEN" not in upper and "OLDRANK" not in upper


def accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "verdict" not in df.columns or "path" not in df.columns:
        return pd.DataFrame()
    rows = df[df["verdict"].astype(str).str.contains("CANDIDATE|ACCEPT|KEEP_CLEAN_REVIEW", regex=True, na=False)].copy()
    if rows.empty:
        return rows
    rows = rows[rows["path"].astype(str).map(lambda p: Path(p).exists() and clean_path_ok(p))]
    return rows


def validation_keep_names() -> set[str] | None:
    df = load_search(VALIDATION_SEARCH)
    if df.empty or "candidate" not in df.columns or "decision" not in df.columns:
        return None
    keep = df[df["decision"].astype(str).eq("KEEP_CLEAN_REVIEW")]
    return set(keep["candidate"].astype(str))


def apply_validation_filter(rows: pd.DataFrame, keep_names: set[str] | None) -> pd.DataFrame:
    if keep_names is None or rows.empty:
        return rows
    return rows[rows["candidate"].astype(str).isin(keep_names) | rows["path"].astype(str).map(lambda p: Path(p).name in keep_names)]


def pick_action(df: pd.DataFrame, keep_names: set[str] | None) -> dict | None:
    rows = apply_validation_filter(accepted_rows(df), keep_names)
    if rows.empty:
        return None
    if "action_churn" in rows.columns:
        rows = rows[pd.to_numeric(rows["action_churn"], errors="coerce").le(0.02)]
    if "serve_15_18_count" in rows.columns:
        rows = rows[pd.to_numeric(rows["serve_15_18_count"], errors="coerce").fillna(999).le(2)]
    if "mean_support" in rows.columns:
        rows = rows[pd.to_numeric(rows["mean_support"], errors="coerce").fillna(0).ge(20)]
    if rows.empty:
        return None
    sort_cols = [c for c in ["ordinary_action_delta_vs_anchor", "weak_action_mean_f1"] if c in rows.columns]
    return rows.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else rows.iloc[0].to_dict()


def pick_point(df: pd.DataFrame, keep_names: set[str] | None) -> dict | None:
    rows = apply_validation_filter(accepted_rows(df), keep_names)
    if rows.empty:
        return None
    if "point_churn" in rows.columns:
        rows = rows[pd.to_numeric(rows["point_churn"], errors="coerce").le(0.015)]
    if "point0_rate_test" in rows.columns:
        rows = rows[pd.to_numeric(rows["point0_rate_test"], errors="coerce").between(0.24, 0.31)]
    if "point0_added_rows" in rows.columns:
        rows = rows[pd.to_numeric(rows["point0_added_rows"], errors="coerce").fillna(999).le(8)]
    if rows.empty:
        return None
    sort_cols = [c for c in ["ordinary_delta_vs_base", "rare_point_mean_f1"] if c in rows.columns]
    return rows.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else rows.iloc[0].to_dict()


def pick_server(df: pd.DataFrame, keep_names: set[str] | None) -> dict | None:
    rows = apply_validation_filter(accepted_rows(df), keep_names)
    if rows.empty:
        return None
    if "server_mad_vs_anchor" in rows.columns:
        rows = rows[pd.to_numeric(rows["server_mad_vs_anchor"], errors="coerce").le(0.002)]
    if rows.empty:
        return None
    sort_cols = [c for c in ["delta_vs_proxy_base", "server_auc"] if c in rows.columns]
    return rows.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else rows.iloc[0].to_dict()


def copy_to_upload(path: Path) -> None:
    if UPLOAD_DIR.exists():
        shutil.copy2(path, UPLOAD_DIR / path.name)


def write_candidate(path: Path, df: pd.DataFrame) -> None:
    write_local_submission(path, df)
    copy_to_upload(path)


def churn_metrics(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    return {
        "action_changed_rows": int(np.sum(base["actionId"].to_numpy() != cand["actionId"].to_numpy())),
        "point_changed_rows": int(np.sum(base["pointId"].to_numpy() != cand["pointId"].to_numpy())),
        "server_mad_vs_anchor": float(
            np.mean(
                np.abs(
                    base["serverGetPoint"].to_numpy(dtype=float)
                    - cand["serverGetPoint"].to_numpy(dtype=float)
                )
            )
        ),
    }


def add_package(rows: list[dict], base: pd.DataFrame, name: str, df: pd.DataFrame, components: list[str]) -> None:
    path = OUTDIR / name
    write_candidate(path, df)
    rows.append(
        {
            "candidate": name,
            "verdict": "LOCAL_COMPONENTS_PACKAGED_REVIEW_BEFORE_UPLOAD",
            "accepted_components": ",".join(components),
            "path": str(path),
            **churn_metrics(base, df),
        }
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = load_v261_cap1_anchor()
    keep_names = validation_keep_names()

    action_row = pick_action(load_search(ACTION_SEARCH), keep_names)
    point_row = pick_point(load_search(POINT_SEARCH), keep_names)
    server_row = pick_server(load_search(SERVER_SEARCH), keep_names)

    action_sub = validate_component(Path(str(action_row["path"]))) if action_row else None
    point_sub = validate_component(Path(str(point_row["path"]))) if point_row else None
    server_sub = validate_component(Path(str(server_row["path"]))) if server_row else None

    rows: list[dict] = []
    add_package(rows, base, "submission_v276_anchor_copy__clean.csv", base.copy(), ["anchor"])

    if server_sub is not None:
        out = base.copy()
        out["serverGetPoint"] = server_sub["serverGetPoint"].astype(float).to_numpy()
        add_package(rows, base, "submission_v276_best_server_only__clean.csv", out, ["server"])

    if point_sub is not None:
        out = base.copy()
        out["pointId"] = point_sub["pointId"].astype(int).to_numpy()
        add_package(rows, base, "submission_v276_best_point_only__clean.csv", out, ["point"])

    if action_sub is not None:
        out = base.copy()
        out["actionId"] = action_sub["actionId"].astype(int).to_numpy()
        add_package(rows, base, "submission_v276_best_action_only__clean.csv", out, ["action"])

    if point_sub is not None and server_sub is not None:
        out = base.copy()
        out["pointId"] = point_sub["pointId"].astype(int).to_numpy()
        out["serverGetPoint"] = server_sub["serverGetPoint"].astype(float).to_numpy()
        add_package(rows, base, "submission_v276_best_point_server__clean.csv", out, ["point", "server"])

    pd.DataFrame(rows).to_csv(PACKAGE_SEARCH, index=False)
    report = {
        "accepted": {
            "action": action_row,
            "point": point_row,
            "server": server_row,
        },
        "validation_filter_available": keep_names is not None,
        "generated": rows,
        "policy": "clean_no_old_no_ttmatch_review_only",
    }
    (OUTDIR / "v276_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUTDIR / "v276_report.md").write_text(
        "# V276 Clean Next-Stage Packager\n\n"
        f"- Generated packages: `{len(rows)}`\n"
        f"- Accepted action: `{action_row.get('candidate') if action_row else 'none'}`\n"
        f"- Accepted point: `{point_row.get('candidate') if point_row else 'none'}`\n"
        f"- Accepted server: `{server_row.get('candidate') if server_row else 'none'}`\n"
        "- Policy: `clean_no_old_no_ttmatch_review_only`\n",
        encoding="utf-8",
    )
    print(json.dumps({"generated": len(rows), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
