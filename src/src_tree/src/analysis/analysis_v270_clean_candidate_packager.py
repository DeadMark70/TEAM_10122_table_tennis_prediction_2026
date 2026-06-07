"""V270 clean candidate packager.

Packages only clean no-old/no-TTMATCH components from V267/V268/V269.
This script never creates an upload decision; it writes review candidates and
keeps the current V261 cap1 anchor as the default.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v263_questionnaire_baseline_helpers import EXPECTED_COLUMNS, load_v261_cap1_anchor, write_local_submission


OUTDIR = Path("v270_clean_candidate_packager")
UPLOAD_DIR = Path("upload_candidates_20260519")
ACTION_SEARCH = Path("v267_macro_f1_action_teacher/v267_action_search.csv")
POINT_SEARCH = Path("v268_macro_f1_point_residual/v268_point_search.csv")
SERVER_SEARCH = Path("v269_clean_server_value_ranker/v269_server_search.csv")
PACKAGE_SEARCH = OUTDIR / "v270_package_search.csv"


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


def accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "verdict" not in df.columns or "path" not in df.columns:
        return pd.DataFrame()
    mask = df["verdict"].astype(str).str.contains("CANDIDATE|ACCEPT|LOCAL_POSITIVE", regex=True, na=False)
    out = df[mask].copy()
    if out.empty:
        return out
    out = out[out["path"].astype(str).map(lambda p: Path(p).exists())]
    return out


def pick_action(df: pd.DataFrame) -> dict | None:
    rows = accepted_rows(df)
    if rows.empty:
        return None
    if "action_churn" in rows.columns:
        rows = rows[pd.to_numeric(rows["action_churn"], errors="coerce").le(0.05)]
    if "serve_15_18_count" in rows.columns:
        rows = rows[pd.to_numeric(rows["serve_15_18_count"], errors="coerce").fillna(0).le(2)]
    if rows.empty:
        return None
    sort_cols = [c for c in ["ordinary_action_delta_vs_anchor", "ordinary_action_macro_f1", "weak_action_mean_f1"] if c in rows.columns]
    if sort_cols:
        return rows.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
    return rows.iloc[0].to_dict()


def pick_point(df: pd.DataFrame) -> dict | None:
    rows = accepted_rows(df)
    if rows.empty:
        return None
    if "point_churn" in rows.columns:
        rows = rows[pd.to_numeric(rows["point_churn"], errors="coerce").le(0.02)]
    if "point0_rate_test" in rows.columns:
        rows = rows[pd.to_numeric(rows["point0_rate_test"], errors="coerce").between(0.24, 0.31)]
    if "point0_added_rows" in rows.columns:
        rows = rows[pd.to_numeric(rows["point0_added_rows"], errors="coerce").fillna(999).le(18)]
    if rows.empty:
        return None
    sort_cols = [c for c in ["ordinary_delta_vs_base", "ordinary_point_macro_f1", "rare_point_mean_f1"] if c in rows.columns]
    if sort_cols:
        return rows.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
    return rows.iloc[0].to_dict()


def pick_server(df: pd.DataFrame) -> dict | None:
    rows = accepted_rows(df)
    if rows.empty:
        return None
    if "server_mad_vs_anchor" in rows.columns:
        rows = rows[pd.to_numeric(rows["server_mad_vs_anchor"], errors="coerce").le(0.002)]
    if rows.empty:
        return None
    sort_cols = [c for c in ["delta_vs_proxy_base", "server_auc"] if c in rows.columns]
    if sort_cols:
        return rows.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
    return rows.iloc[0].to_dict()


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
                    base["serverGetPoint"].to_numpy(dtype=float) - cand["serverGetPoint"].to_numpy(dtype=float)
                )
            )
        ),
    }


def add_package(rows: list[dict], base: pd.DataFrame, name: str, df: pd.DataFrame, components: list[str]) -> None:
    path = OUTDIR / name
    write_candidate(path, df)
    metrics = churn_metrics(base, df)
    rows.append(
        {
            "candidate": name,
            "verdict": "LOCAL_COMPONENTS_PACKAGED_REVIEW_BEFORE_UPLOAD",
            "accepted_components": ",".join(components),
            "path": str(path),
            **metrics,
        }
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = load_v261_cap1_anchor()

    action_row = pick_action(load_search(ACTION_SEARCH))
    point_row = pick_point(load_search(POINT_SEARCH))
    server_row = pick_server(load_search(SERVER_SEARCH))

    action_sub = validate_component(Path(str(action_row["path"]))) if action_row else None
    point_sub = validate_component(Path(str(point_row["path"]))) if point_row else None
    server_sub = validate_component(Path(str(server_row["path"]))) if server_row else None

    package_rows: list[dict] = []

    add_package(package_rows, base, "submission_v270_anchor_copy__clean.csv", base.copy(), ["anchor"])

    if action_sub is not None:
        out = base.copy()
        out["actionId"] = action_sub["actionId"].astype(int).to_numpy()
        add_package(package_rows, base, "submission_v270_best_action_only__pv261cap1__sr121.csv", out, ["action"])

    if point_sub is not None:
        out = base.copy()
        out["pointId"] = point_sub["pointId"].astype(int).to_numpy()
        add_package(package_rows, base, "submission_v270_best_point_only__v173action_r121server.csv", out, ["point"])

    if server_sub is not None:
        out = base.copy()
        out["serverGetPoint"] = server_sub["serverGetPoint"].astype(float).to_numpy()
        add_package(package_rows, base, "submission_v270_best_server_only__v173_v261cap1.csv", out, ["server"])

    if action_sub is not None and server_sub is not None:
        out = base.copy()
        out["actionId"] = action_sub["actionId"].astype(int).to_numpy()
        out["serverGetPoint"] = server_sub["serverGetPoint"].astype(float).to_numpy()
        add_package(package_rows, base, "submission_v270_best_action_server__pv261cap1.csv", out, ["action", "server"])

    if point_sub is not None and server_sub is not None:
        out = base.copy()
        out["pointId"] = point_sub["pointId"].astype(int).to_numpy()
        out["serverGetPoint"] = server_sub["serverGetPoint"].astype(float).to_numpy()
        add_package(package_rows, base, "submission_v270_best_point_server__v173action.csv", out, ["point", "server"])

    pd.DataFrame(package_rows).to_csv(PACKAGE_SEARCH, index=False)

    report = {
        "accepted": {
            "action": action_row,
            "point": point_row,
            "server": server_row,
        },
        "generated": package_rows,
        "policy": "clean_no_old_no_ttmatch_review_only",
    }
    (OUTDIR / "v270_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUTDIR / "v270_report.md").write_text(
        "# V270 Clean Candidate Packager\n\n"
        f"- Generated packages: `{len(package_rows)}`\n"
        f"- Accepted action: `{action_row.get('candidate') if action_row else 'none'}`\n"
        f"- Accepted point: `{point_row.get('candidate') if point_row else 'none'}`\n"
        f"- Accepted server: `{server_row.get('candidate') if server_row else 'none'}`\n"
        "- Policy: `clean_no_old_no_ttmatch_review_only`\n",
        encoding="utf-8",
    )
    print(json.dumps({"generated": len(package_rows), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
