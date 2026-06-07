"""V271 server microblend probe.

Repackage the best clean V269 server-only candidates against the current
V173 action / V261 cap1 point / R121 server anchor.  This script reads only
the clean V269 search table and its generated submissions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v271_server_microblend_probe")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
V269_SEARCH = Path("v269_clean_server_value_ranker/v269_server_search.csv")
SEARCH_PATH = OUTDIR / "v271_server_probe_search.csv"
REPORT_PATH = OUTDIR / "v271_report.md"
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]

SAFE_NAME = "submission_v271_server_safe__v173_v261cap1.csv"
EXPLORATORY_NAME = "submission_v271_server_exploratory__v173_v261cap1.csv"


def validate_submission(df: pd.DataFrame, path: Path) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != 1845:
        raise ValueError(f"{path} rows={len(df)} expected 1845")
    server = pd.to_numeric(df["serverGetPoint"], errors="coerce")
    if server.isna().any() or not server.between(0.0, 1.0).all():
        raise ValueError(f"{path} has invalid serverGetPoint values")


def no_ttmatch_path_guard(paths: list[Path | str]) -> None:
    bad = [str(path) for path in paths if "TTMATCH" in str(path).upper()]
    if bad:
        raise ValueError(f"TTMATCH is banned from clean branch: {bad}")


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    no_ttmatch_path_guard([path])
    df = pd.read_csv(path)
    validate_submission(df, path)
    return df


def best_row(search: pd.DataFrame, verdict_token: str, mad_limit: float) -> pd.Series:
    verdict = search["verdict"].astype(str).str.contains(verdict_token, case=False, regex=False, na=False)
    mad = pd.to_numeric(search["server_mad_vs_anchor"], errors="coerce").le(float(mad_limit))
    rows = search[verdict & mad].copy()
    if rows.empty:
        raise ValueError(f"No V269 row matched verdict={verdict_token} mad<={mad_limit}")
    rows["delta_vs_proxy_base"] = pd.to_numeric(rows["delta_vs_proxy_base"], errors="coerce")
    rows["server_auc"] = pd.to_numeric(rows["server_auc"], errors="coerce")
    rows["server_mad_vs_anchor"] = pd.to_numeric(rows["server_mad_vs_anchor"], errors="coerce")
    return rows.sort_values(
        ["delta_vs_proxy_base", "server_auc", "server_mad_vs_anchor"],
        ascending=[False, False, True],
    ).iloc[0]


def repackage(row: pd.Series, output_name: str, anchor: pd.DataFrame) -> dict[str, object]:
    source = Path(str(row["path"]))
    source_df = load_submission(source)
    if not source_df["rally_uid"].equals(anchor["rally_uid"]):
        raise ValueError(f"{source} rally_uid does not match anchor")
    if not source_df["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
        raise ValueError(f"{source} changes actionId")
    if not source_df["pointId"].astype(int).equals(anchor["pointId"].astype(int)):
        raise ValueError(f"{source} changes pointId")

    out = anchor.copy()
    out["serverGetPoint"] = source_df["serverGetPoint"].astype(float).to_numpy()
    validate_submission(out, OUTDIR / output_name)

    output_path = OUTDIR / output_name
    out.to_csv(output_path, index=False, float_format="%.8f")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / output_name
    shutil.copy2(output_path, upload_path)

    server_mad = float(
        np.mean(
            np.abs(
                out["serverGetPoint"].to_numpy(dtype=float)
                - anchor["serverGetPoint"].to_numpy(dtype=float)
            )
        )
    )
    return {
        "candidate": output_name,
        "path": str(output_path),
        "upload_path": str(upload_path),
        "source_candidate": row["candidate"],
        "source_path": str(source),
        "server_auc": float(row["server_auc"]),
        "delta_vs_proxy_base": float(row["delta_vs_proxy_base"]),
        "server_mad_vs_anchor": server_mad,
        "server_corr_vs_anchor": float(row["server_corr_vs_anchor"]),
        "risk_tier": row["risk_tier"],
        "verdict": row["verdict"],
    }


def write_report(records: list[dict[str, object]]) -> None:
    lines = [
        "# V271 Server Microblend Probe",
        "",
        "Repackages clean V269 server-only candidates while keeping V173 action and V261 cap1 point fixed.",
        "",
        "## Policy",
        "",
        "- No TTMATCH input.",
        "- No old-server or old-test labels.",
        "- No manual public-label edits.",
        "- No automatic upload; files are copied only to the local upload candidate directory.",
        "",
        "## Candidates",
        "",
        "| rank | candidate | source | delta | server_mad | verdict |",
        "|---:|---|---|---:|---:|---|",
    ]
    for i, row in enumerate(records, start=1):
        lines.append(
            f"| {i} | `{row['candidate']}` | `{row['source_candidate']}` | "
            f"{row['delta_vs_proxy_base']:.6f} | {row['server_mad_vs_anchor']:.6f} | "
            f"`{row['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Low-risk first probe: `{SAFE_NAME}`",
            f"- Exploratory follow-up only if safe probe is positive: `{EXPLORATORY_NAME}`",
            f"- Search CSV: `{SEARCH_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    no_ttmatch_path_guard([ANCHOR_PATH, V269_SEARCH])

    if not V269_SEARCH.exists():
        raise FileNotFoundError(V269_SEARCH)
    search = pd.read_csv(V269_SEARCH)
    required = {"candidate", "path", "server_auc", "delta_vs_proxy_base", "server_mad_vs_anchor", "server_corr_vs_anchor", "risk_tier", "verdict"}
    missing = required - set(search.columns)
    if missing:
        raise ValueError(f"{V269_SEARCH} missing columns: {sorted(missing)}")

    anchor = load_submission(ANCHOR_PATH)
    safe = best_row(search, "CANDIDATE", 0.002)
    exploratory = best_row(search, "EXPLORATORY", 0.003)

    records = [
        repackage(safe, SAFE_NAME, anchor),
        repackage(exploratory, EXPLORATORY_NAME, anchor),
    ]
    pd.DataFrame(records).to_csv(SEARCH_PATH, index=False)
    write_report(records)

    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "safe_candidate": records[0]["source_candidate"],
                "exploratory_candidate": records[1]["source_candidate"],
                "generated": [row["candidate"] for row in records],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
