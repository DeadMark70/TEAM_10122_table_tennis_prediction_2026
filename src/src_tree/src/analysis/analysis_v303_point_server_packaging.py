"""V303 combined point-server packaging sweep.

Package selected low-churn point candidates with the V300 best-safe server.
The output is clean-only: no TTMATCH, no old-server, and no upload copy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "v303_point_server_packaging"
ANCHOR_V300_PATH = (
    ROOT
    / "v300_clean_server_blend_recycler"
    / "submission_v300_best_safe_repack__v173action_v261point_server.csv"
)
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
PACKAGE_SEARCH_PATH = OUT_DIR / "v303_package_search.csv"
REPORT_JSON_PATH = OUT_DIR / "v303_report.json"
REPORT_MD_PATH = OUT_DIR / "v303_report.md"
BANNED_PATH_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER")


@dataclass(frozen=True)
class CandidateSpec:
    package_name: str
    source_candidate: str
    source_path: str
    source_search_path: str
    source_local_delta: float | None = None
    source_local_delta_column: str = ""
    source_public_like_delta: float | None = None


BASE_CANDIDATES = [
    (
        "v303_v298_no_point0_serverv300",
        "v298_support_no_point0_cap0p01",
        "v298_action_point_support_prior/v298_candidate_search.csv",
    ),
    (
        "v303_v298_long789005_serverv300",
        "v298_support_long789_cap0p005",
        "v298_action_point_support_prior/v298_candidate_search.csv",
    ),
    (
        "v303_v298_long789010_serverv300",
        "v298_support_long789_cap0p01",
        "v298_action_point_support_prior/v298_candidate_search.csv",
    ),
    (
        "v303_v295_rare134_serverv300",
        "v295_rare134_ovr_cap0p0025",
        "v295_true_oof_point_specialists/v295_candidate_search.csv",
    ),
    (
        "v303_v297_norare_serverv300",
        "v297_no_rare134_cap0p005",
        "v297_multisource_point_agreement/v297_candidate_search.csv",
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def no_banned_path_guard(paths: list[Path | str]) -> None:
    bad = []
    for path in paths:
        upper = str(path).upper()
        if any(token in upper for token in BANNED_PATH_TOKENS):
            bad.append(str(path))
    if bad:
        raise ValueError(f"Banned clean-branch input path: {bad}")


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={SUBMISSION_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
    if not df["pointId"].between(0, 9).all():
        raise ValueError("pointId out of range")
    if not df["actionId"].between(0, 18).all():
        raise ValueError("actionId out of range")
    server = pd.to_numeric(df["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    if not server.between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")


def load_submission(path: Path, *, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    no_banned_path_guard([path])
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    validate_submission_frame(df, expected_rows=expected_rows)
    return df


def build_package_submission(
    v300_anchor: pd.DataFrame,
    point_source: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    validate_submission_frame(v300_anchor, expected_rows=expected_rows)
    validate_submission_frame(point_source, expected_rows=expected_rows)
    if not point_source["rally_uid"].equals(v300_anchor["rally_uid"]):
        raise ValueError("point source rally_uid does not match V300 anchor")
    out = v300_anchor.copy()
    out["pointId"] = point_source["pointId"].astype(int).to_numpy()
    out = out.loc[:, SUBMISSION_COLUMNS]
    validate_submission_frame(out, expected_rows=expected_rows)
    return out


def point0_rate_delta(anchor_point: np.ndarray, candidate_point: np.ndarray) -> float:
    anchor = np.asarray(anchor_point, dtype=int)
    candidate = np.asarray(candidate_point, dtype=int)
    return float(np.mean(candidate == 0) - np.mean(anchor == 0))


def recommendation_for(source_local_delta: float | None) -> str:
    if source_local_delta is not None and np.isfinite(source_local_delta) and source_local_delta >= 0.001:
        return "REVIEW_UPLOAD"
    return "DO_NOT_UPLOAD"


def server_mad(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def summarize_package(
    spec: CandidateSpec,
    v300_anchor: pd.DataFrame,
    packaged: pd.DataFrame,
    output_path: str,
) -> dict[str, Any]:
    anchor_point = v300_anchor["pointId"].astype(int).to_numpy()
    cand_point = packaged["pointId"].astype(int).to_numpy()
    point_changed = int(np.sum(cand_point != anchor_point))
    action_changed = int(
        np.sum(packaged["actionId"].astype(int).to_numpy() != v300_anchor["actionId"].astype(int).to_numpy())
    )
    mad = server_mad(packaged["serverGetPoint"], v300_anchor["serverGetPoint"])
    return {
        "candidate": spec.package_name,
        "path": output_path,
        "source_candidate": spec.source_candidate,
        "source_path": spec.source_path,
        "source_search_path": spec.source_search_path,
        "point_changed_rows_vs_v300": point_changed,
        "point_churn_vs_v300": float(point_changed / len(v300_anchor)),
        "action_changed_rows_vs_v300": action_changed,
        "server_mad_vs_v300": mad,
        "point0_rate_delta_vs_v300": point0_rate_delta(anchor_point, cand_point),
        "inherited_point_local_delta": spec.source_local_delta,
        "inherited_point_local_delta_column": spec.source_local_delta_column,
        "inherited_point_public_like_delta": spec.source_public_like_delta,
        "recommendation": recommendation_for(spec.source_local_delta),
    }


def write_submission(path: Path, df: pd.DataFrame) -> None:
    no_banned_path_guard([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission_frame(df, expected_rows=len(df))
    df.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def first_present_number(row: pd.Series, columns: list[str]) -> tuple[float | None, str]:
    for column in columns:
        if column not in row or pd.isna(row[column]):
            continue
        value = float(pd.to_numeric(row[column], errors="coerce"))
        if np.isfinite(value):
            return value, column
    return None, ""


def load_candidate_spec(package_name: str, source_candidate: str, source_search_path: str) -> CandidateSpec:
    search_path = ROOT / source_search_path
    no_banned_path_guard([search_path])
    if not search_path.exists():
        raise FileNotFoundError(search_path)
    search = pd.read_csv(search_path)
    match = search[search["candidate"].astype(str).eq(source_candidate)]
    if match.empty:
        raise ValueError(f"{source_candidate} not found in {source_search_path}")
    row = match.iloc[0]
    source_path = Path(str(row["path"]))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    local_delta, delta_column = first_present_number(
        row,
        [
            "delta_vs_v294_base",
            "delta_vs_aligned_base",
            "point_delta_vs_base",
            "local_delta",
        ],
    )
    public_like_delta, _public_column = first_present_number(row, ["public_like_delta"])
    return CandidateSpec(
        package_name=package_name,
        source_candidate=source_candidate,
        source_path=relative_path(source_path),
        source_search_path=source_search_path,
        source_local_delta=local_delta,
        source_local_delta_column=delta_column,
        source_public_like_delta=public_like_delta,
    )


def output_filename(package_name: str) -> str:
    return f"submission_{package_name}.csv"


def choose_ultra_safe_source(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in records if int(row["point_changed_rows_vs_v300"]) <= 9]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            float(row["inherited_point_local_delta"])
            if row["inherited_point_local_delta"] is not None and np.isfinite(row["inherited_point_local_delta"])
            else -np.inf,
            -int(row["point_changed_rows_vs_v300"]),
            str(row["candidate"]),
        ),
        reverse=True,
    )[0]


def write_reports(search: pd.DataFrame, summary: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V303 combined point-server packaging sweep",
        "",
        f"Anchor: `{summary['anchor_v300']}`",
        "Clean policy: no TTMATCH, no old-server, no upload copy.",
        f"Generated submissions: `{summary['generated_submission_count']}`",
        f"Upload recommendation: `{summary['upload_recommendation']}`",
        "",
        "## Candidates",
        "",
        "| candidate | source | point rows | action rows | server MAD | point0 delta | local delta | recommendation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.to_dict("records"):
        local_delta = row["inherited_point_local_delta"]
        local_text = "" if pd.isna(local_delta) else f"{float(local_delta):.6f}"
        lines.append(
            f"| `{row['candidate']}` | `{row['source_candidate']}` | "
            f"{int(row['point_changed_rows_vs_v300'])} | "
            f"{int(row['action_changed_rows_vs_v300'])} | "
            f"{float(row['server_mad_vs_v300']):.8f} | "
            f"{float(row['point0_rate_delta_vs_v300']):.6f} | "
            f"{local_text} | `{row['recommendation']}` |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Recommendation is DO_NOT_UPLOAD unless the inherited point source local delta is at least +0.001.",
            f"Search CSV: `{relative_path(PACKAGE_SEARCH_PATH)}`",
            "",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v300_anchor = load_submission(ANCHOR_V300_PATH)
    no_banned_path_guard([ANCHOR_V300_PATH])

    specs = [load_candidate_spec(*item) for item in BASE_CANDIDATES]
    records: list[dict[str, Any]] = []
    generated: list[str] = []
    packaged_by_name: dict[str, pd.DataFrame] = {}

    for spec in specs:
        source = load_submission(ROOT / spec.source_path)
        packaged = build_package_submission(v300_anchor, source)
        out_path = OUT_DIR / output_filename(spec.package_name)
        write_submission(out_path, packaged)
        path_text = relative_path(out_path)
        records.append(summarize_package(spec, v300_anchor, packaged, path_text))
        generated.append(path_text)
        packaged_by_name[spec.package_name] = packaged

    ultra_source = choose_ultra_safe_source(records)
    if ultra_source is not None:
        ultra_spec = CandidateSpec(
            package_name="v303_ultra_safe_serverv300",
            source_candidate=str(ultra_source["source_candidate"]),
            source_path=str(ultra_source["source_path"]),
            source_search_path=str(ultra_source["source_search_path"]),
            source_local_delta=ultra_source["inherited_point_local_delta"],
            source_local_delta_column=str(ultra_source["inherited_point_local_delta_column"]),
            source_public_like_delta=ultra_source["inherited_point_public_like_delta"],
        )
        packaged = packaged_by_name[str(ultra_source["candidate"])]
        out_path = OUT_DIR / output_filename(ultra_spec.package_name)
        write_submission(out_path, packaged)
        path_text = relative_path(out_path)
        row = summarize_package(ultra_spec, v300_anchor, packaged, path_text)
        row["ultra_safe_source_package"] = ultra_source["candidate"]
        records.append(row)
        generated.append(path_text)

    search = pd.DataFrame(records).sort_values(
        ["recommendation", "inherited_point_local_delta", "point_changed_rows_vs_v300"],
        ascending=[False, False, True],
    )
    search.to_csv(PACKAGE_SEARCH_PATH, index=False)

    action_fixed = bool(search["action_changed_rows_vs_v300"].eq(0).all())
    server_fixed = bool(np.allclose(search["server_mad_vs_v300"].astype(float).to_numpy(), 0.0))
    summary = {
        "version": "V303",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": relative_path(OUT_DIR),
        "anchor_v300": relative_path(ANCHOR_V300_PATH),
        "current_best_clean_public": {
            "name": "V300 best_safe_repack",
            "public_lb": 0.3576975,
            "components": "V173 action + V261 point + V300 server",
        },
        "policy": {
            "clean_only": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_copy": True,
        },
        "fixed_output": {
            "action_changed_rows_vs_v300_all_zero": action_fixed,
            "server_mad_vs_v300_all_zero": server_fixed,
        },
        "generated_submissions": generated,
        "generated_submission_count": len(generated),
        "package_search": relative_path(PACKAGE_SEARCH_PATH),
        "report_md": relative_path(REPORT_MD_PATH),
        "ultra_safe_generated": ultra_source is not None,
        "ultra_safe_source_package": None if ultra_source is None else ultra_source["candidate"],
        "upload_recommendation": "REVIEW_UPLOAD"
        if search["recommendation"].eq("REVIEW_UPLOAD").any()
        else "DO_NOT_UPLOAD",
        "recommendation_gate": "DO_NOT_UPLOAD unless inherited point source local delta >= 0.001.",
    }
    write_reports(search, summary)
    return summary


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": report["outdir"],
                "generated_submissions": report["generated_submission_count"],
                "ultra_safe_generated": report["ultra_safe_generated"],
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
