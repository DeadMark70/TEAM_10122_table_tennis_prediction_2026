"""V309 server packaging around the V306 point0-positive anchor.

Package the fixed V306 p0 cap0p01 point/action anchor with clean server
sources from R121/V300/V302 artifacts. Outputs stay local under
v309_v306_server_packaging.
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


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v309_v306_server_packaging"
V306_ANCHOR_PATH = (
    ROOT
    / "v306_point0_addition_probe"
    / "submission_v306_p0_cap0p01__v173action_v300server.csv"
)
V300_BEST_PATH = (
    ROOT
    / "v300_clean_server_blend_recycler"
    / "submission_v300_best_safe_repack__v173action_v261point_server.csv"
)
R121_FALLBACK_PATH = (
    ROOT
    / "v261_action_conditioned_point_residual"
    / "submission_v261_cap0p01__v173action_r121server.csv"
)
V300_DIR = ROOT / "v300_clean_server_blend_recycler"
V302_DIR = ROOT / "v302_clean_server_calibration_sweep"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
BANNED_PATH_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER")
SEARCH_PATH = OUT_DIR / "v309_server_packaging_search.csv"
REPORT_JSON_PATH = OUT_DIR / "v309_report.json"
REPORT_MD_PATH = OUT_DIR / "v309_report.md"


@dataclass(frozen=True)
class ServerSpec:
    source_key: str
    source_path: str
    source_family: str
    source_is_clean: bool


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
        return path.resolve().relative_to(ROOT).as_posix()
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


def source_is_clean(path: Path | str) -> bool:
    upper = str(path).upper()
    return not any(token in upper for token in BANNED_PATH_TOKENS)


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={SUBMISSION_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
    if not df["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of range")
    if not df["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
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


def corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def build_server_packaged_submission(
    v306_anchor: pd.DataFrame,
    server_source: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    validate_submission_frame(v306_anchor, expected_rows=expected_rows)
    validate_submission_frame(server_source, expected_rows=expected_rows)
    if not server_source["rally_uid"].equals(v306_anchor["rally_uid"]):
        raise ValueError("server source rally_uid does not match V306 anchor")
    out = v306_anchor.copy()
    out["serverGetPoint"] = pd.to_numeric(server_source["serverGetPoint"], errors="raise").to_numpy(dtype=float)
    out = out.loc[:, SUBMISSION_COLUMNS]
    validate_submission_frame(out, expected_rows=expected_rows)
    return out


def decision_for_server(server_mad: float, clean: bool) -> str:
    if clean and np.isfinite(server_mad) and float(server_mad) <= 0.02:
        return "REVIEW_SERVER"
    return "DIAGNOSTIC"


def summarize_server_variant(
    spec: ServerSpec,
    v306_anchor: pd.DataFrame,
    packaged: pd.DataFrame,
    output_path: str,
) -> dict[str, Any]:
    validate_submission_frame(v306_anchor, expected_rows=len(v306_anchor))
    validate_submission_frame(packaged, expected_rows=len(v306_anchor))
    anchor_server = v306_anchor["serverGetPoint"].to_numpy(dtype=float)
    server = packaged["serverGetPoint"].to_numpy(dtype=float)
    mad = float(np.mean(np.abs(server - anchor_server)))
    action_changed = int(
        np.sum(packaged["actionId"].astype(int).to_numpy() != v306_anchor["actionId"].astype(int).to_numpy())
    )
    point_changed = int(
        np.sum(packaged["pointId"].astype(int).to_numpy() != v306_anchor["pointId"].astype(int).to_numpy())
    )
    return {
        "candidate": Path(output_path).name,
        "path": output_path,
        "server_source": spec.source_key,
        "server_source_path": spec.source_path,
        "source_family": spec.source_family,
        "source_is_clean": bool(spec.source_is_clean),
        "server_mad_vs_v306_best_server": mad,
        "server_corr_vs_v306_best_server": corr(server, anchor_server),
        "server_min": float(np.min(server)),
        "server_max": float(np.max(server)),
        "row_count": int(len(packaged)),
        "action_changed_rows_vs_v306_anchor": action_changed,
        "point_changed_rows_vs_v306_anchor": point_changed,
        "decision": decision_for_server(mad, spec.source_is_clean),
    }


def write_submission(path: Path, df: pd.DataFrame) -> None:
    no_banned_path_guard([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission_frame(df, expected_rows=len(df))
    df.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def _existing_spec(path: Path, source_key: str, family: str) -> ServerSpec | None:
    if not path.exists():
        return None
    return ServerSpec(
        source_key=source_key,
        source_path=relative_path(path),
        source_family=family,
        source_is_clean=source_is_clean(path),
    )


def discover_server_specs() -> list[ServerSpec]:
    specs: list[ServerSpec] = []
    seen: set[str] = set()

    def add(spec: ServerSpec | None) -> None:
        if spec is None or spec.source_path in seen:
            return
        specs.append(spec)
        seen.add(spec.source_path)

    add(_existing_spec(R121_FALLBACK_PATH, "r121_v261_cap0p01_fallback", "r121_fallback"))
    add(_existing_spec(V300_BEST_PATH, "v300_best_safe_repack", "v300"))

    for kind in ("mean", "rankavg"):
        for token in ("w0p005", "w0p01", "w0p02"):
            path = V300_DIR / f"submission_v300_{kind}_{token}__v173action_v261point_server.csv"
            add(_existing_spec(path, f"v300_{kind}_{token}", "v300"))

    for path in sorted(V302_DIR.glob("submission_v302_*.csv")):
        name_upper = path.name.upper()
        if "OLD" in name_upper or "TTMATCH" in name_upper:
            continue
        source_key = path.stem.replace("submission_", "")
        add(_existing_spec(path, source_key, "v302"))

    if not specs:
        raise ValueError("No eligible clean server sources found.")
    no_banned_path_guard([spec.source_path for spec in specs])
    return specs


def output_filename(spec: ServerSpec) -> str:
    key = spec.source_key.replace(".", "p").replace("-", "_")
    return f"submission_v309_{key}__v306p0cap0p01_v173action.csv"


def write_reports(search: pd.DataFrame, summary: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V309 V306 Server Packaging",
        "",
        "Server-only packaging around fixed V306 p0 cap0p01 point and V173 action.",
        "",
        f"- Anchor: `{summary['v306_anchor_path']}`",
        f"- Generated submissions: `{summary['generated_submission_count']}`",
        f"- Review server variants: `{summary['review_server_count']}`",
        "",
        "## Top Variants",
        "",
        "| candidate | source | MAD | corr | min | max | rows | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.head(15).to_dict("records"):
        lines.append(
            f"| `{row['candidate']}` | `{row['server_source']}` | "
            f"{float(row['server_mad_vs_v306_best_server']):.8f} | "
            f"{float(row['server_corr_vs_v306_best_server']):.8f} | "
            f"{float(row['server_min']):.8f} | {float(row['server_max']):.8f} | "
            f"{int(row['row_count'])} | `{row['decision']}` |"
        )
    lines.extend(["", f"Search CSV: `{relative_path(SEARCH_PATH)}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v306_anchor = load_submission(V306_ANCHOR_PATH)
    specs = discover_server_specs()

    records: list[dict[str, Any]] = []
    generated: list[str] = []
    for spec in specs:
        source = load_submission(ROOT / spec.source_path)
        packaged = build_server_packaged_submission(v306_anchor, source)
        out_path = OUT_DIR / output_filename(spec)
        write_submission(out_path, packaged)
        out_text = relative_path(out_path)
        records.append(summarize_server_variant(spec, v306_anchor, packaged, out_text))
        generated.append(out_text)

    search = pd.DataFrame(records).sort_values(
        ["decision", "server_mad_vs_v306_best_server", "server_source"],
        ascending=[False, True, True],
    )
    search.to_csv(SEARCH_PATH, index=False)

    review = search[search["decision"].eq("REVIEW_SERVER")]
    summary = {
        "version": "V309",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": relative_path(OUT_DIR),
        "v306_anchor_path": relative_path(V306_ANCHOR_PATH),
        "fixed_anchor": {
            "point": "V306 p0 cap0p01",
            "action": "V173",
            "source_public_pl": 0.3577905,
            "source_submission": V306_ANCHOR_PATH.name,
        },
        "policy": {
            "clean_sources_only": True,
            "server_only": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_copy": True,
        },
        "generated_submissions": generated,
        "generated_submission_count": len(generated),
        "search_path": relative_path(SEARCH_PATH),
        "report_json_path": relative_path(REPORT_JSON_PATH),
        "report_md_path": relative_path(REPORT_MD_PATH),
        "review_server_count": int(len(review)),
        "top_server_variants": review.head(10).to_dict(orient="records"),
        "decision_rule": "REVIEW_SERVER if server MAD <= 0.02 and source is clean; otherwise DIAGNOSTIC.",
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
                "review_server_count": report["review_server_count"],
                "top": [
                    row["server_source"]
                    for row in report["top_server_variants"][:5]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
