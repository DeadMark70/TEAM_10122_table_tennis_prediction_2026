"""V321 server robustness and rank-blend validation.

Server-only rank/value blends around the V306 clean action/point anchor.
The script reads clean V300/V302/V319 server artifacts if present, validates
source agreement, and writes local-only outputs under
v321_server_robust_rankblend.
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
OUT_DIR = ROOT / "v321_server_robust_rankblend"
ANCHOR_PATH = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"

V319_SEARCH_PATH = ROOT / "v319_clean_server_value_state" / "v319_server_value_state_search.csv"
V300_SEARCH_PATH = ROOT / "v300_clean_server_blend_recycler" / "v300_server_search.csv"
V302_SEARCH_PATH = ROOT / "v302_clean_server_calibration_sweep" / "v302_server_search.csv"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
BANNED_PATH_TOKENS = (
    "TTMATCH",
    "OLD_SERVER",
    "OLDSERVER",
    "OLD-SERVER",
    "UPLOAD_CANDIDATES_20260519",
    "SUBMISSIONS/SELECTED",
    "SUBMISSIONS\\SELECTED",
)

SEARCH_PATH = OUT_DIR / "v321_server_rankblend_search.csv"
REPORT_JSON_PATH = OUT_DIR / "v321_report.json"
REPORT_MD_PATH = OUT_DIR / "v321_report.md"

EXPECTED_EXPORTS = {
    "rankblend_mad0p001": "submission_v321_server_rankblend_mad0p001__v173action_v306point.csv",
    "rankblend_mad0p002": "submission_v321_server_rankblend_mad0p002__v173action_v306point.csv",
    "value_consensus_mad0p002": "submission_v321_server_value_consensus_mad0p002__v173action_v306point.csv",
    "robust_mean_mad0p003": "submission_v321_server_robust_mean_mad0p003__v173action_v306point.csv",
    "temperature_mad0p005": "submission_v321_server_temperature_mad0p005__v173action_v306point.csv",
}


@dataclass(frozen=True)
class ServerSource:
    name: str
    family: str
    server: np.ndarray
    evidence_delta: float
    path: str


@dataclass(frozen=True)
class ServerCandidate:
    key: str
    filename: str
    kind: str
    target_mad: float
    server: np.ndarray
    source_count: int
    source_family_count: int
    min_agree_sources: int
    mean_agree_sources: float
    evidence_delta: float
    source_names: tuple[str, ...]


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


def relative_path(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path_obj.as_posix()


def validate_output_path(path: Path | str) -> None:
    text = str(path)
    upper = text.upper()
    bad = [token for token in BANNED_PATH_TOKENS if token in upper]
    if bad:
        raise ValueError(f"banned V321 path {text!r}: {bad}")
    path_obj = Path(path)
    if path_obj.is_absolute():
        try:
            path_obj.resolve().relative_to(OUT_DIR.resolve())
        except ValueError as exc:
            raise ValueError(f"V321 outputs must stay under {relative_path(OUT_DIR)}: {text}") from exc


def no_banned_input_guard(paths: list[Path | str]) -> None:
    bad = []
    for path in paths:
        upper = str(path).upper()
        if "TTMATCH" in upper or "OLD_SERVER" in upper or "OLDSERVER" in upper or "OLD-SERVER" in upper:
            bad.append(str(path))
    if bad:
        raise ValueError(f"banned clean-branch input path: {bad}")


def cap_prob(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


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
    no_banned_input_guard([path])
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    validate_submission_frame(df, expected_rows=expected_rows)
    return df


def rank_normalize_to_anchor(source: np.ndarray | pd.Series, anchor: np.ndarray | pd.Series) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    finite = source_arr[np.isfinite(source_arr)]
    fill = float(np.median(finite)) if len(finite) else 0.0
    source_arr = np.nan_to_num(source_arr, nan=fill, posinf=fill, neginf=fill)
    anchor_sorted = np.sort(cap_prob(anchor))
    if len(source_arr) != len(anchor_sorted):
        raise ValueError("source and anchor lengths differ")
    if len(source_arr) == 1:
        return np.array([anchor_sorted[0]], dtype=float)
    ranks = pd.Series(source_arr).rank(method="average").to_numpy(dtype=float) - 1.0
    return cap_prob(np.interp(ranks, np.arange(len(anchor_sorted), dtype=float), anchor_sorted))


def blend_to_target_mad(anchor: np.ndarray | pd.Series, target: np.ndarray | pd.Series, *, target_mad: float) -> np.ndarray:
    if not np.isfinite(target_mad) or target_mad < 0.0:
        raise ValueError(f"target_mad must be finite and non-negative, got {target_mad}")
    anchor_arr = cap_prob(anchor)
    target_arr = cap_prob(target)
    if len(anchor_arr) != len(target_arr):
        raise ValueError("anchor and target lengths differ")
    delta = target_arr - anchor_arr
    full_mad = float(np.mean(np.abs(delta)))
    if full_mad == 0.0 or target_mad == 0.0:
        return anchor_arr.copy()
    scale = min(1.0, float(target_mad) / full_mad)
    return cap_prob(anchor_arr + scale * delta)


def apply_temperature(server: np.ndarray | pd.Series, *, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    p = cap_prob(server)
    logits = np.log(p) - np.log1p(-p)
    scaled = logits / float(temperature)
    return cap_prob(1.0 / (1.0 + np.exp(-scaled)))


def direction_agreement(anchor: np.ndarray, target: np.ndarray, source_matrix: np.ndarray) -> np.ndarray:
    anchor_arr = cap_prob(anchor)
    target_delta = cap_prob(target) - anchor_arr
    source_delta = np.asarray(source_matrix, dtype=float) - anchor_arr[:, None]
    target_sign = np.sign(target_delta)
    source_sign = np.sign(source_delta)
    same = (target_sign[:, None] != 0) & (source_sign == target_sign[:, None])
    return same.sum(axis=1).astype(int)


def _resolve_source_path(raw_path: object, search_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    sibling = search_path.parent / path.name
    if sibling.exists():
        return sibling
    return candidate


def _metric_value(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            value = pd.to_numeric(row[name], errors="coerce")
            if pd.notna(value):
                return float(value)
    return float("nan")


def _source_from_row(
    *,
    row: pd.Series,
    search_path: Path,
    family: str,
    anchor: pd.DataFrame,
    evidence_names: list[str],
) -> ServerSource | None:
    if "path" not in row or "candidate" not in row:
        return None
    path = _resolve_source_path(row["path"], search_path)
    if not path.exists():
        return None
    source = load_submission(path)
    if not source["rally_uid"].astype(int).equals(anchor["rally_uid"].astype(int)):
        raise ValueError(f"{relative_path(path)} rally_uid does not match V306 anchor")
    evidence = _metric_value(row, evidence_names)
    if not np.isfinite(evidence):
        evidence = 0.0
    return ServerSource(
        name=str(row["candidate"]),
        family=family,
        server=cap_prob(source["serverGetPoint"].to_numpy(dtype=float)),
        evidence_delta=float(evidence),
        path=relative_path(path),
    )


def _fingerprint(server: np.ndarray) -> bytes:
    return np.round(cap_prob(server), 10).tobytes()


def load_clean_server_sources(anchor: pd.DataFrame) -> list[ServerSource]:
    no_banned_input_guard([V319_SEARCH_PATH, V300_SEARCH_PATH, V302_SEARCH_PATH])
    by_fp: dict[bytes, ServerSource] = {}

    search_specs = [
        (V319_SEARCH_PATH, "v319", ["oof_auc_delta_vs_anchor"]),
        (V300_SEARCH_PATH, "v300", ["proxy_delta_vs_proxy_base", "delta_vs_proxy_base"]),
        (V302_SEARCH_PATH, "v302", ["proxy_delta_vs_v300_estimate"]),
    ]
    for search_path, family, evidence_names in search_specs:
        if not search_path.exists():
            continue
        search = pd.read_csv(search_path)
        for _, row in search.iterrows():
            source = _source_from_row(
                row=row,
                search_path=search_path,
                family=family,
                anchor=anchor,
                evidence_names=evidence_names,
            )
            if source is None:
                continue
            fp = _fingerprint(source.server)
            if fp not in by_fp or source.evidence_delta > by_fp[fp].evidence_delta:
                by_fp[fp] = source

    sources = list(by_fp.values())
    if len(sources) < 2:
        raise ValueError(f"Need at least two clean server sources, found {len(sources)}")
    return sources


def _mean_finite(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return 0.0
    return float(np.mean(finite))


def _rank_matrix(anchor_server: np.ndarray, sources: list[ServerSource]) -> np.ndarray:
    return np.column_stack([rank_normalize_to_anchor(source.server, anchor_server) for source in sources])


def _trimmed_mean(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] < 5:
        return np.mean(matrix, axis=1)
    ordered = np.sort(matrix, axis=1)
    return np.mean(ordered[:, 1:-1], axis=1)


def _target_for_kind(anchor_server: np.ndarray, sources: list[ServerSource], kind: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = _rank_matrix(anchor_server, sources)
    if kind == "rankblend":
        target = np.mean(matrix, axis=1)
    elif kind == "value_consensus":
        value_sources = [source for source in sources if source.family == "v319"]
        use_sources = value_sources if len(value_sources) >= 2 else sources
        matrix = _rank_matrix(anchor_server, use_sources)
        target = np.mean(matrix, axis=1)
    elif kind == "robust_mean":
        target = _trimmed_mean(matrix)
    elif kind == "temperature":
        rank_target = np.mean(matrix, axis=1)
        target = apply_temperature(rank_target, temperature=1.05)
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    return cap_prob(target), matrix


def build_candidate(
    key: str,
    anchor_server: np.ndarray,
    sources: list[ServerSource],
    *,
    target_mad: float,
    kind: str,
    filename: str,
) -> ServerCandidate:
    if len(sources) < 2:
        raise ValueError("candidate needs at least two clean server sources")
    target, agreement_matrix = _target_for_kind(anchor_server, sources, kind)
    agreement = direction_agreement(cap_prob(anchor_server), target, agreement_matrix)
    gated_target = np.where(agreement >= 2, target, cap_prob(anchor_server))
    server = blend_to_target_mad(anchor_server, gated_target, target_mad=target_mad)
    final_agreement = direction_agreement(cap_prob(anchor_server), server, agreement_matrix)
    changed = np.abs(server - cap_prob(anchor_server)) > 1e-12
    changed_agreement = final_agreement[changed]
    return ServerCandidate(
        key=key,
        filename=filename,
        kind=kind,
        target_mad=float(target_mad),
        server=server,
        source_count=len(sources),
        source_family_count=len({source.family for source in sources}),
        min_agree_sources=int(changed_agreement.min()) if len(changed_agreement) else 0,
        mean_agree_sources=float(np.mean(changed_agreement)) if len(changed_agreement) else 0.0,
        evidence_delta=_mean_finite([source.evidence_delta for source in sources]),
        source_names=tuple(source.name for source in sources),
    )


def build_packaged_submission(
    anchor: pd.DataFrame,
    candidate: ServerCandidate,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    validate_submission_frame(anchor, expected_rows=expected_rows)
    if len(anchor) != len(candidate.server):
        raise ValueError("anchor and candidate server lengths differ")
    out = anchor.copy()
    out["serverGetPoint"] = cap_prob(candidate.server)
    out = out.loc[:, SUBMISSION_COLUMNS]
    validate_submission_frame(out, expected_rows=expected_rows)
    return out


def decision_for_candidate(
    *,
    action_changed_rows: int,
    point_changed_rows: int,
    mad: float,
    server_min: float,
    server_max: float,
    min_agree_sources: int,
    source_family_count: int,
    evidence_delta: float,
) -> str:
    sane_distribution = (
        np.isfinite(server_min)
        and np.isfinite(server_max)
        and 0.0 <= server_min < server_max <= 1.0
        and (server_max - server_min) >= 0.01
    )
    if (
        action_changed_rows == 0
        and point_changed_rows == 0
        and np.isfinite(mad)
        and mad <= 0.0050000001
        and sane_distribution
        and min_agree_sources >= 2
        and source_family_count >= 2
        and np.isfinite(evidence_delta)
        and evidence_delta > -0.002
    ):
        return "REVIEW_SERVER"
    return "DIAGNOSTIC"


def summarize_candidate(candidate: ServerCandidate, anchor: pd.DataFrame, packaged: pd.DataFrame) -> dict[str, Any]:
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    server = cap_prob(packaged["serverGetPoint"].to_numpy(dtype=float))
    action_changed = int(
        np.sum(packaged["actionId"].astype(int).to_numpy() != anchor["actionId"].astype(int).to_numpy())
    )
    point_changed = int(
        np.sum(packaged["pointId"].astype(int).to_numpy() != anchor["pointId"].astype(int).to_numpy())
    )
    mad = float(np.mean(np.abs(server - anchor_server)))
    server_min = float(np.min(server))
    server_max = float(np.max(server))
    decision = decision_for_candidate(
        action_changed_rows=action_changed,
        point_changed_rows=point_changed,
        mad=mad,
        server_min=server_min,
        server_max=server_max,
        min_agree_sources=candidate.min_agree_sources,
        source_family_count=candidate.source_family_count,
        evidence_delta=candidate.evidence_delta,
    )
    return {
        "candidate": candidate.filename,
        "path": relative_path(OUT_DIR / candidate.filename),
        "key": candidate.key,
        "kind": candidate.kind,
        "target_mad": candidate.target_mad,
        "server_mad_vs_v306_server": mad,
        "server_corr_vs_v306_server": corr(server, anchor_server),
        "server_min": server_min,
        "server_max": server_max,
        "server_mean": float(np.mean(server)),
        "server_std": float(np.std(server)),
        "row_count": int(len(packaged)),
        "source_count": candidate.source_count,
        "source_family_count": candidate.source_family_count,
        "min_agree_sources": candidate.min_agree_sources,
        "mean_agree_sources": candidate.mean_agree_sources,
        "mean_source_evidence_delta": candidate.evidence_delta,
        "source_names": ";".join(candidate.source_names),
        "action_changed_rows_vs_anchor": action_changed,
        "point_changed_rows_vs_anchor": point_changed,
        "decision": decision,
    }


def write_submission(path: Path, df: pd.DataFrame) -> None:
    validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission_frame(df, expected_rows=len(df))
    df.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def write_reports(search: pd.DataFrame, summary: dict[str, Any]) -> None:
    validate_output_path(REPORT_JSON_PATH)
    validate_output_path(REPORT_MD_PATH)
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V321 Server Robust Rankblend",
        "",
        "Server-only robustness validation around the V306 action/point anchor.",
        "",
        "## Policy",
        "",
        "- No TTMATCH input.",
        "- No old-server input.",
        "- No upload-candidate or selected submission writes.",
        "- `actionId` and `pointId` remain fixed to V306.",
        "",
        "## Sources",
        "",
        f"- Unique clean server sources: `{summary['source_count']}`",
        f"- Source families: `{', '.join(summary['source_families'])}`",
        "",
        "## Candidates",
        "",
        "| candidate | kind | target MAD | actual MAD | corr | min agree | families | evidence | action | point | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"| `{row['candidate']}` | `{row['kind']}` | {float(row['target_mad']):.6f} | "
            f"{float(row['server_mad_vs_v306_server']):.8f} | "
            f"{float(row['server_corr_vs_v306_server']):.8f} | "
            f"{int(row['min_agree_sources'])} | {int(row['source_family_count'])} | "
            f"{float(row['mean_source_evidence_delta']):.6f} | "
            f"{int(row['action_changed_rows_vs_anchor'])} | "
            f"{int(row['point_changed_rows_vs_anchor'])} | `{row['decision']}` |"
        )
    lines.extend(["", f"Search CSV: `{relative_path(SEARCH_PATH)}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_all_candidates(anchor_server: np.ndarray, sources: list[ServerSource]) -> list[ServerCandidate]:
    specs = [
        ("rankblend_mad0p001", "rankblend", 0.001),
        ("rankblend_mad0p002", "rankblend", 0.002),
        ("value_consensus_mad0p002", "value_consensus", 0.002),
        ("robust_mean_mad0p003", "robust_mean", 0.003),
        ("temperature_mad0p005", "temperature", 0.005),
    ]
    return [
        build_candidate(
            key,
            anchor_server,
            sources,
            target_mad=target_mad,
            kind=kind,
            filename=EXPECTED_EXPORTS[key],
        )
        for key, kind, target_mad in specs
    ]


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_output_path(OUT_DIR / "probe.csv")
    no_banned_input_guard([ANCHOR_PATH, V319_SEARCH_PATH, V300_SEARCH_PATH, V302_SEARCH_PATH])

    anchor = load_submission(ANCHOR_PATH)
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    sources = load_clean_server_sources(anchor)
    candidates = build_all_candidates(anchor_server, sources)

    rows: list[dict[str, Any]] = []
    generated: list[str] = []
    for candidate in candidates:
        packaged = build_packaged_submission(anchor, candidate)
        out_path = OUT_DIR / candidate.filename
        write_submission(out_path, packaged)
        generated.append(relative_path(out_path))
        rows.append(summarize_candidate(candidate, anchor, packaged))

    search = pd.DataFrame(rows)
    search["decision_rank"] = search["decision"].map({"REVIEW_SERVER": 0, "DIAGNOSTIC": 1}).fillna(9)
    search = search.sort_values(
        ["decision_rank", "target_mad", "mean_source_evidence_delta"],
        ascending=[True, True, False],
    ).drop(columns=["decision_rank"])
    validate_output_path(SEARCH_PATH)
    search.to_csv(SEARCH_PATH, index=False)
    review = search[search["decision"].eq("REVIEW_SERVER")]

    summary = {
        "version": "V321",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": relative_path(OUT_DIR),
        "anchor_path": relative_path(ANCHOR_PATH),
        "policy": {
            "server_only": True,
            "fixed_action_point_anchor": "V306 submission_v306_p0_cap0p01__v173action_v300server.csv",
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_copy": True,
            "review_server_requires_two_source_direction_agreement": True,
        },
        "source_count": len(sources),
        "source_families": sorted({source.family for source in sources}),
        "source_inventory": [
            {
                "name": source.name,
                "family": source.family,
                "path": source.path,
                "evidence_delta": source.evidence_delta,
                "corr_vs_v306_server": corr(source.server, anchor_server),
                "mad_vs_v306_server": float(np.mean(np.abs(source.server - anchor_server))),
            }
            for source in sources
        ],
        "generated_submissions": generated,
        "generated_submission_count": len(generated),
        "search_path": relative_path(SEARCH_PATH),
        "report_json_path": relative_path(REPORT_JSON_PATH),
        "report_md_path": relative_path(REPORT_MD_PATH),
        "review_server_count": int(len(review)),
        "top_candidates": review.head(10).to_dict(orient="records"),
        "decision_rule": (
            "REVIEW_SERVER only if server-only, MAD <= 0.005, sane distribution, "
            "at least two clean sources agree on direction, at least two source families are present, "
            "and mean source evidence is not materially negative."
        ),
    }
    write_reports(search, summary)
    return summary


def main() -> None:
    summary = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": summary["outdir"],
                "generated_submission_count": summary["generated_submission_count"],
                "review_server_count": summary["review_server_count"],
                "source_count": summary["source_count"],
                "top": [row["candidate"] for row in summary["top_candidates"][:5]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
