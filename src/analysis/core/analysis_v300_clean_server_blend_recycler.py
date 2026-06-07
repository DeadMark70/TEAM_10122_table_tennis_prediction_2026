"""V300 clean server blend recycler.

Recycle existing clean server-only candidates into tiny serverGetPoint-only
rank-average and linear-average blends against the current clean anchor.
No TTMATCH, no old-server, and no upload-candidate copying.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v300_clean_server_blend_recycler")
ANCHOR_PATH = Path("upload_candidates_20260519/submission_v261_cap0p01__v173action_r121server.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
WEIGHTS = [0.005, 0.010, 0.020]
SEARCH_PATH = OUTDIR / "v300_server_search.csv"
REPORT_JSON_PATH = OUTDIR / "v300_report.json"
REPORT_MD_PATH = OUTDIR / "v300_report.md"

SOURCE_SEARCHES = [
    Path("v266_clean_autoresearch_loop/v266_candidate_search.csv"),
    Path("v269_clean_server_value_ranker/v269_server_search.csv"),
    Path("v271_server_microblend_probe/v271_server_probe_search.csv"),
]

BANNED_PATH_TOKENS = ["TTMATCH", "OLD_SERVER", "OLDSERVER"]


@dataclass(frozen=True)
class ServerSource:
    candidate: str
    path: Path
    server: np.ndarray
    source_family: str
    source_candidates: tuple[str, ...]
    proxy_local_auc: float
    proxy_delta: float
    server_mad_vs_anchor: float
    server_corr_vs_anchor: float
    risk_tier: str
    verdict: str


def no_banned_path_guard(paths: list[Path | str]) -> None:
    bad = []
    for path in paths:
        upper = str(path).upper()
        if any(token in upper for token in BANNED_PATH_TOKENS):
            bad.append(str(path))
    if bad:
        raise ValueError(f"Banned clean-branch input path: {bad}")


def cap_prob(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
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


def rank_normalize_to_anchor(source: np.ndarray, anchor: np.ndarray) -> np.ndarray:
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
    normalized = np.interp(ranks, np.arange(len(anchor_sorted), dtype=float), anchor_sorted)
    return cap_prob(normalized)


def blend_server_only(anchor: pd.DataFrame, target_server: np.ndarray, *, weight: float) -> pd.DataFrame:
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    if len(anchor) != len(target_server):
        raise ValueError("anchor and target_server lengths differ")
    out = anchor.copy()
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    target = cap_prob(target_server)
    out["serverGetPoint"] = cap_prob((1.0 - weight) * anchor_server + weight * target)
    validate_submission_frame(out, expected_rows=len(anchor))
    return out


def weight_name(weight: float) -> str:
    return f"{weight:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def classify_risk(mad: float) -> str:
    if mad <= 0.002:
        return "safe"
    if mad <= 0.005:
        return "exploratory"
    return "high_churn"


def verdict_for(proxy_delta: float, mad: float) -> str:
    if np.isfinite(proxy_delta) and proxy_delta > 0.0 and mad <= 0.002:
        return "CANDIDATE_FOR_REVIEW"
    if np.isfinite(proxy_delta) and proxy_delta > 0.0 and mad <= 0.005:
        return "EXPLORATORY_REVIEW"
    return "LOCAL_PROXY_UNKNOWN_REVIEW"


def resolve_source_path(raw_path: object, search_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path
    candidate = search_path.parent / path.name
    if candidate.exists():
        return candidate
    return path


def source_family_from_search(search_path: Path) -> str:
    name = search_path.parent.name
    if name.startswith("v266"):
        return "v266"
    if name.startswith("v269"):
        return "v269"
    if name.startswith("v271"):
        return "v271"
    return name


def metric_value(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(pd.to_numeric(row[name], errors="coerce"))
    return float("nan")


def row_proxy_delta(row: pd.Series) -> float:
    return metric_value(row, ["delta_vs_proxy_base", "proxy_delta_linear_from_v263c"])


def source_fingerprint(server: np.ndarray) -> bytes:
    return np.round(cap_prob(server), 10).tobytes()


def load_sources(anchor: pd.DataFrame) -> tuple[list[ServerSource], int]:
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    deduped: dict[bytes, ServerSource] = {}
    raw_rows = 0

    for search_path in SOURCE_SEARCHES:
        no_banned_path_guard([search_path])
        if not search_path.exists():
            continue
        search = pd.read_csv(search_path)
        if "path" not in search.columns or "candidate" not in search.columns:
            raise ValueError(f"{search_path} must contain candidate and path columns")
        family = source_family_from_search(search_path)
        for _, row in search.iterrows():
            raw_rows += 1
            source_path = resolve_source_path(row["path"], search_path)
            source_df = load_submission(source_path)
            if not source_df["rally_uid"].equals(anchor["rally_uid"]):
                raise ValueError(f"{source_path} rally_uid does not match anchor")
            if not source_df["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
                raise ValueError(f"{source_path} changes actionId")
            if not source_df["pointId"].astype(int).equals(anchor["pointId"].astype(int)):
                raise ValueError(f"{source_path} changes pointId")

            server = cap_prob(source_df["serverGetPoint"].to_numpy(dtype=float))
            fingerprint = source_fingerprint(server)
            proxy_auc = metric_value(row, ["server_auc", "proxy_local_auc"])
            proxy_delta = row_proxy_delta(row)
            mad = float(np.mean(np.abs(server - anchor_server)))
            server_corr = corr(server, anchor_server)
            source = ServerSource(
                candidate=str(row["candidate"]),
                path=source_path,
                server=server,
                source_family=family,
                source_candidates=(str(row["candidate"]),),
                proxy_local_auc=proxy_auc,
                proxy_delta=proxy_delta,
                server_mad_vs_anchor=mad,
                server_corr_vs_anchor=server_corr,
                risk_tier=str(row.get("risk_tier", classify_risk(mad))),
                verdict=str(row.get("verdict", verdict_for(proxy_delta, mad))),
            )
            if fingerprint in deduped:
                previous = deduped[fingerprint]
                deduped[fingerprint] = ServerSource(
                    candidate=previous.candidate,
                    path=previous.path,
                    server=previous.server,
                    source_family=previous.source_family,
                    source_candidates=previous.source_candidates + (source.candidate,),
                    proxy_local_auc=max(previous.proxy_local_auc, source.proxy_local_auc)
                    if np.isfinite(previous.proxy_local_auc) or np.isfinite(source.proxy_local_auc)
                    else float("nan"),
                    proxy_delta=max(previous.proxy_delta, source.proxy_delta)
                    if np.isfinite(previous.proxy_delta) or np.isfinite(source.proxy_delta)
                    else float("nan"),
                    server_mad_vs_anchor=previous.server_mad_vs_anchor,
                    server_corr_vs_anchor=previous.server_corr_vs_anchor,
                    risk_tier=previous.risk_tier,
                    verdict=previous.verdict,
                )
            else:
                deduped[fingerprint] = source

    sources = list(deduped.values())
    if not sources:
        raise ValueError("No clean server sources found.")
    return sources, raw_rows


def mean_finite(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def write_submission(path: Path, df: pd.DataFrame) -> None:
    no_banned_path_guard([path])
    validate_submission_frame(df, expected_rows=len(df))
    df[EXPECTED_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def summarize_candidate(
    *,
    candidate: str,
    path: Path,
    output: pd.DataFrame,
    anchor_server: np.ndarray,
    blend_kind: str,
    weight: float,
    sources: list[ServerSource],
) -> dict[str, object]:
    server = cap_prob(output["serverGetPoint"].to_numpy(dtype=float))
    proxy_auc = mean_finite([source.proxy_local_auc for source in sources])
    proxy_delta = mean_finite([source.proxy_delta for source in sources])
    mad = float(np.mean(np.abs(server - anchor_server)))
    row = {
        "candidate": candidate,
        "path": str(path),
        "blend_kind": blend_kind,
        "weight": float(weight),
        "source_count": int(len(sources)),
        "source_candidates": ";".join(source.candidate for source in sources),
        "source_aliases": ";".join("|".join(source.source_candidates) for source in sources),
        "proxy_local_auc": proxy_auc,
        "proxy_delta_vs_proxy_base": proxy_delta,
        "server_mad_vs_anchor": mad,
        "server_corr_vs_anchor": corr(server, anchor_server),
        "action_changed_vs_anchor": 0,
        "point_changed_vs_anchor": 0,
        "risk_tier": classify_risk(mad),
        "verdict": verdict_for(proxy_delta, mad),
    }
    return row


def best_safe_source_for_repack(sources: list[ServerSource]) -> ServerSource | None:
    eligible = [
        source
        for source in sources
        if source.source_family in {"v266", "v269"}
        and classify_risk(source.server_mad_vs_anchor) == "safe"
        and np.isfinite(source.proxy_delta)
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda source: (
            source.proxy_delta,
            source.proxy_local_auc if np.isfinite(source.proxy_local_auc) else -np.inf,
            -source.server_mad_vs_anchor,
        ),
        reverse=True,
    )[0]


def build_outputs(anchor: pd.DataFrame, sources: list[ServerSource]) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    source_matrix = np.column_stack([source.server for source in sources])
    mean_target = cap_prob(np.mean(source_matrix, axis=1))
    rank_matrix = np.column_stack(
        [rank_normalize_to_anchor(source.server, anchor_server) for source in sources]
    )
    rank_target = cap_prob(np.mean(rank_matrix, axis=1))

    rows: list[dict[str, object]] = []
    outputs: dict[str, pd.DataFrame] = {}

    for kind, target in [("rankavg", rank_target), ("mean", mean_target)]:
        for weight in WEIGHTS:
            candidate = f"submission_v300_{kind}_w{weight_name(weight)}__v173action_v261point_server.csv"
            output = blend_server_only(anchor, target, weight=weight)
            path = OUTDIR / candidate
            outputs[candidate] = output
            rows.append(
                summarize_candidate(
                    candidate=candidate,
                    path=path,
                    output=output,
                    anchor_server=anchor_server,
                    blend_kind=kind,
                    weight=weight,
                    sources=sources,
                )
            )

    best_safe = best_safe_source_for_repack(sources)
    if best_safe is not None:
        candidate = "submission_v300_best_safe_repack__v173action_v261point_server.csv"
        output = anchor.copy()
        output["serverGetPoint"] = best_safe.server
        validate_submission_frame(output, expected_rows=len(anchor))
        path = OUTDIR / candidate
        outputs[candidate] = output
        rows.append(
            summarize_candidate(
                candidate=candidate,
                path=path,
                output=output,
                anchor_server=anchor_server,
                blend_kind="best_safe_repack",
                weight=1.0,
                sources=[best_safe],
            )
        )

    return rows, outputs


def write_reports(search: pd.DataFrame, summary: dict[str, object]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V300 Clean Server Blend Recycler",
        "",
        "Clean server-only recycler. Action and point stay fixed to the V261/V173 clean anchor.",
        "",
        "## Policy",
        "",
        "- No TTMATCH input.",
        "- No old-server input.",
        "- No copying to upload_candidates.",
        "",
        "## Sources",
        "",
        f"- Raw source rows read: `{summary['raw_source_rows']}`",
        f"- Unique server sources used: `{summary['unique_server_sources']}`",
        f"- Anchor: `{summary['anchor_path']}`",
        "",
        "## Candidates",
        "",
        "| candidate | kind | weight | sources | proxy_auc | proxy_delta | MAD | corr | risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"| `{row['candidate']}` | {row['blend_kind']} | {row['weight']:.3f} | "
            f"{int(row['source_count'])} | {row['proxy_local_auc']:.6f} | "
            f"{row['proxy_delta_vs_proxy_base']:.6f} | {row['server_mad_vs_anchor']:.6f} | "
            f"{row['server_corr_vs_anchor']:.6f} | `{row['risk_tier']}` |"
        )
    lines.extend(["", f"Search CSV: `{SEARCH_PATH}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    no_banned_path_guard([ANCHOR_PATH, *SOURCE_SEARCHES])
    anchor = load_submission(ANCHOR_PATH)
    sources, raw_source_rows = load_sources(anchor)
    rows, outputs = build_outputs(anchor, sources)

    for candidate, output in outputs.items():
        write_submission(OUTDIR / candidate, output)

    search = pd.DataFrame(rows)
    search.to_csv(SEARCH_PATH, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(OUTDIR),
        "anchor_path": str(ANCHOR_PATH),
        "policy": {
            "clean_only": True,
            "server_only": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_copy": True,
        },
        "raw_source_rows": int(raw_source_rows),
        "unique_server_sources": int(len(sources)),
        "generated_candidates": list(outputs.keys()),
        "search_path": str(SEARCH_PATH),
        "report_md_path": str(REPORT_MD_PATH),
    }
    write_reports(search, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
