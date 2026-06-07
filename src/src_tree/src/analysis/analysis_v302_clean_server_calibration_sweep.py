"""V302 clean server calibration sweep.

Server-only calibration variants around the V300 best-safe-repack submission.
This script reads only clean V300/V266/V269/V271 artifacts, does not read
TTMATCH or old-server inputs, and does not copy outputs to upload candidates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v302_clean_server_calibration_sweep")
V300_BASE_PATH = Path(
    "v300_clean_server_blend_recycler/"
    "submission_v300_best_safe_repack__v173action_v261point_server.csv"
)
V300_SEARCH_PATH = Path("v300_clean_server_blend_recycler/v300_server_search.csv")
V261_ANCHOR_PATH = Path(
    "v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv"
)
SOURCE_SEARCHES = [
    Path("v266_clean_autoresearch_loop/v266_candidate_search.csv"),
    Path("v269_clean_server_value_ranker/v269_server_search.csv"),
    Path("v271_server_microblend_probe/v271_server_probe_search.csv"),
]
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
BANNED_PATH_TOKENS = ["TTMATCH", "OLD_SERVER", "OLDSERVER"]

SHRINK_STRENGTHS = [0.25, 0.50, 0.75]
TEMPERATURES = [0.95, 0.90, 1.05]
MIX_WEIGHTS = [0.25, 0.50]

SEARCH_PATH = OUTDIR / "v302_server_search.csv"
REPORT_JSON_PATH = OUTDIR / "v302_report.json"
REPORT_MD_PATH = OUTDIR / "v302_report.md"

V300_PUBLIC_PL = 0.3576975
V300_PUBLIC_DELTA_VS_V261_CLEAN_ANCHOR = 0.0000255


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


def resolve_source_path(raw_path: object, search_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path
    candidate = search_path.parent / path.name
    if candidate.exists():
        return candidate
    return path


def weight_name(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def metric_value(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(pd.to_numeric(row[name], errors="coerce"))
    return float("nan")


def mean_finite(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def shrink_to_anchor(v300_server: np.ndarray, anchor_server: np.ndarray, *, strength: float) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if len(v300_server) != len(anchor_server):
        raise ValueError("v300_server and anchor_server lengths differ")
    base = cap_prob(v300_server)
    anchor = cap_prob(anchor_server)
    return cap_prob((1.0 - strength) * base + strength * anchor)


def apply_temperature(server: np.ndarray, *, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    p = cap_prob(server)
    logits = np.log(p) - np.log1p(-p)
    scaled = logits / float(temperature)
    out = 1.0 / (1.0 + np.exp(-scaled))
    return cap_prob(out)


def mix_server(base_server: np.ndarray, target_server: np.ndarray, *, weight: float) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    if len(base_server) != len(target_server):
        raise ValueError("base_server and target_server lengths differ")
    base = cap_prob(base_server)
    target = cap_prob(target_server)
    return cap_prob((1.0 - weight) * base + weight * target)


def build_server_only_output(base: pd.DataFrame, server: np.ndarray) -> pd.DataFrame:
    if len(base) != len(server):
        raise ValueError("base and server lengths differ")
    out = base.copy()
    out["serverGetPoint"] = cap_prob(server)
    validate_submission_frame(out, expected_rows=len(base))
    return out[EXPECTED_COLUMNS]


def assert_action_point_unchanged(candidate: pd.DataFrame, reference: pd.DataFrame, label: str) -> None:
    if not candidate["rally_uid"].equals(reference["rally_uid"]):
        raise ValueError(f"{label} rally_uid changed")
    if not candidate["actionId"].astype(int).equals(reference["actionId"].astype(int)):
        raise ValueError(f"{label} actionId changed")
    if not candidate["pointId"].astype(int).equals(reference["pointId"].astype(int)):
        raise ValueError(f"{label} pointId changed")


def load_v300_search() -> pd.DataFrame:
    no_banned_path_guard([V300_SEARCH_PATH])
    if not V300_SEARCH_PATH.exists():
        raise FileNotFoundError(V300_SEARCH_PATH)
    search = pd.read_csv(V300_SEARCH_PATH)
    required = {"candidate", "path", "blend_kind"}
    missing = required - set(search.columns)
    if missing:
        raise ValueError(f"{V300_SEARCH_PATH} missing columns: {sorted(missing)}")
    return search


def v300_proxy_delta(search: pd.DataFrame) -> float:
    hit = search[search["candidate"].astype(str).eq(V300_BASE_PATH.name)]
    if hit.empty:
        return float("nan")
    return metric_value(hit.iloc[0], ["proxy_delta_vs_proxy_base", "delta_vs_proxy_base"])


def load_v300_kind_target(
    search: pd.DataFrame,
    kind: str,
    v300_base: pd.DataFrame,
) -> tuple[np.ndarray, float, list[str]]:
    rows = search[search["blend_kind"].astype(str).eq(kind)].copy()
    if rows.empty:
        raise ValueError(f"No V300 search rows for blend_kind={kind}")

    servers: list[np.ndarray] = []
    proxy_deltas: list[float] = []
    source_candidates: list[str] = []
    for _, row in rows.iterrows():
        source_path = resolve_source_path(row["path"], V300_SEARCH_PATH)
        source_df = load_submission(source_path)
        assert_action_point_unchanged(source_df, v300_base, str(source_path))
        servers.append(cap_prob(source_df["serverGetPoint"].to_numpy(dtype=float)))
        proxy_deltas.append(metric_value(row, ["proxy_delta_vs_proxy_base", "delta_vs_proxy_base"]))
        source_candidates.append(str(row["candidate"]))

    return cap_prob(np.mean(np.column_stack(servers), axis=1)), mean_finite(proxy_deltas), source_candidates


def count_clean_source_rows() -> dict[str, int]:
    counts: dict[str, int] = {}
    for search_path in SOURCE_SEARCHES:
        no_banned_path_guard([search_path])
        if not search_path.exists():
            counts[str(search_path)] = 0
            continue
        search = pd.read_csv(search_path)
        if "path" in search.columns:
            no_banned_path_guard([resolve_source_path(path, search_path) for path in search["path"]])
        counts[str(search_path)] = int(len(search))
    return counts


def verdict_for(proxy_delta_vs_v300: float, mad_vs_v300: float) -> str:
    if not np.isfinite(proxy_delta_vs_v300):
        return "LOCAL_PROXY_UNKNOWN_REVIEW"
    if mad_vs_v300 <= 0.001 and proxy_delta_vs_v300 >= -0.001:
        return "CANDIDATE_FOR_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def summarize_candidate(
    *,
    candidate: str,
    path: Path,
    output: pd.DataFrame,
    v300_base: pd.DataFrame,
    v261_anchor: pd.DataFrame,
    variant_kind: str,
    parameter: float,
    proxy_delta_estimate: float,
    v300_proxy_delta_estimate: float,
    source_candidates: list[str],
) -> dict[str, object]:
    server = cap_prob(output["serverGetPoint"].to_numpy(dtype=float))
    v300_server = cap_prob(v300_base["serverGetPoint"].to_numpy(dtype=float))
    v261_server = cap_prob(v261_anchor["serverGetPoint"].to_numpy(dtype=float))
    mad_vs_v300 = float(np.mean(np.abs(server - v300_server)))
    proxy_delta_vs_v300 = (
        float(proxy_delta_estimate - v300_proxy_delta_estimate)
        if np.isfinite(proxy_delta_estimate) and np.isfinite(v300_proxy_delta_estimate)
        else float("nan")
    )
    finite = bool(np.isfinite(server).all())
    row = {
        "candidate": candidate,
        "path": str(path),
        "variant_kind": variant_kind,
        "parameter": float(parameter),
        "source_candidates": ";".join(source_candidates),
        "proxy_delta_estimate": proxy_delta_estimate,
        "v300_proxy_delta_estimate": v300_proxy_delta_estimate,
        "proxy_delta_vs_v300_estimate": proxy_delta_vs_v300,
        "server_mad_vs_v300": mad_vs_v300,
        "server_corr_vs_v300": corr(server, v300_server),
        "server_mad_vs_v261": float(np.mean(np.abs(server - v261_server))),
        "server_corr_vs_v261": corr(server, v261_server),
        "server_min": float(np.min(server)),
        "server_max": float(np.max(server)),
        "server_finite": finite,
        "action_changed_vs_v300": int(
            not output["actionId"].astype(int).equals(v300_base["actionId"].astype(int))
        ),
        "point_changed_vs_v300": int(
            not output["pointId"].astype(int).equals(v300_base["pointId"].astype(int))
        ),
        "verdict": verdict_for(proxy_delta_vs_v300, mad_vs_v300),
    }
    return row


def write_submission(path: Path, df: pd.DataFrame) -> None:
    no_banned_path_guard([path])
    validate_submission_frame(df, expected_rows=len(df))
    df[EXPECTED_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def build_outputs(
    v300_base: pd.DataFrame,
    v261_anchor: pd.DataFrame,
    search: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    v300_server = cap_prob(v300_base["serverGetPoint"].to_numpy(dtype=float))
    v261_server = cap_prob(v261_anchor["serverGetPoint"].to_numpy(dtype=float))
    base_proxy = v300_proxy_delta(search)
    rank_target, rank_proxy, rank_sources = load_v300_kind_target(search, "rankavg", v300_base)
    mean_target, mean_proxy, mean_sources = load_v300_kind_target(search, "mean", v300_base)

    rows: list[dict[str, object]] = []
    outputs: dict[str, pd.DataFrame] = {}

    def add_variant(
        *,
        kind: str,
        parameter: float,
        server: np.ndarray,
        proxy_delta: float,
        source_candidates: list[str],
    ) -> None:
        candidate = f"submission_v302_{kind}_{weight_name(parameter)}__v173action_v261point_server.csv"
        output = build_server_only_output(v300_base, server)
        assert_action_point_unchanged(output, v300_base, candidate)
        assert_action_point_unchanged(output, v261_anchor, candidate)
        path = OUTDIR / candidate
        outputs[candidate] = output
        rows.append(
            summarize_candidate(
                candidate=candidate,
                path=path,
                output=output,
                v300_base=v300_base,
                v261_anchor=v261_anchor,
                variant_kind=kind,
                parameter=parameter,
                proxy_delta_estimate=proxy_delta,
                v300_proxy_delta_estimate=base_proxy,
                source_candidates=source_candidates,
            )
        )

    for strength in SHRINK_STRENGTHS:
        proxy = (1.0 - strength) * base_proxy if np.isfinite(base_proxy) else float("nan")
        add_variant(
            kind="shrink_to_anchor_s",
            parameter=strength,
            server=shrink_to_anchor(v300_server, v261_server, strength=strength),
            proxy_delta=proxy,
            source_candidates=[V261_ANCHOR_PATH.name],
        )

    for temperature in TEMPERATURES:
        add_variant(
            kind="temperature_t",
            parameter=temperature,
            server=apply_temperature(v300_server, temperature=temperature),
            proxy_delta=base_proxy,
            source_candidates=[V300_BASE_PATH.name],
        )

    for weight in MIX_WEIGHTS:
        rank_proxy_est = (
            (1.0 - weight) * base_proxy + weight * rank_proxy
            if np.isfinite(base_proxy) and np.isfinite(rank_proxy)
            else float("nan")
        )
        add_variant(
            kind="rankmix_w",
            parameter=weight,
            server=mix_server(v300_server, rank_target, weight=weight),
            proxy_delta=rank_proxy_est,
            source_candidates=rank_sources,
        )

    for weight in MIX_WEIGHTS:
        mean_proxy_est = (
            (1.0 - weight) * base_proxy + weight * mean_proxy
            if np.isfinite(base_proxy) and np.isfinite(mean_proxy)
            else float("nan")
        )
        add_variant(
            kind="meanmix_w",
            parameter=weight,
            server=mix_server(v300_server, mean_target, weight=weight),
            proxy_delta=mean_proxy_est,
            source_candidates=mean_sources,
        )

    return rows, outputs


def write_reports(search: pd.DataFrame, summary: dict[str, object]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# V302 Clean Server Calibration Sweep",
        "",
        "Server-only calibration variants around V300 best_safe_repack.",
        "",
        "## Policy",
        "",
        "- No TTMATCH input.",
        "- No old-server input.",
        "- No upload-candidate copy.",
        "- `actionId` and `pointId` remain unchanged.",
        "",
        "## Anchor",
        "",
        f"- V300 base: `{summary['v300_base_path']}`",
        f"- V261 reference: `{summary['v261_anchor_path']}`",
        f"- V300 public PL: `{summary['v300_public_pl']:.7f}`",
        f"- V300 public delta vs V261 clean anchor: `{summary['v300_public_delta_vs_v261_clean_anchor']:.7f}`",
        "",
        "## Candidates",
        "",
        "| candidate | kind | param | proxy_vs_v300 | MAD_v300 | corr_v300 | MAD_v261 | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"| `{row['candidate']}` | {row['variant_kind']} | {row['parameter']:.3f} | "
            f"{row['proxy_delta_vs_v300_estimate']:.6f} | "
            f"{row['server_mad_vs_v300']:.6f} | {row['server_corr_vs_v300']:.6f} | "
            f"{row['server_mad_vs_v261']:.6f} | `{row['verdict']}` |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Review candidates: `{summary['candidate_for_review_count']}`",
            f"- Recommended first review: `{summary['recommended_candidate']}`",
            f"- Search CSV: `{SEARCH_PATH}`",
            "",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    no_banned_path_guard([V300_BASE_PATH, V300_SEARCH_PATH, V261_ANCHOR_PATH, *SOURCE_SEARCHES])

    v300_base = load_submission(V300_BASE_PATH)
    v261_anchor = load_submission(V261_ANCHOR_PATH)
    assert_action_point_unchanged(v300_base, v261_anchor, "V300 base vs V261 anchor")

    search = load_v300_search()
    clean_source_rows = count_clean_source_rows()
    rows, outputs = build_outputs(v300_base, v261_anchor, search)

    for candidate, output in outputs.items():
        write_submission(OUTDIR / candidate, output)

    result = pd.DataFrame(rows).sort_values(
        ["verdict", "proxy_delta_vs_v300_estimate", "server_mad_vs_v300"],
        ascending=[True, False, True],
    )
    result.to_csv(SEARCH_PATH, index=False)

    candidates = result[result["verdict"].eq("CANDIDATE_FOR_REVIEW")].copy()
    if candidates.empty:
        recommended = "NONE"
    else:
        recommended = str(
            candidates.sort_values(
                ["proxy_delta_vs_v300_estimate", "server_mad_vs_v300"],
                ascending=[False, True],
            ).iloc[0]["candidate"]
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(OUTDIR),
        "v300_base_path": str(V300_BASE_PATH),
        "v300_search_path": str(V300_SEARCH_PATH),
        "v261_anchor_path": str(V261_ANCHOR_PATH),
        "policy": {
            "clean_only": True,
            "server_only": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_copy": True,
        },
        "v300_public_pl": V300_PUBLIC_PL,
        "v300_public_delta_vs_v261_clean_anchor": V300_PUBLIC_DELTA_VS_V261_CLEAN_ANCHOR,
        "v300_proxy_delta_estimate": v300_proxy_delta(search),
        "clean_source_rows": clean_source_rows,
        "generated_candidates": list(outputs.keys()),
        "candidate_for_review_count": int(len(candidates)),
        "recommended_candidate": recommended,
        "search_path": str(SEARCH_PATH),
        "report_json_path": str(REPORT_JSON_PATH),
        "report_md_path": str(REPORT_MD_PATH),
    }
    write_reports(result, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
