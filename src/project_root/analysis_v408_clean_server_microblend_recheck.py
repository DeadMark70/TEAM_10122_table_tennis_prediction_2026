"""V408 clean server microblend recheck.

Build tiny serverGetPoint-only blends on top of the V362 public anchor while
preserving actionId and pointId exactly.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v408_clean_server_microblend_recheck"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
EXPECTED_ROWS = 1845
SERVER_MAD_LIMIT = 0.02
PROB_MIN = 0.001
PROB_MAX = 0.999

SOURCE_DIRS = (
    "v300_clean_server_blend_recycler",
    "v321_server_robust_rankblend",
    "v319_clean_server_value_state",
    "v302_clean_server_calibration_sweep",
    "v271_server_microblend_probe",
    "v269_clean_server_value_ranker",
    "v266_clean_autoresearch_loop",
)
ROOT_SOURCE_GLOBS = (
    "submission_r121*.csv",
    "submission_v300*.csv",
    "submission_v321*.csv",
)
BANNED_PATH_TOKENS = (
    "TTMATCH",
    "OLD_SERVER",
    "OLDSERVER",
    "OLD-SERVER",
    "UPLOAD_CANDIDATES_20260519",
)
RANKED_COLUMNS = [
    "candidate",
    "path",
    "selected_rows",
    "selected_row_count",
    "action_churn",
    "point_churn",
    "point0_additions",
    "server_changed",
    "risk",
    "evidence",
    "server_mad",
    "server_corr",
    "server_min",
    "server_max",
    "source_count",
    "source_names",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def clip_prob(values: Iterable[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=PROB_MAX, neginf=PROB_MIN)
    return np.clip(arr, PROB_MIN, PROB_MAX)


def server_corr(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def reject_banned_paths(paths: Iterable[Path | str]) -> None:
    bad = []
    for path in paths:
        upper = str(path).upper()
        if any(token in upper for token in BANNED_PATH_TOKENS):
            bad.append(str(path))
    if bad:
        raise ValueError(f"banned V408 path(s): {bad}")


def load_submission(path: Path, *, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    reject_banned_paths([path])
    frame = pd.read_csv(path)
    frame = frame.loc[:, SUBMISSION_COLUMNS]
    validate_submission_schema(frame, expected_rows=expected_rows)
    frame = frame.copy()
    frame["serverGetPoint"] = clip_prob(frame["serverGetPoint"])
    return frame


def align_to_anchor(frame: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    if frame["rally_uid"].equals(anchor["rally_uid"]):
        return frame.reset_index(drop=True)
    if set(frame["rally_uid"]) != set(anchor["rally_uid"]):
        raise ValueError("rally_uid set differs from V362 anchor")
    aligned = frame.set_index("rally_uid").loc[anchor["rally_uid"]].reset_index()
    return aligned.loc[:, SUBMISSION_COLUMNS]


def discover_source_paths(*, root: Path = ROOT, source_dirs: Iterable[str] = SOURCE_DIRS) -> list[Path]:
    root = Path(root)
    seen: set[Path] = set()
    paths: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        reject_banned_paths([path])
        seen.add(resolved)
        paths.append(path)

    for dirname in source_dirs:
        directory = root / dirname
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("submission_*.csv")):
            if path.is_file():
                add(path)

    for pattern in ROOT_SOURCE_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                add(path)
    return paths


def _fingerprint(server: np.ndarray) -> bytes:
    return np.round(clip_prob(server), 10).tobytes()


def load_server_sources(
    anchor: pd.DataFrame,
    *,
    root: Path = ROOT,
    expected_rows: int | None = EXPECTED_ROWS,
) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, str]]]:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    sources: dict[bytes, tuple[str, np.ndarray]] = {}
    ignored: list[dict[str, str]] = []

    for path in discover_source_paths(root=root):
        try:
            frame = align_to_anchor(load_submission(path, expected_rows=expected_rows), anchor)
        except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
            ignored.append({"path": str(path), "reason": "unusable_submission", "detail": str(exc)})
            continue

        server = clip_prob(frame["serverGetPoint"])
        fp = _fingerprint(server)
        if fp == _fingerprint(anchor_server) or fp in sources:
            ignored.append({"path": str(path), "reason": "duplicate_or_anchor_server"})
            continue
        sources[fp] = (path.parent.name + "/" + path.name, server)

    return list(sources.values()), ignored


def rank_normalize_to_anchor(source: Iterable[float], anchor: Iterable[float]) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    finite = source_arr[np.isfinite(source_arr)]
    fill = float(np.median(finite)) if len(finite) else 0.5
    source_arr = np.nan_to_num(source_arr, nan=fill, posinf=fill, neginf=fill)
    anchor_sorted = np.sort(clip_prob(anchor))
    if len(source_arr) != len(anchor_sorted):
        raise ValueError("source and anchor lengths differ")
    if len(source_arr) == 1:
        return np.array([anchor_sorted[0]], dtype=float)
    ranks = pd.Series(source_arr).rank(method="average").to_numpy(dtype=float) - 1.0
    normalized = np.interp(ranks, np.arange(len(anchor_sorted), dtype=float), anchor_sorted)
    return clip_prob(normalized)


def blend_server(anchor_server: np.ndarray, target_server: np.ndarray, *, weight: float) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return clip_prob((1.0 - weight) * clip_prob(anchor_server) + weight * clip_prob(target_server))


def should_emit_candidate(*, server_mad: float, action_churn: int, point_churn: int) -> bool:
    return (
        action_churn == 0
        and point_churn == 0
        and np.isfinite(server_mad)
        and float(server_mad) <= SERVER_MAD_LIMIT
    )


def _scorestate_mask(root: Path, anchor: pd.DataFrame) -> np.ndarray | None:
    feature_path = root / "test_new.csv"
    if not feature_path.exists():
        return None
    try:
        features = pd.read_csv(feature_path, usecols=["rally_uid", "scoreSelf", "scoreOther"])
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if features["rally_uid"].duplicated().any():
        return None
    if set(features["rally_uid"]) != set(anchor["rally_uid"]):
        return None
    aligned = features.set_index("rally_uid").loc[anchor["rally_uid"]].reset_index()
    if len(aligned) != len(anchor):
        return None
    score_self = pd.to_numeric(aligned["scoreSelf"], errors="coerce")
    score_other = pd.to_numeric(aligned["scoreOther"], errors="coerce")
    if score_self.isna().all() or score_other.isna().all():
        return None
    return (score_self.sub(score_other).abs() <= 2).fillna(False).to_numpy(dtype=bool)


def _write_submission(path: Path, frame: pd.DataFrame, *, expected_rows: int | None) -> None:
    validate_submission_schema(frame, expected_rows=expected_rows)
    frame.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def _selected_rows(anchor: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    server = clip_prob(submission["serverGetPoint"])
    changed = np.abs(server - anchor_server) > 1e-12
    return pd.DataFrame(
        {
            "row_id": np.flatnonzero(changed),
            "rally_uid": anchor.loc[changed, "rally_uid"].to_numpy(),
            "anchor_serverGetPoint": anchor_server[changed],
            "candidate_serverGetPoint": server[changed],
            "server_delta": server[changed] - anchor_server[changed],
        }
    )


def _package(anchor: pd.DataFrame, server: np.ndarray, *, expected_rows: int | None) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["serverGetPoint"] = clip_prob(server)
    if not out["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
        raise AssertionError("V408 changed actionId")
    if not out["pointId"].astype(int).equals(anchor["pointId"].astype(int)):
        raise AssertionError("V408 changed pointId")
    validate_submission_schema(out, expected_rows=expected_rows)
    return out


def _risk_for_mad(mad: float) -> str:
    if mad <= 0.005:
        return "safe"
    if mad <= SERVER_MAD_LIMIT:
        return "microblend"
    return "blocked_high_mad"


def build_microblend_candidates(
    anchor: pd.DataFrame,
    sources: list[tuple[str, np.ndarray]],
    *,
    outdir: Path,
    root: Path = ROOT,
    expected_rows: int | None = EXPECTED_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    outdir = Path(outdir)
    anchor_server = clip_prob(anchor["serverGetPoint"])
    if not sources:
        return [], {}

    source_names = [name for name, _ in sources]
    source_matrix = np.column_stack([clip_prob(server) for _, server in sources])
    mean_target = clip_prob(np.mean(source_matrix, axis=1))
    rank_matrix = np.column_stack([rank_normalize_to_anchor(server, anchor_server) for _, server in sources])
    rank_target = clip_prob(np.mean(rank_matrix, axis=1))

    specs: list[tuple[str, np.ndarray, float, str]] = [
        ("mean_w0p005", mean_target, 0.005, "mean"),
        ("mean_w0p010", mean_target, 0.010, "mean"),
        ("rankavg_w0p005", rank_target, 0.005, "rankavg"),
        ("rankavg_w0p010", rank_target, 0.010, "rankavg"),
    ]

    mask = _scorestate_mask(Path(root), anchor)
    if mask is not None and mask.any():
        safe_target = np.where(mask, mean_target, anchor_server)
        specs.append(("scorestate_safe", safe_target, 0.010, "scorestate_safe"))

    rows: list[dict[str, Any]] = []
    submissions: dict[str, pd.DataFrame] = {}
    for candidate, target, weight, evidence in specs:
        server = blend_server(anchor_server, target, weight=weight)
        submission = _package(anchor, server, expected_rows=expected_rows)
        action_churn = int((submission["actionId"].astype(int) != anchor["actionId"].astype(int)).sum())
        point_churn = int((submission["pointId"].astype(int) != anchor["pointId"].astype(int)).sum())
        server_changed = int(np.sum(np.abs(clip_prob(submission["serverGetPoint"]) - anchor_server) > 1e-12))
        mad = float(np.mean(np.abs(clip_prob(submission["serverGetPoint"]) - anchor_server)))

        if not should_emit_candidate(server_mad=mad, action_churn=action_churn, point_churn=point_churn):
            continue

        submission_path = safe_output_path(outdir, f"submission_v408_{candidate}__v173action_v362point_server.csv")
        selected_path = safe_output_path(outdir, f"selected_rows_{candidate}.csv")
        selected = _selected_rows(anchor, submission)
        row = {
            "candidate": candidate,
            "path": str(submission_path),
            "selected_rows": str(selected_path),
            "selected_row_count": int(len(selected)),
            "action_churn": action_churn,
            "point_churn": point_churn,
            "point0_additions": 0,
            "server_changed": server_changed,
            "risk": _risk_for_mad(mad),
            "evidence": evidence,
            "server_mad": mad,
            "server_corr": server_corr(submission["serverGetPoint"], anchor_server),
            "server_min": float(np.min(clip_prob(submission["serverGetPoint"]))),
            "server_max": float(np.max(clip_prob(submission["serverGetPoint"]))),
            "source_count": int(len(sources)),
            "source_names": ";".join(source_names),
        }
        rows.append(row)
        submissions[candidate] = submission

    rows = sorted(rows, key=lambda item: (item["server_mad"], item["candidate"]))
    return rows, submissions


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    root: Path = ROOT,
    anchor_path: Path | None = None,
    expected_rows: int | None = EXPECTED_ROWS,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir)
    anchor_path = Path(anchor_path) if anchor_path is not None else root / ANCHOR_PATH.relative_to(ROOT)
    outdir.mkdir(parents=True, exist_ok=True)
    reject_banned_paths([outdir, anchor_path])

    anchor = load_submission(anchor_path, expected_rows=expected_rows)
    sources, ignored = load_server_sources(anchor, root=root, expected_rows=expected_rows)
    rows, submissions = build_microblend_candidates(
        anchor,
        sources,
        outdir=outdir,
        root=root,
        expected_rows=expected_rows,
    )

    for row in rows:
        candidate = str(row["candidate"])
        _write_submission(Path(row["path"]), submissions[candidate], expected_rows=expected_rows)
        _selected_rows(anchor, submissions[candidate]).to_csv(row["selected_rows"], index=False)

    ranked = pd.DataFrame(rows, columns=RANKED_COLUMNS)
    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    ranked.to_csv(ranked_path, index=False)

    report = {
        "version": "V408",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": str(anchor_path),
        "outdir": str(outdir),
        "source_count": int(len(sources)),
        "ignored_source_count": int(len(ignored)),
        "ignored_sources": ignored,
        "generated_submission_count": int(len(rows)),
        "generated_candidates": rows,
        "ranked_candidates": str(ranked_path),
        "policy": {
            "anchor": "V362",
            "server_only": True,
            "preserve_action": True,
            "preserve_point": True,
            "clip_probability_bounds": [PROB_MIN, PROB_MAX],
            "max_server_mad": SERVER_MAD_LIMIT,
            "no_ttmatch": True,
            "no_old_server": True,
        },
    }
    report_path = safe_output_path(outdir, "search_report.json")
    write_json(report_path, report)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
