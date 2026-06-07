"""V400 public-proven component recombination.

This script recombines existing public-positive point submissions against the
V362 public-proven anchor. It only emits point-only candidates, preserves V173
action and V300 server columns exactly, and blocks point0 additions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v400_public_component_recombination"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
SOURCE_DIRS = (
    "v338_joint_moe_pack",
    "v341_no_p0_point_pack",
    "v306_point0_addition_probe",
    "v300_clean_server_blend_recycler",
    "v261_action_conditioned_point_residual",
    "v272_action_conditioned_point_residual",
    "v277_v272b_point_refinement",
    "v300_clean_server_blend_recycler",
    "v306_point0_addition_probe",
    "v338_joint_moe_pack",
    "v362_point_hierarchical_specialists",
)
EXTRA_GLOBS = (
    "submission_v261*.csv",
    "submission_v272*.csv",
    "submission_v277*.csv",
    "submission_v300*.csv",
    "submission_v306*.csv",
    "submission_v338*.csv",
    "submission_v362*.csv",
)
CANDIDATE_SIZES = (9, 15, 24)


@dataclass(frozen=True)
class PublicSource:
    path: Path
    frame: pd.DataFrame
    source_name: str
    content_hash: str


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_submission_paths(
    *,
    root: Path = ROOT,
    source_dirs: Iterable[str] = SOURCE_DIRS,
    extra_globs: Iterable[str] = EXTRA_GLOBS,
) -> list[Path]:
    """Return existing candidate submission CSV paths without recursive scans."""
    root = Path(root)
    seen: set[Path] = set()
    out: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(path)

    for dirname in dict.fromkeys(source_dirs):
        directory = root / dirname
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.csv")):
            if path.name.startswith("submission_"):
                add(path)

    for pattern in extra_globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.name.startswith("submission_"):
                add(path)
    return out


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[:, SUBMISSION_COLUMNS]
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.copy()


def load_anchor_submission(expected_rows: int | None = 1845) -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"missing V362 anchor submission: {ANCHOR_PATH}")
    return load_submission(ANCHOR_PATH, expected_rows=expected_rows)


def align_to_anchor(frame: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    if frame["rally_uid"].equals(anchor["rally_uid"]):
        return frame.reset_index(drop=True)
    if set(frame["rally_uid"]) != set(anchor["rally_uid"]):
        raise ValueError("rally_uid set differs from anchor")
    aligned = frame.set_index("rally_uid").loc[anchor["rally_uid"]].reset_index()
    return aligned.loc[:, SUBMISSION_COLUMNS]


def load_public_positive_sources(
    *,
    anchor: pd.DataFrame,
    paths: Iterable[Path],
    expected_rows: int | None = 1845,
) -> tuple[list[PublicSource], list[dict[str, Any]]]:
    sources: list[PublicSource] = []
    ignored: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    anchor_hash: str | None = None
    if ANCHOR_PATH.exists():
        try:
            anchor_hash = file_sha256(ANCHOR_PATH)
        except OSError:
            anchor_hash = None

    for path in paths:
        try:
            content_hash = file_sha256(path)
            if content_hash in seen_hashes or (anchor_hash is not None and content_hash == anchor_hash):
                ignored.append({"path": relative_path(path), "reason": "duplicate_content"})
                continue
            frame = align_to_anchor(load_submission(path, expected_rows=expected_rows), anchor)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            ignored.append({"path": relative_path(path), "reason": "bad_schema", "detail": str(exc)})
            continue

        seen_hashes.add(content_hash)
        if not frame["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
            ignored.append({"path": relative_path(path), "reason": "action_changed"})
            continue
        server = pd.to_numeric(frame["serverGetPoint"], errors="coerce").to_numpy(dtype=float)
        anchor_server = pd.to_numeric(anchor["serverGetPoint"], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(server, anchor_server):
            ignored.append({"path": relative_path(path), "reason": "server_changed"})
            continue
        sources.append(
            PublicSource(
                path=path,
                frame=frame,
                source_name=path.parent.name,
                content_hash=content_hash,
            )
        )
    return sources, ignored


def point_depth(point: int) -> str:
    value = int(point)
    if value == 0:
        return "terminal"
    if value <= 3:
        return "short"
    if value <= 6:
        return "half"
    return "long"


def build_ranked_point_votes(
    *,
    anchor: pd.DataFrame,
    sources: list[PublicSource],
    ignored_sources: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchor_points = anchor["pointId"].astype(int).to_numpy()
    ignored_sources = ignored_sources or []

    for row_idx, anchor_point in enumerate(anchor_points):
        votes: dict[int, list[PublicSource]] = defaultdict(list)
        for source in sources:
            point = int(source.frame.at[row_idx, "pointId"])
            if point != int(anchor_point):
                votes[point].append(source)

        for new_point, supporters in votes.items():
            if new_point == 0:
                continue
            if int(anchor_point) != 0 and new_point == 0:
                continue
            if len(supporters) < 2:
                continue

            source_names = [source.source_name for source in supporters]
            source_paths = [relative_path(source.path) for source in supporters]
            source_diversity = len(set(source_names))
            rows.append(
                {
                    "rank": 0,
                    "row_id": row_idx,
                    "rally_uid": anchor.at[row_idx, "rally_uid"],
                    "anchor_point": int(anchor_point),
                    "new_point": int(new_point),
                    "agreement_count": len(supporters),
                    "number_of_public_positive_sources": len(supporters),
                    "nonterminal_only": bool(int(anchor_point) != 0 and new_point != 0),
                    "source_diversity": source_diversity,
                    "stable_depth_change": bool(point_depth(int(anchor_point)) == point_depth(new_point)),
                    "point0_additions": 0,
                    "supporting_sources": "|".join(source_names),
                    "supporting_paths": "|".join(source_paths),
                    "ignored_source_count": len(ignored_sources),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rank",
                "row_id",
                "rally_uid",
                "anchor_point",
                "new_point",
                "agreement_count",
                "number_of_public_positive_sources",
                "nonterminal_only",
                "source_diversity",
                "stable_depth_change",
                "point0_additions",
                "supporting_sources",
                "supporting_paths",
                "ignored_source_count",
            ]
        )

    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        [
            "agreement_count",
            "number_of_public_positive_sources",
            "nonterminal_only",
            "source_diversity",
            "stable_depth_change",
            "row_id",
            "new_point",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def package_candidate(anchor: pd.DataFrame, selected_rows: pd.DataFrame) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    for row in selected_rows.itertuples(index=False):
        out.at[int(row.row_id), "pointId"] = int(row.new_point)
    if not out["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
        raise AssertionError("action changed")
    if not np.array_equal(
        pd.to_numeric(out["serverGetPoint"]).to_numpy(dtype=float),
        pd.to_numeric(anchor["serverGetPoint"]).to_numpy(dtype=float),
    ):
        raise AssertionError("server changed")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def candidate_filename(size: int) -> str:
    return f"submission_v400_public_agree_top{size}__v173action_v300server.csv"


def selected_filename(size: int) -> str:
    return f"selected_rows_v400_public_agree_top{size}.csv"


def transition_counts(base_point: pd.Series, cand_point: pd.Series) -> dict[str, int]:
    counts = Counter()
    for old, new in zip(base_point.astype(int), cand_point.astype(int)):
        if int(old) != int(new):
            counts[f"{int(old)}->{int(new)}"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def run_pipeline(*, outdir: Path = OUTDIR, expected_rows: int | None = 1845) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    anchor = load_anchor_submission(expected_rows=expected_rows)
    paths = discover_submission_paths()
    sources, ignored = load_public_positive_sources(anchor=anchor, paths=paths, expected_rows=expected_rows)
    row_votes = build_ranked_point_votes(anchor=anchor, sources=sources, ignored_sources=ignored)
    row_votes_path = safe_output_path(outdir, "row_votes.csv")
    row_votes.to_csv(row_votes_path, index=False)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for size in CANDIDATE_SIZES:
        selected = row_votes.head(size).copy()
        submission = package_candidate(anchor, selected)
        dist = point_distribution_report(anchor["pointId"], submission["pointId"])
        if dist["point0_additions"] != 0:
            raise AssertionError("point0 addition escaped candidate packaging")

        submission_path = safe_output_path(outdir, candidate_filename(size))
        selected_path = safe_output_path(outdir, selected_filename(size))
        submission.to_csv(submission_path, index=False)
        selected.to_csv(selected_path, index=False)

        action_churn = int((submission["actionId"].astype(int) != anchor["actionId"].astype(int)).sum())
        server_changed = int(
            np.sum(
                pd.to_numeric(submission["serverGetPoint"]).to_numpy(dtype=float)
                != pd.to_numeric(anchor["serverGetPoint"]).to_numpy(dtype=float)
            )
        )
        candidate = f"v400_public_agree_top{size}"
        row = {
            "candidate": candidate,
            "path": str(submission_path),
            "selected_rows": str(selected_path),
            "selected_row_count": int(len(selected)),
            "action_churn": action_churn,
            "point_churn": int(dist["changed_rows"]),
            "point0_additions": int(dist["point0_additions"]),
            "server_changed": server_changed,
            "risk": "safe" if len(selected) <= 15 else "medium",
            "evidence": "deterministic_public_source_agreement",
            "transition_counts": transition_counts(anchor["pointId"], submission["pointId"]),
        }
        summary_rows.append(row)
        generated.append(row.copy())

    ranked_candidates = pd.DataFrame(summary_rows)
    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    ranked_candidates.to_csv(ranked_path, index=False)

    ignored_reasons = Counter(str(item["reason"]) for item in ignored)
    report = {
        "version": "V400",
        "anchor": relative_path(ANCHOR_PATH),
        "anchor_rows": int(len(anchor)),
        "discovered_path_count": len(paths),
        "usable_source_count": len(sources),
        "ignored_source_count": len(ignored),
        "ignored_reasons": dict(sorted(ignored_reasons.items())),
        "row_vote_count": int(len(row_votes)),
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "ranked_candidates": str(ranked_path),
        "row_votes": str(row_votes_path),
        "ignored_sources": ignored,
        "policy": {
            "anchor": "V362",
            "point_only": True,
            "no_point0_additions": True,
            "action_preserved": True,
            "server_preserved": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "minimum_non_anchor_agreement": 2,
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
