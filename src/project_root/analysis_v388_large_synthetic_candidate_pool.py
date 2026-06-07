"""V388 large synthetic candidate evidence pool.

This script scans clean historical candidate directories and extracts point and
action changes relative to the V362 anchor. It emits evidence pools only; it
does not create submission files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis_v335_moe_anchor_contract import SERVE_ACTION_CLASSES, SUBMISSION_COLUMNS, safe_output_path


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v388_large_synthetic_candidate_pool"
ANCHOR = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)

SOURCE_DIRS = [
    "v261_action_conditioned_point_residual",
    "v272_action_conditioned_point_residual",
    "v277_v272b_point_refinement",
    "v300_clean_server_blend_recycler",
    "v306_point0_addition_probe",
    "v307_point0_dose_extension",
    "v338_joint_moe_pack",
    "v341_no_p0_point_pack",
    "v345_nonpoint0_utility_optimizer",
    "v362_point_hierarchical_specialists",
    "v370_point_breakthrough_pool",
    "v374_physical_rule_audit",
    "v383_synthetic_adjusted_packager",
    "v387_expanded_synthetic_packager",
]

POINT_POOL_COLUMNS = [
    "rally_uid",
    "base_point",
    "candidate_point",
    "transition",
    "source_file",
    "source_dir",
    "support_count",
    "source_family_count",
    "is_point0_addition",
    "is_point0_removal",
    "same_depth",
    "same_side",
]

ACTION_POOL_COLUMNS = [
    "rally_uid",
    "base_action",
    "candidate_action",
    "source_file",
    "source_dir",
    "support_count",
    "source_family_count",
    "is_serve_15_18_addition",
    "same_family",
]

SUMMARY_COLUMNS = [
    "source_dir",
    "status",
    "csv_files_seen",
    "valid_candidate_files",
    "point_change_rows",
    "action_change_rows",
    "skipped_reason",
]

POINT_DEPTH = {
    0: "terminal",
    1: "short",
    2: "short",
    3: "short",
    4: "half",
    5: "half",
    6: "half",
    7: "long",
    8: "long",
    9: "long",
}

POINT_SIDE = {
    0: "terminal",
    1: "left",
    2: "middle",
    3: "right",
    4: "left",
    5: "middle",
    6: "right",
    7: "left",
    8: "middle",
    9: "right",
}

ACTION_FAMILY_BY_ID = {
    0: "unknown",
    1: "attack",
    2: "attack",
    3: "attack",
    4: "receive",
    5: "control",
    6: "control",
    7: "receive",
    8: "defensive",
    9: "defensive",
    10: "defensive",
    11: "control",
    12: "attack",
    13: "setup",
    14: "setup",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}


def output_filenames() -> list[str]:
    return [
        "point_change_pool.csv",
        "action_change_pool.csv",
        "candidate_source_summary.csv",
        "search_report.json",
    ]


def _norm_int(value: object, default: int = -1) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: bool) -> bool:
    return bool(value)


def _join_unique(values: Iterable[object]) -> str:
    cleaned = sorted({str(value) for value in values if str(value)})
    return "|".join(cleaned)


def _empty_point_pool() -> pd.DataFrame:
    return pd.DataFrame(columns=POINT_POOL_COLUMNS)


def _empty_action_pool() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_POOL_COLUMNS)


def validate_anchor_frame(frame: pd.DataFrame) -> list[str]:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"anchor schema mismatch: {list(frame.columns)}")
    if frame["rally_uid"].duplicated().any():
        raise ValueError("anchor rally_uid values must be unique")
    if not pd.to_numeric(frame["actionId"], errors="coerce").between(0, 18).all():
        raise ValueError("anchor actionId out of range")
    if not pd.to_numeric(frame["pointId"], errors="coerce").between(0, 9).all():
        raise ValueError("anchor pointId out of range")
    server = pd.to_numeric(frame["serverGetPoint"], errors="coerce")
    if server.isna().any() or not server.between(0.0, 1.0).all():
        raise ValueError("anchor serverGetPoint out of range")
    return list(frame.columns)


def _validate_candidate_submission(frame: pd.DataFrame, anchor_rows: int) -> bool:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        return False
    if len(frame) != anchor_rows:
        return False
    return not frame["rally_uid"].duplicated().any()


def _point_depth(point_id: object) -> str:
    return POINT_DEPTH.get(_norm_int(point_id), "unknown")


def _point_side(point_id: object) -> str:
    return POINT_SIDE.get(_norm_int(point_id), "unknown")


def _action_family(action_id: object) -> str:
    return ACTION_FAMILY_BY_ID.get(_norm_int(action_id), "unknown")


def _point_rows_from_submission(
    anchor: pd.DataFrame, candidate: pd.DataFrame, source_file: str, source_dir: str
) -> list[dict[str, object]]:
    merged = anchor.merge(candidate, on="rally_uid", suffixes=("_base", "_candidate"))
    changed = merged[
        pd.to_numeric(merged["pointId_base"], errors="coerce")
        != pd.to_numeric(merged["pointId_candidate"], errors="coerce")
    ]
    rows = []
    for _, row in changed.iterrows():
        base = _norm_int(row["pointId_base"])
        cand = _norm_int(row["pointId_candidate"])
        rows.append(
            {
                "rally_uid": row["rally_uid"],
                "base_point": base,
                "candidate_point": cand,
                "transition": f"{base}->{cand}",
                "source_file": source_file,
                "source_dir": source_dir,
                "is_point0_addition": _bool(base != 0 and cand == 0),
                "is_point0_removal": _bool(base == 0 and cand != 0),
                "same_depth": _bool(_point_depth(base) == _point_depth(cand)),
                "same_side": _bool(_point_side(base) == _point_side(cand)),
            }
        )
    return rows


def _action_rows_from_submission(
    anchor: pd.DataFrame, candidate: pd.DataFrame, source_file: str, source_dir: str
) -> list[dict[str, object]]:
    merged = anchor.merge(candidate, on="rally_uid", suffixes=("_base", "_candidate"))
    changed = merged[
        pd.to_numeric(merged["actionId_base"], errors="coerce")
        != pd.to_numeric(merged["actionId_candidate"], errors="coerce")
    ]
    rows = []
    for _, row in changed.iterrows():
        base = _norm_int(row["actionId_base"])
        cand = _norm_int(row["actionId_candidate"])
        rows.append(
            {
                "rally_uid": row["rally_uid"],
                "base_action": base,
                "candidate_action": cand,
                "source_file": source_file,
                "source_dir": source_dir,
                "is_serve_15_18_addition": _bool(base not in SERVE_ACTION_CLASSES and cand in SERVE_ACTION_CLASSES),
                "same_family": _bool(_action_family(base) == _action_family(cand)),
            }
        )
    return rows


def _point_rows_from_candidate_table(frame: pd.DataFrame, source_file: str, source_dir: str) -> list[dict[str, object]]:
    required = {"rally_uid", "base_point", "candidate_point"}
    if not required.issubset(frame.columns):
        return []
    rows = []
    for _, row in frame.iterrows():
        base = _norm_int(row["base_point"])
        cand = _norm_int(row["candidate_point"])
        if base == cand or cand < 0:
            continue
        rows.append(
            {
                "rally_uid": row["rally_uid"],
                "base_point": base,
                "candidate_point": cand,
                "transition": str(row.get("transition", f"{base}->{cand}")),
                "source_file": source_file,
                "source_dir": source_dir,
                "is_point0_addition": _bool(base != 0 and cand == 0),
                "is_point0_removal": _bool(base == 0 and cand != 0),
                "same_depth": _bool(_point_depth(base) == _point_depth(cand)),
                "same_side": _bool(_point_side(base) == _point_side(cand)),
            }
        )
    return rows


def _action_rows_from_candidate_table(frame: pd.DataFrame, source_file: str, source_dir: str) -> list[dict[str, object]]:
    required = {"rally_uid", "base_action", "candidate_action"}
    if not required.issubset(frame.columns):
        return []
    rows = []
    for _, row in frame.iterrows():
        base = _norm_int(row["base_action"])
        cand = _norm_int(row["candidate_action"])
        if base == cand or cand < 0:
            continue
        rows.append(
            {
                "rally_uid": row["rally_uid"],
                "base_action": base,
                "candidate_action": cand,
                "source_file": source_file,
                "source_dir": source_dir,
                "is_serve_15_18_addition": _bool(base not in SERVE_ACTION_CLASSES and cand in SERVE_ACTION_CLASSES),
                "same_family": _bool(_action_family(base) == _action_family(cand)),
            }
        )
    return rows


def _aggregate_rows(rows: list[dict[str, object]], group_cols: list[str], output_cols: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=output_cols)
    frame = pd.DataFrame(rows)
    aggregated = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            source_file=("source_file", _join_unique),
            source_dir=("source_dir", _join_unique),
            support_count=("source_file", "size"),
            source_family_count=("source_dir", lambda values: len({str(value) for value in values})),
        )
        .reset_index()
    )
    for col in output_cols:
        if col in aggregated.columns:
            continue
        if col == "transition":
            aggregated[col] = aggregated["base_point"].astype(str) + "->" + aggregated["candidate_point"].astype(str)
        elif col == "is_point0_addition":
            aggregated[col] = (aggregated["base_point"].astype(int) != 0) & (
                aggregated["candidate_point"].astype(int) == 0
            )
        elif col == "is_point0_removal":
            aggregated[col] = (aggregated["base_point"].astype(int) == 0) & (
                aggregated["candidate_point"].astype(int) != 0
            )
        elif col == "same_depth":
            aggregated[col] = [
                _point_depth(base) == _point_depth(cand)
                for base, cand in zip(aggregated["base_point"], aggregated["candidate_point"])
            ]
        elif col == "same_side":
            aggregated[col] = [
                _point_side(base) == _point_side(cand)
                for base, cand in zip(aggregated["base_point"], aggregated["candidate_point"])
            ]
        elif col == "is_serve_15_18_addition":
            aggregated[col] = [
                _norm_int(base) not in SERVE_ACTION_CLASSES and _norm_int(cand) in SERVE_ACTION_CLASSES
                for base, cand in zip(aggregated["base_action"], aggregated["candidate_action"])
            ]
        elif col == "same_family":
            aggregated[col] = [
                _action_family(base) == _action_family(cand)
                for base, cand in zip(aggregated["base_action"], aggregated["candidate_action"])
            ]
        else:
            aggregated[col] = None

    bool_cols = [
        "is_point0_addition",
        "is_point0_removal",
        "same_depth",
        "same_side",
        "is_serve_15_18_addition",
        "same_family",
    ]
    for col in bool_cols:
        if col in aggregated.columns:
            aggregated[col] = aggregated[col].map(bool).astype(object)
    return aggregated.loc[:, output_cols].sort_values(
        ["support_count", "source_family_count", "rally_uid"], ascending=[False, False, True]
    )


def aggregate_point_pool(anchor: pd.DataFrame, candidates: Iterable[tuple[pd.DataFrame, str, str]]) -> pd.DataFrame:
    validate_anchor_frame(anchor)
    rows: list[dict[str, object]] = []
    for frame, source_file, source_dir in candidates:
        if _validate_candidate_submission(frame, len(anchor)):
            rows.extend(_point_rows_from_submission(anchor, frame, source_file, source_dir))
        else:
            rows.extend(_point_rows_from_candidate_table(frame, source_file, source_dir))
    return _aggregate_rows(
        rows,
        ["rally_uid", "base_point", "candidate_point"],
        POINT_POOL_COLUMNS,
    )


def aggregate_action_pool(anchor: pd.DataFrame, candidates: Iterable[tuple[pd.DataFrame, str, str]]) -> pd.DataFrame:
    validate_anchor_frame(anchor)
    rows: list[dict[str, object]] = []
    for frame, source_file, source_dir in candidates:
        if _validate_candidate_submission(frame, len(anchor)):
            rows.extend(_action_rows_from_submission(anchor, frame, source_file, source_dir))
        else:
            rows.extend(_action_rows_from_candidate_table(frame, source_file, source_dir))
    return _aggregate_rows(
        rows,
        ["rally_uid", "base_action", "candidate_action"],
        ACTION_POOL_COLUMNS,
    )


def _read_candidate_csvs(root: Path, source_dirs: list[str], anchor: pd.DataFrame) -> tuple[list[tuple[pd.DataFrame, str, str]], pd.DataFrame, list[str]]:
    candidates: list[tuple[pd.DataFrame, str, str]] = []
    summary_rows: list[dict[str, object]] = []
    skipped_dirs: list[str] = []

    for source_dir in source_dirs:
        directory = root / source_dir
        if not directory.exists():
            skipped_dirs.append(source_dir)
            summary_rows.append(
                {
                    "source_dir": source_dir,
                    "status": "skipped",
                    "csv_files_seen": 0,
                    "valid_candidate_files": 0,
                    "point_change_rows": 0,
                    "action_change_rows": 0,
                    "skipped_reason": "missing_source_dir",
                }
            )
            continue

        csv_paths = sorted(path for path in directory.glob("*.csv") if path.is_file())
        valid_files = 0
        point_rows = 0
        action_rows = 0
        for path in csv_paths:
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            is_submission = _validate_candidate_submission(frame, len(anchor))
            has_point_candidates = {"rally_uid", "base_point", "candidate_point"}.issubset(frame.columns)
            has_action_candidates = {"rally_uid", "base_action", "candidate_action"}.issubset(frame.columns)
            if not (is_submission or has_point_candidates or has_action_candidates):
                continue
            valid_files += 1
            candidates.append((frame, path.name, source_dir))
            if is_submission:
                merged = anchor.merge(frame, on="rally_uid", suffixes=("_base", "_candidate"))
                point_rows += int((merged["pointId_base"].astype(int) != merged["pointId_candidate"].astype(int)).sum())
                action_rows += int(
                    (merged["actionId_base"].astype(int) != merged["actionId_candidate"].astype(int)).sum()
                )
            else:
                point_rows += len(_point_rows_from_candidate_table(frame, path.name, source_dir))
                action_rows += len(_action_rows_from_candidate_table(frame, path.name, source_dir))

        summary_rows.append(
            {
                "source_dir": source_dir,
                "status": "ok",
                "csv_files_seen": len(csv_paths),
                "valid_candidate_files": valid_files,
                "point_change_rows": point_rows,
                "action_change_rows": action_rows,
                "skipped_reason": "",
            }
        )

    return candidates, pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS), skipped_dirs


def run_pipeline(
    root: str | Path = ROOT,
    outdir: str | Path = OUTDIR,
    source_dirs: list[str] | None = None,
) -> dict[str, object]:
    root = Path(root)
    output_dir = Path(outdir)
    if output_dir == OUTDIR:
        output_dir = root / OUTDIR.name
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = SOURCE_DIRS if source_dirs is None else source_dirs
    anchor_path = root / ANCHOR.relative_to(ROOT)
    anchor = pd.read_csv(anchor_path)
    anchor_columns = validate_anchor_frame(anchor)

    candidates, summary, skipped_dirs = _read_candidate_csvs(root, source_dirs, anchor)
    point_pool = aggregate_point_pool(anchor, candidates)
    action_pool = aggregate_action_pool(anchor, candidates)

    point_pool.to_csv(safe_output_path(output_dir, "point_change_pool.csv"), index=False)
    action_pool.to_csv(safe_output_path(output_dir, "action_change_pool.csv"), index=False)
    summary.to_csv(safe_output_path(output_dir, "candidate_source_summary.csv"), index=False)

    report = {
        "version": "v388_large_synthetic_candidate_pool",
        "purpose": "Large clean historical point/action candidate evidence pool for synthetic OOF/proxy stages.",
        "anchor": anchor_path.relative_to(root).as_posix() if anchor_path.is_relative_to(root) else str(anchor_path),
        "anchor_columns": anchor_columns,
        "source_dirs_requested": source_dirs,
        "skipped_dirs": skipped_dirs,
        "source_dirs_scanned": [row for row in source_dirs if row not in skipped_dirs],
        "candidate_files_loaded": int(len(candidates)),
        "point_pool_rows": int(len(point_pool)),
        "action_pool_rows": int(len(action_pool)),
        "point0_additions": int(point_pool["is_point0_addition"].sum()) if not point_pool.empty else 0,
        "point0_removals": int(point_pool["is_point0_removal"].sum()) if not point_pool.empty else 0,
        "serve_15_18_action_additions": int(action_pool["is_serve_15_18_addition"].sum())
        if not action_pool.empty
        else 0,
        "outputs": output_filenames(),
        "policy": [
            "No submission CSVs emitted by V388.",
            "Source scan limited to the clean V388 source directory list.",
            "Synthetic data is not used for direct answer generation.",
        ],
    }
    safe_output_path(output_dir, "search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
