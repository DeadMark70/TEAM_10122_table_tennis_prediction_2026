"""V347 changed-row differential audit for V338/V341/V345/V344.

This script writes reports only under v347_v338_v341_diff_audit. It compares
pointId edits against the V306 anchor and never writes upload candidates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v347_v338_v341_diff_audit"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_PUBLIC_POSITIVE = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V341_DIR = ROOT / "v341_no_p0_point_pack"
V345_B36 = ROOT / "v345_nonpoint0_utility_optimizer" / "submission_v345_nonp0_util_b36__v173action_v300server.csv"
V344_K12 = ROOT / "v344_point0_swap_optimizer" / "submission_v344_point0_swap_k12__v173action_v300server.csv"
V343_BANK = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
TEST_FEATURES = ROOT / "test_new.csv"

REPORT_COLUMNS = [
    "row_id",
    "rally_uid",
    "old_point",
    "new_point",
    "transition",
    "in_v338",
    "in_v341",
    "in_v345_b36",
    "in_v344_k12",
    "source_count_from_candidate_bank",
    "same_depth",
    "old_depth",
    "new_depth",
    "old_side",
    "new_side",
    "candidate_sources",
    "v341_candidate_count",
    "row_transition_count",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return (point - 1) // 3


def point_side(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return (point - 1) % 3


def same_depth(old_point: int, new_point: int) -> bool:
    return bool(point_depth(old_point) == point_depth(new_point))


def discover_v341_paths(v341_dir: Path = V341_DIR) -> list[Path]:
    if not v341_dir.exists():
        return []
    return [
        path
        for path in sorted(v341_dir.glob("submission*.csv"))
        if "ttmatch" not in path.name.lower() and "old_server" not in path.name.lower()
    ]


def extract_changed_transitions(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    family: str,
    source: str,
) -> pd.DataFrame:
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError(f"{source} rally_uid order differs from V306")
    old = base["pointId"].to_numpy(dtype=int)
    new = candidate["pointId"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for row_id in np.flatnonzero(old != new):
        rows.append(
            {
                "row_id": int(row_id),
                "rally_uid": base.at[int(row_id), "rally_uid"],
                "old_point": int(old[row_id]),
                "new_point": int(new[row_id]),
                "transition": f"{int(old[row_id])}->{int(new[row_id])}",
                "family": family,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def load_candidate_bank_counts(bank_path: Path = V343_BANK) -> pd.DataFrame:
    columns = ["row_id", "old_point", "new_point", "source_count_from_candidate_bank"]
    if not bank_path.exists():
        return pd.DataFrame(columns=columns)
    bank = pd.read_csv(bank_path)
    required = {"row_id", "anchor_value", "candidate_value", "source"}
    if not required.issubset(bank.columns):
        return pd.DataFrame(columns=columns)
    work = bank.copy()
    work["row_id"] = pd.to_numeric(work["row_id"], errors="coerce")
    work["old_point"] = pd.to_numeric(work["anchor_value"], errors="coerce")
    work["new_point"] = pd.to_numeric(work["candidate_value"], errors="coerce")
    work = work.dropna(subset=["row_id", "old_point", "new_point"])
    work["row_id"] = work["row_id"].astype(int)
    work["old_point"] = work["old_point"].astype(int)
    work["new_point"] = work["new_point"].astype(int)
    grouped = (
        work.groupby(["row_id", "old_point", "new_point"], as_index=False)["source"]
        .nunique()
        .rename(columns={"source": "source_count_from_candidate_bank"})
    )
    return grouped.loc[:, columns]


def load_feature_slice(feature_path: Path = TEST_FEATURES) -> pd.DataFrame:
    if not feature_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(feature_path)
    keep = ["rally_uid"]
    for col in frame.columns:
        lower = str(col).lower()
        if lower in {"strikenumber", "rally_id", "match", "numbergame"}:
            keep.append(col)
        elif "phase" in lower or "prefix" in lower or "lag0" in lower:
            keep.append(col)
    keep = list(dict.fromkeys([col for col in keep if col in frame.columns]))
    out = frame.loc[:, keep].copy()
    if "strikeNumber" in out.columns and "prefix_len" not in out.columns:
        out["prefix_len"] = pd.to_numeric(out["strikeNumber"], errors="coerce")
    return out


def build_row_diff(
    base: pd.DataFrame,
    named_candidates: list[tuple[str, str, pd.DataFrame]],
    *,
    bank_counts: pd.DataFrame | None = None,
    feature_slice: pd.DataFrame | None = None,
) -> pd.DataFrame:
    changed_parts = [
        extract_changed_transitions(base, candidate, family, source)
        for family, source, candidate in named_candidates
    ]
    changed = pd.concat([part for part in changed_parts if not part.empty], ignore_index=True) if changed_parts else pd.DataFrame()
    if changed.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    keys = ["row_id", "rally_uid", "old_point", "new_point", "transition"]
    records: list[dict[str, Any]] = []
    for key_values, group in changed.groupby(keys, sort=True):
        record = dict(zip(keys, key_values))
        families = set(group["family"].astype(str))
        sources = sorted(set(group["source"].astype(str)))
        v341_sources = sorted(set(group.loc[group["family"].eq("v341"), "source"].astype(str)))
        old_point = int(record["old_point"])
        new_point = int(record["new_point"])
        record.update(
            {
                "in_v338": "v338" in families,
                "in_v341": "v341" in families,
                "in_v345_b36": "v345_b36" in families,
                "in_v344_k12": "v344_k12" in families,
                "source_count_from_candidate_bank": 0,
                "same_depth": same_depth(old_point, new_point),
                "old_depth": point_depth(old_point),
                "new_depth": point_depth(new_point),
                "old_side": point_side(old_point),
                "new_side": point_side(new_point),
                "candidate_sources": "|".join(sources),
                "v341_candidate_count": int(len(v341_sources)),
            }
        )
        records.append(record)

    out = pd.DataFrame(records)
    out["row_transition_count"] = out.groupby("row_id")["new_point"].transform("nunique").astype(int)

    if bank_counts is not None and not bank_counts.empty:
        out = out.merge(bank_counts, on=["row_id", "old_point", "new_point"], how="left", suffixes=("", "_bank"))
        if "source_count_from_candidate_bank_bank" in out.columns:
            out["source_count_from_candidate_bank"] = (
                pd.to_numeric(out["source_count_from_candidate_bank_bank"], errors="coerce").fillna(0).astype(int)
            )
            out = out.drop(columns=["source_count_from_candidate_bank_bank"])

    if feature_slice is not None and not feature_slice.empty:
        features = feature_slice.copy()
        features.insert(0, "row_id", np.arange(len(features), dtype=int))
        features = features.drop(columns=["rally_uid"], errors="ignore")
        out = out.merge(features, on="row_id", how="left")

    fixed = [col for col in REPORT_COLUMNS if col in out.columns]
    extra = [col for col in out.columns if col not in fixed]
    return out.loc[:, fixed + extra].sort_values(["row_id", "new_point"]).reset_index(drop=True)


def build_transition_summary(row_diff: pd.DataFrame) -> pd.DataFrame:
    if row_diff.empty:
        return pd.DataFrame(
            columns=[
                "transition",
                "rows",
                "in_v338",
                "in_v341",
                "in_v345_b36",
                "in_v344_k12",
                "same_depth_rows",
                "mean_source_count_from_candidate_bank",
            ]
        )
    summary = (
        row_diff.groupby("transition", as_index=False)
        .agg(
            rows=("row_id", "count"),
            in_v338=("in_v338", "sum"),
            in_v341=("in_v341", "sum"),
            in_v345_b36=("in_v345_b36", "sum"),
            in_v344_k12=("in_v344_k12", "sum"),
            same_depth_rows=("same_depth", "sum"),
            mean_source_count_from_candidate_bank=("source_count_from_candidate_bank", "mean"),
        )
        .sort_values(["rows", "transition"], ascending=[False, True])
    )
    return summary


def build_slice_summary(row_diff: pd.DataFrame) -> pd.DataFrame:
    if row_diff.empty:
        return pd.DataFrame(columns=["slice", "rows", "unique_rows", "same_depth_rows", "point0_additions"])
    work = row_diff.copy()
    work["slice"] = work.apply(
        lambda row: "|".join(
            name
            for name, enabled in [
                ("v338", bool(row["in_v338"])),
                ("v341", bool(row["in_v341"])),
                ("v345_b36", bool(row["in_v345_b36"])),
                ("v344_k12", bool(row["in_v344_k12"])),
            ]
            if enabled
        ),
        axis=1,
    )
    work["point0_addition"] = (work["old_point"].astype(int) != 0) & (work["new_point"].astype(int) == 0)
    summary = (
        work.groupby("slice", as_index=False)
        .agg(
            rows=("row_id", "count"),
            unique_rows=("row_id", "nunique"),
            same_depth_rows=("same_depth", "sum"),
            point0_additions=("point0_addition", "sum"),
            mean_source_count_from_candidate_bank=("source_count_from_candidate_bank", "mean"),
        )
        .sort_values(["rows", "slice"], ascending=[False, True])
    )
    return summary


def run_pipeline(*, outdir: Path = OUTDIR, expected_rows: int | None = 1845) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_paths = [V306_ANCHOR, V338_PUBLIC_POSITIVE, V345_B36, V344_K12]
    v341_paths = discover_v341_paths()
    missing = [relative_path(path) for path in fixed_paths if not path.exists()]
    if not v341_paths:
        missing.append(relative_path(V341_DIR / "submission*.csv"))
    if missing:
        report = {"version": "V347", "decision": "BLOCKED_MISSING_INPUT", "missing": missing}
        write_json(safe_output_path(outdir, "search_report.json"), report)
        return report

    base = load_submission(V306_ANCHOR, expected_rows=expected_rows)
    v338 = load_submission(V338_PUBLIC_POSITIVE, expected_rows=expected_rows)
    v345_b36 = load_submission(V345_B36, expected_rows=expected_rows)
    v344_k12 = load_submission(V344_K12, expected_rows=expected_rows)

    named_candidates: list[tuple[str, str, pd.DataFrame]] = [
        ("v338", relative_path(V338_PUBLIC_POSITIVE), v338),
        ("v345_b36", relative_path(V345_B36), v345_b36),
        ("v344_k12", relative_path(V344_K12), v344_k12),
    ]
    skipped_v341: list[dict[str, Any]] = []
    for path in v341_paths:
        try:
            frame = load_submission(path, expected_rows=expected_rows)
            named_candidates.append(("v341", relative_path(path), frame))
        except Exception as exc:  # noqa: BLE001 - report all skipped candidates.
            skipped_v341.append({"path": relative_path(path), "reason": str(exc)})

    row_diff = build_row_diff(
        base,
        named_candidates,
        bank_counts=load_candidate_bank_counts(),
        feature_slice=load_feature_slice(),
    )
    transition_summary = build_transition_summary(row_diff)
    slice_summary = build_slice_summary(row_diff)

    row_diff_path = safe_output_path(outdir, "row_diff.csv")
    transition_path = safe_output_path(outdir, "transition_summary.csv")
    slice_path = safe_output_path(outdir, "slice_summary.csv")
    row_diff.to_csv(row_diff_path, index=False)
    transition_summary.to_csv(transition_path, index=False)
    slice_summary.to_csv(slice_path, index=False)

    v341_extra = row_diff[row_diff["in_v341"].astype(bool) & ~row_diff["in_v338"].astype(bool)]
    v338_shared_with_v341 = row_diff[row_diff["in_v341"].astype(bool) & row_diff["in_v338"].astype(bool)]
    v345_new_vs_v338 = row_diff[row_diff["in_v345_b36"].astype(bool) & ~row_diff["in_v338"].astype(bool)]
    v344_new_vs_v338 = row_diff[row_diff["in_v344_k12"].astype(bool) & ~row_diff["in_v338"].astype(bool)]
    report = {
        "version": "V347",
        "decision": "REPORTS_EXPORTED",
        "outputs": {
            "row_diff": relative_path(row_diff_path),
            "transition_summary": relative_path(transition_path),
            "slice_summary": relative_path(slice_path),
            "search_report": relative_path(outdir / "search_report.json"),
        },
        "inputs": {
            "v306_anchor": relative_path(V306_ANCHOR),
            "v338_public_positive": relative_path(V338_PUBLIC_POSITIVE),
            "v341_candidate_count": len(v341_paths),
            "v341_candidates": [relative_path(path) for path in v341_paths],
            "v345_b36": relative_path(V345_B36),
            "v344_k12": relative_path(V344_K12),
            "candidate_bank": relative_path(V343_BANK) if V343_BANK.exists() else None,
            "feature_slice": relative_path(TEST_FEATURES) if TEST_FEATURES.exists() else None,
        },
        "skipped_v341_candidates": skipped_v341,
        "row_diff_rows": int(len(row_diff)),
        "unique_changed_rows": int(row_diff["row_id"].nunique()) if not row_diff.empty else 0,
        "v338_rows": int(row_diff["in_v338"].sum()) if not row_diff.empty else 0,
        "v341_rows": int(row_diff["in_v341"].sum()) if not row_diff.empty else 0,
        "v341_extra_rows_beyond_v338": int(len(v341_extra)),
        "v338_rows_shared_with_v341": int(len(v338_shared_with_v341)),
        "v345_b36_rows": int(row_diff["in_v345_b36"].sum()) if not row_diff.empty else 0,
        "v345_b36_new_rows_beyond_v338": int(len(v345_new_vs_v338)),
        "v344_k12_rows": int(row_diff["in_v344_k12"].sum()) if not row_diff.empty else 0,
        "v344_k12_new_rows_beyond_v338": int(len(v344_new_vs_v338)),
        "top_transitions": transition_summary.head(10).to_dict("records"),
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "manual_row_edits": False,
            "upload_candidates_writes": False,
            "submission_exports": False,
            "reports_only": True,
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), _json_safe(report))
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
