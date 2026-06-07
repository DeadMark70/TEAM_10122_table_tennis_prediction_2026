"""V414 masked/coarse pretraining input builder.

Turns V413 clean canonical events into coarse objectives for later
representation learning. No exact AICUP labels are emitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V413_CLEAN = ROOT / "v413_external_license_overlap_guard" / "canonical_clean_events.csv"
OUTDIR = ROOT / "v414_masked_pretraining_inputs"

FORBIDDEN_COLUMNS = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
EXCLUDED_SOURCES = {"TTMATCH", "sonytabletennis", "TT-MatchDynamics"}

EXPECTED_BINS = {
    "landing_depth_bin": {"short", "half", "long", "terminal_or_unknown", "unknown"},
    "landing_side_bin": {"left", "middle", "right", "unknown"},
    "speed_bin": {"low", "medium", "high", "unknown"},
    "spin_bin": {"low", "medium", "high", "unknown"},
}


def _clean_source_rows(canonical: pd.DataFrame) -> pd.DataFrame:
    if canonical.empty:
        return canonical.copy()
    source = canonical.get("source_dataset", pd.Series(dtype=str)).astype(str)
    mask = ~source.isin(EXCLUDED_SOURCES)
    mask &= ~source.str.contains("ttmatch", case=False, na=False)
    return canonical[mask].copy()


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _axis_bins(values: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    numeric = _numeric(values)
    result = pd.Series(["unknown"] * len(values), index=values.index, dtype=object)
    valid = numeric.dropna()
    if len(valid) < 3 or valid.nunique() < 3:
        return result
    q1, q2 = valid.quantile([1 / 3, 2 / 3]).tolist()
    result.loc[numeric <= q1] = labels[0]
    result.loc[(numeric > q1) & (numeric <= q2)] = labels[1]
    result.loc[numeric > q2] = labels[2]
    return result


def _value_bins_by_source(frame: pd.DataFrame, column: str, labels: tuple[str, str, str]) -> pd.Series:
    result = pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object)
    if column not in frame.columns:
        return result
    for _, idx in frame.groupby("source_dataset", dropna=False).groups.items():
        result.loc[idx] = _axis_bins(frame.loc[idx, column], labels)
    return result


def _derive_depth_side(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    depth = frame.get("landing_depth_bin", pd.Series(["unknown"] * len(frame), index=frame.index)).map(_normalize_text)
    side = frame.get("landing_side_bin", pd.Series(["unknown"] * len(frame), index=frame.index)).map(_normalize_text)
    depth = depth.where(depth.isin(EXPECTED_BINS["landing_depth_bin"]), "unknown")
    side = side.where(side.isin(EXPECTED_BINS["landing_side_bin"]), "unknown")

    missing_depth = depth == "unknown"
    missing_side = side == "unknown"
    if "landing_y" in frame.columns and missing_depth.any():
        derived = _value_bins_by_source(frame.loc[missing_depth], "landing_y", ("short", "half", "long"))
        depth.loc[missing_depth] = derived
    if "landing_x" in frame.columns and missing_side.any():
        derived = _value_bins_by_source(frame.loc[missing_side], "landing_x", ("left", "middle", "right"))
        side.loc[missing_side] = derived
    return depth.fillna("unknown"), side.fillna("unknown")


def _build_sequences(clean: pd.DataFrame) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "sequence_id",
                "event_index",
                "token_family",
                "phase",
                "terminal_label",
                "landing_depth_bin",
                "landing_side_bin",
                "speed_bin",
                "spin_bin",
                "maskable",
            ]
        )
    clean = clean.copy()
    depth, side = _derive_depth_side(clean)
    speed_bin = _value_bins_by_source(clean, "speed_norm", ("low", "medium", "high"))
    spin_bin = _value_bins_by_source(clean, "spin_norm", ("low", "medium", "high"))
    seq = pd.DataFrame(
        {
            "source_dataset": clean["source_dataset"].map(lambda x: _normalize_text(x)),
            "sequence_id": clean["sequence_id"].map(lambda x: _normalize_text(x)),
            "event_index": pd.to_numeric(clean.get("event_index", pd.Series(range(len(clean)))), errors="coerce").fillna(0).astype(int),
            "token_family": clean.get("coarse_family", pd.Series(index=clean.index)).map(_normalize_text),
            "phase": clean.get("phase", pd.Series(index=clean.index)).map(_normalize_text),
            "terminal_label": clean.get("terminal_label", pd.Series(index=clean.index)).map(_normalize_text),
            "landing_depth_bin": depth,
            "landing_side_bin": side,
            "speed_bin": speed_bin,
            "spin_bin": spin_bin,
        }
    )
    seq["maskable"] = seq["token_family"].ne("unknown")
    return seq.sort_values(["source_dataset", "sequence_id", "event_index"]).reset_index(drop=True)


def _context_for(group: pd.DataFrame, position: int) -> list[int]:
    start = max(position - 2, 0)
    stop = min(position + 3, len(group))
    indices = [int(group.iloc[i]["event_index"]) for i in range(start, stop) if i != position]
    return indices


def _build_masked_examples(seq: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, sequence_id), group in seq.groupby(["source_dataset", "sequence_id"], sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        for pos, item in group.iterrows():
            if not bool(item["maskable"]):
                continue
            rows.append(
                {
                    "source_dataset": item["source_dataset"],
                    "sequence_id": item["sequence_id"],
                    "context_indices": json.dumps(_context_for(group, pos)),
                    "masked_index": int(item["event_index"]),
                    "target_family": item["token_family"],
                    "target_phase": item["phase"],
                    "target_terminal": item["terminal_label"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["source_dataset", "sequence_id", "context_indices", "masked_index", "target_family", "target_phase", "target_terminal"],
    )


def _build_landing_examples(seq: pd.DataFrame) -> pd.DataFrame:
    mask = seq["landing_depth_bin"].ne("unknown") | seq["landing_side_bin"].ne("unknown")
    out = seq.loc[mask, ["source_dataset", "sequence_id", "event_index", "landing_depth_bin", "landing_side_bin"]].copy()
    out = out.rename(columns={"landing_depth_bin": "target_depth_bin", "landing_side_bin": "target_side_bin"})
    out["has_xy"] = True
    return out[["source_dataset", "sequence_id", "event_index", "target_depth_bin", "target_side_bin", "has_xy"]]


def _build_physics_examples(clean: pd.DataFrame) -> pd.DataFrame:
    cols = ["source_dataset", "sequence_id", "event_index", "speed_norm", "spin_norm", "landing_x", "landing_y", "landing_z"]
    for col in cols:
        if col not in clean.columns:
            clean[col] = pd.NA
    out = clean[cols].copy()
    numeric_cols = ["speed_norm", "spin_norm", "landing_x", "landing_y", "landing_z"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = out[numeric_cols].notna().any(axis=1)
    return out.loc[mask].reset_index(drop=True)


def _build_response_pairs(seq: pd.DataFrame, max_pairs: int = 5000) -> pd.DataFrame:
    usable = seq[seq["token_family"].ne("unknown")].drop_duplicates(["source_dataset", "sequence_id", "token_family"])
    rows: list[dict[str, Any]] = []
    for source, source_group in usable.groupby("source_dataset", sort=False):
        families = list(source_group["token_family"].drop_duplicates())
        if len(families) < 2:
            continue
        by_family = {family: list(group["sequence_id"].drop_duplicates()) for family, group in source_group.groupby("token_family")}
        for family, seq_ids in by_family.items():
            if len(seq_ids) < 2:
                continue
            negative_family = next((candidate for candidate in families if candidate != family), None)
            if negative_family is None or not by_family.get(negative_family):
                continue
            for idx in range(min(len(seq_ids) - 1, 20)):
                rows.append(
                    {
                        "source_dataset": source,
                        "anchor_sequence_id": seq_ids[idx],
                        "positive_sequence_id": seq_ids[idx + 1],
                        "negative_sequence_id": by_family[negative_family][idx % len(by_family[negative_family])],
                        "pair_type": "same_family_vs_other_family",
                    }
                )
                if len(rows) >= max_pairs:
                    return pd.DataFrame(rows)
    return pd.DataFrame(
        rows,
        columns=["source_dataset", "anchor_sequence_id", "positive_sequence_id", "negative_sequence_id", "pair_type"],
    )


def _assert_no_forbidden_columns(outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in outputs.items():
        overlap = FORBIDDEN_COLUMNS & set(frame.columns)
        if overlap:
            raise ValueError(f"{name} contains forbidden exact AICUP columns: {sorted(overlap)}")


def build_pretraining_inputs(canonical: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    clean = _clean_source_rows(canonical)
    seq = _build_sequences(clean)
    outputs = {
        "pretrain_sequences": seq,
        "masked_event_examples": _build_masked_examples(seq),
        "landing_intent_examples": _build_landing_examples(seq),
        "physics_reconstruction_examples": _build_physics_examples(clean),
        "response_style_pairs": _build_response_pairs(seq),
    }
    _assert_no_forbidden_columns(outputs)
    report = {
        "version": "V414",
        "input_rows": int(len(canonical)),
        "clean_rows": int(len(clean)),
        "source_counts": seq.groupby("source_dataset").size().to_dict() if not seq.empty else {},
        "objective_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "excluded_sources": sorted(EXCLUDED_SOURCES),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
    }
    return outputs, report


def run_pipeline(
    *,
    canonical_path: Path = V413_CLEAN,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(canonical_path, low_memory=False)
    outputs, report = build_pretraining_inputs(canonical)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = outdir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)
    report_path = outdir / "pretraining_input_report.json"
    report["outputs"] = {**output_paths, "pretraining_input_report": str(report_path)}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
