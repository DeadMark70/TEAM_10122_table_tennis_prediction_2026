"""V412 clean canonical external corpus builder.

Converts allowed external sources into a coarse, license-aware event schema for
representation pretraining. It intentionally avoids AICUP exact labels.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
V411_MANIFEST = ROOT / "v411_external_inventory_lockfile" / "external_file_manifest.csv"
OUTDIR = ROOT / "v412_clean_canonical_external"

CANONICAL_COLUMNS = [
    "source_dataset",
    "source_file",
    "license_tag",
    "risk_tier",
    "sequence_id",
    "match_id",
    "rally_id",
    "event_index",
    "frame",
    "timestamp",
    "phase",
    "event_type",
    "coarse_family",
    "terminal_label",
    "remaining_hint",
    "landing_x",
    "landing_y",
    "landing_z",
    "landing_depth_bin",
    "landing_side_bin",
    "speed_x",
    "speed_y",
    "speed_z",
    "speed_norm",
    "spin_x",
    "spin_y",
    "spin_z",
    "spin_norm",
    "player_x",
    "player_y",
    "opponent_x",
    "opponent_y",
    "raw_label",
    "raw_payload_hash",
]


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_safe_json(value).encode("utf-8", errors="ignore")).hexdigest()


def _empty_row(source_row: dict[str, Any], source_file: str, payload: Any) -> dict[str, Any]:
    row = {col: pd.NA for col in CANONICAL_COLUMNS}
    row.update(
        {
            "source_dataset": source_row.get("source_dataset"),
            "source_file": source_file,
            "license_tag": source_row.get("license_tag"),
            "risk_tier": source_row.get("risk_tier"),
            "raw_payload_hash": _payload_hash({"source_file": source_file, "payload": payload}),
        }
    )
    return row


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _norm(*values: Any) -> float | None:
    floats = [_to_float(value) for value in values]
    if any(value is None for value in floats):
        return None
    return math.sqrt(sum(float(value) ** 2 for value in floats))


def _first_present(data: dict[str, Any], names: Iterable[str]) -> Any:
    lowered = {str(key).lower(): key for key in data.keys()}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return data.get(key)
    return pd.NA


def _area_to_bins(area: Any) -> tuple[str | Any, str | Any]:
    try:
        value = int(float(area))
    except Exception:
        return pd.NA, pd.NA
    if value <= 0:
        return "terminal_or_unknown", "unknown"
    depth = "short" if value in {1, 2, 3} else "half" if value in {4, 5, 6} else "long"
    side = "left" if value in {1, 4, 7} else "middle" if value in {2, 5, 8} else "right"
    return depth, side


def _badminton_family(label: Any) -> str | Any:
    if pd.isna(label):
        return pd.NA
    text = str(label).strip()
    if not text:
        return pd.NA
    return "badminton_" + "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def _opentt_family(data: dict[str, Any]) -> str:
    raw = _first_present(data, ["safe_action_family", "technique", "event_type", "type"])
    if not pd.isna(raw):
        text = str(raw).strip().lower().replace(" ", "_")
        if text:
            return "table_tennis_" + text
    return "table_tennis_trajectory"


def _finalize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    frame = pd.DataFrame(rows)
    for col in CANONICAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[CANONICAL_COLUMNS]


def _convert_deepmind_json(source_row: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    phase = "serve" if "serve" in path.stem.lower() else "rally"
    source_file = str(source_row.get("relative_path") or path)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        row = _empty_row(source_row, source_file, {"idx": idx, **item})
        row.update(
            {
                "sequence_id": f"{path.stem}_{item.get('id', idx)}",
                "event_index": idx,
                "phase": phase,
                "event_type": "trajectory_state",
                "coarse_family": "table_tennis_physics",
                "landing_x": _first_present(item, ["pos_x", "x"]),
                "landing_y": _first_present(item, ["pos_y", "y"]),
                "landing_z": _first_present(item, ["pos_z", "z"]),
                "speed_x": _first_present(item, ["vel_x", "vx"]),
                "speed_y": _first_present(item, ["vel_y", "vy"]),
                "speed_z": _first_present(item, ["vel_z", "vz"]),
                "spin_x": _first_present(item, ["w_vel_x", "spin_x"]),
                "spin_y": _first_present(item, ["w_vel_y", "spin_y"]),
                "spin_z": _first_present(item, ["w_vel_z", "spin_z"]),
            }
        )
        row["speed_norm"] = _norm(row["speed_x"], row["speed_y"], row["speed_z"])
        row["spin_norm"] = _norm(row["spin_x"], row["spin_y"], row["spin_z"])
        rows.append(row)
    return rows


def _iter_json_records(data: Any) -> Iterable[tuple[str, int, dict[str, Any]]]:
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                yield "json_list", idx, item
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                yield str(key), 0, value
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        yield str(key), idx, item


def _convert_opentt_json(source_row: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    source_file = str(source_row.get("relative_path") or path)
    rows: list[dict[str, Any]] = []
    for sequence_key, idx, item in _iter_json_records(data):
        row = _empty_row(source_row, source_file, {"sequence": sequence_key, "idx": idx, **item})
        row.update(
            {
                "sequence_id": f"{path.stem}_{sequence_key}",
                "event_index": idx,
                "phase": "rally",
                "event_type": _first_present(item, ["event_type", "type"]),
                "coarse_family": _opentt_family(item),
                "terminal_label": _first_present(item, ["safe_terminal_label", "is_rally_ending", "rally_ending_type"]),
                "landing_x": _first_present(item, ["x", "pos_x", "landing_x"]),
                "landing_y": _first_present(item, ["y", "pos_y", "landing_y"]),
                "landing_z": _first_present(item, ["z", "pos_z", "landing_z"]),
                "raw_label": _first_present(item, ["label", "technique", "event_type", "type"]),
            }
        )
        rows.append(row)
    return rows


def _convert_tt3d_csv(source_row: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    required = {"X", "Y", "Z"}
    if not required.issubset(set(df.columns)):
        return []
    source_file = str(source_row.get("relative_path") or path)
    rows: list[dict[str, Any]] = []
    for idx, item in df.iterrows():
        row = _empty_row(source_row, source_file, item.to_dict())
        row.update(
            {
                "sequence_id": Path(source_file).with_suffix("").as_posix(),
                "event_index": idx,
                "timestamp": item.get("Timestamp", pd.NA),
                "phase": "trajectory",
                "event_type": "3d_ball",
                "coarse_family": "table_tennis_trajectory",
                "landing_x": item.get("X", pd.NA),
                "landing_y": item.get("Y", pd.NA),
                "landing_z": item.get("Z", pd.NA),
            }
        )
        rows.append(row)
    return rows


def _convert_coachai_csv(source_row: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if "test_gt" in str(path).lower():
        return []
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return []
    columns = set(header.columns)
    if not ({"rally", "ball_round"} & columns) or not ({"type", "landing_area", "landing_x", "landing_y"} & columns):
        return []
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    source_file = str(source_row.get("relative_path") or path)
    rows: list[dict[str, Any]] = []
    for idx, item in df.iterrows():
        data = item.to_dict()
        rally_id = _first_present(data, ["rally", "rally_id"])
        ball_round = _first_present(data, ["ball_round", "round", "stroke_id"])
        landing_area = _first_present(data, ["landing_area", "landing_area_id"])
        depth, side = _area_to_bins(landing_area)
        row = _empty_row(source_row, source_file, data)
        row.update(
            {
                "sequence_id": f"{Path(source_file).stem}_{rally_id}",
                "match_id": _first_present(data, ["match_id", "match"]),
                "rally_id": rally_id,
                "event_index": ball_round if not pd.isna(ball_round) else idx,
                "timestamp": _first_present(data, ["time", "timestamp"]),
                "phase": "serve" if _to_float(ball_round) == 1 else "rally",
                "event_type": "badminton_stroke",
                "coarse_family": _badminton_family(_first_present(data, ["type", "shot_type"])),
                "terminal_label": _first_present(data, ["lose_reason", "win_reason", "getpoint_player"]),
                "landing_x": _first_present(data, ["landing_x", "landing_x_court"]),
                "landing_y": _first_present(data, ["landing_y", "landing_y_court"]),
                "landing_depth_bin": depth,
                "landing_side_bin": side,
                "player_x": _first_present(data, ["player_location_x", "player_x", "hit_x"]),
                "player_y": _first_present(data, ["player_location_y", "player_y", "hit_y"]),
                "opponent_x": _first_present(data, ["opponent_location_x", "opponent_x"]),
                "opponent_y": _first_present(data, ["opponent_location_y", "opponent_y"]),
                "raw_label": _first_present(data, ["type", "shot_type"]),
            }
        )
        rows.append(row)
    return rows


def _convert_spindoe_zip(source_row: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        archive = zipfile.ZipFile(path)
    except Exception:
        return rows
    source_file = str(source_row.get("relative_path") or path)
    for name in archive.namelist():
        if not name.lower().endswith(".csv"):
            continue
        try:
            payload = archive.read(name)
            df = pd.read_csv(io.BytesIO(payload), sep=";", header=None)
        except Exception:
            continue
        if df.shape[1] < 4:
            continue
        for idx, item in df.iterrows():
            row = _empty_row(source_row, f"{source_file}!{name}", item.to_dict())
            row.update(
                {
                    "sequence_id": f"spindoe_{Path(name).stem}",
                    "event_index": idx,
                    "timestamp": item.iloc[0],
                    "phase": "trajectory",
                    "event_type": "spin_trajectory",
                    "coarse_family": "table_tennis_spin_trajectory",
                    "landing_x": item.iloc[1],
                    "landing_y": item.iloc[2],
                    "landing_z": item.iloc[3],
                }
            )
            rows.append(row)
    return rows


def _convert_aimy_hdf5(source_row: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    try:
        import h5py  # type: ignore
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    source_file = str(source_row.get("relative_path") or path)
    try:
        handle = h5py.File(path, "r")
    except Exception:
        return rows
    with handle:
        datasets: list[tuple[str, Any]] = []

        def visitor(name: str, obj: Any) -> None:
            if hasattr(obj, "shape") and len(getattr(obj, "shape", ())) >= 2:
                datasets.append((name, obj))

        handle.visititems(visitor)
        for dataset_name, dataset in datasets[:3]:
            try:
                values = dataset[:]
            except Exception:
                continue
            max_rows = min(len(values), 1000)
            for idx in range(max_rows):
                vector = values[idx]
                if len(vector) < 3:
                    continue
                row = _empty_row(source_row, f"{source_file}::{dataset_name}", {"idx": idx})
                row.update(
                    {
                        "sequence_id": f"aimy_{path.stem}_{dataset_name}",
                        "event_index": idx,
                        "phase": "trajectory",
                        "event_type": "aimy_hdf5_trajectory",
                        "coarse_family": "table_tennis_physics",
                        "landing_x": vector[0],
                        "landing_y": vector[1],
                        "landing_z": vector[2],
                    }
                )
                rows.append(row)
    return rows


def _convert_one(source_row: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    source = str(source_row.get("source_dataset", ""))
    path = Path(str(source_row.get("path", "")))
    extension = str(source_row.get("extension", path.suffix)).lower()
    if not path.exists():
        return [], "missing_path"
    if source == "DeepMindrobottabletennis" and extension == ".json":
        return _convert_deepmind_json(source_row, path), None
    if source == "openttgames" and extension == ".json":
        return _convert_opentt_json(source_row, path), None
    if source == "TT3D" and extension == ".csv":
        return _convert_tt3d_csv(source_row, path), None
    if source == "CoachAI-Projects-main" and extension == ".csv":
        return _convert_coachai_csv(source_row, path), None
    if source.lower() == "spindoe" and extension == ".zip":
        return _convert_spindoe_zip(source_row, path), None
    if source == "AIMY" and extension in {".hdf5", ".h5"}:
        converted = _convert_aimy_hdf5(source_row, path)
        return converted, None if converted else "aimy_hdf5_reader_unavailable_or_no_numeric_dataset"
    return [], "unsupported_file_type_or_schema"


def build_canonical(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    blocked = 0
    for _, manifest_row in manifest.iterrows():
        source_row = manifest_row.to_dict()
        if not bool(source_row.get("allowed_first_version", False)):
            blocked += 1
            continue
        converted, reason = _convert_one(source_row)
        if converted:
            rows.extend(converted)
        else:
            skipped.append(
                {
                    "source_dataset": source_row.get("source_dataset"),
                    "source_file": source_row.get("relative_path") or source_row.get("path"),
                    "reason": reason or "no_convertible_rows",
                }
            )
    canonical = _finalize(rows)
    report = {
        "version": "V412",
        "canonical_rows": int(len(canonical)),
        "blocked_manifest_rows": int(blocked),
        "skipped_files": skipped[:500],
        "skipped_file_count": int(len(skipped)),
        "source_counts": canonical.groupby("source_dataset").size().to_dict() if not canonical.empty else {},
        "columns": CANONICAL_COLUMNS,
    }
    return canonical, report


def run_pipeline(
    *,
    manifest_path: Path = V411_MANIFEST,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    canonical, report = build_canonical(manifest)
    canonical_path = outdir / "canonical_external_events.csv"
    summary_path = outdir / "canonical_source_summary.csv"
    report_path = outdir / "canonical_schema_report.json"
    canonical.to_csv(canonical_path, index=False)
    if canonical.empty:
        summary = pd.DataFrame(columns=["source_dataset", "license_tag", "risk_tier", "rows"])
    else:
        summary = (
            canonical.groupby(["source_dataset", "license_tag", "risk_tier"], dropna=False)
            .size()
            .reset_index(name="rows")
        )
    summary.to_csv(summary_path, index=False)
    report["outputs"] = {
        "canonical_external_events": str(canonical_path),
        "canonical_source_summary": str(summary_path),
        "canonical_schema_report": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
