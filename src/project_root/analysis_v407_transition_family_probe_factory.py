"""V407 transition-family probe factory.

This script splits V400 public-agreement point rows into smaller transition
families on top of the V362 anchor. It preserves action/server exactly and
blocks point0 additions before packaging.
"""

from __future__ import annotations

import json
import math
from collections import Counter
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
OUTDIR = ROOT / "v407_transition_family_probe_factory"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V400_SELECTED_PATHS = {
    "top9": ROOT / "v400_public_component_recombination" / "selected_rows_v400_public_agree_top9.csv",
    "top15": ROOT / "v400_public_component_recombination" / "selected_rows_v400_public_agree_top15.csv",
    "top24": ROOT / "v400_public_component_recombination" / "selected_rows_v400_public_agree_top24.csv",
}

TRANSITION_FAMILIES: dict[str, set[tuple[int, int]]] = {
    "longside_centering": {(7, 8), (9, 8)},
    "longside_corner": {(8, 9), (8, 7), (7, 9), (9, 7)},
    "long_to_half": {(9, 6), (8, 6), (7, 4)},
    "short_to_middle": {(2, 5), (3, 6), (1, 4)},
}
EMIT_FAMILIES = (
    "longside_centering",
    "longside_corner",
    "long_to_half",
    "short_to_middle",
    "mixed_high_agreement",
)
REQUIRED_SELECTED_COLUMNS = ["rank", "row_id", "rally_uid", "anchor_point", "new_point"]


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


def load_anchor_submission(
    path: Path = ANCHOR_PATH,
    *,
    expected_rows: int | None = 1845,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame


def classify_transition(anchor_point: int, new_point: int) -> str | None:
    transition = (int(anchor_point), int(new_point))
    for family, transitions in TRANSITION_FAMILIES.items():
        if transition in transitions:
            return family
    return None


def transition_label(anchor_point: int, new_point: int) -> str:
    return f"{int(anchor_point)}->{int(new_point)}"


def _source_size(label: str) -> int:
    return int(str(label).replace("top", ""))


def load_v400_selected_rows(
    selected_paths: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_paths = selected_paths or V400_SELECTED_PATHS
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for label in ("top9", "top15", "top24"):
        path = selected_paths[label]
        if not path.exists():
            missing.append(label)
            continue
        frame = pd.read_csv(path)
        missing_cols = [col for col in REQUIRED_SELECTED_COLUMNS if col not in frame.columns]
        if missing_cols:
            raise ValueError(f"{path} missing selected row columns: {missing_cols}")
        frame = frame.copy()
        frame["source_tier"] = label
        frame["source_size"] = _source_size(label)
        frames.append(frame)

    if not frames:
        columns = REQUIRED_SELECTED_COLUMNS + [
            "source_tier",
            "source_size",
            "transition",
            "transition_group",
            "in_top9",
            "in_top15",
            "in_top24",
        ]
        return pd.DataFrame(columns=columns), {"missing_selected_inputs": missing, "loaded_selected_inputs": []}

    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("rank", "row_id", "anchor_point", "new_point"):
        all_rows[col] = all_rows[col].astype(int)
    all_rows["transition"] = [
        transition_label(old, new) for old, new in zip(all_rows["anchor_point"], all_rows["new_point"])
    ]
    all_rows["transition_group"] = [
        classify_transition(old, new) for old, new in zip(all_rows["anchor_point"], all_rows["new_point"])
    ]

    all_rows = all_rows.sort_values(["source_size", "rank", "row_id", "new_point"], kind="mergesort")
    dedup = all_rows.drop_duplicates(["row_id", "new_point"], keep="first").copy()
    for label in ("top9", "top15", "top24"):
        keys = set(
            zip(
                all_rows.loc[all_rows["source_tier"] == label, "row_id"].astype(int),
                all_rows.loc[all_rows["source_tier"] == label, "new_point"].astype(int),
            )
        )
        dedup[f"in_{label}"] = [(int(row_id), int(new_point)) in keys for row_id, new_point in zip(dedup["row_id"], dedup["new_point"])]

    dedup = dedup.sort_values(["rank", "source_size", "row_id", "new_point"], kind="mergesort").reset_index(drop=True)
    return dedup, {
        "missing_selected_inputs": missing,
        "loaded_selected_inputs": [label for label in ("top9", "top15", "top24") if label not in missing],
        "raw_selected_row_count": int(len(all_rows)),
        "dedup_selected_row_count": int(len(dedup)),
    }


def block_point0_additions(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    mask = ~((rows["anchor_point"].astype(int) != 0) & (rows["new_point"].astype(int) == 0))
    return rows.loc[mask].copy()


def build_transition_family_rows(selected_rows: pd.DataFrame, *, mixed_limit: int = 9) -> dict[str, pd.DataFrame]:
    safe_rows = block_point0_additions(selected_rows)
    if "source_size" not in safe_rows.columns:
        safe_rows = safe_rows.copy()
        safe_rows["source_size"] = 24
    families: dict[str, pd.DataFrame] = {}
    for family in TRANSITION_FAMILIES:
        group = safe_rows.loc[safe_rows["transition_group"] == family].copy()
        group["transition_group"] = family
        families[family] = group.sort_values(["rank", "row_id", "new_point"], kind="mergesort").reset_index(drop=True)

    mixed = safe_rows.loc[safe_rows["transition_group"].isin(TRANSITION_FAMILIES)].copy()
    if not mixed.empty:
        mixed = mixed.sort_values(
            ["in_top9", "in_top15", "rank", "source_size", "row_id", "new_point"],
            ascending=[False, False, True, True, True, True],
            kind="mergesort",
        ).head(mixed_limit)
        mixed["transition_group"] = "mixed_high_agreement"
    families["mixed_high_agreement"] = mixed.reset_index(drop=True)
    return families


def package_candidate(anchor: pd.DataFrame, selected_rows: pd.DataFrame) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    for row in selected_rows.itertuples(index=False):
        row_id = int(row.row_id)
        if row_id < 0 or row_id >= len(out):
            raise IndexError(f"row_id outside anchor: {row_id}")
        if int(row.anchor_point) != int(anchor.at[row_id, "pointId"]):
            raise ValueError(f"anchor point mismatch at row_id={row_id}")
        if int(row.anchor_point) != 0 and int(row.new_point) == 0:
            raise AssertionError("point0 addition escaped family filtering")
        out.at[row_id, "pointId"] = int(row.new_point)

    if not out["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
        raise AssertionError("action changed")
    if not np.array_equal(
        pd.to_numeric(out["serverGetPoint"]).to_numpy(dtype=float),
        pd.to_numeric(anchor["serverGetPoint"]).to_numpy(dtype=float),
    ):
        raise AssertionError("server changed")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def transition_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts = Counter(str(value) for value in rows.get("transition", pd.Series(dtype=str)).tolist())
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def candidate_filename(family: str) -> str:
    return f"submission_v407_{family}__v173action_v300server.csv"


def selected_filename(family: str) -> str:
    return f"selected_rows_v407_{family}.csv"


def _risk(selected_count: int) -> str:
    return "safe" if selected_count <= 9 else "medium"


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
    anchor_path: Path = ANCHOR_PATH,
    selected_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    anchor = load_anchor_submission(anchor_path, expected_rows=expected_rows)
    selected_rows, input_report = load_v400_selected_rows(selected_paths)
    selected_rows = block_point0_additions(selected_rows)
    family_rows = build_transition_family_rows(selected_rows)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for family in EMIT_FAMILIES:
        selected = family_rows.get(family, pd.DataFrame()).copy()
        if selected.empty:
            skipped[family] = "zero_selected_rows"
            continue

        submission = package_candidate(anchor, selected)
        dist = point_distribution_report(anchor["pointId"], submission["pointId"])
        if int(dist["point0_additions"]) != 0:
            raise AssertionError("point0 addition escaped candidate packaging")

        submission_path = safe_output_path(outdir, candidate_filename(family))
        selected_path = safe_output_path(outdir, selected_filename(family))
        selected.to_csv(selected_path, index=False)
        submission.to_csv(submission_path, index=False)

        action_churn = int((submission["actionId"].astype(int) != anchor["actionId"].astype(int)).sum())
        server_changed = int(
            np.sum(
                pd.to_numeric(submission["serverGetPoint"]).to_numpy(dtype=float)
                != pd.to_numeric(anchor["serverGetPoint"]).to_numpy(dtype=float)
            )
        )
        row = {
            "candidate": f"v407_{family}",
            "path": str(submission_path),
            "selected_rows": str(selected_path),
            "selected_row_count": int(len(selected)),
            "action_churn": action_churn,
            "point_churn": int(dist["changed_rows"]),
            "point0_additions": int(dist["point0_additions"]),
            "server_changed": server_changed,
            "risk": _risk(len(selected)),
            "evidence": "v400_public_agreement_transition_family",
            "transition_group": family,
            "transition_counts": transition_counts(selected),
            "source_top9_rows": int(selected["in_top9"].sum()) if "in_top9" in selected else 0,
            "source_top15_rows": int(selected["in_top15"].sum()) if "in_top15" in selected else 0,
            "source_top24_rows": int(selected["in_top24"].sum()) if "in_top24" in selected else 0,
        }
        summary_rows.append(row)
        generated.append(row.copy())

    ranked = pd.DataFrame(summary_rows)
    if not ranked.empty:
        risk_order = {"safe": 0, "medium": 1, "blocked": 9}
        ranked["_risk_order"] = ranked["risk"].map(risk_order).fillna(8).astype(int)
        ranked = ranked.sort_values(
            ["_risk_order", "source_top9_rows", "selected_row_count", "transition_group"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).drop(columns=["_risk_order"]).reset_index(drop=True)
    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    ranked.to_csv(ranked_path, index=False)

    report = {
        "version": "V407",
        "anchor": relative_path(anchor_path),
        "anchor_rows": int(len(anchor)),
        "inputs": input_report,
        "candidate_family_order": list(EMIT_FAMILIES),
        "transition_families": {
            family: [transition_label(old, new) for old, new in sorted(transitions)]
            for family, transitions in TRANSITION_FAMILIES.items()
        },
        "selected_row_count_after_point0_gate": int(len(selected_rows)),
        "generated_submission_count": int(len(generated)),
        "generated_submissions": generated,
        "skipped_families": skipped,
        "ranked_candidates": str(ranked_path),
        "policy": {
            "anchor": "V362",
            "source": "V400 selected rows top9/top15/top24",
            "point_only": True,
            "no_point0_additions": True,
            "action_preserved": True,
            "server_preserved": True,
            "emit_nonempty_only": True,
            "mixed_high_agreement_max_rows": 9,
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
