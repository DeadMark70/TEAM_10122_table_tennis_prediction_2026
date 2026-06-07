"""V405 V362 pruning lab.

This experiment compares the public-best V362 point submission against the
closest available predecessor, classifies V362's point changes, and emits small
point-only pruning candidates while preserving V362 action/server columns.
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
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v405_v362_pruning_lab"
V362_RELATIVE = Path(
    "v362_point_hierarchical_specialists/submission_v362_depth_agree_only__v173action_v300server.csv"
)
V362_SELECTED_RELATIVE = Path("v362_point_hierarchical_specialists/selected_v362_depth_agree_only.csv")
PREDECESSOR_PATTERNS = (
    ("v338_public_positive_style", ("v338_joint_moe_pack/submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv",)),
    ("v338_public_positive_style", ("v338_joint_moe_pack/submission_v338*.csv",)),
    ("v341_no_p0_point_pack", ("v341_no_p0_point_pack/submission_v341*.csv",)),
    (
        "v300_best_safe_repack",
        ("v300_clean_server_blend_recycler/submission_v300_best_safe_repack__v173action_v261point_server.csv",),
    ),
    ("v261_action_conditioned_point_residual", ("v261_action_conditioned_point_residual/submission_v261*.csv",)),
    ("v306_point0_addition_probe", ("v306_point0_addition_probe/submission_v306*.csv",)),
)
PUBLIC_SELECTED_DIRS = (
    "v400_public_component_recombination",
    "v401_action_point_compatibility",
    "v402_rare_point_specialist_lab",
    "v403_neural_posterior_gate",
)


def relative_path(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
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


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame


def align_to_anchor(frame: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    if frame["rally_uid"].reset_index(drop=True).equals(anchor["rally_uid"].reset_index(drop=True)):
        return frame.reset_index(drop=True).loc[:, SUBMISSION_COLUMNS].copy()
    if set(frame["rally_uid"]) != set(anchor["rally_uid"]):
        raise ValueError("rally_uid set differs from anchor")
    return frame.set_index("rally_uid").loc[anchor["rally_uid"]].reset_index().loc[:, SUBMISSION_COLUMNS].copy()


def _first_matching_path(root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        for path in matches:
            if path.is_file():
                return path
    return None


def _normalize_selected_rows(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if "row_id" not in frame.columns:
        return pd.DataFrame(columns=["row_id", "old_point", "new_point", "source"])

    old_col = next((col for col in ("base_point", "anchor_point", "old_point", "from_point") if col in frame.columns), None)
    new_col = next(
        (col for col in ("candidate_point", "new_point", "point_pred", "to_point") if col in frame.columns),
        None,
    )
    if old_col is None or new_col is None:
        return pd.DataFrame(columns=["row_id", "old_point", "new_point", "source"])

    out = pd.DataFrame(
        {
            "row_id": pd.to_numeric(frame["row_id"], errors="coerce"),
            "old_point": pd.to_numeric(frame[old_col], errors="coerce"),
            "new_point": pd.to_numeric(frame[new_col], errors="coerce"),
            "source": source,
        }
    )
    out = out.dropna(subset=["row_id", "old_point", "new_point"]).copy()
    out[["row_id", "old_point", "new_point"]] = out[["row_id", "old_point", "new_point"]].astype(int)
    return out


def derive_predecessor_from_v362_selected(
    *,
    root: Path,
    v362: pd.DataFrame,
    expected_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_path = root / V362_SELECTED_RELATIVE
    predecessor = v362.copy()
    if not selected_path.exists():
        return predecessor, {"kind": "v362_identity_fallback", "path": None, "selected_rows": 0}

    selected = _normalize_selected_rows(pd.read_csv(selected_path), source=relative_path(selected_path, root))
    selected = selected[selected["row_id"].between(0, len(predecessor) - 1)].copy()
    for row in selected.itertuples(index=False):
        predecessor.at[int(row.row_id), "pointId"] = int(row.old_point)
    validate_submission_schema(predecessor, expected_rows=expected_rows)
    return predecessor, {
        "kind": "pseudo_from_v362_selected_rows",
        "path": relative_path(selected_path, root),
        "selected_rows": int(len(selected)),
    }


def choose_predecessor(
    *,
    root: Path,
    v362: pd.DataFrame,
    expected_rows: int | None = 1845,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for kind, patterns in PREDECESSOR_PATTERNS:
        path = _first_matching_path(root, patterns)
        if path is None:
            continue
        try:
            frame = align_to_anchor(load_submission(path, expected_rows=expected_rows), v362)
        except (ValueError, KeyError, pd.errors.ParserError):
            continue
        return frame, {"kind": kind, "path": relative_path(path, root)}
    return derive_predecessor_from_v362_selected(root=root, v362=v362, expected_rows=expected_rows)


def _change_key(row_id: int, old_point: int, new_point: int) -> tuple[int, int, int]:
    return int(row_id), int(old_point), int(new_point)


def load_public_agreement_keys(root: Path) -> tuple[set[tuple[int, int, int]], list[dict[str, Any]]]:
    keys: set[tuple[int, int, int]] = set()
    sources: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for dirname in PUBLIC_SELECTED_DIRS:
        directory = root / dirname
        if not directory.exists():
            continue
        for path in sorted(directory.glob("selected_rows*.csv")) + sorted(directory.glob("selected_*.csv")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                selected = _normalize_selected_rows(pd.read_csv(path), source=relative_path(path, root))
            except (OSError, pd.errors.ParserError):
                continue
            for row in selected.itertuples(index=False):
                keys.add(_change_key(row.row_id, row.old_point, row.new_point))
            sources.append({"path": relative_path(path, root), "rows": int(len(selected))})
    return keys, sources


def _load_v362_support(root: Path) -> dict[int, dict[str, Any]]:
    path = root / V362_SELECTED_RELATIVE
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "row_id" not in frame.columns:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        row_id = int(row["row_id"])
        support = 0.0
        for col in ("depth_support", "train_backoff_support_score"):
            if col in frame.columns and pd.notna(row.get(col)):
                support = max(support, float(row[col]))
        depth_agree = bool(row.get("depth_agree", False)) if "depth_agree" in frame.columns else False
        out[row_id] = {"train_support": support, "depth_agree": depth_agree}
    return out


def classify_v362_changes(*, root: Path, v362: pd.DataFrame, predecessor: pd.DataFrame) -> pd.DataFrame:
    public_keys, _sources = load_public_agreement_keys(root)
    support_by_row = _load_v362_support(root)
    rows: list[dict[str, Any]] = []
    pred_points = predecessor["pointId"].astype(int).to_numpy()
    v362_points = v362["pointId"].astype(int).to_numpy()

    for row_id, (old_point, new_point) in enumerate(zip(pred_points, v362_points)):
        if int(old_point) == int(new_point):
            continue
        old_i = int(old_point)
        new_i = int(new_point)
        support = support_by_row.get(row_id, {})
        train_support = float(support.get("train_support", 0.0) or 0.0)
        depth_agree = bool(support.get("depth_agree", False))
        public_agreement = _change_key(row_id, old_i, new_i) in public_keys
        has_train_support = train_support > 0 or depth_agree
        point0_related = old_i == 0 or new_i == 0
        rows.append(
            {
                "row_id": row_id,
                "rally_uid": v362.at[row_id, "rally_uid"],
                "old_point": old_i,
                "new_point": new_i,
                "transition": f"{old_i}->{new_i}",
                "long_side": old_i in {7, 8, 9} and new_i in {7, 8, 9},
                "half_boundary": old_i in {4, 5, 6} or new_i in {4, 5, 6},
                "short_control": old_i in {1, 2, 3} or new_i in {1, 2, 3},
                "point0_related": point0_related,
                "public_agreement": public_agreement,
                "train_backoff_support": train_support,
                "depth_agree": depth_agree,
                "low_support": bool((not public_agreement) and (not has_train_support)),
            }
        )

    columns = [
        "row_id",
        "rally_uid",
        "old_point",
        "new_point",
        "transition",
        "long_side",
        "half_boundary",
        "short_control",
        "point0_related",
        "public_agreement",
        "train_backoff_support",
        "depth_agree",
        "low_support",
    ]
    return pd.DataFrame(rows, columns=columns)


def package_candidate(
    *,
    candidate: str,
    v362: pd.DataFrame,
    predecessor: pd.DataFrame,
    changes: pd.DataFrame,
    mode: str,
    keep_mask: pd.Series | None = None,
    remove_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = v362.loc[:, SUBMISSION_COLUMNS].copy()
    selected = changes.iloc[0:0].copy()
    if changes.empty:
        validate_submission_schema(out, expected_rows=len(v362))
        return out, selected

    if mode == "keep_mask":
        if keep_mask is None:
            raise ValueError("keep_mask is required")
        selected = changes.loc[pd.Series(keep_mask, index=changes.index).astype(bool)].copy()
        out["pointId"] = predecessor["pointId"].astype(int).to_numpy()
        for row in selected.itertuples(index=False):
            out.at[int(row.row_id), "pointId"] = int(row.new_point)
    elif mode == "remove_mask":
        if remove_mask is None:
            raise ValueError("remove_mask is required")
        selected = changes.loc[pd.Series(remove_mask, index=changes.index).astype(bool)].copy()
        for row in selected.itertuples(index=False):
            out.at[int(row.row_id), "pointId"] = int(row.old_point)
    else:
        raise ValueError(f"unknown mode: {mode}")

    point0_addition_mask = v362["pointId"].astype(int).ne(0) & out["pointId"].astype(int).eq(0)
    if point0_addition_mask.any():
        blocked_ids = set(np.flatnonzero(point0_addition_mask.to_numpy()))
        out.loc[point0_addition_mask, "pointId"] = v362.loc[point0_addition_mask, "pointId"].astype(int)
        selected = selected[~selected["row_id"].astype(int).isin(blocked_ids)].copy()

    if not out["actionId"].astype(int).equals(v362["actionId"].astype(int)):
        raise AssertionError(f"{candidate} changed actionId")
    if not np.array_equal(
        pd.to_numeric(out["serverGetPoint"]).to_numpy(dtype=float),
        pd.to_numeric(v362["serverGetPoint"]).to_numpy(dtype=float),
    ):
        raise AssertionError(f"{candidate} changed serverGetPoint")
    validate_submission_schema(out, expected_rows=len(v362))
    return out, selected


def transition_counts(base_point: pd.Series, cand_point: pd.Series) -> dict[str, int]:
    counts = Counter()
    for old, new in zip(base_point.astype(int), cand_point.astype(int)):
        if int(old) != int(new):
            counts[f"{int(old)}->{int(new)}"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _candidate_specs(changes: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    if changes.empty:
        return []
    specs = [
        ("v405_v362_keep_public_agreement", "keep_mask", changes["public_agreement"].astype(bool)),
        ("v405_v362_remove_low_support", "remove_mask", changes["low_support"].astype(bool)),
        ("v405_v362_longside_only", "keep_mask", changes["long_side"].astype(bool)),
        ("v405_v362_half_boundary_only", "keep_mask", changes["half_boundary"].astype(bool)),
    ]
    return [(name, mode, mask) for name, mode, mask in specs if bool(mask.any())]


def _submission_filename(candidate: str) -> str:
    return f"submission_{candidate}__v173action_v300server.csv"


def _selected_filename(candidate: str) -> str:
    return f"selected_rows_{candidate}.csv"


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    v362_path = root / V362_RELATIVE
    v362 = load_submission(v362_path, expected_rows=expected_rows).reset_index(drop=True)
    predecessor, predecessor_meta = choose_predecessor(root=root, v362=v362, expected_rows=expected_rows)
    predecessor = align_to_anchor(predecessor, v362)
    changes = classify_v362_changes(root=root, v362=v362, predecessor=predecessor)
    changes_path = safe_output_path(outdir, "v362_change_classification.csv")
    changes.to_csv(changes_path, index=False)

    _public_keys, public_sources = load_public_agreement_keys(root)
    generated: list[dict[str, Any]] = []
    for candidate, mode, mask in _candidate_specs(changes):
        kwargs: dict[str, Any] = {"keep_mask": mask} if mode == "keep_mask" else {"remove_mask": mask}
        submission, selected = package_candidate(
            candidate=candidate,
            v362=v362,
            predecessor=predecessor,
            changes=changes,
            mode=mode,
            **kwargs,
        )
        if selected.empty:
            continue

        submission_path = safe_output_path(outdir, _submission_filename(candidate))
        selected_path = safe_output_path(outdir, _selected_filename(candidate))
        submission.to_csv(submission_path, index=False)
        selected.to_csv(selected_path, index=False)

        point_dist = point_distribution_report(v362["pointId"], submission["pointId"])
        action_churn = int((submission["actionId"].astype(int) != v362["actionId"].astype(int)).sum())
        server_changed = int(
            np.sum(
                pd.to_numeric(submission["serverGetPoint"]).to_numpy(dtype=float)
                != pd.to_numeric(v362["serverGetPoint"]).to_numpy(dtype=float)
            )
        )
        row = {
            "candidate": candidate,
            "path": str(submission_path),
            "selected_rows": str(selected_path),
            "selected_row_count": int(len(selected)),
            "action_churn": action_churn,
            "point_churn": int(point_dist["changed_rows"]),
            "point0_additions": int(point_dist["point0_additions"]),
            "server_changed": server_changed,
            "risk": "safe" if len(selected) <= 15 and int(point_dist["point0_additions"]) == 0 else "medium",
            "evidence": "v362_pruning_public_agreement_train_backoff",
            "transition_counts": json.dumps(transition_counts(v362["pointId"], submission["pointId"]), sort_keys=True),
            "mode": mode,
        }
        generated.append(row)

    ranked = pd.DataFrame(
        generated,
        columns=[
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
            "transition_counts",
            "mode",
        ],
    )
    if not ranked.empty:
        ranked = ranked.sort_values(
            ["risk", "point0_additions", "action_churn", "server_changed", "selected_row_count", "candidate"],
            ascending=[True, True, True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    ranked.to_csv(ranked_path, index=False)

    classification_counts = {
        "long_side": int(changes["long_side"].sum()) if "long_side" in changes else 0,
        "half_boundary": int(changes["half_boundary"].sum()) if "half_boundary" in changes else 0,
        "short_control": int(changes["short_control"].sum()) if "short_control" in changes else 0,
        "point0_related": int(changes["point0_related"].sum()) if "point0_related" in changes else 0,
        "public_agreement": int(changes["public_agreement"].sum()) if "public_agreement" in changes else 0,
        "low_support": int(changes["low_support"].sum()) if "low_support" in changes else 0,
    }
    report = {
        "version": "V405",
        "anchor": relative_path(v362_path, root),
        "anchor_rows": int(len(v362)),
        "predecessor": predecessor_meta,
        "v362_change_count": int(len(changes)),
        "classification_counts": classification_counts,
        "public_selected_sources": public_sources,
        "generated_submission_count": int(len(generated)),
        "generated_submissions": generated,
        "ranked_candidates": str(ranked_path),
        "change_classification": str(changes_path),
        "policy": {
            "anchor": "V362",
            "action_preserved": True,
            "server_preserved": True,
            "point0_additions_blocked_vs_v362": True,
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
