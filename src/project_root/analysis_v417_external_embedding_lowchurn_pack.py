"""V417 low-churn packaging for V416 external-representation predictions.

This script repacks V416 predictions against the V362 anchor as small,
bounded probes. It does not train models and does not write upload candidates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v417_external_embedding_lowchurn_pack"
V416_DIR = ROOT / "v416_external_embedding_aicup_finetune"
ANCHOR_RELATIVE = Path(
    "v362_point_hierarchical_specialists/submission_v362_depth_agree_only__v173action_v300server.csv"
)

POINT_SIZES = (5, 10, 20)
ACTION_SIZES = (5, 10, 20)
JOINT_SIZE = 10


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
        return relative_path(value)
    return value


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str]:
    action_col = _first_existing_column(
        frame,
        (
            "pred_actionId",
            "action_pred",
            "pred_action",
            "candidate_actionId",
            "v416_actionId",
            "actionId_pred",
        ),
    )
    point_col = _first_existing_column(
        frame,
        (
            "pred_pointId",
            "point_pred",
            "pred_point",
            "candidate_pointId",
            "v416_pointId",
            "pointId_pred",
        ),
    )
    if action_col is None or point_col is None:
        raise ValueError("V416 test_predictions.csv must include predicted action and point columns")
    return action_col, point_col


def _prefix_probability_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        text = str(col).lower()
        if any(text.startswith(prefix) for prefix in prefixes):
            cols.append(col)
    return cols


def _probability_margin_from_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.Series | None:
    cols = _prefix_probability_columns(frame, prefixes)
    if len(cols) < 2:
        return None
    probs = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if probs.shape[1] < 2:
        return None
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return pd.Series(margin, index=frame.index).clip(lower=0.0)


def _normalized_low_entropy(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.Series | None:
    cols = _prefix_probability_columns(frame, prefixes)
    if len(cols) < 2:
        return None
    probs = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    totals = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, totals, out=np.zeros_like(probs), where=totals > 0)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    max_entropy = math.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    return pd.Series(1.0 - (entropy / max_entropy), index=frame.index).clip(lower=0.0, upper=1.0)


def _confidence(
    predictions: pd.DataFrame,
    *,
    kind: str,
    pred_values: pd.Series,
    anchor_values: pd.Series,
) -> pd.Series:
    direct_cols = {
        "action": ("action_margin", "action_probability_margin", "action_confidence", "action_prob_max"),
        "point": ("point_margin", "point_probability_margin", "point_confidence", "point_prob_max"),
        "joint": ("joint_margin", "joint_probability_margin", "joint_confidence"),
    }
    direct = _first_existing_column(predictions, direct_cols[kind])
    if direct is not None:
        return pd.to_numeric(predictions[direct], errors="coerce").fillna(0.0).astype(float)

    if kind == "action":
        margin = _probability_margin_from_columns(predictions, ("action_proba_", "action_prob_"))
        entropy = _normalized_low_entropy(predictions, ("action_proba_", "action_prob_"))
    elif kind == "point":
        margin = _probability_margin_from_columns(predictions, ("point_proba_", "point_prob_"))
        entropy = _normalized_low_entropy(predictions, ("point_proba_", "point_prob_"))
    else:
        margin = None
        entropy = None

    if margin is not None:
        return margin.astype(float)
    if entropy is not None:
        return entropy.astype(float)

    changed = pred_values.astype(int).ne(anchor_values.astype(int)).astype(float)
    stable = pd.util.hash_pandas_object(predictions["rally_uid"].astype(str), index=False).astype(float)
    stable = (stable % 1000) / 1_000_000.0
    return (0.25 + (0.50 * changed) + stable).astype(float)


def align_predictions_to_anchor(predictions: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in predictions.columns:
        if len(predictions) != len(anchor):
            raise ValueError("V416 predictions without rally_uid must match anchor row count")
        out = predictions.copy()
        out.insert(0, "rally_uid", anchor["rally_uid"].to_numpy())
        return out.reset_index(drop=True)

    pred_uid = predictions["rally_uid"].reset_index(drop=True).astype(str)
    anchor_uid = anchor["rally_uid"].reset_index(drop=True).astype(str)
    if len(predictions) == len(anchor) and pred_uid.equals(anchor_uid):
        return predictions.reset_index(drop=True).copy()

    reduced = predictions.copy()
    if reduced["rally_uid"].duplicated().any():
        if "strikeNumber" in reduced.columns:
            reduced = reduced.sort_values(["rally_uid", "strikeNumber"])
        reduced = reduced.groupby("rally_uid", sort=False).tail(1)

    if set(reduced["rally_uid"].astype(str)) != set(anchor["rally_uid"].astype(str)):
        missing = sorted(set(anchor["rally_uid"].astype(str)) - set(reduced["rally_uid"].astype(str)))
        raise ValueError(f"V416 predictions cannot align to anchor; missing rally_uid values: {missing[:5]}")

    return (
        reduced.assign(rally_uid=reduced["rally_uid"].astype(str))
        .set_index("rally_uid")
        .loc[anchor["rally_uid"].astype(str)]
        .reset_index()
    )


def build_ranked_changes(anchor: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    action_col, point_col = _prediction_columns(predictions)
    rows = pd.DataFrame(
        {
            "row_id": np.arange(len(anchor), dtype=int),
            "rally_uid": anchor["rally_uid"].to_numpy(),
            "anchor_action": anchor["actionId"].astype(int).to_numpy(),
            "pred_action": pd.to_numeric(predictions[action_col], errors="coerce").fillna(anchor["actionId"]).astype(int),
            "anchor_point": anchor["pointId"].astype(int).to_numpy(),
            "pred_point": pd.to_numeric(predictions[point_col], errors="coerce").fillna(anchor["pointId"]).astype(int),
        }
    )
    rows["action_changed"] = rows["pred_action"].ne(rows["anchor_action"])
    rows["point_changed"] = rows["pred_point"].ne(rows["anchor_point"])
    rows["action_eligible"] = rows["action_changed"] & ~(
        rows["pred_action"].isin(SERVE_ACTION_CLASSES) & ~rows["anchor_action"].isin(SERVE_ACTION_CLASSES)
    )
    rows["point_eligible"] = rows["point_changed"] & ~(
        rows["pred_point"].eq(0) & rows["anchor_point"].ne(0)
    )
    rows["action_confidence"] = _confidence(
        predictions,
        kind="action",
        pred_values=rows["pred_action"],
        anchor_values=rows["anchor_action"],
    ).to_numpy()
    rows["point_confidence"] = _confidence(
        predictions,
        kind="point",
        pred_values=rows["pred_point"],
        anchor_values=rows["anchor_point"],
    ).to_numpy()
    joint_direct = _first_existing_column(predictions, ("joint_margin", "joint_probability_margin", "joint_confidence"))
    if joint_direct is not None:
        rows["joint_confidence"] = pd.to_numeric(predictions[joint_direct], errors="coerce").fillna(0.0).to_numpy()
    else:
        rows["joint_confidence"] = (rows["action_confidence"] + rows["point_confidence"]) / 2.0
    return rows


def _select_rows(changes: pd.DataFrame, *, mode: str, limit: int) -> pd.DataFrame:
    if mode == "point":
        mask = changes["point_eligible"]
        confidence = "point_confidence"
    elif mode == "action":
        mask = changes["action_eligible"]
        confidence = "action_confidence"
    elif mode == "joint":
        mask = changes["point_eligible"] | changes["action_eligible"]
        confidence = "joint_confidence"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return (
        changes.loc[mask]
        .sort_values([confidence, "row_id"], ascending=[False, True])
        .head(int(limit))
        .reset_index(drop=True)
    )


def _package(anchor: pd.DataFrame, selected: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    for row in selected.itertuples(index=False):
        row_id = int(row.row_id)
        if mode in {"action", "joint"} and bool(row.action_eligible):
            out.at[row_id, "actionId"] = int(row.pred_action)
        if mode in {"point", "joint"} and bool(row.point_eligible):
            out.at[row_id, "pointId"] = int(row.pred_point)
    return out.loc[:, SUBMISSION_COLUMNS].copy()


def _candidate_stats(
    *,
    candidate: str,
    path: Path,
    anchor: pd.DataFrame,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    root: Path,
) -> dict[str, Any]:
    action_changed = frame["actionId"].astype(int).ne(anchor["actionId"].astype(int))
    point_changed = frame["pointId"].astype(int).ne(anchor["pointId"].astype(int))
    server_changed = frame["serverGetPoint"].ne(anchor["serverGetPoint"])
    serve_additions = (
        frame["actionId"].astype(int).isin(SERVE_ACTION_CLASSES)
        & ~anchor["actionId"].astype(int).isin(SERVE_ACTION_CLASSES)
    )
    point0_additions = frame["pointId"].astype(int).eq(0) & anchor["pointId"].astype(int).ne(0)
    return {
        "candidate": candidate,
        "path": str(path.resolve()),
        "selected_rows": " ".join(str(int(v)) for v in selected["row_id"].tolist()),
        "selected_row_count": int(len(selected)),
        "action_churn": int(action_changed.sum()),
        "point_churn": int(point_changed.sum()),
        "server_changed": int(server_changed.sum()),
        "serve_15_18_additions": int(serve_additions.sum()),
        "point0_additions": int(point0_additions.sum()),
    }


def write_candidate(
    *,
    candidate: str,
    filename: str,
    mode: str,
    limit: int,
    anchor: pd.DataFrame,
    changes: pd.DataFrame,
    outdir: Path,
    root: Path,
    expected_rows: int | None,
) -> dict[str, Any]:
    selected = _select_rows(changes, mode=mode, limit=limit)
    frame = _package(anchor, selected, mode=mode)
    validate_submission_schema(frame, expected_rows=expected_rows)
    path = safe_output_path(outdir, filename)
    frame.to_csv(path, index=False)
    return _candidate_stats(candidate=candidate, path=path, anchor=anchor, frame=frame, selected=selected, root=root)


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path | None = None,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / "v417_external_embedding_lowchurn_pack"
    outdir.mkdir(parents=True, exist_ok=True)

    anchor_path = root / ANCHOR_RELATIVE
    v416_dir = root / "v416_external_embedding_aicup_finetune"
    test_predictions_path = v416_dir / "test_predictions.csv"
    oof_predictions_path = v416_dir / "oof_predictions.csv"
    metrics_path = v416_dir / "local_metrics.json"

    anchor = load_submission(anchor_path, expected_rows=expected_rows)
    raw_predictions = pd.read_csv(test_predictions_path)
    predictions = align_predictions_to_anchor(raw_predictions, anchor)
    changes = build_ranked_changes(anchor, predictions)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    oof_rows = int(len(pd.read_csv(oof_predictions_path))) if oof_predictions_path.exists() else 0

    generated: list[dict[str, Any]] = []
    for size in POINT_SIZES:
        generated.append(
            write_candidate(
                candidate=f"point_top{size}",
                filename=f"submission_v417_point_top{size}__externalrepr_v362anchor.csv",
                mode="point",
                limit=size,
                anchor=anchor,
                changes=changes,
                outdir=outdir,
                root=root,
                expected_rows=expected_rows,
            )
        )
    for size in ACTION_SIZES:
        generated.append(
            write_candidate(
                candidate=f"action_top{size}",
                filename=f"submission_v417_action_top{size}__externalrepr_v362anchor.csv",
                mode="action",
                limit=size,
                anchor=anchor,
                changes=changes,
                outdir=outdir,
                root=root,
                expected_rows=expected_rows,
            )
        )
    generated.append(
        write_candidate(
            candidate="joint_top10",
            filename="submission_v417_joint_top10__externalrepr_v362anchor.csv",
            mode="joint",
            limit=JOINT_SIZE,
            anchor=anchor,
            changes=changes,
            outdir=outdir,
            root=root,
            expected_rows=expected_rows,
        )
    )

    summary = pd.DataFrame(generated)
    summary_path = safe_output_path(outdir, "candidate_summary.csv")
    summary.to_csv(summary_path, index=False)

    report = {
        "version": "v417_external_embedding_lowchurn_pack",
        "anchor_path": relative_path(anchor_path, root),
        "v416_test_predictions_path": relative_path(test_predictions_path, root),
        "v416_oof_predictions_path": relative_path(oof_predictions_path, root),
        "v416_local_metrics_path": relative_path(metrics_path, root),
        "anchor_rows": int(len(anchor)),
        "raw_prediction_rows": int(len(raw_predictions)),
        "aligned_prediction_rows": int(len(predictions)),
        "oof_prediction_rows": oof_rows,
        "local_metrics": json_safe(metrics),
        "eligible_action_changes": int(changes["action_eligible"].sum()),
        "eligible_point_changes": int(changes["point_eligible"].sum()),
        "blocked_serve_15_18_additions": int(
            (
                changes["action_changed"]
                & changes["pred_action"].isin(SERVE_ACTION_CLASSES)
                & ~changes["anchor_action"].isin(SERVE_ACTION_CLASSES)
            ).sum()
        ),
        "blocked_point0_additions": int(
            (changes["point_changed"] & changes["pred_point"].eq(0) & changes["anchor_point"].ne(0)).sum()
        ),
        "generated_submission_count": int(len(generated)),
        "generated_submissions": json_safe(generated),
    }
    write_json(safe_output_path(outdir, "packaging_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
