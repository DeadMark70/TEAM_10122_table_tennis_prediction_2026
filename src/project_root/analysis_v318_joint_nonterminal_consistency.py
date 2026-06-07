"""V318 joint action/point nonterminal consistency research.

This branch only writes local artifacts under v318_joint_nonterminal_consistency.
It searches paired nonterminal action/point edits where source probabilities and
train support both prefer the paired edit over action-only or point-only moves.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from analysis_v261_action_conditioned_point_residual import (
    EXPECTED_COLUMNS,
    action_family,
    build_frames,
    distribution,
    normalize_rows_safe,
    point_depth,
)
from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column
from analysis_v306_point0_addition_probe import V300_SUBMISSION, load_artifacts, load_submission


OUTDIR = Path("v318_joint_nonterminal_consistency")
V306_SUBMISSION = Path("v306_point0_addition_probe/submission_v306_p0_cap0p01__v173action_v300server.csv")
V286_OOF = Path("v286_weak_action_specialist_pretraining/v286_specialist_oof.csv")
ACTION_OOF_PROB = Path("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_oof_action_prob.npy")
ACTION_TEST_PROB = Path("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_test_action_prob.npy")

NONTERMINAL_POINTS = np.arange(1, 10, dtype=int)
NONTERMINAL_ACTIONS = np.arange(1, 19, dtype=int)


@dataclass(frozen=True)
class CandidateSpec:
    candidate: str
    submission: str
    test_budget: int
    min_score_gain: float
    min_pair_support: int
    single_gain_margin: float


def validate_submission_schema(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"submission columns {list(frame.columns)} != {EXPECTED_COLUMNS}")
    return frame.loc[:, EXPECTED_COLUMNS].copy()


def _phase_from_prefix(prefix_len: pd.Series) -> pd.Series:
    values = pd.to_numeric(prefix_len, errors="coerce").fillna(0).astype(int)
    return pd.Series(
        np.select([values <= 1, values.eq(2), values.le(4)], [0, 1, 2], default=3),
        index=prefix_len.index,
    ).astype(int)


def _coerce_phase_id(frame: pd.DataFrame) -> pd.Series:
    if "phase_id" in frame:
        return pd.to_numeric(frame["phase_id"], errors="coerce").fillna(0).astype(int)
    if "phase_bin" in frame:
        codes, _ = pd.factorize(frame["phase_bin"].astype(str), sort=True)
        return pd.Series(codes, index=frame.index).astype(int)
    if "prefix_len" in frame:
        return _phase_from_prefix(frame["prefix_len"])
    return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)


def _coerce_lag0_depth(frame: pd.DataFrame) -> pd.Series:
    if "lag0_depth" in frame:
        return pd.to_numeric(frame["lag0_depth"], errors="coerce").fillna(0).astype(int)
    if "lag0_point_depth" in frame:
        return pd.to_numeric(frame["lag0_point_depth"], errors="coerce").fillna(0).astype(int)
    if "lag0_pointId" not in frame:
        raise KeyError("frame must include lag0_pointId or lag0_depth")
    return frame["lag0_pointId"].astype(int).map(point_depth).astype(int)


def _coerce_lag0_action_family(frame: pd.DataFrame) -> pd.Series:
    if "lag0_action_family" in frame:
        return pd.to_numeric(frame["lag0_action_family"], errors="coerce").fillna(0).astype(int)
    if "lag0_actionId" not in frame:
        raise KeyError("frame must include lag0_actionId or lag0_action_family")
    return frame["lag0_actionId"].astype(int).map(action_family).astype(int)


def add_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["phase_id"] = _coerce_phase_id(out)
    out["lag0_depth"] = _coerce_lag0_depth(out)
    out["lag0_action_family"] = _coerce_lag0_action_family(out)
    return out


def _conditional_table(
    frame: pd.DataFrame,
    group_cols: list[str],
    label_col: str,
    label_name: str,
) -> pd.DataFrame:
    counts = frame.groupby(group_cols + [label_col], dropna=False).size().reset_index(name="support")
    totals = counts.groupby(group_cols, dropna=False)["support"].transform("sum")
    counts["total_support"] = totals.astype(int)
    counts["prob"] = counts["support"].astype(float) / totals.clip(lower=1).astype(float)
    return counts.rename(columns={label_col: label_name})


def build_support_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"next_actionId", "next_pointId"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"support frame missing columns: {missing}")
    ctx = add_context_columns(frame)
    nonterminal = ctx[ctx["next_actionId"].astype(int).ne(0) & ctx["next_pointId"].astype(int).ne(0)].copy()
    nonterminal["next_actionId"] = nonterminal["next_actionId"].astype(int)
    nonterminal["next_pointId"] = nonterminal["next_pointId"].astype(int)
    point_given_action = _conditional_table(
        nonterminal,
        ["next_actionId", "phase_id", "lag0_depth"],
        "next_pointId",
        "pointId",
    ).rename(columns={"next_actionId": "actionId"})
    action_given_point = _conditional_table(
        nonterminal,
        ["next_pointId", "phase_id", "lag0_action_family"],
        "next_actionId",
        "actionId",
    ).rename(columns={"next_pointId": "pointId"})
    return {
        "point_given_action": point_given_action[
            ["actionId", "phase_id", "lag0_depth", "pointId", "support", "total_support", "prob"]
        ].reset_index(drop=True),
        "action_given_point": action_given_point[
            ["pointId", "phase_id", "lag0_action_family", "actionId", "support", "total_support", "prob"]
        ].reset_index(drop=True),
    }


def _lookup_support(table: pd.DataFrame, filters: dict[str, int]) -> tuple[float, int, int]:
    if table.empty:
        return 0.0, 0, 0
    key_cols = tuple(filters.keys())
    cache_name = "__v318_lookup_" + "|".join(key_cols)
    lookup = table.attrs.get(cache_name)
    if lookup is None:
        lookup = {
            tuple(int(getattr(row, col)) for col in key_cols): (
                float(row.prob),
                int(row.support),
                int(row.total_support),
            )
            for row in table.itertuples(index=False)
        }
        table.attrs[cache_name] = lookup
    key = tuple(int(value) for value in filters.values())
    if key not in lookup:
        return 0.0, 0, 0
    return lookup[key]


def compatibility_score(
    tables: dict[str, pd.DataFrame],
    *,
    action: int,
    point: int,
    phase: int,
    lag0_depth: int,
    lag0_action_family: int,
) -> dict[str, float | int]:
    point_prob, point_support, point_total = _lookup_support(
        tables["point_given_action"],
        {
            "actionId": int(action),
            "phase_id": int(phase),
            "lag0_depth": int(lag0_depth),
            "pointId": int(point),
        },
    )
    action_prob, action_support, action_total = _lookup_support(
        tables["action_given_point"],
        {
            "pointId": int(point),
            "phase_id": int(phase),
            "lag0_action_family": int(lag0_action_family),
            "actionId": int(action),
        },
    )
    return {
        "score": float(0.5 * (point_prob + action_prob)),
        "point_prob": float(point_prob),
        "action_prob": float(action_prob),
        "point_support": int(point_support),
        "action_support": int(action_support),
        "min_support": int(min(point_support, action_support)),
        "point_total_support": int(point_total),
        "action_total_support": int(action_total),
    }


def _best_nonterminal_target(prob: np.ndarray, base: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = normalize_rows_safe(prob)
    base = np.asarray(base, dtype=int)
    scoped = np.full_like(p, -np.inf, dtype=float)
    valid = allowed[(allowed >= 0) & (allowed < p.shape[1])]
    scoped[:, valid] = p[:, valid]
    for row_id, cls in enumerate(base):
        if 0 <= int(cls) < scoped.shape[1]:
            scoped[row_id, int(cls)] = -np.inf
    target = scoped.argmax(axis=1).astype(int)
    target_prob = p[np.arange(len(p)), np.clip(target, 0, p.shape[1] - 1)]
    base_prob = p[np.arange(len(p)), np.clip(base, 0, p.shape[1] - 1)]
    margin = target_prob - base_prob
    return target, target_prob, margin


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_id",
            "old_pointId",
            "new_pointId",
            "old_actionId",
            "new_actionId",
            "point_margin",
            "action_margin",
            "base_compat",
            "pair_compat",
            "point_only_compat",
            "action_only_compat",
            "compat_gain",
            "point_only_gain",
            "action_only_gain",
            "pair_support",
            "point_support",
            "action_support",
        ]
    )


def select_joint_nonterminal_candidates(
    frame: pd.DataFrame,
    base_point: np.ndarray,
    base_action: np.ndarray,
    point_prob: np.ndarray,
    action_prob: np.ndarray,
    tables: dict[str, pd.DataFrame],
    *,
    budget: int,
    min_score_gain: float,
    min_pair_support: int,
    single_gain_margin: float = 0.0,
    min_action_margin: float = -0.35,
) -> pd.DataFrame:
    ctx = add_context_columns(frame)
    base_point = np.asarray(base_point, dtype=int)
    base_action = np.asarray(base_action, dtype=int)
    if not (len(ctx) == len(base_point) == len(base_action) == len(point_prob) == len(action_prob)):
        raise ValueError("frame, base labels, and probabilities must have matching row counts")
    if budget <= 0:
        return _empty_candidates()

    point_target, _point_target_prob, point_margin = _best_nonterminal_target(
        point_prob, base_point, NONTERMINAL_POINTS
    )
    p_action = normalize_rows_safe(action_prob)
    eligible = (
        (base_point != 0)
        & (base_action != 0)
        & (point_target != 0)
        & (point_target != base_point)
        & np.isfinite(point_margin)
        & (point_margin > 0.0)
    )
    rows: list[dict[str, Any]] = []
    for row_id in np.where(eligible)[0]:
        phase = int(ctx.iloc[row_id]["phase_id"])
        lag0_depth = int(ctx.iloc[row_id]["lag0_depth"])
        lag0_family = int(ctx.iloc[row_id]["lag0_action_family"])
        old_point = int(base_point[row_id])
        old_action = int(base_action[row_id])
        new_point = int(point_target[row_id])

        action_options: list[tuple[float, int, float, int, dict[str, float | int]]] = []
        for candidate_action in NONTERMINAL_ACTIONS.tolist():
            if int(candidate_action) == old_action:
                continue
            score_for_action = compatibility_score(
                tables,
                action=int(candidate_action),
                point=new_point,
                phase=phase,
                lag0_depth=lag0_depth,
                lag0_action_family=lag0_family,
            )
            if int(score_for_action["min_support"]) <= 0:
                continue
            candidate_margin = float(
                p_action[row_id, int(candidate_action)]
                - p_action[row_id, np.clip(old_action, 0, p_action.shape[1] - 1)]
            )
            action_options.append(
                (
                    float(score_for_action["score"]),
                    int(score_for_action["min_support"]),
                    candidate_margin,
                    int(candidate_action),
                    score_for_action,
                )
            )
        if not action_options:
            continue
        action_options.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _score, _support, action_margin, new_action, pair = action_options[0]
        if float(action_margin) < float(min_action_margin):
            continue

        base = compatibility_score(
            tables,
            action=old_action,
            point=old_point,
            phase=phase,
            lag0_depth=lag0_depth,
            lag0_action_family=lag0_family,
        )
        point_only = compatibility_score(
            tables,
            action=old_action,
            point=new_point,
            phase=phase,
            lag0_depth=lag0_depth,
            lag0_action_family=lag0_family,
        )
        action_only = compatibility_score(
            tables,
            action=new_action,
            point=old_point,
            phase=phase,
            lag0_depth=lag0_depth,
            lag0_action_family=lag0_family,
        )
        compat_gain = float(pair["score"]) - float(base["score"])
        point_only_gain = float(point_only["score"]) - float(base["score"])
        action_only_gain = float(action_only["score"]) - float(base["score"])
        if compat_gain < float(min_score_gain):
            continue
        if int(pair["min_support"]) < int(min_pair_support):
            continue
        if compat_gain <= max(point_only_gain, action_only_gain) + float(single_gain_margin):
            continue
        rows.append(
            {
                "row_id": int(row_id),
                "old_pointId": old_point,
                "new_pointId": new_point,
                "old_actionId": old_action,
                "new_actionId": new_action,
                "point_margin": float(point_margin[row_id]),
                "action_margin": float(action_margin),
                "base_compat": float(base["score"]),
                "pair_compat": float(pair["score"]),
                "point_only_compat": float(point_only["score"]),
                "action_only_compat": float(action_only["score"]),
                "compat_gain": compat_gain,
                "point_only_gain": point_only_gain,
                "action_only_gain": action_only_gain,
                "pair_support": int(pair["min_support"]),
                "point_support": int(pair["point_support"]),
                "action_support": int(pair["action_support"]),
            }
        )
    if not rows:
        return _empty_candidates()
    selected = pd.DataFrame(rows)
    selected = selected.sort_values(
        ["compat_gain", "pair_support", "point_margin", "action_margin", "row_id"],
        ascending=[False, False, False, False, True],
    )
    return selected.head(int(budget)).reset_index(drop=True)


def _apply_joint_changes(
    base_point: np.ndarray,
    base_action: np.ndarray,
    selected: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(base_point, dtype=int).copy()
    action = np.asarray(base_action, dtype=int).copy()
    for row in selected.itertuples(index=False):
        point[int(row.row_id)] = int(row.new_pointId)
        action[int(row.row_id)] = int(row.new_actionId)
    return point, action


def _macro(y: np.ndarray, pred: np.ndarray, labels: list[int] | np.ndarray) -> float:
    return float(f1_score(np.asarray(y, dtype=int), np.asarray(pred, dtype=int), labels=list(labels), average="macro", zero_division=0))


def _joint_metric(y_point: np.ndarray, point: np.ndarray, y_action: np.ndarray, action: np.ndarray) -> float:
    return 0.5 * (_macro(y_point, point, POINT_CLASSES) + _macro(y_action, action, ACTION_CLASSES))


def _point0_stats(base: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    base = np.asarray(base, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return int(np.sum((base != 0) & (pred == 0))), int(np.sum((base == 0) & (pred != 0)))


def _distribution_json(values: np.ndarray) -> str:
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(unique, counts)}, sort_keys=True)


def _candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "v318_joint_nonterminal_budget12",
            "submission_v318_joint_nonterminal_budget12__v300server.csv",
            12,
            0.040,
            15,
            0.000,
        ),
        CandidateSpec(
            "v318_joint_nonterminal_budget24",
            "submission_v318_joint_nonterminal_budget24__v300server.csv",
            24,
            0.030,
            10,
            0.000,
        ),
        CandidateSpec(
            "v318_joint_actionpoint_agree_safe",
            "submission_v318_joint_actionpoint_agree_safe__v300server.csv",
            12,
            0.060,
            25,
            0.010,
        ),
    ]


def _load_action_prob(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing action probability artifact: {path}")
    prob = normalize_rows_safe(np.load(path))
    if prob.shape != (expected_rows, len(ACTION_CLASSES)):
        raise ValueError(f"{path} shape {prob.shape} != {(expected_rows, len(ACTION_CLASSES))}")
    return prob


def _load_v286_oof(expected_rows: int) -> pd.DataFrame:
    if not V286_OOF.exists():
        raise FileNotFoundError(f"missing V286 action OOF anchor: {V286_OOF}")
    oof = pd.read_csv(V286_OOF)
    if len(oof) != expected_rows:
        raise ValueError(f"{V286_OOF} rows {len(oof)} != {expected_rows}")
    for col in ["y_true_action", "anchor_action"]:
        if col not in oof:
            raise KeyError(f"{V286_OOF} missing {col}")
    return oof


def _load_aligned_frames(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df, _ = build_frames()
    train_df = align_train_to_literal_meta(train_df, meta)
    return add_context_columns(train_df).reset_index(drop=True), add_context_columns(test_df).reset_index(drop=True)


def _write_submission(anchor: pd.DataFrame, point: np.ndarray, action: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out["actionId"] = np.asarray(action, dtype=int)
    out = validate_submission_schema(out)
    if len(out) != 1845:
        raise ValueError(f"{name} has {len(out)} rows, expected 1845")
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def _decision_label(
    joint_delta: float,
    point_only_delta: float,
    action_only_delta: float,
    changed_rows: int,
    mean_support: float,
) -> str:
    if (
        int(changed_rows) > 0
        and float(joint_delta) >= max(float(point_only_delta), float(action_only_delta), 0.0) + 0.0005
        and float(mean_support) >= 15.0
    ):
        return "REVIEW"
    return "DO_NOT_UPLOAD"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report_md(report: dict[str, Any]) -> None:
    rows = report.get("top_candidates", [])
    table = "\n".join(
        "- `{candidate}` joint_delta `{joint_oof_delta:.6f}` point_only `{point_only_oof_delta:.6f}` "
        "action_only `{action_only_oof_delta:.6f}` rows `{test_changed_rows}` support `{mean_pair_support:.2f}` decision `{decision}`".format(
            **row
        )
        for row in rows
    )
    (OUTDIR / "v318_report.md").write_text(
        "# V318 Joint Nonterminal Consistency\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n"
        f"- Best candidate: `{report['best_candidate'].get('candidate', 'none')}`\n"
        f"- Copied to upload/selected: `False`\n\n"
        "## Top Candidates\n\n"
        + (table if table else "- None")
        + "\n",
        encoding="utf-8",
    )


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    meta = pd.DataFrame(artifacts["meta"]).reset_index(drop=True)
    train_df, test_df = _load_aligned_frames(meta)
    support_tables = build_support_tables(train_df)
    support_tables["point_given_action"].to_csv(OUTDIR / "v318_support_point_given_action.csv", index=False)
    support_tables["action_given_point"].to_csv(OUTDIR / "v318_support_action_given_point.csv", index=False)

    point_oof_prob = normalize_rows_safe(np.asarray(artifacts["v188_oof_prob"]))
    point_test_prob = normalize_rows_safe(np.asarray(artifacts["v188_test_prob"]))
    cap5_oof = pd.DataFrame(artifacts["cap5_oof"])
    cap5_test = pd.DataFrame(artifacts["cap5_test"])
    oof_base_point = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    cap5_test_point = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    y_point = meta["next_pointId"].astype(int).to_numpy()
    if len(oof_base_point) != len(y_point):
        raise ValueError(f"OOF point rows {len(oof_base_point)} != labels {len(y_point)}")

    v286 = _load_v286_oof(len(y_point))
    y_action = v286["y_true_action"].astype(int).to_numpy()
    oof_base_action = v286["anchor_action"].astype(int).to_numpy()
    action_oof_prob = _load_action_prob(ACTION_OOF_PROB, len(y_point))
    action_test_prob = _load_action_prob(ACTION_TEST_PROB, len(test_df))

    anchor = load_submission(V306_SUBMISSION if V306_SUBMISSION.exists() else V300_SUBMISSION)
    test_base_point = anchor["pointId"].astype(int).to_numpy()
    test_base_action = anchor["actionId"].astype(int).to_numpy()
    if len(test_base_point) != len(test_df):
        raise ValueError(f"anchor rows {len(anchor)} != test rows {len(test_df)}")
    current_v300 = load_submission(V300_SUBMISSION)

    base_point_score = _macro(y_point, oof_base_point, POINT_CLASSES)
    base_action_score = _macro(y_action, oof_base_action, ACTION_CLASSES)
    base_joint = _joint_metric(y_point, oof_base_point, y_action, oof_base_action)

    records: list[dict[str, Any]] = []
    changed_frames: list[pd.DataFrame] = []
    submissions: list[dict[str, str]] = []
    for spec in _candidate_specs():
        oof_budget = int(math.floor(len(oof_base_point) * (float(spec.test_budget) / max(len(test_base_point), 1))))
        oof_selected = select_joint_nonterminal_candidates(
            train_df,
            oof_base_point,
            oof_base_action,
            point_oof_prob,
            action_oof_prob,
            support_tables,
            budget=oof_budget,
            min_score_gain=spec.min_score_gain,
            min_pair_support=spec.min_pair_support,
            single_gain_margin=spec.single_gain_margin,
        )
        test_selected = select_joint_nonterminal_candidates(
            test_df,
            test_base_point,
            test_base_action,
            point_test_prob,
            action_test_prob,
            support_tables,
            budget=spec.test_budget,
            min_score_gain=spec.min_score_gain,
            min_pair_support=spec.min_pair_support,
            single_gain_margin=spec.single_gain_margin,
        )

        pred_point_oof, pred_action_oof = _apply_joint_changes(oof_base_point, oof_base_action, oof_selected)
        pred_point_test, pred_action_test = _apply_joint_changes(test_base_point, test_base_action, test_selected)
        point_only_oof, _ = _apply_joint_changes(oof_base_point, oof_base_action, oof_selected.assign(new_actionId=oof_selected["old_actionId"]))
        _, action_only_oof = _apply_joint_changes(oof_base_point, oof_base_action, oof_selected.assign(new_pointId=oof_selected["old_pointId"]))

        point_score = _macro(y_point, pred_point_oof, POINT_CLASSES)
        action_score = _macro(y_action, pred_action_oof, ACTION_CLASSES)
        joint_score = _joint_metric(y_point, pred_point_oof, y_action, pred_action_oof)
        point_only_delta = _joint_metric(y_point, point_only_oof, y_action, oof_base_action) - base_joint
        action_only_delta = _joint_metric(y_point, oof_base_point, y_action, action_only_oof) - base_joint
        joint_delta = joint_score - base_joint
        p0_add, p0_remove = _point0_stats(test_base_point, pred_point_test)
        mean_support = float(test_selected["pair_support"].mean()) if len(test_selected) else 0.0
        decision = _decision_label(joint_delta, point_only_delta, action_only_delta, len(test_selected), mean_support)
        path = _write_submission(anchor, pred_point_test, pred_action_test, spec.submission)
        submissions.append({"candidate": spec.candidate, "submission": spec.submission, "path": path})

        if len(test_selected):
            changed = test_selected.copy()
            changed["candidate"] = spec.candidate
            changed["rally_uid"] = anchor.loc[changed["row_id"].astype(int), "rally_uid"].astype(int).to_numpy()
            changed["serverGetPoint"] = anchor.loc[changed["row_id"].astype(int), "serverGetPoint"].to_numpy()
            changed_frames.append(changed)

        records.append(
            {
                "candidate": spec.candidate,
                "submission": spec.submission,
                "test_budget": int(spec.test_budget),
                "oof_budget": int(oof_budget),
                "min_score_gain": float(spec.min_score_gain),
                "min_pair_support": int(spec.min_pair_support),
                "single_gain_margin": float(spec.single_gain_margin),
                "point_macro_f1": float(point_score),
                "action_macro_f1": float(action_score),
                "joint_oof_metric": float(joint_score),
                "point_oof_delta": float(point_score - base_point_score),
                "action_oof_delta": float(action_score - base_action_score),
                "joint_oof_delta": float(joint_delta),
                "point_only_oof_delta": float(point_only_delta),
                "action_only_oof_delta": float(action_only_delta),
                "test_changed_rows": int(len(test_selected)),
                "oof_changed_rows": int(len(oof_selected)),
                "point0_additions": int(p0_add),
                "point0_removals": int(p0_remove),
                "mean_compat_gain": float(test_selected["compat_gain"].mean()) if len(test_selected) else 0.0,
                "min_compat_gain": float(test_selected["compat_gain"].min()) if len(test_selected) else 0.0,
                "mean_pair_support": float(mean_support),
                "min_pair_support_changed": int(test_selected["pair_support"].min()) if len(test_selected) else 0,
                "mean_point_margin": float(test_selected["point_margin"].mean()) if len(test_selected) else 0.0,
                "mean_action_margin": float(test_selected["action_margin"].mean()) if len(test_selected) else 0.0,
                "test_churn_vs_v306_anchor": float(np.mean((pred_point_test != test_base_point) | (pred_action_test != test_base_action))),
                "test_point_churn_vs_v305_cap5": float(np.mean(pred_point_test != cap5_test_point)),
                "test_point_churn_vs_v300": float(np.mean(pred_point_test != current_v300["pointId"].astype(int).to_numpy())),
                "test_changed_point_distribution": _distribution_json(pred_point_test[pred_point_test != test_base_point])
                if len(test_selected)
                else "{}",
                "test_changed_action_distribution": _distribution_json(pred_action_test[pred_action_test != test_base_action])
                if len(test_selected)
                else "{}",
                "test_point_distribution": json.dumps(distribution(pred_point_test), sort_keys=True),
                "test_action_distribution": _distribution_json(pred_action_test),
                "decision": decision,
                "path": path,
            }
        )

    search = pd.DataFrame(records).sort_values(
        ["decision", "joint_oof_delta", "mean_compat_gain", "test_changed_rows"],
        ascending=[True, False, False, True],
    )
    search.to_csv(OUTDIR / "v318_joint_search.csv", index=False)
    changed_rows = pd.concat(changed_frames, ignore_index=True) if changed_frames else _empty_candidates()
    changed_rows.to_csv(OUTDIR / "v318_local_candidate_rows.csv", index=False)

    best = search.sort_values(["joint_oof_delta", "mean_compat_gain"], ascending=[False, False]).head(1)
    best_dict = best.iloc[0].to_dict() if len(best) else {}
    review = search[search["decision"].eq("REVIEW")]
    report = _json_safe(
        {
            "version": "V318",
            "verdict": "HAS_REVIEW_CANDIDATE" if len(review) else "NO_UPLOAD_WORTHY_CANDIDATE",
            "upload_recommendation": "REVIEW" if len(review) else "DO_NOT_UPLOAD",
            "copied_to_upload_or_selected": False,
            "no_ttmatch_or_old_server": True,
            "anchor_submission": str(V306_SUBMISSION if V306_SUBMISSION.exists() else V300_SUBMISSION),
            "point_probability_source": {
                "oof": "v305_literal_v188_point_artifact/v305_v188_r186_w005_oof_proba.npy",
                "test": "v305_literal_v188_point_artifact/v305_v188_r186_w005_test_proba.npy",
            },
            "action_probability_source": {"oof": str(ACTION_OOF_PROB), "test": str(ACTION_TEST_PROB)},
            "support_tables": {
                "point_given_action": "v318_joint_nonterminal_consistency/v318_support_point_given_action.csv",
                "action_given_point": "v318_joint_nonterminal_consistency/v318_support_action_given_point.csv",
            },
            "base_point_macro_f1": float(base_point_score),
            "base_action_macro_f1": float(base_action_score),
            "base_joint_metric": float(base_joint),
            "best_candidate": best_dict,
            "top_candidates": search.head(5).to_dict(orient="records"),
            "submissions": submissions,
            "notes": [
                "Only paired action and point edits are applied.",
                "Rows with base or candidate point0/action0 are rejected.",
                "Each row must improve pair compatibility beyond action-only and point-only compatibility.",
                "All files are local to v318_joint_nonterminal_consistency.",
            ],
        }
    )
    (OUTDIR / "v318_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(report)
    return report


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "verdict": report.get("verdict"),
                "upload_recommendation": report.get("upload_recommendation"),
                "best_candidate": best.get("candidate", ""),
                "best_joint_oof_delta": best.get("joint_oof_delta", 0.0),
                "best_changed_rows": best.get("test_changed_rows", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
