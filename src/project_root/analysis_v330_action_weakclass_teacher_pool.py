"""V330 weak-class teacher pool over the strict V173 action anchor.

This research line keeps the fixed V306 point + V300 server package and only
considers action edits over a rebuilt V173 OOF/test action anchor.  If the V173
anchor cannot be rebuilt and aligned, the script raises before exporting any
submission CSVs.
"""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_lgbm import ACTION_CLASSES  # noqa: E402


OUTDIR = ROOT / "v330_action_weakclass_teacher_pool"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V209_OOF = ROOT / "v209_action_selector_reranker" / "v209_v208_action_point_aux_oof.npy"
V209_TEST = ROOT / "v209_action_selector_reranker" / "v209_v208_action_point_aux_test.npy"
V286_OOF = ROOT / "v286_weak_action_specialist_pretraining" / "v286_specialist_oof.csv"
V286_SUB_DIR = ROOT / "v286_weak_action_specialist_pretraining"
V291_SUB_DIR = ROOT / "v291_weak_class_training_upgrade"
R197_SUB_DIR = ROOT / "v197_action_teacher_surgery"

N_ACTIONS = 19
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
WEAK_ACTIONS = np.array([0, 3, 4, 5, 7, 8, 9, 12, 14], dtype=int)
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
PROTECTED_ANCHOR_ACTIONS = np.array([1, 10, 13, 15, 16, 17, 18], dtype=int)

MIN_ACTION_OOF_DELTA = 0.0015
MIN_CHANGED_ROW_PRECISION = 0.45
MIN_CHANGED_ACTION_ROWS = 5
MAX_CHANGED_ACTION_ROWS = 80
MAX_SERVE_ACTION_ROWS = 0

SPECIALIST_GROUPS: "OrderedDict[str, tuple[int, ...]]" = OrderedDict(
    [
        ("all_weak_03457891214", tuple(int(a) for a in WEAK_ACTIONS)),
        ("terminal_03", (0, 3)),
        ("short_control_4", (4,)),
        ("fast_attack_57", (5, 7)),
        ("style_control_89", (8, 9)),
        ("defensive_1214", (12, 14)),
    ]
)
TEST_BUDGETS = (5, 10, 20, 40, 80)


@dataclass(frozen=True)
class ExportSpec:
    filename: str


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def require_rebuilt_v173_anchor(meta: dict[str, Any]) -> None:
    if meta.get("anchor_oof_source") != "rebuilt_v173_pred_oof":
        raise RuntimeError("strict V173 action anchor required: rebuilt V173 OOF source missing")
    if "rebuild_v173_best_actions" not in str(meta.get("row_source", "")):
        raise RuntimeError("strict V173 action anchor required: rows are not from the R184 V173 rebuild")


def protected_output_path(outdir: Path, spec: ExportSpec) -> Path:
    root = Path(outdir)
    path = root / spec.filename
    parts = {part.lower() for part in path.parts}
    if any("upload_candidates" in part for part in parts) or "selected" in parts or "submissions" in parts:
        raise ValueError(f"refusing non-local V330 export path: {path}")
    if path.parent != root:
        raise ValueError(f"refusing non-local V330 export path: {path}")
    return path


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("matrix must be a non-empty 2D array")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    row_sum = arr.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0.0
    if bad.any():
        arr[bad] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def macro_f1(y: np.ndarray, pred: np.ndarray, labels: Iterable[int] = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def class_f1_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for action in ACTION_CLASSES:
        rows.append(
            {
                "action": int(action),
                "is_focus_weak": int(int(action) in set(WEAK_ACTIONS.tolist())),
                "support": int(np.sum(np.asarray(y, dtype=int) == int(action))),
                "anchor_f1": macro_f1(y, anchor, [int(action)]),
                "v330_best_f1": macro_f1(y, pred, [int(action)]),
            }
        )
    out = pd.DataFrame(rows)
    out["delta"] = out["v330_best_f1"] - out["anchor_f1"]
    return out.sort_values(["is_focus_weak", "delta"], ascending=[False, False]).reset_index(drop=True)


def action_distribution(values: np.ndarray) -> str:
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(unique, counts)}, sort_keys=True)


def changed_row_precision(y_true: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(anchor, dtype=int)
    pred = np.asarray(candidate, dtype=int)
    if not (len(y) == len(base) == len(pred)):
        raise ValueError("y_true, anchor, and candidate must have matching lengths")
    changed = pred != base
    rows = int(changed.sum())
    correct = int(np.sum(changed & (pred == y)))
    return {
        "changed_rows": rows,
        "changed_correct": correct,
        "changed_precision": float(correct / rows) if rows else 0.0,
    }


def evidence_passes(row: dict[str, Any] | pd.Series) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    source_family = str(data.get("source_family", "")).lower()
    if "fallback" in source_family or "fallback" in str(data.get("anchor_oof_source", "")).lower():
        return False
    changed = int(data.get("changed_action_rows", 0))
    return bool(
        float(data.get("action_oof_delta", 0.0)) >= MIN_ACTION_OOF_DELTA
        and float(data.get("changed_row_oof_precision", 0.0)) >= MIN_CHANGED_ROW_PRECISION
        and MIN_CHANGED_ACTION_ROWS <= changed <= MAX_CHANGED_ACTION_ROWS
        and int(data.get("serve_action_rows", 0)) <= MAX_SERVE_ACTION_ROWS
    )


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={SUBMISSION_COLUMNS}")
    if len(df) != int(expected_rows):
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
    if not df["actionId"].astype(int).between(0, N_ACTIONS - 1).all():
        raise ValueError("actionId out of range")
    if not df["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
    server = pd.to_numeric(df["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    if not server.between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")


def build_export_frame(anchor_sub: pd.DataFrame, action: np.ndarray) -> pd.DataFrame:
    pred = np.asarray(action, dtype=int)
    if len(anchor_sub) != len(pred):
        raise ValueError(f"action rows {len(pred)} != anchor submission rows {len(anchor_sub)}")
    out = pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": pred,
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )
    validate_submission_frame(out, expected_rows=len(anchor_sub))
    return out


def load_strict_v173_anchor() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing fixed V306/V300 package: {ANCHOR_SUBMISSION}")
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    validate_submission_frame(anchor_sub, expected_rows=len(anchor_sub))

    try:
        from analysis_v290_shortcontrol411_specialist import _set_pickle_dataclasses
        from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

        _set_pickle_dataclasses()
        state = rebuild_v173_best_actions()
    except Exception as exc:
        raise RuntimeError(f"strict V173 action anchor required: rebuild_v173_best_actions failed: {exc}") from exc

    required = {"rows", "test_rows", "v173_pred_oof", "v173_pred_test"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"strict V173 action anchor required: rebuild state missing {missing}")

    rows = state["rows"].reset_index(drop=True).copy()
    test_rows = state["test_rows"].reset_index(drop=True).copy()
    y = rows["next_actionId"].astype(int).to_numpy()
    anchor_oof = np.asarray(state["v173_pred_oof"], dtype=int)
    rebuilt_test = np.asarray(state["v173_pred_test"], dtype=int)
    if len(rows) != len(anchor_oof):
        raise RuntimeError("strict V173 action anchor required: OOF length mismatch")
    if len(test_rows) != len(rebuilt_test):
        raise RuntimeError("strict V173 action anchor required: test length mismatch")
    if len(anchor_sub) != len(test_rows):
        raise RuntimeError(f"strict V173 action anchor required: package rows {len(anchor_sub)} != test rows {len(test_rows)}")
    if not anchor_sub["rally_uid"].astype(int).reset_index(drop=True).equals(test_rows["rally_uid"].astype(int).reset_index(drop=True)):
        raise RuntimeError("strict V173 action anchor required: package rally_uid does not align to rebuilt test rows")
    package_action = anchor_sub["actionId"].astype(int).to_numpy()
    mismatch = int(np.sum(package_action != rebuilt_test))
    if mismatch:
        raise RuntimeError(f"strict V173 action anchor required: fixed package action differs from rebuilt V173 on {mismatch} rows")
    if "fold" not in rows:
        rows["fold"] = np.arange(len(rows), dtype=int) % 5
    test_rows["anchor_action"] = package_action
    meta = {
        "row_source": "analysis_r184_receiver_affordance_refiner.rebuild_v173_best_actions",
        "anchor_oof_source": "rebuilt_v173_pred_oof",
        "anchor_test_source": "fixed V306 package actionId verified equal to rebuilt_v173_pred_test",
        "v173_best_candidate": state.get("best_candidate", ""),
        "v173_schedule": state.get("schedule", ""),
        "v173_alpha": state.get("alpha", None),
        "anchor_rows": int(len(anchor_sub)),
    }
    require_rebuilt_v173_anchor(meta)
    return rows, test_rows, y, anchor_oof, anchor_sub, meta


def _action_family(action: int) -> str:
    a = int(action)
    if a == 0:
        return "Zero"
    if 1 <= a <= 7:
        return "Attack"
    if 8 <= a <= 11:
        return "Control"
    if 12 <= a <= 14:
        return "Defensive"
    if 15 <= a <= 18:
        return "Serve"
    return "Zero"


def _point_depth(point: int) -> str:
    p = int(point)
    if p <= 0:
        return "zero"
    if p in {1, 4, 7}:
        return "short"
    if p in {2, 5, 8}:
        return "half"
    return "long"


def _phase(prefix_len: int) -> str:
    p = int(prefix_len)
    if p <= 1:
        return "receive"
    if p == 2:
        return "third"
    if p == 3:
        return "fourth"
    return "rally"


def context_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["phase"] = rows["r184_phase"].astype(str) if "r184_phase" in rows else rows["prefix_len"].map(_phase).astype(str)
    out["lag_action"] = pd.to_numeric(rows.get("lag0_actionId", 0), errors="coerce").fillna(0).astype(int)
    out["lag_point"] = pd.to_numeric(rows.get("lag0_pointId", 0), errors="coerce").fillna(0).astype(int)
    out["lag_spin"] = pd.to_numeric(rows.get("lag0_spinId", 0), errors="coerce").fillna(0).astype(int)
    out["lag_strength"] = pd.to_numeric(rows.get("lag0_strengthId", 0), errors="coerce").fillna(0).astype(int)
    out["lag_family"] = rows["r184_lag0_family"].astype(str) if "r184_lag0_family" in rows else out["lag_action"].map(_action_family)
    out["lag_depth"] = rows["r184_lag0_depth"].astype(str) if "r184_lag0_depth" in rows else out["lag_point"].map(_point_depth)
    return out.reset_index(drop=True)


def fold_splits(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = rows["fold"].astype(int).to_numpy() if "fold" in rows else np.arange(len(rows), dtype=int) % 5
    return [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(np.unique(folds))]


def _rate_score(train_ctx: pd.DataFrame, train_y: np.ndarray, pred_ctx: pd.DataFrame, key_cols: list[str]) -> np.ndarray:
    scores = np.zeros((len(pred_ctx), N_ACTIONS), dtype=float)
    counts = np.bincount(np.asarray(train_y, dtype=int), minlength=N_ACTIONS).astype(float) + 1.0
    prior = counts / counts.sum()
    alpha = 20.0
    base = train_ctx[key_cols].copy()
    base["y"] = np.asarray(train_y, dtype=int)
    totals = base.groupby(key_cols, dropna=False).size().reset_index(name="total")
    for action in WEAK_ACTIONS.tolist():
        hits = (
            base.assign(hit=(base["y"].astype(int) == int(action)).astype(int))
            .groupby(key_cols, dropna=False)["hit"]
            .sum()
            .reset_index(name="hit")
        )
        table = totals.merge(hits, on=key_cols, how="left")
        table["hit"] = pd.to_numeric(table["hit"], errors="coerce").fillna(0.0)
        table["score"] = ((table["hit"] + prior[int(action)] * alpha) / (table["total"] + alpha)) * (
            table["total"] / (table["total"] + 10.0)
        )
        merged = pred_ctx[key_cols].merge(table[key_cols + ["score"]], on=key_cols, how="left")
        scores[:, int(action)] = pd.to_numeric(merged["score"], errors="coerce").fillna(prior[int(action)] * 0.05).to_numpy()
    return scores


def support_backoff_source(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    ctx = context_frame(rows)
    test_ctx = context_frame(test_rows)
    levels = [
        (0.50, ["phase", "lag_action", "lag_point", "lag_spin"]),
        (0.30, ["phase", "lag_family", "lag_depth"]),
        (0.20, ["phase"]),
    ]
    oof = np.zeros((len(rows), N_ACTIONS), dtype=float)
    for train_idx, valid_idx in fold_splits(rows):
        fold_score = np.zeros((len(valid_idx), N_ACTIONS), dtype=float)
        for weight, cols in levels:
            fold_score += float(weight) * _rate_score(ctx.iloc[train_idx].reset_index(drop=True), y[train_idx], ctx.iloc[valid_idx].reset_index(drop=True), cols)
        oof[valid_idx] = fold_score
    test = np.zeros((len(test_rows), N_ACTIONS), dtype=float)
    for weight, cols in levels:
        test += float(weight) * _rate_score(ctx, y, test_ctx, cols)
    return {
        "name": "foldsafe_transition_backoff",
        "source_family": "fold_safe_transition_backoff_tables",
        "oof_score": np.clip(oof, 0.0, 1.0),
        "test_score": np.clip(test, 0.0, 1.0),
        "fitted_folds": len(fold_splits(rows)),
    }


def _aligned_feature_matrices(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocked = {
        "rally_uid",
        "rally_id",
        "match",
        "fold",
        "next_actionId",
        "next_pointId",
        "serverGetPoint",
        "anchor_action",
    }
    cols = [c for c in train_frame.columns if c not in blocked and not str(c).startswith("next_")]
    train = train_frame.reindex(columns=cols, fill_value=0).copy()
    test = test_frame.reindex(columns=cols, fill_value=0).copy()
    both = pd.concat([train, test], axis=0, ignore_index=True)
    numeric_cols = [c for c in both.columns if pd.api.types.is_numeric_dtype(both[c])]
    pieces: list[pd.DataFrame] = []
    if numeric_cols:
        pieces.append(both[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float))
    cat_cols = [c for c in both.columns if c not in numeric_cols]
    if cat_cols:
        pieces.append(pd.get_dummies(both[cat_cols].fillna("__missing__").astype(str), dtype=float))
    matrix = pd.concat(pieces, axis=1) if pieces else pd.DataFrame({"bias": np.ones(len(both), dtype=float)})
    return matrix.iloc[: len(train_frame)].reset_index(drop=True), matrix.iloc[len(train_frame) :].reset_index(drop=True)


def build_v291_style_feature_frames(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    anchor_oof: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"v286_oof_status": "missing"}
    v286 = None
    if V286_OOF.exists():
        oof = pd.read_csv(V286_OOF)
        if len(oof) == len(rows):
            if "y_true_action" not in oof or np.array_equal(oof["y_true_action"].astype(int).to_numpy(), np.asarray(y, dtype=int)):
                v286 = oof
                meta["v286_oof_status"] = "loaded_as_feature_teacher"
            else:
                meta["v286_oof_status"] = "skipped_label_mismatch"
        else:
            meta["v286_oof_status"] = f"skipped_length_{len(oof)}_expected_{len(rows)}"

    def build_safe(base_rows: pd.DataFrame, anchor_action: np.ndarray, teacher: pd.DataFrame | None) -> pd.DataFrame:
        ctx = context_frame(base_rows)
        frame = pd.DataFrame(index=base_rows.index)
        for col in [
            "prefix_len",
            "phase_id",
            "lag0_actionId",
            "lag0_pointId",
            "lag0_spinId",
            "lag0_strengthId",
            "lag0_positionId",
            "scoreSelf",
            "scoreOther",
            "scoreTotal",
            "serverScoreDiff",
        ]:
            if col in base_rows:
                frame[col] = pd.to_numeric(base_rows[col], errors="coerce").fillna(0.0)
        frame["v330_anchor_action"] = np.asarray(anchor_action, dtype=int)
        frame["v330_anchor_is_weak"] = np.isin(anchor_action, WEAK_ACTIONS).astype(int)
        frame["v330_anchor_family"] = pd.Series(anchor_action).map(_action_family).astype(str).to_numpy()
        for col in ctx.columns:
            frame[f"ctx_{col}"] = ctx[col].to_numpy()
        if teacher is not None and len(teacher) == len(frame):
            for action in WEAK_ACTIONS.tolist():
                for prefix in ["specialist_p", "support"]:
                    col = f"{prefix}_{int(action)}"
                    if col in teacher:
                        frame[f"v286_{col}"] = pd.to_numeric(teacher[col], errors="coerce").fillna(0.0).to_numpy()
        else:
            for action in WEAK_ACTIONS.tolist():
                frame[f"v286_specialist_p_{int(action)}"] = 0.0
                frame[f"v286_support_{int(action)}"] = 0.0
        return frame.reset_index(drop=True)

    train_frame = build_safe(rows.reset_index(drop=True), np.asarray(anchor_oof, dtype=int), v286)
    # The strict V173 package action is inserted by load_strict_v173_anchor.
    test_frame = build_safe(test_rows.reset_index(drop=True), test_rows["anchor_action"].astype(int).to_numpy(), None)
    meta["feature_builder"] = "v330_safe_aicup_context_plus_v286_oof_no_external_priors"
    meta["external_priors_used"] = False
    return train_frame, test_frame, meta


def fit_weak_ovr_source(
    name: str,
    family: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
    *,
    kind: str,
) -> dict[str, Any]:
    oof = np.zeros((len(x_train), N_ACTIONS), dtype=float)
    test = np.zeros((len(x_test), N_ACTIONS), dtype=float)
    metrics = []
    fitted_total = 0
    for action in WEAK_ACTIONS.tolist():
        target = (np.asarray(y, dtype=int) == int(action)).astype(int)
        test_sum = np.zeros(len(x_test), dtype=float)
        fitted = 0
        for fold_id, (train_idx, valid_idx) in enumerate(fold_splits(rows)):
            if len(np.unique(target[train_idx])) < 2:
                continue
            if kind == "logreg":
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced", C=0.30, random_state=33000 + action * 17 + fold_id),
                )
            else:
                model = ExtraTreesClassifier(
                    n_estimators=140,
                    min_samples_leaf=9,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=33100 + action * 19 + fold_id,
                    n_jobs=1,
                )
            model.fit(x_train.iloc[train_idx], target[train_idx])
            oof[valid_idx, int(action)] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
            test_sum += model.predict_proba(x_test)[:, 1]
            fitted += 1
        if fitted:
            test[:, int(action)] = test_sum / float(fitted)
        else:
            base = float(target.mean()) if len(target) else 0.0
            oof[:, int(action)] = base
            test[:, int(action)] = base
        fitted_total += fitted
        metrics.append({"action": int(action), "positive_rows": int(target.sum()), "fitted_folds": int(fitted)})
    return {
        "name": name,
        "source_family": family,
        "oof_score": np.clip(oof, 0.0, 1.0),
        "test_score": np.clip(test, 0.0, 1.0),
        "fitted_folds": int(fitted_total),
        "metrics": metrics,
    }


def pred_to_score(pred: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros((rows, N_ACTIONS), dtype=float)
    p = np.asarray(pred, dtype=int)
    valid = (p >= 0) & (p < N_ACTIONS)
    out[np.arange(rows)[valid], p[valid]] = 1.0
    return out


def load_v209_source(rows: pd.DataFrame, test_rows: pd.DataFrame) -> dict[str, Any] | None:
    if not (V209_OOF.exists() and V209_TEST.exists()):
        return None
    oof = np.load(V209_OOF)
    test = np.load(V209_TEST)
    if oof.shape != (len(rows), N_ACTIONS) or test.shape != (len(test_rows), N_ACTIONS):
        return None
    return {
        "name": "v209_v208_aux_probability",
        "source_family": "existing_v209_probability_artifact",
        "oof_score": normalize_rows_safe(oof),
        "test_score": normalize_rows_safe(test),
        "fitted_folds": 0,
        "metrics": [],
    }


def load_r184_base_source(state_anchor_oof: np.ndarray, rows: pd.DataFrame, test_rows: pd.DataFrame) -> dict[str, Any] | None:
    try:
        from analysis_v290_shortcontrol411_specialist import _set_pickle_dataclasses
        from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

        _set_pickle_dataclasses()
        state = rebuild_v173_best_actions()
        base_oof = np.asarray(state["base_pred_oof"], dtype=int)
        base_test = np.asarray(state["base_pred_test"], dtype=int)
    except Exception:
        return None
    if len(base_oof) != len(rows) or len(base_test) != len(test_rows) or len(base_oof) != len(state_anchor_oof):
        return None
    return {
        "name": "r184_rebuilt_pre_v173_base",
        "source_family": "rebuilt_r184_base_action_pred",
        "oof_score": pred_to_score(base_oof, len(rows)),
        "test_score": pred_to_score(base_test, len(test_rows)),
        "fitted_folds": 0,
        "metrics": [],
    }


def _load_action_submission_score(path: Path, anchor_sub: pd.DataFrame, *, name: str, family: str) -> dict[str, Any] | None:
    try:
        sub = pd.read_csv(path)
    except Exception:
        return None
    if len(sub) != len(anchor_sub) or "actionId" not in sub:
        return None
    if "rally_uid" in sub and not sub["rally_uid"].astype(int).reset_index(drop=True).equals(anchor_sub["rally_uid"].astype(int).reset_index(drop=True)):
        return None
    pred = sub["actionId"].astype(int).to_numpy()
    return {
        "name": name,
        "source_family": family,
        "test_pred_only": pred,
        "path": relative_path(path),
    }


def load_existing_submission_audit(anchor_sub: pd.DataFrame) -> dict[str, Any]:
    roots = [
        ("v286", V286_SUB_DIR),
        ("v291", V291_SUB_DIR),
        ("r197", R197_SUB_DIR),
    ]
    records = []
    anchor = anchor_sub["actionId"].astype(int).to_numpy()
    for family, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("submission*.csv")):
            lower = str(path).lower()
            if "ttmatch" in lower or "tt-match" in lower or "old_server" in lower or "old-server" in lower:
                continue
            source = _load_action_submission_score(path, anchor_sub, name=path.stem, family=f"existing_{family}_submission_test_only")
            if source is None:
                continue
            pred = np.asarray(source["test_pred_only"], dtype=int)
            changed = pred != anchor
            records.append(
                {
                    "source": source["name"],
                    "source_family": source["source_family"],
                    "path": source["path"],
                    "test_changed_rows": int(changed.sum()),
                    "test_changed_distribution": action_distribution(pred[changed]) if changed.any() else "{}",
                    "used_for_evidence": False,
                    "reason": "test-only source; no strict V173 OOF prediction available",
                }
            )
    return {"test_only_sources": records, "test_only_source_count": len(records)}


def blend_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    if not sources:
        raise ValueError("sources cannot be empty")
    weights = {
        "v330_v291_style_weak_ovr_logreg": 0.25,
        "v330_v291_style_weak_ovr_extratrees": 0.35,
        "foldsafe_transition_backoff": 0.20,
        "v209_v208_aux_probability": 0.15,
        "r184_rebuilt_pre_v173_base": 0.05,
    }
    denom = 0.0
    oof = np.zeros_like(np.asarray(sources[0]["oof_score"], dtype=float))
    test = np.zeros_like(np.asarray(sources[0]["test_score"], dtype=float))
    for src in sources:
        w = float(weights.get(str(src["name"]), 0.10))
        oof += w * np.asarray(src["oof_score"], dtype=float)
        test += w * np.asarray(src["test_score"], dtype=float)
        denom += w
    return {
        "name": "v330_weighted_teacher_pool",
        "source_family": "weighted_weakclass_teacher_pool",
        "oof_score": np.clip(oof / max(denom, 1e-12), 0.0, 1.0),
        "test_score": np.clip(test / max(denom, 1e-12), 0.0, 1.0),
        "fitted_folds": int(sum(int(src.get("fitted_folds", 0)) for src in sources)),
        "metrics": [],
    }


def targets_and_margin(score: np.ndarray, anchor: np.ndarray, allowed_actions: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(score, dtype=float)
    base = np.asarray(anchor, dtype=int)
    scoped = np.zeros_like(arr)
    scoped[:, list(allowed_actions)] = arr[:, list(allowed_actions)]
    target = scoped.argmax(axis=1).astype(int)
    rows = np.arange(len(base))
    target_score = scoped[rows, target]
    anchor_score = np.zeros(len(base), dtype=float)
    anchor_in_scope = np.isin(base, list(allowed_actions))
    safe = np.clip(base, 0, arr.shape[1] - 1)
    anchor_score[anchor_in_scope] = arr[rows[anchor_in_scope], safe[anchor_in_scope]]
    return target, target_score - anchor_score


def select_rows(anchor: np.ndarray, target: np.ndarray, margin: np.ndarray, allowed_actions: tuple[int, ...], budget: int) -> np.ndarray:
    base = np.asarray(anchor, dtype=int)
    cand = np.asarray(target, dtype=int)
    score = np.asarray(margin, dtype=float)
    eligible = (
        (cand != base)
        & np.isin(cand, list(allowed_actions))
        & ~np.isin(base, PROTECTED_ANCHOR_ACTIONS)
        & ~np.isin(cand, SERVE_ACTIONS)
        & np.isfinite(score)
        & (score > 0.0)
    )
    selected = np.zeros(len(base), dtype=bool)
    if budget <= 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected


def apply_selected(anchor: np.ndarray, target: np.ndarray, selected: np.ndarray) -> np.ndarray:
    out = np.asarray(anchor, dtype=int).copy()
    mask = np.asarray(selected, dtype=bool)
    out[mask] = np.asarray(target, dtype=int)[mask]
    return out


def evaluate_candidate(
    source: dict[str, Any],
    group_name: str,
    allowed: tuple[int, ...],
    budget: int,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    oof_target, oof_margin = targets_and_margin(source["oof_score"], anchor_oof, allowed)
    test_target, test_margin = targets_and_margin(source["test_score"], anchor_test, allowed)
    oof_budget = int(math.floor(len(anchor_oof) * (float(budget) / max(len(anchor_test), 1))))
    oof_selected = select_rows(anchor_oof, oof_target, oof_margin, allowed, oof_budget)
    test_selected = select_rows(anchor_test, test_target, test_margin, allowed, budget)
    pred_oof = apply_selected(anchor_oof, oof_target, oof_selected)
    pred_test = apply_selected(anchor_test, test_target, test_selected)
    base_score = macro_f1(y, anchor_oof)
    score = macro_f1(y, pred_oof)
    precision = changed_row_precision(y, anchor_oof, pred_oof)
    changed_actions = pred_test[pred_test != anchor_test]
    serve_rows = int(np.isin(changed_actions, SERVE_ACTIONS).sum())
    candidate = f"v330_{source['name']}__{group_name}__b{int(budget)}"
    rec = {
        "candidate": candidate,
        "candidate_file": f"submission_{candidate}__v306point_v300server.csv",
        "source": source["name"],
        "source_family": source["source_family"],
        "specialist_group": group_name,
        "allowed_actions": "/".join(str(a) for a in allowed),
        "test_budget": int(budget),
        "oof_budget": int(oof_budget),
        "action_macro_f1": float(score),
        "anchor_action_macro_f1": float(base_score),
        "action_oof_delta": float(score - base_score),
        "specialist_group_delta": float(macro_f1(y, pred_oof, allowed) - macro_f1(y, anchor_oof, allowed)),
        "changed_action_rows": int(np.sum(pred_test != anchor_test)),
        "oof_changed_rows": int(precision["changed_rows"]),
        "changed_correct": int(precision["changed_correct"]),
        "changed_row_oof_precision": float(precision["changed_precision"]),
        "serve_action_rows": serve_rows,
        "test_changed_distribution": action_distribution(changed_actions) if len(changed_actions) else "{}",
        "test_action_distribution": action_distribution(pred_test),
        "min_test_margin_changed": float(test_margin[test_selected].min()) if test_selected.any() else 0.0,
        "mean_test_margin_changed": float(test_margin[test_selected].mean()) if test_selected.any() else 0.0,
        "evidence_pass": 0,
        "decision": "DO_NOT_UPLOAD",
    }
    rec["evidence_pass"] = int(evidence_passes(rec))
    rec["decision"] = "REVIEW_ACTION" if rec["evidence_pass"] else "DO_NOT_UPLOAD"
    return rec, pred_oof, pred_test


def build_search(
    sources: list[dict[str, Any]],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for source in sources:
        for group_name, allowed in SPECIALIST_GROUPS.items():
            for budget in TEST_BUDGETS:
                rec, pred_oof, pred_test = evaluate_candidate(source, group_name, allowed, budget, y, anchor_oof, anchor_test)
                key = rec["candidate"]
                records.append(rec)
                oof_predictions[key] = pred_oof
                test_predictions[key] = pred_test
    search = pd.DataFrame(records)
    if not search.empty:
        search = search.sort_values(
            [
                "evidence_pass",
                "action_oof_delta",
                "changed_row_oof_precision",
                "changed_action_rows",
                "source",
            ],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
    return search, oof_predictions, test_predictions


def export_submissions(search: pd.DataFrame, test_predictions: dict[str, np.ndarray], anchor_sub: pd.DataFrame) -> list[str]:
    generated: list[str] = []
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v330*.csv"):
        stale.unlink()
    if search.empty:
        return generated
    passed = search[search["evidence_pass"].astype(int).eq(1)].copy()
    for _, row in passed.iterrows():
        filename = str(row["candidate_file"])
        pred = test_predictions[str(row["candidate"])]
        out = build_export_frame(anchor_sub, pred)
        path = protected_output_path(OUTDIR, ExportSpec(filename))
        out.to_csv(path, index=False, float_format="%.8f")
        generated.append(relative_path(path))
    return generated


def markdown_table(rows: pd.DataFrame, columns: list[str]) -> str:
    if rows.empty:
        return "_None_"

    def cell(value: Any) -> str:
        if isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        return text.replace("|", "\\|")

    records = rows[columns].to_dict("records")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in records]
    return "\n".join([header, sep, *body])


def write_reports(
    search: pd.DataFrame,
    class_report: pd.DataFrame,
    generated: list[str],
    frame_meta: dict[str, Any],
    model_meta: dict[str, Any],
    source_audit: dict[str, Any],
    anchor_sub: pd.DataFrame,
) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    search.to_csv(OUTDIR / "v330_action_search.csv", index=False)
    class_report.to_csv(OUTDIR / "v330_class_report.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    reviewable = search[search["evidence_pass"].astype(int).eq(1)].copy() if len(search) else pd.DataFrame()
    decision = "REVIEW_ACTION" if not reviewable.empty else "DO_NOT_UPLOAD"
    report = json_safe(
        {
            "version": "V330",
            "decision": decision,
            "upload_worthy": decision == "REVIEW_ACTION",
            "anchor_submission": relative_path(ANCHOR_SUBMISSION),
            "action_anchor": "strict rebuilt V173 OOF/test; package action verified equal to rebuilt V173 test",
            "point_fixed_to": "V306 p0 cap0p01 pointId",
            "server_fixed_to": "V300 serverGetPoint",
            "ttmatch_used": False,
            "old_server_used": False,
            "copied_to_upload_or_selected": False,
            "manual_row_edits": False,
            "weak_actions": WEAK_ACTIONS.tolist(),
            "evidence_thresholds": {
                "min_action_oof_delta": MIN_ACTION_OOF_DELTA,
                "min_changed_row_precision": MIN_CHANGED_ROW_PRECISION,
                "min_changed_action_rows": MIN_CHANGED_ACTION_ROWS,
                "max_changed_action_rows": MAX_CHANGED_ACTION_ROWS,
                "max_serve_action_rows": MAX_SERVE_ACTION_ROWS,
            },
            "frame_meta": frame_meta,
            "model_meta": model_meta,
            "source_audit": source_audit,
            "best_candidate": best,
            "reviewable_candidates": reviewable.to_dict("records") if not reviewable.empty else [],
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "anchor_rows": int(len(anchor_sub)),
        }
    )
    (OUTDIR / "v330_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    top = search.head(12)
    md = [
        "# V330 action weak-class teacher pool",
        "",
        f"Anchor submission: `{relative_path(ANCHOR_SUBMISSION)}`",
        "Action anchor: strict rebuilt V173 OOF/test.",
        "Point/server: fixed to V306 p0 cap0p01 and V300.",
        f"Decision: `{decision}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate', '')}`",
        f"OOF action delta: {float(best.get('action_oof_delta', 0.0)):.6f}",
        f"Changed test rows: {int(best.get('changed_action_rows', 0))}",
        f"Changed-row OOF precision: {float(best.get('changed_row_oof_precision', 0.0)):.4f}",
        f"Evidence pass: `{bool(best.get('evidence_pass', 0))}`",
        "",
        "## Top search rows",
        "",
        markdown_table(
            top,
            [
                "candidate",
                "source",
                "specialist_group",
                "action_oof_delta",
                "changed_action_rows",
                "changed_row_oof_precision",
                "decision",
            ],
        ),
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v330_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, test_rows, y, anchor_oof, anchor_sub, frame_meta = load_strict_v173_anchor()
    require_rebuilt_v173_anchor(frame_meta)
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()

    train_frame, test_frame, feature_meta = build_v291_style_feature_frames(rows, test_rows, y, anchor_oof)
    x_train, x_test = _aligned_feature_matrices(train_frame, test_frame)
    sources: list[dict[str, Any]] = [
        fit_weak_ovr_source(
            "v330_v291_style_weak_ovr_logreg",
            "v291_style_fold_safe_weak_ovr_logreg",
            x_train,
            x_test,
            rows,
            y,
            kind="logreg",
        ),
        fit_weak_ovr_source(
            "v330_v291_style_weak_ovr_extratrees",
            "v291_style_fold_safe_weak_ovr_extratrees",
            x_train,
            x_test,
            rows,
            y,
            kind="extratrees",
        ),
        support_backoff_source(rows, test_rows, y),
    ]
    for optional in [load_v209_source(rows, test_rows), load_r184_base_source(anchor_oof, rows, test_rows)]:
        if optional is not None:
            sources.append(optional)
    sources.append(blend_sources(sources))

    search, oof_predictions, test_predictions = build_search(sources, y, anchor_oof, anchor_test)
    best_key = str(search.iloc[0]["candidate"]) if len(search) else ""
    best_oof = oof_predictions.get(best_key, anchor_oof)
    class_report = class_f1_report(y, anchor_oof, best_oof)
    generated = export_submissions(search, test_predictions, anchor_sub)
    source_audit = load_existing_submission_audit(anchor_sub)
    model_meta = {
        "feature_meta": feature_meta,
        "feature_columns": int(x_train.shape[1]),
        "sources": [
            {
                "name": src["name"],
                "source_family": src["source_family"],
                "fitted_folds": int(src.get("fitted_folds", 0)),
                "metrics": src.get("metrics", [])[:12],
            }
            for src in sources
        ],
    }
    return write_reports(search, class_report, generated, frame_meta, model_meta, source_audit, anchor_sub)


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": relative_path(OUTDIR),
                "decision": report.get("decision", "DO_NOT_UPLOAD"),
                "upload_worthy": bool(report.get("upload_worthy", False)),
                "best_candidate": best.get("candidate", ""),
                "best_action_oof_delta": best.get("action_oof_delta", 0.0),
                "best_changed_action_rows": best.get("changed_action_rows", 0),
                "best_changed_row_oof_precision": best.get("changed_row_oof_precision", 0.0),
                "generated_submission_count": report.get("generated_submission_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
