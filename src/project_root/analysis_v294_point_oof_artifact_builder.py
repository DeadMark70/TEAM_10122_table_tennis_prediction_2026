"""V294 point OOF artifact builder.

Builds a reusable row-level point artifact aligned to current train prefix rows
and the V261/V173/R121 clean test anchor. If no literal V261 point OOF artifact
is available, the base is rebuilt with fold-safe ExtraTrees probabilities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


POINT_CLASSES = list(range(10))
EXPECTED_TEST_ROWS = 1845
BASE_POINT_SOURCE = "rebuilt_v261_like"
MAX_LAG = 6
OOF_BASE_COLUMNS = [
    "row_id",
    "rally_uid",
    "fold",
    "y_true_point",
    "base_point_oof",
    "base_point_source",
    "prefix_len",
    "phase",
    "lag0_pointId",
    "lag0_actionId",
    "anchor_action",
]
TEST_BASE_COLUMNS = [
    "row_id",
    "rally_uid",
    "base_point_test",
    "base_point_source",
    "prefix_len",
    "phase",
    "lag0_pointId",
    "lag0_actionId",
    "anchor_action",
]
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
OUT_DIR_NAME = "v294_point_oof_artifact_builder"
ANCHOR_REL = Path("upload_candidates_20260519") / "submission_v261_cap0p01__v173action_r121server.csv"


def project_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test_new.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing train.csv and test_new.csv")


ROOT = project_root()
OUT_DIR = ROOT / OUT_DIR_NAME
ANCHOR_PATH = ROOT / ANCHOR_REL


def normalize_rows_safe(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr[~np.isfinite(arr)] = 0.0
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= eps
    if np.any(bad):
        arr[bad] = 1.0 / arr.shape[1]
        denom = arr.sum(axis=1, keepdims=True)
    return arr / denom


def point_depth(point_id: int) -> int:
    value = int(point_id)
    if value == 0:
        return 0
    if 1 <= value <= 3:
        return 1
    if 4 <= value <= 6:
        return 2
    if 7 <= value <= 9:
        return 3
    raise ValueError(f"unknown pointId: {point_id}")


def point_side(point_id: int) -> int:
    value = int(point_id)
    if value == 0:
        return 0
    if value in {1, 4, 7}:
        return 1
    if value in {2, 5, 8}:
        return 2
    if value in {3, 6, 9}:
        return 3
    raise ValueError(f"unknown pointId: {point_id}")


def action_family(action_id: int) -> int:
    value = int(action_id)
    if value == 0:
        return 0
    if 1 <= value <= 7:
        return 1
    if 8 <= value <= 11:
        return 2
    if 12 <= value <= 14:
        return 3
    if 15 <= value <= 18:
        return 4
    return 0


def phase_from_prefix(prefix_len: int) -> int:
    value = int(prefix_len)
    if value <= 1:
        return 0
    if value == 2:
        return 1
    if value == 3:
        return 2
    return 3


def validate_submission_frame(df: pd.DataFrame, expected_rows: int = EXPECTED_TEST_ROWS) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"bad submission columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"bad submission rows: {len(df)}")
    if not df["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of [0, 9]")
    if not df["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of [0, 18]")


def validate_artifact_schema(df: pd.DataFrame, kind: str) -> None:
    expected_by_kind = {
        "oof_base": OOF_BASE_COLUMNS,
        "test_base": TEST_BASE_COLUMNS,
    }
    if kind not in expected_by_kind:
        raise ValueError(f"unknown artifact kind: {kind}")
    expected = expected_by_kind[kind]
    if list(df.columns) != expected:
        raise ValueError(f"{kind}: bad columns {list(df.columns)}; expected {expected}")


def assign_folds(train_rows: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    folds = pd.Series(-1, index=train_rows.index, dtype=int)
    group_col = "match" if "match" in train_rows else "rally_uid"
    groups = train_rows[group_col].astype(str)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid_idx) in enumerate(splitter.split(train_rows, groups=groups)):
        folds.iloc[valid_idx] = int(fold)
    if folds.lt(0).any():
        raise RuntimeError("fold assignment failed")
    return folds


def _add_context_columns(
    frame: pd.DataFrame,
    *,
    anchor_action: np.ndarray,
    base_point: np.ndarray,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["phase"] = out["prefix_len"].astype(int).map(phase_from_prefix)
    out["anchor_action"] = np.asarray(anchor_action, dtype=int)
    out["anchor_action_family"] = out["anchor_action"].map(action_family)
    out["base_point"] = np.asarray(base_point, dtype=int)
    out["base_point_depth"] = out["base_point"].map(point_depth)
    out["base_point_side"] = out["base_point"].map(point_side)

    for lag in range(MAX_LAG):
        point_col = f"lag{lag}_pointId"
        action_col = f"lag{lag}_actionId"
        if point_col in out:
            clipped = out[point_col].astype(int).clip(0, 9)
            out[f"lag{lag}_point_depth"] = clipped.map(point_depth)
            out[f"lag{lag}_point_side"] = clipped.map(point_side)
            out[f"lag{lag}_point_is_zero"] = clipped.eq(0).astype(int)
        if action_col in out:
            out[f"lag{lag}_action_family"] = out[action_col].astype(int).map(action_family)

    if "serverScore" in out and "receiverScore" in out:
        out["score_total"] = out["serverScore"].astype(float) + out["receiverScore"].astype(float)
        out["score_diff"] = out["serverScore"].astype(float) - out["receiverScore"].astype(float)
    elif "scoreTotal" in out:
        out["score_total"] = out["scoreTotal"].astype(float)
        out["score_diff"] = out.get("serverScoreDiff", 0).astype(float)
    else:
        out["score_total"] = 0.0
        out["score_diff"] = 0.0
    out["score_close"] = out["score_diff"].abs().le(2).astype(int)
    out["prefix_bin"] = out["prefix_len"].astype(int).map(lambda v: 1 if v <= 1 else 2 if v == 2 else 3 if v == 3 else 4 if v <= 6 else 5)
    return out


def build_feature_frame(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    train_anchor_action: np.ndarray,
    train_base_point: np.ndarray,
    test_anchor_action: np.ndarray,
    test_base_point: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_feat = _add_context_columns(
        train_rows,
        anchor_action=train_anchor_action,
        base_point=train_base_point,
    )
    test_feat = _add_context_columns(
        test_rows,
        anchor_action=test_anchor_action,
        base_point=test_base_point,
    )

    for col in train_feat.columns:
        if col not in test_feat and pd.api.types.is_numeric_dtype(train_feat[col]):
            test_feat[col] = 0
    for col in test_feat.columns:
        if col not in train_feat and pd.api.types.is_numeric_dtype(test_feat[col]):
            train_feat[col] = 0

    blocked = {
        "rally_uid",
        "rally_id",
        "match",
        "server_id",
        "receiver_id",
        "gamePlayerId",
        "gamePlayerOtherId",
        "scoreSelf",
        "scoreOther",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "remaining_len",
        "final_parity_even",
        "num_prefixes_in_rally",
        "fold",
    }
    features: list[str] = []
    for col in train_feat.columns:
        if col in blocked or col not in test_feat:
            continue
        if pd.api.types.is_numeric_dtype(train_feat[col]):
            features.append(col)
    if not features:
        raise ValueError("no numeric feature columns available")
    leaked = [col for col in features if "PlayerId" in col or col in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"raw player leakage features detected: {leaked}")
    return train_feat.reset_index(drop=True), test_feat.reset_index(drop=True), features


def clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].replace([np.inf, -np.inf], 0).fillna(0.0)


def safe_predict_proba(model: ExtraTreesClassifier, frame: pd.DataFrame, classes: list[int] = POINT_CLASSES) -> np.ndarray:
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(classes)), dtype=float)
    pos = {int(cls): i for i, cls in enumerate(classes)}
    for j, cls in enumerate(model.classes_):
        if int(cls) in pos:
            out[:, pos[int(cls)]] = raw[:, j]
    return normalize_rows_safe(out)


def make_point_model(
    fold: int,
    *,
    n_estimators: int = 320,
    min_samples_leaf: int = 8,
    seed: int = 294,
    class_weight: str = "balanced_subsample",
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        class_weight=class_weight,
        max_features="sqrt",
        random_state=int(seed) + int(fold),
        n_jobs=1,
    )


def train_extratrees_point_base(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y_true_point: np.ndarray,
    features: list[str],
    *,
    n_estimators: int = 320,
    min_samples_leaf: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    if "fold" not in train_rows:
        raise ValueError("train_rows must contain fold")
    y = np.asarray(y_true_point, dtype=int)
    oof = np.zeros((len(train_rows), len(POINT_CLASSES)), dtype=float)
    test_sum = np.zeros((len(test_rows), len(POINT_CLASSES)), dtype=float)
    fold_report: list[dict[str, int]] = []
    x_test = clean_matrix(test_rows, features)
    fitted = 0
    for fold in sorted(train_rows["fold"].astype(int).unique()):
        valid = train_rows["fold"].astype(int).eq(fold).to_numpy()
        fit = ~valid
        if len(np.unique(y[fit])) < 2:
            raise RuntimeError(f"fold {fold} has fewer than two fit classes")
        model = make_point_model(fold, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
        model.fit(clean_matrix(train_rows.loc[fit], features), y[fit])
        oof[valid] = safe_predict_proba(model, clean_matrix(train_rows.loc[valid], features))
        test_sum += safe_predict_proba(model, x_test)
        fitted += 1
        fold_report.append(
            {
                "fold": int(fold),
                "train_rows": int(fit.sum()),
                "valid_rows": int(valid.sum()),
                "fit_classes": int(len(np.unique(y[fit]))),
            }
        )
    if fitted <= 0:
        raise RuntimeError("no ExtraTrees folds were fitted")
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / fitted), fold_report


def train_oof_prob(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: np.ndarray,
    classes: list[int],
    features: list[str],
    *,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    if "fold" not in train_df:
        raise ValueError("train_df must contain fold")
    y = np.asarray(target, dtype=int)
    oof = np.zeros((len(train_df), len(classes)), dtype=float)
    test_sum = np.zeros((len(test_df), len(classes)), dtype=float)
    x_test = clean_matrix(test_df, features)
    fold_report: list[dict[str, int]] = []
    fitted = 0
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(fold).to_numpy()
        fit = ~valid
        model = make_point_model(
            fold,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
            class_weight="balanced",
        )
        model.fit(clean_matrix(train_df.loc[fit], features), y[fit])
        oof[valid] = safe_predict_proba(model, clean_matrix(train_df.loc[valid], features), classes)
        test_sum += safe_predict_proba(model, x_test, classes)
        fitted += 1
        fold_report.append(
            {
                "fold": int(fold),
                "train_rows": int(fit.sum()),
                "valid_rows": int(valid.sum()),
                "fit_classes": int(len(np.unique(y[fit]))),
            }
        )
    if fitted <= 0:
        raise RuntimeError("no folds were fitted")
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / fitted), fold_report


def v261_numeric_feature_columns(df: pd.DataFrame, *, include_proxy: bool) -> list[str]:
    blocked = {
        "rally_uid",
        "rally_id",
        "match",
        "server_id",
        "receiver_id",
        "gamePlayerId",
        "gamePlayerOtherId",
        "scoreSelf",
        "scoreOther",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "fold",
        "anchor_action",
        "anchor_action_family",
        "base_point",
        "base_point_depth",
        "base_point_side",
    }
    if not include_proxy:
        blocked.update(
            {
                "v261_action_proxy",
                "v261_action_family",
                "v261_terminal_proxy",
                "v261_anchor_point",
                "v261_anchor_depth",
                "v261_anchor_side",
            }
        )
    return [
        c
        for c in df.columns
        if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
    ]


def add_v261_like_proxy_columns(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    train_out = train_rows.copy()
    test_out = test_rows.copy()
    base_features = [c for c in v261_numeric_feature_columns(train_out, include_proxy=False) if c in test_out]
    if not base_features:
        raise ValueError("no V261-like base features available")

    action_oof, action_test, action_folds = train_oof_prob(
        train_out,
        test_out,
        train_out["next_actionId"].astype(int).to_numpy(),
        list(range(19)),
        base_features,
        seed=2610,
        n_estimators=120,
        min_samples_leaf=5,
    )
    terminal_oof, terminal_test, terminal_folds = train_oof_prob(
        train_out,
        test_out,
        train_out["next_pointId"].eq(0).astype(int).to_numpy(),
        [0, 1],
        base_features,
        seed=2710,
        n_estimators=120,
        min_samples_leaf=8,
    )

    train_out["v261_action_proxy"] = action_oof.argmax(axis=1).astype(int)
    train_out["v261_action_family"] = train_out["v261_action_proxy"].map(action_family)
    train_out["v261_terminal_proxy"] = terminal_oof[:, 1]
    train_out["v261_anchor_point"] = -1
    train_out["v261_anchor_depth"] = -1
    train_out["v261_anchor_side"] = -1
    test_out["v261_action_proxy"] = test_out["anchor_action"].astype(int)
    test_out["v261_action_family"] = test_out["v261_action_proxy"].map(action_family)
    test_out["v261_terminal_proxy"] = terminal_test[:, 1]
    test_out["v261_anchor_point"] = test_out["base_point"].astype(int)
    test_out["v261_anchor_depth"] = test_out["v261_anchor_point"].map(point_depth)
    test_out["v261_anchor_side"] = test_out["v261_anchor_point"].map(point_side)
    return (
        train_out,
        test_out,
        [{"stage": "v261_like_action_proxy", **r} for r in action_folds]
        + [{"stage": "v261_like_terminal_proxy", **r} for r in terminal_folds],
    )


def train_v261_like_point_base(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    train_proxy, test_proxy, proxy_folds = add_v261_like_proxy_columns(train_rows, test_rows)
    for col in train_proxy.columns:
        if col not in test_proxy and pd.api.types.is_numeric_dtype(train_proxy[col]):
            test_proxy[col] = 0
    point_features = [c for c in v261_numeric_feature_columns(train_proxy, include_proxy=True) if c in test_proxy]
    if not point_features:
        raise ValueError("no V261-like point features available")
    point_oof, point_test, point_folds = train_oof_prob(
        train_proxy,
        test_proxy,
        train_proxy["next_pointId"].astype(int).to_numpy(),
        POINT_CLASSES,
        point_features,
        seed=2910,
        n_estimators=220,
        min_samples_leaf=4,
    )
    return (
        point_oof,
        point_test,
        point_features,
        proxy_folds + [{"stage": "v261_like_action_conditioned_point", **r} for r in point_folds],
        train_proxy.reset_index(drop=True),
        test_proxy.reset_index(drop=True),
    )


def load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"missing anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_frame(anchor)
    return anchor.reset_index(drop=True)


def build_prefix_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from baseline_lgbm import (
        add_role_and_score_features,
        build_test_prefix_table,
        build_train_prefix_table,
        validate_raw_data,
    )

    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train = add_role_and_score_features(train_raw)
    test = add_role_and_score_features(test_raw)
    train_rows = build_train_prefix_table(train, MAX_LAG).reset_index(drop=True)
    test_rows = build_test_prefix_table(test, MAX_LAG).reset_index(drop=True)
    if "next_pointId" not in train_rows:
        raise RuntimeError("train prefix rows lack next_pointId")
    if "lag0_pointId" not in train_rows or "lag0_actionId" not in train_rows:
        raise RuntimeError("train prefix rows lack required lag0 point/action columns")
    anchor = load_anchor_submission()
    if len(test_rows) != len(anchor):
        raise RuntimeError(f"test rows {len(test_rows)} do not align with anchor rows {len(anchor)}")
    if not test_rows["rally_uid"].astype(int).reset_index(drop=True).equals(anchor["rally_uid"].astype(int)):
        raise RuntimeError("test prefix rally_uid order does not align with V261 anchor")
    train_rows["fold"] = assign_folds(train_rows)
    return train_rows, test_rows, anchor


def make_oof_base_frame(
    train_rows: pd.DataFrame,
    y_true_point: np.ndarray,
    base_point_oof: np.ndarray,
    *,
    base_point_source: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_id": np.arange(len(train_rows), dtype=int),
            "rally_uid": train_rows["rally_uid"].astype(int).to_numpy(),
            "fold": train_rows["fold"].astype(int).to_numpy(),
            "y_true_point": np.asarray(y_true_point, dtype=int),
            "base_point_oof": np.asarray(base_point_oof, dtype=int),
            "base_point_source": base_point_source,
            "prefix_len": train_rows["prefix_len"].astype(int).to_numpy(),
            "phase": train_rows["phase"].astype(int).to_numpy(),
            "lag0_pointId": train_rows["lag0_pointId"].astype(int).to_numpy(),
            "lag0_actionId": train_rows["lag0_actionId"].astype(int).to_numpy(),
            "anchor_action": train_rows["anchor_action"].astype(int).to_numpy(),
        }
    )
    out = out[OOF_BASE_COLUMNS]
    validate_artifact_schema(out, "oof_base")
    return out


def make_test_base_frame(
    test_rows: pd.DataFrame,
    base_point_test: np.ndarray,
    *,
    base_point_source: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_id": np.arange(len(test_rows), dtype=int),
            "rally_uid": test_rows["rally_uid"].astype(int).to_numpy(),
            "base_point_test": np.asarray(base_point_test, dtype=int),
            "base_point_source": base_point_source,
            "prefix_len": test_rows["prefix_len"].astype(int).to_numpy(),
            "phase": test_rows["phase"].astype(int).to_numpy(),
            "lag0_pointId": test_rows["lag0_pointId"].astype(int).to_numpy(),
            "lag0_actionId": test_rows["lag0_actionId"].astype(int).to_numpy(),
            "anchor_action": test_rows["anchor_action"].astype(int).to_numpy(),
        }
    )
    out = out[TEST_BASE_COLUMNS]
    validate_artifact_schema(out, "test_base")
    return out


def audit_literal_point_source() -> dict[str, Any]:
    candidates = [
        ROOT / "v261_action_conditioned_point_residual" / "v261_point_oof_base.csv",
        ROOT / "v261_action_conditioned_point_residual" / "v261_point_oof_proba.npy",
        ROOT / "v188_point_intent_gru" / "v188_point_oof_base.csv",
        ROOT / "v188_point_intent_gru" / "v188_point_oof_proba.npy",
    ]
    existing = [str(path.relative_to(ROOT)) for path in candidates if path.exists()]
    return {
        "literal_source_available": False,
        "checked_paths": [str(path.relative_to(ROOT)) for path in candidates],
        "existing_checked_paths": existing,
        "decision": "No literal V261 row-level point OOF/proba pair was found; rebuilding a V261-like fold-safe action-conditioned point base.",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_reports(
    *,
    oof_base: pd.DataFrame,
    test_base: pd.DataFrame,
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    feature_count: int,
    fold_report: list[dict[str, int]],
    source_audit: dict[str, Any],
    base_macro_f1: float,
) -> dict[str, Any]:
    report = _json_safe(
        {
            "version": "V294",
            "base_point_source": BASE_POINT_SOURCE,
            "anchor_submission": str(ANCHOR_REL),
            "no_ttmatch_no_old_server": True,
            "y_true_point_source": "train prefix next_pointId rows only",
            "train_rows": int(len(oof_base)),
            "test_rows": int(len(test_base)),
            "oof_proba_shape": list(oof_proba.shape),
            "test_proba_shape": list(test_proba.shape),
            "feature_count": int(feature_count),
            "oof_base_macro_f1": float(base_macro_f1),
            "fold_report": fold_report,
            "source_audit": source_audit,
            "outputs": {
                "oof_base": f"{OUT_DIR_NAME}/v294_point_oof_base.csv",
                "oof_proba": f"{OUT_DIR_NAME}/v294_point_oof_proba.npy",
                "test_base": f"{OUT_DIR_NAME}/v294_point_test_base.csv",
                "test_proba": f"{OUT_DIR_NAME}/v294_point_test_proba.npy",
                "alignment_report": f"{OUT_DIR_NAME}/v294_alignment_report.json",
                "report": f"{OUT_DIR_NAME}/v294_report.md",
            },
            "concerns": [
                "No literal V261 row-level point OOF/proba artifact was found; V294 base is rebuilt_v261_like and not literal V261.",
                "Train anchor_action is the rebuilt V261-like fold-safe action proxy; test anchor_action is copied from the V261/V173 action anchor.",
            ],
        }
    )
    alignment = {
        "train_row_id_min": int(oof_base["row_id"].min()) if len(oof_base) else None,
        "train_row_id_max": int(oof_base["row_id"].max()) if len(oof_base) else None,
        "test_row_id_min": int(test_base["row_id"].min()) if len(test_base) else None,
        "test_row_id_max": int(test_base["row_id"].max()) if len(test_base) else None,
        "train_rows": int(len(oof_base)),
        "test_rows": int(len(test_base)),
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "oof_proba_shape": list(oof_proba.shape),
        "test_proba_shape": list(test_proba.shape),
        "oof_proba_no_nan": bool(np.isfinite(oof_proba).all()),
        "test_proba_no_nan": bool(np.isfinite(test_proba).all()),
        "oof_base_columns": list(oof_base.columns),
        "test_base_columns": list(test_base.columns),
        "base_point_source": BASE_POINT_SOURCE,
        "source_audit": source_audit,
    }
    (OUT_DIR / "v294_alignment_report.json").write_text(
        json.dumps(_json_safe(alignment), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# V294 point OOF artifact builder",
        "",
        f"Base point source: `{BASE_POINT_SOURCE}`",
        f"Anchor submission: `{ANCHOR_REL}`",
        "TTMATCH/old-server: not used.",
        "Labels: `y_true_point` comes from train prefix `next_pointId` rows only.",
        "",
        "## Shapes",
        "",
        f"- OOF base rows: `{len(oof_base)}`",
        f"- OOF proba shape: `{tuple(oof_proba.shape)}`",
        f"- Test base rows: `{len(test_base)}`",
        f"- Test proba shape: `{tuple(test_proba.shape)}`",
        f"- Feature count: `{feature_count}`",
        "",
        "## Metric",
        "",
        f"- Rebuilt V261-like OOF base Macro-F1: `{base_macro_f1:.6f}`",
        "",
        "## Source Audit",
        "",
        f"- Decision: {source_audit['decision']}",
        "",
        "## Concerns",
        "",
        *[f"- {item}" for item in report["concerns"]],
        "",
    ]
    (OUT_DIR / "v294_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_audit = audit_literal_point_source()
    train_rows_raw, test_rows_raw, anchor = build_prefix_rows()
    y_true_point = train_rows_raw["next_pointId"].astype(int).to_numpy()
    train_base_point_proxy = train_rows_raw["lag0_pointId"].astype(int).clip(0, 9).to_numpy()
    train_anchor_action = train_rows_raw["lag0_actionId"].astype(int).clip(0, 18).to_numpy()
    test_base_point_anchor = anchor["pointId"].astype(int).to_numpy()
    test_anchor_action = anchor["actionId"].astype(int).to_numpy()
    train_rows, test_rows, features = build_feature_frame(
        train_rows_raw,
        test_rows_raw,
        train_anchor_action=train_anchor_action,
        train_base_point=train_base_point_proxy,
        test_anchor_action=test_anchor_action,
        test_base_point=test_base_point_anchor,
    )
    oof_proba, test_proba, model_features, fold_report, model_train_rows, model_test_rows = train_v261_like_point_base(
        train_rows,
        test_rows,
    )
    model_train_rows = model_train_rows.copy()
    model_test_rows = model_test_rows.copy()
    model_train_rows["anchor_action"] = model_train_rows["v261_action_proxy"].astype(int)
    model_test_rows["anchor_action"] = model_test_rows["v261_action_proxy"].astype(int)
    classes = np.asarray(POINT_CLASSES, dtype=int)
    base_point_oof = classes[oof_proba.argmax(axis=1)]
    base_point_test = classes[test_proba.argmax(axis=1)]
    base_macro_f1 = float(
        f1_score(y_true_point, base_point_oof, labels=POINT_CLASSES, average="macro", zero_division=0)
    )
    oof_base = make_oof_base_frame(
        model_train_rows,
        y_true_point,
        base_point_oof,
        base_point_source=BASE_POINT_SOURCE,
    )
    test_base = make_test_base_frame(
        model_test_rows,
        base_point_test,
        base_point_source=BASE_POINT_SOURCE,
    )
    if oof_proba.shape != (len(oof_base), len(POINT_CLASSES)):
        raise RuntimeError(f"bad OOF proba shape: {oof_proba.shape}")
    if test_proba.shape != (EXPECTED_TEST_ROWS, len(POINT_CLASSES)):
        raise RuntimeError(f"bad test proba shape: {test_proba.shape}")
    if not np.isfinite(oof_proba).all() or not np.isfinite(test_proba).all():
        raise RuntimeError("probability arrays contain NaN/Inf")
    oof_base.to_csv(OUT_DIR / "v294_point_oof_base.csv", index=False)
    test_base.to_csv(OUT_DIR / "v294_point_test_base.csv", index=False)
    np.save(OUT_DIR / "v294_point_oof_proba.npy", oof_proba)
    np.save(OUT_DIR / "v294_point_test_proba.npy", test_proba)
    report = write_reports(
        oof_base=oof_base,
        test_base=test_base,
        oof_proba=oof_proba,
        test_proba=test_proba,
        feature_count=len(model_features),
        fold_report=fold_report,
        source_audit=source_audit,
        base_macro_f1=base_macro_f1,
    )
    (OUT_DIR / "v294_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": OUT_DIR_NAME,
                "base_point_source": report["base_point_source"],
                "train_rows": report["train_rows"],
                "test_rows": report["test_rows"],
                "oof_proba_shape": report["oof_proba_shape"],
                "test_proba_shape": report["test_proba_shape"],
                "oof_base_macro_f1": report["oof_base_macro_f1"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
