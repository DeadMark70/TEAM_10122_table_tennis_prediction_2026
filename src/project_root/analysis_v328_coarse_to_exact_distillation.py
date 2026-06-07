"""V328 coarse-to-exact AICUP distillation.

V328 consumes optional V326/V327 AICUP-prefix representation features, trains
exact action students only against AICUP `next_actionId`, and packages any
evidence-cleared action edits with the fixed V306 point + V300 server anchor.
No external exact action labels, TTMATCH inputs, old-server sources, or upload
directory writes are used.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "v328_coarse_to_exact_distillation"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V326_FEATURES = ROOT / "v326_masked_family_pretrain" / "v326_aicup_prefix_family_features.csv"
V327_FEATURES = ROOT / "v327_response_style_contrastive" / "v327_aicup_response_style_features.csv"
V286_OOF = ROOT / "v286_weak_action_specialist_pretraining" / "v286_specialist_oof.csv"
V317_SEARCH = ROOT / "v317_action_specialist_ensemble" / "v317_action_search.csv"
V323_SEARCH = ROOT / "v323_action_disagreement_mining" / "v323_action_search.csv"

ACTION_CLASSES = list(range(19))
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
WEAK_ACTIONS = [0, 3, 4, 5, 7, 8, 9, 12, 14]
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
PROTECTED_ANCHOR_ACTIONS = np.array([1, 10, 13, 15, 16, 17, 18], dtype=int)
BANNED_PATH_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER", "upload_candidates", "selected", "submissions")

MIN_ACTION_OOF_DELTA = 0.002
MIN_CHANGED_ROW_PRECISION = 0.451
MAX_SERVE_ACTION_ROWS = 0
STRICT_V173_ANCHOR_SOURCE = "rebuilt_v173_pred_oof"


@dataclass(frozen=True)
class ExportSpec:
    filename: str
    source_key: str
    test_budget: int = 18


CANDIDATE_SPECS: dict[str, ExportSpec] = {
    "family_feature": ExportSpec(
        "submission_v328_external_family_student__v306point_v300server.csv",
        "family_feature",
        18,
    ),
    "response_style": ExportSpec(
        "submission_v328_response_style_student__v306point_v300server.csv",
        "response_style",
        18,
    ),
    "v173_kd_external": ExportSpec(
        "submission_v328_v173kd_external_residual__v306point_v300server.csv",
        "v173_kd_external",
        18,
    ),
    "low_churn_selector": ExportSpec(
        "submission_v328_lowchurn_selector__v306point_v300server.csv",
        "low_churn_selector",
        10,
    ),
}


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


def path_has_banned_token(path: Path | str) -> bool:
    upper = str(path).upper()
    return any(token.upper() in upper for token in BANNED_PATH_TOKENS[:3])


def protected_output_path(outdir: Path, spec: ExportSpec) -> Path:
    root = Path(outdir)
    path = root / spec.filename
    parts = {part.lower() for part in path.parts}
    if any("upload_candidates" in part for part in parts) or "selected" in parts or "submissions" in parts:
        raise ValueError(f"refusing non-local V328 export path: {path}")
    if path.parent != root:
        raise ValueError(f"refusing non-local V328 export path: {path}")
    return path


def require_rebuilt_v173_anchor(frame_meta: dict[str, Any]) -> None:
    """Prevent fallback OOF anchors from driving V328 upload evidence."""
    source = str(frame_meta.get("anchor_oof_source", ""))
    if source != STRICT_V173_ANCHOR_SOURCE:
        raise RuntimeError(
            "strict V173 action anchor required for V328 evidence/export; "
            f"got anchor_oof_source={source!r}. "
            "Fallback anchors are diagnostic-only because they do not measure "
            "candidate deltas against the public-positive V173 action anchor."
        )


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
    return bool(
        float(data.get("action_oof_delta", 0.0)) >= MIN_ACTION_OOF_DELTA
        and float(data.get("changed_row_oof_precision", 0.0)) > MIN_CHANGED_ROW_PRECISION - 1e-12
        and int(data.get("changed_action_rows", 0)) > 0
        and int(data.get("serve_action_rows", 0)) <= MAX_SERVE_ACTION_ROWS
    )


def _is_split_column(series: pd.Series) -> bool:
    vals = {str(v).lower() for v in series.dropna().unique().tolist()}
    return bool(vals) and vals.issubset({"train", "test", "valid", "validation", "oof"})


def _find_split_column(df: pd.DataFrame) -> str | None:
    for col in ["split", "dataset", "partition", "stage"]:
        if col in df and _is_split_column(df[col]):
            return col
    return None


def _label_like_column(column: str) -> bool:
    lower = column.lower()
    blocked_exact = {
        "row_key",
        "actionid",
        "pointid",
        "servergetpoint",
        "next_actionid",
        "next_pointid",
        "target",
        "label",
        "truth",
        "y",
        "y_true",
        "fold",
        "match",
    }
    if lower in blocked_exact:
        return True
    return (
        lower.startswith("next_")
        or lower.endswith("_key")
        or lower.endswith("_uid")
        or lower.endswith("_id")
        or lower.endswith("_label")
        or lower.startswith("label_")
        or lower.startswith("target_")
        or "exact_action" in lower
        or "actionid" in lower
        or "pointid" in lower
        or "servergetpoint" in lower
    )


def _feature_keys(left: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    keys = ["rally_uid"]
    if "prefix_len" in left.columns and "prefix_len" in features.columns:
        keys.append("prefix_len")
    missing = [key for key in keys if key not in left.columns or key not in features.columns]
    if missing:
        return []
    return keys


def _split_feature_frame(features: pd.DataFrame, split_col: str | None, split_name: str) -> pd.DataFrame:
    if split_col is None:
        return features.copy()
    values = features[split_col].astype(str).str.lower()
    if split_name == "train":
        return features[values.isin(["train", "valid", "validation", "oof"])].copy()
    return features[values.eq("test")].copy()


def _prepare_external_features(
    base: pd.DataFrame,
    features: pd.DataFrame,
    split_col: str | None,
    split_name: str,
    source: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    part = _split_feature_frame(features, split_col, split_name)
    keys = _feature_keys(base, part)
    if not keys:
        return pd.DataFrame(index=base.index), [], []
    feature_cols = [
        col
        for col in part.columns
        if col not in keys and col != split_col and not _label_like_column(col)
    ]
    dropped = [
        col
        for col in part.columns
        if col not in keys and col != split_col and col not in feature_cols
    ]
    if not feature_cols:
        return pd.DataFrame(index=base.index), [], dropped
    subset = part[keys + feature_cols].copy()
    if subset.duplicated(keys).any():
        subset = subset.groupby(keys, as_index=False).first()
    renamed = {col: f"v328_ext_{source}_{col}" for col in feature_cols}
    subset = subset.rename(columns=renamed)
    merged = base[keys].merge(subset, on=keys, how="left", validate="many_to_one")
    out = merged[list(renamed.values())].copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = out[col].fillna("__missing__").astype(str)
    return out.reset_index(drop=True), list(out.columns), dropped


def merge_optional_external_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_paths: list[tuple[str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    train_out = train_df.reset_index(drop=True).copy()
    test_out = test_df.reset_index(drop=True).copy()
    report: dict[str, dict[str, Any]] = {}
    for source, path in feature_paths:
        if path_has_banned_token(path):
            raise ValueError(f"banned external feature path for V328: {path}")
        if not Path(path).exists():
            report[source] = {"status": "missing", "path": relative_path(Path(path)), "feature_columns": []}
            continue
        features = pd.read_csv(path)
        split_col = _find_split_column(features)
        train_features, train_cols, dropped_train = _prepare_external_features(train_out, features, split_col, "train", source)
        test_features, test_cols, dropped_test = _prepare_external_features(test_out, features, split_col, "test", source)
        if train_cols:
            train_out = pd.concat([train_out, train_features], axis=1)
        if test_cols:
            test_out = pd.concat([test_out, test_features], axis=1)
        report[source] = {
            "status": "loaded",
            "path": relative_path(Path(path)),
            "split_column": split_col,
            "feature_columns": sorted(set(train_cols + test_cols)),
            "dropped_label_like_columns": sorted(set(dropped_train + dropped_test)),
            "train_feature_count": len(train_cols),
            "test_feature_count": len(test_cols),
        }
    return train_out, test_out, report


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


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={SUBMISSION_COLUMNS}")
    if len(df) != int(expected_rows):
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
    if not df["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of range")
    if not df["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
    server = pd.to_numeric(df["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    if not server.between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")


def select_low_churn_predictions(
    anchor: np.ndarray,
    prob: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(anchor, dtype=int)
    p = normalize_rows_safe(prob)
    if len(base) != len(p):
        raise ValueError("anchor and prob must have matching row counts")
    target = p.argmax(axis=1).astype(int)
    rows = np.arange(len(base))
    base_safe = np.clip(base, 0, p.shape[1] - 1)
    margin = p[rows, target] - p[rows, base_safe]
    eligible = (target != base) & ~np.isin(target, SERVE_ACTIONS) & np.isfinite(margin) & (margin > 0.0)
    selected = np.zeros(len(base), dtype=bool)
    if int(budget) > 0 and eligible.any():
        idx = np.where(eligible)[0]
        order = idx[np.argsort(-margin[idx], kind="mergesort")]
        selected[order[: min(int(budget), len(order))]] = True
    pred = base.copy()
    pred[selected] = target[selected]
    return pred, selected, margin


def _load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing V306/V300 anchor submission: {ANCHOR_SUBMISSION}")
    sub = pd.read_csv(ANCHOR_SUBMISSION)
    validate_submission_frame(sub, expected_rows=len(sub))
    return sub


def load_aicup_prefix_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    anchor_sub = _load_anchor_submission()
    try:
        from analysis_v290_shortcontrol411_specialist import _set_pickle_dataclasses
        from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

        _set_pickle_dataclasses()
        state = rebuild_v173_best_actions()
        rows = state["rows"].reset_index(drop=True).copy()
        test_rows = state["test_rows"].reset_index(drop=True).copy()
        y = rows["next_actionId"].astype(int).to_numpy()
        anchor_oof = np.asarray(state["v173_pred_oof"], dtype=int)
        v173_test = np.asarray(state["v173_pred_test"], dtype=int)
        if len(anchor_oof) != len(rows) or len(v173_test) != len(test_rows):
            raise ValueError("rebuilt V173 predictions do not match rebuilt prefix rows")
        if len(anchor_sub) != len(test_rows):
            raise ValueError(f"anchor submission rows {len(anchor_sub)} != V173 test rows {len(test_rows)}")
        if not anchor_sub["rally_uid"].astype(int).reset_index(drop=True).equals(test_rows["rally_uid"].astype(int).reset_index(drop=True)):
            raise ValueError("V306/V300 anchor rally_uid does not match V173 test rows")
        meta: dict[str, Any] = {
            "row_source": "analysis_r184_receiver_affordance_refiner.rebuild_v173_best_actions",
            "anchor_oof_source": "rebuilt_v173_pred_oof",
            "anchor_test_source": "V306/V300 anchor submission actionId",
            "rebuilt_v173_test_mismatch_rows": int(
                np.sum(anchor_sub["actionId"].astype(int).to_numpy() != v173_test)
            ),
        }
        return rows, test_rows, y, anchor_oof, anchor_sub, meta
    except Exception as exc:
        fallback_reason = str(exc)

    from baseline_lgbm import add_role_and_score_features, build_test_prefix_table, build_train_prefix_table, validate_raw_data

    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    rows = build_train_prefix_table(train_raw, 6).reset_index(drop=True)
    test_rows = build_test_prefix_table(test_raw, 6).reset_index(drop=True)
    y = rows["next_actionId"].astype(int).to_numpy()
    meta: dict[str, Any] = {
        "row_source": "baseline_lgbm_prefix_tables",
        "anchor_oof_source": "fallback_lag0_actionId",
        "v173_rebuild_fallback_reason": fallback_reason,
    }

    if "match" in rows and rows["match"].nunique() >= 5:
        splitter = GroupKFold(n_splits=5)
        folds = np.full(len(rows), -1, dtype=int)
        rally_meta = rows[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
        for fold, (_, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"])):
            valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"].astype(int))
            folds[rows["rally_uid"].astype(int).isin(valid_rallies).to_numpy()] = int(fold)
        rows["fold"] = folds
    else:
        rows["fold"] = np.arange(len(rows), dtype=int) % 5

    anchor_oof = rows.get("lag0_actionId", pd.Series(np.zeros(len(rows), dtype=int))).astype(int).to_numpy()
    if V286_OOF.exists():
        oof = pd.read_csv(V286_OOF)
        if len(oof) == len(rows):
            oof_y = oof["y_true_action"].astype(int).to_numpy() if "y_true_action" in oof else y
            if np.array_equal(oof_y, y):
                anchor_oof = oof["anchor_action"].astype(int).to_numpy()
                if "fold" in oof:
                    rows["fold"] = oof["fold"].astype(int).to_numpy()
                meta["anchor_oof_source"] = relative_path(V286_OOF)
            else:
                meta["v286_oof_warning"] = "length matched but y_true_action did not align; used fallback anchor"
        else:
            meta["v286_oof_warning"] = f"length {len(oof)} did not match train rows {len(rows)}"

    if len(anchor_sub) != len(test_rows):
        raise ValueError(f"anchor submission rows {len(anchor_sub)} != test rows {len(test_rows)}")
    if not anchor_sub["rally_uid"].astype(int).reset_index(drop=True).equals(test_rows["rally_uid"].astype(int).reset_index(drop=True)):
        raise ValueError("V306/V300 anchor rally_uid does not match test prefix rows")
    return rows, test_rows, y, anchor_oof, anchor_sub, meta


def _base_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "rally_uid",
        "rally_id",
        "match",
        "fold",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "remaining_len",
        "final_parity_even",
        "num_prefixes_in_rally",
    }
    return [col for col in df.columns if col not in blocked and not col.startswith("next_") and not col.startswith("v328_ext_")]


def _source_columns(df: pd.DataFrame, source: str) -> list[str]:
    prefix = f"v328_ext_{source}_"
    return [col for col in df.columns if col.startswith(prefix)]


def _align_feature_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part = train_df.reindex(columns=columns, fill_value=0).copy()
    test_part = test_df.reindex(columns=columns, fill_value=0).copy()
    all_data = pd.concat([train_part, test_part], axis=0, ignore_index=True)
    numeric_cols = [col for col in all_data.columns if pd.api.types.is_numeric_dtype(all_data[col])]
    object_cols = [col for col in all_data.columns if col not in numeric_cols]
    pieces: list[pd.DataFrame] = []
    if numeric_cols:
        pieces.append(all_data[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float))
    if object_cols:
        pieces.append(pd.get_dummies(all_data[object_cols].fillna("__missing__").astype(str), prefix=object_cols, dtype=float))
    if pieces:
        matrix = pd.concat(pieces, axis=1)
    else:
        matrix = pd.DataFrame({"v328_bias": np.ones(len(all_data), dtype=float)})
    x_train = matrix.iloc[: len(train_df)].reset_index(drop=True)
    x_test = matrix.iloc[len(train_df) :].reset_index(drop=True)
    return x_train, x_test


def _predict_full(model: Any, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(ACTION_CLASSES)), dtype=float)
    for j, cls in enumerate(model.classes_):
        cls_i = int(cls)
        if cls_i in ACTION_CLASSES:
            out[:, cls_i] = raw[:, j]
    return normalize_rows_safe(out)


def _make_model(kind: str, seed: int) -> Any:
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.35,
                random_state=seed,
            ),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    return ExtraTreesClassifier(
        n_estimators=220,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def train_oof_action_prob(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = np.asarray(y, dtype=int)
    oof = np.zeros((len(x_train), len(ACTION_CLASSES)), dtype=float)
    test_sum = np.zeros((len(x_test), len(ACTION_CLASSES)), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    fitted = 0
    for fold in sorted(np.unique(folds.astype(int))):
        valid = folds.astype(int) == int(fold)
        train = ~valid
        if len(np.unique(y[train])) < 2:
            continue
        model = _make_model(kind, seed + int(fold))
        model.fit(x_train.iloc[train], y[train])
        oof[valid] = _predict_full(model, x_train.iloc[valid])
        test_sum += _predict_full(model, x_test)
        fitted += 1
        fold_rows.append(
            {
                "fold": int(fold),
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "train_classes": int(len(np.unique(y[train]))),
            }
        )
    if fitted == 0:
        prior = np.bincount(y, minlength=len(ACTION_CLASSES)).astype(float) + 1.0
        prior = prior / prior.sum()
        oof[:, :] = prior
        test_sum[:, :] = prior
        fitted = 1
        fold_rows.append({"fold": -1, "train_rows": int(len(y)), "valid_rows": int(len(y)), "train_classes": int(len(np.unique(y)))})
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / float(fitted)), fold_rows


def _add_anchor_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    train["v328_anchor_action"] = np.asarray(anchor_oof, dtype=int)
    test["v328_anchor_action"] = np.asarray(anchor_test, dtype=int)
    return train, test


def build_student_sources(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base_cols = _base_feature_columns(train_df)
    v326_cols = _source_columns(train_df, "v326")
    v327_cols = _source_columns(train_df, "v327")
    folds = train_df["fold"].astype(int).to_numpy() if "fold" in train_df else np.arange(len(train_df), dtype=int) % 5
    feature_plan = {
        "family_feature": sorted(set(base_cols + v326_cols)),
        "response_style": sorted(set(base_cols + v327_cols)),
    }
    train_kd, test_kd = _add_anchor_feature(train_df, test_df, anchor_oof, anchor_test)
    feature_plan["v173_kd_external"] = sorted(set(_base_feature_columns(train_kd) + v326_cols + v327_cols + ["v328_anchor_action"]))
    feature_plan["low_churn_selector"] = feature_plan["v173_kd_external"]

    kinds = {
        "family_feature": "logreg",
        "response_style": "et",
        "v173_kd_external": "rf",
        "low_churn_selector": "rf",
    }
    sources: dict[str, dict[str, Any]] = {}
    for i, (name, cols) in enumerate(feature_plan.items()):
        tdf, xdf = (train_kd, test_kd) if name in {"v173_kd_external", "low_churn_selector"} else (train_df, test_df)
        x_train, x_test = _align_feature_matrix(tdf, xdf, cols)
        oof, test, fold_rows = train_oof_action_prob(
            x_train,
            x_test,
            y,
            folds,
            kind=kinds[name],
            seed=32800 + i * 100,
        )
        sources[name] = {
            "name": name,
            "model_kind": kinds[name],
            "feature_count": int(x_train.shape[1]),
            "input_column_count": int(len(cols)),
            "external_feature_count": int(sum(col.startswith("v328_ext_") for col in cols)),
            "oof_prob": oof,
            "test_prob": test,
            "folds": fold_rows,
        }
    return sources, {"feature_plan": {k: len(v) for k, v in feature_plan.items()}}


def evaluate_candidate(
    spec: ExportSpec,
    source: dict[str, Any],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    oof_budget = int(math.floor(len(anchor_oof) * (float(spec.test_budget) / max(len(anchor_test), 1))))
    if spec.source_key == "low_churn_selector":
        oof_budget = max(1, int(math.floor(oof_budget * 0.6)))
    pred_oof, oof_selected, oof_margin = select_low_churn_predictions(anchor_oof, source["oof_prob"], oof_budget)
    pred_test, test_selected, test_margin = select_low_churn_predictions(anchor_test, source["test_prob"], spec.test_budget)
    base_score = macro_f1(y, anchor_oof)
    score = macro_f1(y, pred_oof)
    precision = changed_row_precision(y, anchor_oof, pred_oof)
    changed_test_actions = pred_test[pred_test != anchor_test]
    serve_rows = int(np.isin(changed_test_actions, SERVE_ACTIONS).sum())
    rec = {
        "candidate_file": spec.filename,
        "candidate": spec.filename.removesuffix(".csv"),
        "source": source["name"],
        "model_kind": source["model_kind"],
        "feature_count": int(source["feature_count"]),
        "external_feature_count": int(source["external_feature_count"]),
        "test_budget": int(spec.test_budget),
        "oof_budget": int(oof_budget),
        "action_macro_f1": float(score),
        "anchor_action_macro_f1": float(base_score),
        "action_oof_delta": float(score - base_score),
        "changed_action_rows": int(test_selected.sum()),
        "oof_changed_rows": int(oof_selected.sum()),
        "changed_correct": int(precision["changed_correct"]),
        "changed_row_oof_precision": float(precision["changed_precision"]),
        "serve_action_rows": serve_rows,
        "test_churn_vs_v173": float(np.mean(pred_test != anchor_test)),
        "test_changed_distribution": action_distribution(changed_test_actions) if len(changed_test_actions) else "{}",
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
    sources: dict[str, dict[str, Any]],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for key, spec in CANDIDATE_SPECS.items():
        rec, pred_oof, pred_test = evaluate_candidate(spec, sources[key], y, anchor_oof, anchor_test)
        records.append(rec)
        oof_predictions[key] = pred_oof
        test_predictions[key] = pred_test
    search = pd.DataFrame(records)
    if len(search):
        search = search.sort_values(
            ["evidence_pass", "action_oof_delta", "changed_row_oof_precision", "changed_action_rows"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return search, oof_predictions, test_predictions


def build_class_report(y: np.ndarray, anchor_oof: np.ndarray, best_oof: np.ndarray) -> pd.DataFrame:
    rows = []
    for action in WEAK_ACTIONS:
        anchor = macro_f1(y, anchor_oof, [action])
        best = macro_f1(y, best_oof, [action])
        rows.append(
            {
                "action": int(action),
                "support": int(np.sum(np.asarray(y, dtype=int) == int(action))),
                "anchor_f1": float(anchor),
                "v328_best_f1": float(best),
                "delta": float(best - anchor),
            }
        )
    return pd.DataFrame(rows)


def read_reference_evidence() -> dict[str, Any]:
    ref: dict[str, Any] = {
        "v317_best_changed_precision": 0.30,
        "v323_far_above_changed_precision": 0.45,
        "required_changed_precision": MIN_CHANGED_ROW_PRECISION,
    }
    for key, path in [("v317", V317_SEARCH), ("v323", V323_SEARCH)]:
        if not path.exists():
            ref[f"{key}_search_status"] = "missing"
            continue
        try:
            df = pd.read_csv(path)
            if len(df) and "action_oof_delta" in df:
                ref[f"{key}_best_delta"] = float(pd.to_numeric(df["action_oof_delta"], errors="coerce").max())
            if len(df) and "changed_row_oof_precision" in df:
                ref[f"{key}_best_changed_precision"] = float(pd.to_numeric(df["changed_row_oof_precision"], errors="coerce").max())
        except Exception as exc:
            ref[f"{key}_search_status"] = f"unreadable: {exc}"
    return ref


def export_submissions(
    search: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
    anchor_sub: pd.DataFrame,
) -> list[str]:
    generated: list[str] = []
    if search.empty:
        return generated
    for stale in OUTDIR.glob("submission_v328*.csv"):
        stale.unlink()
    for _, row in search[search["evidence_pass"].astype(int).eq(1)].iterrows():
        key = str(row["source"])
        spec = CANDIDATE_SPECS[key]
        out = build_export_frame(anchor_sub, test_predictions[key])
        path = protected_output_path(OUTDIR, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False, float_format="%.8f")
        generated.append(relative_path(path))
    return generated


def markdown_table(rows: pd.DataFrame, columns: list[str]) -> str:
    if rows.empty:
        return ""

    def cell(value: Any) -> str:
        if isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        return text.replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in rows[columns].to_dict("records")]
    return "\n".join([header, sep, *body])


def write_reports(
    search: pd.DataFrame,
    class_report: pd.DataFrame,
    generated: list[str],
    external_report: dict[str, Any],
    frame_meta: dict[str, Any],
    model_meta: dict[str, Any],
    anchor_sub: pd.DataFrame,
) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    search.to_csv(OUTDIR / "v328_action_search.csv", index=False)
    class_report.to_csv(OUTDIR / "v328_weak_class_report.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    reviewable = search[search["evidence_pass"].astype(int).eq(1)].copy() if len(search) else pd.DataFrame()
    decision = "REVIEW_ACTION" if not reviewable.empty else "DO_NOT_UPLOAD"
    report = json_safe(
        {
            "version": "V328",
            "decision": decision,
            "anchor_submission": relative_path(ANCHOR_SUBMISSION),
            "action_anchor": "V173 action from V306/V300 anchor and V286 OOF when available",
            "point_fixed_to": "V306 p0 cap0p01 pointId",
            "server_fixed_to": "V300 serverGetPoint",
            "exact_action_supervision": "AICUP train next_actionId only",
            "external_exact_labels_used": False,
            "ttmatch_used": False,
            "old_server_used": False,
            "copied_to_upload_or_selected": False,
            "external_features": external_report,
            "frame_meta": frame_meta,
            "model_meta": model_meta,
            "evidence_thresholds": {
                "min_action_oof_delta": MIN_ACTION_OOF_DELTA,
                "min_changed_row_oof_precision": MIN_CHANGED_ROW_PRECISION,
                "max_serve_action_rows": MAX_SERVE_ACTION_ROWS,
            },
            "reference_evidence": read_reference_evidence(),
            "best_candidate": best,
            "reviewable_candidates": reviewable.to_dict("records") if not reviewable.empty else [],
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "anchor_rows": int(len(anchor_sub)),
        }
    )
    (OUTDIR / "v328_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    top = search.head(12)
    md = [
        "# V328 coarse-to-exact AICUP distillation",
        "",
        f"Anchor submission: `{relative_path(ANCHOR_SUBMISSION)}`",
        "Point/server: fixed to V306 point line and V300 server.",
        "Exact action labels: AICUP `next_actionId` only.",
        f"Decision: `{decision}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate_file', '')}`",
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
                "candidate_file",
                "source",
                "model_kind",
                "external_feature_count",
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
    (OUTDIR / "v328_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, test_rows, y, anchor_oof, anchor_sub, frame_meta = load_aicup_prefix_frames()
    require_rebuilt_v173_anchor(frame_meta)
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    rows, test_rows, external_report = merge_optional_external_features(
        rows,
        test_rows,
        [("v326", V326_FEATURES), ("v327", V327_FEATURES)],
    )
    sources, model_meta = build_student_sources(rows, test_rows, y, anchor_oof, anchor_test)
    search, oof_predictions, test_predictions = build_search(sources, y, anchor_oof, anchor_test)
    best_source = str(search.iloc[0]["source"]) if len(search) else ""
    best_oof = oof_predictions.get(best_source, anchor_oof)
    class_report = build_class_report(y, anchor_oof, best_oof)
    generated = export_submissions(search, test_predictions, anchor_sub)
    return write_reports(search, class_report, generated, external_report, frame_meta, model_meta, anchor_sub)


def main() -> None:
    report = run_pipeline()
    best = report.get("best_candidate", {})
    print(
        json.dumps(
            {
                "outdir": relative_path(OUTDIR),
                "decision": report.get("decision", "DO_NOT_UPLOAD"),
                "best_candidate": best.get("candidate_file", ""),
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
