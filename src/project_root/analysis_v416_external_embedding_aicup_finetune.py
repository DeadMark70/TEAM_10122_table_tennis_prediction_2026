"""V416 AICUP exact fine-tune from clean external token embeddings.

Uses V415 coarse token embeddings as row features for local exact action/point
OOF models. Test inference is one row per anchor rally, aligned to submission
order.
"""

from __future__ import annotations

import json
import math
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
TOKEN_EMBEDDINGS_PATH = ROOT / "v415_clean_external_representation" / "token_embeddings.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v416_external_embedding_aicup_finetune"

NUMERIC_CONTEXT_COLUMNS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def build_train_transition_rows(train: pd.DataFrame) -> pd.DataFrame:
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train rows missing required columns: {sorted(missing)}")
    rows = train.copy()
    rows["_source_row_id"] = np.arange(len(rows), dtype=int)
    rows = rows.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").reset_index(drop=True)
    rows["target_actionId"] = rows.groupby("rally_uid", sort=False)["actionId"].shift(-1)
    rows["target_pointId"] = rows.groupby("rally_uid", sort=False)["pointId"].shift(-1)
    out = rows.loc[rows["target_actionId"].notna() & rows["target_pointId"].notna()].copy()
    out["source_row_id"] = out["_source_row_id"].astype(int)
    out["target_actionId"] = pd.to_numeric(out["target_actionId"], errors="coerce").astype(int)
    out["target_pointId"] = pd.to_numeric(out["target_pointId"], errors="coerce").astype(int)
    return out.drop(columns=["_source_row_id"]).reset_index(drop=True)


def build_test_rows(test: pd.DataFrame, anchor: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"rally_uid", "strikeNumber"}
    missing = required - set(test.columns)
    if missing:
        raise ValueError(f"test rows missing required columns: {sorted(missing)}")
    sorted_rows = test.copy().sort_values(["rally_uid", "strikeNumber"], kind="mergesort")
    last_rows = sorted_rows.groupby("rally_uid", sort=False, as_index=False).tail(1).copy()
    if anchor is None:
        return last_rows.sort_values(["rally_uid"], kind="mergesort").reset_index(drop=True)

    if "rally_uid" not in anchor.columns:
        raise ValueError("anchor must include rally_uid")
    aligned = anchor[["rally_uid"]].merge(last_rows, on="rally_uid", how="left", validate="one_to_one")
    if aligned["strikeNumber"].isna().any():
        missing_ids = aligned.loc[aligned["strikeNumber"].isna(), "rally_uid"].head(10).tolist()
        raise ValueError(f"test_new.csv missing observed prefix rows for anchor rally_uid values: {missing_ids}")
    return aligned.reset_index(drop=True)


def action_family(action_id: Any) -> str:
    value = _int_or_none(action_id)
    if value is None:
        return "unknown"
    if value in {15, 16, 17, 18}:
        return "serve"
    if value in {0, 1, 2, 3, 4}:
        return "short_control"
    if value in {5, 6, 7, 8, 9}:
        return "rally_drive"
    if value in {10, 11, 12, 13, 14}:
        return "attack_defense"
    return "other"


def prefix_len_bin(strike_number: Any) -> str:
    value = _int_or_none(strike_number)
    if value is None:
        return "unknown"
    if value <= 2:
        return "early"
    if value <= 5:
        return "middle"
    return "late"


def point_depth(point_id: Any) -> str:
    value = _int_or_none(point_id)
    if value is None or value == 0:
        return "terminal_or_unknown"
    if value in {1, 2, 3}:
        return "short"
    if value in {4, 5, 6}:
        return "half"
    if value in {7, 8, 9}:
        return "long"
    return "unknown"


def point_side(point_id: Any) -> str:
    value = _int_or_none(point_id)
    if value is None or value == 0:
        return "unknown"
    if value in {1, 4, 7}:
        return "left"
    if value in {2, 5, 8}:
        return "middle"
    if value in {3, 6, 9}:
        return "right"
    return "unknown"


def external_family_token(action_id: Any) -> str:
    value = _int_or_none(action_id)
    if value is None:
        return "fam=unknown"
    if value in {15, 16}:
        return "fam=badminton_short_service"
    if value in {17, 18}:
        return "fam=badminton_long_service"
    if value == 0:
        return "fam=unknown"
    if value in {1, 2, 5, 6, 7}:
        return "fam=badminton_drive"
    if value in {3, 14}:
        return "fam=badminton_smash"
    if value in {4, 8, 9, 10, 11}:
        return "fam=badminton_net_shot"
    if value in {12, 13}:
        return "fam=badminton_defensive_shot"
    return "fam=unknown"


def external_phase_token(strike_number: Any, action_id: Any) -> str:
    value = _int_or_none(action_id)
    if value in {15, 16, 17, 18}:
        return "phase=serve"
    return "phase=rally"


def external_spin_token(spin_id: Any) -> str:
    value = _int_or_none(spin_id)
    if value is None:
        return "spin=unknown"
    if value <= 1:
        return "spin=low"
    if value <= 3:
        return "spin=medium"
    return "spin=high"


def external_speed_token(strength_id: Any) -> str:
    value = _int_or_none(strength_id)
    if value is None:
        return "speed=unknown"
    if value <= 1:
        return "speed=high"
    if value == 2:
        return "speed=medium"
    return "speed=low"


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def derive_tokens(row: pd.Series) -> list[str]:
    return [
        external_family_token(row.get("actionId")),
        external_phase_token(row.get("strikeNumber"), row.get("actionId")),
        f"depth={point_depth(row.get('pointId'))}",
        f"side={point_side(row.get('pointId'))}",
        external_spin_token(row.get("spinId")),
        external_speed_token(row.get("strengthId")),
    ]


def _token_value(value: Any) -> str:
    parsed = _int_or_none(value)
    return str(parsed) if parsed is not None else "unknown"


def _load_embedding_lookup(token_embeddings: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[str]]:
    if token_embeddings.empty:
        return {}, []
    token_col = "token" if "token" in token_embeddings.columns else token_embeddings.columns[0]
    emb_cols = [col for col in token_embeddings.columns if col != token_col and pd.api.types.is_numeric_dtype(token_embeddings[col])]
    lookup = {
        str(row[token_col]): row[emb_cols].to_numpy(dtype=float)
        for _, row in token_embeddings[[token_col] + emb_cols].iterrows()
    }
    return lookup, emb_cols


def build_feature_frame(rows: pd.DataFrame, token_embeddings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup, emb_cols = _load_embedding_lookup(token_embeddings)
    dim = len(emb_cols)
    feature = pd.DataFrame(index=rows.index)
    for col in NUMERIC_CONTEXT_COLUMNS:
        source = rows[col] if col in rows.columns else pd.Series(0, index=rows.index)
        feature[col] = pd.to_numeric(source, errors="coerce").fillna(0.0).astype(float)

    families = rows.apply(lambda row: action_family(row.get("actionId")), axis=1)
    phases = rows.apply(lambda row: prefix_len_bin(row.get("strikeNumber")), axis=1)
    depths = rows.apply(lambda row: point_depth(row.get("pointId")), axis=1)
    sides = rows.apply(lambda row: point_side(row.get("pointId")), axis=1)
    for name, values in {"family": families, "phase": phases, "depth": depths, "side": sides}.items():
        feature[f"{name}_code"] = pd.Categorical(values).codes.astype(float)

    meta_rows: list[dict[str, Any]] = []
    mean_vectors: list[np.ndarray] = []
    sum_vectors: list[np.ndarray] = []
    for _, row in rows.iterrows():
        tokens = derive_tokens(row)
        vectors = [lookup.get(token, np.zeros(dim, dtype=float)) for token in tokens]
        matrix = np.vstack(vectors) if vectors and dim else np.zeros((len(tokens), dim), dtype=float)
        mean = matrix.mean(axis=0) if dim else np.zeros(0, dtype=float)
        total = matrix.sum(axis=0) if dim else np.zeros(0, dtype=float)
        mean_vectors.append(mean)
        sum_vectors.append(total)
        meta_rows.append({"tokens_json": json.dumps(tokens), "matched_token_count": int(sum(token in lookup for token in tokens))})

    for idx in range(dim):
        feature[f"v416_emb_mean_{idx}"] = [float(vec[idx]) for vec in mean_vectors]
        feature[f"v416_emb_sum_{idx}"] = [float(vec[idx]) for vec in sum_vectors]
    return feature.reset_index(drop=True), pd.DataFrame(meta_rows)


def _splitter(y: np.ndarray, groups: pd.Series | None) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    class_counts = pd.Series(y).value_counts()
    if groups is not None and groups.nunique(dropna=True) >= 3:
        n_splits = int(min(5, groups.nunique(dropna=True)))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    min_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_count >= 2:
        n_splits = int(min(5, min_count))
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=416).split(np.zeros(len(y)), y))
    indices = np.arange(len(y))
    return [(indices, indices)]


def _fit_model(x: pd.DataFrame, y: np.ndarray) -> Any:
    kwargs: dict[str, Any] = {"max_iter": 1000, "class_weight": "balanced", "random_state": 416}
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "auto"
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**kwargs),
    ).fit(x, y)


def _fit_oof_and_test(
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    groups: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    classes = sorted(int(v) for v in np.unique(y))
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    oof_prob = np.zeros((len(y), len(classes)), dtype=float)
    for train_idx, valid_idx in _splitter(y, groups):
        model = _fit_model(x_train.iloc[train_idx], y[train_idx])
        fold_prob = model.predict_proba(x_train.iloc[valid_idx])
        for local_idx, label in enumerate(model.classes_):
            oof_prob[valid_idx, class_to_index[int(label)]] = fold_prob[:, local_idx]
    missing = oof_prob.sum(axis=1) <= 0
    if missing.any():
        oof_prob[missing] = 1.0 / len(classes)
    model = _fit_model(x_train, y)
    test_local = model.predict_proba(x_test)
    test_prob = np.zeros((len(x_test), len(classes)), dtype=float)
    for local_idx, label in enumerate(model.classes_):
        test_prob[:, class_to_index[int(label)]] = test_local[:, local_idx]
    oof_pred = np.array([classes[idx] for idx in oof_prob.argmax(axis=1)], dtype=int)
    test_pred = np.array([classes[idx] for idx in test_prob.argmax(axis=1)], dtype=int)
    return oof_pred, oof_prob, test_pred, test_prob, classes


def _confidence(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prob.shape[1] == 0:
        return np.zeros(len(prob)), np.zeros(len(prob))
    sorted_prob = np.sort(prob, axis=1)
    confidence = sorted_prob[:, -1]
    runner_up = sorted_prob[:, -2] if prob.shape[1] > 1 else np.zeros(len(prob))
    return confidence, confidence - runner_up


def _append_prob_columns(frame: pd.DataFrame, prefix: str, classes: list[int], prob: np.ndarray) -> None:
    for idx, label in enumerate(classes):
        frame[f"{prefix}_prob_{label}"] = prob[:, idx]


def _class_metrics(y_true: np.ndarray, pred: np.ndarray, label_name: str) -> pd.DataFrame:
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    rows = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        row = {"label": label, "target": label_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    token_embedding_path: Path = TOKEN_EMBEDDINGS_PATH,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    token_embeddings = pd.read_csv(token_embedding_path)
    anchor = pd.read_csv(anchor_path)
    if not set(SUBMISSION_COLUMNS).issubset(anchor.columns):
        raise ValueError(f"anchor missing submission columns: {SUBMISSION_COLUMNS}")

    train_rows = build_train_transition_rows(train_raw)
    test_rows = build_test_rows(test_raw, anchor)
    x_train, train_meta = build_feature_frame(train_rows, token_embeddings)
    x_test, test_meta = build_feature_frame(test_rows, token_embeddings)

    groups = train_rows["match"] if "match" in train_rows.columns else None
    y_action = train_rows["target_actionId"].to_numpy(dtype=int)
    y_point = train_rows["target_pointId"].to_numpy(dtype=int)
    pred_action, prob_action, test_action, test_prob_action, action_classes = _fit_oof_and_test(x_train, y_action, x_test, groups)
    pred_point, prob_point, test_point, test_prob_point, point_classes = _fit_oof_and_test(x_train, y_point, x_test, groups)
    action_conf, action_margin = _confidence(prob_action)
    point_conf, point_margin = _confidence(prob_point)
    test_action_conf, test_action_margin = _confidence(test_prob_action)
    test_point_conf, test_point_margin = _confidence(test_prob_point)

    oof = train_rows[["rally_uid", "match", "strikeNumber", "source_row_id", "target_actionId", "target_pointId"]].copy()
    oof["pred_actionId"] = pred_action.astype(int)
    oof["pred_pointId"] = pred_point.astype(int)
    oof["action_confidence"] = action_conf
    oof["point_confidence"] = point_conf
    oof["action_margin"] = action_margin
    oof["point_margin"] = point_margin
    oof = pd.concat([oof.reset_index(drop=True), train_meta.reset_index(drop=True)], axis=1)
    _append_prob_columns(oof, "action", action_classes, prob_action)
    _append_prob_columns(oof, "point", point_classes, prob_point)

    test_pred = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    test_pred["pred_actionId"] = test_action.astype(int)
    test_pred["pred_pointId"] = test_point.astype(int)
    test_pred["actionId"] = test_pred["pred_actionId"]
    test_pred["pointId"] = test_pred["pred_pointId"]
    test_pred["action_confidence"] = test_action_conf
    test_pred["point_confidence"] = test_point_conf
    test_pred["action_margin"] = test_action_margin
    test_pred["point_margin"] = test_point_margin
    test_pred = pd.concat([test_pred.reset_index(drop=True), test_meta.reset_index(drop=True)], axis=1)
    _append_prob_columns(test_pred, "action", action_classes, test_prob_action)
    _append_prob_columns(test_pred, "point", point_classes, test_prob_point)

    metrics = {
        "version": "V416",
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows)),
        "test_observed_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "action_accuracy": float(accuracy_score(y_action, pred_action)),
        "point_accuracy": float(accuracy_score(y_point, pred_point)),
        "action_macro_f1": float(f1_score(y_action, pred_action, average="macro", zero_division=0)),
        "point_macro_f1": float(f1_score(y_point, pred_point, average="macro", zero_division=0)),
        "action_log_loss": _safe_log_loss(y_action, prob_action, action_classes),
        "point_log_loss": _safe_log_loss(y_point, prob_point, point_classes),
        "action_classes": action_classes,
        "point_classes": point_classes,
        "embedding_dimensions": int(len([col for col in token_embeddings.columns if col != "token" and pd.api.types.is_numeric_dtype(token_embeddings[col])])),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(outdir / "oof_predictions.csv", index=False)
    test_pred.to_csv(outdir / "test_predictions.csv", index=False)
    _class_metrics(y_action, pred_action, "action").to_csv(outdir / "class_metrics_action.csv", index=False)
    _class_metrics(y_point, pred_point, "point").to_csv(outdir / "class_metrics_point.csv", index=False)
    write_json(outdir / "local_metrics.json", metrics)
    return metrics


def _safe_log_loss(y_true: np.ndarray, prob: np.ndarray, labels: list[int]) -> float | None:
    try:
        return float(log_loss(y_true, prob, labels=labels))
    except ValueError:
        return None


if __name__ == "__main__":
    report = run_pipeline()
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
