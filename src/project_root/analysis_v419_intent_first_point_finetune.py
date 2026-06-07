"""V419 intent-first AICUP action/point fine-tune.

Builds next-row AICUP transition targets, enriches rows with clean external
token embeddings, trains fold-safe intent/action/point models, and packages
low-churn V362-anchored submissions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
V418_TOKEN_EMBEDDINGS_PATH = ROOT / "v418_clean_external_sequence_pretrain" / "token_embeddings.csv"
V415_TOKEN_EMBEDDINGS_PATH = ROOT / "v415_clean_external_representation" / "token_embeddings.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v419_intent_first_point_finetune"

NUMERIC_CONTEXT_COLUMNS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]


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


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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
    if value in SERVE_ACTION_CLASSES:
        return "serve"
    if value in {0, 1, 2, 3, 4}:
        return "short_control"
    if value in {5, 6, 7, 8, 9}:
        return "rally_drive"
    if value in {10, 11, 12, 13, 14}:
        return "attack_defense"
    return "other"


def action_intent(action_id: Any) -> str:
    value = _int_or_none(action_id)
    if value is None:
        return "unknown"
    if value in SERVE_ACTION_CLASSES:
        return "serve"
    if value in {1, 2, 5, 6, 7}:
        return "drive"
    if value in {3, 14}:
        return "attack"
    if value in {4, 8, 9, 10, 11}:
        return "net_control"
    if value in {12, 13}:
        return "defense"
    if value == 0:
        return "unknown"
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


def terminal_intent(point_id: Any) -> str:
    value = _int_or_none(point_id)
    return "terminal" if value == 0 else "nonterminal"


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
    if value in SERVE_ACTION_CLASSES:
        return "phase=serve"
    return f"phase={prefix_len_bin(strike_number)}"


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


def derive_tokens(row: pd.Series) -> list[str]:
    return [
        external_family_token(row.get("actionId")),
        external_phase_token(row.get("strikeNumber"), row.get("actionId")),
        f"depth={point_depth(row.get('pointId'))}",
        f"side={point_side(row.get('pointId'))}",
        external_spin_token(row.get("spinId")),
        external_speed_token(row.get("strengthId")),
        f"terminal={terminal_intent(row.get('pointId'))}",
    ]


def _load_embedding_lookup(token_embeddings: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[str]]:
    if token_embeddings.empty:
        return {}, []
    token_col = "token" if "token" in token_embeddings.columns else token_embeddings.columns[0]
    emb_cols = [
        col
        for col in token_embeddings.columns
        if col != token_col and pd.api.types.is_numeric_dtype(token_embeddings[col])
    ]
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
        feature[f"num_{col}"] = pd.to_numeric(source, errors="coerce").fillna(0.0).astype(float)

    families = rows.apply(lambda row: action_family(row.get("actionId")), axis=1)
    intents = rows.apply(lambda row: action_intent(row.get("actionId")), axis=1)
    phases = rows.apply(lambda row: prefix_len_bin(row.get("strikeNumber")), axis=1)
    depths = rows.apply(lambda row: point_depth(row.get("pointId")), axis=1)
    sides = rows.apply(lambda row: point_side(row.get("pointId")), axis=1)
    terminals = rows.apply(lambda row: terminal_intent(row.get("pointId")), axis=1)
    categorical = pd.DataFrame(
        {
            "family": families,
            "intent": intents,
            "prefix_phase": phases,
            "point_depth": depths,
            "point_side": sides,
            "terminal_intent": terminals,
        },
        index=rows.index,
    )
    dummies = pd.get_dummies(categorical, prefix=categorical.columns, dtype=float)
    feature = pd.concat([feature, dummies], axis=1)
    feature["intent_is_serve"] = families.eq("serve").astype(float).to_numpy()
    feature["intent_is_terminal"] = terminals.eq("terminal").astype(float).to_numpy()
    feature["intent_depth_ord"] = depths.map({"short": 1.0, "half": 2.0, "long": 3.0}).fillna(0.0).to_numpy()
    feature["intent_side_ord"] = sides.map({"left": 1.0, "middle": 2.0, "right": 3.0}).fillna(0.0).to_numpy()

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
        feature[f"v419_ext_emb_mean_{idx}"] = [float(vec[idx]) for vec in mean_vectors]
        feature[f"v419_ext_emb_sum_{idx}"] = [float(vec[idx]) for vec in sum_vectors]
    return feature.reset_index(drop=True), pd.DataFrame(meta_rows)


def align_feature_columns(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(x_train.columns) | set(x_test.columns))
    return (
        x_train.reindex(columns=columns, fill_value=0.0).astype(float),
        x_test.reindex(columns=columns, fill_value=0.0).astype(float),
    )


class ConstantClassifier:
    def __init__(self, value: Any):
        self.classes_ = np.array([value])
        self.value = value

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.ones((len(x), 1), dtype=float)


def _splitter(y: np.ndarray, groups: pd.Series | None, seed: int = 419) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    indices = np.arange(len(y))
    if len(y) < 2:
        return [(indices, indices)]
    if groups is not None and groups.nunique(dropna=True) >= 2:
        n_splits = int(min(3, groups.nunique(dropna=True)))
        if n_splits >= 2:
            return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    class_counts = pd.Series(y).value_counts()
    min_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_count >= 2:
        n_splits = int(min(3, min_count))
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return [(indices, indices)]


def _fit_model(x: pd.DataFrame, y: np.ndarray, *, class_weight: str | None = "balanced") -> Any:
    unique = pd.unique(pd.Series(y))
    if len(unique) <= 1:
        return ConstantClassifier(unique[0])
    return DecisionTreeClassifier(
        max_depth=14,
        min_samples_leaf=25,
        class_weight=class_weight,
        random_state=419,
    ).fit(x, y)


def _sort_labels(values: list[Any]) -> list[Any]:
    if all(isinstance(value, (int, np.integer)) for value in values):
        return sorted(values, key=int)
    return sorted(values, key=lambda value: str(value))


def _fit_oof_and_test(
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    groups: pd.Series | None,
    *,
    class_weight: str | None = "balanced",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    classes = _sort_labels(pd.unique(pd.Series(y)).tolist())
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    oof_prob = np.zeros((len(y), len(classes)), dtype=float)
    for train_idx, valid_idx in _splitter(y, groups):
        model = _fit_model(x_train.iloc[train_idx], y[train_idx], class_weight=class_weight)
        fold_prob = model.predict_proba(x_train.iloc[valid_idx])
        for local_idx, label in enumerate(model.classes_):
            oof_prob[valid_idx, class_to_index[label]] = fold_prob[:, local_idx]
    missing = oof_prob.sum(axis=1) <= 0
    if missing.any() and classes:
        oof_prob[missing] = 1.0 / len(classes)

    model = _fit_model(x_train, y, class_weight=class_weight)
    test_local = model.predict_proba(x_test)
    test_prob = np.zeros((len(x_test), len(classes)), dtype=float)
    for local_idx, label in enumerate(model.classes_):
        test_prob[:, class_to_index[label]] = test_local[:, local_idx]

    oof_pred = np.array([classes[idx] for idx in oof_prob.argmax(axis=1)])
    test_pred = np.array([classes[idx] for idx in test_prob.argmax(axis=1)])
    return oof_pred, oof_prob, test_pred, test_prob, classes


def _confidence(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prob.shape[1] == 0:
        return np.zeros(len(prob)), np.zeros(len(prob))
    sorted_prob = np.sort(prob, axis=1)
    confidence = sorted_prob[:, -1]
    runner_up = sorted_prob[:, -2] if prob.shape[1] > 1 else np.zeros(len(prob))
    return confidence, confidence - runner_up


def _safe_suffix(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return text or "blank"


def _append_prob_columns(frame: pd.DataFrame, prefix: str, classes: list[Any], prob: np.ndarray) -> None:
    for idx, label in enumerate(classes):
        frame[f"{prefix}_prob_{_safe_suffix(label)}"] = prob[:, idx]


def _conditioning_frame(
    pred_action: np.ndarray,
    depth_classes: list[Any],
    depth_prob: np.ndarray,
    side_classes: list[Any],
    side_prob: np.ndarray,
) -> pd.DataFrame:
    rows = pd.DataFrame({"pred_action_family": [action_family(value) for value in pred_action]})
    out = pd.get_dummies(rows, prefix=["cond_action_family"], dtype=float)
    for idx, label in enumerate(depth_classes):
        out[f"cond_depth_prob_{_safe_suffix(label)}"] = depth_prob[:, idx]
    for idx, label in enumerate(side_classes):
        out[f"cond_side_prob_{_safe_suffix(label)}"] = side_prob[:, idx]
    return out.reset_index(drop=True)


def _safe_log_loss(y_true: np.ndarray, prob: np.ndarray, labels: list[Any]) -> float | None:
    try:
        return float(log_loss(y_true, prob, labels=labels))
    except ValueError:
        return None


def load_token_embeddings(
    token_embedding_path: Path = V418_TOKEN_EMBEDDINGS_PATH,
    fallback_embedding_path: Path = V415_TOKEN_EMBEDDINGS_PATH,
) -> tuple[pd.DataFrame, Path, bool]:
    token_embedding_path = Path(token_embedding_path)
    fallback_embedding_path = Path(fallback_embedding_path)
    if token_embedding_path.exists():
        return pd.read_csv(token_embedding_path), token_embedding_path, False
    if fallback_embedding_path.exists():
        return pd.read_csv(fallback_embedding_path), fallback_embedding_path, True
    raise FileNotFoundError(
        f"missing token embeddings: {token_embedding_path} and fallback {fallback_embedding_path}"
    )


def _align_predictions_to_anchor(predictions: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    if "rally_uid" not in predictions.columns:
        if len(predictions) != len(anchor):
            raise ValueError("predictions without rally_uid must match anchor row count")
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
            reduced = reduced.sort_values(["rally_uid", "strikeNumber"], kind="mergesort")
        reduced = reduced.groupby("rally_uid", sort=False).tail(1)
    if set(reduced["rally_uid"].astype(str)) != set(anchor["rally_uid"].astype(str)):
        missing = sorted(set(anchor["rally_uid"].astype(str)) - set(reduced["rally_uid"].astype(str)))
        raise ValueError(f"predictions cannot align to anchor; missing rally_uid values: {missing[:5]}")
    return (
        reduced.assign(rally_uid=reduced["rally_uid"].astype(str))
        .set_index("rally_uid")
        .loc[anchor["rally_uid"].astype(str)]
        .reset_index()
    )


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _confidence_column(predictions: pd.DataFrame, kind: str, fallback: float = 0.0) -> pd.Series:
    candidates = {
        "action": ("action_margin", "action_confidence", "action_prob_max"),
        "point": ("point_margin", "point_confidence", "point_prob_max"),
        "joint": ("joint_margin", "joint_confidence"),
    }[kind]
    col = _first_existing_column(predictions, candidates)
    if col is not None:
        return pd.to_numeric(predictions[col], errors="coerce").fillna(fallback).astype(float)
    stable = pd.util.hash_pandas_object(predictions["rally_uid"].astype(str), index=False).astype(float)
    return pd.Series(fallback + ((stable % 1000) / 1_000_000.0), index=predictions.index)


def build_ranked_changes(anchor: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = _align_predictions_to_anchor(predictions, anchor)
    action_col = _first_existing_column(predictions, ("pred_actionId", "action_pred", "candidate_actionId"))
    point_col = _first_existing_column(predictions, ("pred_pointId", "point_pred", "candidate_pointId"))
    if action_col is None or point_col is None:
        raise ValueError("predictions must include pred_actionId and pred_pointId")
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
    rows["action_confidence"] = _confidence_column(predictions, "action").to_numpy()
    rows["point_confidence"] = _confidence_column(predictions, "point").to_numpy()
    rows["joint_confidence"] = _confidence_column(
        predictions,
        "joint",
        fallback=float((rows["action_confidence"].mean() + rows["point_confidence"].mean()) / 2.0)
        if len(rows)
        else 0.0,
    ).to_numpy()
    return rows


def _select_changes(changes: pd.DataFrame, eligible_col: str, confidence_col: str, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return changes.head(0).copy()
    return (
        changes.loc[changes[eligible_col]]
        .sort_values([confidence_col, "row_id"], ascending=[False, True])
        .head(int(limit))
        .reset_index(drop=True)
    )


def _apply_selected(
    anchor: pd.DataFrame,
    action_selected: pd.DataFrame,
    point_selected: pd.DataFrame,
) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    for row in action_selected.itertuples(index=False):
        if bool(row.action_eligible):
            out.at[int(row.row_id), "actionId"] = int(row.pred_action)
    for row in point_selected.itertuples(index=False):
        if bool(row.point_eligible):
            out.at[int(row.row_id), "pointId"] = int(row.pred_point)
    return out.loc[:, SUBMISSION_COLUMNS].copy()


def _submission_stats(
    candidate: str,
    frame: pd.DataFrame,
    anchor: pd.DataFrame,
    path: Path | None,
    action_selected: pd.DataFrame,
    point_selected: pd.DataFrame,
    changes: pd.DataFrame,
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
        "path": str(path.resolve()) if path is not None else "",
        "action_selected_count": int(len(action_selected)),
        "point_selected_count": int(len(point_selected)),
        "action_churn": int(action_changed.sum()),
        "point_churn": int(point_changed.sum()),
        "server_changed": int(server_changed.sum()),
        "serve_15_18_additions": int(serve_additions.sum()),
        "point0_additions": int(point0_additions.sum()),
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
    }


def package_low_churn(
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    point_limit: int,
    action_limit: int = 0,
    candidate: str = "low_churn",
    expected_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    changes = build_ranked_changes(anchor, predictions)
    action_selected = _select_changes(changes, "action_eligible", "action_confidence", action_limit)
    point_selected = _select_changes(changes, "point_eligible", "point_confidence", point_limit)
    frame = _apply_selected(anchor, action_selected, point_selected)
    validate_submission_schema(frame, expected_rows=expected_rows)
    report = _submission_stats(candidate, frame, anchor, None, action_selected, point_selected, changes)
    return frame, report


def package_joint_low_churn(
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    limit: int,
    candidate: str = "joint",
    expected_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    changes = build_ranked_changes(anchor, predictions)
    selected = (
        changes.loc[changes["action_eligible"] | changes["point_eligible"]]
        .sort_values(["joint_confidence", "row_id"], ascending=[False, True])
        .head(int(limit))
        .reset_index(drop=True)
    )
    action_selected = selected.loc[selected["action_eligible"]].copy()
    point_selected = selected.loc[selected["point_eligible"]].copy()
    frame = _apply_selected(anchor, action_selected, point_selected)
    validate_submission_schema(frame, expected_rows=expected_rows)
    report = _submission_stats(candidate, frame, anchor, None, action_selected, point_selected, changes)
    return frame, report


def _write_candidate(
    *,
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
    outdir: Path,
    filename: str,
    candidate: str,
    expected_rows: int | None,
    point_limit: int | None = None,
    action_limit: int = 0,
    joint_limit: int | None = None,
) -> dict[str, Any]:
    if joint_limit is None:
        frame, stats = package_low_churn(
            anchor,
            predictions,
            point_limit=int(point_limit or 0),
            action_limit=action_limit,
            candidate=candidate,
            expected_rows=expected_rows,
        )
    else:
        frame, stats = package_joint_low_churn(
            anchor,
            predictions,
            limit=joint_limit,
            candidate=candidate,
            expected_rows=expected_rows,
        )
    path = safe_output_path(outdir, filename)
    frame.to_csv(path, index=False)
    stats["path"] = str(path.resolve())
    return stats


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    token_embedding_path: Path = V418_TOKEN_EMBEDDINGS_PATH,
    fallback_embedding_path: Path = V415_TOKEN_EMBEDDINGS_PATH,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    token_embeddings, resolved_embedding_path, fallback_used = load_token_embeddings(
        token_embedding_path, fallback_embedding_path
    )
    anchor = pd.read_csv(anchor_path).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(anchor, expected_rows=expected_rows)

    train_rows = build_train_transition_rows(train_raw)
    test_rows = build_test_rows(test_raw, anchor)
    x_train_base, train_meta = build_feature_frame(train_rows, token_embeddings)
    x_test_base, test_meta = build_feature_frame(test_rows, token_embeddings)
    x_train_base, x_test_base = align_feature_columns(x_train_base, x_test_base)
    groups = train_rows["match"] if "match" in train_rows.columns else None

    y_action = train_rows["target_actionId"].to_numpy(dtype=int)
    y_point = train_rows["target_pointId"].to_numpy(dtype=int)
    y_depth = np.array([point_depth(value) for value in y_point], dtype=object)
    y_side = np.array([point_side(value) for value in y_point], dtype=object)

    pred_action, prob_action, test_action, test_prob_action, action_classes = _fit_oof_and_test(
        x_train_base, y_action, x_test_base, groups, class_weight="balanced"
    )
    pred_depth, prob_depth, test_depth, test_prob_depth, depth_classes = _fit_oof_and_test(
        x_train_base, y_depth, x_test_base, groups, class_weight="balanced"
    )
    pred_side, prob_side, test_side, test_prob_side, side_classes = _fit_oof_and_test(
        x_train_base, y_side, x_test_base, groups, class_weight="balanced"
    )

    train_cond = _conditioning_frame(pred_action, depth_classes, prob_depth, side_classes, prob_side)
    test_cond = _conditioning_frame(test_action, depth_classes, test_prob_depth, side_classes, test_prob_side)
    x_point_train, x_point_test = align_feature_columns(
        pd.concat([x_train_base.reset_index(drop=True), train_cond], axis=1),
        pd.concat([x_test_base.reset_index(drop=True), test_cond], axis=1),
    )
    pred_point, prob_point, test_point, test_prob_point, point_classes = _fit_oof_and_test(
        x_point_train, y_point, x_point_test, groups, class_weight="balanced"
    )

    action_conf, action_margin = _confidence(prob_action)
    point_conf, point_margin = _confidence(prob_point)
    depth_conf, depth_margin = _confidence(prob_depth)
    side_conf, side_margin = _confidence(prob_side)
    test_action_conf, test_action_margin = _confidence(test_prob_action)
    test_point_conf, test_point_margin = _confidence(test_prob_point)
    test_depth_conf, test_depth_margin = _confidence(test_prob_depth)
    test_side_conf, test_side_margin = _confidence(test_prob_side)

    oof_base_cols = ["rally_uid", "strikeNumber", "source_row_id", "target_actionId", "target_pointId"]
    if "match" in train_rows.columns:
        oof_base_cols.insert(1, "match")
    oof = train_rows[oof_base_cols].copy()
    oof["pred_actionId"] = pred_action.astype(int)
    oof["pred_pointId"] = pred_point.astype(int)
    oof["target_point_depth"] = y_depth
    oof["pred_point_depth"] = pred_depth
    oof["target_point_side"] = y_side
    oof["pred_point_side"] = pred_side
    oof["action_confidence"] = action_conf
    oof["point_confidence"] = point_conf
    oof["depth_confidence"] = depth_conf
    oof["side_confidence"] = side_conf
    oof["action_margin"] = action_margin
    oof["point_margin"] = point_margin
    oof["depth_margin"] = depth_margin
    oof["side_margin"] = side_margin
    oof["joint_confidence"] = (action_margin + point_margin + depth_margin + side_margin) / 4.0
    oof = pd.concat([oof.reset_index(drop=True), train_meta.reset_index(drop=True)], axis=1)
    _append_prob_columns(oof, "action", action_classes, prob_action)
    _append_prob_columns(oof, "point", point_classes, prob_point)
    _append_prob_columns(oof, "depth", depth_classes, prob_depth)
    _append_prob_columns(oof, "side", side_classes, prob_side)

    test_pred = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    test_pred["strikeNumber"] = test_rows["strikeNumber"].to_numpy()
    test_pred["pred_actionId"] = test_action.astype(int)
    test_pred["pred_pointId"] = test_point.astype(int)
    test_pred["pred_point_depth"] = test_depth
    test_pred["pred_point_side"] = test_side
    test_pred["action_confidence"] = test_action_conf
    test_pred["point_confidence"] = test_point_conf
    test_pred["depth_confidence"] = test_depth_conf
    test_pred["side_confidence"] = test_side_conf
    test_pred["action_margin"] = test_action_margin
    test_pred["point_margin"] = test_point_margin
    test_pred["depth_margin"] = test_depth_margin
    test_pred["side_margin"] = test_side_margin
    test_pred["joint_confidence"] = (
        test_action_margin + test_point_margin + test_depth_margin + test_side_margin
    ) / 4.0
    test_pred = pd.concat([test_pred.reset_index(drop=True), test_meta.reset_index(drop=True)], axis=1)
    _append_prob_columns(test_pred, "action", action_classes, test_prob_action)
    _append_prob_columns(test_pred, "point", point_classes, test_prob_point)
    _append_prob_columns(test_pred, "depth", depth_classes, test_prob_depth)
    _append_prob_columns(test_pred, "side", side_classes, test_prob_side)

    outdir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(safe_output_path(outdir, "oof_predictions.csv"), index=False)
    test_pred.to_csv(safe_output_path(outdir, "test_predictions.csv"), index=False)

    generated = [
        _write_candidate(
            anchor=anchor,
            predictions=test_pred,
            outdir=outdir,
            filename="submission_v419_intent_point_top5__v362anchor.csv",
            candidate="intent_point_top5",
            expected_rows=expected_rows,
            point_limit=5,
        ),
        _write_candidate(
            anchor=anchor,
            predictions=test_pred,
            outdir=outdir,
            filename="submission_v419_intent_point_top10__v362anchor.csv",
            candidate="intent_point_top10",
            expected_rows=expected_rows,
            point_limit=10,
        ),
        _write_candidate(
            anchor=anchor,
            predictions=test_pred,
            outdir=outdir,
            filename="submission_v419_joint_top10__v362anchor.csv",
            candidate="joint_top10",
            expected_rows=expected_rows,
            joint_limit=10,
        ),
    ]
    pd.DataFrame(generated).to_csv(safe_output_path(outdir, "candidate_summary.csv"), index=False)

    emb_cols = [
        col
        for col in token_embeddings.columns
        if col != "token" and pd.api.types.is_numeric_dtype(token_embeddings[col])
    ]
    report = {
        "version": "V419",
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows)),
        "test_observed_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "token_embedding_path": str(Path(resolved_embedding_path).resolve()),
        "fallback_used": bool(fallback_used),
        "embedding_dimensions": int(len(emb_cols)),
        "feature_columns": int(x_train_base.shape[1]),
        "point_conditioned_feature_columns": int(x_point_train.shape[1]),
        "action_accuracy": float(accuracy_score(y_action, pred_action)),
        "point_accuracy": float(accuracy_score(y_point, pred_point)),
        "depth_accuracy": float(accuracy_score(y_depth, pred_depth)),
        "side_accuracy": float(accuracy_score(y_side, pred_side)),
        "action_macro_f1": float(f1_score(y_action, pred_action, average="macro", zero_division=0)),
        "point_macro_f1": float(f1_score(y_point, pred_point, average="macro", zero_division=0)),
        "action_log_loss": _safe_log_loss(y_action, prob_action, action_classes),
        "point_log_loss": _safe_log_loss(y_point, prob_point, point_classes),
        "action_classes": [int(value) for value in action_classes],
        "point_classes": [int(value) for value in point_classes],
        "depth_classes": [str(value) for value in depth_classes],
        "side_classes": [str(value) for value in side_classes],
        "generated_submissions": generated,
        "blocked_point0_additions": int(max(item["blocked_point0_additions"] for item in generated)),
        "blocked_serve_15_18_additions": int(max(item["blocked_serve_15_18_additions"] for item in generated)),
    }
    write_json(safe_output_path(outdir, "training_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
