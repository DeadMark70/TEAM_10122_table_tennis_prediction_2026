"""V432 AICUP exact fine-tune model zoo.

Builds next-row AICUP transition targets, aligns test prefix rows to the clean
anchor order, consumes the newest available external token/sequence embeddings,
and exports probability diagnostics only. V435 owns submission packaging.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v432_aicup_exact_model_zoo_finetune"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
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
LEAK_COLUMNS = {
    "target_actionId",
    "target_pointId",
    "target_serverGetPoint",
    "serverGetPoint",
    "future_actionId",
    "future_pointId",
}
SERVE_ACTION_CLASSES = {15, 16, 17, 18}


@dataclass(frozen=True)
class EmbeddingSource:
    name: str
    source_version: str
    token_path: Path
    sequence_path: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    max_iter: int = 300
    n_estimators: int = 80
    max_depth: int | None = 12
    min_samples_leaf: int = 2


class ConstantClassifier:
    def __init__(self, value: Any):
        self.classes_ = np.array([value])
        self.value = value

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.ones((len(x), 1), dtype=float)


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


def _safe_suffix(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return text or "blank"


def _normalise_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, None)
    row_sum = prob.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 0
    if empty.any() and prob.shape[1]:
        prob[empty, :] = 1.0 / prob.shape[1]
        row_sum = prob.sum(axis=1, keepdims=True)
    return prob / np.maximum(row_sum, 1e-12)


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
    if "serverGetPoint" in rows.columns:
        rows["target_serverGetPoint"] = rows.groupby("rally_uid", sort=False)["serverGetPoint"].shift(-1)
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
    if value == 0:
        return "terminal_or_unknown"
    if value in {1, 2, 5, 6, 7}:
        return "badminton_drive"
    if value in {3, 14}:
        return "badminton_attack"
    if value in {4, 8, 9, 10, 11}:
        return "badminton_net_shot"
    if value in {12, 13}:
        return "badminton_defensive_shot"
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


def phase_bin(row: pd.Series) -> str:
    action = _int_or_none(row.get("actionId"))
    if action in SERVE_ACTION_CLASSES:
        return "serve"
    return prefix_len_bin(row.get("strikeNumber"))


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
    return f"fam={action_family(action_id)}"


def external_phase_token(strike_number: Any, action_id: Any) -> str:
    if _int_or_none(action_id) in SERVE_ACTION_CLASSES:
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


def build_intent_features(rows: pd.DataFrame) -> pd.DataFrame:
    safe_rows = rows.drop(columns=[col for col in LEAK_COLUMNS if col in rows.columns], errors="ignore").copy()
    feature = pd.DataFrame(index=safe_rows.index)
    for col in NUMERIC_CONTEXT_COLUMNS:
        source = safe_rows[col] if col in safe_rows.columns else pd.Series(0, index=safe_rows.index)
        feature[f"num_{col}"] = pd.to_numeric(source, errors="coerce").fillna(0.0).astype(float)
    categorical = pd.DataFrame(
        {
            "intent_family": safe_rows.apply(lambda row: action_family(row.get("actionId")), axis=1),
            "phase": safe_rows.apply(phase_bin, axis=1),
            "prefix_len": safe_rows.apply(lambda row: prefix_len_bin(row.get("strikeNumber")), axis=1),
            "point_depth": safe_rows.apply(lambda row: point_depth(row.get("pointId")), axis=1),
            "point_side": safe_rows.apply(lambda row: point_side(row.get("pointId")), axis=1),
            "terminal": safe_rows.apply(lambda row: terminal_intent(row.get("pointId")), axis=1),
        },
        index=safe_rows.index,
    )
    dummies = pd.get_dummies(categorical, prefix=categorical.columns, dtype=float)
    feature = pd.concat([feature, dummies], axis=1)
    feature["intent_is_serve"] = categorical["intent_family"].eq("serve").astype(float).to_numpy()
    feature["intent_is_terminal"] = categorical["terminal"].eq("terminal").astype(float).to_numpy()
    feature["point_depth_ord"] = categorical["point_depth"].map({"short": 1.0, "half": 2.0, "long": 3.0}).fillna(0.0)
    feature["point_side_ord"] = categorical["point_side"].map({"left": 1.0, "middle": 2.0, "right": 3.0}).fillna(0.0)
    return feature.reset_index(drop=True)


def _load_embedding_lookup(token_embeddings: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[str]]:
    if token_embeddings.empty:
        return {}, []
    token_col = "token" if "token" in token_embeddings.columns else token_embeddings.columns[0]
    emb_cols = [
        col for col in token_embeddings.columns if col != token_col and pd.api.types.is_numeric_dtype(token_embeddings[col])
    ]
    lookup = {
        str(row[token_col]): row[emb_cols].to_numpy(dtype=float)
        for _, row in token_embeddings[[token_col] + emb_cols].iterrows()
    }
    return lookup, emb_cols


def _sequence_embedding_mean(sequence_path: Path | None) -> np.ndarray:
    if sequence_path is None or not sequence_path.exists():
        return np.zeros(0, dtype=float)
    frame = pd.read_csv(sequence_path)
    emb_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col]) and col != "sequence_id"]
    if not emb_cols:
        return np.zeros(0, dtype=float)
    return frame[emb_cols].mean(axis=0).to_numpy(dtype=float)


def build_feature_frame(
    rows: pd.DataFrame,
    token_embeddings: pd.DataFrame,
    *,
    sequence_embedding_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup, emb_cols = _load_embedding_lookup(token_embeddings)
    dim = len(emb_cols)
    feature = build_intent_features(rows)
    seq_mean = _sequence_embedding_mean(sequence_embedding_path)
    meta_rows: list[dict[str, Any]] = []
    mean_vectors: list[np.ndarray] = []
    sum_vectors: list[np.ndarray] = []
    for _, row in rows.iterrows():
        tokens = derive_tokens(row)
        vectors = [lookup.get(token, np.zeros(dim, dtype=float)) for token in tokens]
        matrix = np.vstack(vectors) if vectors and dim else np.zeros((len(tokens), dim), dtype=float)
        mean_vectors.append(matrix.mean(axis=0) if dim else np.zeros(0, dtype=float))
        sum_vectors.append(matrix.sum(axis=0) if dim else np.zeros(0, dtype=float))
        meta_rows.append({"tokens_json": json.dumps(tokens), "matched_token_count": int(sum(token in lookup for token in tokens))})
    for idx in range(dim):
        feature[f"token_emb_mean_{idx}"] = [float(vec[idx]) for vec in mean_vectors]
        feature[f"token_emb_sum_{idx}"] = [float(vec[idx]) for vec in sum_vectors]
    for idx, value in enumerate(seq_mean):
        feature[f"sequence_emb_global_mean_{idx}"] = float(value)
    return feature.reset_index(drop=True), pd.DataFrame(meta_rows)


def align_feature_columns(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(x_train.columns) | set(x_test.columns))
    return (
        x_train.reindex(columns=columns, fill_value=0.0).astype(float),
        x_test.reindex(columns=columns, fill_value=0.0).astype(float),
    )


def discover_embedding_sources(root: Path = ROOT) -> list[EmbeddingSource]:
    root = Path(root)
    v431_dir = root / "v431_external_sequence_model_zoo"
    sources: list[EmbeddingSource] = []
    if v431_dir.exists():
        for token_path in sorted(v431_dir.glob("*/token_embeddings.csv")):
            model_dir = token_path.parent
            sequence_path = model_dir / "sequence_embeddings.csv"
            sources.append(
                EmbeddingSource(
                    name=model_dir.name,
                    source_version="V431",
                    token_path=token_path,
                    sequence_path=sequence_path if sequence_path.exists() else None,
                )
            )
    if sources:
        return sources
    v418 = root / "v418_clean_external_sequence_pretrain" / "token_embeddings.csv"
    if v418.exists():
        seq = v418.parent / "sequence_embeddings.csv"
        return [EmbeddingSource("v418", "V418", v418, seq if seq.exists() else None)]
    v415 = root / "v415_clean_external_representation" / "token_embeddings.csv"
    if v415.exists():
        seq = v415.parent / "sequence_embeddings.csv"
        return [EmbeddingSource("v415", "V415", v415, seq if seq.exists() else None)]
    raise FileNotFoundError("no V431, V418, or V415 token embeddings found")


def build_model_registry(*, quick: bool = False) -> dict[str, ModelConfig]:
    if quick:
        return {
            "logistic": ModelConfig("logistic", "logistic", max_iter=120),
            "extratrees": ModelConfig("extratrees", "extratrees", n_estimators=30, max_depth=8, min_samples_leaf=1),
            "sgd_log": ModelConfig("sgd_log", "sgd", max_iter=120),
        }
    return {
        "logistic": ModelConfig("logistic", "logistic", max_iter=500),
        "extratrees": ModelConfig("extratrees", "extratrees", n_estimators=120, max_depth=14, min_samples_leaf=2),
        "randomforest": ModelConfig("randomforest", "randomforest", n_estimators=90, max_depth=14, min_samples_leaf=2),
        "sgd_log": ModelConfig("sgd_log", "sgd", max_iter=700),
        "mlp_tiny": ModelConfig("mlp_tiny", "mlp", max_iter=90),
    }


def _sort_labels(values: Iterable[Any]) -> list[Any]:
    values = list(values)
    if all(isinstance(value, (int, np.integer)) for value in values):
        return sorted(values, key=int)
    return sorted(values, key=lambda value: str(value))


def _splitter(y: np.ndarray, groups: pd.Series | None, seed: int = 432) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    indices = np.arange(len(y))
    if len(y) < 2:
        return [(indices, indices)]
    if groups is not None and groups.nunique(dropna=True) >= 2:
        n_splits = int(min(3, groups.nunique(dropna=True)))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    class_counts = pd.Series(y).value_counts()
    min_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_count >= 2:
        n_splits = int(min(3, min_count))
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return [(indices, indices)]


def _fit_model(x: pd.DataFrame, y: np.ndarray, config: ModelConfig) -> Any:
    unique = pd.unique(pd.Series(y))
    if len(unique) <= 1:
        return ConstantClassifier(unique[0])
    if config.family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=config.max_iter, class_weight="balanced", random_state=432),
        ).fit(x, y)
    if config.family == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=432,
            n_jobs=1,
        ).fit(x, y)
    if config.family == "randomforest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=432,
            n_jobs=1,
        ).fit(x, y)
    if config.family == "sgd":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                max_iter=config.max_iter,
                tol=1e-3,
                class_weight="balanced",
                random_state=432,
            ),
        ).fit(x, y)
    if config.family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=1e-3,
                max_iter=config.max_iter,
                early_stopping=True,
                random_state=432,
            ),
        ).fit(x, y)
    raise ValueError(f"unknown model family: {config.family}")


def _predict_proba_aligned(model: Any, x: pd.DataFrame, classes: list[Any]) -> np.ndarray:
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    local = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=float)
    for local_idx, label in enumerate(model.classes_):
        out[:, class_to_index[label]] = local[:, local_idx]
    return _normalise_prob(out)


def fit_oof_and_test(
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    groups: pd.Series | None,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    classes = _sort_labels(pd.unique(pd.Series(y)).tolist())
    oof_prob = np.zeros((len(y), len(classes)), dtype=float)
    for train_idx, valid_idx in _splitter(y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _fit_model(x_train.iloc[train_idx], y[train_idx], config)
        oof_prob[valid_idx] = _predict_proba_aligned(model, x_train.iloc[valid_idx], classes)
    missing = oof_prob.sum(axis=1) <= 0
    if missing.any() and classes:
        oof_prob[missing] = 1.0 / len(classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _fit_model(x_train, y, config)
    test_prob = _predict_proba_aligned(model, x_test, classes)
    oof_pred = np.array([classes[idx] for idx in oof_prob.argmax(axis=1)])
    test_pred = np.array([classes[idx] for idx in test_prob.argmax(axis=1)])
    return oof_pred, oof_prob, test_pred, test_prob, classes


def _append_prob_columns(frame: pd.DataFrame, prefix: str, classes: list[Any], prob: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for idx, label in enumerate(classes):
        out[f"{prefix}_prob_{_safe_suffix(label)}"] = prob[:, idx]
    return out


def _probability_frame(
    base: pd.DataFrame,
    *,
    target: str,
    classes: list[Any],
    prob: np.ndarray,
    pred: np.ndarray,
    target_values: np.ndarray | None = None,
) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    if target_values is not None:
        out[f"target_{target}"] = target_values
    out[f"pred_{target}"] = pred
    out[f"{target}_confidence"] = prob.max(axis=1) if prob.shape[1] else 0.0
    if prob.shape[1] > 1:
        sorted_prob = np.sort(prob, axis=1)
        out[f"{target}_margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    else:
        out[f"{target}_margin"] = out[f"{target}_confidence"]
    return _append_prob_columns(out, target, classes, prob)


def _conditioning_frame(
    action_classes: list[Any],
    action_prob: np.ndarray,
    family_classes: list[Any],
    family_prob: np.ndarray,
    depth_classes: list[Any],
    depth_prob: np.ndarray,
    side_classes: list[Any],
    side_prob: np.ndarray,
    terminal_classes: list[Any],
    terminal_prob: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(action_prob.shape[0]))
    for prefix, classes, prob in [
        ("cond_action", action_classes, action_prob),
        ("cond_family", family_classes, family_prob),
        ("cond_depth", depth_classes, depth_prob),
        ("cond_side", side_classes, side_prob),
        ("cond_terminal", terminal_classes, terminal_prob),
    ]:
        for idx, label in enumerate(classes):
            out[f"{prefix}_prob_{_safe_suffix(label)}"] = prob[:, idx]
    return out.reset_index(drop=True)


def _safe_log_loss(y_true: np.ndarray, prob: np.ndarray, labels: list[Any]) -> float | None:
    try:
        return float(log_loss(y_true, prob, labels=labels))
    except ValueError:
        return None


def _slice_metrics(base: pd.DataFrame, y_action: np.ndarray, pred_action: np.ndarray, y_point: np.ndarray, pred_point: np.ndarray) -> pd.DataFrame:
    frame = base.copy().reset_index(drop=True)
    frame["prefix_len_bin"] = frame["strikeNumber"].map(prefix_len_bin)
    frame["phase_bin"] = frame.apply(phase_bin, axis=1)
    rows: list[dict[str, Any]] = []
    for slice_name, col in [("prefix_len", "prefix_len_bin"), ("phase", "phase_bin")]:
        for value, idx in frame.groupby(col, sort=True).groups.items():
            index = np.asarray(list(idx), dtype=int)
            rows.append(
                {
                    "slice": slice_name,
                    "value": value,
                    "rows": int(len(index)),
                    "action_macro_f1": float(f1_score(y_action[index], pred_action[index], average="macro", zero_division=0)),
                    "point_macro_f1": float(f1_score(y_point[index], pred_point[index], average="macro", zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def _write_probs(path_base: Path, frame: pd.DataFrame, prob: np.ndarray) -> None:
    frame.to_csv(path_base.with_suffix(".csv"), index=False)
    np.save(path_base.with_suffix(".npy"), prob)


def train_one_source_model(
    *,
    source: EmbeddingSource,
    config: ModelConfig,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    outdir: Path,
) -> dict[str, Any]:
    token_embeddings = pd.read_csv(source.token_path)
    x_train_base, train_meta = build_feature_frame(train_rows, token_embeddings, sequence_embedding_path=source.sequence_path)
    x_test_base, test_meta = build_feature_frame(test_rows, token_embeddings, sequence_embedding_path=source.sequence_path)
    x_train_base, x_test_base = align_feature_columns(x_train_base, x_test_base)
    groups = train_rows["match"] if "match" in train_rows.columns else None

    y_action = train_rows["target_actionId"].to_numpy(dtype=int)
    y_point = train_rows["target_pointId"].to_numpy(dtype=int)
    y_family = np.array([action_family(value) for value in y_action], dtype=object)
    y_depth = np.array([point_depth(value) for value in y_point], dtype=object)
    y_side = np.array([point_side(value) for value in y_point], dtype=object)
    y_terminal = np.array([terminal_intent(value) for value in y_point], dtype=object)

    pred_action, prob_action, test_action, test_prob_action, action_classes = fit_oof_and_test(
        x_train_base, y_action, x_test_base, groups, config
    )
    pred_family, prob_family, test_family, test_prob_family, family_classes = fit_oof_and_test(
        x_train_base, y_family, x_test_base, groups, config
    )
    pred_depth, prob_depth, test_depth, test_prob_depth, depth_classes = fit_oof_and_test(
        x_train_base, y_depth, x_test_base, groups, config
    )
    pred_side, prob_side, test_side, test_prob_side, side_classes = fit_oof_and_test(
        x_train_base, y_side, x_test_base, groups, config
    )
    pred_terminal, prob_terminal, test_terminal, test_prob_terminal, terminal_classes = fit_oof_and_test(
        x_train_base, y_terminal, x_test_base, groups, config
    )

    train_cond = _conditioning_frame(
        action_classes,
        prob_action,
        family_classes,
        prob_family,
        depth_classes,
        prob_depth,
        side_classes,
        prob_side,
        terminal_classes,
        prob_terminal,
    )
    test_cond = _conditioning_frame(
        action_classes,
        test_prob_action,
        family_classes,
        test_prob_family,
        depth_classes,
        test_prob_depth,
        side_classes,
        test_prob_side,
        terminal_classes,
        test_prob_terminal,
    )
    x_point_train, x_point_test = align_feature_columns(
        pd.concat([x_train_base.reset_index(drop=True), train_cond], axis=1),
        pd.concat([x_test_base.reset_index(drop=True), test_cond], axis=1),
    )
    pred_point, prob_point, test_point, test_prob_point, point_classes = fit_oof_and_test(
        x_point_train, y_point, x_point_test, groups, config
    )

    slug = f"{source.name.lower()}__{config.name}"
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_").lower()
    source_slug = re.sub(r"[^0-9a-zA-Z]+", "_", source.name).strip("_").lower()
    if source.source_version != "V431":
        source_slug = source.source_version.lower()
        slug = f"{source_slug}__{config.name}"

    oof_base_cols = ["rally_uid", "strikeNumber", "source_row_id"]
    if "match" in train_rows.columns:
        oof_base_cols.insert(1, "match")
    oof_base = pd.concat([train_rows[oof_base_cols].reset_index(drop=True), train_meta.reset_index(drop=True)], axis=1)
    test_base = pd.concat([test_rows[["rally_uid", "strikeNumber"]].reset_index(drop=True), test_meta.reset_index(drop=True)], axis=1)

    _write_probs(
        outdir / f"oof_action_probs_{slug}",
        _probability_frame(oof_base, target="action", classes=action_classes, prob=prob_action, pred=pred_action, target_values=y_action),
        prob_action,
    )
    _write_probs(
        outdir / f"test_action_probs_{slug}",
        _probability_frame(test_base, target="action", classes=action_classes, prob=test_prob_action, pred=test_action),
        test_prob_action,
    )
    _write_probs(
        outdir / f"oof_point_probs_{slug}",
        _probability_frame(oof_base, target="point", classes=point_classes, prob=prob_point, pred=pred_point, target_values=y_point),
        prob_point,
    )
    _write_probs(
        outdir / f"test_point_probs_{slug}",
        _probability_frame(test_base, target="point", classes=point_classes, prob=test_prob_point, pred=test_point),
        test_prob_point,
    )

    slices = _slice_metrics(train_rows, y_action, pred_action.astype(int), y_point, pred_point.astype(int))
    slices.insert(0, "model", config.name)
    slices.insert(0, "embedding_source", source.name)
    slices.to_csv(outdir / f"slice_metrics_{slug}.csv", index=False)

    report = {
        "embedding_source": source.name,
        "source_version": source.source_version,
        "model": config.name,
        "family": config.family,
        "slug": slug,
        "token_embedding_path": str(source.token_path.resolve()),
        "sequence_embedding_path": str(source.sequence_path.resolve()) if source.sequence_path else "",
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "feature_columns": int(x_train_base.shape[1]),
        "point_feature_columns": int(x_point_train.shape[1]),
        "action_accuracy": float(accuracy_score(y_action, pred_action)),
        "point_accuracy": float(accuracy_score(y_point, pred_point)),
        "family_accuracy": float(accuracy_score(y_family, pred_family)),
        "depth_accuracy": float(accuracy_score(y_depth, pred_depth)),
        "side_accuracy": float(accuracy_score(y_side, pred_side)),
        "terminal_accuracy": float(accuracy_score(y_terminal, pred_terminal)),
        "action_macro_f1": float(f1_score(y_action, pred_action, average="macro", zero_division=0)),
        "point_macro_f1": float(f1_score(y_point, pred_point, average="macro", zero_division=0)),
        "action_log_loss": _safe_log_loss(y_action, prob_action, action_classes),
        "point_log_loss": _safe_log_loss(y_point, prob_point, point_classes),
        "action_classes": [int(value) for value in action_classes],
        "point_classes": [int(value) for value in point_classes],
        "family_classes": [str(value) for value in family_classes],
        "depth_classes": [str(value) for value in depth_classes],
        "side_classes": [str(value) for value in side_classes],
        "terminal_classes": [str(value) for value in terminal_classes],
    }
    write_json(outdir / f"model_card_{slug}.json", report)
    return report


def _validate_anchor(anchor: pd.DataFrame, expected_rows: int | None) -> pd.DataFrame:
    missing = set(SUBMISSION_COLUMNS) - set(anchor.columns)
    if missing:
        raise ValueError(f"anchor missing submission columns: {sorted(missing)}")
    if expected_rows is not None and len(anchor) != expected_rows:
        raise ValueError(f"anchor row count {len(anchor)} != expected {expected_rows}")
    return anchor.loc[:, SUBMISSION_COLUMNS].copy()


def _bounded_transition_sample(train_rows: pd.DataFrame, max_train_transitions: int | None) -> tuple[pd.DataFrame, bool]:
    if max_train_transitions is None or max_train_transitions <= 0 or len(train_rows) <= max_train_transitions:
        return train_rows.reset_index(drop=True), False
    sampled = train_rows.sample(n=int(max_train_transitions), random_state=432)
    if "source_row_id" in sampled.columns:
        sampled = sampled.sort_values("source_row_id", kind="mergesort")
    else:
        sampled = sampled.sort_index(kind="mergesort")
    return sampled.reset_index(drop=True), True


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    anchor_path: Path = ANCHOR_PATH,
    root: Path = ROOT,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
    model_names: tuple[str, ...] | None = None,
    quick: bool = False,
    max_train_transitions: int | None = None,
    max_embedding_sources: int | None = None,
) -> dict[str, Any]:
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    anchor = _validate_anchor(pd.read_csv(anchor_path), expected_rows)
    train_rows_all = build_train_transition_rows(train_raw)
    if quick and max_train_transitions is None:
        max_train_transitions = 6000
    train_rows, train_limited = _bounded_transition_sample(train_rows_all, max_train_transitions)
    test_rows = build_test_rows(test_raw, anchor)

    sources = discover_embedding_sources(root)
    if quick and max_embedding_sources is None:
        max_embedding_sources = 1
    source_limited = False
    if max_embedding_sources is not None and max_embedding_sources > 0 and len(sources) > max_embedding_sources:
        sources = sources[: int(max_embedding_sources)]
        source_limited = True
    registry = build_model_registry(quick=quick)
    selected_model_names = model_names or (("logistic", "extratrees", "sgd_log") if quick else tuple(registry.keys()))
    missing_models = [name for name in selected_model_names if name not in registry]
    if missing_models:
        raise ValueError(f"unknown model names: {missing_models}")

    outdir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for source in sources:
        for name in selected_model_names:
            reports.append(
                train_one_source_model(
                    source=source,
                    config=registry[name],
                    train_rows=train_rows,
                    test_rows=test_rows,
                    outdir=outdir,
                )
            )
    reports_frame = pd.DataFrame(reports).sort_values(["point_macro_f1", "action_macro_f1"], ascending=[False, False])
    reports_frame.to_csv(outdir / "model_reports.csv", index=False)

    best = reports_frame.iloc[0].to_dict() if len(reports_frame) else {}
    summary = {
        "version": "V432",
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows_all)),
        "train_transition_rows_used": int(len(train_rows)),
        "test_observed_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "embedding_sources": [asdict(source) for source in sources],
        "fallback_source": sources[0].source_version if sources and sources[0].source_version != "V431" else "",
        "partial_run": bool(train_limited or source_limited),
        "max_train_transitions": int(max_train_transitions) if max_train_transitions else None,
        "max_embedding_sources": int(max_embedding_sources) if max_embedding_sources else None,
        "model_names": list(selected_model_names),
        "model_count": int(len(reports)),
        "best_model": json_safe(best),
        "submission_exports": 0,
        "probability_exports": int(len(list(outdir.glob("*_probs_*.npy")))),
    }
    write_json(outdir / "summary.json", summary)
    return json_safe(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run bounded quick configs.")
    parser.add_argument("--models", default="", help="Comma-separated model names to run.")
    parser.add_argument(
        "--max-train-transitions",
        type=int,
        default=None,
        help="Cap AICUP transition rows for bounded or partial diagnostics.",
    )
    parser.add_argument(
        "--max-embedding-sources",
        type=int,
        default=None,
        help="Cap embedding sources, useful while V431 is still producing models.",
    )
    args = parser.parse_args()
    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip()) or None
    summary = run_pipeline(
        quick=args.quick,
        model_names=model_names,
        max_train_transitions=args.max_train_transitions,
        max_embedding_sources=args.max_embedding_sources,
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
