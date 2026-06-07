"""V449 intent-conditioned GRU point model.

Trains a bounded GRU point model on AICUP prefix transitions, conditions point
features on observed/predicted action intent without future point leakage, and
packages conservative V362-anchored point residual submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS, safe_output_path, validate_submission_schema
from analysis_v419_intent_first_point_finetune import build_test_rows, build_train_transition_rows


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v449_intent_gru_point_full"

LEAK_COLUMNS = {
    "target_actionId",
    "target_pointId",
    "target_serverGetPoint",
    "serverGetPoint",
    "future_actionId",
    "future_pointId",
    "future_serverGetPoint",
}
NUMERIC_COLUMNS = [
    "sex",
    "match",
    "numberGame",
    "rally_id",
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
    "pred_action",
    "action_confidence",
]
POINT_CLASSES = list(range(10))
DEPTH_CLASSES = ["zero", "short", "half", "long"]
INTENT_CLASSES = ["terminal", "attack", "control", "defense", "serve"]


@dataclass(frozen=True)
class GRUConfig:
    hidden_dim: int = 48
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    dropout: float = 0.20
    feature_mask_probability: float = 0.10
    max_train_transitions: int | None = 6000
    max_sequence_len: int = 10
    folds: int = 2
    seed: int = 449
    device: str = "cpu"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _uid_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def action_intent(action_id: Any) -> str:
    value = _int_or_none(action_id)
    if value is None or value == 0:
        return "terminal"
    if value in {15, 16, 17, 18}:
        return "serve"
    if value in {3, 7, 14}:
        return "attack"
    if value in {12, 13}:
        return "defense"
    return "control"


def point_depth(point_id: Any) -> str:
    value = _int_or_none(point_id)
    if value is None or value == 0:
        return "zero"
    if value in {1, 2, 3}:
        return "short"
    if value in {4, 5, 6}:
        return "half"
    return "long"


def prefix_phase(strike_number: Any) -> str:
    value = _int_or_none(strike_number)
    if value is None:
        return "unknown"
    if value <= 2:
        return "early"
    if value <= 5:
        return "middle"
    return "late"


def point_side(point_id: Any) -> str:
    value = _int_or_none(point_id)
    if value is None or value == 0:
        return "unknown"
    if value in {1, 4, 7}:
        return "left"
    if value in {2, 5, 8}:
        return "middle"
    return "right"


def build_intent_conditioned_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build row-safe GRU input features using current prefix and predicted intent.

    Future target labels are intentionally dropped before any feature is built.
    If ``pred_action`` is absent, the current observed ``actionId`` is used as a
    conservative intent proxy.
    """

    safe = frame.drop(columns=[col for col in LEAK_COLUMNS if col in frame.columns], errors="ignore").copy()
    if "pred_action" not in safe.columns:
        safe["pred_action"] = safe.get("actionId", 0)
    if "action_confidence" not in safe.columns:
        safe["action_confidence"] = 0.75

    features = pd.DataFrame(index=safe.index)
    for col in NUMERIC_COLUMNS:
        source = safe[col] if col in safe.columns else pd.Series(0, index=safe.index)
        features[f"num_{col}"] = pd.to_numeric(source, errors="coerce").fillna(0.0).astype(float)

    categorical = pd.DataFrame(
        {
            "intent": safe["pred_action"].map(action_intent),
            "observed_intent": safe.get("actionId", pd.Series(0, index=safe.index)).map(action_intent),
            "current_depth": safe.get("pointId", pd.Series(0, index=safe.index)).map(point_depth),
            "current_side": safe.get("pointId", pd.Series(0, index=safe.index)).map(point_side),
            "prefix_phase": safe.get("strikeNumber", pd.Series(0, index=safe.index)).map(prefix_phase),
        },
        index=safe.index,
    )
    dummies = pd.get_dummies(categorical.astype(str), prefix=categorical.columns, dtype=float)
    features = pd.concat([features.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    features["intent_is_terminal"] = categorical["intent"].eq("terminal").astype(float).to_numpy()
    features["intent_is_serve"] = categorical["intent"].eq("serve").astype(float).to_numpy()
    features["current_depth_ord"] = categorical["current_depth"].map({"zero": 0.0, "short": 1.0, "half": 2.0, "long": 3.0}).to_numpy()
    return features.drop(columns=[col for col in LEAK_COLUMNS if col in features.columns], errors="ignore").reset_index(drop=True)


def _align_feature_columns(train_features: pd.DataFrame, test_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(train_features.columns) | set(test_features.columns))
    return (
        train_features.reindex(columns=columns, fill_value=0.0).astype(float),
        test_features.reindex(columns=columns, fill_value=0.0).astype(float),
    )


def _standardize(train_features: pd.DataFrame, test_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return (train_features - mean) / std, (test_features - mean) / std, mean, std


def _build_sequence_tensor(rows: pd.DataFrame, features: pd.DataFrame, max_sequence_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) != len(features):
        raise ValueError("rows and features length mismatch")
    work = rows[["rally_uid", "strikeNumber"]].copy()
    work["_row_index"] = np.arange(len(rows), dtype=int)
    work = work.sort_values(["rally_uid", "strikeNumber", "_row_index"], kind="mergesort")
    feature_values = features.to_numpy(dtype=np.float32)
    seq = np.zeros((len(rows), max_sequence_len, features.shape[1]), dtype=np.float32)
    lengths = np.ones(len(rows), dtype=np.int64)
    history: dict[Any, list[int]] = {}
    for _, row in work.iterrows():
        uid = row["rally_uid"]
        idx = int(row["_row_index"])
        indices = (history.get(uid, []) + [idx])[-max_sequence_len:]
        lengths[idx] = len(indices)
        seq[idx, -len(indices):, :] = feature_values[indices]
        history.setdefault(uid, []).append(idx)
    return seq, lengths


class PointSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        y_point: np.ndarray | None = None,
        y_depth: np.ndarray | None = None,
        y_intent: np.ndarray | None = None,
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y_point = None if y_point is None else torch.tensor(y_point, dtype=torch.long)
        self.y_depth = None if y_depth is None else torch.tensor(y_depth, dtype=torch.long)
        self.y_intent = None if y_intent is None else torch.tensor(y_intent, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int):
        if self.y_point is None:
            return self.sequences[idx], self.lengths[idx]
        return self.sequences[idx], self.lengths[idx], self.y_point[idx], self.y_depth[idx], self.y_intent[idx]


class IntentGRUPointModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, point_count: int, depth_count: int, intent_count: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.point_head = nn.Linear(hidden_dim, point_count)
        self.depth_head = nn.Linear(hidden_dim, depth_count)
        self.intent_head = nn.Linear(hidden_dim, intent_count)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _out, hidden = self.gru(x)
        state = self.dropout(hidden[-1])
        return self.point_head(state), self.depth_head(state), self.intent_head(state)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _train_model(
    train_seq: np.ndarray,
    train_len: np.ndarray,
    y_point: np.ndarray,
    y_depth: np.ndarray,
    y_intent: np.ndarray,
    config: GRUConfig,
) -> tuple[IntentGRUPointModel, list[float]]:
    model = IntentGRUPointModel(
        input_dim=train_seq.shape[2],
        hidden_dim=config.hidden_dim,
        point_count=len(POINT_CLASSES),
        depth_count=len(DEPTH_CLASSES),
        intent_count=len(INTENT_CLASSES),
        dropout=config.dropout,
    ).to(config.device)
    point_loss = nn.CrossEntropyLoss(weight=_class_weights(y_point, len(POINT_CLASSES)).to(config.device))
    depth_loss = nn.CrossEntropyLoss(weight=_class_weights(y_depth, len(DEPTH_CLASSES)).to(config.device))
    intent_loss = nn.CrossEntropyLoss(weight=_class_weights(y_intent, len(INTENT_CLASSES)).to(config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        PointSequenceDataset(train_seq, train_len, y_point, y_depth, y_intent),
        batch_size=config.batch_size,
        shuffle=True,
    )
    losses: list[float] = []
    for _epoch in range(config.epochs):
        model.train()
        running = 0.0
        batches = 0
        for x, _lengths, point, depth, intent in loader:
            x = x.to(config.device)
            if config.feature_mask_probability > 0:
                mask = torch.rand_like(x) >= config.feature_mask_probability
                x = x * mask
            point = point.to(config.device)
            depth = depth.to(config.device)
            intent = intent.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            point_logits, depth_logits, intent_logits = model(x)
            loss = point_loss(point_logits, point) + 0.25 * depth_loss(depth_logits, depth) + 0.25 * intent_loss(intent_logits, intent)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        losses.append(running / max(batches, 1))
    return model, losses


def _predict_probs(model: IntentGRUPointModel, seq: np.ndarray, lengths: np.ndarray, config: GRUConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(PointSequenceDataset(seq, lengths), batch_size=config.batch_size, shuffle=False)
    point_probs: list[np.ndarray] = []
    depth_probs: list[np.ndarray] = []
    intent_probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _lengths in loader:
            point_logits, depth_logits, intent_logits = model(x.to(config.device))
            point_probs.append(torch.softmax(point_logits, dim=1).cpu().numpy())
            depth_probs.append(torch.softmax(depth_logits, dim=1).cpu().numpy())
            intent_probs.append(torch.softmax(intent_logits, dim=1).cpu().numpy())
    return np.vstack(point_probs), np.vstack(depth_probs), np.vstack(intent_probs)


def _normalise_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.nan_to_num(np.asarray(prob, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, None)
    row_sum = prob.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 0
    if empty.any() and prob.shape[1]:
        prob[empty, :] = 1.0 / prob.shape[1]
        row_sum = prob.sum(axis=1, keepdims=True)
    return prob / np.maximum(row_sum, 1e-12)


def _splitter(y: np.ndarray, groups: pd.Series | None, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y), dtype=int)
    if len(y) < 2:
        return [(indices, indices)]
    if groups is not None and groups.nunique(dropna=True) >= 2:
        n_splits = min(folds, int(groups.nunique(dropna=True)))
        if n_splits >= 2:
            return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    counts = pd.Series(y).value_counts()
    if len(counts) and counts.min() >= 2:
        n_splits = min(folds, int(counts.min()))
        if n_splits >= 2:
            return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return [(indices, indices)]


def _prob_frame(base: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    out = base.reset_index(drop=True).copy()
    pred = np.asarray(POINT_CLASSES, dtype=int)[prob.argmax(axis=1)]
    out["pred_pointId"] = pred.astype(int)
    out["point_confidence"] = prob.max(axis=1)
    sorted_prob = np.sort(prob, axis=1)
    out["point_margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    for idx, label in enumerate(POINT_CLASSES):
        out[f"prob_{label}"] = prob[:, idx]
    return out


def point_residual_candidates_from_probs(
    anchor: pd.DataFrame,
    probs: pd.DataFrame,
    *,
    block_point0_additions: bool = True,
) -> pd.DataFrame:
    if "rally_uid" not in anchor.columns or "pointId" not in anchor.columns:
        raise ValueError("anchor must include rally_uid and pointId")
    if "rally_uid" not in probs.columns:
        raise ValueError("probability frame must include rally_uid")
    prob_cols = [col for col in probs.columns if col.startswith("prob_") or col.startswith("point_prob_")]
    if not prob_cols:
        raise ValueError("probability frame must include prob_* columns")
    aligned = anchor[["rally_uid", "pointId"]].merge(probs, on="rally_uid", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for idx, row in aligned.iterrows():
        anchor_point = int(row["pointId"])
        scores: list[tuple[int, float]] = []
        for col in prob_cols:
            try:
                candidate = int(str(col).split("_")[-1])
            except ValueError:
                continue
            if candidate < 0 or candidate > 9:
                continue
            prob = float(pd.to_numeric(row[col], errors="coerce")) if pd.notna(row[col]) else 0.0
            scores.append((candidate, prob))
        if not scores:
            continue
        anchor_prob = next((prob for candidate, prob in scores if candidate == anchor_point), 0.0)
        for candidate, prob in sorted(scores, key=lambda item: item[1], reverse=True):
            if candidate == anchor_point:
                continue
            if block_point0_additions and candidate == 0 and anchor_point != 0:
                continue
            rows.append(
                {
                    "rally_uid": row["rally_uid"],
                    "row_id": int(idx),
                    "anchor_pointId": anchor_point,
                    "candidate_pointId": int(candidate),
                    "candidate_probability": float(prob),
                    "anchor_probability": float(anchor_prob),
                    "probability_delta": float(prob - anchor_prob),
                    "utility": float(prob - anchor_prob + 0.25 * prob),
                    "source": "v449_intent_gru_point_full",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "rally_uid",
                "row_id",
                "anchor_pointId",
                "candidate_pointId",
                "candidate_probability",
                "anchor_probability",
                "probability_delta",
                "utility",
                "source",
            ]
        )
    return pd.DataFrame(rows).sort_values(["utility", "candidate_probability", "row_id"], ascending=[False, False, True]).reset_index(drop=True)


def _package_point_submission(
    anchor: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    top_k: int,
    expected_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    applied = 0
    skipped_duplicate = 0
    seen: set[str] = set()
    uid_to_idx = {_uid_key(uid): idx for idx, uid in enumerate(out["rally_uid"])}
    for row in candidates.itertuples(index=False):
        if applied >= top_k:
            break
        uid = _uid_key(row.rally_uid)
        if uid in seen:
            skipped_duplicate += 1
            continue
        seen.add(uid)
        if uid not in uid_to_idx:
            continue
        idx = uid_to_idx[uid]
        old_value = int(out.at[idx, "pointId"])
        new_value = int(row.candidate_pointId)
        if new_value == old_value or (new_value == 0 and old_value != 0):
            continue
        out.at[idx, "pointId"] = new_value
        applied += 1
    validate_submission_schema(out, expected_rows=expected_rows)
    point0_additions = out["pointId"].astype(int).eq(0) & anchor["pointId"].astype(int).ne(0)
    server_preserved = np.allclose(
        pd.to_numeric(out["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(anchor["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
    )
    action_preserved = out["actionId"].astype(int).equals(anchor["actionId"].astype(int))
    return out, {
        "top_k": int(top_k),
        "candidate_rows": int(len(candidates)),
        "applied_changes": int(applied),
        "skipped_duplicate_rally_uid": int(skipped_duplicate),
        "point0_additions": int(point0_additions.sum()),
        "server_preserved": bool(server_preserved),
        "action_preserved": bool(action_preserved),
    }


def _bounded_rows(train_rows: pd.DataFrame, config: GRUConfig) -> pd.DataFrame:
    if config.max_train_transitions is None or len(train_rows) <= config.max_train_transitions:
        return train_rows.reset_index(drop=True)
    sampled = train_rows.sample(n=int(config.max_train_transitions), random_state=config.seed)
    return sampled.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").reset_index(drop=True)


def _target_arrays(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_point = rows["target_pointId"].astype(int).clip(0, 9).to_numpy()
    depth_lookup = {label: idx for idx, label in enumerate(DEPTH_CLASSES)}
    intent_lookup = {label: idx for idx, label in enumerate(INTENT_CLASSES)}
    y_depth = np.array([depth_lookup[point_depth(value)] for value in y_point], dtype=int)
    y_intent = np.array([intent_lookup[action_intent(value)] for value in rows["target_actionId"]], dtype=int)
    return y_point, y_depth, y_intent


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
    expected_rows: int | None = 1845,
    quick: bool = False,
) -> dict[str, Any]:
    config = GRUConfig() if quick else GRUConfig(hidden_dim=64, epochs=5, max_train_transitions=12000, folds=3)
    _set_seed(config.seed)
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    anchor = pd.read_csv(anchor_path).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(anchor, expected_rows=expected_rows)

    train_rows_all = build_train_transition_rows(train_raw)
    train_rows = _bounded_rows(train_rows_all, config)
    test_rows = build_test_rows(test_raw, anchor)
    train_rows = train_rows.assign(pred_action=train_rows["actionId"], action_confidence=0.75)
    test_rows = test_rows.assign(pred_action=test_rows["actionId"], action_confidence=0.75)

    train_features = build_intent_conditioned_sequence_features(train_rows)
    test_features = build_intent_conditioned_sequence_features(test_rows)
    train_features, test_features = _align_feature_columns(train_features, test_features)
    train_features, test_features, _mean, _std = _standardize(train_features, test_features)
    train_seq, train_lengths = _build_sequence_tensor(train_rows, train_features, config.max_sequence_len)
    test_seq, test_lengths = _build_sequence_tensor(test_rows, test_features, config.max_sequence_len)
    y_point, y_depth, y_intent = _target_arrays(train_rows)

    groups = train_rows["match"] if "match" in train_rows.columns else None
    oof_prob = np.zeros((len(train_rows), len(POINT_CLASSES)), dtype=float)
    fold_reports: list[dict[str, Any]] = []
    for fold, (fit_idx, valid_idx) in enumerate(_splitter(y_point, groups, config.folds, config.seed), start=1):
        fold_config = GRUConfig(**{**config.__dict__, "epochs": max(1, config.epochs - 1)})
        model, losses = _train_model(
            train_seq[fit_idx],
            train_lengths[fit_idx],
            y_point[fit_idx],
            y_depth[fit_idx],
            y_intent[fit_idx],
            fold_config,
        )
        valid_prob, _valid_depth, _valid_intent = _predict_probs(model, train_seq[valid_idx], train_lengths[valid_idx], config)
        oof_prob[valid_idx] = valid_prob
        fold_reports.append({"fold": fold, "train_rows": int(len(fit_idx)), "valid_rows": int(len(valid_idx)), "losses": losses})
    missing = oof_prob.sum(axis=1) <= 0
    if missing.any():
        oof_prob[missing, :] = 1.0 / len(POINT_CLASSES)
    oof_prob = _normalise_prob(oof_prob)

    final_model, train_losses = _train_model(train_seq, train_lengths, y_point, y_depth, y_intent, config)
    test_prob, test_depth_prob, test_intent_prob = _predict_probs(final_model, test_seq, test_lengths, config)
    test_prob = _normalise_prob(test_prob)
    test_depth_prob = _normalise_prob(test_depth_prob)
    test_intent_prob = _normalise_prob(test_intent_prob)
    oof_frame = _prob_frame(
        train_rows[["rally_uid", "strikeNumber", "source_row_id"]].copy(),
        oof_prob,
    )
    test_frame = _prob_frame(anchor[["rally_uid"]].copy(), test_prob)
    for idx, label in enumerate(DEPTH_CLASSES):
        test_frame[f"depth_prob_{label}"] = test_depth_prob[:, idx]
    for idx, label in enumerate(INTENT_CLASSES):
        test_frame[f"intent_prob_{label}"] = test_intent_prob[:, idx]

    oof_path = safe_output_path(outdir, "oof_point_probs_gru.csv")
    test_path_out = safe_output_path(outdir, "test_point_probs_gru.csv")
    oof_frame.to_csv(oof_path, index=False)
    test_frame.to_csv(test_path_out, index=False)
    np.save(outdir / "oof_point_probs_gru.npy", oof_prob)
    np.save(outdir / "test_point_probs_gru.npy", test_prob)

    candidates = point_residual_candidates_from_probs(anchor, test_frame, block_point0_additions=True)
    candidates.to_csv(safe_output_path(outdir, "point_candidate_table.csv"), index=False)

    exports: list[dict[str, Any]] = []
    for top_k in (5, 10):
        submission, report = _package_point_submission(anchor, candidates, top_k=top_k, expected_rows=expected_rows)
        filename = f"submission_v449_point_top{top_k}__v362anchor.csv"
        path = safe_output_path(outdir, filename)
        submission.to_csv(path, index=False)
        report["filename"] = filename
        report["path"] = str(path.resolve())
        exports.append(report)

    pred_oof = np.asarray(POINT_CLASSES, dtype=int)[oof_prob.argmax(axis=1)]
    summary = {
        "version": "V449",
        "quick": bool(quick),
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows_all)),
        "train_transition_rows_used": int(len(train_rows)),
        "test_observed_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "feature_columns": int(train_features.shape[1]),
        "sequence_len": int(config.max_sequence_len),
        "oof_point_accuracy": float(accuracy_score(y_point, pred_oof)),
        "oof_point_macro_f1": float(f1_score(y_point, pred_oof, average="macro", zero_division=0)),
        "oof_point_log_loss": float(log_loss(y_point, oof_prob, labels=POINT_CLASSES)),
        "fold_reports": fold_reports,
        "final_train_losses": train_losses,
        "candidate_rows": int(len(candidates)),
        "exports": exports,
        "point0_additions": int(max(item["point0_additions"] for item in exports) if exports else 0),
        "server_preserved": bool(all(item["server_preserved"] for item in exports)),
        "action_preserved": bool(all(item["action_preserved"] for item in exports)),
    }
    write_json(safe_output_path(outdir, "summary.json"), summary)
    return _json_safe(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run bounded quick GRU training.")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(quick=args.quick), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
