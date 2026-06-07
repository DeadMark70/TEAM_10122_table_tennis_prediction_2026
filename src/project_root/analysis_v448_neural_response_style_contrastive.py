"""V448 neural response-style contrastive features.

Builds safe row-aligned response-style embeddings from observed train/test
prefix rows. Raw player identifiers are used only to create internal hashed
actor clusters for contrastive pair labels; exported artifacts contain only
allowed style features and no submission CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
OUTDIR = ROOT / "v448_neural_response_style_contrastive"

FAMILY_ORDER = ["serve", "attack", "control", "defense", "other", "zero", "missing"]
PHASE_ORDER = ["serve", "receive", "third_ball", "fourth_ball", "rally"]
DEPTH_ORDER = ["zero", "short", "half", "long", "missing"]
FORBIDDEN_EXPORT_COLUMNS = {
    "gamePlayerId",
    "gamePlayerOtherId",
    "target_actionId",
    "target_pointId",
    "target_serverGetPoint",
    "next_actionId",
    "next_pointId",
    "next_serverGetPoint",
    "actor_cluster",
    "actor_key",
    "row_pos",
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _series(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _stable_hash(value: Any, modulo: int = 997) -> int:
    text = "" if pd.isna(value) else str(value)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(modulo)


def _action_family(value: Any) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        text = "" if value is None else str(value).strip().lower()
        if "serve" in text:
            return "serve"
        if any(token in text for token in ("attack", "drive", "smash", "loop")):
            return "attack"
        if any(token in text for token in ("control", "push", "drop", "short")):
            return "control"
        if any(token in text for token in ("defense", "defensive", "lob", "chop")):
            return "defense"
        return "missing"
    action = int(parsed)
    if action == 0:
        return "zero"
    if action in {15, 16, 17, 18}:
        return "serve"
    if action in {1, 2, 3, 4, 5}:
        return "control"
    if action in {6, 7, 8, 9, 10, 11, 12}:
        return "attack"
    if action in {13, 14}:
        return "defense"
    return "other"


def _point_depth(value: Any) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "missing"
    point = int(parsed)
    if point == 0:
        return "zero"
    if 1 <= point <= 3:
        return "short"
    if 4 <= point <= 6:
        return "half"
    if 7 <= point <= 9:
        return "long"
    return "missing"


def _phase_from_strike(strike_number: Any) -> str:
    parsed = pd.to_numeric(pd.Series([strike_number]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "rally"
    strike = int(parsed)
    if strike <= 1:
        return "serve"
    if strike == 2:
        return "receive"
    if strike == 3:
        return "third_ball"
    if strike == 4:
        return "fourth_ball"
    return "rally"


def _normalize_phase(value: Any, strike_number: Any) -> str:
    if value is None or pd.isna(value):
        return _phase_from_strike(strike_number)
    text = str(value).strip().lower()
    if "serve" in text:
        return "serve"
    if "receive" in text:
        return "receive"
    if "third" in text:
        return "third_ball"
    if "fourth" in text:
        return "fourth_ball"
    return "rally" if text else _phase_from_strike(strike_number)


def build_response_context_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build internal response context rows from observed prefixes.

    The returned table may contain internal hashed actor clusters for training.
    Export functions intentionally strip those columns.
    """

    out = pd.DataFrame(index=frame.index)
    out["row_pos"] = np.arange(len(frame), dtype=int)
    out["rally_uid"] = _series(frame, "rally_uid", np.arange(len(frame))).astype(str)
    out["strikeNumber"] = pd.to_numeric(_series(frame, "strikeNumber", 1), errors="coerce").fillna(1).astype(int)
    actor = _series(frame, "gamePlayerId", _series(frame, "gamePlayerOtherId", "unknown")).astype(str).fillna("unknown")
    out["actor_cluster"] = actor.map(lambda value: _stable_hash(value, modulo=4096)).astype(int)
    phase_source = frame["phase"] if "phase" in frame.columns else pd.Series([None] * len(frame), index=frame.index)
    out["phase"] = [_normalize_phase(value, strike) for value, strike in zip(phase_source, out["strikeNumber"])]
    out["action_family"] = _series(frame, "actionId", pd.NA).map(_action_family)
    out["point_depth"] = _series(frame, "pointId", pd.NA).map(_point_depth)
    out = out.sort_values(["rally_uid", "strikeNumber", "row_pos"]).reset_index(drop=True)
    out["incoming_family"] = out.groupby("rally_uid", sort=False)["action_family"].shift(1).fillna("missing")
    context_key = out["phase"] + "|" + out["incoming_family"]
    out["context_cluster"] = context_key.map(lambda value: _stable_hash(value, modulo=128)).astype(int)
    out["same_rally_position"] = out.groupby("rally_uid", sort=False).cumcount().astype(int)
    return out


def _feature_columns() -> list[str]:
    cols = ["strike_scaled", "position_scaled"]
    cols += [f"phase_{phase}" for phase in PHASE_ORDER]
    cols += [f"incoming_{family}" for family in FAMILY_ORDER]
    cols += [f"action_{family}" for family in FAMILY_ORDER]
    cols += [f"depth_{depth}" for depth in DEPTH_ORDER]
    cols += ["context_scaled"]
    return cols


def _context_matrix(table: pd.DataFrame) -> pd.DataFrame:
    cols = _feature_columns()
    if table.empty:
        return pd.DataFrame(columns=cols)
    features = pd.DataFrame(index=table.index)
    strike = pd.to_numeric(table.get("strikeNumber", 1), errors="coerce").fillna(1.0).astype(float)
    pos = pd.to_numeric(table.get("same_rally_position", 0), errors="coerce").fillna(0.0).astype(float)
    context = pd.to_numeric(table.get("context_cluster", 0), errors="coerce").fillna(0.0).astype(float)
    features["strike_scaled"] = np.log1p(strike) / np.log(32.0)
    features["position_scaled"] = np.log1p(pos) / np.log(32.0)
    for phase in PHASE_ORDER:
        features[f"phase_{phase}"] = (table["phase"].astype(str) == phase).astype(float)
    for family in FAMILY_ORDER:
        features[f"incoming_{family}"] = (table["incoming_family"].astype(str) == family).astype(float)
        features[f"action_{family}"] = (table["action_family"].astype(str) == family).astype(float)
    for depth in DEPTH_ORDER:
        features[f"depth_{depth}"] = (table["point_depth"].astype(str) == depth).astype(float)
    features["context_scaled"] = context / 127.0
    return features.reindex(columns=cols, fill_value=0.0).astype(float)


def _deterministic_projection(matrix: np.ndarray, embedding_dim: int) -> tuple[np.ndarray, str]:
    dim = int(embedding_dim)
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], dim), dtype=float), "empty"
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    centered = (matrix - mean) / std
    emb = np.zeros((matrix.shape[0], dim), dtype=float)
    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        usable = min(dim, u.shape[1])
        emb[:, :usable] = u[:, :usable] * s[:usable]
        method = "deterministic_svd"
    except np.linalg.LinAlgError:
        usable = min(dim, centered.shape[1])
        emb[:, :usable] = centered[:, :usable]
        method = "deterministic_standardized_slice"
    return emb, method


def _build_pairs(table: pd.DataFrame, max_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if table.empty or max_pairs <= 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, np.asarray([], dtype=np.float32)
    rows: list[tuple[int, int, float]] = []
    limit = int(max_pairs)
    grouped = table.groupby(["actor_cluster", "phase", "incoming_family"], sort=True)
    for _, group in grouped:
        if len(rows) >= limit:
            break
        if len(group) < 2:
            continue
        group = group.sort_values(["rally_uid", "strikeNumber", "row_pos"]).head(16)
        idxs = group.index.to_numpy(dtype=int)
        rallies = group["rally_uid"].astype(str).to_numpy()
        for left_pos in range(len(idxs) - 1):
            for right_pos in range(left_pos + 1, len(idxs)):
                if rallies[left_pos] == rallies[right_pos]:
                    continue
                rows.append((int(idxs[left_pos]), int(idxs[right_pos]), 1.0))
                break
            if len(rows) >= limit:
                break

    positive_count = len(rows)
    ordered = table.sort_values(["context_cluster", "actor_cluster", "rally_uid", "row_pos"]).reset_index()
    n = len(ordered)
    target_negatives = max(positive_count, min(n, limit // 2))
    for pos, left in ordered.iterrows():
        if len(rows) >= limit or len(rows) >= positive_count + target_negatives:
            break
        for offset in range(1, min(n, 31)):
            right = ordered.iloc[(pos + offset) % n]
            same_actor = int(left["actor_cluster"]) == int(right["actor_cluster"])
            same_context = int(left["context_cluster"]) == int(right["context_cluster"])
            if same_actor and same_context:
                continue
            rows.append((int(left["index"]), int(right["index"]), 0.0))
            break

    if not rows:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, np.asarray([], dtype=np.float32)
    pairs = np.asarray(rows[:limit], dtype=np.float64)
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64), pairs[:, 2].astype(np.float32)


def _train_torch_encoder(
    matrix: np.ndarray,
    table: pd.DataFrame,
    *,
    embedding_dim: int,
    epochs: int,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        emb, method = _deterministic_projection(matrix, embedding_dim)
        return emb, {"method": method, "torch_available": False, "fallback_reason": str(exc), "loss": None, "pairs": 0}

    left_idx, right_idx, labels = _build_pairs(table, max_pairs=max_pairs)
    if len(labels) == 0:
        emb, method = _deterministic_projection(matrix, embedding_dim)
        return emb, {"method": method, "torch_available": True, "fallback_reason": "no_pairs", "loss": None, "pairs": 0}

    torch.manual_seed(int(seed))
    x = torch.as_tensor(matrix, dtype=torch.float32)
    left = torch.as_tensor(left_idx, dtype=torch.long)
    right = torch.as_tensor(right_idx, dtype=torch.long)
    y = torch.as_tensor(labels, dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(x.shape[1], max(16, int(embedding_dim) * 3)),
        nn.ReLU(),
        nn.Linear(max(16, int(embedding_dim) * 3), int(embedding_dim)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = min(512, max(32, len(y)))
    last_loss = float("nan")
    order = torch.arange(len(y), dtype=torch.long)
    for _ in range(max(1, int(epochs))):
        perm = order[torch.randperm(len(order))]
        for start in range(0, len(perm), batch_size):
            batch = perm[start : start + batch_size]
            emb_left = model(x[left[batch]])
            emb_right = model(x[right[batch]])
            sim = nn.functional.cosine_similarity(emb_left, emb_right)
            logits = sim * 4.0
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
    with torch.no_grad():
        emb = model(x).detach().cpu().numpy().astype(float)
    return emb, {
        "method": "torch_contrastive_mlp",
        "torch_available": True,
        "fallback_reason": None,
        "loss": last_loss,
        "pairs": int(len(y)),
        "positive_pairs": int(float(labels.sum())),
        "negative_pairs": int(len(labels) - float(labels.sum())),
    }


def _safe_embedding_frame(table: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"rally_uid": table["rally_uid"].astype(str).to_numpy()})
    for idx in range(emb.shape[1]):
        out[f"neural_style_{idx}"] = emb[:, idx].astype(float)
    norm = np.linalg.norm(emb, axis=1) if emb.size else np.zeros(len(out), dtype=float)
    out["style_norm"] = norm.astype(float)
    finite_norm = norm[np.isfinite(norm)]
    scale = float(np.nanmedian(finite_norm)) if len(finite_norm) else 1.0
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    out["style_confidence"] = np.clip(norm / (norm + scale), 0.0, 1.0).astype(float)
    out["context_cluster"] = pd.to_numeric(table["context_cluster"], errors="coerce").fillna(0).astype(int).to_numpy()
    forbidden = sorted(set(out.columns) & FORBIDDEN_EXPORT_COLUMNS)
    if forbidden:
        out = out.drop(columns=forbidden)
    neural_cols = [col for col in out.columns if col.startswith("neural_style_")]
    rally_order = pd.Index(out["rally_uid"]).drop_duplicates()
    grouped = out.groupby("rally_uid", sort=False)
    aggregated = grouped[neural_cols].mean().reindex(rally_order).reset_index()
    style_matrix = aggregated[neural_cols].to_numpy(dtype=float)
    aggregated["style_norm"] = np.linalg.norm(style_matrix, axis=1).astype(float)
    confidence = grouped["style_confidence"].mean().reindex(rally_order).to_numpy(dtype=float)
    aggregated["style_confidence"] = np.clip(confidence, 0.0, 1.0).astype(float)
    context_cluster = (
        grouped["context_cluster"]
        .agg(lambda values: int(pd.Series(values).mode(dropna=True).iloc[0]) if len(values) else 0)
        .reindex(rally_order)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    aggregated["context_cluster"] = context_cluster
    return aggregated[["rally_uid", *neural_cols, "style_confidence", "style_norm", "context_cluster"]]


def make_safe_style_features(table: pd.DataFrame, embedding_dim: int = 8) -> pd.DataFrame:
    """Create deterministic safe neural-style features for tests/fallback use."""

    matrix = _context_matrix(table).to_numpy(dtype=float)
    emb, _ = _deterministic_projection(matrix, int(embedding_dim))
    return _safe_embedding_frame(table, emb)


def _standardize_train_test(train_matrix: pd.DataFrame, test_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_np = train_matrix.to_numpy(dtype=float)
    test_np = test_matrix.to_numpy(dtype=float)
    mean = train_np.mean(axis=0, keepdims=True) if len(train_np) else np.zeros((1, train_np.shape[1]), dtype=float)
    std = train_np.std(axis=0, keepdims=True) if len(train_np) else np.ones((1, train_np.shape[1]), dtype=float)
    std[std < 1e-9] = 1.0
    return (train_np - mean) / std, (test_np - mean) / std


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def run_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    outdir: Path = OUTDIR,
    *,
    embedding_dim: int = 8,
    epochs: int = 4,
    max_pairs: int = 50000,
    seed: int = 448,
) -> dict[str, Any]:
    train = _read_csv(train_path)
    test = _read_csv(test_path)
    train_table = build_response_context_table(train)
    test_table = build_response_context_table(test)
    train_matrix, test_matrix = _standardize_train_test(_context_matrix(train_table), _context_matrix(test_table))

    train_emb, train_info = _train_torch_encoder(
        train_matrix,
        train_table,
        embedding_dim=embedding_dim,
        epochs=epochs,
        max_pairs=max_pairs,
        seed=seed,
    )
    if train_info["method"] == "torch_contrastive_mlp":
        try:
            import torch
            from torch import nn

            torch.manual_seed(int(seed))
            # Refit a deterministic linear map for test rows from train embedding by least squares.
            ridge = 1e-4 * np.eye(train_matrix.shape[1], dtype=float)
            weights = np.linalg.solve(train_matrix.T @ train_matrix + ridge, train_matrix.T @ train_emb)
            test_emb = test_matrix @ weights
        except Exception:
            combined_emb, method = _deterministic_projection(np.vstack([train_matrix, test_matrix]), embedding_dim)
            train_emb = combined_emb[: len(train_matrix)]
            test_emb = combined_emb[len(train_matrix) :]
            train_info["method"] = method
            train_info["fallback_reason"] = "test_projection_failed"
    else:
        combined_emb, method = _deterministic_projection(np.vstack([train_matrix, test_matrix]), embedding_dim)
        train_emb = combined_emb[: len(train_matrix)]
        test_emb = combined_emb[len(train_matrix) :]
        train_info["method"] = method

    train_features = _safe_embedding_frame(train_table, train_emb)
    test_features = _safe_embedding_frame(test_table, test_emb)
    export_columns = set(train_features.columns) | set(test_features.columns)
    forbidden_exports = sorted(export_columns & FORBIDDEN_EXPORT_COLUMNS)
    loss_value = train_info.get("loss")
    loss_finite = loss_value is None or bool(math.isfinite(float(loss_value)))
    report = {
        "version": "v448_neural_response_style_contrastive",
        "embedding_dim": int(embedding_dim),
        "epochs": int(epochs),
        "max_pairs": int(max_pairs),
        "seed": int(seed),
        "train_input_rows": int(len(train)),
        "test_input_rows": int(len(test)),
        "train_embedding_rows": int(len(train_features)),
        "test_embedding_rows": int(len(test_features)),
        "train_embedding_shape": [int(train_features.shape[0]), int(train_features.shape[1])],
        "test_embedding_shape": [int(test_features.shape[0]), int(test_features.shape[1])],
        "training_method": train_info.get("method"),
        "torch_available": bool(train_info.get("torch_available", False)),
        "fallback_reason": train_info.get("fallback_reason"),
        "contrastive_pairs": int(train_info.get("pairs", 0)),
        "positive_pairs": int(train_info.get("positive_pairs", 0)),
        "negative_pairs": int(train_info.get("negative_pairs", 0)),
        "loss": loss_value,
        "loss_finite": loss_finite,
        "forbidden_export_columns": forbidden_exports,
        "submission_exports": 0,
        "artifacts": {
            "train_neural_style_embeddings": "v448_neural_response_style_contrastive/train_neural_style_embeddings.csv",
            "test_neural_style_embeddings": "v448_neural_response_style_contrastive/test_neural_style_embeddings.csv",
            "contrastive_training_report": "v448_neural_response_style_contrastive/contrastive_training_report.json",
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(outdir / "train_neural_style_embeddings.csv", index=False)
    test_features.to_csv(outdir / "test_neural_style_embeddings.csv", index=False)
    _write_json(outdir / "contrastive_training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=448)
    args = parser.parse_args()
    report = run_pipeline(
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "outdir": "v448_neural_response_style_contrastive",
                "train_embedding_shape": report["train_embedding_shape"],
                "test_embedding_shape": report["test_embedding_shape"],
                "training_method": report["training_method"],
                "contrastive_pairs": report["contrastive_pairs"],
                "loss": report["loss"],
                "loss_finite": report["loss_finite"],
                "forbidden_export_columns": report["forbidden_export_columns"],
                "submission_exports": report["submission_exports"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
