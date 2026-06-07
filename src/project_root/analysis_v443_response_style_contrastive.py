"""V443 response-style contrastive proxy features.

This script builds deterministic player/response-style features from observed
prefix rows. It writes feature artifacts only, with no submission CSVs and no
exact future target columns in exported tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v443_response_style_contrastive"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"

FAMILY_ORDER = ["serve", "attack", "control", "defense", "other", "zero", "missing"]
PHASE_ORDER = ["serve", "receive", "third_ball", "fourth_ball", "rally"]
DEPTH_ORDER = ["zero", "short", "half", "long", "missing"]
PAIR_COLUMNS = [
    "left_row",
    "right_row",
    "pair_label",
    "left_rally_uid",
    "right_rally_uid",
    "same_actor_cluster",
    "same_context",
    "context_key",
]
EXACT_FUTURE_COLUMNS = {
    "target_actionId",
    "target_pointId",
    "target_serverGetPoint",
    "next_actionId",
    "next_pointId",
    "next_serverGetPoint",
}
RAW_PLAYER_COLUMNS = {"gamePlayerId", "gamePlayerOtherId", "actor_key", "player_cluster"}


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
    if not text:
        return _phase_from_strike(strike_number)
    if "serve" in text:
        return "serve"
    if "receive" in text:
        return "receive"
    if "third" in text:
        return "third_ball"
    if "fourth" in text:
        return "fourth_ball"
    return "rally"


def _prepare_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "row_pos",
                "rally_uid",
                "strikeNumber",
                "actor_key",
                "phase",
                "incoming_family",
                "action_family",
                "point_depth",
                "context_key",
                "response_key",
            ]
        )

    out = pd.DataFrame(index=frame.index)
    out["row_pos"] = np.arange(len(frame), dtype=int)
    out["rally_uid"] = _series(frame, "rally_uid", np.arange(len(frame))).astype(str)
    out["strikeNumber"] = pd.to_numeric(_series(frame, "strikeNumber", 1), errors="coerce").fillna(1).astype(int)
    actor = _series(frame, "gamePlayerId", _series(frame, "gamePlayerOtherId", "unknown"))
    out["actor_key"] = actor.astype(str).fillna("unknown")
    phase_source = frame["phase"] if "phase" in frame.columns else pd.Series([None] * len(frame), index=frame.index)
    out["phase"] = [
        _normalize_phase(value, strike)
        for value, strike in zip(phase_source, out["strikeNumber"])
    ]
    out["action_family"] = _series(frame, "actionId", pd.NA).map(_action_family)
    out["point_depth"] = _series(frame, "pointId", pd.NA).map(_point_depth)
    out = out.sort_values(["rally_uid", "strikeNumber", "row_pos"]).reset_index(drop=True)
    out["incoming_family"] = out.groupby("rally_uid", sort=False)["action_family"].shift(1).fillna("missing")
    out["context_key"] = out["phase"] + "|" + out["incoming_family"]
    out["response_key"] = out["context_key"] + "|" + out["action_family"] + "|" + out["point_depth"]
    return out


def build_contrastive_pairs(frame: pd.DataFrame, max_pairs: int = 50000) -> pd.DataFrame:
    """Build bounded deterministic positive/negative row pairs.

    Positives share actor and incoming phase/context. Negatives differ by actor
    or by context. The output intentionally contains row references and context
    diagnostics only, not raw player IDs or exact target columns.
    """

    events = _prepare_events(frame)
    if events.empty or max_pairs <= 0:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    rows: list[dict[str, Any]] = []
    limit = int(max_pairs)

    for _, group in events.groupby(["actor_key", "context_key"], sort=True):
        if len(rows) >= limit:
            break
        if len(group) < 2:
            continue
        group = group.sort_values(["rally_uid", "strikeNumber", "row_pos"]).head(12).reset_index(drop=True)
        for idx in range(len(group) - 1):
            left = group.iloc[idx]
            for right_idx in range(idx + 1, len(group)):
                right = group.iloc[right_idx]
                if left["rally_uid"] == right["rally_uid"]:
                    continue
                rows.append(
                    {
                        "left_row": int(left["row_pos"]),
                        "right_row": int(right["row_pos"]),
                        "pair_label": 1,
                        "left_rally_uid": left["rally_uid"],
                        "right_rally_uid": right["rally_uid"],
                        "same_actor_cluster": 1,
                        "same_context": 1,
                        "context_key": left["context_key"],
                    }
                )
                break
            if len(rows) >= limit:
                break

    positive_count = len(rows)
    sorted_events = events.sort_values(["context_key", "actor_key", "rally_uid", "row_pos"]).reset_index(drop=True)
    n = len(sorted_events)
    for idx, left in sorted_events.iterrows():
        if len(rows) >= limit or len(rows) >= max(positive_count * 2, 1):
            break
        for offset in range(1, min(n, 17)):
            right = sorted_events.iloc[(idx + offset) % n]
            same_actor = left["actor_key"] == right["actor_key"]
            same_context = left["context_key"] == right["context_key"]
            if same_actor and same_context:
                continue
            rows.append(
                {
                    "left_row": int(left["row_pos"]),
                    "right_row": int(right["row_pos"]),
                    "pair_label": 0,
                    "left_rally_uid": left["rally_uid"],
                    "right_rally_uid": right["rally_uid"],
                    "same_actor_cluster": int(same_actor),
                    "same_context": int(same_context),
                    "context_key": left["context_key"],
                }
            )
            break

    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if len(pairs) > limit:
        pairs = pairs.iloc[:limit].copy()
    return pairs.reset_index(drop=True)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(float) / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / np.log(max(len(counts), 2)))


def _stable_feature_columns() -> list[str]:
    columns = ["event_count", "unique_actor_count", "response_entropy", "phase_entropy"]
    columns += [f"phase_rate_{phase}" for phase in PHASE_ORDER]
    columns += [f"incoming_rate_{family}" for family in FAMILY_ORDER]
    columns += [f"response_rate_{family}" for family in FAMILY_ORDER]
    columns += [f"depth_rate_{depth}" for depth in DEPTH_ORDER]
    columns += [f"phase_response_rate_{phase}_{family}" for phase in PHASE_ORDER for family in FAMILY_ORDER]
    columns += ["pair_familiarity_low", "pair_familiarity_mid", "pair_familiarity_high"]
    return columns


def _response_vector_for_group(group: pd.DataFrame, actor_context_counts: pd.Series) -> dict[str, float]:
    row: dict[str, float] = {}
    row["event_count"] = float(len(group))
    row["unique_actor_count"] = float(group["actor_key"].nunique())

    response_counts = group["action_family"].value_counts().reindex(FAMILY_ORDER, fill_value=0).to_numpy(dtype=float)
    phase_counts = group["phase"].value_counts().reindex(PHASE_ORDER, fill_value=0).to_numpy(dtype=float)
    row["response_entropy"] = _entropy_from_counts(response_counts)
    row["phase_entropy"] = _entropy_from_counts(phase_counts)

    denom = max(float(len(group)), 1.0)
    for phase in PHASE_ORDER:
        row[f"phase_rate_{phase}"] = float((group["phase"] == phase).sum() / denom)
    for family in FAMILY_ORDER:
        row[f"incoming_rate_{family}"] = float((group["incoming_family"] == family).sum() / denom)
        row[f"response_rate_{family}"] = float((group["action_family"] == family).sum() / denom)
    for depth in DEPTH_ORDER:
        row[f"depth_rate_{depth}"] = float((group["point_depth"] == depth).sum() / denom)

    phase_response = (
        group.assign(_value=1.0)
        .pivot_table(index="phase", columns="action_family", values="_value", aggfunc="sum", fill_value=0.0)
        .reindex(index=PHASE_ORDER, columns=FAMILY_ORDER, fill_value=0.0)
    )
    total_phase_response = max(float(phase_response.to_numpy().sum()), 1.0)
    for phase in PHASE_ORDER:
        for family in FAMILY_ORDER:
            row[f"phase_response_rate_{phase}_{family}"] = float(phase_response.loc[phase, family] / total_phase_response)

    familiarity_values = []
    for actor, context in zip(group["actor_key"], group["context_key"]):
        familiarity_values.append(float(actor_context_counts.get((actor, context), 0.0)))
    avg_familiarity = float(np.mean(familiarity_values)) if familiarity_values else 0.0
    row["pair_familiarity_low"] = float(avg_familiarity <= 1.0)
    row["pair_familiarity_mid"] = float(1.0 < avg_familiarity <= 4.0)
    row["pair_familiarity_high"] = float(avg_familiarity > 4.0)
    return row


def _svd_projection(features: pd.DataFrame, embedding_dim: int) -> pd.DataFrame:
    columns = _stable_feature_columns()
    matrix = features[columns].to_numpy(dtype=float)
    if matrix.size:
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std < 1e-9] = 1.0
        centered = (matrix - mean) / std
    else:
        centered = np.zeros((len(features), len(columns)), dtype=float)

    dim = int(embedding_dim)
    emb = np.zeros((len(features), dim), dtype=float)
    if len(features) and dim > 0:
        try:
            u, s, _ = np.linalg.svd(centered, full_matrices=False)
            usable = min(dim, u.shape[1])
            if usable:
                emb[:, :usable] = u[:, :usable] * s[:usable]
        except np.linalg.LinAlgError:
            usable = min(dim, centered.shape[1])
            emb[:, :usable] = centered[:, :usable]
    return pd.DataFrame(emb, columns=[f"style_emb_{idx}" for idx in range(dim)], index=features.index)


def compute_response_style_embeddings(frame: pd.DataFrame, embedding_dim: int = 8) -> pd.DataFrame:
    """Compute one deterministic style embedding row per rally_uid."""

    events = _prepare_events(frame)
    emb_cols = [f"style_emb_{idx}" for idx in range(int(embedding_dim))]
    if events.empty:
        return pd.DataFrame(columns=["rally_uid", *emb_cols])

    stable_columns = _stable_feature_columns()
    actor_context_counts = events.groupby(["actor_key", "context_key"], sort=False).size()
    rally_order = pd.Index(_series(frame, "rally_uid", np.arange(len(frame))).astype(str)).drop_duplicates()
    features = pd.DataFrame({"rally_uid": rally_order})
    features = features.set_index("rally_uid", drop=False)

    event_count = events.groupby("rally_uid", sort=False).size().reindex(rally_order, fill_value=0).astype(float)
    features["event_count"] = event_count.to_numpy(dtype=float)
    features["unique_actor_count"] = (
        events.groupby("rally_uid", sort=False)["actor_key"].nunique().reindex(rally_order, fill_value=0).to_numpy(dtype=float)
    )

    response_counts = pd.crosstab(events["rally_uid"], events["action_family"]).reindex(
        index=rally_order, columns=FAMILY_ORDER, fill_value=0
    )
    phase_counts = pd.crosstab(events["rally_uid"], events["phase"]).reindex(
        index=rally_order, columns=PHASE_ORDER, fill_value=0
    )
    incoming_counts = pd.crosstab(events["rally_uid"], events["incoming_family"]).reindex(
        index=rally_order, columns=FAMILY_ORDER, fill_value=0
    )
    depth_counts = pd.crosstab(events["rally_uid"], events["point_depth"]).reindex(
        index=rally_order, columns=DEPTH_ORDER, fill_value=0
    )
    denom = event_count.replace(0.0, 1.0)

    features["response_entropy"] = [_entropy_from_counts(row) for row in response_counts.to_numpy(dtype=float)]
    features["phase_entropy"] = [_entropy_from_counts(row) for row in phase_counts.to_numpy(dtype=float)]
    for phase in PHASE_ORDER:
        features[f"phase_rate_{phase}"] = (phase_counts[phase].astype(float) / denom).to_numpy(dtype=float)
    for family in FAMILY_ORDER:
        features[f"incoming_rate_{family}"] = (incoming_counts[family].astype(float) / denom).to_numpy(dtype=float)
        features[f"response_rate_{family}"] = (response_counts[family].astype(float) / denom).to_numpy(dtype=float)
    for depth in DEPTH_ORDER:
        features[f"depth_rate_{depth}"] = (depth_counts[depth].astype(float) / denom).to_numpy(dtype=float)

    keyed = events.assign(phase_response_key=events["phase"] + "_" + events["action_family"])
    phase_response_order = [f"{phase}_{family}" for phase in PHASE_ORDER for family in FAMILY_ORDER]
    phase_response_counts = pd.crosstab(keyed["rally_uid"], keyed["phase_response_key"]).reindex(
        index=rally_order, columns=phase_response_order, fill_value=0
    )
    for phase in PHASE_ORDER:
        for family in FAMILY_ORDER:
            source_col = f"{phase}_{family}"
            features[f"phase_response_rate_{phase}_{family}"] = (
                phase_response_counts[source_col].astype(float) / denom
            ).to_numpy(dtype=float)

    familiarity = actor_context_counts.rename("_familiarity").reset_index()
    with_familiarity = events.merge(familiarity, on=["actor_key", "context_key"], how="left")
    avg_familiarity = (
        with_familiarity.groupby("rally_uid", sort=False)["_familiarity"].mean().reindex(rally_order, fill_value=0.0)
    )
    features["pair_familiarity_low"] = (avg_familiarity <= 1.0).astype(float).to_numpy(dtype=float)
    features["pair_familiarity_mid"] = ((avg_familiarity > 1.0) & (avg_familiarity <= 4.0)).astype(float).to_numpy(dtype=float)
    features["pair_familiarity_high"] = (avg_familiarity > 4.0).astype(float).to_numpy(dtype=float)

    for col in stable_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = features.reset_index(drop=True)
    embeddings = _svd_projection(features, int(embedding_dim))
    out = pd.concat([features[["rally_uid"] + stable_columns].reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
    drop_cols = [col for col in out.columns if col in EXACT_FUTURE_COLUMNS or col in RAW_PLAYER_COLUMNS]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_emb: pd.DataFrame,
    test_emb: pd.DataFrame,
    pairs: pd.DataFrame,
    embedding_dim: int,
    quick: bool,
) -> dict[str, Any]:
    exact_leak_columns = sorted((set(train_emb.columns) | set(test_emb.columns) | set(pairs.columns)) & EXACT_FUTURE_COLUMNS)
    raw_player_export_columns = sorted((set(train_emb.columns) | set(test_emb.columns) | set(pairs.columns)) & RAW_PLAYER_COLUMNS)
    pair_counts = pairs["pair_label"].value_counts().to_dict() if not pairs.empty else {}
    return {
        "version": "v443_response_style_contrastive",
        "quick": bool(quick),
        "embedding_dim": int(embedding_dim),
        "train_input_rows": int(len(train)),
        "test_input_rows": int(len(test)),
        "train_rally_uid_rows": int(train["rally_uid"].nunique()) if "rally_uid" in train.columns else int(len(train_emb)),
        "test_rally_uid_rows": int(test["rally_uid"].nunique()) if "rally_uid" in test.columns else int(len(test_emb)),
        "train_embedding_shape": [int(train_emb.shape[0]), int(train_emb.shape[1])],
        "test_embedding_shape": [int(test_emb.shape[0]), int(test_emb.shape[1])],
        "contrastive_pair_rows": int(len(pairs)),
        "positive_pairs": int(pair_counts.get(1, 0)),
        "negative_pairs": int(pair_counts.get(0, 0)),
        "exact_future_export_columns": exact_leak_columns,
        "raw_player_export_columns": raw_player_export_columns,
        "submissions_written": 0,
        "artifacts": {
            "train_style_embeddings": "v443_response_style_contrastive/train_style_embeddings.csv",
            "test_style_embeddings": "v443_response_style_contrastive/test_style_embeddings.csv",
            "contrastive_pairs": "v443_response_style_contrastive/contrastive_pairs.csv",
            "style_report": "v443_response_style_contrastive/style_report.json",
        },
    }


def run_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    outdir: Path = OUTDIR,
    *,
    quick: bool = False,
    embedding_dim: int = 8,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    train = _read_csv(train_path)
    test = _read_csv(test_path)
    pair_limit = int(max_pairs if max_pairs is not None else (20000 if quick else 100000))

    train_emb = compute_response_style_embeddings(train, embedding_dim=embedding_dim)
    test_emb = compute_response_style_embeddings(test, embedding_dim=embedding_dim)
    pairs = build_contrastive_pairs(train, max_pairs=pair_limit)
    report = _report(train, test, train_emb, test_emb, pairs, embedding_dim, quick)

    outdir.mkdir(parents=True, exist_ok=True)
    train_emb.to_csv(outdir / "train_style_embeddings.csv", index=False)
    test_emb.to_csv(outdir / "test_style_embeddings.csv", index=False)
    pairs.to_csv(outdir / "contrastive_pairs.csv", index=False)
    _write_json(outdir / "style_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use bounded pair generation for fast verification.")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    report = run_pipeline(quick=args.quick, embedding_dim=args.embedding_dim, max_pairs=args.max_pairs)
    print(
        json.dumps(
            {
                "outdir": "v443_response_style_contrastive",
                "train_embedding_shape": report["train_embedding_shape"],
                "test_embedding_shape": report["test_embedding_shape"],
                "positive_pairs": report["positive_pairs"],
                "negative_pairs": report["negative_pairs"],
                "submissions_written": report["submissions_written"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
