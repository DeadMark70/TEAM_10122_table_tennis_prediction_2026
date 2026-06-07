"""R42: Golden soft-prob teacher blend candidates.

Uses C:/aicup/tenis_new/submission_golden_blend_probs.csv as a historical
soft teacher. Keeps current R34 point/server fixed for safer candidates and
only blends action probabilities.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis_r7_phase_features import add_phase_features
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, full_predict
from generate_r1_submission import compose_v3_full


OUT_DIR = Path("r42_golden_soft_blends")
UPLOAD_DIR = Path("upload_candidates_20260519")
GOLDEN_PATH = Path(r"C:\aicup\tenis_new\submission_golden_blend_probs.csv")
CURRENT_SUB_PATH = Path("upload_candidates_20260519/submission_r34_r33action_v3point_r28server.csv")


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def compose_r7_full(train: pd.DataFrame, test: pd.DataFrame, r7_tuning: V3Tuning, out_path: Path) -> dict[str, object]:
    if out_path.exists():
        with open(out_path, "rb") as f:
            return pickle.load(f)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    test_prefix = test_prefix[["rally_uid", "match"] + features]
    pred = full_predict(prefix_df, test_prefix, features, SimpleNamespace(seeds=[42], n_estimators=120, ngram_alpha=20.0))
    action = blend_probs(pred["lgbm_action"], pred["ngram_action"], r7_tuning.action_ngram_weight)
    point = blend_probs(pred["lgbm_point"], pred["ngram_point"], r7_tuning.point_ngram_weight)
    sw = r7_tuning.server_weights
    server = sw["direct"] * pred["lgbm_server"] + sw["ngram"] * pred["ngram_server"] + sw["parity"] * pred["parity_server"] + sw["remaining"] * pred["remaining_server"]
    out = {
        "test_prefix": test_prefix[["rally_uid", "prefix_len"]].copy(),
        "r7_action": action,
        "r7_point": point,
        "r7_server": np.clip(server, 1e-6, 1.0 - 1e-6),
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    return out


def build_current_r33_action_prob() -> tuple[pd.DataFrame, np.ndarray, dict]:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    selected = json.loads(Path("r33_safe_oof_ensemble/r33_selected.json").read_text(encoding="utf-8"))
    with open("oof_proba_v3.pkl", "rb") as f:
        v3_oof = pickle.load(f)
    with open("oof_proba_r7.pkl", "rb") as f:
        r7_oof = pickle.load(f)
    with open("r1_full_sequence_proba.pkl", "rb") as f:
        r1_seq = pickle.load(f)
    test_prefix, _, _, _ = compose_v3_full(train, test, v3_oof["tuning"])
    r7_full = compose_r7_full(train, test, r7_oof["tuning"], Path("r7_full_lgbm_proba.pkl"))
    test_meta = r1_seq["test_meta"].reset_index(drop=True)
    for name, ids in [
        ("V3", test_prefix["rally_uid"].reset_index(drop=True)),
        ("R7", r7_full["test_prefix"]["rally_uid"].reset_index(drop=True)),
    ]:
        if not test_meta["rally_uid"].reset_index(drop=True).equals(ids):
            raise ValueError(f"{name} rows are not aligned.")
    r1_action = normalize_rows(0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"])
    action_prob = normalize_rows(0.85 * r1_action + 0.05 * r7_full["r7_action"] + 0.10 * r1_seq["gru_action"])
    return test_meta, action_prob, selected


def read_golden(test_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    golden = pd.read_csv(GOLDEN_PATH)
    action_cols = [f"seq_action_prob_{i:02d}" for i in range(19)]
    point_cols = [f"seq_point_prob_{i:02d}" for i in range(10)]
    merged = test_meta[["rally_uid", "prefix_len"]].merge(golden, on="rally_uid", how="left")
    if merged[action_cols].isna().any().any():
        raise ValueError("Golden soft probs did not align.")
    action = normalize_rows(merged[action_cols].to_numpy(dtype=float))
    point = normalize_rows(merged[point_cols].to_numpy(dtype=float))
    server = merged["serverGetPoint"].to_numpy(dtype=float)
    return action, point, server, merged


def write_submission(test_meta: pd.DataFrame, action_prob: np.ndarray, point_label: np.ndarray, server_prob: np.ndarray, action_mult: dict, name: str) -> Path:
    action_pred = apply_segmented_multipliers(test_meta, action_prob, action_mult, ACTION_CLASSES, "two")
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_label.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return path


def write_hard_submission(test_meta: pd.DataFrame, action_label: np.ndarray, point_label: np.ndarray, server_prob: np.ndarray, name: str) -> Path:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_label.astype(int),
            "pointId": point_label.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    test_meta, current_action_prob, selected = build_current_r33_action_prob()
    current_sub = pd.read_csv(CURRENT_SUB_PATH)
    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(current_sub, on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")
    current_point_label = current_sub["pointId"].to_numpy(dtype=int)
    current_server = current_sub["serverGetPoint"].to_numpy(dtype=float)
    golden_action, golden_point, golden_server, golden = read_golden(test_meta)

    paths = []
    for w in [0.10, 0.20, 0.35, 0.50, 0.65]:
        action = normalize_rows((1 - w) * current_action_prob + w * golden_action)
        paths.append(write_submission(test_meta, action, current_point_label, current_server, selected["action_multipliers"], f"submission_r42_golden_action_w{str(w).replace('.', 'p')}_current_point_server.csv"))

    len1 = test_meta["prefix_len"].eq(1).to_numpy()
    for w in [0.50, 1.00]:
        action = current_action_prob.copy()
        action[len1] = normalize_rows((1 - w) * current_action_prob[len1] + w * golden_action[len1])
        paths.append(write_submission(test_meta, action, current_point_label, current_server, selected["action_multipliers"], f"submission_r42_golden_action_len1_w{str(w).replace('.', 'p')}_current_point_server.csv"))

    # Diagnostic: full golden action/point with current server, and full golden copy.
    paths.append(write_hard_submission(test_meta, golden_action.argmax(axis=1), golden_point.argmax(axis=1), current_server, "submission_r42_golden_hard_action_point_current_server.csv"))
    paths.append(write_hard_submission(test_meta, golden_action.argmax(axis=1), golden_point.argmax(axis=1), golden_server, "submission_r42_golden_full_hard.csv"))

    analysis_rows = []
    for name, action_prob in [
        ("current_r33_prob", current_action_prob),
        ("golden_prob", golden_action),
        ("blend_w0.2", normalize_rows(0.8 * current_action_prob + 0.2 * golden_action)),
        ("blend_w0.35", normalize_rows(0.65 * current_action_prob + 0.35 * golden_action)),
    ]:
        pred = apply_segmented_multipliers(test_meta, action_prob, selected["action_multipliers"], ACTION_CLASSES, "two")
        analysis_rows.append(
            {
                "candidate": name,
                "action_diff_vs_current_r34_hard": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
                "action8_count": int((pred == 8).sum()),
                "action9_count": int((pred == 9).sum()),
                "action12_count": int((pred == 12).sum()),
                "action0_count": int((pred == 0).sum()),
            }
        )
    pd.DataFrame(analysis_rows).to_csv(OUT_DIR / "r42_action_distribution_summary.csv", index=False)
    report = {
        "golden_path": str(GOLDEN_PATH),
        "current_submission": str(CURRENT_SUB_PATH),
        "generated": [str(p) for p in paths],
        "note": "Safer candidates keep current R34 point/server fixed and only blend golden action probabilities.",
    }
    (OUT_DIR / "r42_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
