"""V267 lightweight Macro-F1-aware action teacher.

This is the controlled replacement for the heavy V267 worker that stalled.
It trains a fast class-balanced SGD log-loss action posterior on safe prefix
features, applies class-prior logit adjustment, and exports capped action-only
replacement candidates on top of the V261 cap1 clean anchor.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v263_questionnaire_baseline_helpers import (
    add_questionnaire_columns,
    load_v261_cap1_anchor,
    numeric_features,
    safe_predict_proba,
    write_local_submission,
)
from baseline_lgbm import (
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


OUTDIR = Path("v267_macro_f1_action_teacher")
UPLOAD_DIR = Path("upload_candidates_20260519")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
CLASSES = list(range(19))
WEAK_CLASSES = [0, 3, 4, 5, 7, 8, 9, 12, 14]
MAX_LAG = 6
FOLDS = 5
SEED = 267
V173_REBUILT_ACTION_OOF = 0.335949
BLOCKED_FEATURES = {
    "rally_uid",
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
}


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = 1845) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")


def class_prior_logit_adjustment(class_counts: np.ndarray, tau: float) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=float)
    counts = np.maximum(counts, 1.0)
    prior = counts / counts.sum()
    return -float(tau) * np.log(prior)


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def cap_by_score(scores: np.ndarray, cap: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    budget = int(np.floor(len(scores) * float(cap)))
    mask = np.zeros(len(scores), dtype=bool)
    if budget <= 0:
        return mask
    order = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf), kind="mergesort")[:budget]
    mask[order] = True
    return mask


def clean_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0).astype(np.float32)


def make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=2e-4,
            max_iter=800,
            tol=1e-3,
            random_state=seed,
        ),
    )


def model_predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    return safe_predict_proba(model, frame, CLASSES)


def logit_adjust_probs(prob: np.ndarray, class_counts: np.ndarray, tau: float) -> np.ndarray:
    adj = class_prior_logit_adjustment(class_counts, tau)
    logits = np.log(np.clip(normalize_rows_safe(prob), 1e-12, 1.0)) + adj[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logits))


def distribution_json(labels: np.ndarray) -> str:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=19)
    return pd.Series({str(i): int(v) for i, v in enumerate(counts) if v > 0}).to_json()


def weak_mean_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    vals = [f1_score(y_true == cls, pred == cls, zero_division=0) for cls in WEAK_CLASSES]
    return float(np.mean(vals))


def build_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    train_prefix = add_questionnaire_columns(build_train_prefix_table(train, MAX_LAG))
    test_prefix = add_questionnaire_columns(build_test_prefix_table(test, MAX_LAG))
    features = numeric_features(train_prefix, test_prefix, BLOCKED_FEATURES)
    leaked = [c for c in features if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player leakage features detected: {leaked}")
    return train_prefix, test_prefix, features


def run_oof(train_prefix: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict]:
    y = train_prefix["next_actionId"].astype(int).to_numpy()
    groups = train_prefix["match"].to_numpy()
    oof = np.zeros((len(train_prefix), len(CLASSES)), dtype=float)
    splitter = GroupKFold(n_splits=FOLDS)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_prefix, y, groups), start=1):
        train_part = train_prefix.iloc[tr_idx]
        valid_part = train_prefix.iloc[va_idx]
        if set(train_part["match"].unique()) & set(valid_part["match"].unique()):
            raise RuntimeError("GroupKFold match leakage")
        model = make_model(SEED + fold)
        model.fit(clean_numeric_frame(train_part, features), train_part["next_actionId"].astype(int))
        oof[va_idx] = model_predict_proba(model, clean_numeric_frame(valid_part, features))
        pred = oof[va_idx].argmax(axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(va_idx)),
                "action_macro_f1": float(f1_score(y[va_idx], pred, average="macro", labels=CLASSES, zero_division=0)),
            }
        )
        print(json.dumps(fold_rows[-1], sort_keys=True))
    pred = oof.argmax(axis=1)
    metrics = {
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)),
        "weak_action_mean_f1": weak_mean_f1(y, pred),
        "folds": fold_rows,
    }
    return normalize_rows_safe(oof), metrics


def train_full_predict(train_prefix: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = make_model(SEED)
    model.fit(clean_numeric_frame(train_prefix, features), train_prefix["next_actionId"].astype(int))
    return model_predict_proba(model, clean_numeric_frame(test_prefix, features))


def copy_to_upload(path: Path) -> None:
    if UPLOAD_DIR.exists():
        shutil.copy2(path, UPLOAD_DIR / path.name)


def write_candidate(path: Path, anchor: pd.DataFrame, pred_action: np.ndarray) -> None:
    out = anchor.copy()
    out["actionId"] = np.asarray(pred_action, dtype=int)
    validate_submission_frame(out)
    write_local_submission(path, out)
    copy_to_upload(path)


def candidate_from_probs(anchor: pd.DataFrame, probs: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    anchor_action = anchor["actionId"].astype(int).to_numpy()
    pred = probs.argmax(axis=1).astype(int)
    score = probs[np.arange(len(probs)), pred] - probs[np.arange(len(probs)), anchor_action]
    eligible = (pred != anchor_action) & np.isfinite(score)
    mask = cap_by_score(np.where(eligible, score, -np.inf), cap)
    out = anchor_action.copy()
    out[mask] = pred[mask]
    return out, mask


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_prefix, test_prefix, features = build_feature_tables()
    anchor = load_v261_cap1_anchor()
    validate_submission_frame(anchor)
    y = train_prefix["next_actionId"].astype(int).to_numpy()
    class_counts = np.bincount(y, minlength=19)

    oof_prob, oof_metrics = run_oof(train_prefix, features)
    test_prob = train_full_predict(train_prefix, test_prefix, features)

    rows = []
    variants = [
        ("balanced_raw_diagnostic", 0.0, 1.0, "DIAGNOSTIC_ONLY"),
        ("logadj_tau0p10_cap0p005", 0.10, 0.005, "LOCAL_NEGATIVE_DO_NOT_SUBMIT"),
        ("logadj_tau0p20_cap0p010", 0.20, 0.010, "LOCAL_NEGATIVE_DO_NOT_SUBMIT"),
        ("logadj_tau0p35_cap0p020", 0.35, 0.020, "LOCAL_NEGATIVE_DO_NOT_SUBMIT"),
        ("logadj_tau0p35_cap0p050", 0.35, 0.050, "LOCAL_NEGATIVE_DO_NOT_SUBMIT"),
    ]

    for variant, tau, cap, default_verdict in variants:
        prob = logit_adjust_probs(test_prob, class_counts, tau) if tau > 0 else test_prob
        pred, mask = candidate_from_probs(anchor, prob, cap)
        if variant == "balanced_raw_diagnostic":
            pred = prob.argmax(axis=1).astype(int)
            mask = pred != anchor["actionId"].astype(int).to_numpy()
        name = f"submission_v267_action_{variant}__pv261cap1__sr121.csv"
        path = OUTDIR / name
        write_candidate(path, anchor, pred)
        serve_count = int(np.isin(pred, [15, 16, 17, 18]).sum())
        action_churn = float(np.mean(pred != anchor["actionId"].astype(int).to_numpy()))
        verdict = default_verdict
        if (
            variant != "balanced_raw_diagnostic"
            and oof_metrics["action_macro_f1"] >= V173_REBUILT_ACTION_OOF + 0.003
            and action_churn <= 0.05
            and serve_count <= 2
        ):
            verdict = "CANDIDATE_FOR_REVIEW"
        rows.append(
            {
                "candidate": name,
                "path": str(path),
                "ordinary_action_macro_f1": oof_metrics["action_macro_f1"],
                "ordinary_action_delta_vs_anchor": oof_metrics["action_macro_f1"] - V173_REBUILT_ACTION_OOF,
                "weak_action_mean_f1": oof_metrics["weak_action_mean_f1"],
                "action_churn": action_churn,
                "serve_15_18_count": serve_count,
                "test_action_distribution": distribution_json(pred),
                "tau": tau,
                "cap": cap,
                "features": len(features),
                "train_rows": len(train_prefix),
                "verdict": verdict,
            }
        )

    search = pd.DataFrame(rows)
    search.to_csv(OUTDIR / "v267_action_search.csv", index=False)
    (OUTDIR / "v267_report.md").write_text(
        "# V267 Macro-F1 Action Teacher\n\n"
        f"- Model: `SGDClassifier(log_loss, class_weight=balanced)`\n"
        f"- Train prefix rows: `{len(train_prefix)}`\n"
        f"- Features: `{len(features)}`\n"
        f"- OOF action Macro-F1: `{oof_metrics['action_macro_f1']:.6f}`\n"
        f"- Delta vs V173 rebuilt reference `{V173_REBUILT_ACTION_OOF:.6f}`: "
        f"`{oof_metrics['action_macro_f1'] - V173_REBUILT_ACTION_OOF:+.6f}`\n"
        f"- Weak-action mean F1: `{oof_metrics['weak_action_mean_f1']:.6f}`\n\n"
        "## Interpretation\n\n"
        "This lightweight balanced posterior does not clear the V173-scale local gate unless search rows say otherwise. "
        "Generated capped candidates are diagnostic/review artifacts only.\n",
        encoding="utf-8",
    )
    (OUTDIR / "v267_run_summary.json").write_text(
        json.dumps({"oof": oof_metrics, "candidates": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "candidates": len(rows), "oof_action": oof_metrics["action_macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
