"""V263A simple balanced action residual.

Local-only action residual search on top of the V261 cap1 anchor.  This script
uses safe prefix features from baseline_lgbm plus questionnaire columns, trains
balanced tree classifiers with GroupKFold(match), and writes capped action
submissions under v263_questionnaire_baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis_v263_questionnaire_baseline_helpers import (
    OUTDIR,
    add_questionnaire_columns,
    cap_by_score,
    class_f1_table,
    distribution_json,
    load_v261_cap1_anchor,
    normalize_rows,
    numeric_features,
    safe_predict_proba,
    write_local_submission,
)
from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


CAPS = [0.005, 0.010, 0.020]
MAX_LAG = 6
N_SPLITS = 5
EXPECTED_ROWS = 1845


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)

    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = build_train_prefix_table(train_raw, MAX_LAG)
    test_df = build_test_prefix_table(test_raw, MAX_LAG)
    train_df = add_questionnaire_columns(train_df)
    test_df = add_questionnaire_columns(test_df)

    anchor = load_v261_cap1_anchor()
    if len(anchor) != len(test_df) or len(anchor) != EXPECTED_ROWS:
        raise ValueError(f"Anchor/test row mismatch: anchor={len(anchor)} test={len(test_df)}")
    if not np.array_equal(anchor["rally_uid"].astype(int).to_numpy(), test_df["rally_uid"].astype(int).to_numpy()):
        raise ValueError("V261 cap1 anchor rally_uid order does not match test prefix rows.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), anchor


def assign_group_folds(train_df: pd.DataFrame) -> pd.Series:
    rally_meta = train_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=N_SPLITS)
    rally_meta["fold"] = -1
    for fold, (_, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"])):
        rally_meta.loc[valid_idx, "fold"] = int(fold)
    if rally_meta["fold"].lt(0).any():
        raise RuntimeError("Fold assignment failed.")
    fold_map = rally_meta.set_index("rally_uid")["fold"]
    return train_df["rally_uid"].map(fold_map).astype(int)


def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    blocked = {
        "rally_uid",
        "match",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "remaining_len",
        "final_parity_even",
        "num_prefixes_in_rally",
        "fold",
    }
    features = numeric_features(train_df, test_df, blocked)
    leaked = [c for c in features if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player feature leakage detected: {leaked}")
    return features


def fit_extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def fit_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def train_action_oof_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    y = train_df["next_actionId"].astype(int).to_numpy()
    folds = train_df["fold"].astype(int).to_numpy()
    classes = list(ACTION_CLASSES)
    oof = np.zeros((len(train_df), len(classes)), dtype=float)
    test_sum = np.zeros((len(test_df), len(classes)), dtype=float)
    fold_rows: list[dict[str, int]] = []

    x_test = test_df[features].replace([np.inf, -np.inf], 0).fillna(0)
    for fold in sorted(np.unique(folds)):
        valid_mask = folds == int(fold)
        train_mask = ~valid_mask
        x_train = train_df.loc[train_mask, features].replace([np.inf, -np.inf], 0).fillna(0)
        x_valid = train_df.loc[valid_mask, features].replace([np.inf, -np.inf], 0).fillna(0)

        et = fit_extra_trees(2630 + int(fold))
        rf = fit_random_forest(2640 + int(fold))
        et.fit(x_train, y[train_mask])
        rf.fit(x_train, y[train_mask])

        valid_prob = 0.5 * (
            safe_predict_proba(et, x_valid, classes) + safe_predict_proba(rf, x_valid, classes)
        )
        test_prob = 0.5 * (
            safe_predict_proba(et, x_test, classes) + safe_predict_proba(rf, x_test, classes)
        )
        oof[valid_mask] = valid_prob
        test_sum += test_prob
        fold_rows.append(
            {"fold": int(fold), "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum())}
        )

    return normalize_rows(oof), normalize_rows(test_sum / len(fold_rows)), fold_rows


def simple_proxy_base(train_df: pd.DataFrame) -> np.ndarray:
    """Fold-safe no-fit action proxy for local delta accounting."""
    base = pd.to_numeric(train_df["lag0_actionId"], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.clip(base, 0, len(ACTION_CLASSES) - 1)


def capped_residual_labels(
    base_labels: np.ndarray,
    prob: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    p = normalize_rows(prob)
    top = p.argmax(axis=1).astype(int)
    base_pos = np.clip(base, 0, p.shape[1] - 1)
    score = p[np.arange(len(p)), top] - p[np.arange(len(p)), base_pos]
    eligible = (top != base) & np.isfinite(score) & (score > 0)
    changed = cap_by_score(np.where(eligible, score, -np.inf), cap) & eligible
    out = base.copy()
    out[changed] = top[changed]
    return out, changed, score


def weak_class_mean_f1(y: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray) -> float:
    support = pd.Series(y).value_counts().reindex(ACTION_CLASSES, fill_value=0)
    weak_classes = support.sort_values(kind="mergesort").index[:5].astype(int).tolist()
    table = class_f1_table(y, base_pred, cand_pred, list(ACTION_CLASSES))
    return float(table[table["class_id"].isin(weak_classes)]["candidate_f1"].mean())


def action_distribution(labels: np.ndarray) -> str:
    return distribution_json(np.asarray(labels, dtype=int), len(ACTION_CLASSES))


def write_action_submission(anchor: pd.DataFrame, action: np.ndarray, cap: float) -> Path:
    name = f"submission_v263a_action_cap{cap:0.3f}".replace(".", "p") + "__pv261cap1__sr121.csv"
    out = anchor.copy()
    out["actionId"] = np.asarray(action, dtype=int)
    path = OUTDIR / name
    write_local_submission(path, out)
    return path


def verdict(delta: float, serve_count: int, anchor_serve_count: int) -> str:
    serve_limit = int(np.ceil(anchor_serve_count * 1.25 + 5))
    if delta > 0.0 and serve_count <= serve_limit:
        return "CANDIDATE_FOR_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, anchor = build_frames()
    train_df["fold"] = assign_group_folds(train_df)
    features = make_features(train_df, test_df)
    if not features:
        raise RuntimeError("No numeric features selected.")

    y = train_df["next_actionId"].astype(int).to_numpy()
    proxy_base = simple_proxy_base(train_df)
    proxy_base_f1 = float(f1_score(y, proxy_base, labels=ACTION_CLASSES, average="macro", zero_division=0))
    model_oof_prob, model_test_prob, fold_rows = train_action_oof_and_test(train_df, test_df, features)
    raw_oof_pred = model_oof_prob.argmax(axis=1).astype(int)
    raw_model_f1 = float(f1_score(y, raw_oof_pred, labels=ACTION_CLASSES, average="macro", zero_division=0))

    test_base_action = anchor["actionId"].astype(int).to_numpy()
    anchor_serve_count = int(np.isin(test_base_action, [15, 16, 17, 18]).sum())
    records: list[dict[str, object]] = [
        {
            "candidate": "v263a_raw_balanced_action_diagnostic",
            "action_macro_f1": raw_model_f1,
            "delta_vs_proxy_base": raw_model_f1 - proxy_base_f1,
            "action_churn_vs_anchor": float("nan"),
            "changed_rows": 0,
            "weak_class_mean_f1": weak_class_mean_f1(y, proxy_base, raw_oof_pred),
            "serve_15_18_count_test": int(np.isin(model_test_prob.argmax(axis=1), [15, 16, 17, 18]).sum()),
            "test_action_distribution": action_distribution(model_test_prob.argmax(axis=1)),
            "verdict": "DIAGNOSTIC_ONLY",
            "path": "",
        }
    ]

    submissions: list[dict[str, object]] = []
    for cap in CAPS:
        oof_pred, _, _ = capped_residual_labels(proxy_base, model_oof_prob, cap)
        test_pred, test_changed, _ = capped_residual_labels(test_base_action, model_test_prob, cap)
        score = float(f1_score(y, oof_pred, labels=ACTION_CLASSES, average="macro", zero_division=0))
        serve_count = int(np.isin(test_pred, [15, 16, 17, 18]).sum())
        path = write_action_submission(anchor, test_pred, cap)
        rec = {
            "candidate": f"v263a_action_cap{cap:0.3f}".replace(".", "p"),
            "action_macro_f1": score,
            "delta_vs_proxy_base": score - proxy_base_f1,
            "action_churn_vs_anchor": float(np.mean(test_pred != test_base_action)),
            "changed_rows": int(test_changed.sum()),
            "weak_class_mean_f1": weak_class_mean_f1(y, proxy_base, oof_pred),
            "serve_15_18_count_test": serve_count,
            "test_action_distribution": action_distribution(test_pred),
            "verdict": verdict(score - proxy_base_f1, serve_count, anchor_serve_count),
            "path": str(path),
        }
        records.append(rec)
        submissions.append(rec)

    search = pd.DataFrame(records)
    search.to_csv(OUTDIR / "v263a_action_search.csv", index=False)
    best = search[search["candidate"].str.startswith("v263a_action_cap")].sort_values(
        ["delta_vs_proxy_base", "action_macro_f1"], ascending=[False, False]
    ).iloc[0]

    summary = {
        "outdir": str(OUTDIR),
        "search_path": str(OUTDIR / "v263a_action_search.csv"),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(features)),
        "proxy_base_action_macro_f1": proxy_base_f1,
        "raw_model_action_macro_f1": raw_model_f1,
        "best_candidate": best["candidate"],
        "best_delta_vs_proxy_base": float(best["delta_vs_proxy_base"]),
        "best_verdict": best["verdict"],
        "generated_submissions": [str(s["path"]) for s in submissions],
        "folds": fold_rows,
        "notes": [
            "No TTMATCH, old-server labels, upload directory writes, or submissions directory writes are used.",
            "OOF delta_vs_proxy_base uses lag0_actionId as a fold-safe simple proxy because V173 OOF is unavailable.",
            "Test action residuals are capped edits to load_v261_cap1_anchor().actionId; point/server remain fixed from that anchor.",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
