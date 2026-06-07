"""R46 targeted rare-action expert for actionId 8/9/12/14.

The goal is not to replace the action model. It trains one-vs-rest LightGBM
detectors for rare/control/defensive actions and uses them only as conservative
logit boosts on top of the current public-positive R42 action blend.

Targets:
  8  pimple's long push
  9  pimple's fast push
  12 chop
  14 lob
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r12_rare_action_rescue import align_validation_features, assign_folds_from_report, binary_weights
from analysis_r7_phase_features import add_phase_features
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers
from generate_r42_golden_soft_blends import (
    CURRENT_SUB_PATH,
    UPLOAD_DIR,
    build_current_r33_action_prob,
    normalize_rows,
    read_golden,
)


OUT_DIR = Path("r46_rare_action_expert")
RARE_CLASSES = [8, 9, 12, 14]
CLASS_NAMES = {
    8: "pimple_long_push",
    9: "pimple_fast_push",
    12: "chop",
    14: "lob",
}


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_detector(seed: int, n_estimators: int = 220) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=18,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=1.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def detector_hard_negative(y_action: pd.Series, target: int) -> pd.Series:
    hard = {
        8: {9, 10, 11, 13, 6},
        9: {8, 10, 11, 13, 6, 12},
        12: {9, 10, 13, 14},
        14: {3, 12, 13},
    }.get(target, set())
    return y_action.isin(hard)


def train_detector_oof(
    prefix_df: pd.DataFrame,
    valid_features: pd.DataFrame,
    meta: pd.DataFrame,
    features: list[str],
    target: int,
    seed: int,
) -> np.ndarray:
    out = np.zeros(len(meta), dtype=float)
    valid_rallies_by_fold = {
        fold: set(meta.loc[meta["fold"].eq(fold), "rally_uid"].astype(int).tolist())
        for fold in sorted(meta["fold"].unique())
    }
    for fold in sorted(meta["fold"].unique()):
        valid_mask = meta["fold"].eq(fold).to_numpy()
        train_mask = ~prefix_df["rally_uid"].astype(int).isin(valid_rallies_by_fold[fold])
        train_part = prefix_df.loc[train_mask].copy()
        y = train_part["next_actionId"].eq(target).astype(int)
        model = make_detector(seed + int(fold) * 101 + target)
        model.fit(
            train_part[features],
            y,
            sample_weight=binary_weights(y, detector_hard_negative(train_part["next_actionId"], target)),
        )
        out[valid_mask] = model.predict_proba(valid_features.loc[valid_mask, features])[:, 1]
    return out


def train_detector_full(prefix_df: pd.DataFrame, features: list[str], target: int, seed: int) -> lgb.LGBMClassifier:
    y = prefix_df["next_actionId"].eq(target).astype(int)
    model = make_detector(seed + target)
    model.fit(
        prefix_df[features],
        y,
        sample_weight=binary_weights(y, detector_hard_negative(prefix_df["next_actionId"], target)),
    )
    return model


def build_oof_current_action() -> tuple[pd.DataFrame, np.ndarray, dict]:
    v3 = load_pickle("oof_proba_v3.pkl")
    v5 = load_pickle("oof_proba_v5.pkl")
    v7 = load_pickle("oof_proba_v7.pkl")
    r7 = load_pickle("oof_proba_r7.pkl")
    selected = json.loads(Path("r33_safe_oof_ensemble/r33_selected.json").read_text(encoding="utf-8"))
    meta = assign_folds_from_report(normalize_meta(v3["valid_meta"]), v3["fold_report"])
    for name, oof in [("V5", v5), ("V7", v7), ("R7", r7)]:
        assert_aligned(meta.drop(columns=["fold"]), oof["valid_meta"], name)
    r7_action, _, _ = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    current_action = normalize_rows(0.85 * r1_action + 0.05 * r7_action + 0.10 * v5["gru_action"])
    return meta, current_action, selected


def build_prefix_features(max_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train = add_role_and_score_features(train_raw)
    test = add_role_and_score_features(test_raw)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, max_lag))
    test_prefix = build_test_prefix_table(test, max_lag)
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket"]
    return prefix_df, test_prefix, features


def detector_report(meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for j, cls in enumerate(RARE_CLASSES):
        yy = (y == cls).astype(int)
        rows.append(
            {
                "actionId": cls,
                "name": CLASS_NAMES[cls],
                "support": int(yy.sum()),
                "auc": float(roc_auc_score(yy, scores[:, j])) if yy.sum() and yy.sum() < len(yy) else float("nan"),
                "average_precision": float(average_precision_score(yy, scores[:, j])) if yy.sum() else float("nan"),
                "positive_mean": float(scores[yy == 1, j].mean()) if yy.sum() else float("nan"),
                "negative_mean": float(scores[yy == 0, j].mean()),
                "positive_p90": float(np.quantile(scores[yy == 1, j], 0.90)) if yy.sum() else float("nan"),
                "negative_p99": float(np.quantile(scores[yy == 0, j], 0.99)),
            }
        )
    return pd.DataFrame(rows)


def action_ranks(prob: np.ndarray) -> np.ndarray:
    order = np.argsort(-prob, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(len(prob))[:, None], order] = np.arange(prob.shape[1])[None, :] + 1
    return ranks


def boost_probs(
    base_prob: np.ndarray,
    rare_scores: np.ndarray,
    thresholds: np.ndarray,
    gamma: float,
    rank_max: int,
    min_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    prob = base_prob.copy()
    ranks = action_ranks(base_prob)
    touched = np.zeros(len(prob), dtype=bool)
    for j, cls in enumerate(RARE_CLASSES):
        mask = (rare_scores[:, j] >= thresholds[j]) & (ranks[:, cls] <= rank_max) & (base_prob[:, cls] >= min_prob)
        prob[mask, cls] *= gamma
        touched |= mask
    return normalize_rows(prob), touched


def search_oof(meta: pd.DataFrame, base_prob: np.ndarray, selected: dict, rare_scores: np.ndarray) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    base_pred = apply_segmented_multipliers(meta, base_prob, selected["action_multipliers"], ACTION_CLASSES, "two")
    base_f1 = f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    rows = []
    for q in [0.80, 0.85, 0.90, 0.95, 0.975, 0.99]:
        thresholds = np.quantile(rare_scores, q, axis=0)
        for gamma in [1.25, 1.50, 2.00, 3.00, 5.00, 8.00]:
            for rank_max in [3, 5, 8, 19]:
                for min_prob in [0.001, 0.005, 0.010, 0.020, 0.040]:
                    boosted, touched = boost_probs(base_prob, rare_scores, thresholds, gamma, rank_max, min_prob)
                    pred = apply_segmented_multipliers(meta, boosted, selected["action_multipliers"], ACTION_CLASSES, "two")
                    churn = float((pred != base_pred).mean())
                    if churn > 0.12:
                        continue
                    action_f1 = f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
                    rows.append(
                        {
                            "q": q,
                            "gamma": gamma,
                            "rank_max": rank_max,
                            "min_prob": min_prob,
                            "action_macro_f1": float(action_f1),
                            "gain_vs_base": float(action_f1 - base_f1),
                            "churn_vs_base": churn,
                            "touched_rate": float(touched.mean()),
                            "pred8_count": int((pred == 8).sum()),
                            "pred9_count": int((pred == 9).sum()),
                            "pred12_count": int((pred == 12).sum()),
                            "pred14_count": int((pred == 14).sum()),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["action_macro_f1", "gain_vs_base"], ascending=False)


def write_submission(
    test_meta: pd.DataFrame,
    action_prob: np.ndarray,
    selected: dict,
    current_sub: pd.DataFrame,
    name: str,
) -> dict:
    action_pred = apply_segmented_multipliers(test_meta, action_prob, selected["action_multipliers"], ACTION_CLASSES, "two")
    current_action = current_sub["actionId"].to_numpy(dtype=int)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "action_diff_vs_current_r34": float(np.mean(action_pred != current_action)),
        "action8_count": int((action_pred == 8).sum()),
        "action9_count": int((action_pred == 9).sum()),
        "action12_count": int((action_pred == 12).sum()),
        "action14_count": int((action_pred == 14).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    prefix_df, test_prefix, features = build_prefix_features()
    meta, current_oof_prob, selected = build_oof_current_action()
    valid_features = align_validation_features(prefix_df, meta)

    oof_scores = np.column_stack(
        [train_detector_oof(prefix_df, valid_features, meta, features, cls, 5400) for cls in RARE_CLASSES]
    )
    det = detector_report(meta, oof_scores)
    det.to_csv(OUT_DIR / "r46_detector_report.csv", index=False)

    search = search_oof(meta, current_oof_prob, selected, oof_scores)
    search.to_csv(OUT_DIR / "r46_oof_boost_search.csv", index=False)

    # Full-test rare detector scores.
    test_features = test_prefix[["rally_uid", "match"] + features].copy()
    test_scores = []
    for cls in RARE_CLASSES:
        model = train_detector_full(prefix_df, features, cls, 8800)
        test_scores.append(model.predict_proba(test_features[features])[:, 1])
    test_scores = np.column_stack(test_scores)
    np.save(OUT_DIR / "r46_test_rare_scores.npy", test_scores)

    test_meta, current_test_prob, selected_test = build_current_r33_action_prob()
    golden_action, _, _, _ = read_golden(test_meta)
    r42_base_prob = normalize_rows(0.80 * current_test_prob + 0.20 * golden_action)
    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current submission did not align.")

    generated = []
    if len(search):
        # Generate the top OOF policies and a few safer variants. Thresholds are recomputed on test scores
        # by quantile so the policy uses distributional selectivity rather than validation labels.
        top = search.head(6).copy()
        for _, row in top.iterrows():
            thresholds = np.quantile(test_scores, float(row["q"]), axis=0)
            boosted, touched = boost_probs(
                r42_base_prob,
                test_scores,
                thresholds,
                float(row["gamma"]),
                int(row["rank_max"]),
                float(row["min_prob"]),
            )
            name = (
                "submission_r46_rareexpert"
                f"_q{str(row['q']).replace('.', 'p')}"
                f"_g{str(row['gamma']).replace('.', 'p')}"
                f"_rk{int(row['rank_max'])}"
                f"_mp{str(row['min_prob']).replace('.', 'p')}.csv"
            )
            info = write_submission(test_meta, boosted, selected_test, current_sub, name)
            info.update(
                {
                    "source_oof_action_f1": float(row["action_macro_f1"]),
                    "source_oof_gain": float(row["gain_vs_base"]),
                    "source_oof_churn": float(row["churn_vs_base"]),
                    "test_touched_rate": float(touched.mean()),
                }
            )
            generated.append(info)

    # Conservative hand-picked probes even if OOF top is too aggressive.
    for q, gamma, rank_max, min_prob in [(0.95, 1.5, 5, 0.005), (0.975, 2.0, 5, 0.005), (0.99, 3.0, 8, 0.001)]:
        thresholds = np.quantile(test_scores, q, axis=0)
        boosted, touched = boost_probs(r42_base_prob, test_scores, thresholds, gamma, rank_max, min_prob)
        name = (
            "submission_r46_rareexpert_safe"
            f"_q{str(q).replace('.', 'p')}"
            f"_g{str(gamma).replace('.', 'p')}"
            f"_rk{rank_max}"
            f"_mp{str(min_prob).replace('.', 'p')}.csv"
        )
        info = write_submission(test_meta, boosted, selected_test, current_sub, name)
        info.update({"manual_probe": True, "test_touched_rate": float(touched.mean())})
        generated.append(info)

    pd.DataFrame(generated).to_csv(OUT_DIR / "r46_generated_candidates.csv", index=False)
    report = {
        "rare_classes": {str(k): CLASS_NAMES[k] for k in RARE_CLASSES},
        "detector_report": str(OUT_DIR / "r46_detector_report.csv"),
        "oof_search": str(OUT_DIR / "r46_oof_boost_search.csv"),
        "generated": generated,
        "base_policy": "R42 public-positive action blend w=0.20 + current point/server",
        "public_anchor": {"submission": "submission_r42_golden_action_w0p2_current_point_server.csv", "pl": 0.3342886},
    }
    (OUT_DIR / "r46_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(det.to_string(index=False))
    print(search.head(10).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
