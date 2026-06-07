"""V47-V50 action expert factory.

This script trains several deliberately different action experts and stores
OOF/test probabilities for a later top-k meta-stacker.

Experts:
  V47: golden/V64 soft action expert (OOF from historical V64, test from golden soft file)
  V48: rare/control weighted LightGBM action expert
  V49a: familiar-player LightGBM expert with raw player IDs included
  V49b: robust unseen-player LightGBM expert without player IDs and stronger regularization
  V50: short-prefix weighted LightGBM action expert

It also trains one-vs-rest rare detectors for action 8/9/12/14 and stores their
scores as meta-stacker features.
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

from analysis_r1_oof_ensemble import normalize_meta
from analysis_r12_rare_action_rescue import align_validation_features, assign_folds_from_report, binary_weights
from analysis_r7_phase_features import add_phase_features
from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    validate_raw_data,
)
from baseline_v3 import add_remaining_bucket
from generate_r42_golden_soft_blends import build_current_r33_action_prob, normalize_rows, read_golden


OUT_DIR = Path("v47_v50_action_experts")
OLD_V64_DIR = Path(r"C:\aicup\tenis_new\artifacts\cv_v64_ultimate_knowledge_v1")
RARE_CLASSES = [8, 9, 12, 14]
CONTROL_CLASSES = {6, 8, 9, 10, 11, 13}
HARD_NEGATIVE_CLASSES = {3, 6, 10, 11, 13}


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


def aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        if cls in classes:
            out[:, classes.index(cls)] = proba[:, i]
    zero = out.sum(axis=1) <= 0
    if zero.any():
        out[zero] = 1.0 / len(classes)
    return normalize_rows(out)


def add_player_id_features(prefix_df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    first = (
        raw.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)[["rally_uid", "gamePlayerId", "gamePlayerOtherId"]]
        .rename(columns={"gamePlayerId": "server_id", "gamePlayerOtherId": "receiver_id"})
    )
    out = prefix_df.merge(first, on="rally_uid", how="left")
    next_server = out["next_hitter_is_server"].astype(bool)
    out["next_hitter_id"] = np.where(next_server, out["server_id"], out["receiver_id"]).astype(int)
    out["next_receiver_id"] = np.where(next_server, out["receiver_id"], out["server_id"]).astype(int)
    return out


def build_feature_tables(max_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train = add_role_and_score_features(train_raw)
    test = add_role_and_score_features(test_raw)
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, max_lag))
    test_prefix = build_test_prefix_table(test, max_lag)
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    prefix_df = add_player_id_features(prefix_df, train)
    test_prefix = add_player_id_features(test_prefix, test)

    player_id_cols = {"server_id", "receiver_id", "next_hitter_id", "next_receiver_id"}
    safe_features = [c for c in feature_columns(prefix_df) if c != "remaining_len_bucket" and c not in player_id_cols]
    # Familiar-player expert deliberately uses raw player IDs as a separate expert.
    id_features = list(dict.fromkeys(safe_features + ["server_id", "receiver_id", "next_hitter_id", "next_receiver_id"]))
    return prefix_df, test_prefix, safe_features, id_features


def make_action_model(seed: int, profile: str) -> lgb.LGBMClassifier:
    if profile == "robust":
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTION_CLASSES),
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=23,
            min_child_samples=45,
            subsample=0.75,
            colsample_bytree=0.70,
            reg_alpha=0.4,
            reg_lambda=4.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if profile == "familiar":
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTION_CLASSES),
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=39,
            min_child_samples=18,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=22,
        subsample=0.86,
        colsample_bytree=0.86,
        reg_alpha=0.1,
        reg_lambda=1.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def weights_for_expert(y: pd.Series, prefix_len: pd.Series, expert: str) -> np.ndarray:
    y_int = y.astype(int)
    if expert == "rare_control":
        w = class_weight_sample(y_int, beta=0.45)
        w *= np.where(y_int.isin(RARE_CLASSES), 6.0, 1.0)
        w *= np.where(y_int.isin(HARD_NEGATIVE_CLASSES), 1.45, 1.0)
        w *= np.where(y_int.isin(CONTROL_CLASSES), 1.25, 1.0)
    elif expert == "macro":
        w = class_weight_sample(y_int, beta=0.75)
        w *= np.where(y_int.isin(RARE_CLASSES), 2.0, 1.0)
    elif expert == "short":
        w = class_weight_sample(y_int, beta=0.35)
        plen = prefix_len.astype(int)
        w *= np.where(plen <= 1, 5.0, np.where(plen == 2, 3.0, np.where(plen == 3, 1.5, 0.8)))
    else:
        w = class_weight_sample(y_int, beta=0.25)
    return w / np.mean(w)


def train_action_oof(
    prefix_df: pd.DataFrame,
    valid_features: pd.DataFrame,
    meta: pd.DataFrame,
    features: list[str],
    expert: str,
    seed: int,
    profile: str = "standard",
) -> np.ndarray:
    out = np.zeros((len(meta), len(ACTION_CLASSES)), dtype=float)
    valid_rallies_by_fold = {
        fold: set(meta.loc[meta["fold"].eq(fold), "rally_uid"].astype(int).tolist())
        for fold in sorted(meta["fold"].unique())
    }
    for fold in sorted(meta["fold"].unique()):
        valid_mask = meta["fold"].eq(fold).to_numpy()
        train_mask = ~prefix_df["rally_uid"].astype(int).isin(valid_rallies_by_fold[fold])
        train_part = prefix_df.loc[train_mask].copy()
        model = make_action_model(seed + int(fold) * 17, profile)
        model.fit(
            train_part[features],
            train_part["next_actionId"],
            sample_weight=weights_for_expert(train_part["next_actionId"], train_part["prefix_len"], expert),
        )
        out[valid_mask] = aligned_proba(model, valid_features.loc[valid_mask, features], ACTION_CLASSES)
    return out


def train_action_full(prefix_df: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str], expert: str, seed: int, profile: str) -> np.ndarray:
    model = make_action_model(seed, profile)
    model.fit(
        prefix_df[features],
        prefix_df["next_actionId"],
        sample_weight=weights_for_expert(prefix_df["next_actionId"], prefix_df["prefix_len"], expert),
    )
    return aligned_proba(model, test_prefix[features], ACTION_CLASSES)


def make_detector(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
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


def train_rare_detector_oof(prefix_df: pd.DataFrame, valid_features: pd.DataFrame, meta: pd.DataFrame, features: list[str], target: int, seed: int) -> np.ndarray:
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
        model = make_detector(seed + int(fold) * 37 + target)
        model.fit(
            train_part[features],
            y,
            sample_weight=binary_weights(y, detector_hard_negative(train_part["next_actionId"], target)),
        )
        out[valid_mask] = model.predict_proba(valid_features.loc[valid_mask, features])[:, 1]
    return out


def train_rare_detector_full(prefix_df: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str], target: int, seed: int) -> np.ndarray:
    y = prefix_df["next_actionId"].eq(target).astype(int)
    model = make_detector(seed + target)
    model.fit(
        prefix_df[features],
        y,
        sample_weight=binary_weights(y, detector_hard_negative(prefix_df["next_actionId"], target)),
    )
    return model.predict_proba(test_prefix[features])[:, 1]


def load_v64_oof_action(meta: pd.DataFrame) -> np.ndarray:
    usecols = ["rally_uid", "prefix_len"] + [f"seq_action_prob_{i:02d}" for i in range(19)]
    old = pd.read_csv(OLD_V64_DIR / "oof_distilled_features.csv", usecols=usecols)
    merged = meta[["rally_uid", "prefix_len"]].reset_index().merge(old, on=["rally_uid", "prefix_len"], how="left")
    if merged[[f"seq_action_prob_{i:02d}" for i in range(19)]].isna().any().any():
        raise ValueError("V64 OOF soft probabilities did not align.")
    merged = merged.sort_values("index")
    return normalize_rows(merged[[f"seq_action_prob_{i:02d}" for i in range(19)]].to_numpy(dtype=float))


def expert_report(meta: pd.DataFrame, experts: dict[str, np.ndarray]) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for name, prob in experts.items():
        pred = prob.argmax(axis=1)
        rows.append(
            {
                "expert": name,
                "action_macro_f1_argmax": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "action8_pred_count": int((pred == 8).sum()),
                "action9_pred_count": int((pred == 9).sum()),
                "action12_pred_count": int((pred == 12).sum()),
                "action14_pred_count": int((pred == 14).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("action_macro_f1_argmax", ascending=False)


def rare_detector_report(meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    y = meta["next_actionId"].to_numpy(dtype=int)
    rows = []
    for j, cls in enumerate(RARE_CLASSES):
        yy = (y == cls).astype(int)
        rows.append(
            {
                "actionId": cls,
                "support": int(yy.sum()),
                "auc": float(roc_auc_score(yy, scores[:, j])) if yy.sum() else float("nan"),
                "average_precision": float(average_precision_score(yy, scores[:, j])) if yy.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    prefix_df, test_prefix, safe_features, id_features = build_feature_tables()
    v3 = load_pickle("oof_proba_v3.pkl")
    meta = assign_folds_from_report(normalize_meta(v3["valid_meta"]), v3["fold_report"])
    valid_safe = align_validation_features(prefix_df, meta)
    valid_id = align_validation_features(prefix_df, meta)

    # Current and historical/golden branches.
    test_meta, current_test_action, selected = build_current_r33_action_prob()
    golden_test_action, _, _, _ = read_golden(test_meta)
    v64_oof_action = load_v64_oof_action(meta)

    experts_oof: dict[str, np.ndarray] = {
        "v47_v64_oof_soft": v64_oof_action,
    }
    experts_test: dict[str, np.ndarray] = {
        "v47_golden_test_soft": golden_test_action,
    }

    configs = [
        ("v48_rare_control", "rare_control", safe_features, "standard", 4700),
        ("v48_macro_f1_weighted", "macro", safe_features, "standard", 4750),
        ("v49_familiar_player", "standard", id_features, "familiar", 4900),
        ("v49_robust_unseen", "standard", safe_features, "robust", 4950),
        ("v50_short_prefix", "short", safe_features, "standard", 5000),
    ]
    for name, expert, features, profile, seed in configs:
        print(f"training {name} OOF...")
        experts_oof[name] = train_action_oof(prefix_df, valid_safe if features is safe_features else valid_id, meta, features, expert, seed, profile)
        print(f"training {name} full...")
        experts_test[name] = train_action_full(prefix_df, test_prefix, features, expert, seed + 10000, profile)

    print("training rare one-vs-rest detectors...")
    rare_oof = np.column_stack([train_rare_detector_oof(prefix_df, valid_safe, meta, safe_features, cls, 7000) for cls in RARE_CLASSES])
    rare_test = np.column_stack([train_rare_detector_full(prefix_df, test_prefix, safe_features, cls, 9000) for cls in RARE_CLASSES])

    report = expert_report(meta, experts_oof)
    detector = rare_detector_report(meta, rare_oof)
    report.to_csv(OUT_DIR / "v47_v50_expert_oof_report.csv", index=False)
    detector.to_csv(OUT_DIR / "v47_v50_rare_detector_report.csv", index=False)

    artifact = {
        "valid_meta": meta,
        "test_meta": test_meta,
        "selected": selected,
        "current_test_action": current_test_action,
        "current_oof_action_placeholder": None,
        "experts_oof": experts_oof,
        "experts_test": experts_test,
        "rare_classes": RARE_CLASSES,
        "rare_oof_scores": rare_oof,
        "rare_test_scores": rare_test,
        "safe_features": safe_features,
        "id_features": id_features,
    }
    with open(OUT_DIR / "v47_v50_action_experts.pkl", "wb") as f:
        pickle.dump(artifact, f)
    (OUT_DIR / "v47_v50_report.json").write_text(
        json.dumps(
            {
                "experts": list(experts_oof),
                "expert_report": report.to_dict(orient="records"),
                "rare_detector_report": detector.to_dict(orient="records"),
                "artifact": str(OUT_DIR / "v47_v50_action_experts.pkl"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(detector.to_string(index=False))


if __name__ == "__main__":
    main()
