"""R19 transductive observed-style profile audit.

This experiment revisits player/style profiles, but unlike V9 it does not use
train-history-only player profiles as the main signal. Instead it simulates the
test setting: the validation fold gets style statistics from the observed
prefix strokes only, plus fold-train public history. Raw player IDs are not fed
to the model; only smoothed rate/count features are used.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r7_phase_features import add_phase_features
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    POINT_NONTERMINAL_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    make_lgbm,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v2 import blend_probs
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers, tune_segmented_multipliers


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R19 observed-style profile audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--cv-report", default="cv_report_r19.csv")
    parser.add_argument("--search-report", default="r19_style_profile_search.csv")
    parser.add_argument("--selected", default="r19_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r19.json")
    parser.add_argument("--oof-proba", default="oof_proba_r19.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--smooth-k", type=float, default=50.0)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def add_server_receiver_ids(prefix_df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    first = (
        raw.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)[["rally_uid", "gamePlayerId", "gamePlayerOtherId"]]
        .rename(columns={"gamePlayerId": "server_id", "gamePlayerOtherId": "receiver_id"})
    )
    out = prefix_df.merge(first, on="rally_uid", how="left")
    next_server = out["next_hitter_is_server"].astype(bool)
    out["next_hitter_id_tmp"] = np.where(next_server, out["server_id"], out["receiver_id"]).astype(int)
    out["next_receiver_id_tmp"] = np.where(next_server, out["receiver_id"], out["server_id"]).astype(int)
    return out


def build_profile_tables(corpus: pd.DataFrame, smooth_k: float) -> dict[str, pd.DataFrame]:
    corpus = corpus.copy()
    action_vals = list(range(19))
    point_vals = list(range(10))
    spin_vals = list(range(6))
    hand_vals = list(range(3))
    strength_vals = list(range(4))
    global_priors = {}
    for field, vals in [
        ("actionId", action_vals),
        ("pointId", point_vals),
        ("spinId", spin_vals),
        ("handId", hand_vals),
        ("strengthId", strength_vals),
    ]:
        counts = corpus[field].value_counts().reindex(vals, fill_value=0).to_numpy(dtype=float)
        global_priors[field] = (counts + 1.0) / (counts.sum() + len(vals))

    def table_for(role_col: str, prefix: str, specs: list[tuple[str, list[int]]]) -> pd.DataFrame:
        players = pd.DataFrame({"player_id": sorted(corpus[role_col].astype(int).unique().tolist())})
        total = corpus.groupby(role_col).size().rename(f"{prefix}_seen_count").reset_index().rename(columns={role_col: "player_id"})
        out = players.merge(total, on="player_id", how="left")
        out[f"{prefix}_seen_count"] = out[f"{prefix}_seen_count"].fillna(0).astype(float)
        for field, vals in specs:
            ct = pd.crosstab(corpus[role_col].astype(int), corpus[field].astype(int)).reindex(columns=vals, fill_value=0)
            ct = ct.reindex(out["player_id"], fill_value=0)
            denom = out[f"{prefix}_seen_count"].to_numpy(dtype=float)[:, None] + smooth_k
            rates = (ct.to_numpy(dtype=float) + smooth_k * global_priors[field][None, :]) / denom
            for i, val in enumerate(vals):
                out[f"{prefix}_{field}_{val}_rate"] = rates[:, i]
        out[f"{prefix}_log_seen_count"] = np.log1p(out[f"{prefix}_seen_count"].to_numpy(dtype=float))
        return out

    hitter = table_for(
        "gamePlayerId",
        "style_hitter",
        [("actionId", action_vals), ("pointId", point_vals), ("spinId", spin_vals), ("handId", hand_vals), ("strengthId", strength_vals)],
    )
    receiver = table_for("gamePlayerOtherId", "style_receiver", [("pointId", point_vals), ("spinId", spin_vals)])
    return {"hitter": hitter, "receiver": receiver}


def attach_profiles(prefix_df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = prefix_df.copy()
    out = out.merge(tables["hitter"], left_on="next_hitter_id_tmp", right_on="player_id", how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(tables["receiver"], left_on="next_receiver_id_tmp", right_on="player_id", how="left")
    out = out.drop(columns=["player_id"])
    profile_cols = [c for c in out.columns if c.startswith("style_")]
    for col in profile_cols:
        if out[col].isna().any():
            if col.endswith("_seen_count") or col.endswith("_log_seen_count"):
                out[col] = out[col].fillna(0.0)
            else:
                out[col] = out[col].fillna(out[col].mean())
    return out.drop(columns=["server_id", "receiver_id", "next_hitter_id_tmp", "next_receiver_id_tmp"])


def fit_models(train_df: pd.DataFrame, features: list[str], seed: int, n_estimators: int):
    action = make_lgbm("multiclass", n_estimators, seed, num_class=len(ACTION_CLASSES))
    action.fit(train_df[features], train_df["next_actionId"], sample_weight=class_weight_sample(train_df["next_actionId"]))
    terminal = make_lgbm("binary", n_estimators, seed + 1)
    terminal.fit(train_df[features], train_df["next_is_terminal"])
    point_train = train_df[train_df["next_pointId"].isin(POINT_NONTERMINAL_CLASSES)].copy()
    point = make_lgbm("multiclass", n_estimators, seed + 2, num_class=len(POINT_NONTERMINAL_CLASSES))
    point.fit(point_train[features], point_train["next_pointId"], sample_weight=class_weight_sample(point_train["next_pointId"]))
    return action, terminal, point


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


def predict_models(models, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    action, terminal, point = models
    x = df[features]
    action_prob = aligned_proba(action, x, ACTION_CLASSES)
    term = terminal.predict_proba(x)[:, 1]
    term = np.clip(term, 1e-6, 1.0 - 1e-6)
    point_nt = aligned_proba(point, x, POINT_NONTERMINAL_CLASSES)
    point_prob = np.zeros((len(df), len(POINT_CLASSES)), dtype=float)
    point_prob[:, 0] = term
    point_prob[:, 1:] = (1.0 - term[:, None]) * point_nt
    return action_prob, normalize_rows(point_prob)


def observed_rows_for_valid(raw: pd.DataFrame, fold_valid: pd.DataFrame) -> pd.DataFrame:
    limits = fold_valid[["rally_uid", "prefix_len"]].rename(columns={"prefix_len": "observed_prefix_len"})
    rows = raw.merge(limits, on="rally_uid", how="inner")
    return rows[rows["strikeNumber"].le(rows["observed_prefix_len"])].drop(columns=["observed_prefix_len"])


def cv_oof(prefix_df: pd.DataFrame, raw: pd.DataFrame, test_lengths: np.ndarray, base_features: list[str], args):
    rally_meta = prefix_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    parts = {"valid_meta": [], "action": [], "point": []}
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[train_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"])
        fold_train_base = prefix_df[prefix_df["rally_uid"].isin(train_rallies)].copy()
        valid_pool = prefix_df[prefix_df["rally_uid"].isin(valid_rallies)].copy()
        sampled = sample_validation_prefixes(valid_pool, test_lengths, args.seed + fold)
        fold_valid_base = valid_pool.loc[sampled].copy()

        train_raw = raw[raw["rally_uid"].isin(train_rallies)]
        valid_observed_raw = observed_rows_for_valid(raw, fold_valid_base)
        train_tables = build_profile_tables(train_raw, args.smooth_k)
        valid_tables = build_profile_tables(pd.concat([train_raw, valid_observed_raw], ignore_index=True), args.smooth_k)
        fold_train = attach_profiles(fold_train_base, train_tables)
        fold_valid = attach_profiles(fold_valid_base, valid_tables)
        features = [c for c in fold_train.columns if c in fold_valid.columns and c not in {
            "rally_uid","match","next_actionId","next_pointId","next_is_terminal","serverGetPoint",
            "remaining_len","final_parity_even","num_prefixes_in_rally","remaining_len_bucket",
        }]
        features = [c for c in features if "PlayerId" not in c and c not in {"server_id","receiver_id"}]
        models = fit_models(fold_train, features, args.seed + fold * 31, args.n_estimators)
        action, point = predict_models(models, fold_valid, features)
        parts["valid_meta"].append(fold_valid_base[["rally_uid","match","prefix_len","next_actionId","next_pointId","serverGetPoint"]].reset_index(drop=True))
        parts["action"].append(action)
        parts["point"].append(point)
        rows.append({
            "fold": fold,
            "train_rows": len(fold_train),
            "valid_rows": len(fold_valid),
            "observed_valid_strokes": len(valid_observed_raw),
            "feature_count": len(features),
            "action_argmax": float(f1_score(fold_valid["next_actionId"], np.asarray(ACTION_CLASSES)[np.argmax(action,axis=1)], average="macro", labels=ACTION_CLASSES, zero_division=0)),
            "point_argmax": float(f1_score(fold_valid["next_pointId"], np.asarray(POINT_CLASSES)[np.argmax(point,axis=1)], average="macro", labels=POINT_CLASSES, zero_division=0)),
        })
        print(f"fold {fold}: A={rows[-1]['action_argmax']:.4f} P={rows[-1]['point_argmax']:.4f} features={len(features)}")
    return {
        "valid_meta": pd.concat(parts["valid_meta"], ignore_index=True),
        "fold_report": pd.DataFrame(rows),
        "style_action": np.vstack(parts["action"]),
        "style_point": np.vstack(parts["point"]),
    }


def build_safe_server(v3, v5, v7, v10, selected_v10) -> np.ndarray:
    _, _, v3_server = compose_v3(v3)
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    return (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(selected_v10["server_v10_weight"]) * v10["v10_server"]


def search(oof, v3, v5, v7, v10, selected_v10):
    meta = normalize_meta(oof["valid_meta"])
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    _, v3_point, _ = compose_v3(v3)
    server = build_safe_server(v3, v5, v7, v10, selected_v10)
    server_auc = roc_auc_score(meta["serverGetPoint"], server)
    rows = []
    best = None
    base_action_mult = tune_segmented_multipliers(meta, r1_action, ACTION_CLASSES, "action", "two")
    base_action_pred = apply_segmented_multipliers(meta, r1_action, base_action_mult, ACTION_CLASSES, "two")
    for aw in [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]:
        action = blend_probs(r1_action, oof["style_action"], aw)
        action_mult = tune_segmented_multipliers(meta, action, ACTION_CLASSES, "action", "two")
        action_pred = apply_segmented_multipliers(meta, action, action_mult, ACTION_CLASSES, "two")
        action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        churn = float((action_pred != base_action_pred).mean())
        for pw in [0, 0.02, 0.05, 0.1, 0.2]:
            point = blend_probs(v3_point, oof["style_point"], pw)
            point_pred = apply_segmented_multipliers(meta, point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
            point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
            overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
            row = {"action_blend":aw,"point_blend":pw,"action_macro_f1":float(action_f1),"point_macro_f1":float(point_f1),"server_auc":float(server_auc),"overall":float(overall),"action_churn_vs_r1":churn}
            rows.append(row)
            eligible = overall >= 0.316 and churn <= 0.08 and point_f1 >= 0.203
            if eligible and (best is None or overall > best["overall"]):
                best = dict(row)
                best["action_multipliers"] = action_mult
    report = pd.DataFrame(rows).sort_values("overall", ascending=False)
    if best is None:
        best = report.iloc[0].to_dict()
        best["submit_recommendation"] = False
        best["selected_policy"] = "diagnostic_only"
    else:
        best["submit_recommendation"] = True
        best["selected_policy"] = "observed_style_profile"
    return report, best


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    test_prefix = build_train_prefix_table(test.assign(serverGetPoint=0), args.max_lag)
    prefix = add_server_receiver_ids(add_phase_features(prefix, train), train)
    test_lengths = test.groupby("rally_uid").size().to_numpy(dtype=int)
    oof = cv_oof(prefix, train, test_lengths, feature_columns(prefix), args)
    with open(args.oof_proba, "wb") as f:
        pickle.dump(oof, f)
    oof["fold_report"].to_csv(args.cv_report, index=False)
    v3, v5, v7, v10 = [load_pickle(p) for p in [args.v3_oof,args.v5_oof,args.v7_oof,args.v10b_oof]]
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    assert_aligned(normalize_meta(v3["valid_meta"]), oof["valid_meta"], "R19")
    report, selected = search(oof, v3, v5, v7, v10, selected_v10)
    report.to_csv(args.search_report, index=False)
    out = {"selected": selected, "protocol": "fold-safe observed-style profiles from fold-train plus valid observed prefix"}
    Path(args.selected).write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path(args.feature_report).write_text(json.dumps({"selected": out, "cv_report": args.cv_report, "search_report": args.search_report}, indent=2), encoding="utf-8")
    print("R19 selected:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
