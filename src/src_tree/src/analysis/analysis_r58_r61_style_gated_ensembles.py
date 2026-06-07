"""R58-R61 style-aware action ensembles.

R57 showed that player-style clustering is a strong action signal, but high
weights cause large churn. This script keeps point/server fixed and explores
safer ways to use the style expert:

- R58: trust/confidence gated style blend.
- R59: class-aware style blend for weak/style-sensitive action classes.
- R60: seen/unseen/observed-count segmented style weights.
- R61: combine R56 low-action experts with R57 style expert.
"""

from __future__ import annotations

import ast
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r56_low_action_class_experts import blend_columns
from analysis_r57_player_style_clustering import (
    StyleEncoder,
    add_player_id_features,
    add_style_features,
    aligned_action_proba,
    make_action_model,
    observed_rows_for_prefixes,
)
from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r58_r61_style_gated_ensembles")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R57_DIR = Path("r57_player_style_clustering")
R56_DIR = Path("r56_low_action_class_experts")
PRIMARY_STYLE = "transductive_k8"
STYLE_VARIANTS = [
    ("train_only_k8", 8, False),
    ("train_only_k16", 16, False),
    ("transductive_k8", 8, True),
    ("transductive_k16", 16, True),
]


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


def load_artifact() -> dict:
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


def row_blend(base: np.ndarray, expert: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    return normalize_rows((1.0 - w) * base + w * expert)


def row_class_blend(base: np.ndarray, expert: np.ndarray, weights: np.ndarray, classes: list[int]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    out = base.copy()
    for cls in classes:
        out[:, cls] = (1.0 - w) * base[:, cls] + w * expert[:, cls]
    return normalize_rows(out)


def describe_prob(name: str, prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, mult: dict, extra: dict | None = None) -> dict:
    pred = apply_action(prob, meta, mult)
    row = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "churn_vs_r42": float(np.mean(pred != base_pred)),
        "pred0_count": int((pred == 0).sum()),
        "pred4_count": int((pred == 4).sum()),
        "pred8_count": int((pred == 8).sum()),
        "pred9_count": int((pred == 9).sum()),
        "pred11_count": int((pred == 11).sum()),
        "pred12_count": int((pred == 12).sum()),
        "pred14_count": int((pred == 14).sum()),
    }
    if extra:
        row.update(extra)
    return row


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str, extra: dict | None = None) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    row = {
        "candidate": name,
        "path": str(path),
        "upload_path": str(UPLOAD_DIR / name),
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
    }
    if extra:
        row.update(extra)
    return row


def prepare_prefix_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_raw0 = pd.read_csv("train.csv")
    test_raw0 = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw0, test_raw0)
    train_raw = add_role_and_score_features(train_raw0)
    test_raw = add_role_and_score_features(test_raw0)
    prefix_base = add_remaining_bucket(build_train_prefix_table(train_raw, max_lag=6))
    test_prefix_base = build_test_prefix_table(test_raw, max_lag=6)
    prefix_base = add_phase_features(prefix_base, train_raw)
    test_prefix_base = add_phase_features(test_prefix_base, test_raw)
    prefix_base = add_player_id_features(prefix_base, train_raw)
    test_prefix_base = add_player_id_features(test_prefix_base, test_raw)
    player_cols = {"server_id", "receiver_id", "next_hitter_id", "next_receiver_id"}
    base_features = [c for c in feature_columns(prefix_base) if c != "remaining_len_bucket" and c not in player_cols]
    return train_raw, test_raw, prefix_base, test_prefix_base, base_features


def reconstruct_oof_meta_and_style(
    train_raw: pd.DataFrame,
    prefix_base: pd.DataFrame,
    test_prefix_base: pd.DataFrame,
    base_features: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    rally_meta = prefix_base[["rally_uid", "match"]].drop_duplicates().reset_index(drop=True)
    test_lengths = test_prefix_base["prefix_len"].to_numpy(dtype=int)
    meta_parts = []
    style_parts = {name: [] for name, _, _ in STYLE_VARIANTS}
    segment_parts = []

    for fold, (tr_rally_idx, va_rally_idx) in enumerate(GroupKFold(n_splits=5).split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[tr_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[va_rally_idx]["rally_uid"])
        valid_pool = prefix_base[prefix_base["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_lengths, 42 + fold)
        va = valid_pool.loc[sampled_idx].copy().reset_index(drop=True)
        meta_parts.append(va[["rally_uid", "match", "prefix_len", "next_actionId", "next_hitter_id", "next_receiver_id"]])

        train_obs_raw = train_raw[train_raw["rally_uid"].isin(train_rallies)].copy()
        valid_obs_raw = observed_rows_for_prefixes(train_raw[train_raw["rally_uid"].isin(valid_rallies)], va)
        train_players = set(pd.concat([train_obs_raw["gamePlayerId"], train_obs_raw["gamePlayerOtherId"]]).dropna().astype(int))
        valid_counts = pd.concat([valid_obs_raw["gamePlayerId"], valid_obs_raw["gamePlayerOtherId"]]).dropna().astype(int).value_counts()
        seg = pd.DataFrame(
            {
                "hitter_seen": va["next_hitter_id"].isin(train_players).astype(int).to_numpy(),
                "receiver_seen": va["next_receiver_id"].isin(train_players).astype(int).to_numpy(),
                "hitter_obs_count": va["next_hitter_id"].map(valid_counts).fillna(0).astype(int).to_numpy(),
                "receiver_obs_count": va["next_receiver_id"].map(valid_counts).fillna(0).astype(int).to_numpy(),
                "prefix_len": va["prefix_len"].to_numpy(dtype=int),
            }
        )
        segment_parts.append(seg)

        tr = prefix_base[prefix_base["rally_uid"].isin(train_rallies)].copy().reset_index(drop=True)
        for variant_name, k, transductive in STYLE_VARIANTS:
            observed_for_style = pd.concat([train_obs_raw, valid_obs_raw], ignore_index=True) if transductive else train_obs_raw
            encoder = StyleEncoder(k=k, alpha=25.0, beta=25.0, seed=5700 + fold + k).fit(observed_for_style, train_obs_raw)
            va_style = add_style_features(va, encoder)
            style_cols = [c for c in va_style.columns if c.startswith("style_")]
            style_parts[variant_name].append(va_style[style_cols].reset_index(drop=True))

    meta = pd.concat(meta_parts, ignore_index=True)
    style_oof = {name: pd.concat(parts, ignore_index=True) for name, parts in style_parts.items()}
    segment_oof = pd.concat(segment_parts, ignore_index=True)
    for name, df in style_oof.items():
        df["style_min_trust"] = df[["style_hitter_trust", "style_receiver_trust"]].min(axis=1)
        df["style_mean_confidence"] = df[["style_hitter_confidence", "style_receiver_confidence"]].mean(axis=1)
        df["style_mean_n_strokes"] = df[["style_hitter_n_strokes", "style_receiver_n_strokes"]].mean(axis=1)
        df["style_min_n_strokes"] = df[["style_hitter_n_strokes", "style_receiver_n_strokes"]].min(axis=1)
    oof_probs = {name: np.load(R57_DIR / f"{name}_oof_action.npy") for name, _, _ in STYLE_VARIANTS}
    return meta.reset_index(drop=True), style_oof, oof_probs, {}, {"segments": segment_oof.reset_index(drop=True)}


def build_test_style_probs(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    prefix_base: pd.DataFrame,
    test_prefix_base: pd.DataFrame,
    base_features: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], pd.DataFrame]:
    full_train_obs = train_raw.copy()
    test_obs = test_raw.copy()
    test_probs: dict[str, np.ndarray] = {}
    test_styles: dict[str, pd.DataFrame] = {}
    test_counts = pd.concat([test_obs["gamePlayerId"], test_obs["gamePlayerOtherId"]]).dropna().astype(int).value_counts()
    train_players = set(pd.concat([train_raw["gamePlayerId"], train_raw["gamePlayerOtherId"]]).dropna().astype(int))
    test_segments = pd.DataFrame(
        {
            "hitter_seen": test_prefix_base["next_hitter_id"].isin(train_players).astype(int).to_numpy(),
            "receiver_seen": test_prefix_base["next_receiver_id"].isin(train_players).astype(int).to_numpy(),
            "hitter_obs_count": test_prefix_base["next_hitter_id"].map(test_counts).fillna(0).astype(int).to_numpy(),
            "receiver_obs_count": test_prefix_base["next_receiver_id"].map(test_counts).fillna(0).astype(int).to_numpy(),
            "prefix_len": test_prefix_base["prefix_len"].to_numpy(dtype=int),
        }
    )
    for variant_name, k, transductive in STYLE_VARIANTS:
        observed = pd.concat([full_train_obs, test_obs], ignore_index=True) if transductive else full_train_obs
        encoder = StyleEncoder(k=k, alpha=25.0, beta=25.0, seed=6700 + k).fit(observed, full_train_obs)
        train_style = add_style_features(prefix_base, encoder)
        test_style = add_style_features(test_prefix_base, encoder)
        style_cols = [c for c in train_style.columns if c.startswith("style_")]
        features = base_features + style_cols
        model = make_action_model(seed=6800 + k + (100 if transductive else 0))
        model.fit(train_style[features], train_style["next_actionId"], sample_weight=class_weight_sample(train_style["next_actionId"]))
        test_probs[variant_name] = aligned_action_proba(model, test_style[features])
        df = test_style[style_cols].reset_index(drop=True)
        df["style_min_trust"] = df[["style_hitter_trust", "style_receiver_trust"]].min(axis=1)
        df["style_mean_confidence"] = df[["style_hitter_confidence", "style_receiver_confidence"]].mean(axis=1)
        df["style_mean_n_strokes"] = df[["style_hitter_n_strokes", "style_receiver_n_strokes"]].mean(axis=1)
        df["style_min_n_strokes"] = df[["style_hitter_n_strokes", "style_receiver_n_strokes"]].min(axis=1)
        test_styles[variant_name] = df
    return test_probs, test_styles, test_segments


def trust_weights(style_df: pd.DataFrame, seg_df: pd.DataFrame, base_w: float, trust_min: float, conf_min: float, ent_max: float, min_prefix: int) -> np.ndarray:
    mask = (
        (style_df["style_min_trust"].to_numpy(float) >= trust_min)
        & (style_df["style_mean_confidence"].to_numpy(float) >= conf_min)
        & (style_df["style_hitter_receiver_entropy_sum"].to_numpy(float) <= ent_max)
        & (seg_df["prefix_len"].to_numpy(int) >= min_prefix)
    )
    return np.where(mask, base_w, 0.0)


def segment_weights(seg_df: pd.DataFrame, w_seen: float, w_unseen_many: float, w_unseen_few: float, many_thr: int) -> np.ndarray:
    hitter_seen = seg_df["hitter_seen"].to_numpy(int).astype(bool)
    obs = seg_df["hitter_obs_count"].to_numpy(int)
    weights = np.full(len(seg_df), w_unseen_few, dtype=float)
    weights[(~hitter_seen) & (obs >= many_thr)] = w_unseen_many
    weights[hitter_seen] = w_seen
    return weights


def parse_r56_choices(label: str) -> dict[int, tuple[str, float]]:
    search = pd.read_csv(R56_DIR / "r56_oof_class_blend_search.csv")
    row = search[search["candidate"] == label]
    if row.empty:
        raise ValueError(f"Missing R56 candidate {label}")
    raw = row.iloc[0]["choices"]
    parsed = ast.literal_eval(raw) if isinstance(raw, str) else {}
    return {int(k): (str(v["expert"]), float(v["weight"])) for k, v in parsed.items()}


def make_r56_probs(art: dict, labels: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    experts_oof = {
        "r42_base": r42_oof,
        "current": current_oof,
        "golden": golden_oof,
        "v49_familiar": art["experts_oof"]["v49_familiar_player"],
        "v50_short": art["experts_oof"]["v50_short_prefix"],
        "v49_robust": art["experts_oof"]["v49_robust_unseen"],
        "v48_macro": art["experts_oof"]["v48_macro_f1_weighted"],
        "v48_rare": art["experts_oof"]["v48_rare_control"],
    }
    experts_test = {
        "r42_base": r42_test,
        "current": current_test,
        "golden": golden_test,
        "v49_familiar": art["experts_test"]["v49_familiar_player"],
        "v50_short": art["experts_test"]["v50_short_prefix"],
        "v49_robust": art["experts_test"]["v49_robust_unseen"],
        "v48_macro": art["experts_test"]["v48_macro_f1_weighted"],
        "v48_rare": art["experts_test"]["v48_rare_control"],
    }
    out_oof = {}
    out_test = {}
    for label in labels:
        choices = parse_r56_choices(label)
        out_oof[label] = blend_columns(r42_oof, experts_oof, choices)
        out_test[label] = blend_columns(r42_test, experts_test, choices)
    return out_oof, out_test


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    art = load_artifact()
    train_raw, test_raw, prefix_base, test_prefix_base, base_features = prepare_prefix_tables()
    meta, style_oof, r57_oof_probs, _, aux_oof = reconstruct_oof_meta_and_style(train_raw, prefix_base, test_prefix_base, base_features)
    test_probs, test_styles, test_segments = build_test_style_probs(train_raw, test_raw, prefix_base, test_prefix_base, base_features)
    seg_oof = aux_oof["segments"]

    y = meta["next_actionId"].to_numpy(dtype=int)
    mult = art["selected"]["action_multipliers"]
    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof_full = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    src = art["valid_meta"].copy().reset_index(drop=True)
    src["_row"] = np.arange(len(src))
    align = meta[["rally_uid", "prefix_len", "next_actionId"]].merge(
        src[["rally_uid", "prefix_len", "next_actionId", "_row"]],
        on=["rally_uid", "prefix_len", "next_actionId"],
        how="left",
        validate="one_to_one",
    )
    if align["_row"].isna().any():
        raise ValueError("Could not align artifact meta.")
    idx = align["_row"].to_numpy(dtype=int)
    r42_oof = r42_oof_full[idx]
    base_pred = apply_action(r42_oof, meta, mult)
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    current_sub = test_prefix_base[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current submission did not align.")

    all_rows = [{"candidate": "r42_base", "experiment": "base", "action_macro_f1": base_f1, "churn_vs_r42": 0.0}]
    prob_oof_by_name = {"r42_base": r42_oof}
    prob_test_by_name = {"r42_base": r42_test}

    # R58: trust/confidence gated row-wise blend.
    r58_rows = []
    for base_w in [0.10, 0.15, 0.20, 0.30]:
        for trust_min in [0.0, 0.10, 0.25, 0.40, 0.55]:
            for conf_min in [0.0, 0.18, 0.22, 0.26]:
                for ent_max in [10.0, 4.0, 3.2, 2.6]:
                    for min_prefix in [1, 2, 3]:
                        w_oof = trust_weights(style_oof[PRIMARY_STYLE], seg_oof, base_w, trust_min, conf_min, ent_max, min_prefix)
                        if w_oof.mean() < 0.005:
                            continue
                        prob = row_blend(r42_oof, r57_oof_probs[PRIMARY_STYLE], w_oof)
                        name = f"r58_trust_w{base_w}_t{trust_min}_c{conf_min}_e{ent_max}_p{min_prefix}"
                        row = describe_prob(
                            name,
                            prob,
                            meta,
                            y,
                            base_pred,
                            mult,
                            {
                                "experiment": "R58",
                                "mean_weight": float(w_oof.mean()),
                                "coverage": float(np.mean(w_oof > 0)),
                                "base_weight": base_w,
                                "trust_min": trust_min,
                                "conf_min": conf_min,
                                "ent_max": ent_max,
                                "min_prefix": min_prefix,
                            },
                        )
                        r58_rows.append(row)
                        prob_oof_by_name[name] = prob
                        w_test = trust_weights(test_styles[PRIMARY_STYLE], test_segments, base_w, trust_min, conf_min, ent_max, min_prefix)
                        prob_test_by_name[name] = row_blend(r42_test, test_probs[PRIMARY_STYLE], w_test)
    r58_search = pd.DataFrame(r58_rows).sort_values("action_macro_f1", ascending=False)
    r58_search.to_csv(OUTDIR / "r58_trust_gate_search.csv", index=False)
    all_rows.extend(r58_search.head(30).to_dict(orient="records"))

    # R59: class-aware style blend.
    r59_rows = []
    class_sets = {
        "style_core": [4, 8, 9, 11, 12],
        "style_low": [0, 3, 4, 7, 8, 9, 11, 12],
        "rare_control": [8, 9, 11, 12],
        "rare_only": [8, 9, 12, 14],
    }
    for set_name, classes in class_sets.items():
        for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]:
            prob = row_class_blend(r42_oof, r57_oof_probs[PRIMARY_STYLE], np.full(len(meta), w), classes)
            name = f"r59_{set_name}_w{w}"
            row = describe_prob(name, prob, meta, y, base_pred, mult, {"experiment": "R59", "classes": str(classes), "weight": w})
            r59_rows.append(row)
            prob_oof_by_name[name] = prob
            prob_test_by_name[name] = row_class_blend(r42_test, test_probs[PRIMARY_STYLE], np.full(len(test_prefix_base), w), classes)
    # Greedy per-class, style expert only.
    greedy_choices: dict[int, float] = {}
    current = r42_oof.copy()
    current_score = base_f1
    for _ in range(2):
        improved = False
        for cls in [0, 3, 4, 7, 8, 9, 11, 12, 14]:
            best = (current_score, None, current)
            for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
                choices = dict(greedy_choices)
                choices[cls] = w
                trial = r42_oof.copy()
                for c, cw in choices.items():
                    trial[:, c] = (1.0 - cw) * r42_oof[:, c] + cw * r57_oof_probs[PRIMARY_STYLE][:, c]
                trial = normalize_rows(trial)
                pred = apply_action(trial, meta, mult)
                score = float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0))
                if score > best[0] + 1e-8:
                    best = (score, w, trial)
            if best[1] is not None:
                greedy_choices[cls] = float(best[1])
                current_score = float(best[0])
                current = best[2]
                improved = True
        if not improved:
            break
    test_greedy = r42_test.copy()
    for c, cw in greedy_choices.items():
        test_greedy[:, c] = (1.0 - cw) * r42_test[:, c] + cw * test_probs[PRIMARY_STYLE][:, c]
    test_greedy = normalize_rows(test_greedy)
    name = "r59_greedy_class_style"
    r59_rows.append(describe_prob(name, current, meta, y, base_pred, mult, {"experiment": "R59", "choices": json.dumps(greedy_choices, sort_keys=True)}))
    prob_oof_by_name[name] = current
    prob_test_by_name[name] = test_greedy
    r59_search = pd.DataFrame(r59_rows).sort_values("action_macro_f1", ascending=False)
    r59_search.to_csv(OUTDIR / "r59_class_aware_search.csv", index=False)
    all_rows.extend(r59_search.head(30).to_dict(orient="records"))

    # R60: segmented seen/unseen/observed-count style blend.
    r60_rows = []
    for w_seen in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for w_many in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for w_few in [0.0, 0.03, 0.05, 0.10]:
                for thr in [3, 8, 15, 30]:
                    w_oof = segment_weights(seg_oof, w_seen, w_many, w_few, thr)
                    prob = row_blend(r42_oof, r57_oof_probs[PRIMARY_STYLE], w_oof)
                    name = f"r60_seen{w_seen}_many{w_many}_few{w_few}_thr{thr}"
                    row = describe_prob(
                        name,
                        prob,
                        meta,
                        y,
                        base_pred,
                        mult,
                        {
                            "experiment": "R60",
                            "w_seen": w_seen,
                            "w_unseen_many": w_many,
                            "w_unseen_few": w_few,
                            "many_thr": thr,
                            "mean_weight": float(w_oof.mean()),
                        },
                    )
                    r60_rows.append(row)
                    prob_oof_by_name[name] = prob
                    w_test = segment_weights(test_segments, w_seen, w_many, w_few, thr)
                    prob_test_by_name[name] = row_blend(r42_test, test_probs[PRIMARY_STYLE], w_test)
    r60_search = pd.DataFrame(r60_rows).sort_values("action_macro_f1", ascending=False)
    r60_search.to_csv(OUTDIR / "r60_seen_unseen_search.csv", index=False)
    all_rows.extend(r60_search.head(30).to_dict(orient="records"))

    # R61: combine R56 low-action experts with R57 style.
    r56_labels = ["zero_control_defense", "moderate_low", "rare_only", "conservative_low"]
    r56_oof, r56_test = make_r56_probs(art, r56_labels)
    r61_rows = []
    style_sources = [
        ("r57_row_w0p10", row_blend(r42_oof, r57_oof_probs[PRIMARY_STYLE], np.full(len(meta), 0.10)), row_blend(r42_test, test_probs[PRIMARY_STYLE], np.full(len(test_prefix_base), 0.10))),
        ("r57_row_w0p15", row_blend(r42_oof, r57_oof_probs[PRIMARY_STYLE], np.full(len(meta), 0.15)), row_blend(r42_test, test_probs[PRIMARY_STYLE], np.full(len(test_prefix_base), 0.15))),
        ("r57_class_greedy", current, test_greedy),
    ]
    for r56_label in r56_labels:
        for style_label, style_oof_prob, style_test_prob in style_sources:
            for w_style in [0.20, 0.35, 0.50, 0.65]:
                prob = normalize_rows((1.0 - w_style) * r56_oof[r56_label] + w_style * style_oof_prob)
                name = f"r61_{r56_label}_{style_label}_ws{w_style}"
                row = describe_prob(
                    name,
                    prob,
                    meta,
                    y,
                    base_pred,
                    mult,
                    {"experiment": "R61", "r56_label": r56_label, "style_label": style_label, "style_weight": w_style},
                )
                r61_rows.append(row)
                prob_oof_by_name[name] = prob
                prob_test_by_name[name] = normalize_rows((1.0 - w_style) * r56_test[r56_label] + w_style * style_test_prob)
    r61_search = pd.DataFrame(r61_rows).sort_values("action_macro_f1", ascending=False)
    r61_search.to_csv(OUTDIR / "r61_r56_r57_meta_search.csv", index=False)
    all_rows.extend(r61_search.head(40).to_dict(orient="records"))

    all_search = pd.DataFrame(all_rows).sort_values("action_macro_f1", ascending=False)
    all_search.to_csv(OUTDIR / "r58_r61_all_search.csv", index=False)

    generated = []
    used = set()
    for row in all_search.to_dict(orient="records"):
        label = row["candidate"]
        if label == "r42_base" or label in used:
            continue
        if label not in prob_test_by_name:
            continue
        churn = float(row["churn_vs_r42"])
        # Keep generated files in the plausible public-probe range.
        if churn > 0.10:
            continue
        pred = apply_action(prob_test_by_name[label], test_prefix_base, mult)
        safe_label = label.replace(".", "p").replace(" ", "_")
        name = f"submission_{safe_label}_current_point_server.csv"
        info = write_submission(
            test_prefix_base,
            pred,
            current_sub,
            name,
            {
                "source_oof_action_f1": row["action_macro_f1"],
                "source_oof_churn": row["churn_vs_r42"],
                "experiment": row.get("experiment", ""),
            },
        )
        generated.append(info)
        used.add(label)
        if len(generated) >= 10:
            break
    pd.DataFrame(generated).to_csv(OUTDIR / "r58_r61_generated_candidates.csv", index=False)

    report = {
        "base_action_f1": base_f1,
        "primary_style": PRIMARY_STYLE,
        "best_all": all_search.head(30).to_dict(orient="records"),
        "generated": generated,
        "note": "All generated candidates keep point/server fixed to current safe branch.",
    }
    (OUTDIR / "r58_r61_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(all_search.head(30).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
