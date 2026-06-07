"""R71-R74 special action signals.

R71:
  Infer player handedness from observed position/hand patterns and train an
  action expert with hitter/receiver handedness features.

R72:
  Past-only intra-match tactical shift features. For each row, summarize the
  next hitter's observed earlier strokes in the same match before the current
  game/rally.

R73:
  Rally survival expectation gate. Predict terminal / short remaining length
  and use it to softly rescale action groups.

R74:
  Pure spin-strength-depth physical prior. Fold-safe lookup
  P(action | incoming spin, strength, landing depth, phase) with backoff.

All submissions change action only. Point/server stay fixed to current R34.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r57_player_style_clustering import add_player_id_features, observed_rows_for_prefixes, point_depth
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r71_r74_special_signals")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


def blend_action_prob(base: np.ndarray, expert: np.ndarray, weight: float) -> np.ndarray:
    return normalize_rows((1.0 - weight) * base + weight * expert)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def make_lgbm_multiclass(seed: int, n_estimators: int = 180) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_lgbm_binary(seed: int, n_estimators: int = 220) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fill_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        out[:, cls] = raw[:, i]
    return normalize_rows(out)


def observed_rows_for_prefix_subset(raw: pd.DataFrame, prefixes: pd.DataFrame) -> pd.DataFrame:
    obs = observed_rows_for_prefixes(raw, prefixes[["rally_uid", "prefix_len"]])
    return obs.copy()


def handedness_table(observed: pd.DataFrame, alpha: float = 3.0) -> dict[int, tuple[float, float, int]]:
    table: dict[int, tuple[float, float, int]] = {}
    for pid, g in observed.groupby("gamePlayerId"):
        pos = g["positionId"].astype(int)
        hand = g["handId"].astype(int)
        left_ev = int(((pos == 3) & (hand == 1)).sum() + ((pos == 1) & (hand == 2)).sum())
        right_ev = int(((pos == 1) & (hand == 1)).sum() + ((pos == 3) & (hand == 2)).sum())
        support = left_ev + right_ev
        p_left = (left_ev + alpha) / (support + 2.0 * alpha)
        trust = support / (support + 20.0)
        table[int(pid)] = (float(p_left), float(trust), int(support))
    return table


def add_handedness_features(prefix_df: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    table = handedness_table(observed)
    out = prefix_df.copy()
    hp, ht, hn, rp, rt, rn = [], [], [], [], [], []
    for row in out.itertuples(index=False):
        h = table.get(int(row.next_hitter_id), (0.5, 0.0, 0))
        r = table.get(int(row.next_receiver_id), (0.5, 0.0, 0))
        hp.append(h[0])
        ht.append(h[1])
        hn.append(h[2])
        rp.append(r[0])
        rt.append(r[1])
        rn.append(r[2])
    out["r71_hitter_lefty_prob"] = hp
    out["r71_hitter_lefty_trust"] = ht
    out["r71_hitter_hand_support"] = hn
    out["r71_receiver_lefty_prob"] = rp
    out["r71_receiver_lefty_trust"] = rt
    out["r71_receiver_hand_support"] = rn
    out["r71_same_handedness_prob"] = (
        out["r71_hitter_lefty_prob"] * out["r71_receiver_lefty_prob"]
        + (1.0 - out["r71_hitter_lefty_prob"]) * (1.0 - out["r71_receiver_lefty_prob"])
    )
    out["r71_handedness_delta"] = out["r71_hitter_lefty_prob"] - out["r71_receiver_lefty_prob"]
    out["r71_hitter_lefty_conf"] = np.abs(out["r71_hitter_lefty_prob"] - 0.5) * out["r71_hitter_lefty_trust"]
    out["r71_receiver_lefty_conf"] = np.abs(out["r71_receiver_lefty_prob"] - 0.5) * out["r71_receiver_lefty_trust"]
    return out


def action_group_id(a: int) -> int:
    if a == 0:
        return 0
    if 1 <= a <= 7:
        return 1
    if 8 <= a <= 11:
        return 2
    if 12 <= a <= 14:
        return 3
    if 15 <= a <= 18:
        return 4
    return 0


def build_past_tables(observed: pd.DataFrame) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    obs = observed.copy()
    obs["order_key"] = obs["numberGame"].astype(int) * 100000 + obs["rally_id"].astype(int)
    obs["ag"] = obs["actionId"].astype(int).map(action_group_id).astype(int)
    tables: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for key, g in obs.sort_values("order_key").groupby(["match", "gamePlayerId"], sort=False):
        keys = g["order_key"].to_numpy(dtype=np.int64)
        act_counts = np.zeros((len(g), len(ACTION_CLASSES)), dtype=np.int16)
        grp_counts = np.zeros((len(g), 5), dtype=np.int16)
        rare_counts = np.zeros((len(g), 4), dtype=np.int16)
        running_a = np.zeros(len(ACTION_CLASSES), dtype=np.int16)
        running_g = np.zeros(5, dtype=np.int16)
        running_r = np.zeros(4, dtype=np.int16)
        for i, row in enumerate(g.itertuples(index=False)):
            act_counts[i] = running_a
            grp_counts[i] = running_g
            rare_counts[i] = running_r
            a = int(row.actionId)
            running_a[a] += 1
            running_g[int(row.ag)] += 1
            for ri, ra in enumerate([8, 9, 12, 14]):
                if a == ra:
                    running_r[ri] += 1
        tables[(int(key[0]), int(key[1]))] = {
            "keys": keys,
            "act": act_counts,
            "grp": grp_counts,
            "rare": rare_counts,
        }
    return tables


def add_intra_match_shift_features(prefix_df: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    out = prefix_df.copy()
    tables = build_past_tables(observed)
    global_grp = np.ones(5, dtype=float)
    global_rare = np.ones(4, dtype=float)
    observed_groups = observed["actionId"].astype(int).map(action_group_id).astype(int)
    for g in range(5):
        global_grp[g] += int((observed_groups == g).sum())
    for i, a in enumerate([8, 9, 12, 14]):
        global_rare[i] += int((observed["actionId"].astype(int) == a).sum())
    global_grp = global_grp / global_grp.sum()
    global_rare = global_rare / global_rare.sum()

    rows = []
    for row in out.itertuples(index=False):
        key = (int(row.match), int(row.next_hitter_id))
        order_key = int(row.numberGame) * 100000 + int(row.rally_id)
        table = tables.get(key)
        if table is None:
            grp = np.zeros(5, dtype=float)
            rare = np.zeros(4, dtype=float)
            n = 0
        else:
            pos = int(np.searchsorted(table["keys"], order_key, side="left"))
            if pos <= 0:
                grp = np.zeros(5, dtype=float)
                rare = np.zeros(4, dtype=float)
                n = 0
            else:
                idx = pos - 1
                grp_counts = table["grp"][idx].astype(float)
                rare_counts = table["rare"][idx].astype(float)
                n = int(grp_counts.sum())
                grp = (grp_counts + 2.0 * global_grp) / (grp_counts.sum() + 2.0)
                rare = (rare_counts + 2.0 * global_rare) / (rare_counts.sum() + 2.0)
        feat = {
            "r72_past_n": n,
            "r72_log_past_n": float(np.log1p(n)),
        }
        for gi in range(5):
            feat[f"r72_past_group_rate_{gi}"] = float(grp[gi])
            feat[f"r72_past_group_delta_{gi}"] = float(grp[gi] - global_grp[gi])
        for ri, a in enumerate([8, 9, 12, 14]):
            feat[f"r72_past_rare_rate_{a}"] = float(rare[ri])
            feat[f"r72_past_rare_delta_{a}"] = float(rare[ri] - global_rare[ri])
        rows.append(feat)
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def train_action_expert_oof(
    prefix_aligned: pd.DataFrame,
    prefix: pd.DataFrame,
    features: list[str],
    raw: pd.DataFrame,
    mode: str,
) -> tuple[np.ndarray, list[str]]:
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    feature_cols: list[str] | None = None
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = prefix[~prefix["match"].isin(valid_matches)].copy()
        va = prefix_aligned.loc[idx].copy()
        if mode == "handed":
            obs_source = pd.concat(
                [
                    raw[~raw["match"].isin(valid_matches)],
                    observed_rows_for_prefix_subset(raw, va),
                ],
                ignore_index=True,
            )
            tr_aug = add_handedness_features(tr, obs_source)
            va_aug = add_handedness_features(va, obs_source)
            add_cols = [c for c in tr_aug.columns if c.startswith("r71_")]
        elif mode == "shift":
            tr_obs = observed_rows_for_prefix_subset(raw, tr)
            va_obs = observed_rows_for_prefix_subset(raw, va)
            tr_aug = add_intra_match_shift_features(tr, tr_obs)
            va_aug = add_intra_match_shift_features(va, va_obs)
            add_cols = [c for c in tr_aug.columns if c.startswith("r72_")]
        else:
            raise ValueError(mode)
        cols = [c for c in features if c in tr_aug.columns] + add_cols
        feature_cols = cols
        model = make_lgbm_multiclass(7100 + int(fold) + (100 if mode == "shift" else 0))
        model.fit(tr_aug[cols], tr_aug["next_actionId"], sample_weight=class_weight_sample(tr_aug["next_actionId"]))
        out[idx] = fill_proba(model, va_aug[cols])
    assert feature_cols is not None
    return out, feature_cols


def train_action_expert_test(
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    features: list[str],
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    mode: str,
    feature_cols: list[str],
) -> np.ndarray:
    if mode == "handed":
        source = pd.concat([train_raw, observed_rows_for_prefix_subset(test_raw, test_prefix)], ignore_index=True)
        tr_aug = add_handedness_features(prefix, source)
        te_aug = add_handedness_features(test_prefix, source)
        seed = 7900
    elif mode == "shift":
        tr_aug = add_intra_match_shift_features(prefix, observed_rows_for_prefix_subset(train_raw, prefix))
        te_aug = add_intra_match_shift_features(test_prefix, observed_rows_for_prefix_subset(test_raw, test_prefix))
        seed = 8000
    else:
        raise ValueError(mode)
    model = make_lgbm_multiclass(seed)
    model.fit(tr_aug[feature_cols], tr_aug["next_actionId"], sample_weight=class_weight_sample(tr_aug["next_actionId"]))
    return fill_proba(model, te_aug[feature_cols])


def survival_scores_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    terminal = np.zeros(len(prefix_aligned), dtype=float)
    le2 = np.zeros(len(prefix_aligned), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = prefix[~prefix["match"].isin(valid_matches)].copy()
        va = prefix_aligned.loc[idx].copy()
        y1 = tr["next_is_terminal"].astype(int)
        y2 = tr["remaining_len"].le(2).astype(int)
        m1 = make_lgbm_binary(7300 + int(fold))
        m2 = make_lgbm_binary(7350 + int(fold))
        m1.fit(tr[features], y1)
        m2.fit(tr[features], y2)
        terminal[idx] = m1.predict_proba(va[features])[:, 1]
        le2[idx] = m2.predict_proba(va[features])[:, 1]
    return terminal, le2


def survival_scores_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    m1 = make_lgbm_binary(8300)
    m2 = make_lgbm_binary(8350)
    m1.fit(prefix[features], prefix["next_is_terminal"].astype(int))
    m2.fit(prefix[features], prefix["remaining_len"].le(2).astype(int))
    return m1.predict_proba(test_prefix[features])[:, 1], m2.predict_proba(test_prefix[features])[:, 1]


def survival_gate_prob(base: np.ndarray, p_terminal: np.ndarray, p_le2: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    out = base.copy()
    terminal_classes = [0, 3, 12, 13, 14]
    transition_classes = [10, 11, 13]
    out[:, terminal_classes] *= np.exp(beta * (p_terminal.reshape(-1, 1) - 0.5))
    out[:, transition_classes] *= np.exp(gamma * ((1.0 - p_le2).reshape(-1, 1) - 0.5))
    return normalize_rows(out)


def physical_keys(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "k4": ["phase_id", "lag0_spinId", "lag0_strengthId", "lag0_depth"],
        "k3": ["lag0_spinId", "lag0_strengthId", "lag0_depth"],
        "k2": ["lag0_spinId", "lag0_strengthId"],
        "k1": ["phase_id", "lag0_spinId"],
    }


def add_lag0_depth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag0_depth"] = point_depth(out["lag0_pointId"].to_numpy(dtype=int)).astype(int)
    return out


def build_lookup(df: pd.DataFrame, cols: list[str], alpha: float = 15.0) -> tuple[dict[tuple, np.ndarray], dict[tuple, int], np.ndarray]:
    global_counts = df["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(ACTION_CLASSES))
    lookup: dict[tuple, np.ndarray] = {}
    support: dict[tuple, int] = {}
    for key, g in df.groupby(cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        counts = g["next_actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
        lookup[key] = (counts + alpha * global_prior) / (counts.sum() + alpha)
        support[key] = int(len(g))
    return lookup, support, global_prior


def physical_prior_for_rows(rows: pd.DataFrame, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = add_lag0_depth(rows)
    train_df = add_lag0_depth(train_df)
    defs = physical_keys(rows)
    lookups = {name: build_lookup(train_df, cols) for name, cols in defs.items()}
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    level = np.zeros(len(rows), dtype=int)
    for i, row in enumerate(rows.itertuples(index=False)):
        chosen = None
        for li, name in enumerate(["k4", "k3", "k2", "k1"], start=4):
            cols = defs[name]
            key = tuple(int(getattr(row, c)) for c in cols)
            lookup, supp, global_prior = lookups[name]
            min_supp = 20 if name == "k4" else 35 if name == "k3" else 50 if name == "k2" else 80
            if key in lookup and supp[key] >= min_supp:
                chosen = (lookup[key], supp[key], li)
                break
        if chosen is None:
            _, _, global_prior = lookups["k1"]
            chosen = (global_prior, 0, 0)
        out[i] = chosen[0]
        support[i] = chosen[1]
        level[i] = chosen[2]
    return normalize_rows(out), support, level


def physical_prior_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(prefix_aligned), dtype=float)
    level = np.zeros(len(prefix_aligned), dtype=int)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = prefix[~prefix["match"].isin(valid_matches)].copy()
        p, s, l = physical_prior_for_rows(prefix_aligned.loc[idx], tr)
        out[idx] = p
        support[idx] = s
        level[idx] = l
    return out, support, level


def physical_prior_test(test_prefix: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return physical_prior_for_rows(test_prefix, prefix)


def metrics_row(name: str, prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, mult: dict, extra: dict | None = None) -> dict:
    pred = apply_action(prob, meta, mult)
    row = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "churn_vs_r42": float(np.mean(pred != base_pred)),
        "pred0_count": int((pred == 0).sum()),
        "pred8_count": int((pred == 8).sum()),
        "pred9_count": int((pred == 9).sum()),
        "pred12_count": int((pred == 12).sum()),
        "pred14_count": int((pred == 14).sum()),
    }
    if extra:
        row.update(extra)
    return row


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
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
    return {
        "candidate": name,
        "path": str(path),
        "upload_path": str(UPLOAD_DIR / name),
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train, test, prefix, test_prefix, features = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    prefix_aligned = align_prefix_meta(meta, prefix)

    y = meta["next_actionId"].to_numpy(dtype=int)
    mult = art["selected"]["action_multipliers"]
    current_oof = build_current_oof_action()
    v64_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * v64_oof)
    base_pred = apply_action(r42_oof, meta, mult)
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    rows = [metrics_row("r42_base", r42_oof, meta, y, base_pred, mult, {"kind": "base", "weight": 0.0})]

    # R71/R72 action experts.
    r71_oof, r71_cols = train_action_expert_oof(prefix_aligned, prefix, features, train, "handed")
    r72_oof, r72_cols = train_action_expert_oof(prefix_aligned, prefix, features, train, "shift")

    # R73 survival gate scores.
    p_term_oof, p_le2_oof = survival_scores_oof(prefix_aligned, prefix, features)

    # R74 physical prior.
    r74_oof, r74_support, r74_level = physical_prior_oof(prefix_aligned, prefix)

    experts = {
        "r71_handed": r71_oof,
        "r72_shift": r72_oof,
        "r74_physical": r74_oof,
    }
    for name, prob in experts.items():
        rows.append(metrics_row(f"{name}_direct", prob, meta, y, base_pred, mult, {"kind": name, "weight": 1.0}))
        for w in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            rows.append(
                metrics_row(
                    f"{name}_blend_w{w}",
                    blend_action_prob(r42_oof, prob, w),
                    meta,
                    y,
                    base_pred,
                    mult,
                    {"kind": name, "weight": float(w)},
                )
            )

    for beta in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]:
        for gamma in [-0.50, 0.00, 0.25, 0.50, 1.00]:
            prob = survival_gate_prob(r42_oof, p_term_oof, p_le2_oof, beta=beta, gamma=gamma)
            rows.append(
                metrics_row(
                    f"r73_survival_b{beta}_g{gamma}",
                    prob,
                    meta,
                    y,
                    base_pred,
                    mult,
                    {"kind": "r73_survival", "beta": float(beta), "gamma": float(gamma), "weight": np.nan},
                )
            )

    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True])
    search.to_csv(OUTDIR / "r71_r74_oof_search.csv", index=False)
    pd.DataFrame({"r74_support": r74_support, "r74_level": r74_level}).to_csv(OUTDIR / "r74_physical_support_oof.csv", index=False)

    # Full test probabilities.
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    r71_test = train_action_expert_test(prefix, test_prefix, features, train, test, "handed", r71_cols)
    r72_test = train_action_expert_test(prefix, test_prefix, features, train, test, "shift", r72_cols)
    p_term_test, p_le2_test = survival_scores_test(prefix, test_prefix, features)
    r74_test, r74_support_test, r74_level_test = physical_prior_test(test_prefix, prefix)
    test_experts = {"r71_handed": r71_test, "r72_shift": r72_test, "r74_physical": r74_test}

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    selected = search[
        (search["action_macro_f1"].gt(base_f1))
        & (search["churn_vs_r42"].le(0.12))
        & (~search["candidate"].eq("r42_base"))
    ].head(12)

    generated = []
    for row in selected.itertuples(index=False):
        kind = str(row.kind)
        if kind in test_experts:
            weight = float(row.weight)
            pred = apply_action(blend_action_prob(r42_test, test_experts[kind], weight), test_meta, mult)
            name = f"submission_{kind}_blend_w{clean_float(weight)}_current_point_server.csv"
        elif kind == "r73_survival":
            beta = float(row.beta)
            gamma = float(row.gamma)
            pred = apply_action(survival_gate_prob(r42_test, p_term_test, p_le2_test, beta, gamma), test_meta, mult)
            name = f"submission_r73_survival_b{clean_float(beta)}_g{clean_float(gamma)}_current_point_server.csv"
        else:
            continue
        info = write_submission(test_meta, pred, current_sub, name)
        info["source_candidate"] = str(row.candidate)
        info["source_kind"] = kind
        info["source_oof_action_f1"] = float(row.action_macro_f1)
        info["source_oof_churn"] = float(row.churn_vs_r42)
        if hasattr(row, "weight") and not pd.isna(row.weight):
            info["weight"] = float(row.weight)
        if hasattr(row, "beta") and not pd.isna(row.beta):
            info["beta"] = float(row.beta)
            info["gamma"] = float(row.gamma)
        generated.append(info)
    pd.DataFrame(generated).to_csv(OUTDIR / "r71_r74_generated_candidates.csv", index=False)

    report = {
        "base_action_macro_f1": base_f1,
        "top_oof": search.head(30).to_dict(orient="records"),
        "generated": generated,
        "r71_feature_count": len(r71_cols),
        "r72_feature_count": len(r72_cols),
        "r74_test_support_mean": float(np.mean(r74_support_test)),
        "r74_test_level_counts": pd.Series(r74_level_test).value_counts().sort_index().to_dict(),
    }
    (OUTDIR / "r71_r74_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(30).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
