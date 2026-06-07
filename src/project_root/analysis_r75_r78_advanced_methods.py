"""R75-R78 advanced action experiments.

R75:
  Pairwise matchup target-encoding prior for action:
  P(action | next_hitter, next_receiver, phase) with backoff.

R76:
  Stroke2Vec-style unsupervised token embedding. Tokens are stroke tuples from
  train + public test prefixes. Prefix mean embeddings feed an action expert.

R77:
  Constrained Macro-F1 multiplier search over the R42 action probabilities.
  This is intentionally regularized and churn-aware.

R78:
  Mirror augmentation audit for action models. Three variants:
  position only, position+point-side, position+point-side+side-spin.

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

from analysis_r7_phase_features import add_phase_features
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r57_player_style_clustering import add_player_id_features, observed_rows_for_prefixes
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features
from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
)
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r75_r78_advanced_methods")
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


def apply_extra_multiplier(prob: np.ndarray, meta: pd.DataFrame, base_mult: dict, extra: np.ndarray) -> np.ndarray:
    return apply_action(normalize_rows(prob * extra[None, :]), meta, base_mult)


def blend_action_prob(base: np.ndarray, expert: np.ndarray, weight: float) -> np.ndarray:
    return normalize_rows((1.0 - weight) * base + weight * expert)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def make_lgbm(seed: int, n_estimators: int = 160) -> lgb.LGBMClassifier:
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


def fill_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        out[:, cls] = raw[:, i]
    return normalize_rows(out)


def prediction_phase_from_strike(strike_number: pd.Series | np.ndarray) -> np.ndarray:
    s = np.asarray(strike_number, dtype=int)
    return np.select([s == 2, s == 3, s == 4, s >= 5], [1, 2, 3, 4], default=0).astype(np.int8)


def observed_rows_for_prefix_subset(raw: pd.DataFrame, prefixes: pd.DataFrame) -> pd.DataFrame:
    return observed_rows_for_prefixes(raw, prefixes[["rally_uid", "prefix_len"]]).copy()


def make_action_row_table(observed: pd.DataFrame) -> pd.DataFrame:
    rows = observed[observed["strikeNumber"].astype(int).ge(2)].copy()
    rows["phase_id"] = prediction_phase_from_strike(rows["strikeNumber"])
    rows["hitter_id"] = rows["gamePlayerId"].astype(int)
    rows["receiver_id"] = rows["gamePlayerOtherId"].astype(int)
    rows["target_action"] = rows["actionId"].astype(int)
    return rows


def prior_from_key(rows: pd.DataFrame, source: pd.DataFrame, cols: list[str], alpha: float, global_prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup: dict[tuple, np.ndarray] = {}
    support: dict[tuple, int] = {}
    if len(source) > 0:
        for key, g in source.groupby(cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            counts = g["target_action"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
            lookup[key] = (counts + alpha * global_prior) / (counts.sum() + alpha)
            support[key] = int(len(g))
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    supp = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        key = tuple(int(getattr(row, c)) for c in cols)
        if key in lookup:
            out[i] = lookup[key]
            supp[i] = support[key]
        else:
            out[i] = global_prior
    return out, supp


def matchup_prior_for_rows(rows: pd.DataFrame, source_observed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source = make_action_row_table(source_observed)
    global_counts = source["target_action"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(ACTION_CLASSES))
    query = rows.rename(columns={"next_hitter_id": "hitter_id", "next_receiver_id": "receiver_id"}).copy()
    query["phase_id"] = query["phase_id"].astype(int)

    p_pair, s_pair = prior_from_key(query, source, ["hitter_id", "receiver_id", "phase_id"], 25.0, global_prior)
    p_hit, s_hit = prior_from_key(query, source, ["hitter_id", "phase_id"], 40.0, global_prior)
    p_recv, s_recv = prior_from_key(query, source, ["receiver_id", "phase_id"], 50.0, global_prior)
    p_phase, s_phase = prior_from_key(query, source, ["phase_id"], 80.0, global_prior)

    w_pair = (s_pair / (s_pair + 25.0)).reshape(-1, 1)
    w_hit = (s_hit / (s_hit + 40.0)).reshape(-1, 1)
    w_recv = (s_recv / (s_recv + 50.0)).reshape(-1, 1)
    blended = w_pair * p_pair + (1.0 - w_pair) * (
        0.45 * w_hit * p_hit + 0.30 * w_recv * p_recv + (1.0 - 0.45 * w_hit - 0.30 * w_recv) * p_phase
    )
    return normalize_rows(blended), s_pair


def matchup_prior_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    support = np.zeros(len(prefix_aligned), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        source = pd.concat(
            [
                raw[~raw["match"].isin(valid_matches)],
                observed_rows_for_prefix_subset(raw, prefix_aligned.loc[idx]),
            ],
            ignore_index=True,
        )
        p, s = matchup_prior_for_rows(prefix_aligned.loc[idx], source)
        out[idx] = p
        support[idx] = s
    return out, support


def matchup_prior_test(test_prefix: pd.DataFrame, train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source = pd.concat([train_raw, observed_rows_for_prefix_subset(test_raw, test_prefix)], ignore_index=True)
    return matchup_prior_for_rows(test_prefix, source)


def token_for_row(row: pd.Series) -> str:
    return (
        f"A{int(row.actionId)}_P{int(row.pointId)}_S{int(row.spinId)}_"
        f"H{int(row.handId)}_T{int(row.strengthId)}_X{int(row.positionId)}"
    )


def build_token_embeddings(raws: list[pd.DataFrame], dim: int = 16, max_vocab: int = 1600, window: int = 2) -> tuple[dict[str, int], np.ndarray]:
    token_counts: dict[str, int] = {}
    seqs: list[list[str]] = []
    for raw in raws:
        for _, g in raw.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
            seq = [token_for_row(row) for _, row in g.iterrows()]
            seqs.append(seq)
            for tok in seq:
                token_counts[tok] = token_counts.get(tok, 0) + 1
    vocab = ["<UNK>"] + [t for t, _ in sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max_vocab - 1]]
    tok2id = {t: i for i, t in enumerate(vocab)}
    mat = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    for seq in seqs:
        ids = [tok2id.get(t, 0) for t in seq]
        for i, a in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i == j:
                    continue
                mat[a, ids[j]] += 1.0
    mat = np.log1p(mat)
    mat -= mat.mean(axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    emb = u[:, :dim] * np.sqrt(s[:dim])[None, :]
    emb = emb.astype(np.float32)
    return tok2id, emb


def prefix_token_embeddings(prefix_df: pd.DataFrame, raw: pd.DataFrame, tok2id: dict[str, int], emb: np.ndarray) -> pd.DataFrame:
    seq_by_rally: dict[int, list[int]] = {}
    for rid, g in raw.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        seq_by_rally[int(rid)] = [tok2id.get(token_for_row(row), 0) for _, row in g.iterrows()]
    rows = []
    dim = emb.shape[1]
    for row in prefix_df.itertuples(index=False):
        seq = seq_by_rally.get(int(row.rally_uid), [])[: int(row.prefix_len)]
        if not seq:
            mean = np.zeros(dim, dtype=float)
            rec = np.zeros(dim, dtype=float)
        else:
            arr = emb[np.asarray(seq, dtype=int)]
            mean = arr.mean(axis=0)
            weights = np.linspace(0.5, 1.0, len(seq), dtype=np.float32)
            rec = (arr * weights[:, None]).sum(axis=0) / weights.sum()
        feat = {f"r76_mean_{i}": float(mean[i]) for i in range(dim)}
        feat.update({f"r76_rec_{i}": float(rec[i]) for i in range(dim)})
        rows.append(feat)
    return pd.concat([prefix_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def stroke2vec_expert_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, raw_train: pd.DataFrame, raw_test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    public_test = observed_rows_for_prefix_subset(raw_test, raw_test.groupby("rally_uid").size().reset_index(name="prefix_len"))
    tok2id, emb = build_token_embeddings([raw_train, public_test], dim=16)
    prefix_emb = prefix_token_embeddings(prefix, raw_train, tok2id, emb)
    aligned_emb = align_prefix_meta(prefix_aligned[["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint", "fold"]], prefix_emb)
    emb_cols = [c for c in prefix_emb.columns if c.startswith("r76_")]
    cols = [c for c in features if c in prefix_emb.columns] + emb_cols
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = prefix_emb[~prefix_emb["match"].isin(valid_matches)].copy()
        va = aligned_emb.loc[idx].copy()
        model = make_lgbm(7600 + int(fold))
        model.fit(tr[cols], tr["next_actionId"], sample_weight=class_weight_sample(tr["next_actionId"]))
        out[idx] = fill_proba(model, va[cols])
    return out, cols


def stroke2vec_expert_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, train_raw: pd.DataFrame, test_raw: pd.DataFrame, features: list[str], cols: list[str]) -> np.ndarray:
    public_test = observed_rows_for_prefix_subset(test_raw, test_prefix)
    tok2id, emb = build_token_embeddings([train_raw, public_test], dim=16)
    tr = prefix_token_embeddings(prefix, train_raw, tok2id, emb)
    te = prefix_token_embeddings(test_prefix, test_raw, tok2id, emb)
    model = make_lgbm(8600)
    model.fit(tr[cols], tr["next_actionId"], sample_weight=class_weight_sample(tr["next_actionId"]))
    return fill_proba(model, te[cols])


def constrained_multiplier_search(prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, base_mult: dict, seed: int = 77) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bounds = np.array([(0.5, 1.8)] * len(ACTION_CLASSES), dtype=float)
    for c in [0, 3, 4, 7, 8, 9, 11, 12, 14]:
        bounds[c] = (0.35, 4.0)
    for c in [15, 16, 17, 18]:
        bounds[c] = (0.05, 1.0)
    bounds[1] = (0.4, 1.5)
    bounds[2] = (0.4, 1.5)
    records = []
    best_score = -1e9
    best = np.ones(len(ACTION_CLASSES), dtype=float)
    # Start from neutral and random log-uniform candidates.
    candidates = [np.ones(len(ACTION_CLASSES), dtype=float)]
    for _ in range(2500):
        u = rng.uniform(np.log(bounds[:, 0]), np.log(bounds[:, 1]))
        # Keep most classes near 1; only a subset is strongly perturbed.
        if rng.random() < 0.7:
            mask = rng.random(len(ACTION_CLASSES)) < 0.45
            u[~mask] = 0.0
        candidates.append(np.exp(u))
    for i, mult in enumerate(candidates):
        pred = apply_extra_multiplier(prob, meta, base_mult, mult)
        f1 = float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0))
        churn = float(np.mean(pred != base_pred))
        reg = float(np.mean(np.abs(np.log(mult))))
        score = f1 - 0.035 * churn - 0.003 * reg
        if score > best_score:
            best_score = score
            best = mult.copy()
        if i % 50 == 0 or score >= best_score:
            records.append({"trial": i, "score": score, "action_macro_f1": f1, "churn_vs_r42": churn, "reg": reg})
    return best, pd.DataFrame(records).sort_values("score", ascending=False)


POINT_SIDE_SWAP = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}


def mirror_raw(raw: pd.DataFrame, variant: str, offset: int) -> pd.DataFrame:
    out = raw.copy()
    out["rally_uid"] = out["rally_uid"].astype(int) + offset
    out["match"] = out["match"].astype(int) + offset
    out["positionId"] = out["positionId"].replace({1: 3, 3: 1}).astype(int)
    if "point" in variant:
        out["pointId"] = out["pointId"].replace(POINT_SIDE_SWAP).astype(int)
    if "spin" in variant:
        out["spinId"] = out["spinId"].replace({4: 5, 5: 4}).astype(int)
    return out


def mirror_expert_oof(prefix_aligned: pd.DataFrame, raw_train: pd.DataFrame, features: list[str], variant: str) -> np.ndarray:
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr_raw = raw_train[~raw_train["match"].isin(valid_matches)].copy()
        aug_raw = pd.concat([tr_raw, mirror_raw(tr_raw, variant, 1_000_000_000 + int(fold) * 10_000_000)], ignore_index=True)
        aug_prefix = add_remaining_bucket(build_train_prefix_table(aug_raw, 6))
        aug_prefix = add_phase_features(aug_prefix, aug_raw)
        va = prefix_aligned.loc[idx].copy()
        cols = [c for c in features if c in aug_prefix.columns]
        model = make_lgbm(7800 + int(fold), n_estimators=140)
        model.fit(aug_prefix[cols], aug_prefix["next_actionId"], sample_weight=class_weight_sample(aug_prefix["next_actionId"]))
        out[idx] = fill_proba(model, va[cols])
    return out


def mirror_expert_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, raw_train: pd.DataFrame, features: list[str], variant: str) -> np.ndarray:
    aug_raw = pd.concat([raw_train, mirror_raw(raw_train, variant, 1_500_000_000)], ignore_index=True)
    aug_prefix = add_remaining_bucket(build_train_prefix_table(aug_raw, 6))
    aug_prefix = add_phase_features(aug_prefix, aug_raw)
    cols = [c for c in features if c in aug_prefix.columns]
    model = make_lgbm(8800, n_estimators=140)
    model.fit(aug_prefix[cols], aug_prefix["next_actionId"], sample_weight=class_weight_sample(aug_prefix["next_actionId"]))
    return fill_proba(model, test_prefix[cols])


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

    r75_oof, r75_support = matchup_prior_oof(prefix_aligned, prefix, train)
    r76_oof, r76_cols = stroke2vec_expert_oof(prefix_aligned, prefix, train, test, features)
    r78_oofs = {
        "r78_mirror_pos": mirror_expert_oof(prefix_aligned, train, features, "pos"),
        "r78_mirror_pos_point": mirror_expert_oof(prefix_aligned, train, features, "pos_point"),
        "r78_mirror_pos_point_spin": mirror_expert_oof(prefix_aligned, train, features, "pos_point_spin"),
    }

    experts = {"r75_matchup": r75_oof, "r76_stroke2vec": r76_oof, **r78_oofs}
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

    best_mult, mult_log = constrained_multiplier_search(r42_oof, meta, y, base_pred, mult)
    mult_log.to_csv(OUTDIR / "r77_multiplier_search_log.csv", index=False)
    prob_r77 = normalize_rows(r42_oof * best_mult[None, :])
    rows.append(metrics_row("r77_constrained_multiplier", prob_r77, meta, y, base_pred, mult, {"kind": "r77_multiplier", "weight": 1.0}))
    pd.DataFrame({"actionId": ACTION_CLASSES, "r77_multiplier": best_mult}).to_csv(OUTDIR / "r77_selected_multipliers.csv", index=False)
    pd.DataFrame({"r75_pair_support": r75_support}).to_csv(OUTDIR / "r75_matchup_support_oof.csv", index=False)

    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True])
    search.to_csv(OUTDIR / "r75_r78_oof_search.csv", index=False)

    # Full test.
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    r75_test, _ = matchup_prior_test(test_prefix, train, test)
    r76_test = stroke2vec_expert_test(prefix, test_prefix, train, test, features, r76_cols)
    r78_tests = {
        "r78_mirror_pos": mirror_expert_test(prefix, test_prefix, train, features, "pos"),
        "r78_mirror_pos_point": mirror_expert_test(prefix, test_prefix, train, features, "pos_point"),
        "r78_mirror_pos_point_spin": mirror_expert_test(prefix, test_prefix, train, features, "pos_point_spin"),
    }
    test_experts = {"r75_matchup": r75_test, "r76_stroke2vec": r76_test, **r78_tests}
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
            w = float(row.weight)
            pred = apply_action(blend_action_prob(r42_test, test_experts[kind], w), test_meta, mult)
            name = f"submission_{kind}_blend_w{clean_float(w)}_current_point_server.csv"
        elif kind == "r77_multiplier":
            pred = apply_extra_multiplier(r42_test, test_meta, mult, best_mult)
            name = "submission_r77_constrained_multiplier_current_point_server.csv"
        else:
            continue
        info = write_submission(test_meta, pred, current_sub, name)
        info["source_candidate"] = str(row.candidate)
        info["source_kind"] = kind
        info["source_oof_action_f1"] = float(row.action_macro_f1)
        info["source_oof_churn"] = float(row.churn_vs_r42)
        if hasattr(row, "weight") and not pd.isna(row.weight):
            info["weight"] = float(row.weight)
        generated.append(info)
    pd.DataFrame(generated).to_csv(OUTDIR / "r75_r78_generated_candidates.csv", index=False)
    report = {
        "base_action_macro_f1": base_f1,
        "top_oof": search.head(40).to_dict(orient="records"),
        "generated": generated,
        "r76_feature_count": len(r76_cols),
    }
    (OUTDIR / "r75_r78_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(40).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
