"""R63/R64 conditional-style and momentum action experiments.

R63:
  Conditional player style clustering. Instead of only modeling a player's
  marginal stroke rates, estimate how the player responds to incoming action
  group / spin / landing depth. Add smoothed player conditional-style cluster
  features to an action model.

R64:
  Past-only momentum/fatigue features from scoreboard progression and game
  progress. No future scoreboard and no serverGetPoint labels are used.

Both experiments change action only. Point/server are fixed to current R34.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r57_player_style_clustering import (
    action_group,
    add_player_id_features,
    observed_rows_for_prefixes,
    point_depth,
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


OUTDIR = Path("r63_r64_conditional_momentum")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")

KEY_ACTIONS = [0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14]
ACTION_GROUPS = [0, 1, 2, 3, 4]
SPINS = list(range(6))
DEPTHS = [0, 1, 2, 3]


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


def make_action_model(seed: int, n_estimators: int = 180) -> lgb.LGBMClassifier:
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


def aligned_action_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        out[:, ACTION_CLASSES.index(cls)] = proba[:, i]
    return normalize_rows(out)


def encode_pair_score(row: pd.Series, pid: int) -> int:
    if int(row["gamePlayerId"]) == int(pid):
        return int(row["scoreSelf"])
    if int(row["gamePlayerOtherId"]) == int(pid):
        return int(row["scoreOther"])
    return -1


def add_past_momentum_features(prefix: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Add scoreboard/game-progress features using only previous observed rally starts."""
    first = raw.sort_values(["match", "numberGame", "rally_id", "rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).head(1).copy()
    first["pmin"] = first[["gamePlayerId", "gamePlayerOtherId"]].min(axis=1).astype(int)
    first["pmax"] = first[["gamePlayerId", "gamePlayerOtherId"]].max(axis=1).astype(int)
    feature_rows: list[dict[str, int | float]] = []
    group_cols = ["match", "numberGame", "pmin", "pmax"]
    for _, g in first.sort_values(group_cols + ["rally_id"]).groupby(group_cols, sort=False):
        g = g.reset_index(drop=True)
        for i, cur in g.iterrows():
            row: dict[str, int | float] = {
                "rally_uid": int(cur["rally_uid"]),
                "mg_observed_rally_idx": int(i),
                "mg_rally_id_rank_pct": float(i / max(len(g) - 1, 1)),
                "mg_has_prev_observed": int(i > 0),
                "mg_rally_id_gap_prev": 0,
                "mg_score_total_delta_prev": 0,
                "mg_current_server_score_delta_prev": 0,
                "mg_current_receiver_score_delta_prev": 0,
                "mg_prev_interval_current_server_rate": 0.5,
                "mg_prev_interval_valid": 0,
                "mg_prev_interval_current_server_won_more": 0,
                "mg_prev_lead_delta_current_server": 0,
            }
            if i > 0:
                prev = g.iloc[i - 1]
                cur_server = int(cur["gamePlayerId"])
                cur_receiver = int(cur["gamePlayerOtherId"])
                prev_server_score = encode_pair_score(prev, cur_server)
                prev_receiver_score = encode_pair_score(prev, cur_receiver)
                cur_server_score = int(cur["scoreSelf"])
                cur_receiver_score = int(cur["scoreOther"])
                ds = cur_server_score - prev_server_score if prev_server_score >= 0 else 0
                dr = cur_receiver_score - prev_receiver_score if prev_receiver_score >= 0 else 0
                total_delta = ds + dr
                valid = int(prev_server_score >= 0 and prev_receiver_score >= 0 and ds >= 0 and dr >= 0 and total_delta >= 0)
                row.update(
                    {
                        "mg_rally_id_gap_prev": int(cur["rally_id"] - prev["rally_id"]),
                        "mg_score_total_delta_prev": int(total_delta if valid else 0),
                        "mg_current_server_score_delta_prev": int(ds if valid else 0),
                        "mg_current_receiver_score_delta_prev": int(dr if valid else 0),
                        "mg_prev_interval_current_server_rate": float(ds / total_delta) if valid and total_delta > 0 else 0.5,
                        "mg_prev_interval_valid": valid,
                        "mg_prev_interval_current_server_won_more": int(valid and ds > dr),
                        "mg_prev_lead_delta_current_server": int((ds - dr) if valid else 0),
                    }
                )
            feature_rows.append(row)
    feats = pd.DataFrame(feature_rows)
    out = prefix.merge(feats, on="rally_uid", how="left")
    momentum_cols = [c for c in feats.columns if c != "rally_uid"]
    out[momentum_cols] = out[momentum_cols].fillna(0)
    out["mg_score_total_x_rank_pct"] = out["scoreTotal"].astype(float) * out["mg_rally_id_rank_pct"].astype(float)
    out["mg_late_rank_close_score"] = (
        (out["mg_rally_id_rank_pct"].astype(float) >= 0.70)
        & (np.abs(out["serverScoreDiff"].astype(int)) <= 2)
    ).astype(np.int8)
    return out


def response_events(raw: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in raw.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        cur = g.iloc[1:].copy().reset_index(drop=True)
        prev = g.iloc[:-1].copy().reset_index(drop=True)
        cur["incoming_action_group"] = action_group(prev["actionId"])
        cur["incoming_spin"] = prev["spinId"].to_numpy(dtype=int)
        cur["incoming_depth"] = point_depth(prev["pointId"])
        cur["response_group"] = action_group(cur["actionId"])
        parts.append(
            cur[
                [
                    "gamePlayerId",
                    "actionId",
                    "response_group",
                    "incoming_action_group",
                    "incoming_spin",
                    "incoming_depth",
                ]
            ].copy()
        )
    if not parts:
        return pd.DataFrame(columns=["gamePlayerId", "actionId", "response_group", "incoming_action_group", "incoming_spin", "incoming_depth"])
    return pd.concat(parts, ignore_index=True)


def conditional_matrix(events: pd.DataFrame, cond_col: str, cond_values: list[int], resp_values: list[int], resp_col: str, prior: np.ndarray, alpha: float) -> np.ndarray:
    mat = np.zeros((len(cond_values), len(resp_values)), dtype=float)
    if events.empty:
        return prior.copy()
    for i, cv in enumerate(cond_values):
        sub = events[events[cond_col].astype(int) == int(cv)]
        counts = sub[resp_col].value_counts().reindex(resp_values, fill_value=0).to_numpy(dtype=float)
        mat[i] = (counts + alpha * prior[i]) / (counts.sum() + alpha)
    return mat


class ConditionalStyleEncoder:
    def __init__(self, k: int = 8, alpha: float = 25.0, beta: float = 25.0, seed: int = 63) -> None:
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.kmeans: KMeans | None = None
        self.global_vec: np.ndarray | None = None
        self.global_parts: dict[str, np.ndarray] = {}
        self.player_vectors: dict[int, np.ndarray] = {}
        self.player_counts: dict[int, int] = {}
        self.cluster_prior: np.ndarray | None = None
        self.tau: float = 1.0

    def _build_global_parts(self, global_events: pd.DataFrame) -> None:
        def prior(cond_col: str, cond_values: list[int], resp_values: list[int], resp_col: str) -> np.ndarray:
            base = np.zeros((len(cond_values), len(resp_values)), dtype=float)
            for i, cv in enumerate(cond_values):
                sub = global_events[global_events[cond_col].astype(int) == int(cv)]
                counts = sub[resp_col].value_counts().reindex(resp_values, fill_value=0).to_numpy(dtype=float)
                base[i] = (counts + 1.0) / (counts.sum() + len(resp_values))
            return base

        self.global_parts = {
            "ag_group": prior("incoming_action_group", ACTION_GROUPS, ACTION_GROUPS, "response_group"),
            "spin_group": prior("incoming_spin", SPINS, ACTION_GROUPS, "response_group"),
            "depth_group": prior("incoming_depth", DEPTHS, ACTION_GROUPS, "response_group"),
            "ag_key": prior("incoming_action_group", ACTION_GROUPS, KEY_ACTIONS, "actionId"),
        }
        self.global_vec = np.concatenate([v.reshape(-1) for v in self.global_parts.values()])

    def _player_vector(self, events: pd.DataFrame) -> np.ndarray:
        if self.global_vec is None:
            raise RuntimeError("ConditionalStyleEncoder is not initialized.")
        return np.concatenate(
            [
                conditional_matrix(events, "incoming_action_group", ACTION_GROUPS, ACTION_GROUPS, "response_group", self.global_parts["ag_group"], self.alpha).reshape(-1),
                conditional_matrix(events, "incoming_spin", SPINS, ACTION_GROUPS, "response_group", self.global_parts["spin_group"], self.alpha).reshape(-1),
                conditional_matrix(events, "incoming_depth", DEPTHS, ACTION_GROUPS, "response_group", self.global_parts["depth_group"], self.alpha).reshape(-1),
                conditional_matrix(events, "incoming_action_group", ACTION_GROUPS, KEY_ACTIONS, "actionId", self.global_parts["ag_key"], self.alpha).reshape(-1),
            ]
        )

    def fit(self, observed_raw: pd.DataFrame, global_raw: pd.DataFrame) -> "ConditionalStyleEncoder":
        observed_events = response_events(observed_raw)
        global_events = response_events(global_raw)
        self._build_global_parts(global_events)
        if self.global_vec is None:
            raise RuntimeError("Global vector missing.")
        players, vectors, counts = [], [], []
        for pid, ev in observed_events.groupby("gamePlayerId", sort=False):
            players.append(int(pid))
            vectors.append(self._player_vector(ev))
            counts.append(len(ev))
        if not vectors:
            players, vectors, counts = [-1], [self.global_vec], [0]
        x = np.vstack(vectors)
        x_fit = x if len(x) >= self.k else np.vstack([x, np.repeat(self.global_vec[None, :], self.k - len(x), axis=0)])
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        self.kmeans.fit(x_fit)
        dist = self.kmeans.transform(x_fit)
        self.tau = float(np.median(np.min(dist, axis=1)))
        if not np.isfinite(self.tau) or self.tau <= 1e-6:
            self.tau = 0.10
        post = self._posterior(x)
        self.cluster_prior = post.mean(axis=0)
        self.cluster_prior = self.cluster_prior / self.cluster_prior.sum()
        self.player_vectors = {int(pid): vec for pid, vec in zip(players, vectors)}
        self.player_counts = {int(pid): int(n) for pid, n in zip(players, counts)}
        return self

    def _posterior(self, vecs: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("ConditionalStyleEncoder is not fit.")
        dist = self.kmeans.transform(vecs)
        logits = -dist / max(self.tau, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def transform_player(self, pid: int) -> tuple[np.ndarray, np.ndarray, int, float]:
        if self.global_vec is None:
            raise RuntimeError("ConditionalStyleEncoder is not fit.")
        prior = self.cluster_prior if self.cluster_prior is not None else np.ones(self.k) / self.k
        vec = self.player_vectors.get(int(pid), self.global_vec)
        n = int(self.player_counts.get(int(pid), 0))
        post = self._posterior(vec[None, :])[0]
        trust = n / (n + self.beta)
        post = (1.0 - trust) * prior + trust * post
        post = post / post.sum()
        return post, vec, n, float(trust)


def vec_offsets() -> dict[str, tuple[int, tuple[int, int]]]:
    off = 0
    out = {}
    out["ag_group"] = (off, (len(ACTION_GROUPS), len(ACTION_GROUPS)))
    off += len(ACTION_GROUPS) * len(ACTION_GROUPS)
    out["spin_group"] = (off, (len(SPINS), len(ACTION_GROUPS)))
    off += len(SPINS) * len(ACTION_GROUPS)
    out["depth_group"] = (off, (len(DEPTHS), len(ACTION_GROUPS)))
    off += len(DEPTHS) * len(ACTION_GROUPS)
    out["ag_key"] = (off, (len(ACTION_GROUPS), len(KEY_ACTIONS)))
    return out


def add_conditional_style_features(prefix: pd.DataFrame, encoder: ConditionalStyleEncoder) -> pd.DataFrame:
    offsets = vec_offsets()
    rows = []
    for row in prefix.itertuples(index=False):
        hp, hv, hn, ht = encoder.transform_player(int(row.next_hitter_id))
        rp, _, rn, rt = encoder.transform_player(int(row.next_receiver_id))
        feat: dict[str, int | float] = {}
        for i in range(encoder.k):
            feat[f"cond_hitter_cluster_p{i}"] = float(hp[i])
            feat[f"cond_receiver_cluster_p{i}"] = float(rp[i])
        feat["cond_hitter_cluster_top"] = int(np.argmax(hp))
        feat["cond_receiver_cluster_top"] = int(np.argmax(rp))
        feat["cond_pair_similarity"] = float(np.dot(hp, rp))
        feat["cond_pair_same_top"] = int(np.argmax(hp) == np.argmax(rp))
        feat["cond_hitter_trust"] = float(ht)
        feat["cond_receiver_trust"] = float(rt)
        feat["cond_hitter_n_events"] = int(hn)
        feat["cond_receiver_n_events"] = int(rn)
        feat["cond_hitter_entropy"] = float(-np.sum(np.clip(hp, 1e-12, 1) * np.log(np.clip(hp, 1e-12, 1))))
        feat["cond_receiver_entropy"] = float(-np.sum(np.clip(rp, 1e-12, 1) * np.log(np.clip(rp, 1e-12, 1))))

        incoming_ag = int(action_group(np.array([getattr(row, "lag0_actionId")]))[0])
        incoming_spin = int(getattr(row, "lag0_spinId"))
        incoming_depth = int(point_depth(np.array([getattr(row, "lag0_pointId")]))[0])
        incoming_ag = incoming_ag if incoming_ag in ACTION_GROUPS else 0
        incoming_spin = incoming_spin if incoming_spin in SPINS else 0
        incoming_depth = incoming_depth if incoming_depth in DEPTHS else 0

        off, shape = offsets["ag_group"]
        mat = hv[off : off + shape[0] * shape[1]].reshape(shape)
        for g in ACTION_GROUPS:
            feat[f"cond_resp_group_from_ag_{g}"] = float(mat[incoming_ag, g])
        off, shape = offsets["spin_group"]
        mat = hv[off : off + shape[0] * shape[1]].reshape(shape)
        for g in ACTION_GROUPS:
            feat[f"cond_resp_group_from_spin_{g}"] = float(mat[incoming_spin, g])
        off, shape = offsets["depth_group"]
        mat = hv[off : off + shape[0] * shape[1]].reshape(shape)
        for g in ACTION_GROUPS:
            feat[f"cond_resp_group_from_depth_{g}"] = float(mat[incoming_depth, g])
        off, shape = offsets["ag_key"]
        mat = hv[off : off + shape[0] * shape[1]].reshape(shape)
        for j, a in enumerate(KEY_ACTIONS):
            feat[f"cond_resp_key_action_from_ag_{a}"] = float(mat[incoming_ag, j])
        rows.append(feat)
    return pd.concat([prefix.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def class_blend(base: np.ndarray, expert: np.ndarray, weight: float, classes: list[int]) -> np.ndarray:
    out = base.copy()
    for cls in classes:
        out[:, cls] = (1.0 - weight) * base[:, cls] + weight * expert[:, cls]
    return normalize_rows(out)


def describe(name: str, prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, mult: dict, extra: dict) -> dict:
    pred = apply_action(prob, meta, mult)
    row = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "churn_vs_r42": float(np.mean(pred != base_pred)),
        "pred8_count": int((pred == 8).sum()),
        "pred9_count": int((pred == 9).sum()),
        "pred11_count": int((pred == 11).sum()),
        "pred12_count": int((pred == 12).sum()),
        "pred14_count": int((pred == 14).sum()),
    }
    row.update(extra)
    return row


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str, outdir: Path = OUTDIR) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = outdir / name
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
    art = load_artifact()
    train0 = pd.read_csv("train.csv")
    test0 = pd.read_csv("test_new.csv")
    validate_raw_data(train0, test0)
    train = add_role_and_score_features(train0)
    test = add_role_and_score_features(test0)
    prefix_base = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    prefix_base = add_phase_features(add_past_momentum_features(prefix_base, train), train)
    test_prefix = add_phase_features(add_past_momentum_features(test_prefix, test), test)
    prefix_base = add_player_id_features(prefix_base, train)
    test_prefix = add_player_id_features(test_prefix, test)
    player_cols = {"server_id", "receiver_id", "next_hitter_id", "next_receiver_id"}
    base_features = [c for c in feature_columns(prefix_base) if c != "remaining_len_bucket" and c not in player_cols]
    momentum_cols = [c for c in prefix_base.columns if c.startswith("mg_")]
    base_no_momentum = [c for c in base_features if not c.startswith("mg_")]

    rally_meta = prefix_base[["rally_uid", "match"]].drop_duplicates().reset_index(drop=True)
    test_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    meta_parts = []
    r63_oof_parts = {"r63_train_only_k8": [], "r63_transductive_k8": []}
    r64_parts = []

    for fold, (tr_rally_idx, va_rally_idx) in enumerate(GroupKFold(n_splits=5).split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[tr_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[va_rally_idx]["rally_uid"])
        tr = prefix_base[prefix_base["rally_uid"].isin(train_rallies)].copy().reset_index(drop=True)
        valid_pool = prefix_base[prefix_base["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_lengths, 42 + fold)
        va = valid_pool.loc[sampled_idx].copy().reset_index(drop=True)
        meta_parts.append(va[["rally_uid", "match", "prefix_len", "next_actionId"] + momentum_cols])

        # R64 momentum expert.
        m_features = base_features
        m_model = make_action_model(seed=6400 + fold)
        m_model.fit(tr[m_features], tr["next_actionId"], sample_weight=class_weight_sample(tr["next_actionId"]))
        r64_parts.append(aligned_action_proba(m_model, va[m_features]))

        train_obs = train[train["rally_uid"].isin(train_rallies)].copy()
        valid_obs = observed_rows_for_prefixes(train[train["rally_uid"].isin(valid_rallies)], va)
        for name, transductive in [("r63_train_only_k8", False), ("r63_transductive_k8", True)]:
            observed = pd.concat([train_obs, valid_obs], ignore_index=True) if transductive else train_obs
            encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=6300 + fold).fit(observed, train_obs)
            tr_cond = add_conditional_style_features(tr, encoder)
            va_cond = add_conditional_style_features(va, encoder)
            cond_cols = [c for c in tr_cond.columns if c.startswith("cond_")]
            features = base_no_momentum + cond_cols
            model = make_action_model(seed=6350 + fold + (100 if transductive else 0))
            model.fit(tr_cond[features], tr_cond["next_actionId"], sample_weight=class_weight_sample(tr_cond["next_actionId"]))
            r63_oof_parts[name].append(aligned_action_proba(model, va_cond[features]))
        print(f"fold {fold} done")

    meta = pd.concat(meta_parts, ignore_index=True).reset_index(drop=True)
    y = meta["next_actionId"].to_numpy(dtype=int)
    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_full = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    src = art["valid_meta"].copy().reset_index(drop=True)
    src["_row"] = np.arange(len(src))
    align = meta[["rally_uid", "prefix_len", "next_actionId"]].merge(
        src[["rally_uid", "prefix_len", "next_actionId", "_row"]],
        on=["rally_uid", "prefix_len", "next_actionId"],
        how="left",
        validate="one_to_one",
    )
    if align["_row"].isna().any():
        raise ValueError("Could not align R42 OOF to R63/R64 meta.")
    idx = align["_row"].to_numpy(dtype=int)
    r42_oof = r42_full[idx]
    mult = art["selected"]["action_multipliers"]
    base_pred = apply_action(r42_oof, meta, mult)
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    experts_oof = {k: np.vstack(v) for k, v in r63_oof_parts.items()}
    experts_oof["r64_momentum"] = np.vstack(r64_parts)

    rows = [{"candidate": "r42_base", "experiment": "base", "action_macro_f1": base_f1, "churn_vs_r42": 0.0}]
    oof_by_name = {"r42_base": r42_oof}
    class_sets = {
        "rare_control": [8, 9, 11, 12],
        "control_defense": [8, 9, 11, 12, 13, 14],
        "low_action": [0, 3, 4, 7, 8, 9, 11, 12, 14],
    }
    for expert_name, prob in experts_oof.items():
        pred = apply_action(prob, meta, mult)
        rows.append(describe(expert_name, prob, meta, y, base_pred, mult, {"experiment": expert_name, "blend_type": "single"}))
        for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            blend = normalize_rows((1 - w) * r42_oof + w * prob)
            name = f"{expert_name}_row_w{w}"
            rows.append(describe(name, blend, meta, y, base_pred, mult, {"experiment": expert_name, "blend_type": "row", "weight": w}))
            oof_by_name[name] = blend
            for set_name, classes in class_sets.items():
                cb = class_blend(r42_oof, prob, w, classes)
                cname = f"{expert_name}_cls_{set_name}_w{w}"
                rows.append(describe(cname, cb, meta, y, base_pred, mult, {"experiment": expert_name, "blend_type": "class", "class_set": set_name, "weight": w}))
                oof_by_name[cname] = cb

    # Combined R63 + R64 low-DoF class blend.
    for r63_name in ["r63_transductive_k8", "r63_train_only_k8"]:
        for w63 in [0.05, 0.10, 0.15, 0.20]:
            for w64 in [0.03, 0.05, 0.075, 0.10]:
                for set_name, classes in class_sets.items():
                    tmp = class_blend(r42_oof, experts_oof[r63_name], w63, classes)
                    comb = class_blend(tmp, experts_oof["r64_momentum"], w64, classes)
                    name = f"r63r64_{r63_name}_{set_name}_w63{w63}_w64{w64}"
                    rows.append(describe(name, comb, meta, y, base_pred, mult, {"experiment": "R63R64", "blend_type": "class2", "class_set": set_name, "w63": w63, "w64": w64}))
                    oof_by_name[name] = comb

    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r63_r64_search.csv", index=False)
    for name, prob in experts_oof.items():
        np.save(OUTDIR / f"{name}_oof_action.npy", prob)

    # Full train/test experts.
    experts_test: dict[str, np.ndarray] = {}
    m_model = make_action_model(seed=7400)
    m_model.fit(prefix_base[base_features], prefix_base["next_actionId"], sample_weight=class_weight_sample(prefix_base["next_actionId"]))
    experts_test["r64_momentum"] = aligned_action_proba(m_model, test_prefix[base_features])

    for name, transductive in [("r63_train_only_k8", False), ("r63_transductive_k8", True)]:
        observed = pd.concat([train, test], ignore_index=True) if transductive else train
        encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=7300).fit(observed, train)
        tr_cond = add_conditional_style_features(prefix_base, encoder)
        te_cond = add_conditional_style_features(test_prefix, encoder)
        cond_cols = [c for c in tr_cond.columns if c.startswith("cond_")]
        features = base_no_momentum + cond_cols
        model = make_action_model(seed=7350 + (100 if transductive else 0))
        model.fit(tr_cond[features], tr_cond["next_actionId"], sample_weight=class_weight_sample(tr_cond["next_actionId"]))
        experts_test[name] = aligned_action_proba(model, te_cond[features])

    r42_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    current_sub = test_prefix[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    generated = []
    for _, row in search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.10)].head(10).iterrows():
        label = str(row["candidate"])
        prob = None
        if label in experts_test:
            prob = experts_test[label]
        elif "_row_w" in label:
            expert_name = label.split("_row_w")[0]
            w = float(row["weight"])
            prob = normalize_rows((1 - w) * r42_test + w * experts_test[expert_name])
        elif "_cls_" in label and str(row["experiment"]) != "R63R64":
            expert_name = str(row["experiment"])
            w = float(row["weight"])
            classes = class_sets[str(row["class_set"])]
            prob = class_blend(r42_test, experts_test[expert_name], w, classes)
        elif str(row["experiment"]) == "R63R64":
            r63_name = "r63_transductive_k8" if "r63_transductive_k8" in label else "r63_train_only_k8"
            classes = class_sets[str(row["class_set"])]
            tmp = class_blend(r42_test, experts_test[r63_name], float(row["w63"]), classes)
            prob = class_blend(tmp, experts_test["r64_momentum"], float(row["w64"]), classes)
        if prob is None:
            continue
        pred = apply_action(prob, test_prefix, mult)
        safe = label.replace(".", "p").replace(" ", "_")
        name = f"submission_{safe}_current_point_server.csv"
        info = write_submission(test_prefix, pred, current_sub, name)
        info.update({"source_oof_action_f1": row["action_macro_f1"], "source_oof_churn": row["churn_vs_r42"], "experiment": row["experiment"]})
        generated.append(info)
        if len(generated) >= 8:
            break
    pd.DataFrame(generated).to_csv(OUTDIR / "r63_r64_generated_candidates.csv", index=False)

    report = {
        "base_action_f1": base_f1,
        "best": search.head(25).to_dict(orient="records"),
        "generated": generated,
        "note": "R63/R64 generated candidates change action only; point/server fixed to current R34.",
    }
    (OUTDIR / "r63_r64_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(30).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
