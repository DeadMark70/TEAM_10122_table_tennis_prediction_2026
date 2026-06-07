"""R57 player style clustering for action improvement.

Build smoothed player style vectors, convert them to soft cluster posteriors,
and use them as action-model features. This targets action Macro-F1, especially
style/equipment-sensitive classes 4/8/9/10/11/12/13/14.

Variants:
- train_only_k{K}: style clusters from train strokes only.
- transductive_k{K}: train + public observed test/validation prefixes for style
  vectors and clustering, without using hidden next-stroke labels.

Point/server are never changed in generated submissions.
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
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
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
from analysis_r48_action_meta_stacker import build_current_oof_action


OUTDIR = Path("r57_player_style_clustering")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
KEY_ACTIONS = [4, 8, 9, 10, 11, 12, 13, 14]
ACTION_GROUPS = [0, 1, 2, 3, 4]
SPINS = list(range(6))
POINTS = list(range(10))
HANDS = list(range(3))
STRENGTHS = list(range(4))
POSITIONS = list(range(4))


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


def action_group(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.full(len(arr), -1, dtype=np.int16)
    out[arr == 0] = 0
    out[(arr >= 1) & (arr <= 7)] = 1
    out[(arr >= 8) & (arr <= 11)] = 2
    out[(arr >= 12) & (arr <= 14)] = 3
    out[(arr >= 15) & (arr <= 18)] = 4
    return out


def point_depth(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) // 3 + 1).astype(np.int16)
    return out


def point_side(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    out = np.zeros(len(arr), dtype=np.int16)
    mask = arr > 0
    out[mask] = ((arr[mask] - 1) % 3 + 1).astype(np.int16)
    return out


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


def observed_rows_for_prefixes(raw: pd.DataFrame, prefixes: pd.DataFrame) -> pd.DataFrame:
    keep = prefixes[["rally_uid", "prefix_len"]].drop_duplicates()
    merged = raw.merge(keep, on="rally_uid", how="inner")
    return merged[merged["strikeNumber"].le(merged["prefix_len"])].drop(columns=["prefix_len"])


def global_rate(raw: pd.DataFrame, field: str, values: list[int]) -> np.ndarray:
    counts = raw[field].value_counts().reindex(values, fill_value=0).to_numpy(dtype=float)
    return (counts + 1.0) / (counts.sum() + len(values))


def style_feature_names(k: int) -> list[str]:
    names = []
    for role in ["hitter", "receiver"]:
        names += [f"style_{role}_cluster_p{i}" for i in range(k)]
        names += [
            f"style_{role}_cluster_top",
            f"style_{role}_confidence",
            f"style_{role}_entropy",
            f"style_{role}_trust",
            f"style_{role}_n_strokes",
        ]
        for a in KEY_ACTIONS:
            names.append(f"style_{role}_action_rate_{a}")
        for g in ACTION_GROUPS:
            names.append(f"style_{role}_group_rate_{g}")
        for s in SPINS:
            names.append(f"style_{role}_spin_rate_{s}")
        for p in [0, 1, 2, 3, 6, 7, 8, 9]:
            names.append(f"style_{role}_point_rate_{p}")
    names += [
        "style_pair_similarity",
        "style_pair_same_top",
        "style_hitter_receiver_conf_delta",
        "style_hitter_receiver_entropy_sum",
    ]
    return names


class StyleEncoder:
    def __init__(self, k: int, alpha: float = 25.0, beta: float = 25.0, seed: int = 57) -> None:
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.kmeans: KMeans | None = None
        self.global_parts: dict[str, np.ndarray] = {}
        self.player_vectors: dict[int, np.ndarray] = {}
        self.player_counts: dict[int, int] = {}
        self.cluster_prior: np.ndarray | None = None
        self.tau: float = 1.0

    def _global_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.global_parts["action_key"],
                self.global_parts["action_group"],
                self.global_parts["spin"],
                self.global_parts["point_key"],
                self.global_parts["hand"],
                self.global_parts["strength"],
                self.global_parts["position"],
                self.global_parts["depth"],
                self.global_parts["side"],
            ]
        )

    def _player_vector(self, part: pd.DataFrame) -> np.ndarray:
        n = len(part)
        ag = action_group(part["actionId"])
        depth = point_depth(part["pointId"])
        side = point_side(part["pointId"])

        def smooth_counts(values: np.ndarray, cats: list[int], prior: np.ndarray) -> np.ndarray:
            counts = pd.Series(values).value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
            return (counts + self.alpha * prior) / (n + self.alpha)

        return np.concatenate(
            [
                smooth_counts(part["actionId"].to_numpy(int), KEY_ACTIONS, self.global_parts["action_key"]),
                smooth_counts(ag, ACTION_GROUPS, self.global_parts["action_group"]),
                smooth_counts(part["spinId"].to_numpy(int), SPINS, self.global_parts["spin"]),
                smooth_counts(part["pointId"].to_numpy(int), [0, 1, 2, 3, 6, 7, 8, 9], self.global_parts["point_key"]),
                smooth_counts(part["handId"].to_numpy(int), HANDS, self.global_parts["hand"]),
                smooth_counts(part["strengthId"].to_numpy(int), STRENGTHS, self.global_parts["strength"]),
                smooth_counts(part["positionId"].to_numpy(int), POSITIONS, self.global_parts["position"]),
                smooth_counts(depth, [0, 1, 2, 3], self.global_parts["depth"]),
                smooth_counts(side, [0, 1, 2, 3], self.global_parts["side"]),
            ]
        )

    def fit(self, observed_raw: pd.DataFrame, global_raw: pd.DataFrame) -> "StyleEncoder":
        ag = action_group(global_raw["actionId"])
        depth = point_depth(global_raw["pointId"])
        side = point_side(global_raw["pointId"])
        self.global_parts = {
            "action_key": global_rate(global_raw.assign(_dummy=0), "actionId", KEY_ACTIONS),
            "action_group": (pd.Series(ag).value_counts().reindex(ACTION_GROUPS, fill_value=0).to_numpy(dtype=float) + 1)
            / (len(ag) + len(ACTION_GROUPS)),
            "spin": global_rate(global_raw, "spinId", SPINS),
            "point_key": global_rate(global_raw, "pointId", [0, 1, 2, 3, 6, 7, 8, 9]),
            "hand": global_rate(global_raw, "handId", HANDS),
            "strength": global_rate(global_raw, "strengthId", STRENGTHS),
            "position": global_rate(global_raw, "positionId", POSITIONS),
            "depth": (pd.Series(depth).value_counts().reindex([0, 1, 2, 3], fill_value=0).to_numpy(dtype=float) + 1) / (len(depth) + 4),
            "side": (pd.Series(side).value_counts().reindex([0, 1, 2, 3], fill_value=0).to_numpy(dtype=float) + 1) / (len(side) + 4),
        }
        global_vec = self._global_vector()
        players = []
        vectors = []
        counts = []
        for pid, part in observed_raw.groupby("gamePlayerId", sort=False):
            pid = int(pid)
            players.append(pid)
            vectors.append(self._player_vector(part))
            counts.append(len(part))
        if not vectors:
            vectors = [global_vec]
            players = [-1]
            counts = [0]
        x = np.vstack(vectors)
        n_clusters = min(self.k, len(x))
        if n_clusters < self.k:
            pad = np.repeat(global_vec[None, :], self.k - n_clusters, axis=0)
            x_fit = np.vstack([x, pad])
        else:
            x_fit = x
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        self.kmeans.fit(x_fit)
        dist = self.kmeans.transform(x_fit)
        self.tau = float(np.median(np.min(dist, axis=1)))
        if not np.isfinite(self.tau) or self.tau <= 1e-6:
            self.tau = 0.15
        post = self._posterior_from_vecs(x)
        self.cluster_prior = post.mean(axis=0)
        self.cluster_prior = self.cluster_prior / self.cluster_prior.sum()
        self.player_vectors = {int(pid): vec for pid, vec in zip(players, vectors)}
        self.player_counts = {int(pid): int(n) for pid, n in zip(players, counts)}
        return self

    def _posterior_from_vecs(self, vecs: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("StyleEncoder is not fit.")
        dist = self.kmeans.transform(vecs)
        logits = -dist / max(self.tau, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def transform_player(self, pid: int) -> tuple[np.ndarray, np.ndarray, int, float]:
        prior = self.cluster_prior if self.cluster_prior is not None else np.ones(self.k) / self.k
        vec = self.player_vectors.get(int(pid), self._global_vector())
        n = int(self.player_counts.get(int(pid), 0))
        posterior = self._posterior_from_vecs(vec[None, :])[0]
        trust = n / (n + self.beta)
        posterior = (1.0 - trust) * prior + trust * posterior
        posterior = posterior / posterior.sum()
        return posterior, vec, n, float(trust)


def add_style_features(prefix: pd.DataFrame, encoder: StyleEncoder) -> pd.DataFrame:
    rows = []
    feature_names = style_feature_names(encoder.k)
    vec_len = len(encoder._global_vector())
    key_action_len = len(KEY_ACTIONS)
    group_len = len(ACTION_GROUPS)
    spin_len = len(SPINS)
    point_len = len([0, 1, 2, 3, 6, 7, 8, 9])
    for row in prefix.itertuples(index=False):
        hp, hv, hn, ht = encoder.transform_player(int(row.next_hitter_id))
        rp, rv, rn, rt = encoder.transform_player(int(row.next_receiver_id))

        def role_feats(role: str, p: np.ndarray, vec: np.ndarray, n: int, trust: float) -> dict[str, float]:
            out = {f"style_{role}_cluster_p{i}": float(p[i]) for i in range(encoder.k)}
            out[f"style_{role}_cluster_top"] = int(np.argmax(p))
            out[f"style_{role}_confidence"] = float(np.max(p))
            out[f"style_{role}_entropy"] = float(-np.sum(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))))
            out[f"style_{role}_trust"] = float(trust)
            out[f"style_{role}_n_strokes"] = int(n)
            off = 0
            for j, a in enumerate(KEY_ACTIONS):
                out[f"style_{role}_action_rate_{a}"] = float(vec[off + j])
            off += key_action_len
            for j, g in enumerate(ACTION_GROUPS):
                out[f"style_{role}_group_rate_{g}"] = float(vec[off + j])
            off += group_len
            for j, s in enumerate(SPINS):
                out[f"style_{role}_spin_rate_{s}"] = float(vec[off + j])
            off += spin_len
            for j, pp in enumerate([0, 1, 2, 3, 6, 7, 8, 9]):
                out[f"style_{role}_point_rate_{pp}"] = float(vec[off + j])
            return out

        feat = {}
        feat.update(role_feats("hitter", hp, hv, hn, ht))
        feat.update(role_feats("receiver", rp, rv, rn, rt))
        feat["style_pair_similarity"] = float(np.dot(hp, rp))
        feat["style_pair_same_top"] = int(np.argmax(hp) == np.argmax(rp))
        feat["style_hitter_receiver_conf_delta"] = float(np.max(hp) - np.max(rp))
        feat["style_hitter_receiver_entropy_sum"] = float(feat["style_hitter_entropy"] + feat["style_receiver_entropy"])
        rows.append(feat)
    style_df = pd.DataFrame(rows, columns=feature_names)
    return pd.concat([prefix.reset_index(drop=True), style_df], axis=1)


def make_action_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=180,
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


def train_predict_action(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    model = make_action_model(seed)
    model.fit(train_df[features], train_df["next_actionId"], sample_weight=class_weight_sample(train_df["next_actionId"]))
    return aligned_action_proba(model, valid_df[features])


def class_report(y: np.ndarray, pred: np.ndarray, name: str) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=ACTION_CLASSES, zero_division=0)
    return pd.DataFrame(
        {
            "model": name,
            "actionId": ACTION_CLASSES,
            "support": s,
            "pred_count": [(pred == c).sum() for c in ACTION_CLASSES],
            "precision": p,
            "recall": r,
            "f1": f,
        }
    )


def load_artifact() -> dict:
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


def align_r42(meta: pd.DataFrame, art: dict) -> tuple[np.ndarray, dict]:
    current = build_current_oof_action()
    golden = art["experts_oof"]["v47_v64_oof_soft"]
    r42 = normalize_rows(0.80 * current + 0.20 * golden)
    src = art["valid_meta"].copy().reset_index(drop=True)
    src["_row"] = np.arange(len(src))
    merged = meta[["rally_uid", "prefix_len", "next_actionId"]].merge(
        src[["rally_uid", "prefix_len", "next_actionId", "_row"]],
        on=["rally_uid", "prefix_len", "next_actionId"],
        how="left",
        validate="one_to_one",
    )
    if merged["_row"].isna().any():
        raise ValueError("Could not align R42 action OOF to R57 meta.")
    return r42[merged["_row"].to_numpy(dtype=int)], art["selected"]["action_multipliers"]


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


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
    art = load_artifact()
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

    rally_meta = prefix_base[["rally_uid", "match"]].drop_duplicates().reset_index(drop=True)
    test_lengths = test_prefix_base["prefix_len"].to_numpy(dtype=int)
    variants = [("train_only_k8", 8, False), ("train_only_k16", 16, False), ("transductive_k8", 8, True), ("transductive_k16", 16, True)]
    oof_parts = {name: [] for name, _, _ in variants}
    meta_parts = []

    for fold, (tr_rally_idx, va_rally_idx) in enumerate(GroupKFold(n_splits=5).split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[tr_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[va_rally_idx]["rally_uid"])
        tr = prefix_base[prefix_base["rally_uid"].isin(train_rallies)].copy().reset_index(drop=True)
        valid_pool = prefix_base[prefix_base["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_lengths, 42 + fold)
        va = valid_pool.loc[sampled_idx].copy().reset_index(drop=True)
        meta_parts.append(va[["rally_uid", "match", "prefix_len", "next_actionId"]])
        train_obs_raw = train_raw[train_raw["rally_uid"].isin(train_rallies)].copy()
        valid_obs_raw = observed_rows_for_prefixes(train_raw[train_raw["rally_uid"].isin(valid_rallies)], va)

        for variant_name, k, transductive in variants:
            observed_for_style = pd.concat([train_obs_raw, valid_obs_raw], ignore_index=True) if transductive else train_obs_raw
            encoder = StyleEncoder(k=k, alpha=25.0, beta=25.0, seed=5700 + fold + k).fit(observed_for_style, train_obs_raw)
            tr_style = add_style_features(tr, encoder)
            va_style = add_style_features(va, encoder)
            style_cols = [c for c in tr_style.columns if c.startswith("style_")]
            features = base_features + style_cols
            prob = train_predict_action(tr_style, va_style, features, seed=5800 + fold + k)
            oof_parts[variant_name].append(prob)
        print(f"fold {fold} done")

    meta = pd.concat(meta_parts, ignore_index=True)
    y = meta["next_actionId"].to_numpy(dtype=int)
    r42_oof, action_mult = align_r42(meta, art)
    r42_pred = apply_action(r42_oof, meta, action_mult)
    r42_f1 = float(f1_score(y, r42_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))
    variant_prob = {name: np.vstack(parts) for name, parts in oof_parts.items()}

    rows = [{"variant": "r42_base", "weight": 0.0, "action_macro_f1": r42_f1, "churn_vs_r42": 0.0}]
    blend_grid = [0.0, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
    for name, prob in variant_prob.items():
        pred_single = apply_action(prob, meta, action_mult)
        rows.append(
            {
                "variant": name,
                "weight": 1.0,
                "action_macro_f1": float(f1_score(y, pred_single, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "churn_vs_r42": float(np.mean(pred_single != r42_pred)),
            }
        )
        for w in blend_grid:
            blend = normalize_rows((1.0 - w) * r42_oof + w * prob)
            pred = apply_action(blend, meta, action_mult)
            rows.append(
                {
                    "variant": f"blend_r42_{name}",
                    "weight": w,
                    "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                    "churn_vs_r42": float(np.mean(pred != r42_pred)),
                }
            )

    search = pd.DataFrame(rows).sort_values("action_macro_f1", ascending=False)
    search.to_csv(OUTDIR / "r57_oof_blend_search.csv", index=False)
    report_frames = []
    report_frames.append(class_report(y, r42_pred, "r42_base"))
    for name, prob in variant_prob.items():
        report_frames.append(class_report(y, apply_action(prob, meta, action_mult), name))
    pd.concat(report_frames, ignore_index=True).to_csv(OUTDIR / "r57_action_class_report.csv", index=False)
    for name, prob in variant_prob.items():
        np.save(OUTDIR / f"{name}_oof_action.npy", prob)

    # Full train/test predictions.
    full_train_obs = train_raw.copy()
    test_obs = test_raw.copy()
    test_probs = {}
    for variant_name, k, transductive in variants:
        observed = pd.concat([full_train_obs, test_obs], ignore_index=True) if transductive else full_train_obs
        encoder = StyleEncoder(k=k, alpha=25.0, beta=25.0, seed=6700 + k).fit(observed, full_train_obs)
        train_style = add_style_features(prefix_base, encoder)
        test_style = add_style_features(test_prefix_base, encoder)
        style_cols = [c for c in train_style.columns if c.startswith("style_")]
        features = base_features + style_cols
        model = make_action_model(seed=6800 + k + (100 if transductive else 0))
        model.fit(train_style[features], train_style["next_actionId"], sample_weight=class_weight_sample(train_style["next_actionId"]))
        test_probs[variant_name] = aligned_action_proba(model, test_style[features])

    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    current_sub = test_prefix_base[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    generated = []
    for row in search.to_dict(orient="records"):
        if not str(row["variant"]).startswith("blend_r42_") or float(row["weight"]) <= 0:
            continue
        if float(row["churn_vs_r42"]) > 0.08:
            continue
        source = str(row["variant"]).replace("blend_r42_", "")
        w = float(row["weight"])
        prob = normalize_rows((1.0 - w) * r42_test + w * test_probs[source])
        pred = apply_action(prob, test_prefix_base, action_mult)
        name = f"submission_r57_{source}_w{str(w).replace('.', 'p')}_current_point_server.csv"
        info = write_submission(test_prefix_base, pred, current_sub, name)
        info.update({"source_oof_action_f1": row["action_macro_f1"], "source_oof_churn": row["churn_vs_r42"], "weight": w, "source": source})
        generated.append(info)
        if len(generated) >= 6:
            break
    pd.DataFrame(generated).to_csv(OUTDIR / "r57_generated_candidates.csv", index=False)

    report = {
        "r42_base_action_f1": r42_f1,
        "best_oof": search.head(20).to_dict(orient="records"),
        "generated": generated,
        "variants": [v[0] for v in variants],
    }
    (OUTDIR / "r57_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(20).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
