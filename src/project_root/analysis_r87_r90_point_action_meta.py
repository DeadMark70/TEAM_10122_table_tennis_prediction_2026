"""R87-R90 point/action meta experiments.

R87:
  Non-linear action-guided point meta model. Train a point LGBM using V3 point
  probabilities, R83 point-style probabilities, R67 action probabilities, and
  fold-safe point motif priors.

R88:
  Style-gated dynamic action multipliers. Use R63 style/action probabilities to
  boost only player/style-supported low-F1 action classes.

R89:
  Joint candidate sweep combining R67/R88 action branches with R87/R83 point
  branches.

R90:
  Scoreboard pseudo-label diagnostic only. It writes reports but does not emit
  upload submissions because this branch is compliance-sensitive.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r82_r86_point_style import (
    R63_OOF_PATH,
    action_conditioned_point_oof,
    action_conditioned_point_for_rows,
    align_prefix_meta,
    blend_action_classwise,
    blend_point,
    clean_float,
    compose_v3_full_point,
    point_depth_arr,
    point_side_arr,
    prepare_prefix_features,
    train_ordinal_oof,
    train_ordinal_test,
    train_point_style_oof,
    train_point_style_test,
)
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r87_r90_point_action_meta")
SELECTED_DIR = Path("submissions/selected")
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


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(prob: np.ndarray) -> np.ndarray:
    order = np.sort(prob, axis=1)
    return order[:, -1] - order[:, -2]


def top_class(prob: np.ndarray) -> np.ndarray:
    return prob.argmax(axis=1)


def build_point_motif_lookup(train_df: pd.DataFrame, alpha: float = 30.0) -> dict:
    global_counts = train_df["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(POINT_CLASSES))

    def make(cols: list[str]) -> dict[tuple[int, ...], tuple[np.ndarray, int]]:
        out = {}
        for key, sub in train_df.groupby(cols, sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            counts = sub["next_pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
            prior = (counts + alpha * global_prior) / (counts.sum() + alpha)
            out[tuple(int(x) for x in key_tuple)] = (prior, int(len(sub)))
        return out

    return {
        "global": global_prior,
        "k3": make(["phase_id", "lag1_pointId", "lag0_pointId", "lag0_actionId"]),
        "k2": make(["phase_id", "lag1_pointId", "lag0_pointId"]),
        "k1": make(["lag0_pointId"]),
    }


def point_motif_for_rows(rows: pd.DataFrame, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = build_point_motif_lookup(train_df)
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    level = np.zeros(len(rows), dtype=int)
    for i, (_, row) in enumerate(rows.iterrows()):
        candidates = [
            ("k3", (int(row["phase_id"]), int(row["lag1_pointId"]), int(row["lag0_pointId"]), int(row["lag0_actionId"])), 20, 3),
            ("k2", (int(row["phase_id"]), int(row["lag1_pointId"]), int(row["lag0_pointId"])), 35, 2),
            ("k1", (int(row["lag0_pointId"]),), 60, 1),
        ]
        chosen = (lookup["global"], 0, 0)
        for name, key, min_support, lev in candidates:
            item = lookup[name].get(key)
            if item is not None and item[1] >= min_support:
                chosen = (item[0], item[1], lev)
                break
        out[i] = chosen[0]
        support[i] = float(chosen[1])
        level[i] = int(chosen[2])
    return normalize_rows(out), support, level


def point_motif_oof(rows: pd.DataFrame, prefix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    support = np.zeros(len(rows), dtype=float)
    level = np.zeros(len(rows), dtype=int)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        train_df = prefix[~prefix["match"].isin(valid_matches)].copy()
        p, s, l = point_motif_for_rows(rows.loc[idx], train_df)
        out[idx] = p
        support[idx] = s
        level[idx] = l
    return out, support, level


def make_point_meta_frame(
    rows: pd.DataFrame,
    v3_point: np.ndarray,
    r83_point: np.ndarray,
    r85_point: np.ndarray,
    r82_point: np.ndarray,
    r67_action: np.ndarray,
    motif: np.ndarray,
    motif_support: np.ndarray,
    motif_level: np.ndarray,
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for i in POINT_CLASSES:
        data[f"v3_p{i}"] = v3_point[:, i]
        data[f"r83_p{i}"] = r83_point[:, i]
        data[f"r85_p{i}"] = r85_point[:, i]
        data[f"r82_p{i}"] = r82_point[:, i]
        data[f"motif_p{i}"] = motif[:, i]
    for i in ACTION_CLASSES:
        data[f"r67_a{i}"] = r67_action[:, i]
    data["v3_entropy"] = entropy(v3_point)
    data["r83_entropy"] = entropy(r83_point)
    data["r67_action_entropy"] = entropy(r67_action)
    data["v3_margin"] = margin(v3_point)
    data["r83_margin"] = margin(r83_point)
    data["r67_action_margin"] = margin(r67_action)
    data["v3_top"] = top_class(v3_point)
    data["r83_top"] = top_class(r83_point)
    data["r67_action_top"] = top_class(r67_action)
    data["motif_support_log"] = np.log1p(motif_support)
    data["motif_level"] = motif_level.astype(float)
    for c in [
        "prefix_len",
        "phase_id",
        "sex",
        "scoreTotal",
        "serverScoreDiff",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_handId",
        "lag0_positionId",
        "lag1_actionId",
        "lag1_pointId",
        "lag1_spinId",
        "serve_action",
        "serve_spin",
        "serve_point",
        "receive_action",
        "receive_point",
    ]:
        if c in rows.columns:
            data[c] = rows[c].to_numpy()
    data["lag0_depth"] = point_depth_arr(rows["lag0_pointId"].to_numpy(dtype=int))
    data["lag0_side"] = point_side_arr(rows["lag0_pointId"].to_numpy(dtype=int))
    data["lag1_depth"] = point_depth_arr(rows["lag1_pointId"].to_numpy(dtype=int))
    data["lag1_side"] = point_side_arr(rows["lag1_pointId"].to_numpy(dtype=int))
    return pd.DataFrame(data)


def train_point_meta_oof(x: pd.DataFrame, y: pd.Series, rows: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    out = np.zeros((len(x), len(POINT_CLASSES)), dtype=float)
    fold_rows = []
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        train_idx = rows.index[~rows["fold"].eq(fold)].to_numpy()
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(POINT_CLASSES),
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=39,
            min_child_samples=28,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.25,
            reg_lambda=3.0,
            random_state=8700 + int(fold),
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x.iloc[train_idx], y.iloc[train_idx], sample_weight=class_weight_sample(y.iloc[train_idx]))
        pred = model.predict_proba(x.iloc[idx])
        aligned = np.zeros((len(idx), len(POINT_CLASSES)), dtype=float)
        for j, cls in enumerate([int(c) for c in model.classes_]):
            aligned[:, POINT_CLASSES.index(cls)] = pred[:, j]
        out[idx] = normalize_rows(aligned)
        fold_pred = out[idx].argmax(axis=1)
        fold_rows.append(
            {
                "fold": int(fold),
                "point_macro_f1": float(f1_score(y.iloc[idx], fold_pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
                "n_valid": int(len(idx)),
            }
        )
    return normalize_rows(out), pd.DataFrame(fold_rows)


def train_point_meta_full(x: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(POINT_CLASSES),
        n_estimators=280,
        learning_rate=0.035,
        num_leaves=39,
        min_child_samples=28,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.25,
        reg_lambda=3.0,
        random_state=8799,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=class_weight_sample(y))
    return model


def align_point_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    pred = model.predict_proba(x)
    out = np.zeros((len(x), len(POINT_CLASSES)), dtype=float)
    for j, cls in enumerate([int(c) for c in model.classes_]):
        out[:, POINT_CLASSES.index(cls)] = pred[:, j]
    return normalize_rows(out)


def style_gated_multiplier(base: np.ndarray, style: np.ndarray, alpha: float, beta: float, cap: float, target_classes: list[int]) -> np.ndarray:
    prior = base.mean(axis=0)
    out = base.copy()
    for c in target_classes:
        shift = (style[:, c] - prior[c]) / max(prior[c], 0.02)
        factor = 1.0 + alpha * np.maximum(shift, 0.0) - beta * np.maximum(-shift, 0.0)
        factor = np.clip(factor, 1.0 / cap, cap)
        out[:, c] *= factor
    return normalize_rows(out)


def write_submission(
    test_meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
    name: str,
    extra: dict | None = None,
    selected: bool = True,
) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path = None
    if selected:
        SELECTED_DIR.mkdir(parents=True, exist_ok=True)
        selected_path = SELECTED_DIR / name
        selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path) if selected_path else ""}
    if extra:
        info.update(extra)
    return info


def future_score_diagnostic(test_raw: pd.DataFrame) -> pd.DataFrame:
    first = test_raw.sort_values(["match", "numberGame", "rally_id", "rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).head(1).copy()
    first["pmin"] = first[["gamePlayerId", "gamePlayerOtherId"]].min(axis=1).astype(int)
    first["pmax"] = first[["gamePlayerId", "gamePlayerOtherId"]].max(axis=1).astype(int)
    rows = []
    for _, g in first.sort_values(["match", "numberGame", "pmin", "pmax", "rally_id"]).groupby(["match", "numberGame", "pmin", "pmax"], sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            next_score = {int(nxt["gamePlayerId"]): int(nxt["scoreSelf"]), int(nxt["gamePlayerOtherId"]): int(nxt["scoreOther"])}
            server_id = int(cur["gamePlayerId"])
            receiver_id = int(cur["gamePlayerOtherId"])
            ds = next_score.get(server_id, -999) - int(cur["scoreSelf"])
            dr = next_score.get(receiver_id, -999) - int(cur["scoreOther"])
            gap = int(nxt["rally_id"] - cur["rally_id"])
            valid = gap > 0 and ds >= 0 and dr >= 0 and ds + dr == gap
            if valid:
                rows.append(
                    {
                        "rally_uid": int(cur["rally_uid"]),
                        "future_gap": gap,
                        "future_server_score_rate": float(ds / gap),
                        "future_score_valid": 1,
                        "future_server_points": int(ds),
                        "future_receiver_points": int(dr),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, features = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)

    v3_oof = load_pickle("oof_proba_v3.pkl")
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]):
        raise ValueError("V3 OOF does not align.")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    test_prefix_v3, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    if not test_prefix_v3["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 test point rows do not align.")

    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    r63_oof = np.load(R63_OOF_PATH)
    r67_oof = normalize_rows(0.80 * r42_oof + 0.20 * r63_oof)

    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    # Recompute R63 test action using the same transductive encoder as R82/R86.
    from analysis_r82_r86_point_style import ConditionalStyleEncoder, add_conditional_style_features

    encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=7350).fit(pd.concat([train_raw, test_raw], ignore_index=True), train_raw)
    train_cond = add_conditional_style_features(prefix, encoder)
    test_cond = add_conditional_style_features(test_prefix, encoder)
    cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
    cond_features = [c for c in features if c in train_cond.columns] + cond_cols
    r63_model = lgb.LGBMClassifier(objective="multiclass", num_class=len(ACTION_CLASSES), n_estimators=180, learning_rate=0.04, num_leaves=39, min_child_samples=24, subsample=0.88, subsample_freq=1, colsample_bytree=0.88, reg_alpha=0.15, reg_lambda=2.0, random_state=7350, n_jobs=-1, verbosity=-1)
    r63_model.fit(train_cond[cond_features], train_cond["next_actionId"], sample_weight=class_weight_sample(train_cond["next_actionId"]))
    r63_test_raw = r63_model.predict_proba(test_cond[cond_features])
    r63_test = np.zeros((len(test_prefix), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in r63_model.classes_]):
        r63_test[:, ACTION_CLASSES.index(cls)] = r63_test_raw[:, i]
    r63_test = normalize_rows(r63_test)
    r67_test = normalize_rows(0.80 * r42_test + 0.20 * r63_test)

    # Recompute point experts; save arrays for later reuse.
    r82_oof, _ = action_conditioned_point_oof(r67_oof, rows, prefix)
    r82_test, _ = action_conditioned_point_for_rows(r67_test, test_prefix, prefix)
    r83_oof = train_point_style_oof(train_raw, prefix, rows, features)
    r83_test = train_point_style_test(train_raw, test_raw, prefix, test_prefix, features)
    r85_oof = train_ordinal_oof(prefix, rows, features)
    r85_test = train_ordinal_test(prefix, test_prefix, features)
    motif_oof, motif_support, motif_level = point_motif_oof(rows, prefix)
    motif_test, motif_support_test, motif_level_test = point_motif_for_rows(test_prefix, prefix)
    np.save(OUTDIR / "r87_r82_point_oof.npy", r82_oof)
    np.save(OUTDIR / "r87_r83_point_oof.npy", r83_oof)
    np.save(OUTDIR / "r87_r85_point_oof.npy", r85_oof)
    np.save(OUTDIR / "r87_point_motif_oof.npy", motif_oof)

    x_oof = make_point_meta_frame(rows, v3_point_oof, r83_oof, r85_oof, r82_oof, r67_oof, motif_oof, motif_support, motif_level)
    x_test = make_point_meta_frame(test_prefix, v3_point_test, r83_test, r85_test, r82_test, r67_test, motif_test, motif_support_test, motif_level_test)
    y_point = meta["next_pointId"].astype(int)
    r87_oof, fold_report = train_point_meta_oof(x_oof, y_point, rows)
    fold_report.to_csv(OUTDIR / "r87_point_meta_fold_report.csv", index=False)
    r87_model = train_point_meta_full(x_oof, y_point)
    r87_test = align_point_proba(r87_model, x_test)
    np.save(OUTDIR / "r87_point_meta_oof.npy", r87_oof)
    np.save(OUTDIR / "r87_point_meta_test.npy", r87_test)

    y_action = meta["next_actionId"].to_numpy(dtype=int)
    base_action_pred = apply_segmented_multipliers(meta, r67_oof, art["selected"]["action_multipliers"], ACTION_CLASSES, "two")
    base_point_pred = apply_segmented_multipliers(meta, v3_point_oof, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
    base_action_f1 = f1_score(y_action, base_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    base_point_f1 = f1_score(y_point, base_point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)

    search_rows = [
        {
            "variant": "r67_w0p2_v3point",
            "kind": "base",
            "action_macro_f1": float(base_action_f1),
            "point_macro_f1": float(base_point_f1),
            "action_churn": 0.0,
            "point_churn": 0.0,
        }
    ]

    point_sources = {"r87_meta": r87_oof, "r83_style": r83_oof, "r85_ordinal": r85_oof, "motif": motif_oof}
    point_test_sources = {"r87_meta": r87_test, "r83_style": r83_test, "r85_ordinal": r85_test, "motif": motif_test}
    best_point = []
    for source, prob in point_sources.items():
        for w in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
            p = normalize_rows((1.0 - w) * v3_point_oof + w * prob)
            pred = apply_segmented_multipliers(meta, p, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
            f1 = f1_score(y_point, pred, average="macro", labels=POINT_CLASSES, zero_division=0)
            churn = float(np.mean(pred != base_point_pred))
            row = {
                "variant": f"{source}_w{clean_float(w)}",
                "kind": "point_blend",
                "point_source": source,
                "point_w": float(w),
                "action_macro_f1": float(base_action_f1),
                "point_macro_f1": float(f1),
                "action_churn": 0.0,
                "point_churn": churn,
            }
            search_rows.append(row)
            if churn <= 0.10 and f1 > base_point_f1:
                best_point.append(row)

    # R88 style-gated action multipliers.
    target_classes = [0, 3, 7, 8, 9, 11, 12, 14]
    best_action = []
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for beta in [0.0, 0.05, 0.10]:
            for cap in [1.25, 1.5, 2.0, 3.0]:
                prob = style_gated_multiplier(r67_oof, r63_oof, alpha, beta, cap, target_classes)
                pred = apply_segmented_multipliers(meta, prob, art["selected"]["action_multipliers"], ACTION_CLASSES, "two")
                f1 = f1_score(y_action, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
                churn = float(np.mean(pred != base_action_pred))
                row = {
                    "variant": f"r88_stylemult_a{clean_float(alpha)}_b{clean_float(beta)}_cap{clean_float(cap)}",
                    "kind": "action_style_multiplier",
                    "alpha": alpha,
                    "beta": beta,
                    "cap": cap,
                    "action_macro_f1": float(f1),
                    "point_macro_f1": float(base_point_f1),
                    "action_churn": churn,
                    "point_churn": 0.0,
                }
                search_rows.append(row)
                if f1 > base_action_f1 and churn <= 0.10:
                    best_action.append(row)

    search = pd.DataFrame(search_rows).sort_values(["point_macro_f1", "action_macro_f1"], ascending=[False, False])
    search.to_csv(OUTDIR / "r87_r90_oof_search.csv", index=False)

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    server_test = current_sub["serverGetPoint"].to_numpy(dtype=float)
    generated = []

    action_candidates: dict[str, tuple[np.ndarray, dict]] = {
        "r67_w0p2": (
            apply_segmented_multipliers(test_meta, r67_test, art["selected"]["action_multipliers"], ACTION_CLASSES, "two"),
            {"action_variant": "r67_w0p2", "oof_action_f1": float(base_action_f1), "oof_action_churn": 0.0},
        )
    }
    for row in sorted(best_action, key=lambda r: (r["action_macro_f1"], -r["action_churn"]), reverse=True)[:5]:
        prob = style_gated_multiplier(r67_test, r63_test, row["alpha"], row["beta"], row["cap"], target_classes)
        pred = apply_segmented_multipliers(test_meta, prob, art["selected"]["action_multipliers"], ACTION_CLASSES, "two")
        action_candidates[row["variant"]] = (
            pred,
            {"action_variant": row["variant"], "oof_action_f1": row["action_macro_f1"], "oof_action_churn": row["action_churn"]},
        )

    point_candidates: dict[str, tuple[np.ndarray, dict]] = {
        "v3": (
            apply_segmented_multipliers(test_meta, v3_point_test, art["selected"]["point_multipliers"], POINT_CLASSES, "two"),
            {"point_variant": "v3", "oof_point_f1": float(base_point_f1), "oof_point_churn": 0.0},
        )
    }
    for row in sorted(best_point, key=lambda r: (r["point_macro_f1"], -r["point_churn"]), reverse=True)[:8]:
        source = row["point_source"]
        w = float(row["point_w"])
        test_prob = normalize_rows((1.0 - w) * v3_point_test + w * point_test_sources[source])
        pred = apply_segmented_multipliers(test_meta, test_prob, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
        point_candidates[row["variant"]] = (
            pred,
            {"point_variant": row["variant"], "oof_point_f1": row["point_macro_f1"], "oof_point_churn": row["point_churn"]},
        )

    for action_key, (action_pred, action_info) in action_candidates.items():
        for point_key, (point_pred, point_info) in point_candidates.items():
            if action_key != "r67_w0p2" and point_key != "v3":
                # Keep joint grid compact: only combine best action variants with
                # the top two point variants.
                if point_key not in list(point_candidates.keys())[1:3]:
                    continue
            name = f"submission_r89_{action_key}_{point_key}_current_server.csv"
            generated.append(write_submission(test_meta, action_pred, point_pred, server_test, name, {**action_info, **point_info}))

    pd.DataFrame(generated).to_csv(OUTDIR / "r87_r90_generated_candidates.csv", index=False)

    # R90 diagnostic only.
    diag = future_score_diagnostic(test_raw)
    diag.to_csv(OUTDIR / "r90_scoreboard_pseudolabel_diagnostic.csv", index=False)
    r90_report = {
        "covered_rows": int(len(diag)),
        "coverage": float(len(diag) / max(len(test_meta), 1)),
        "high_conf_server_rate_rows": int((np.abs(diag["future_server_score_rate"] - 0.5) >= 0.5).sum()) if len(diag) else 0,
        "note": "Diagnostic only; no R90 submission is generated pending organizer response.",
    }
    (OUTDIR / "r90_report.json").write_text(json.dumps(r90_report, indent=2), encoding="utf-8")

    report = {
        "base": {
            "r67_w0p2_action_f1": float(base_action_f1),
            "v3_point_f1": float(base_point_f1),
        },
        "best_point": sorted(best_point, key=lambda r: (r["point_macro_f1"], -r["point_churn"]), reverse=True)[:10],
        "best_action": sorted(best_action, key=lambda r: (r["action_macro_f1"], -r["action_churn"]), reverse=True)[:10],
        "generated": generated,
        "r90": r90_report,
    }
    (OUTDIR / "r87_r90_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(30).to_string(index=False))
    print(pd.DataFrame(generated).head(40).to_string(index=False))
    print(json.dumps(r90_report, indent=2))


if __name__ == "__main__":
    main()
