"""R79-R81 structural target experiments.

R79:
  Joint action-point target model. Train a multiclass model on
  joint_target = actionId * 10 + pointId, then marginalize back to action/point.

R80:
  Next-next action distillation. Train a t -> t+2 action model and use its
  probability vector as meta-features for the real t -> t+1 action model.

R81:
  Adversarial validation / pruning. Train a train-vs-test classifier, report
  shifted features, and retrain action experts after dropping top-shift fields.

Submissions generated here change action only. Point/server stay fixed to the
current R34 branch.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r79_r81_structural_targets")
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
    return str(float(x)).replace(".", "p").replace("-", "m")


def make_lgbm_multiclass(seed: int, n_estimators: int = 180, num_leaves: int = 39) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=num_leaves,
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


def make_lgbm_binary(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=260,
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


def fill_action_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        if cls in ACTION_CLASSES:
            out[:, cls] = raw[:, i]
    return normalize_rows(out)


def joint_sample_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    weights = y.map(lambda v: min(8.0, 1.0 / np.sqrt(float(counts[int(v)])))).to_numpy(dtype=float)
    return weights / np.mean(weights)


def joint_marginalize(raw: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.zeros((len(raw), len(ACTION_CLASSES)), dtype=float)
    point = np.zeros((len(raw), len(POINT_CLASSES)), dtype=float)
    for i, cls in enumerate(classes.astype(int)):
        a = cls // 10
        p = cls % 10
        if 0 <= a < len(ACTION_CLASSES) and 0 <= p < len(POINT_CLASSES):
            action[:, a] += raw[:, i]
            point[:, p] += raw[:, i]
    return normalize_rows(action), normalize_rows(point)


def r79_joint_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    action = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    point = np.zeros((len(prefix_aligned), len(POINT_CLASSES)), dtype=float)
    train_df = prefix.copy()
    train_df["joint_target"] = train_df["next_actionId"].astype(int) * 10 + train_df["next_pointId"].astype(int)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = train_df[~train_df["match"].isin(valid_matches)].copy()
        va = prefix_aligned.loc[idx].copy()
        model = make_lgbm_multiclass(7900 + int(fold), n_estimators=220, num_leaves=47)
        model.fit(tr[features], tr["joint_target"], sample_weight=joint_sample_weight(tr["joint_target"]))
        raw = model.predict_proba(va[features])
        a, p = joint_marginalize(raw, model.classes_)
        action[idx] = a
        point[idx] = p
    return action, point


def r79_joint_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tr = prefix.copy()
    tr["joint_target"] = tr["next_actionId"].astype(int) * 10 + tr["next_pointId"].astype(int)
    model = make_lgbm_multiclass(8900, n_estimators=220, num_leaves=47)
    model.fit(tr[features], tr["joint_target"], sample_weight=joint_sample_weight(tr["joint_target"]))
    raw = model.predict_proba(test_prefix[features])
    return joint_marginalize(raw, model.classes_)


def add_next2_target(prefix: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    lut = raw.set_index(["rally_uid", "strikeNumber"])["actionId"].astype(int).to_dict()
    out = prefix.copy()
    out["next2_actionId"] = [
        int(lut.get((int(row.rally_uid), int(row.prefix_len) + 2), -1))
        for row in out.itertuples(index=False)
    ]
    return out


def next2_oof_for_rows(rows: pd.DataFrame, train_df: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    tr = train_df[train_df["next2_actionId"].ge(0)].copy()
    if len(tr) == 0:
        return np.ones((len(rows), len(ACTION_CLASSES)), dtype=float) / len(ACTION_CLASSES)
    model = make_lgbm_multiclass(seed, n_estimators=150)
    model.fit(tr[features], tr["next2_actionId"], sample_weight=class_weight_sample(tr["next2_actionId"]))
    return fill_action_proba(model, rows[features])


def inner_next2_oof(train_df: pd.DataFrame, features: list[str], fold_seed: int) -> np.ndarray:
    out = np.zeros((len(train_df), len(ACTION_CLASSES)), dtype=float)
    valid_mask = train_df["next2_actionId"].ge(0).to_numpy()
    if valid_mask.sum() < 100:
        return out + 1.0 / len(ACTION_CLASSES)
    groups = train_df["match"].to_numpy()
    for inner, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=3).split(train_df, groups=groups), start=1):
        tr = train_df.iloc[tr_idx]
        tr = tr[tr["next2_actionId"].ge(0)].copy()
        va = train_df.iloc[va_idx].copy()
        model = make_lgbm_multiclass(fold_seed + inner, n_estimators=130)
        model.fit(tr[features], tr["next2_actionId"], sample_weight=class_weight_sample(tr["next2_actionId"]))
        out[va_idx] = fill_action_proba(model, va[features])
    return out


def add_next2_prob_features(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for c in ACTION_CLASSES:
        out[f"r80_next2_p_{c}"] = prob[:, c]
    order = np.argsort(-prob, axis=1)
    out["r80_next2_top"] = order[:, 0].astype(int)
    out["r80_next2_margin"] = prob[np.arange(len(prob)), order[:, 0]] - prob[np.arange(len(prob)), order[:, 1]]
    return out


def r80_next2_action_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, raw: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    prefix2 = add_next2_target(prefix, raw)
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    feature_cols: list[str] | None = None
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = prefix2[~prefix2["match"].isin(valid_matches)].copy().reset_index(drop=True)
        va = add_next2_target(prefix_aligned.loc[idx].copy(), raw)
        tr_next2 = inner_next2_oof(tr, features, 8000 + int(fold) * 10)
        va_next2 = next2_oof_for_rows(va, tr, features, 8050 + int(fold))
        tr_aug = add_next2_prob_features(tr, tr_next2)
        va_aug = add_next2_prob_features(va, va_next2)
        cols = [c for c in features if c in tr_aug.columns] + [c for c in tr_aug.columns if c.startswith("r80_")]
        feature_cols = cols
        model = make_lgbm_multiclass(8100 + int(fold), n_estimators=170)
        model.fit(tr_aug[cols], tr_aug["next_actionId"], sample_weight=class_weight_sample(tr_aug["next_actionId"]))
        out[idx] = fill_action_proba(model, va_aug[cols])
    assert feature_cols is not None
    return out, feature_cols


def r80_next2_action_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, raw: pd.DataFrame, features: list[str], feature_cols: list[str]) -> np.ndarray:
    prefix2 = add_next2_target(prefix, raw)
    tr_next2 = inner_next2_oof(prefix2.reset_index(drop=True), features, 8800)
    te_next2 = next2_oof_for_rows(test_prefix, prefix2, features, 8850)
    tr_aug = add_next2_prob_features(prefix2.reset_index(drop=True), tr_next2)
    te_aug = add_next2_prob_features(test_prefix.copy(), te_next2)
    model = make_lgbm_multiclass(8901, n_estimators=170)
    model.fit(tr_aug[feature_cols], tr_aug["next_actionId"], sample_weight=class_weight_sample(tr_aug["next_actionId"]))
    return fill_action_proba(model, te_aug[feature_cols])


def adversarial_report(prefix_aligned: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, float]:
    train_sample = prefix_aligned[features].copy()
    test_sample = test_prefix[features].copy()
    n = min(len(train_sample), len(test_sample))
    train_sample = train_sample.sample(n=n, random_state=8100)
    test_sample = test_sample.sample(n=n, random_state=8101)
    x = pd.concat([train_sample, test_sample], ignore_index=True)
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    oof = np.zeros(len(x), dtype=float)
    importances = np.zeros(len(features), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=8123).split(x, y), start=1):
        model = make_lgbm_binary(8200 + fold)
        model.fit(x.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
        importances += model.feature_importances_
    auc = float(roc_auc_score(y, oof))
    report = pd.DataFrame({"feature": features, "importance": importances / 5.0}).sort_values("importance", ascending=False)
    return report, auc


def r81_pruned_oof(prefix_aligned: pd.DataFrame, prefix: pd.DataFrame, features: list[str], drop_features: set[str]) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in features if c not in drop_features]
    out = np.zeros((len(prefix_aligned), len(ACTION_CLASSES)), dtype=float)
    for fold in sorted(prefix_aligned["fold"].unique()):
        idx = prefix_aligned.index[prefix_aligned["fold"].eq(fold)].to_numpy()
        valid_matches = set(prefix_aligned.loc[idx, "match"])
        tr = prefix[~prefix["match"].isin(valid_matches)].copy()
        va = prefix_aligned.loc[idx].copy()
        model = make_lgbm_multiclass(8300 + int(fold), n_estimators=170)
        model.fit(tr[cols], tr["next_actionId"], sample_weight=class_weight_sample(tr["next_actionId"]))
        out[idx] = fill_action_proba(model, va[cols])
    return out, cols


def r81_pruned_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, cols: list[str]) -> np.ndarray:
    model = make_lgbm_multiclass(8950, n_estimators=170)
    model.fit(prefix[cols], prefix["next_actionId"], sample_weight=class_weight_sample(prefix["next_actionId"]))
    return fill_action_proba(model, test_prefix[cols])


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

    r79_action_oof, r79_point_oof = r79_joint_oof(prefix_aligned, prefix, features)
    r80_oof, r80_cols = r80_next2_action_oof(prefix_aligned, prefix, train, features)
    adv_imp, adv_auc = adversarial_report(prefix_aligned, test_prefix, features)
    adv_imp.to_csv(OUTDIR / "r81_adversarial_feature_importance.csv", index=False)
    r81_oofs: dict[str, tuple[np.ndarray, list[str]]] = {}
    for cut in [10, 20, 40, 60]:
        drop = set(adv_imp.head(cut)["feature"].tolist())
        r81_oofs[f"r81_pruned_top{cut}"] = r81_pruned_oof(prefix_aligned, prefix, features, drop)

    experts = {"r79_joint_action": r79_action_oof, "r80_next2": r80_oof}
    experts.update({name: val[0] for name, val in r81_oofs.items()})
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
    # Point marginal diagnostics only; point is not submitted from this script.
    point_diag = []
    for w in [0.02, 0.05, 0.10, 0.20]:
        point_pred = np.argmax(r79_point_oof, axis=1)
        point_diag.append(
            {
                "source": "r79_joint_point_direct",
                "weight": w,
                "point_macro_f1_direct": float(
                    f1_score(meta["next_pointId"].to_numpy(dtype=int), point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
                ),
            }
        )
    pd.DataFrame(point_diag).to_csv(OUTDIR / "r79_point_diagnostic.csv", index=False)

    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True])
    search.to_csv(OUTDIR / "r79_r81_oof_search.csv", index=False)

    # Full test.
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    r79_action_test, _ = r79_joint_test(prefix, test_prefix, features)
    r80_test = r80_next2_action_test(prefix, test_prefix, train, features, r80_cols)
    r81_tests = {name: r81_pruned_test(prefix, test_prefix, cols) for name, (_, cols) in r81_oofs.items()}
    test_experts = {"r79_joint_action": r79_action_test, "r80_next2": r80_test, **r81_tests}
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
        if kind not in test_experts:
            continue
        w = float(row.weight)
        pred = apply_action(blend_action_prob(r42_test, test_experts[kind], w), test_meta, mult)
        name = f"submission_{kind}_blend_w{clean_float(w)}_current_point_server.csv"
        info = write_submission(test_meta, pred, current_sub, name)
        info["source_candidate"] = str(row.candidate)
        info["source_kind"] = kind
        info["source_oof_action_f1"] = float(row.action_macro_f1)
        info["source_oof_churn"] = float(row.churn_vs_r42)
        info["weight"] = w
        generated.append(info)
    pd.DataFrame(generated).to_csv(OUTDIR / "r79_r81_generated_candidates.csv", index=False)
    report = {
        "base_action_macro_f1": base_f1,
        "adversarial_auc": adv_auc,
        "top_shift_features": adv_imp.head(30).to_dict(orient="records"),
        "top_oof": search.head(40).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "r79_r81_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(40).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))
    print(f"Adversarial AUC: {adv_auc:.6f}")
    print(adv_imp.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
