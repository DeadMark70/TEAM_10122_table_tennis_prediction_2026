"""V237 deepened V234 phase/style action teacher.

V237 is the concrete follow-up to V234:

- phase-specific exact action experts
- auxiliary family head reconstructed into action probability
- fold-safe player/style response prior features
- public-like density-ratio sample weights
- class-balanced weak/tail weighting

Point is fixed at V188 cap5 and server is fixed at R121.  No TTMATCH and no
old-server labels are read.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v230_action_soft_teacher_factory import geometric_log_blend, normalize_rows_safe
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v235_player_conditional_response_teacher import _merge_target_players, build_response_prior, response_context
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v237_deep_phase_style_action")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v237_deep_phase_style_action.py")

ACTION_FAMILY_IDS = {
    0: [0],
    1: [1, 2, 3, 4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14],
    4: [15, 16, 17, 18],
}
WEAK_ACTIONS = [0, 3, 4, 5, 7, 8, 9, 12, 14]


def family_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    out = np.zeros(len(y), dtype=int)
    for fam, ids in ACTION_FAMILY_IDS.items():
        out[np.isin(y, ids)] = int(fam)
    return out


def hierarchical_action_probability(exact_prob: np.ndarray, family_prob: np.ndarray) -> np.ndarray:
    exact = normalize_rows_safe(exact_prob)
    fam = normalize_rows_safe(family_prob)
    out = np.zeros_like(exact, dtype=float)
    for family_id, ids in ACTION_FAMILY_IDS.items():
        local = exact[:, ids]
        local_sum = local.sum(axis=1, keepdims=True)
        cond = np.divide(local, local_sum, out=np.full_like(local, 1.0 / len(ids)), where=local_sum > 0)
        out[:, ids] = cond * fam[:, [family_id]]
    return normalize_rows_safe(out)


def class_balanced_sample_weight(y: np.ndarray, base_weight: np.ndarray | None = None, power: float = 0.5, cap: float = 4.0) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if base_weight is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(base_weight, dtype=float).copy()
    counts = np.bincount(y, minlength=19).astype(float)
    median = np.median(counts[counts > 0])
    class_scale = np.ones(19, dtype=float)
    for cls in range(19):
        if counts[cls] > 0:
            class_scale[cls] = min(float(cap), max(0.25, (median / counts[cls]) ** float(power)))
    return w * class_scale[y]


def append_prior_features(rows: pd.DataFrame, prior: np.ndarray, prefix: str = "resp") -> pd.DataFrame:
    out = rows.copy()
    p = normalize_rows_safe(prior)
    fam_names = ["zero", "attack", "control", "defensive", "serve"]
    for fam_id, ids in ACTION_FAMILY_IDS.items():
        out[f"{prefix}_family_{fam_names[fam_id]}"] = p[:, ids].sum(axis=1)
    for action in WEAK_ACTIONS:
        out[f"{prefix}_action_{action}"] = p[:, action]
    out[f"{prefix}_entropy"] = -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1)
    out[f"{prefix}_margin"] = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
    return out


def _phase_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    phase = rows["audit_phase"].astype(str) if "audit_phase" in rows.columns else pd.Series(["rally"] * len(rows), index=rows.index)
    return {
        "receive": phase.eq("receive").to_numpy(),
        "third_ball": phase.eq("third_ball").to_numpy(),
        "rally": phase.eq("rally").to_numpy(),
        "other": ~(phase.isin(["receive", "third_ball", "rally"]).to_numpy()),
    }


def _feature_columns(rows: pd.DataFrame) -> list[str]:
    blocked = {"rally_uid", "match", "next_actionId", "next_pointId", "serverGetPoint", "fold"}
    return [c for c in rows.columns if c not in blocked and pd.api.types.is_numeric_dtype(rows[c])]


def _fit_lgbm(x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray, num_class: int, seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=int(num_class),
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.10,
        reg_lambda=0.70,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _predict_classes(model: lgb.LGBMClassifier, x: pd.DataFrame, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), int(n_classes)), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return normalize_rows_safe(out)


def _context_weight_frame(rows: pd.DataFrame) -> pd.DataFrame:
    prefix = pd.to_numeric(rows["prefix_len"], errors="coerce").fillna(0).astype(int)
    return pd.DataFrame(
        {
            "prefix_bin": prefix.map(lambda v: "1" if v <= 1 else "2" if v == 2 else "3" if v == 3 else "4_6" if v <= 6 else "7_plus"),
            "phase": rows["audit_phase"].astype(str),
            "lag0_family": rows["audit_lag0_action_family"].astype(str),
            "lag0_depth": rows["audit_lag0_depth"].astype(str),
        }
    )


def train_deep_phase_probs(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y_family = family_targets(y)
    oof_exact = np.zeros((len(rows), 19), dtype=float)
    oof_family = np.zeros((len(rows), 5), dtype=float)
    test_exact = np.zeros((len(test_rows), 19), dtype=float)
    test_family = np.zeros((len(test_rows), 5), dtype=float)
    metrics = []
    phase_masks = _phase_masks(rows)
    test_phase_masks = _phase_masks(test_rows)
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        fold_exact = np.zeros((valid.sum(), 19), dtype=float)
        fold_family = np.zeros((valid.sum(), 5), dtype=float)
        valid_index = np.where(valid)[0]
        for phase_name, all_mask in phase_masks.items():
            valid_phase = all_mask[valid]
            if valid_phase.sum() == 0:
                continue
            train_phase = all_mask[train]
            if train_phase.sum() < 120 or len(np.unique(y[train][train_phase])) < 2:
                train_phase = np.ones(train.sum(), dtype=bool)
            x_train = rows.loc[train, feature_cols].reset_index(drop=True).loc[train_phase].fillna(0)
            sw_train = sample_weight[train][train_phase]
            exact_model = _fit_lgbm(x_train, y[train][train_phase], sw_train, 19, 237 + int(fold))
            family_model = _fit_lgbm(x_train, y_family[train][train_phase], sw_train, 5, 537 + int(fold))
            x_valid = rows.loc[valid, feature_cols].reset_index(drop=True).loc[valid_phase].fillna(0)
            fold_exact[valid_phase] = _predict_classes(exact_model, x_valid, 19)
            fold_family[valid_phase] = _predict_classes(family_model, x_valid, 5)
            metrics.append({"fold": int(fold), "phase": phase_name, "train_rows": int(len(x_train)), "valid_rows": int(valid_phase.sum())})
        empty = fold_exact.sum(axis=1) == 0
        if empty.any():
            x_train = rows.loc[train, feature_cols].fillna(0)
            exact_model = _fit_lgbm(x_train, y[train], sample_weight[train], 19, 837 + int(fold))
            family_model = _fit_lgbm(x_train, y_family[train], sample_weight[train], 5, 937 + int(fold))
            x_valid = rows.loc[valid, feature_cols].reset_index(drop=True).loc[empty].fillna(0)
            fold_exact[empty] = _predict_classes(exact_model, x_valid, 19)
            fold_family[empty] = _predict_classes(family_model, x_valid, 5)
        oof_exact[valid_index] = fold_exact
        oof_family[valid_index] = fold_family

    for phase_name, test_mask in test_phase_masks.items():
        if test_mask.sum() == 0:
            continue
        train_mask = phase_masks.get(phase_name, np.zeros(len(rows), dtype=bool))
        if train_mask.sum() < 120 or len(np.unique(y[train_mask])) < 2:
            train_mask = np.ones(len(rows), dtype=bool)
        x_train = rows.loc[train_mask, feature_cols].fillna(0)
        exact_model = _fit_lgbm(x_train, y[train_mask], sample_weight[train_mask], 19, 1237)
        family_model = _fit_lgbm(x_train, y_family[train_mask], sample_weight[train_mask], 5, 1537)
        x_test = test_rows.loc[test_mask, feature_cols].fillna(0)
        test_exact[test_mask] = _predict_classes(exact_model, x_test, 19)
        test_family[test_mask] = _predict_classes(family_model, x_test, 5)
    empty_test = test_exact.sum(axis=1) == 0
    if empty_test.any():
        x_train = rows[feature_cols].fillna(0)
        exact_model = _fit_lgbm(x_train, y, sample_weight, 19, 2237)
        family_model = _fit_lgbm(x_train, y_family, sample_weight, 5, 2537)
        x_test = test_rows.loc[empty_test, feature_cols].fillna(0)
        test_exact[empty_test] = _predict_classes(exact_model, x_test, 19)
        test_family[empty_test] = _predict_classes(family_model, x_test, 5)
    return hierarchical_action_probability(oof_exact, oof_family), hierarchical_action_probability(test_exact, test_family), metrics


def _evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    iw = weighted_macro_f1(y, pred, weights)
    base_iw = weighted_macro_f1(y, anchor, weights)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "iw_action_macro_f1": float(iw),
        "iw_delta_vs_v173": float(iw - base_iw),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
    }


def _write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    raw_train = pd.read_csv("train.csv")
    raw_test = pd.read_csv("test_new.csv")
    rows = add_audit_columns(_merge_target_players(data["rows"].copy(), raw_train))
    test_rows = add_audit_columns(_merge_target_players(state["test_rows"].copy(), raw_test))
    y = rows["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    train_ctx = response_context(rows)
    test_ctx = response_context(test_rows)
    oof_prior = np.zeros((len(rows), 19), dtype=float)
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        oof_prior[valid] = build_response_prior(train_ctx.loc[train].reset_index(drop=True), y[train], train_ctx.loc[valid].reset_index(drop=True), smoothing=12.0)
    test_prior = build_response_prior(train_ctx.reset_index(drop=True), y, test_ctx.reset_index(drop=True), smoothing=12.0)
    rows_feat = append_prior_features(rows, oof_prior, prefix="resp")
    test_feat = append_prior_features(test_rows, test_prior, prefix="resp")
    cols = _feature_columns(rows_feat)
    for c in cols:
        if c not in test_feat.columns:
            test_feat[c] = 0
    density_w = density_ratio_weights(_context_weight_frame(rows_feat), _context_weight_frame(test_feat), ["prefix_bin", "phase", "lag0_family", "lag0_depth"])
    sample_weight = class_balanced_sample_weight(y, density_w, power=0.45, cap=3.5)
    hier_oof, hier_test, fold_metrics = train_deep_phase_probs(rows_feat, test_feat, y, cols, sample_weight)
    variants = {
        "v237_deep_phase_hier_raw": (hier_oof, hier_test),
        "v237_deep_phase_hier_w0p35": (geometric_log_blend(v173_prob_oof, hier_oof, 0.35), geometric_log_blend(v173_prob_test, hier_test, 0.35)),
        "v237_deep_phase_hier_w0p50": (geometric_log_blend(v173_prob_oof, hier_oof, 0.50), geometric_log_blend(v173_prob_test, hier_test, 0.50)),
        "v237_deep_phase_hier_w0p65": (geometric_log_blend(v173_prob_oof, hier_oof, 0.65), geometric_log_blend(v173_prob_test, hier_test, 0.65)),
    }
    records = [_evaluate("v173_anchor", y, v173_oof, v173_oof, density_w)]
    generated = []
    for name, (prob_oof, prob_test) in variants.items():
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = _evaluate(name, y, pred, v173_oof, density_w)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", prob_test)
        generated.append(_write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v237_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v237_fold_phase_metrics.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}
    (OUTDIR / "v237_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v237_report.md").write_text(f"# V237 Deep Phase Style Action\n\n- Verdict: `{verdict}`\n- Best delta vs V173: `{best_delta:.6f}`\n", encoding="utf-8")
    shutil.copy2("analysis_v237_deep_phase_style_action.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
