"""V234 V173 phase expert reconstruction.

This is the first post-deep-research action experiment.  It reconstructs V173 as
phase-specific action experts with exact 19-class output and family/transition
context features.  It is intentionally a model source, not a tiny row selector.

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
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v234_v173_phase_expert_reconstruction")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v234_v173_phase_expert_reconstruction.py")


def feature_columns(rows: pd.DataFrame) -> list[str]:
    blocked = {
        "rally_uid",
        "match",
        "next_actionId",
        "next_pointId",
        "serverGetPoint",
        "fold",
    }
    cols = []
    for c in rows.columns:
        if c in blocked:
            continue
        if pd.api.types.is_numeric_dtype(rows[c]):
            cols.append(c)
    return cols


def phase_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    phase = rows["audit_phase"].astype(str) if "audit_phase" in rows.columns else pd.Series(["rally"] * len(rows), index=rows.index)
    return {
        "receive": phase.eq("receive").to_numpy(),
        "third_ball": phase.eq("third_ball").to_numpy(),
        "rally": phase.eq("rally").to_numpy(),
        "other": ~(phase.isin(["receive", "third_ball", "rally"]).to_numpy()),
    }


def fit_action_model(x: pd.DataFrame, y: np.ndarray, seed: int = 234) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=19,
        n_estimators=160,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.60,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def predict_full_classes(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 19), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return normalize_rows_safe(out)


def train_phase_expert_probs(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    folds = sorted(rows["fold"].astype(int).unique())
    oof = np.zeros((len(rows), 19), dtype=float)
    metrics = []
    train_phase = phase_masks(rows)
    test_prob_accum = np.zeros((len(test_rows), 19), dtype=float)
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        fold_oof = np.zeros((valid.sum(), 19), dtype=float)
        valid_index = np.where(valid)[0]
        for phase_name, full_mask in train_phase.items():
            valid_phase = full_mask[valid]
            if valid_phase.sum() == 0:
                continue
            train_phase_mask = full_mask[train]
            if train_phase_mask.sum() < 100 or len(np.unique(y[train][train_phase_mask])) < 2:
                train_phase_mask = np.ones(train.sum(), dtype=bool)
            x_train = rows.loc[train, feature_cols].reset_index(drop=True).loc[train_phase_mask].fillna(0)
            y_train = y[train][train_phase_mask]
            model = fit_action_model(x_train, y_train, seed=234 + int(fold))
            x_valid = rows.loc[valid, feature_cols].reset_index(drop=True).loc[valid_phase].fillna(0)
            fold_oof[valid_phase] = predict_full_classes(model, x_valid)
            metrics.append({"fold": int(fold), "phase": phase_name, "train_rows": int(len(x_train)), "valid_rows": int(valid_phase.sum())})
        empty = fold_oof.sum(axis=1) == 0
        if empty.any():
            model = fit_action_model(rows.loc[train, feature_cols].fillna(0), y[train], seed=734 + int(fold))
            fold_oof[empty] = predict_full_classes(model, rows.loc[valid, feature_cols].reset_index(drop=True).loc[empty].fillna(0))
        oof[valid_index] = normalize_rows_safe(fold_oof)

    # Full-data test models per phase.
    test_phase = phase_masks(test_rows)
    for phase_name, test_mask in test_phase.items():
        if test_mask.sum() == 0:
            continue
        train_mask = train_phase.get(phase_name, np.zeros(len(rows), dtype=bool))
        if train_mask.sum() < 100 or len(np.unique(y[train_mask])) < 2:
            train_mask = np.ones(len(rows), dtype=bool)
        model = fit_action_model(rows.loc[train_mask, feature_cols].fillna(0), y[train_mask], seed=934)
        test_prob_accum[test_mask] = predict_full_classes(model, test_rows.loc[test_mask, feature_cols].fillna(0))
    empty_test = test_prob_accum.sum(axis=1) == 0
    if empty_test.any():
        model = fit_action_model(rows[feature_cols].fillna(0), y, seed=1234)
        test_prob_accum[empty_test] = predict_full_classes(model, test_rows.loc[empty_test, feature_cols].fillna(0))
    return normalize_rows_safe(oof), normalize_rows_safe(test_prob_accum), metrics


def _context_weights(rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
    def frame(r: pd.DataFrame) -> pd.DataFrame:
        prefix = pd.to_numeric(r["prefix_len"], errors="coerce").fillna(0).astype(int)
        return pd.DataFrame(
            {
                "prefix_bin": prefix.map(lambda v: "1" if v <= 1 else "2" if v == 2 else "3" if v == 3 else "4_6" if v <= 6 else "7_plus"),
                "phase": r["audit_phase"].astype(str),
                "lag0_family": r["audit_lag0_action_family"].astype(str),
                "lag0_depth": r["audit_lag0_depth"].astype(str),
            }
        )
    return density_ratio_weights(frame(rows), frame(test_rows), ["prefix_bin", "phase", "lag0_family", "lag0_depth"])


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
    rows = add_audit_columns(data["rows"].copy())
    test_rows = add_audit_columns(state["test_rows"].copy())
    y = rows["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    cols = feature_columns(rows)
    for c in cols:
        if c not in test_rows.columns:
            test_rows[c] = 0
    test_rows = test_rows.copy()
    phase_oof, phase_test, fold_metrics = train_phase_expert_probs(rows, test_rows, y, cols)
    weights = _context_weights(rows, test_rows)
    records = [_evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    variants = {
        "v234_phase_raw": (phase_oof, phase_test),
        "v234_phase_v173kd_w0p20": (geometric_log_blend(v173_prob_oof, phase_oof, 0.20), geometric_log_blend(v173_prob_test, phase_test, 0.20)),
        "v234_phase_v173kd_w0p35": (geometric_log_blend(v173_prob_oof, phase_oof, 0.35), geometric_log_blend(v173_prob_test, phase_test, 0.35)),
        "v234_phase_v173kd_w0p50": (geometric_log_blend(v173_prob_oof, phase_oof, 0.50), geometric_log_blend(v173_prob_test, phase_test, 0.50)),
    }
    generated = []
    for name, (prob_oof, prob_test) in variants.items():
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = _evaluate(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", prob_test)
        generated.append(_write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v234_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v234_fold_phase_metrics.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}
    (OUTDIR / "v234_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v234_report.md").write_text(f"# V234 V173 Phase Expert Reconstruction\n\n- Verdict: `{verdict}`\n- Best delta vs V173: `{best_delta:.6f}`\n", encoding="utf-8")
    shutil.copy2("analysis_v234_v173_phase_expert_reconstruction.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
