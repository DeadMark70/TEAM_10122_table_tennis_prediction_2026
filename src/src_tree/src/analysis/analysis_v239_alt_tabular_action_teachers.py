"""V239 alternative tabular action teachers.

Trains non-LGBM tabular teachers (ExtraTrees and HistGradientBoosting) as bold
alternative action sources.  Outputs raw and V173-blended variants with point
fixed at V188 cap5 and server fixed at R121.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v237_deep_phase_style_action import class_balanced_sample_weight
from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v239_alt_tabular_action_teachers")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v239_alt_tabular_action_teachers.py")


def feature_columns(rows: pd.DataFrame) -> list[str]:
    blocked = {"rally_uid", "match", "next_actionId", "next_pointId", "serverGetPoint", "fold"}
    return [c for c in rows.columns if c not in blocked and pd.api.types.is_numeric_dtype(rows[c])]


def context_weights(rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
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


def predict_full(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 19), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return normalize_probability_rows(out)


def train_oof(model_name: str, rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray, cols: list[str], sample_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof = np.zeros((len(rows), 19), dtype=float)
    test_sum = np.zeros((len(test_rows), 19), dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        if model_name == "extratrees":
            model = ExtraTreesClassifier(n_estimators=160, max_depth=None, min_samples_leaf=4, class_weight="balanced", random_state=239 + int(fold), n_jobs=1)
            model.fit(rows.loc[train, cols].fillna(0), y[train])
        else:
            model = LogisticRegression(
                solver="saga",
                penalty="l2",
                C=0.35,
                class_weight="balanced",
                max_iter=500,
                random_state=539 + int(fold),
                n_jobs=1,
            )
            model.fit(rows.loc[train, cols].fillna(0), y[train], sample_weight=sample_weight[train])
        oof[valid] = predict_full(model, rows.loc[valid, cols].fillna(0))
        test_sum += predict_full(model, test_rows[cols].fillna(0))
        metrics.append({"fold": int(fold), "model": model_name, "valid_rows": int(valid.sum())})
    return normalize_probability_rows(oof), normalize_probability_rows(test_sum / len(metrics)), metrics


def evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    iw = weighted_macro_f1(y, pred, weights)
    base_iw = weighted_macro_f1(y, anchor, weights)
    return {"candidate": name, "action_macro_f1": float(score), "delta_vs_v173_anchor": float(score - base), "iw_delta_vs_v173": float(iw - base_iw), "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)), "changed_rows": int(np.sum(pred != anchor))}


def write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"rally_uid": point_src["rally_uid"].astype(int), "actionId": np.asarray(action, dtype=int), "pointId": point_src["pointId"].astype(int), "serverGetPoint": server_src["serverGetPoint"].astype(float)})
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
        if c not in test_rows:
            test_rows[c] = 0
    weights = context_weights(rows, test_rows)
    sample_weight = class_balanced_sample_weight(y, weights, power=0.35, cap=3.0)
    records = [evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    all_metrics = []
    for model_name in ["extratrees", "logreg"]:
        prob_oof, prob_test, metrics = train_oof(model_name, rows, test_rows, y, cols, sample_weight)
        all_metrics.extend(metrics)
        variants = {
            f"v239_{model_name}_raw": (prob_oof, prob_test),
            f"v239_{model_name}_v173blend_w0p20": (blend_probabilities(v173_prob_oof, prob_oof, 0.20), blend_probabilities(v173_prob_test, prob_test, 0.20)),
            f"v239_{model_name}_v173blend_w0p35": (blend_probabilities(v173_prob_oof, prob_oof, 0.35), blend_probabilities(v173_prob_test, prob_test, 0.35)),
        }
        for name, (p_oof, p_test) in variants.items():
            pred = p_oof.argmax(axis=1).astype(int)
            test_pred = p_test.argmax(axis=1).astype(int)
            rec = evaluate(name, y, pred, v173_oof, weights)
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
            records.append(rec)
            np.save(OUTDIR / f"{name}_oof_action_prob.npy", p_oof)
            np.save(OUTDIR / f"{name}_test_action_prob.npy", p_test)
            generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v239_action_search.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(OUTDIR / "v239_fold_metrics.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    (OUTDIR / "v239_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v239_alt_tabular_action_teachers.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
