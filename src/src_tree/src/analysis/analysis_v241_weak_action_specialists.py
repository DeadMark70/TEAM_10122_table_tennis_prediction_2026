"""V241 weak action specialists with precision-constrained posterior changes."""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows, precision_constrained_threshold, select_top_changes
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v241_weak_action_specialists")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v241_weak_action_specialists.py")
WEAK_ACTIONS = [0, 3, 4, 5, 7, 8, 9, 12, 14]


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


def train_ovr_scores(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof = np.zeros((len(rows), len(WEAK_ACTIONS)), dtype=float)
    test_sum = np.zeros((len(test_rows), len(WEAK_ACTIONS)), dtype=float)
    metrics = []
    for j, action in enumerate(WEAK_ACTIONS):
        for fold in sorted(rows["fold"].astype(int).unique()):
            valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
            train = ~valid
            target = (y[train] == int(action)).astype(int)
            if target.sum() < 5:
                continue
            clf = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=3, class_weight="balanced", random_state=241 + action * 10 + int(fold), n_jobs=1)
            clf.fit(rows.loc[train, cols].fillna(0), target)
            p = clf.predict_proba(rows.loc[valid, cols].fillna(0))[:, 1]
            oof[valid, j] = p
            test_sum[:, j] += clf.predict_proba(test_rows[cols].fillna(0))[:, 1] / rows["fold"].nunique()
        auc = roc_auc_score((y == int(action)).astype(int), oof[:, j]) if len(np.unique((y == int(action)).astype(int))) > 1 else np.nan
        metrics.append({"action": int(action), "auc": float(auc), "mean_score": float(oof[:, j].mean())})
    return oof, test_sum, metrics


def specialist_probability(v173_prob: np.ndarray, scores: np.ndarray, strength: float) -> np.ndarray:
    p = normalize_probability_rows(v173_prob).copy()
    for j, action in enumerate(WEAK_ACTIONS):
        p[:, action] *= np.exp(float(strength) * scores[:, j])
    return normalize_probability_rows(p)


def best_specialist_action(anchor: np.ndarray, scores: np.ndarray, thresholds: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    cand = np.asarray(anchor, dtype=int).copy()
    gain = np.zeros(len(anchor), dtype=float)
    for j, action in enumerate(WEAK_ACTIONS):
        mask = scores[:, j] >= thresholds.get(action, np.inf)
        better = mask & (scores[:, j] > gain)
        cand[better] = int(action)
        gain[better] = scores[better, j]
    return cand, gain


def evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    iw = weighted_macro_f1(y, pred, weights)
    base_iw = weighted_macro_f1(y, anchor, weights)
    weak = f1_score(y, pred, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    weak_base = f1_score(y, anchor, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    return {"candidate": name, "action_macro_f1": float(score), "delta_vs_v173_anchor": float(score - base), "iw_delta_vs_v173": float(iw - base_iw), "weak_delta_vs_v173": float(weak - weak_base), "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)), "changed_rows": int(np.sum(pred != anchor))}


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
    scores_oof, scores_test, metrics = train_ovr_scores(rows, test_rows, y, cols)
    thresholds = {}
    for j, action in enumerate(WEAK_ACTIONS):
        thresholds[action] = precision_constrained_threshold(scores_oof[:, j], (y == action).astype(int), min_precision=0.55)
    weights = context_weights(rows, test_rows)
    records = [evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    for strength in [0.5, 1.0, 1.5]:
        p_oof = specialist_probability(v173_prob_oof, scores_oof, strength)
        p_test = specialist_probability(v173_prob_test, scores_test, strength)
        name = f"v241_weakposterior_s{str(strength).replace('.', 'p')}"
        pred = p_oof.argmax(axis=1).astype(int)
        test_pred = p_test.argmax(axis=1).astype(int)
        rec = evaluate(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", p_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", p_test)
        generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    cand_oof, gain_oof = best_specialist_action(v173_oof, scores_oof, thresholds)
    cand_test, gain_test = best_specialist_action(v173_test, scores_test, thresholds)
    for cap in [0.002, 0.005, 0.01]:
        pred, _ = select_top_changes(v173_oof, cand_oof, gain_oof, cap=cap, min_score=0.0)
        test_pred, test_changed = select_top_changes(v173_test, cand_test, gain_test, cap=cap, min_score=0.0)
        name = f"v241_precision_cap{str(cap).replace('.', 'p')}"
        rec = evaluate(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(test_changed.sum())
        records.append(rec)
        generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "weak_delta_vs_v173", "iw_delta_vs_v173"], ascending=[False, False, False])
    search.to_csv(OUTDIR / "v241_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v241_specialist_metrics.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    (OUTDIR / "v241_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v241_weak_action_specialists.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
