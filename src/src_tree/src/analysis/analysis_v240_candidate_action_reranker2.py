"""V240 candidate action reranker 2.0.

This is a broader candidate-ranking version of V209/V217.  It uses action
candidate probabilities from V173, R166, V234, V236, and V238 when available.
The final output is still low/medium capped action changes over V173 with V188
cap5 point and R121 server fixed.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor, rebuild_r166_best_action
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v238_v242_action_model_helpers import normalize_probability_rows, select_top_changes, topk_candidate_frame
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v240_candidate_action_reranker2")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v240_candidate_action_reranker2.py")
CAPS = [0.002, 0.005, 0.01, 0.02]


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


def optional_sources(v173_oof: np.ndarray, v173_test: np.ndarray, r166_oof: np.ndarray, r166_test: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof = {"v173": v173_oof, "r166": r166_oof}
    test = {"v173": v173_test, "r166": r166_test}
    paths = {
        "v234": ("v234_v173_phase_expert_reconstruction/v234_phase_v173kd_w0p50_oof_action_prob.npy", "v234_v173_phase_expert_reconstruction/v234_phase_v173kd_w0p50_test_action_prob.npy"),
        "v236": ("v236_distributional_action_calibrator/v236_v234_phase_v173kd_w0p35_weaktemp_oof_action_prob.npy", "v236_distributional_action_calibrator/v236_v234_phase_v173kd_w0p35_weaktemp_test_action_prob.npy"),
        "v238": ("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_oof_action_prob.npy", "v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_test_action_prob.npy"),
    }
    for name, (a, b) in paths.items():
        pa, pb = Path(a), Path(b)
        if pa.exists() and pb.exists():
            oof[name] = normalize_probability_rows(np.load(pa))
            test[name] = normalize_probability_rows(np.load(pb))
    return oof, test


def candidate_features(frame: pd.DataFrame, probs: dict[str, np.ndarray], rows: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cand = out["candidate_action"].astype(int).to_numpy()
    row_id = out["row_id"].astype(int).to_numpy()
    for name, prob in probs.items():
        p = normalize_probability_rows(prob)
        out[f"{name}_p"] = p[row_id, cand]
        out[f"{name}_top"] = p[row_id].argmax(axis=1)
        out[f"{name}_agrees"] = (out[f"{name}_top"].astype(int).to_numpy() == cand).astype(int)
        part = np.partition(p[row_id], -2, axis=1)
        out[f"{name}_margin"] = part[:, -1] - part[:, -2]
    meta_cols = ["prefix_len", "lag0_actionId", "lag0_pointId", "lag0_spinId", "lag0_strengthId", "audit_phase", "audit_lag0_action_family", "audit_lag0_depth"]
    meta = rows.reset_index(drop=True).loc[row_id, [c for c in meta_cols if c in rows.columns]].reset_index(drop=True)
    out = pd.concat([out.reset_index(drop=True), meta], axis=1)
    x = out[["candidate_action", "source_prob", "source_rank"] + [c for c in out.columns if c.endswith("_p") or c.endswith("_agrees") or c.endswith("_margin")]].apply(pd.to_numeric, errors="coerce").fillna(0)
    cats = [c for c in ["source", "audit_phase", "audit_lag0_action_family", "audit_lag0_depth"] if c in out.columns]
    if cats:
        x = pd.concat([x, pd.get_dummies(out[cats].astype(str), prefix=cats, dtype=float)], axis=1)
    return x


def align(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for c in cols:
        if c not in out:
            out[c] = 0.0
    return out[cols].astype(float)


def best_by_row(frame: pd.DataFrame, scores: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    best_action = np.full(n_rows, -1, dtype=int)
    best_score = np.full(n_rows, -np.inf, dtype=float)
    for row, action, score in zip(frame["row_id"].astype(int), frame["candidate_action"].astype(int), scores):
        if score > best_score[row]:
            best_score[row] = float(score)
            best_action[row] = int(action)
    return best_action, best_score


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
    _r166_oof, _r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    sources_oof, sources_test = optional_sources(v173_prob_oof, v173_prob_test, r166_prob_oof, r166_prob_test)
    frame = topk_candidate_frame(v173_oof, sources_oof, top_k=3)
    test_frame = topk_candidate_frame(v173_test, sources_test, top_k=3)
    frame["is_correct"] = (frame["candidate_action"].astype(int).to_numpy() == y[frame["row_id"].astype(int).to_numpy()]).astype(int)
    oof_best = np.zeros(len(rows), dtype=int)
    oof_score = np.full(len(rows), -np.inf, dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows = set(np.where(~valid)[0])
        valid_rows = set(np.where(valid)[0])
        train_f = frame[frame["row_id"].isin(train_rows)].copy()
        valid_f = frame[frame["row_id"].isin(valid_rows)].copy()
        x_train = candidate_features(train_f, sources_oof, rows)
        x_valid = align(candidate_features(valid_f, sources_oof, rows), list(x_train.columns))
        y_train = train_f["is_correct"].astype(int).to_numpy()
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.35, max_iter=1000, random_state=240)
        clf.fit(x_train, y_train)
        p = clf.predict_proba(x_valid)[:, 1]
        best_action, best_score = best_by_row(valid_f, p, len(rows))
        idx = np.where(valid)[0]
        oof_best[idx] = best_action[idx]
        oof_score[idx] = best_score[idx]
        metrics.append({"fold": int(fold), "candidate_auc": float(roc_auc_score(valid_f["is_correct"].astype(int), p)) if valid_f["is_correct"].nunique() > 1 else np.nan})
    x_full = candidate_features(frame, sources_oof, rows)
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.35, max_iter=1000, random_state=240)
    clf.fit(x_full, frame["is_correct"].astype(int).to_numpy())
    p_test = clf.predict_proba(align(candidate_features(test_frame, sources_test, test_rows), list(x_full.columns)))[:, 1]
    test_best, test_score = best_by_row(test_frame, p_test, len(test_rows))
    weights = context_weights(rows, test_rows)
    records = [evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    for cap in CAPS:
        pred, changed = select_top_changes(v173_oof, oof_best, oof_score, cap=cap, min_score=0.0)
        test_pred, test_changed = select_top_changes(v173_test, test_best, test_score, cap=cap, min_score=0.0)
        name = f"v240_reranker2_cap{str(cap).replace('.', 'p')}"
        rec = evaluate(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(test_changed.sum())
        records.append(rec)
        generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v240_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v240_fold_metrics.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    (OUTDIR / "v240_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v240_candidate_action_reranker2.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
