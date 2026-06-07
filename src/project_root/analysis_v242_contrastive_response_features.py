"""V242 contrastive response feature probe.

Builds simple response-context embeddings and nearest-centroid action posterior.
This is a lightweight probe for the "same incoming state -> similar response"
hypothesis without training a heavy neural contrastive model.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v242_contrastive_response_features")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v242_contrastive_response_features.py")


def response_feature_columns(rows: pd.DataFrame) -> list[str]:
    cols = []
    patterns = ["prefix_len", "serverScore", "receiverScore", "serverScoreDiff", "scoreTotal", "lag0_", "lag1_", "is_receive", "is_third", "is_fourth", "is_rally", "count_actionId_", "count_pointId_"]
    for c in rows.columns:
        if pd.api.types.is_numeric_dtype(rows[c]) and any(str(c).startswith(p) for p in patterns):
            cols.append(c)
    return cols


def centroid_posterior(x_train: np.ndarray, y_train: np.ndarray, x_apply: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    centroids = np.zeros((19, x_train.shape[1]), dtype=float)
    counts = np.bincount(y_train, minlength=19).astype(float)
    for cls in range(19):
        if counts[cls] > 0:
            centroids[cls] = x_train[y_train == cls].mean(axis=0)
    d2 = ((x_apply[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    logits = -d2 / max(float(temperature), 1e-6)
    logits[:, counts == 0] = -1e9
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probability_rows(np.exp(logits))


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


def build_oof_centroid(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(rows), 19), dtype=float)
    test_sum = np.zeros((len(test_rows), 19), dtype=float)
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        scaler = StandardScaler()
        x_train = scaler.fit_transform(rows.loc[train, cols].fillna(0))
        x_valid = scaler.transform(rows.loc[valid, cols].fillna(0))
        x_test = scaler.transform(test_rows[cols].fillna(0))
        oof[valid] = centroid_posterior(x_train, y[train], x_valid, temperature=0.75)
        test_sum += centroid_posterior(x_train, y[train], x_test, temperature=0.75) / rows["fold"].nunique()
    return normalize_probability_rows(oof), normalize_probability_rows(test_sum)


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
    cols = response_feature_columns(rows)
    for c in cols:
        if c not in test_rows:
            test_rows[c] = 0
    centroid_oof, centroid_test = build_oof_centroid(rows, test_rows, y, cols)
    weights = context_weights(rows, test_rows)
    variants = {
        "v242_centroid_raw": (centroid_oof, centroid_test),
        "v242_centroid_v173blend_w0p10": (blend_probabilities(v173_prob_oof, centroid_oof, 0.10), blend_probabilities(v173_prob_test, centroid_test, 0.10)),
        "v242_centroid_v173blend_w0p20": (blend_probabilities(v173_prob_oof, centroid_oof, 0.20), blend_probabilities(v173_prob_test, centroid_test, 0.20)),
        "v242_centroid_v173blend_w0p35": (blend_probabilities(v173_prob_oof, centroid_oof, 0.35), blend_probabilities(v173_prob_test, centroid_test, 0.35)),
    }
    records = [evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
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
    search.to_csv(OUTDIR / "v242_action_search.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    (OUTDIR / "v242_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v242_contrastive_response_features.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
