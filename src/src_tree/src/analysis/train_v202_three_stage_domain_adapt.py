"""V202 three-stage domain-adaptive point/action residual.

Minimum viable implementation of the proposed three-stage idea:

Stage 1/2 proxy:
  Train the existing V188-style GRU on the full generated prefix pool with
  point0-calibrated loss.  This acts as the all-prefix grammar model.

Stage 3:
  Freeze those probabilities and train a fold-safe, test-like weighted
  multinomial adapter over base + Stage2 probabilities and context features.

Final exports:
  - point residual cap2/cap5 with action=V173, server=R121
  - action adapter wrappers from V197 low-churn surgery
  - joint low candidate: V197 low action + V202 point cap2

Raw neural argmax is diagnostic only and is never exported.  TTMATCH is not
read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_v188_point_intent_gru import capped_residual_labels, row_log_blend
from analysis_v195_distribution_matched_point_gru import MATCH_COLS, distribution_match_weights, prepare_data
from analysis_v196_point0_calibrated_gru import CalibrationSetting, run_scheme
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v202_three_stage_domain_adapt")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/train_v202_three_stage_domain_adapt.py")

CALIBRATION = CalibrationSetting("v202_stage2_p0t026_conf075", 0.26, 1.0, 0.75, 0.50, 0.05)
ALPHAS = [0.05, 0.075]
CAPS = [0.02, 0.05]
V197_ACTION_LOW = UPLOAD_DIR / "submission_v197_v166_r184_agree_attack_control__pv188_r186_w005_cap5__sr121.csv"


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


def adapter_sample_weights(rows: pd.DataFrame, test_rows: pd.DataFrame, cols: list[str], testlike_share: float = 0.7) -> np.ndarray:
    density = distribution_match_weights(rows, test_rows, cols, clip=(0.25, 4.0), smooth=3.0)
    w = float(testlike_share) * density + (1.0 - float(testlike_share)) * np.ones(len(rows), dtype=float)
    return w / max(float(w.mean()), 1e-12)


def build_adapter_features(
    rows: pd.DataFrame,
    pred_rows: pd.DataFrame,
    base_prob: np.ndarray,
    stage_prob: np.ndarray,
    base_pred_prob: np.ndarray,
    stage_pred_prob: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["prefix_len", "scoreTotal", "serverScoreDiff", "lag0_actionId", "lag0_pointId", "lag0_spinId", "lag0_strengthId"]
    cat_cols = ["audit_phase", "audit_lag0_depth", "audit_lag0_action_family"]
    train = pd.DataFrame(index=np.arange(len(rows)))
    pred = pd.DataFrame(index=np.arange(len(pred_rows)))
    for i in range(base_prob.shape[1]):
        train[f"base_p{i}"] = base_prob[:, i]
        train[f"stage_p{i}"] = stage_prob[:, i]
        train[f"log_ratio_p{i}"] = np.log(np.clip(stage_prob[:, i], 1e-8, 1.0)) - np.log(np.clip(base_prob[:, i], 1e-8, 1.0))
        pred[f"base_p{i}"] = base_pred_prob[:, i]
        pred[f"stage_p{i}"] = stage_pred_prob[:, i]
        pred[f"log_ratio_p{i}"] = np.log(np.clip(stage_pred_prob[:, i], 1e-8, 1.0)) - np.log(np.clip(base_pred_prob[:, i], 1e-8, 1.0))
    for c in cols:
        train_src = rows[c] if c in rows.columns else pd.Series([0] * len(rows))
        pred_src = pred_rows[c] if c in pred_rows.columns else pd.Series([0] * len(pred_rows))
        train[c] = pd.to_numeric(train_src, errors="coerce").fillna(0).to_numpy()
        pred[c] = pd.to_numeric(pred_src, errors="coerce").fillna(0).to_numpy()
    for c in cat_cols:
        train_src = rows[c] if c in rows.columns else pd.Series(["unknown"] * len(rows))
        pred_src = pred_rows[c] if c in pred_rows.columns else pd.Series(["unknown"] * len(pred_rows))
        train[c] = train_src.astype(str).to_numpy()
        pred[c] = pred_src.astype(str).to_numpy()
    both = pd.concat([train, pred], ignore_index=True)
    both = pd.get_dummies(both, columns=cat_cols, dummy_na=True)
    return both.iloc[: len(train)].reset_index(drop=True), both.iloc[len(train) :].reset_index(drop=True)


def fit_adapter_oof(data: dict, stage_oof: np.ndarray, stage_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = data["rows"]
    test_rows = data["test_rows"]
    y = data["y_oof"]
    oof = np.zeros((len(rows), 10), dtype=float)
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        x_train, x_valid = build_adapter_features(
            rows.iloc[train].reset_index(drop=True),
            rows.iloc[valid].reset_index(drop=True),
            data["base_prob_oof"][train],
            stage_oof[train],
            data["base_prob_oof"][valid],
            stage_oof[valid],
        )
        sw = adapter_sample_weights(rows.iloc[train].reset_index(drop=True), test_rows, MATCH_COLS)
        model = LogisticRegression(max_iter=300, C=0.7, class_weight="balanced", n_jobs=-1)
        model.fit(x_train, y[train], sample_weight=sw)
        prob = model.predict_proba(x_valid)
        tmp = np.zeros((len(x_valid), 10), dtype=float)
        for j, cls in enumerate(model.classes_):
            tmp[:, int(cls)] = prob[:, j]
        oof[valid] = normalize_rows_safe(tmp)
    x_full, x_test = build_adapter_features(rows, test_rows, data["base_prob_oof"], stage_oof, data["base_prob_test"], stage_test)
    sw = adapter_sample_weights(rows, test_rows, MATCH_COLS)
    model = LogisticRegression(max_iter=300, C=0.7, class_weight="balanced", n_jobs=-1)
    model.fit(x_full, y, sample_weight=sw)
    prob = model.predict_proba(x_test)
    test = np.zeros((len(x_test), 10), dtype=float)
    for j, cls in enumerate(model.classes_):
        test[:, int(cls)] = prob[:, j]
    return normalize_rows_safe(oof), normalize_rows_safe(test)


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    score = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_score = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "point_macro_f1": score,
        "delta_vs_base": score - base_score,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "point0_rate": float(np.mean(pred == 0)),
    }
    rec.update(meta)
    return rec


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray | None = None, action: np.ndarray | None = None) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "pointId", "serverGetPoint"]].copy()
    if point is not None:
        out["pointId"] = np.asarray(point, dtype=int)
    if action is not None:
        out["actionId"] = np.asarray(action, dtype=int)
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    stage_oof, stage_test, folds = run_scheme("v202_stage2_fullprefix", data, "full", CALIBRATION)
    adapter_oof, adapter_test = fit_adapter_oof(data, stage_oof, stage_test)
    y = data["y_oof"]
    base = data["base_pred_oof"]
    test_base = data["test_base_point"]
    records = []
    pred_store = {}
    for source_name, oof_prob, test_prob in [("stage2", stage_oof, stage_test), ("stage3_adapter", adapter_oof, adapter_test)]:
        raw = oof_prob.argmax(axis=1).astype(int)
        raw_test = test_prob.argmax(axis=1).astype(int)
        records.append(eval_candidate(f"v202_{source_name}_raw_diagnostic", y, raw, base, {"source": source_name, "test_raw_point0_rate": float(np.mean(raw_test == 0))}))
        for alpha in ALPHAS:
            blend = row_log_blend(data["base_prob_oof"], oof_prob, alpha)
            blend_test = row_log_blend(data["base_prob_test"], test_prob, alpha)
            for cap in CAPS:
                pred, _ = capped_residual_labels(base, blend, cap)
                test_pred, test_changed = capped_residual_labels(test_base, blend_test, cap)
                name = f"v202a_{source_name}_point_residual_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                rec = eval_candidate(name, y, pred, base, {"source": source_name, "alpha": alpha, "cap": cap})
                rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base))
                rec["test_changed_rows"] = int(np.sum(test_changed))
                records.append(rec)
                pred_store[name] = test_pred
    search = pd.DataFrame(records).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True])
    search.to_csv(OUTDIR / "v202_search.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTDIR / "v202_fold_metrics.csv", index=False)

    generated = []
    for cap in [0.02, 0.05]:
        part = search[(search["candidate"].str.contains("stage3_adapter_point")) & np.isclose(search["cap"].fillna(-1).astype(float), cap)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], point=pred_store[name])
        info.update(rec)
        info["submission"] = sub_name
        generated.append(info)
    if V197_ACTION_LOW.exists():
        v197 = pd.read_csv(V197_ACTION_LOW)
        action = data["base_sub"][["rally_uid"]].merge(v197[["rally_uid", "actionId"]], on="rally_uid", how="left", validate="one_to_one")["actionId"].astype(int).to_numpy()
        info = write_submission("submission_v202a_action_adapter_low__pv188cap5__sr121.csv", data["base_sub"], action=action)
        generated.append({**info, "candidate": "v202a_action_adapter_low"})
        if generated:
            cap2 = next((g for g in generated if "cap0p02" in g["submission"]), None)
            if cap2 is not None:
                point = pd.read_csv(cap2["upload_path"])["pointId"].astype(int).to_numpy()
                info = write_submission("submission_v202a_joint_low__sr121.csv", data["base_sub"], point=point, action=action)
                generated.append({**info, "candidate": "v202a_joint_low"})

    report = {
        "verdict": "GENERATED",
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": [
            "V202 minimum implementation uses full-prefix GRU as Stage2 and a fold-safe weighted logistic adapter as Stage3.",
            "Raw predictions are diagnostic only.",
            "Point submissions are residual/cap only.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v202_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v202_report.md").write_text(
        "# V202 Three-Stage Domain Adapt\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("train_v202_three_stage_domain_adapt.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
