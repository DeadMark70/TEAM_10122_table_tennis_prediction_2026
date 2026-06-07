"""V236 distributional / macro-F1 action calibration.

V236 tests long-tail posterior calibration around V173 and available V234/V235
probability sources.  It is not a row override script: it adjusts full 19-class
probabilities with class-prior logit adjustment and classwise temperatures.

Point is fixed at V188 cap5 and server is fixed at R121.  No TTMATCH and no
old-server labels are read.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

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


OUTDIR = Path("v236_distributional_action_calibrator")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v236_distributional_action_calibrator.py")
WEAK_CLASSES = np.array([0, 3, 4, 5, 7, 8, 9, 12, 14], dtype=int)


def class_prior_logit_adjust(prob: np.ndarray, class_counts: np.ndarray, tau: float = 0.5) -> np.ndarray:
    p = np.clip(normalize_rows_safe(prob), 1e-8, 1.0)
    counts = np.asarray(class_counts, dtype=float) + 1.0
    prior = counts / counts.sum()
    logits = np.log(p) - float(tau) * np.log(np.clip(prior, 1e-8, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logits))


def classwise_temperature(prob: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    p = np.clip(normalize_rows_safe(prob), 1e-8, 1.0)
    t = np.asarray(temperatures, dtype=float)
    logits = np.log(p) / np.clip(t, 1e-3, 100.0)
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logits))


def weak_tail_temperature(n_classes: int = 19, weak_temp: float = 0.85, head_temp: float = 1.10) -> np.ndarray:
    t = np.full(n_classes, float(head_temp), dtype=float)
    t[WEAK_CLASSES] = float(weak_temp)
    return t


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
    weak_score = f1_score(y, pred, labels=WEAK_CLASSES.tolist(), average="macro", zero_division=0)
    weak_base = f1_score(y, anchor, labels=WEAK_CLASSES.tolist(), average="macro", zero_division=0)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "iw_action_macro_f1": float(iw),
        "iw_delta_vs_v173": float(iw - base_iw),
        "weak_macro_f1": float(weak_score),
        "weak_delta_vs_v173": float(weak_score - weak_base),
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


def _available_source_prob(v173_oof: np.ndarray, v173_test: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    sources = [("v173", v173_oof, v173_test)]
    for name in ["v234_phase_v173kd_w0p35", "v235_response_w0p20", "v235_response_aggressive_w0p35"]:
        if name.startswith("v234"):
            base = Path("v234_v173_phase_expert_reconstruction")
        else:
            base = Path("v235_player_conditional_response_teacher")
        oof = base / f"{name}_oof_action_prob.npy"
        test = base / f"{name}_test_action_prob.npy"
        if oof.exists() and test.exists():
            sources.append((name, np.load(oof), np.load(test)))
    return sources


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
    weights = _context_weights(rows, test_rows)
    class_counts = np.bincount(y, minlength=19).astype(float)
    records = [_evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    for source_name, prob_oof, prob_test in _available_source_prob(v173_prob_oof, v173_prob_test):
        variants = {
            f"v236_{source_name}_logadj_tau0p25": (
                class_prior_logit_adjust(prob_oof, class_counts, 0.25),
                class_prior_logit_adjust(prob_test, class_counts, 0.25),
            ),
            f"v236_{source_name}_logadj_tau0p50": (
                class_prior_logit_adjust(prob_oof, class_counts, 0.50),
                class_prior_logit_adjust(prob_test, class_counts, 0.50),
            ),
            f"v236_{source_name}_weaktemp": (
                classwise_temperature(prob_oof, weak_tail_temperature()),
                classwise_temperature(prob_test, weak_tail_temperature()),
            ),
        }
        if source_name != "v173":
            variants[f"v236_{source_name}_blend_v173_w0p30_tau0p25"] = (
                class_prior_logit_adjust(geometric_log_blend(v173_prob_oof, prob_oof, 0.30), class_counts, 0.25),
                class_prior_logit_adjust(geometric_log_blend(v173_prob_test, prob_test, 0.30), class_counts, 0.25),
            )
        for name, (cal_oof, cal_test) in variants.items():
            pred = cal_oof.argmax(axis=1).astype(int)
            test_pred = cal_test.argmax(axis=1).astype(int)
            rec = _evaluate(name, y, pred, v173_oof, weights)
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
            records.append(rec)
            np.save(OUTDIR / f"{name}_oof_action_prob.npy", cal_oof)
            np.save(OUTDIR / f"{name}_test_action_prob.npy", cal_test)
            generated.append(_write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"], ascending=[False, False, False])
    search.to_csv(OUTDIR / "v236_action_search.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(12).to_dict(orient="records"), "generated": generated}
    (OUTDIR / "v236_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v236_report.md").write_text(f"# V236 Distributional Action Calibrator\n\n- Verdict: `{verdict}`\n- Best delta vs V173: `{best_delta:.6f}`\n", encoding="utf-8")
    shutil.copy2("analysis_v236_distributional_action_calibrator.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
