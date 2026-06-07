"""V238 faithful V173 reconstruction / ablation.

This experiment does not claim to fully reproduce V173 internals.  It builds a
controlled ablation around known available sources: V173 soft anchor, R166
teacher, V230/V232 curriculum probabilities, and phase-weighted variants.  The
goal is to identify whether any source can move V173 positively under the V233
public-like gate.

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
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor, rebuild_r166_best_action
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v238_v173_reconstruction_ablation")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v238_v173_reconstruction_ablation.py")


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


def evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
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


def write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
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


def optional_prob(name: str, n_oof: int, n_test: int, fallback_oof: np.ndarray, fallback_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        (Path("v230_action_soft_teacher_factory") / f"{name}_oof_action_prob.npy", Path("v230_action_soft_teacher_factory") / f"{name}_test_action_prob.npy"),
        (Path("v232_v173_curriculum_deepening") / f"{name}_oof_action_prob.npy", Path("v232_v173_curriculum_deepening") / f"{name}_test_action_prob.npy"),
    ]
    for oof, test in candidates:
        if oof.exists() and test.exists():
            a = np.load(oof)
            b = np.load(test)
            if len(a) == n_oof and len(b) == n_test:
                return normalize_probability_rows(a), normalize_probability_rows(b)
    return fallback_oof, fallback_test


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
    phase_oof, phase_test = optional_prob("v230_phase_teacher_receive", len(rows), len(test_rows), v173_prob_oof, v173_prob_test)
    ext_oof, ext_test = optional_prob("v232_external_plus_v173_kd", len(rows), len(test_rows), v173_prob_oof, v173_prob_test)
    weights = context_weights(rows, test_rows)
    variants = {
        "v238_v173_soft_anchor": (v173_prob_oof, v173_prob_test),
        "v238_r166_only": (r166_prob_oof, r166_prob_test),
        "v238_v173_r166_w0p10": (blend_probabilities(v173_prob_oof, r166_prob_oof, 0.10), blend_probabilities(v173_prob_test, r166_prob_test, 0.10)),
        "v238_v173_r166_w0p20": (blend_probabilities(v173_prob_oof, r166_prob_oof, 0.20), blend_probabilities(v173_prob_test, r166_prob_test, 0.20)),
        "v238_v173_phase_external_proxy": (blend_probabilities(phase_oof, ext_oof, 0.25), blend_probabilities(phase_test, ext_test, 0.25)),
        "v238_v173_phase_external_r166": (
            blend_probabilities(blend_probabilities(phase_oof, ext_oof, 0.20), r166_prob_oof, 0.12),
            blend_probabilities(blend_probabilities(phase_test, ext_test, 0.20), r166_prob_test, 0.12),
        ),
    }
    records = [evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    for name, (prob_oof, prob_test) in variants.items():
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = evaluate(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", normalize_probability_rows(prob_oof))
        np.save(OUTDIR / f"{name}_test_action_prob.npy", normalize_probability_rows(prob_test))
        generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v238_action_search.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}
    (OUTDIR / "v238_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v238_report.md").write_text(f"# V238 V173 Reconstruction Ablation\n\n- Verdict: `{verdict}`\n- Best delta vs V173: `{best_delta:.6f}`\n", encoding="utf-8")
    shutil.copy2("analysis_v238_v173_reconstruction_ablation.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
