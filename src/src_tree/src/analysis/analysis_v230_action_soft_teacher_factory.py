"""V230 V173-centered soft action teacher factory.

V230 is deliberately not a low-row selector.  It builds full 19-class action
probability teachers centered on the public-positive V173 anchor, then exports
argmax action submissions with V188 cap5 point and R121 server fixed.

No TTMATCH, old-server labels, or external exact action labels are read.
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
from analysis_r179_action_physics_hierarchy import action_family
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    V3Tuning,
    GrUTuning,
    TransformerTuning,
    distill_v173_soft_anchor,
    rebuild_r166_best_action,
)
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v230_action_soft_teacher_factory")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v230_action_soft_teacher_factory.py")

ACTION_FAMILY_TO_IDS = {
    "Zero": [0],
    "Attack": [1, 2, 3, 4, 5, 6, 7],
    "Control": [8, 9, 10, 11],
    "Defensive": [12, 13, 14],
    "Serve": [15, 16, 17, 18],
}
FAMILY_COLS = ["Zero", "Attack", "Control", "Defensive", "Serve"]


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    sums = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, sums, out=np.full_like(arr, 1.0 / arr.shape[1]), where=sums > 0)


def geometric_log_blend(anchor_prob: np.ndarray, teacher_prob: np.ndarray, weight: float, eps: float = 1e-8) -> np.ndarray:
    anchor = np.clip(normalize_rows_safe(anchor_prob), eps, 1.0)
    teacher = np.clip(normalize_rows_safe(teacher_prob), eps, 1.0)
    logp = (1.0 - float(weight)) * np.log(anchor) + float(weight) * np.log(teacher)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


def soften_probability(prob: np.ndarray, temperature: float = 2.5) -> np.ndarray:
    p = np.clip(normalize_rows_safe(prob), 1e-8, 1.0)
    return normalize_rows_safe(p ** (1.0 / float(temperature)))


def family_mass(prob: np.ndarray) -> np.ndarray:
    p = normalize_rows_safe(prob)
    out = np.zeros((len(p), len(FAMILY_COLS)), dtype=float)
    for j, fam in enumerate(FAMILY_COLS):
        out[:, j] = p[:, ACTION_FAMILY_TO_IDS[fam]].sum(axis=1)
    return normalize_rows_safe(out)


def action_family_name(action_id: int) -> str:
    return action_family(int(action_id))


def apply_family_calibration(prob: np.ndarray, family_priors: list[dict[str, float]] | np.ndarray, weight: float) -> np.ndarray:
    p = normalize_rows_safe(prob)
    if isinstance(family_priors, np.ndarray):
        fam_prior = normalize_rows_safe(family_priors)
    else:
        fam_prior = np.zeros((len(p), len(FAMILY_COLS)), dtype=float)
        for i, row in enumerate(family_priors):
            for j, fam in enumerate(FAMILY_COLS):
                fam_prior[i, j] = float(row.get(fam, 0.0))
        fam_prior = normalize_rows_safe(fam_prior)
    current = family_mass(p)
    scale = np.ones_like(p)
    for j, fam in enumerate(FAMILY_COLS):
        ids = ACTION_FAMILY_TO_IDS[fam]
        ratio = np.divide(fam_prior[:, j], current[:, j], out=np.ones(len(p)), where=current[:, j] > 0)
        scale[:, ids] *= ratio[:, None] ** float(weight)
    return normalize_rows_safe(p * scale)


def public_like_slice_score(y: np.ndarray, pred: np.ndarray, rows: pd.DataFrame) -> float:
    scores = []
    weights = []
    prefix = pd.to_numeric(rows["prefix_len"], errors="coerce").fillna(0)
    phase = rows["audit_phase"].astype(str) if "audit_phase" in rows.columns else pd.Series([""] * len(rows))
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "prefix_1": prefix.eq(1).to_numpy(),
        "prefix_2": prefix.eq(2).to_numpy(),
        "prefix_3": prefix.eq(3).to_numpy(),
        "receive": phase.eq("receive").to_numpy(),
        "third_ball": phase.eq("third_ball").to_numpy(),
        "rally": phase.eq("rally").to_numpy(),
    }
    for name, mask in masks.items():
        if mask.sum() < 20 and name != "all":
            continue
        w = 2.0 if name in {"prefix_1", "prefix_2", "prefix_3", "receive", "third_ball"} else 1.0
        scores.append(f1_score(y[mask], pred[mask], labels=ACTION_CLASSES, average="macro", zero_division=0))
        weights.append(w)
    return float(np.average(scores, weights=weights))


def no_serve_explosion(pred: np.ndarray, anchor: np.ndarray, tolerance: int = 2) -> bool:
    pred_count = int(np.isin(pred, ACTION_FAMILY_TO_IDS["Serve"]).sum())
    anchor_count = int(np.isin(anchor, ACTION_FAMILY_TO_IDS["Serve"]).sum())
    return pred_count <= anchor_count + int(tolerance)


def phase_weights(rows: pd.DataFrame, receive: float, third: float, rally: float, default: float = 0.05) -> np.ndarray:
    phase = rows["audit_phase"].astype(str) if "audit_phase" in rows.columns else pd.Series([""] * len(rows))
    w = np.full(len(rows), float(default), dtype=float)
    w[phase.eq("receive").to_numpy()] = float(receive)
    w[phase.eq("third_ball").to_numpy()] = float(third)
    w[phase.eq("rally").to_numpy()] = float(rally)
    return w


def rowwise_log_blend(anchor: np.ndarray, teacher: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(anchor, dtype=float)
    for value in sorted(set(np.round(weights, 6))):
        mask = np.isclose(weights, value)
        out[mask] = geometric_log_blend(anchor[mask], teacher[mask], float(value))
    return normalize_rows_safe(out)


def source_family_prior(v173_prob: np.ndarray, r166_prob: np.ndarray, v208_prob: np.ndarray) -> np.ndarray:
    return normalize_rows_safe(0.50 * family_mass(v173_prob) + 0.35 * family_mass(r166_prob) + 0.15 * family_mass(v208_prob))


def evaluate_action_candidate(name: str, prob: np.ndarray, y: np.ndarray, anchor: np.ndarray, rows: pd.DataFrame) -> dict:
    pred = normalize_rows_safe(prob).argmax(axis=1).astype(int)
    anchor_score = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - anchor_score),
        "public_like_action_macro_f1": public_like_slice_score(y, pred, rows),
        "public_like_delta_vs_v173": public_like_slice_score(y, pred, rows) - public_like_slice_score(y, anchor, rows),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
        "serve_count": int(np.isin(pred, ACTION_FAMILY_TO_IDS["Serve"]).sum()),
        "action_distribution": json.dumps(pd.Series(pred).value_counts().sort_index().to_dict(), sort_keys=True),
    }


def class_f1_table(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray, candidate: str) -> pd.DataFrame:
    records = []
    for label in ACTION_CLASSES:
        records.append(
            {
                "candidate": candidate,
                "action": int(label),
                "support": int((y == int(label)).sum()),
                "anchor_f1": float(f1_score(y, anchor, labels=[label], average="macro", zero_division=0)),
                "candidate_f1": float(f1_score(y, pred, labels=[label], average="macro", zero_division=0)),
            }
        )
    df = pd.DataFrame(records)
    df["delta"] = df["candidate_f1"] - df["anchor_f1"]
    return df


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


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    rows = data["rows"].copy()
    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    _r166_oof, _r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    v208_oof_path = Path("v209_action_selector_reranker/v209_v208_action_point_aux_oof.npy")
    v208_test_path = Path("v209_action_selector_reranker/v209_v208_action_point_aux_test.npy")
    v208_prob_oof = np.load(v208_oof_path) if v208_oof_path.exists() else v173_prob_oof
    v208_prob_test = np.load(v208_test_path) if v208_test_path.exists() else v173_prob_test

    family_prior_oof = source_family_prior(v173_prob_oof, r166_prob_oof, v208_prob_oof)
    family_prior_test = source_family_prior(v173_prob_test, r166_prob_test, v208_prob_test)
    soft_v173_oof = soften_probability(v173_prob_oof, 2.5)
    soft_v173_test = soften_probability(v173_prob_test, 2.5)
    very_soft_v173_oof = soften_probability(v173_prob_oof, 4.0)
    very_soft_v173_test = soften_probability(v173_prob_test, 4.0)
    variants = {
        "v230_v173_r166_logblend_w0p03": (
            geometric_log_blend(soft_v173_oof, r166_prob_oof, 0.03),
            geometric_log_blend(soft_v173_test, r166_prob_test, 0.03),
        ),
        "v230_v173_r166_logblend_w0p05": (
            geometric_log_blend(soft_v173_oof, r166_prob_oof, 0.05),
            geometric_log_blend(soft_v173_test, r166_prob_test, 0.05),
        ),
        "v230_v173_r166_logblend_w0p075": (
            geometric_log_blend(soft_v173_oof, r166_prob_oof, 0.075),
            geometric_log_blend(soft_v173_test, r166_prob_test, 0.075),
        ),
        "v230_phase_teacher_receive": (
            rowwise_log_blend(soft_v173_oof, r166_prob_oof, phase_weights(rows, 0.18, 0.06, 0.04)),
            rowwise_log_blend(soft_v173_test, r166_prob_test, phase_weights(state["test_rows"], 0.18, 0.06, 0.04)),
        ),
        "v230_phase_teacher_third": (
            rowwise_log_blend(soft_v173_oof, r166_prob_oof, phase_weights(rows, 0.04, 0.18, 0.06)),
            rowwise_log_blend(soft_v173_test, r166_prob_test, phase_weights(state["test_rows"], 0.04, 0.18, 0.06)),
        ),
        "v230_phase_teacher_rally": (
            rowwise_log_blend(soft_v173_oof, r166_prob_oof, phase_weights(rows, 0.04, 0.06, 0.22)),
            rowwise_log_blend(soft_v173_test, r166_prob_test, phase_weights(state["test_rows"], 0.04, 0.06, 0.22)),
        ),
        "v230_family_calibrated_teacher": (
            apply_family_calibration(geometric_log_blend(soft_v173_oof, r166_prob_oof, 0.10), family_prior_oof, 0.65),
            apply_family_calibration(geometric_log_blend(soft_v173_test, r166_prob_test, 0.10), family_prior_test, 0.65),
        ),
        "v230_aggressive_teacher_mix": (
            apply_family_calibration(normalize_rows_safe(0.35 * very_soft_v173_oof + 0.45 * r166_prob_oof + 0.20 * normalize_rows_safe(v208_prob_oof)), family_prior_oof, 0.75),
            apply_family_calibration(normalize_rows_safe(0.35 * very_soft_v173_test + 0.45 * r166_prob_test + 0.20 * normalize_rows_safe(v208_prob_test)), family_prior_test, 0.75),
        ),
    }

    records = [
        {
            "candidate": "v173_anchor",
            "action_macro_f1": float(f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0)),
            "delta_vs_v173_anchor": 0.0,
            "public_like_action_macro_f1": public_like_slice_score(y, v173_oof, rows),
            "public_like_delta_vs_v173": 0.0,
            "action_churn_vs_v173_anchor": 0.0,
            "changed_rows": 0,
            "serve_count": int(np.isin(v173_oof, ACTION_FAMILY_TO_IDS["Serve"]).sum()),
            "action_distribution": json.dumps(pd.Series(v173_oof).value_counts().sort_index().to_dict(), sort_keys=True),
        }
    ]
    generated = []
    class_tables = []
    export_map = {
        "v230_v173_r166_logblend_w0p05": "submission_v230_logblend_w0p05__pv188cap5__sr121.csv",
        "v230_v173_r166_logblend_w0p075": "submission_v230_logblend_w0p075__pv188cap5__sr121.csv",
        "v230_phase_teacher_receive": "submission_v230_phase_receive__pv188cap5__sr121.csv",
        "v230_phase_teacher_third": "submission_v230_phase_third__pv188cap5__sr121.csv",
        "v230_phase_teacher_rally": "submission_v230_phase_rally__pv188cap5__sr121.csv",
        "v230_family_calibrated_teacher": "submission_v230_family_calibrated__pv188cap5__sr121.csv",
        "v230_aggressive_teacher_mix": "submission_v230_aggressive_mix__pv188cap5__sr121.csv",
    }
    for name, (prob_oof, prob_test) in variants.items():
        pred_oof = normalize_rows_safe(prob_oof).argmax(axis=1).astype(int)
        pred_test = normalize_rows_safe(prob_test).argmax(axis=1).astype(int)
        rec = evaluate_action_candidate(name, prob_oof, y, v173_oof, rows)
        rec["test_churn_vs_v173"] = float(np.mean(pred_test != v173_test))
        rec["test_changed_rows"] = int(np.sum(pred_test != v173_test))
        rec["test_serve_count"] = int(np.isin(pred_test, ACTION_FAMILY_TO_IDS["Serve"]).sum())
        records.append(rec)
        class_tables.append(class_f1_table(y, v173_oof, pred_oof, name))
        if name in export_map:
            info = write_submission(export_map[name], pred_test, point, server)
            info.update(rec)
            generated.append(info)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", normalize_rows_safe(prob_oof))
        np.save(OUTDIR / f"{name}_test_action_prob.npy", normalize_rows_safe(prob_test))

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "public_like_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v230_action_search.csv", index=False)
    pd.DataFrame(distill_metrics).to_csv(OUTDIR / "v230_v173_distill_metrics.csv", index=False)
    if class_tables:
        pd.concat(class_tables, ignore_index=True).to_csv(OUTDIR / "v230_class_f1_delta.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": [
            "V230 centers all teachers on V173 and never direct-replaces with V166.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "No TTMATCH and no old-server labels are read.",
        ],
    }
    (OUTDIR / "v230_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v230_report.md").write_text(
        "# V230 Action Soft Teacher Factory\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v230_action_soft_teacher_factory.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
