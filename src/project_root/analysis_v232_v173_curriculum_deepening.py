"""V232 V173 deepening / multi-source curriculum 2.0.

This is a no-old, no-TTMATCH curriculum wrapper around the existing V173/R166
teachers.  External tables are used only as coarse action-family priors, never
as exact AICUP actionId labels.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r179_action_physics_hierarchy import action_family
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor, rebuild_r166_best_action
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v230_action_soft_teacher_factory import (
    ACTION_FAMILY_TO_IDS,
    apply_family_calibration,
    geometric_log_blend,
    normalize_rows_safe,
    public_like_slice_score,
    soften_probability,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v232_v173_curriculum_deepening")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v232_v173_curriculum_deepening.py")
OPEN_PRIOR = Path("v173_external_curriculum_pretrain/v173_opentt_prior_table.csv")
COACHAI_PRIOR = Path("v173_external_curriculum_pretrain/v173_coachai_transition_stats.csv")

FAMILY_ORDER = ["Zero", "Attack", "Control", "Defensive", "Serve"]
EXTERNAL_TO_INTERNAL = {
    "attack": "Attack",
    "control": "Control",
    "defensive": "Defensive",
    "serve": "Serve",
    "unknown": "Zero",
}


def family_from_lag_action(action_id: int) -> str:
    return str(action_family(int(action_id))).lower()


def row_contexts(rows: pd.DataFrame) -> pd.DataFrame:
    phase = rows["audit_phase"].astype(str) if "audit_phase" in rows.columns else pd.Series(["rally"] * len(rows))
    lag = rows["lag0_actionId"].astype(int) if "lag0_actionId" in rows.columns else pd.Series([1] * len(rows))
    return pd.DataFrame({"phase": phase.to_numpy(), "current_family": [family_from_lag_action(a) for a in lag]})


def build_family_prior_matrix(prior_table: pd.DataFrame, contexts: pd.DataFrame) -> np.ndarray:
    table = {}
    for row in prior_table.itertuples(index=False):
        phase = str(getattr(row, "phase"))
        current = str(getattr(row, "current_family"))
        values = {
            "Zero": float(getattr(row, "next_family_unknown", 0.0)),
            "Attack": float(getattr(row, "next_family_attack", 0.0)),
            "Control": float(getattr(row, "next_family_control", 0.0)),
            "Defensive": float(getattr(row, "next_family_defensive", 0.0)),
            "Serve": float(getattr(row, "next_family_serve", 0.0)),
        }
        table[(phase, current)] = values
    default = {"Zero": 0.02, "Attack": 0.50, "Control": 0.25, "Defensive": 0.22, "Serve": 0.01}
    priors = []
    for row in contexts.itertuples(index=False):
        priors.append(table.get((str(row.phase), str(row.current_family)), default))
    out = np.zeros((len(priors), len(FAMILY_ORDER)), dtype=float)
    for i, rec in enumerate(priors):
        for j, fam in enumerate(FAMILY_ORDER):
            out[i, j] = float(rec.get(fam, 0.0))
    return normalize_rows_safe(out)


def coachai_family_prior(path: Path, contexts: pd.DataFrame) -> np.ndarray:
    if not path.exists():
        return np.full((len(contexts), len(FAMILY_ORDER)), 1.0 / len(FAMILY_ORDER))
    df = pd.read_csv(path)
    pivot = df.pivot_table(index="phase", columns="next_family", values="rows", aggfunc="sum", fill_value=0)
    out = np.zeros((len(contexts), len(FAMILY_ORDER)), dtype=float)
    for i, phase in enumerate(contexts["phase"].astype(str)):
        if phase in pivot.index:
            for ext, internal in EXTERNAL_TO_INTERNAL.items():
                out[i, FAMILY_ORDER.index(internal)] += float(pivot.loc[phase].get(ext, 0.0))
        else:
            out[i] = 1.0
    return normalize_rows_safe(out + 1.0)


def apply_curriculum_family_prior(prob: np.ndarray, family_prior: np.ndarray, weight: float) -> np.ndarray:
    return apply_family_calibration(prob, family_prior, weight)


def evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, rows: pd.DataFrame) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "public_like_action_macro_f1": public_like_slice_score(y, pred, rows),
        "public_like_delta_vs_v173": public_like_slice_score(y, pred, rows) - public_like_slice_score(y, anchor, rows),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
        "action_distribution": json.dumps(pd.Series(pred).value_counts().sort_index().to_dict(), sort_keys=True),
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


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    _r166_oof, _r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    soft_v173_oof = soften_probability(v173_prob_oof, 2.5)
    soft_v173_test = soften_probability(v173_prob_test, 2.5)
    very_soft_v173_oof = soften_probability(v173_prob_oof, 4.0)
    very_soft_v173_test = soften_probability(v173_prob_test, 4.0)
    oof_context = row_contexts(data["rows"])
    test_context = row_contexts(state["test_rows"])
    opentt = pd.read_csv(OPEN_PRIOR) if OPEN_PRIOR.exists() else pd.DataFrame()
    opentt_prior_oof = build_family_prior_matrix(opentt, oof_context) if not opentt.empty else np.full((len(oof_context), 5), 0.2)
    opentt_prior_test = build_family_prior_matrix(opentt, test_context) if not opentt.empty else np.full((len(test_context), 5), 0.2)
    coach_prior_oof = coachai_family_prior(COACHAI_PRIOR, oof_context)
    coach_prior_test = coachai_family_prior(COACHAI_PRIOR, test_context)
    external_prior_oof = normalize_rows_safe(0.65 * opentt_prior_oof + 0.35 * coach_prior_oof)
    external_prior_test = normalize_rows_safe(0.65 * opentt_prior_test + 0.35 * coach_prior_test)
    variants = {
        "v232_aicup_only_curriculum": (
            apply_curriculum_family_prior(soft_v173_oof, external_prior_oof, 0.30),
            apply_curriculum_family_prior(soft_v173_test, external_prior_test, 0.30),
        ),
        "v232_external_family_pretrain": (
            apply_curriculum_family_prior(soft_v173_oof, external_prior_oof, 0.55),
            apply_curriculum_family_prior(soft_v173_test, external_prior_test, 0.55),
        ),
        "v232_external_plus_v173_kd": (
            apply_curriculum_family_prior(geometric_log_blend(soft_v173_oof, r166_prob_oof, 0.05), external_prior_oof, 0.45),
            apply_curriculum_family_prior(geometric_log_blend(soft_v173_test, r166_prob_test, 0.05), external_prior_test, 0.45),
        ),
        "v232_external_plus_r166_kd": (
            apply_curriculum_family_prior(geometric_log_blend(soft_v173_oof, r166_prob_oof, 0.12), external_prior_oof, 0.55),
            apply_curriculum_family_prior(geometric_log_blend(soft_v173_test, r166_prob_test, 0.12), external_prior_test, 0.55),
        ),
        "v232_full_aggressive_teacher": (
            apply_curriculum_family_prior(geometric_log_blend(very_soft_v173_oof, r166_prob_oof, 0.18), external_prior_oof, 0.70),
            apply_curriculum_family_prior(geometric_log_blend(very_soft_v173_test, r166_prob_test, 0.18), external_prior_test, 0.70),
        ),
    }
    records = [
        {
            "candidate": "v173_anchor",
            "action_macro_f1": float(f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0)),
            "delta_vs_v173_anchor": 0.0,
            "public_like_action_macro_f1": public_like_slice_score(y, v173_oof, data["rows"]),
            "public_like_delta_vs_v173": 0.0,
            "action_churn_vs_v173_anchor": 0.0,
            "changed_rows": 0,
        }
    ]
    generated = []
    export = {
        "v232_aicup_only_curriculum": "submission_v232_aicup_only__pv188cap5__sr121.csv",
        "v232_external_family_pretrain": "submission_v232_external_family__pv188cap5__sr121.csv",
        "v232_external_plus_v173_kd": "submission_v232_v173kd__pv188cap5__sr121.csv",
        "v232_external_plus_r166_kd": "submission_v232_r166kd__pv188cap5__sr121.csv",
        "v232_full_aggressive_teacher": "submission_v232_full_aggressive__pv188cap5__sr121.csv",
    }
    for name, (prob_oof, prob_test) in variants.items():
        pred = normalize_rows_safe(prob_oof).argmax(axis=1).astype(int)
        test_pred = normalize_rows_safe(prob_test).argmax(axis=1).astype(int)
        rec = evaluate(name, y, pred, v173_oof, data["rows"])
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        generated.append(write_submission(export[name], test_pred, point, server))
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", normalize_rows_safe(prob_oof))
        np.save(OUTDIR / f"{name}_test_action_prob.npy", normalize_rows_safe(prob_test))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "public_like_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v232_action_search.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": [
            "V232 uses external tables only as coarse family priors.",
            "No external exact AICUP actionId labels, no TTMATCH, and no old-server labels are read.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
        ],
    }
    (OUTDIR / "v232_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v232_report.md").write_text(
        "# V232 V173 Curriculum Deepening\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v232_v173_curriculum_deepening.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
