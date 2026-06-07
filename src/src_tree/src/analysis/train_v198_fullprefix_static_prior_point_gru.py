"""V198 full-prefix static-prior point GRU.

This is the heavier point experiment after V195/V196.  It trains only on the
full generated prefix pool, keeps V188-style architecture, uses point0-calibrated
loss, and reports ordinary plus test-like slice OOF before exporting residual
candidates.

Raw argmax is diagnostic only and is never written as a submission.  TTMATCH is
not read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_v188_point_intent_gru import capped_residual_labels, row_log_blend
from analysis_v195_distribution_matched_point_gru import distribution, prepare_data
from analysis_v196_point0_calibrated_gru import CalibrationSetting, run_scheme
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v198_fullprefix_static_prior_point_gru")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/train_v198_fullprefix_static_prior_point_gru.py")

ALPHAS = [0.05, 0.075]
CAPS = [0.02, 0.03, 0.05]
CALIBRATION = CalibrationSetting("v198_p0t026_conf075", 0.26, 1.0, 0.75, 0.50, 0.05)


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


def slice_macro_f1(rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    phase = rows.get("audit_phase", pd.Series(["unknown"] * len(rows))).astype(str)
    depth = rows.get("audit_lag0_depth", pd.Series(["unknown"] * len(rows))).astype(str)
    family = rows.get("audit_lag0_action_family", pd.Series(["unknown"] * len(rows))).astype(str)
    prefix = pd.to_numeric(rows.get("prefix_len", pd.Series([0] * len(rows))), errors="coerce").fillna(0)
    masks = {
        "phase_receive": phase.eq("receive").to_numpy(),
        "phase_third_ball": phase.eq("third_ball").to_numpy(),
        "phase_fourth_ball": phase.eq("fourth_ball").to_numpy(),
        "phase_rally": phase.eq("rally").to_numpy(),
        "lag0_long": depth.eq("long").to_numpy(),
        "lag0_attack": family.eq("Attack").to_numpy(),
        "prefix_ge3": prefix.ge(3).to_numpy(),
        "testlike_union": (phase.eq("rally").to_numpy() | depth.eq("long").to_numpy() | prefix.ge(3).to_numpy()),
    }
    out = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(f1_score(y[mask], pred[mask], labels=POINT_CLASSES, average="macro", zero_division=0))
    return out


def eval_candidate(name: str, rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    score = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_score = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "point_macro_f1": score,
        "delta_vs_base": score - base_score,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "pred_point0_rate": float(np.mean(pred == 0)),
    }
    rec.update({f"slice_{k}": v for k, v in slice_macro_f1(rows, y, pred).items()})
    rec.update(meta)
    return rec


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
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
    oof_prob, test_prob, folds = run_scheme("v198_fullprefix_static_prior", data, "full", CALIBRATION)
    y = data["y_oof"]
    base = data["base_pred_oof"]
    test_base = data["test_base_point"]
    raw_oof = oof_prob.argmax(axis=1).astype(int)
    raw_test = test_prob.argmax(axis=1).astype(int)

    records = [
        eval_candidate(
            "v198_raw_argmax_diagnostic",
            data["rows"],
            y,
            raw_oof,
            base,
            {
                "alpha": 1.0,
                "cap": 1.0,
                "test_raw_point0_rate": float(np.mean(raw_test == 0)),
                "test_raw_distribution": json.dumps(distribution(raw_test), sort_keys=True),
            },
        )
    ]
    pred_store = {}
    for alpha in ALPHAS:
        blend = row_log_blend(data["base_prob_oof"], normalize_rows_safe(oof_prob), alpha)
        blend_test = row_log_blend(data["base_prob_test"], normalize_rows_safe(test_prob), alpha)
        for cap in CAPS:
            pred, _ = capped_residual_labels(base, blend, cap)
            test_pred, test_changed = capped_residual_labels(test_base, blend_test, cap)
            name = f"v198_fullprefix_static_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
            rec = eval_candidate(name, data["rows"], y, pred, base, {"alpha": alpha, "cap": cap})
            rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base))
            rec["test_changed_rows"] = int(np.sum(test_changed))
            rec["test_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
            records.append(rec)
            pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True])
    search.to_csv(OUTDIR / "v198_search.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTDIR / "v198_fold_metrics.csv", index=False)

    generated = []
    positive = search[(search["candidate"].str.startswith("v198_fullprefix")) & search["delta_vs_base"].gt(0)]
    for cap in [0.02, 0.05]:
        part = positive[np.isclose(positive["cap"].astype(float), cap)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        info.update(rec)
        info["submission"] = sub_name
        generated.append(info)

    report = {
        "verdict": "GENERATED" if generated else "NO_POSITIVE_RESIDUAL",
        "raw_test_point0_rate": float(np.mean(raw_test == 0)),
        "raw_test_distribution": distribution(raw_test),
        "generated": generated,
        "notes": [
            "V198 trains on the full generated prefix pool with static priors from the V195 data builder.",
            "Raw argmax is diagnostic only and is not exported.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v198_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v198_report.md").write_text(
        "# V198 Full-Prefix Static-Prior Point GRU\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Raw test point0 rate: `{report['raw_test_point0_rate']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("train_v198_fullprefix_static_prior_point_gru.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
