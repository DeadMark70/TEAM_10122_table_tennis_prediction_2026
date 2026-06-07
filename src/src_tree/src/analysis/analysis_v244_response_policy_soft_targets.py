"""V244 fold-safe response-policy soft target augmentation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import context_weights, evaluate_action, finalize_search, load_action_context, write_submission


OUTDIR = Path("v244_response_policy_soft_targets")
SRC_DEST = Path("src/analysis/analysis_v244_response_policy_soft_targets.py")
POLICY_SPECS = [
    ("phase_family_depth", ["audit_phase", "audit_lag0_action_family", "audit_lag0_depth"]),
    ("phase_lagaction_spin", ["audit_phase", "lag0_actionId", "lag0_spinId"]),
    ("prefix_family", ["prefix_bin", "audit_lag0_action_family"]),
]


def prefix_bin_series(rows: pd.DataFrame) -> pd.Series:
    prefix = pd.to_numeric(rows["prefix_len"], errors="coerce").fillna(0).astype(int)
    return prefix.map(lambda v: "1" if v <= 1 else "2" if v == 2 else "3" if v == 3 else "4_6" if v <= 6 else "7_plus").astype(str)


def prepare_policy_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["prefix_bin"] = prefix_bin_series(out)
    for col in ["audit_phase", "audit_lag0_action_family", "audit_lag0_depth", "lag0_actionId", "lag0_spinId"]:
        if col not in out:
            out[col] = "missing"
        out[col] = out[col].astype(str).fillna("missing")
    return out


def smoothed_policy(train: pd.DataFrame, y: np.ndarray, key_cols: list[str], alpha: float = 3.0) -> tuple[dict[tuple, np.ndarray], np.ndarray]:
    counts = np.bincount(y.astype(int), minlength=19).astype(float)
    global_prob = normalize_probability_rows((counts + alpha)[None, :])[0]
    table: dict[tuple, np.ndarray] = {}
    tmp = train.loc[:, key_cols].copy()
    tmp["_y"] = y.astype(int)
    for key, grp in tmp.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        c = np.bincount(grp["_y"].astype(int).to_numpy(), minlength=19).astype(float)
        table[tuple(map(str, key))] = normalize_probability_rows((c + alpha * global_prob)[None, :])[0]
    return table, global_prob


def apply_policy(frame: pd.DataFrame, key_cols: list[str], table: dict[tuple, np.ndarray], global_prob: np.ndarray) -> np.ndarray:
    out = np.zeros((len(frame), 19), dtype=float)
    vals = frame.loc[:, key_cols].astype(str).to_numpy()
    for i, row in enumerate(vals):
        out[i] = table.get(tuple(row.tolist()), global_prob)
    return normalize_probability_rows(out)


def fold_policy_probs(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows_p = prepare_policy_frame(rows)
    test_p = prepare_policy_frame(test_rows)
    oof_parts = {name: np.zeros((len(rows), 19), dtype=float) for name, _ in POLICY_SPECS}
    test_parts = {name: np.zeros((len(test_rows), 19), dtype=float) for name, _ in POLICY_SPECS}
    metrics = []
    folds = sorted(rows["fold"].astype(int).unique())
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        for name, cols in POLICY_SPECS:
            table, global_prob = smoothed_policy(rows_p.loc[train], y[train], cols, alpha=3.0)
            oof_parts[name][valid] = apply_policy(rows_p.loc[valid], cols, table, global_prob)
            test_parts[name] += apply_policy(test_p, cols, table, global_prob) / len(folds)
            metrics.append({"fold": int(fold), "policy": name, "keys": int(len(table))})
    weights = {"phase_family_depth": 0.45, "phase_lagaction_spin": 0.35, "prefix_family": 0.20}
    oof = sum(weights[name] * oof_parts[name] for name in weights)
    test = sum(weights[name] * test_parts[name] for name in weights)
    return normalize_probability_rows(oof), normalize_probability_rows(test), metrics


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    v173_oof = ctx["v173_oof"]
    v173_test = ctx["v173_test"]
    weights = context_weights(rows, test_rows)
    policy_oof, policy_test, metrics = fold_policy_probs(rows, test_rows, y)
    records = [evaluate_action("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    variants = {
        "v244_policy_raw": (policy_oof, policy_test),
        "v244_policy_v173blend_w0p10": (blend_probabilities(ctx["v173_prob_oof"], policy_oof, 0.10), blend_probabilities(ctx["v173_prob_test"], policy_test, 0.10)),
        "v244_policy_v173blend_w0p20": (blend_probabilities(ctx["v173_prob_oof"], policy_oof, 0.20), blend_probabilities(ctx["v173_prob_test"], policy_test, 0.20)),
        "v244_policy_v173blend_w0p35": (blend_probabilities(ctx["v173_prob_oof"], policy_oof, 0.35), blend_probabilities(ctx["v173_prob_test"], policy_test, 0.35)),
    }
    for name, (prob_oof, prob_test) in variants.items():
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = evaluate_action(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", prob_test)
        generated.append(write_submission(OUTDIR, f"submission_{name}__pv188cap5__sr121.csv", test_pred, ctx["point"], ctx["server"]))
    search, best_delta, verdict = finalize_search(records)
    search.to_csv(OUTDIR / "v244_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v244_policy_metrics.csv", index=False)
    (OUTDIR / "v244_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v244_response_policy_soft_targets.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
