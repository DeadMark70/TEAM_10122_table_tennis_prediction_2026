"""V247 action-point consistency auxiliary posterior."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import context_weights, evaluate_action, finalize_search, load_action_context, write_submission


OUTDIR = Path("v247_action_point_consistency_aux")
SRC_DEST = Path("src/analysis/analysis_v247_action_point_consistency_aux.py")


def point_group(point: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(point, dtype=int)
    out = np.full(len(p), "zero", dtype=object)
    out[np.isin(p, [1, 2, 3])] = "short"
    out[np.isin(p, [4, 5, 6])] = "half"
    out[np.isin(p, [7, 8, 9])] = "long"
    return out


def prep(rows: pd.DataFrame, point_values: np.ndarray) -> pd.DataFrame:
    out = rows.copy()
    out["point_group_target"] = point_group(point_values)
    for col in ["audit_phase", "audit_lag0_action_family", "audit_lag0_depth", "point_group_target"]:
        if col not in out:
            out[col] = "missing"
        out[col] = out[col].astype(str).fillna("missing")
    return out


def build_policy(train: pd.DataFrame, y: np.ndarray, cols: list[str], alpha: float = 3.0) -> tuple[dict[tuple, np.ndarray], np.ndarray]:
    counts = np.bincount(y.astype(int), minlength=19).astype(float)
    global_prob = normalize_probability_rows((counts + alpha)[None, :])[0]
    tmp = train.loc[:, cols].copy()
    tmp["_y"] = y.astype(int)
    table = {}
    for key, grp in tmp.groupby(cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        c = np.bincount(grp["_y"].astype(int).to_numpy(), minlength=19).astype(float)
        table[tuple(map(str, key))] = normalize_probability_rows((c + alpha * global_prob)[None, :])[0]
    return table, global_prob


def apply_policy(frame: pd.DataFrame, cols: list[str], table: dict[tuple, np.ndarray], global_prob: np.ndarray) -> np.ndarray:
    out = np.zeros((len(frame), 19), dtype=float)
    vals = frame.loc[:, cols].astype(str).to_numpy()
    for i, row in enumerate(vals):
        out[i] = table.get(tuple(row.tolist()), global_prob)
    return normalize_probability_rows(out)


def consistency_probs(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray, test_point: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows_p = prep(rows, rows["next_pointId"].astype(int).to_numpy())
    test_p = prep(test_rows, test_point)
    specs = [
        ("point_phase_family", ["point_group_target", "audit_phase", "audit_lag0_action_family"]),
        ("point_phase_depth", ["point_group_target", "audit_phase", "audit_lag0_depth"]),
    ]
    oof_parts = {name: np.zeros((len(rows), 19), dtype=float) for name, _ in specs}
    test_parts = {name: np.zeros((len(test_rows), 19), dtype=float) for name, _ in specs}
    metrics = []
    folds = sorted(rows["fold"].astype(int).unique())
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        for name, cols in specs:
            table, global_prob = build_policy(rows_p.loc[train], y[train], cols)
            oof_parts[name][valid] = apply_policy(rows_p.loc[valid], cols, table, global_prob)
            test_parts[name] += apply_policy(test_p, cols, table, global_prob) / len(folds)
            metrics.append({"fold": int(fold), "policy": name, "keys": int(len(table))})
    oof = 0.60 * oof_parts["point_phase_family"] + 0.40 * oof_parts["point_phase_depth"]
    test = 0.60 * test_parts["point_phase_family"] + 0.40 * test_parts["point_phase_depth"]
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
    test_point = ctx["point"]["pointId"].astype(int).to_numpy()
    cons_oof, cons_test, metrics = consistency_probs(rows, test_rows, y, test_point)
    variants = {
        "v247_consistency_raw": (cons_oof, cons_test),
        "v247_consistency_v173blend_w0p10": (blend_probabilities(ctx["v173_prob_oof"], cons_oof, 0.10), blend_probabilities(ctx["v173_prob_test"], cons_test, 0.10)),
        "v247_consistency_v173blend_w0p20": (blend_probabilities(ctx["v173_prob_oof"], cons_oof, 0.20), blend_probabilities(ctx["v173_prob_test"], cons_test, 0.20)),
        "v247_consistency_v173blend_w0p35": (blend_probabilities(ctx["v173_prob_oof"], cons_oof, 0.35), blend_probabilities(ctx["v173_prob_test"], cons_test, 0.35)),
    }
    records = [evaluate_action("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
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
    search.to_csv(OUTDIR / "v247_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v247_policy_metrics.csv", index=False)
    (OUTDIR / "v247_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v247_action_point_consistency_aux.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
