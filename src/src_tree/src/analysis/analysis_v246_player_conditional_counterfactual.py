"""V246 player-conditional response backoff / counterfactual soft teacher."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import context_weights, evaluate_action, finalize_search, load_action_context, write_submission


OUTDIR = Path("v246_player_conditional_counterfactual")
SRC_DEST = Path("src/analysis/analysis_v246_player_conditional_counterfactual.py")


def prep(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for col in ["next_hitter_id", "next_receiver_id", "audit_phase", "audit_lag0_action_family", "audit_lag0_depth"]:
        if col not in out:
            out[col] = -1 if col.endswith("_id") else "missing"
        out[col] = out[col].fillna(-1 if col.endswith("_id") else "missing").astype(str)
    return out


def build_table(train: pd.DataFrame, y: np.ndarray, key_cols: list[str], alpha: float = 4.0) -> tuple[dict[tuple, tuple[np.ndarray, float]], np.ndarray]:
    counts = np.bincount(y.astype(int), minlength=19).astype(float)
    global_prob = normalize_probability_rows((counts + alpha)[None, :])[0]
    table: dict[tuple, tuple[np.ndarray, float]] = {}
    tmp = train.loc[:, key_cols].copy()
    tmp["_y"] = y.astype(int)
    for key, grp in tmp.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        n = len(grp)
        rel = n / (n + 25.0)
        c = np.bincount(grp["_y"].astype(int).to_numpy(), minlength=19).astype(float)
        prob = normalize_probability_rows((c + alpha * global_prob)[None, :])[0]
        table[tuple(map(str, key))] = (prob, float(rel))
    return table, global_prob


def apply_table(frame: pd.DataFrame, key_cols: list[str], table: dict[tuple, tuple[np.ndarray, float]], global_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = np.zeros((len(frame), 19), dtype=float)
    rel = np.zeros(len(frame), dtype=float)
    vals = frame.loc[:, key_cols].astype(str).to_numpy()
    for i, row in enumerate(vals):
        p, r = table.get(tuple(row.tolist()), (global_prob, 0.0))
        prob[i] = p
        rel[i] = r
    return normalize_probability_rows(prob), rel


def player_policy_probs(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows_p = prep(rows)
    test_p = prep(test_rows)
    specs = [
        ("hitter_phase_family", ["next_hitter_id", "audit_phase", "audit_lag0_action_family"]),
        ("receiver_phase_depth", ["next_receiver_id", "audit_phase", "audit_lag0_depth"]),
        ("hitter_receiver_phase", ["next_hitter_id", "next_receiver_id", "audit_phase"]),
    ]
    oof = np.zeros((len(rows), 19), dtype=float)
    test = np.zeros((len(test_rows), 19), dtype=float)
    oof_rel = np.zeros(len(rows), dtype=float)
    metrics = []
    folds = sorted(rows["fold"].astype(int).unique())
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        fold_oof = np.zeros((valid.sum(), 19), dtype=float)
        fold_rel = np.zeros(valid.sum(), dtype=float)
        fold_test = np.zeros((len(test_rows), 19), dtype=float)
        for name, cols in specs:
            table, global_prob = build_table(rows_p.loc[train], y[train], cols)
            p_valid, r_valid = apply_table(rows_p.loc[valid], cols, table, global_prob)
            p_test, r_test = apply_table(test_p, cols, table, global_prob)
            fold_oof += p_valid / len(specs)
            fold_rel += r_valid / len(specs)
            fold_test += p_test / len(specs)
            metrics.append({"fold": int(fold), "policy": name, "keys": int(len(table)), "test_reliability": float(r_test.mean())})
        oof[valid] = normalize_probability_rows(fold_oof)
        oof_rel[valid] = fold_rel
        test += normalize_probability_rows(fold_test) / len(folds)
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
    player_oof, player_test, metrics = player_policy_probs(rows, test_rows, y)
    variants = {
        "v246_player_policy_raw": (player_oof, player_test),
        "v246_player_policy_v173blend_w0p10": (blend_probabilities(ctx["v173_prob_oof"], player_oof, 0.10), blend_probabilities(ctx["v173_prob_test"], player_test, 0.10)),
        "v246_player_policy_v173blend_w0p20": (blend_probabilities(ctx["v173_prob_oof"], player_oof, 0.20), blend_probabilities(ctx["v173_prob_test"], player_test, 0.20)),
        "v246_player_policy_v173blend_w0p35": (blend_probabilities(ctx["v173_prob_oof"], player_oof, 0.35), blend_probabilities(ctx["v173_prob_test"], player_test, 0.35)),
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
    search.to_csv(OUTDIR / "v246_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v246_policy_metrics.csv", index=False)
    (OUTDIR / "v246_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v246_player_conditional_counterfactual.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
