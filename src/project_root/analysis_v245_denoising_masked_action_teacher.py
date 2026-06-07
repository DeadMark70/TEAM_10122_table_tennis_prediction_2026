"""V245 denoising/masked feature action teachers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import (
    align_test_columns,
    class_balanced_weights,
    context_weights,
    evaluate_action,
    feature_columns,
    finalize_search,
    load_action_context,
    train_extratrees_oof,
    write_submission,
)


OUTDIR = Path("v245_denoising_masked_action_teacher")
SRC_DEST = Path("src/analysis/analysis_v245_denoising_masked_action_teacher.py")
VIEWS = {
    "drop_player_score": ("player", "receiver", "hitter", "score"),
    "drop_spin_strength": ("spin", "strength"),
    "robust_core": ("player", "receiver", "hitter", "score", "spin", "strength"),
}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    v173_oof = ctx["v173_oof"]
    v173_test = ctx["v173_test"]
    weights = context_weights(rows, test_rows)
    sample_weight = class_balanced_weights(y, weights, power=0.35, cap=4.0)
    records = [evaluate_action("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    view_oof = []
    view_test = []
    metrics = []
    for view, drops in VIEWS.items():
        cols = feature_columns(rows, drop_keywords=drops)
        aligned = align_test_columns(rows, test_rows, cols)
        prob_oof, prob_test, fold_metrics = train_extratrees_oof(rows, aligned, y, cols, sample_weight, seed=2450 + len(view), n_estimators=90, min_samples_leaf=5)
        view_oof.append(prob_oof)
        view_test.append(prob_test)
        metrics.extend(dict(m, view=view) for m in fold_metrics)
        for weight in [0.20, 0.35]:
            name = f"v245_{view}_v173blend_w{str(weight).replace('.', 'p')}"
            p_oof = blend_probabilities(ctx["v173_prob_oof"], prob_oof, weight)
            p_test = blend_probabilities(ctx["v173_prob_test"], prob_test, weight)
            pred = p_oof.argmax(axis=1).astype(int)
            test_pred = p_test.argmax(axis=1).astype(int)
            rec = evaluate_action(name, y, pred, v173_oof, weights)
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
            records.append(rec)
            generated.append(write_submission(OUTDIR, f"submission_{name}__pv188cap5__sr121.csv", test_pred, ctx["point"], ctx["server"]))
        np.save(OUTDIR / f"v245_{view}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"v245_{view}_test_action_prob.npy", prob_test)

    ens_oof = normalize_probability_rows(np.mean(view_oof, axis=0))
    ens_test = normalize_probability_rows(np.mean(view_test, axis=0))
    for weight in [0.20, 0.35, 0.50]:
        name = f"v245_masked_ensemble_v173blend_w{str(weight).replace('.', 'p')}"
        p_oof = blend_probabilities(ctx["v173_prob_oof"], ens_oof, weight)
        p_test = blend_probabilities(ctx["v173_prob_test"], ens_test, weight)
        pred = p_oof.argmax(axis=1).astype(int)
        test_pred = p_test.argmax(axis=1).astype(int)
        rec = evaluate_action(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        generated.append(write_submission(OUTDIR, f"submission_{name}__pv188cap5__sr121.csv", test_pred, ctx["point"], ctx["server"]))

    search, best_delta, verdict = finalize_search(records)
    search.to_csv(OUTDIR / "v245_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v245_fold_metrics.csv", index=False)
    (OUTDIR / "v245_report.json").write_text(json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2), encoding="utf-8")
    shutil.copy2("analysis_v245_denoising_masked_action_teacher.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
