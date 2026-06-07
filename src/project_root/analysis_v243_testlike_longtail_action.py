"""V243 test-like sampler plus long-tail action retraining.

This is a no-old action-only experiment.  Point is fixed to V188 cap5 and
server is fixed to R121.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities
from analysis_v243_v247_action_augmentation_helpers import balanced_softmax_adjustment, mix_probabilities
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


OUTDIR = Path("v243_testlike_longtail_action")
SRC_DEST = Path("src/analysis/analysis_v243_testlike_longtail_action.py")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    v173_oof = ctx["v173_oof"]
    v173_test = ctx["v173_test"]
    point = ctx["point"]
    server = ctx["server"]
    v173_prob_oof = ctx["v173_prob_oof"]
    v173_prob_test = ctx["v173_prob_test"]

    cols = feature_columns(rows)
    test_rows = align_test_columns(rows, test_rows, cols)
    weights = context_weights(rows, test_rows, low=0.20, high=5.0)
    sample_weight = class_balanced_weights(y, weights, power=0.45, cap=5.0)
    teacher_oof, teacher_test, fold_metrics = train_extratrees_oof(rows, test_rows, y, cols, sample_weight, seed=2430, n_estimators=180, min_samples_leaf=3)

    counts = np.bincount(y, minlength=19)
    variants = {
        "v243_testlike_et_raw": (teacher_oof, teacher_test),
        "v243_testlike_et_v173blend_w0p20": (blend_probabilities(v173_prob_oof, teacher_oof, 0.20), blend_probabilities(v173_prob_test, teacher_test, 0.20)),
        "v243_testlike_et_v173blend_w0p35": (blend_probabilities(v173_prob_oof, teacher_oof, 0.35), blend_probabilities(v173_prob_test, teacher_test, 0.35)),
        "v243_balanced_softmax_s0p25": (
            mix_probabilities(v173_prob_oof, balanced_softmax_adjustment(teacher_oof, counts, 0.25), 0.35),
            mix_probabilities(v173_prob_test, balanced_softmax_adjustment(teacher_test, counts, 0.25), 0.35),
        ),
        "v243_balanced_softmax_s0p50": (
            mix_probabilities(v173_prob_oof, balanced_softmax_adjustment(teacher_oof, counts, 0.50), 0.35),
            mix_probabilities(v173_prob_test, balanced_softmax_adjustment(teacher_test, counts, 0.50), 0.35),
        ),
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
        generated.append(write_submission(OUTDIR, f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))

    search, best_delta, verdict = finalize_search(records)
    search.to_csv(OUTDIR / "v243_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v243_fold_metrics.csv", index=False)
    (OUTDIR / "v243_report.json").write_text(
        json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2),
        encoding="utf-8",
    )
    shutil.copy2("analysis_v243_testlike_longtail_action.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
