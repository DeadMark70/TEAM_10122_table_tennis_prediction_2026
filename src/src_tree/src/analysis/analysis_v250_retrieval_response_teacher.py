"""V250 retrieval / kNN response teacher."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v250_v253_action_source_common import align_feature_frames, evaluate_and_export_variants, report_json, response_feature_frame
from analysis_v250_v253_action_source_helpers import logit_adjust_probability, standardize_train_test, weighted_neighbor_action_prob
from analysis_v243_v247_action_experiment_common import load_action_context


OUTDIR = Path("v250_retrieval_response_teacher")
SRC_DEST = Path("src/analysis/analysis_v250_retrieval_response_teacher.py")


def retrieval_probs(ctx: dict, k: int = 80, temperature: float = 2.0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    x_rows, x_test = align_feature_frames(response_feature_frame(rows), response_feature_frame(test_rows))
    oof = np.zeros((len(rows), 19), dtype=float)
    test_sum = np.zeros((len(test_rows), 19), dtype=float)
    metrics = []
    folds = sorted(rows["fold"].astype(int).unique())
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        x_train, x_valid = standardize_train_test(x_rows.loc[train].to_numpy(), x_rows.loc[valid].to_numpy())
        _, x_test_std = standardize_train_test(x_rows.loc[train].to_numpy(), x_test.to_numpy())
        nn = NearestNeighbors(n_neighbors=min(k, int(train.sum())), metric="euclidean", n_jobs=1)
        nn.fit(x_train)
        dist, idx = nn.kneighbors(x_valid)
        oof[valid] = weighted_neighbor_action_prob(y[train][idx], dist, n_classes=19, temperature=temperature)
        dist_t, idx_t = nn.kneighbors(x_test_std)
        test_sum += weighted_neighbor_action_prob(y[train][idx_t], dist_t, n_classes=19, temperature=temperature) / len(folds)
        metrics.append({"fold": int(fold), "train_rows": int(train.sum()), "valid_rows": int(valid.sum()), "k": int(min(k, int(train.sum())))})
    return normalize_probability_rows(oof), normalize_probability_rows(test_sum), metrics


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    teacher_oof, teacher_test, metrics = retrieval_probs(ctx)
    counts = np.bincount(ctx["y"], minlength=19)
    extra = {
        "v250_retrieval_logadj_tau0p25": (logit_adjust_probability(teacher_oof, counts, 0.25), logit_adjust_probability(teacher_test, counts, 0.25)),
        "v250_retrieval_logadj_tau0p50": (logit_adjust_probability(teacher_oof, counts, 0.50), logit_adjust_probability(teacher_test, counts, 0.50)),
        "v250_retrieval_v173_logadj_w0p35": (
            blend_probabilities(ctx["v173_prob_oof"], logit_adjust_probability(teacher_oof, counts, 0.25), 0.35),
            blend_probabilities(ctx["v173_prob_test"], logit_adjust_probability(teacher_test, counts, 0.25), 0.35),
        ),
    }
    search, generated = evaluate_and_export_variants(OUTDIR, "v250_retrieval", teacher_oof, teacher_test, ctx, extra)
    search.to_csv(OUTDIR / "v250_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v250_fold_metrics.csv", index=False)
    report_json(OUTDIR, "v250", search, generated)
    shutil.copy2("analysis_v250_retrieval_response_teacher.py", SRC_DEST)


if __name__ == "__main__":
    main()
