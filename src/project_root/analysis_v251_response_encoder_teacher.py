"""V251 response-context MLP encoder teacher.

This is not a sequence model.  It uses incoming-ball/phase/player response
features and trains a small fold-safe MLP posterior.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import load_action_context
from analysis_v250_v253_action_source_common import align_feature_frames, evaluate_and_export_variants, report_json, response_feature_frame
from analysis_v250_v253_action_source_helpers import standardize_train_test


OUTDIR = Path("v251_response_encoder_teacher")
SRC_DEST = Path("src/analysis/analysis_v251_response_encoder_teacher.py")


def predict_full(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 19), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return normalize_probability_rows(out)


def mlp_oof(ctx: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
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
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=80,
            random_state=2510 + int(fold),
            early_stopping=False,
            solver="adam",
        )
        model.fit(x_train, y[train])
        oof[valid] = predict_full(model, x_valid)
        test_sum += predict_full(model, x_test_std) / len(folds)
        metrics.append({"fold": int(fold), "iters": int(model.n_iter_), "loss": float(model.loss_)})
    return normalize_probability_rows(oof), normalize_probability_rows(test_sum), metrics


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    teacher_oof, teacher_test, metrics = mlp_oof(ctx)
    extra = {
        "v251_mlp_v173kd_w0p15": (
            blend_probabilities(ctx["v173_prob_oof"], teacher_oof, 0.15),
            blend_probabilities(ctx["v173_prob_test"], teacher_test, 0.15),
        ),
        "v251_mlp_v173kd_w0p30": (
            blend_probabilities(ctx["v173_prob_oof"], teacher_oof, 0.30),
            blend_probabilities(ctx["v173_prob_test"], teacher_test, 0.30),
        ),
    }
    search, generated = evaluate_and_export_variants(OUTDIR, "v251_mlp_response", teacher_oof, teacher_test, ctx, extra)
    search.to_csv(OUTDIR / "v251_action_search.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v251_fold_metrics.csv", index=False)
    report_json(OUTDIR, "v251", search, generated)
    shutil.copy2("analysis_v251_response_encoder_teacher.py", SRC_DEST)


if __name__ == "__main__":
    main()
