"""V253 curriculum 2.0 proxy with retrieval response as a new component."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import load_action_context
from analysis_v250_v253_action_source_common import evaluate_and_export_variants, report_json


OUTDIR = Path("v253_curriculum2_action_proxy")
SRC_DEST = Path("src/analysis/analysis_v253_curriculum2_action_proxy.py")


def load_prob(path: str) -> np.ndarray:
    return normalize_probability_rows(np.load(path))


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    retrieval_oof = load_prob("v250_retrieval_response_teacher/v250_retrieval_raw_oof_action_prob.npy")
    retrieval_test = load_prob("v250_retrieval_response_teacher/v250_retrieval_raw_test_action_prob.npy")
    # Use existing V173 probability as the stable curriculum anchor, but add a
    # new response-retrieval component before final argmax.
    variants = {}
    for w in [0.10, 0.20, 0.30, 0.45]:
        variants[f"v253_v173_retrieval_curriculum_w{str(w).replace('.', 'p')}"] = (
            blend_probabilities(ctx["v173_prob_oof"], retrieval_oof, w),
            blend_probabilities(ctx["v173_prob_test"], retrieval_test, w),
        )
    # A more aggressive mixture: retrieval plus V252 calibrated ensemble if it exists.
    v252_oof_path = Path("v252_longtail_calibrated_classifier/v252_ensemble_logadj_tau0p30_oof_action_prob.npy")
    v252_test_path = Path("v252_longtail_calibrated_classifier/v252_ensemble_logadj_tau0p30_test_action_prob.npy")
    if v252_oof_path.exists() and v252_test_path.exists():
        v252_oof = load_prob(str(v252_oof_path))
        v252_test = load_prob(str(v252_test_path))
        mixed_oof = normalize_probability_rows(0.65 * retrieval_oof + 0.35 * v252_oof)
        mixed_test = normalize_probability_rows(0.65 * retrieval_test + 0.35 * v252_test)
        variants["v253_retrieval_v252_curriculum_w0p30"] = (
            blend_probabilities(ctx["v173_prob_oof"], mixed_oof, 0.30),
            blend_probabilities(ctx["v173_prob_test"], mixed_test, 0.30),
        )
    # evaluate_and_export_variants also creates a raw source, so pass retrieval as
    # the primary teacher and append curriculum variants.
    search, generated = evaluate_and_export_variants(OUTDIR, "v253_retrieval_curriculum", retrieval_oof, retrieval_test, ctx, variants)
    search.to_csv(OUTDIR / "v253_action_search.csv", index=False)
    report_json(OUTDIR, "v253", search, generated)
    shutil.copy2("analysis_v253_curriculum2_action_proxy.py", SRC_DEST)


if __name__ == "__main__":
    main()
