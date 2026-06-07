"""V252 long-tail calibration over V250/V251 action posteriors."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import load_action_context
from analysis_v250_v253_action_source_common import evaluate_and_export_variants, report_json
from analysis_v250_v253_action_source_helpers import logit_adjust_probability


OUTDIR = Path("v252_longtail_calibrated_classifier")
SRC_DEST = Path("src/analysis/analysis_v252_longtail_calibrated_classifier.py")


def load_prob(path: str) -> np.ndarray:
    return normalize_probability_rows(np.load(path))


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    y = ctx["y"]
    counts = np.bincount(y, minlength=19)
    sources = {}
    for name, oof_path, test_path in [
        ("retrieval", "v250_retrieval_response_teacher/v250_retrieval_raw_oof_action_prob.npy", "v250_retrieval_response_teacher/v250_retrieval_raw_test_action_prob.npy"),
        ("mlp", "v251_response_encoder_teacher/v251_mlp_response_raw_oof_action_prob.npy", "v251_response_encoder_teacher/v251_mlp_response_raw_test_action_prob.npy"),
    ]:
        if Path(oof_path).exists() and Path(test_path).exists():
            sources[name] = (load_prob(oof_path), load_prob(test_path))
    if not sources:
        raise FileNotFoundError("V252 requires V250 or V251 probability files")

    base_oof = normalize_probability_rows(np.mean([p[0] for p in sources.values()], axis=0))
    base_test = normalize_probability_rows(np.mean([p[1] for p in sources.values()], axis=0))
    extra = {}
    for tau in [0.15, 0.30, 0.50]:
        adj_oof = logit_adjust_probability(base_oof, counts, tau)
        adj_test = logit_adjust_probability(base_test, counts, tau)
        extra[f"v252_ensemble_logadj_tau{str(tau).replace('.', 'p')}"] = (adj_oof, adj_test)
        extra[f"v252_ensemble_logadj_v173blend_tau{str(tau).replace('.', 'p')}_w0p35"] = (
            blend_probabilities(ctx["v173_prob_oof"], adj_oof, 0.35),
            blend_probabilities(ctx["v173_prob_test"], adj_test, 0.35),
        )
    search, generated = evaluate_and_export_variants(OUTDIR, "v252_response_ensemble", base_oof, base_test, ctx, extra)
    search.to_csv(OUTDIR / "v252_action_search.csv", index=False)
    report_json(OUTDIR, "v252", search, generated)
    shutil.copy2("analysis_v252_longtail_calibrated_classifier.py", SRC_DEST)


if __name__ == "__main__":
    main()
