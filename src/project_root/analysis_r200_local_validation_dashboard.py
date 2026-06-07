"""R200 local validation dashboard.

This script gives every local candidate the same no-label test sanity and
available OOF summary checks.  It is deliberately conservative: when row-level
OOF predictions are unavailable, the ordinary/test-like OOF fields are left
missing instead of being inferred.

No model is trained and TTMATCH is not read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v165_combined_external_pretrain_proxy import prepare_prefix_features
from analysis_v194_train_test_split_distribution_audit import add_audit_columns


OUTDIR = Path("r200_local_validation_dashboard")
UPLOAD_DIR = Path("upload_candidates_20260519")
SRC_DEST = Path("src/analysis/analysis_r200_local_validation_dashboard.py")

ANCHOR = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")

DEFAULT_CANDIDATES = [
    ANCHOR,
    UPLOAD_DIR / "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv",
    UPLOAD_DIR / "submission_v191_r184_attack_to_control__pv188_r186_w005_cap5__sr121.csv",
    UPLOAD_DIR / "submission_v191_r184_state_pair_supported__pv188_r186_w005_cap5__sr121.csv",
    UPLOAD_DIR / "submission_v191_r184_receive_affordance_control__pv188_r186_w005_cap5__sr121.csv",
    UPLOAD_DIR / "submission_v193_p0match0p29_all_a0p075_cap0p05__v173action_r121server.csv",
    UPLOAD_DIR / "submission_v196a_r111_importance_p0t029_rw1_conf085_cw025_tc005_a0p075_cap0p05__v173action_r121server.csv",
]


def churn_tier(task: str, churn: float) -> str:
    if task == "action":
        if churn <= 0.05:
            return "low"
        if churn <= 0.10:
            return "medium"
        return "high"
    if task == "point":
        if churn <= 0.02:
            return "safe_probe"
        if churn <= 0.05:
            return "normal"
        return "high"
    if task == "server":
        if churn <= 0.02:
            return "clean"
        if churn <= 0.04:
            return "medium"
        return "high"
    raise ValueError(task)


def decision_label(point_churn: float, action_churn: float, server_mad: float, point0_rate: float) -> str:
    if point0_rate > 0.75:
        return "REJECT_POINT0_COLLAPSE"
    if point_churn > 0.05:
        return "REJECT_POINT_CHURN"
    if action_churn > 0.12:
        return "REVIEW_ACTION_CHURN"
    if server_mad > 0.04:
        return "REVIEW_SERVER_MAD"
    return "KEEP"


def slice_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    prefix = pd.to_numeric(rows["prefix_len"], errors="coerce").fillna(0)
    phase = rows["audit_phase"].astype(str)
    depth = rows["audit_lag0_depth"].astype(str)
    family = rows["audit_lag0_action_family"].astype(str)
    return {
        "all": np.ones(len(rows), dtype=bool),
        "prefix_1": prefix.eq(1).to_numpy(),
        "prefix_2": prefix.eq(2).to_numpy(),
        "prefix_3": prefix.eq(3).to_numpy(),
        "prefix_4_6": prefix.between(4, 6).to_numpy(),
        "prefix_7_plus": prefix.ge(7).to_numpy(),
        "phase_receive": phase.eq("receive").to_numpy(),
        "phase_third_ball": phase.eq("third_ball").to_numpy(),
        "phase_fourth_ball": phase.eq("fourth_ball").to_numpy(),
        "phase_rally": phase.eq("rally").to_numpy(),
        "lag0_short": depth.eq("short").to_numpy(),
        "lag0_half": depth.eq("half").to_numpy(),
        "lag0_long": depth.eq("long").to_numpy(),
        "lag0_attack": family.eq("Attack").to_numpy(),
        "lag0_control": family.eq("Control").to_numpy(),
        "lag0_defensive": family.eq("Defensive").to_numpy(),
    }


def load_submission(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def count_distribution(values: np.ndarray, n: int) -> dict[str, int]:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=n)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def known_oof_summary(candidate: str) -> dict:
    """Read already produced search reports for known candidates."""
    tables = [
        Path("v193_v188_calibrated_residual/v193_search.csv"),
        Path("v196_point0_calibrated_gru/v196_search.csv"),
        Path("v188_point_intent_gru/v188_search.csv"),
        Path("v202_three_stage_domain_adapt/v202_search.csv"),
        Path("v203_tt_shuttlenet/v203_search.csv"),
        Path("v204_ttshuttle_residual_selector/v204_search.csv"),
        Path("v207_anchor_relative_ttselector/v207_search.csv"),
        Path("v208_action_ttshuttlenet/v208_action_search.csv"),
        Path("v209_action_selector_reranker/v209_action_search.csv"),
        Path("v211_true_shuttlenet_selector/v211_action_search.csv"),
        Path("v214_shuttlenet_component_ablation/v214_action_search.csv"),
        Path("v216_terminal_action_tuner/v216_action_search.csv"),
        Path("v217_macro_f1_utility_reranker/v217_action_search.csv"),
        Path("v218_action_weakclass_booster/v218_action_search.csv"),
        Path("v219_action_classwise_budget/v219_action_search.csv"),
        Path("v220_action_backoff_support_filter/v220_action_search.csv"),
        Path("v221_action_backoff_candidate_generator/v221_action_search.csv"),
        Path("v222_v225_action_improvement_suite/v222_v225_action_search.csv"),
        Path("v230_action_soft_teacher_factory/v230_action_search.csv"),
        Path("v231_action_only_sequence_teacher/v231_action_search.csv"),
        Path("v232_v173_curriculum_deepening/v232_action_search.csv"),
        Path("v234_v173_phase_expert_reconstruction/v234_action_search.csv"),
        Path("v235_player_conditional_response_teacher/v235_action_search.csv"),
        Path("v236_distributional_action_calibrator/v236_action_search.csv"),
        Path("v237_deep_phase_style_action/v237_action_search.csv"),
        Path("v238_v173_reconstruction_ablation/v238_action_search.csv"),
        Path("v239_alt_tabular_action_teachers/v239_action_search.csv"),
        Path("v240_candidate_action_reranker2/v240_action_search.csv"),
        Path("v241_weak_action_specialists/v241_action_search.csv"),
        Path("v242_contrastive_response_features/v242_action_search.csv"),
        Path("v243_testlike_longtail_action/v243_action_search.csv"),
        Path("v244_response_policy_soft_targets/v244_action_search.csv"),
        Path("v245_denoising_masked_action_teacher/v245_action_search.csv"),
        Path("v246_player_conditional_counterfactual/v246_action_search.csv"),
        Path("v247_action_point_consistency_aux/v247_action_search.csv"),
        Path("v248_v173_curriculum_decomposition/v248_candidate_search.csv"),
        Path("v250_retrieval_response_teacher/v250_action_search.csv"),
        Path("v251_response_encoder_teacher/v251_action_search.csv"),
        Path("v252_longtail_calibrated_classifier/v252_action_search.csv"),
        Path("v253_curriculum2_action_proxy/v253_action_search.csv"),
        Path("v255_external_action_family_smoke/v255_action_search.csv"),
        Path("v256_external_representation_action/v256_action_search.csv"),
        Path("v257_aicup_action_finetune/v257_action_search.csv"),
        Path("v258_true_encoder_finetune/v258_action_search.csv"),
        Path("v260_longtail_action_teacher/v260_action_search.csv"),
        Path("v261_action_conditioned_point_residual/v261_point_search.csv"),
        Path("v261b_direct_v188_point_gate/v261b_point_search.csv"),
        Path("v262_tabletennis_external_sequence_teacher/v262_action_search.csv"),
        Path("v263_questionnaire_baseline/v263a_action_search.csv"),
        Path("v263_questionnaire_baseline/v263b_point_search.csv"),
        Path("v263_questionnaire_baseline/v263c_server_search.csv"),
        Path("v263_questionnaire_baseline/v263d_package_search.csv"),
        Path("v264_current_best_structure_packager/v264_packaging_report.csv"),
        Path("v265_ttmatch_diagnostic/v265_candidate_search.csv"),
        Path("v266_clean_autoresearch_loop/v266_candidate_search.csv"),
        Path("v267_macro_f1_action_teacher/v267_action_search.csv"),
        Path("v268_macro_f1_point_residual/v268_point_search.csv"),
        Path("v269_clean_server_value_ranker/v269_server_search.csv"),
        Path("v270_clean_candidate_packager/v270_package_search.csv"),
        Path("v271_server_microblend_probe/v271_server_probe_search.csv"),
        Path("v272_action_conditioned_point_residual/v272_point_search.csv"),
        Path("v273_player_conditional_action_response/v273_action_search.csv"),
        Path("v276_clean_next_stage_packager/v276_package_search.csv"),
        Path("v277_v272b_point_refinement/v277_point_search.csv"),
        Path("v280_joint_action_point_optimizer/v280_pair_search.csv"),
        Path("v282_joint_context_support_optimizer/v282_pair_search.csv"),
        Path("v283_pair_level_selector/v283_pair_search.csv"),
        Path("v285_mambalite_sequence_teacher/v285_action_search.csv"),
        Path("v286_weak_action_specialist_pretraining/v286_action_search.csv"),
        Path("v287_weak_action_gated_ensemble/v287_action_search.csv"),
        Path("v288_specialist_feature_discovery/v288_group_search.csv"),
        Path("v289_terminal03_specialist/v289_terminal03_search.csv"),
        Path("v290_shortcontrol411_specialist/v290_shortcontrol_search.csv"),
        Path("v291_weak_class_training_upgrade/v291_candidate_search.csv"),
        Path("v292_weak_class_pretraining_action_teacher/v292_teacher_search.csv"),
        Path("v293_point_weakclass_residual_lab/v293_candidate_search.csv"),
        Path("v295_true_oof_point_specialists/v295_candidate_search.csv"),
        Path("v297_multisource_point_agreement/v297_candidate_search.csv"),
        Path("v298_action_point_support_prior/v298_candidate_search.csv"),
        Path("v299_point_hybrid_selector/v299_candidate_search.csv"),
        Path("v300_clean_server_blend_recycler/v300_server_search.csv"),
        Path("v301_action_point_consistency_explorer/v301_pair_search.csv"),
        Path("v302_clean_server_calibration_sweep/v302_server_search.csv"),
        Path("v303_point_server_packaging/v303_package_search.csv"),
        Path("v281_ttmatch_diagnostic_validator/v281_ttmatch_diagnostic.csv"),
    ]
    empty = {
        "ordinary_point_macro_f1": np.nan,
        "ordinary_delta_vs_base": np.nan,
        "ordinary_point_churn": np.nan,
        "ordinary_action_macro_f1": np.nan,
        "ordinary_action_delta_vs_anchor": np.nan,
        "ordinary_action_churn": np.nan,
        "oof_source": "",
    }
    for path in tables:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        name_col = "candidate" if "candidate" in df.columns else "submission" if "submission" in df.columns else None
        if name_col is None:
            continue
        stem = candidate.replace("submission_", "").replace("__v173action_r121server.csv", "")
        stem = stem.replace("__pv188_r186_w005_cap5__sr121.csv", "")
        stem = stem.replace("__pv188cap5__sr121.csv", "")
        stem = stem.replace("__pv261cap1__sr121.csv", "")
        if stem == "v292_best_gatedblend":
            stem = "v292_extratrees_weighted_weak_w0p03_gatedblend"
        elif stem == "v292_best_softblend_diag":
            stem = "v292_logreg_balanced_w0p03_softblend"
        elif stem.startswith("v292_") and "w0p10" in stem:
            stem = stem.replace("w0p10", "w0p1")
        hit = df[df[name_col].astype(str).str.contains(stem, regex=False, na=False)]
        if not hit.empty:
            rec = hit.iloc[0].to_dict()
            out = empty.copy()
            out.update(
                {
                    "ordinary_point_macro_f1": rec.get("point_macro_f1", np.nan),
                    "ordinary_delta_vs_base": rec.get(
                        "delta_vs_base",
                        rec.get(
                            "delta_vs_v261",
                            rec.get(
                                "delta_vs_v294_base",
                                rec.get("delta_vs_aligned_base", np.nan),
                            ),
                        ),
                    ),
                    "ordinary_point_churn": rec.get("point_churn_vs_base", rec.get("point_churn", np.nan)),
                    "ordinary_action_macro_f1": rec.get("action_macro_f1", np.nan),
                    "ordinary_action_delta_vs_anchor": rec.get(
                        "delta_vs_v173_anchor", rec.get("delta_vs_v173", np.nan)
                    ),
                    "ordinary_action_churn": rec.get("action_churn_vs_v173_anchor", np.nan),
                "oof_source": str(path),
                }
            )
            return out
    return empty


def evaluate_candidate(path: Path, anchor: pd.DataFrame, rows: pd.DataFrame) -> tuple[dict, list[dict], list[dict]]:
    cand = load_submission(path, rows["rally_uid"].to_numpy())
    action_diff = cand["actionId"].astype(int).to_numpy() != anchor["actionId"].astype(int).to_numpy()
    point_diff = cand["pointId"].astype(int).to_numpy() != anchor["pointId"].astype(int).to_numpy()
    server_delta = cand["serverGetPoint"].astype(float).to_numpy() - anchor["serverGetPoint"].astype(float).to_numpy()
    action_churn = float(action_diff.mean())
    point_churn = float(point_diff.mean())
    server_mad = float(np.mean(np.abs(server_delta)))
    point0_rate = float(cand["pointId"].astype(int).eq(0).mean())
    rec = {
        "candidate": path.name,
        "rows": int(len(cand)),
        "action_churn_vs_anchor": action_churn,
        "point_churn_vs_anchor": point_churn,
        "server_mad_vs_anchor": server_mad,
        "server_corr_vs_anchor": float(np.corrcoef(cand["serverGetPoint"].astype(float), anchor["serverGetPoint"].astype(float))[0, 1]) if len(cand) > 2 else np.nan,
        "action_tier": churn_tier("action", action_churn),
        "point_tier": churn_tier("point", point_churn),
        "server_tier": churn_tier("server", server_mad),
        "point0_rate": point0_rate,
        "action_distribution": json.dumps(count_distribution(cand["actionId"].astype(int).to_numpy(), 19), sort_keys=True),
        "point_distribution": json.dumps(count_distribution(cand["pointId"].astype(int).to_numpy(), 10), sort_keys=True),
        "decision": decision_label(point_churn, action_churn, server_mad, point0_rate),
    }
    rec.update(known_oof_summary(path.name))

    churn_rows = []
    for col, diff in [("actionId", action_diff), ("pointId", point_diff)]:
        if diff.any():
            trans = pd.DataFrame({"from": anchor.loc[diff, col].astype(int).to_numpy(), "to": cand.loc[diff, col].astype(int).to_numpy()})
            for row in trans.value_counts(["from", "to"]).reset_index(name="rows").itertuples(index=False):
                churn_rows.append({"candidate": path.name, "task": col, "from": int(row[0]), "to": int(row[1]), "rows": int(row[2])})

    slice_rows = []
    for name, mask in slice_masks(rows).items():
        if mask.sum() == 0:
            continue
        slice_rows.append(
            {
                "candidate": path.name,
                "slice": name,
                "rows": int(mask.sum()),
                "action_churn": float(action_diff[mask].mean()),
                "point_churn": float(point_diff[mask].mean()),
                "server_mad": float(np.mean(np.abs(server_delta[mask]))),
                "point0_rate": float(cand.loc[mask, "pointId"].astype(int).eq(0).mean()),
            }
        )
    return rec, churn_rows, slice_rows


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    _, _, _, test_prefix, _ = prepare_prefix_features()
    rows = add_audit_columns(test_prefix.reset_index(drop=True))
    rally_uids = rows["rally_uid"].astype(int).to_numpy()
    anchor = load_submission(ANCHOR, rally_uids)
    candidates = [p for p in DEFAULT_CANDIDATES if p.exists()]
    for pattern in [
        "submission_v197_*__pv188_r186_w005_cap5__sr121.csv",
        "submission_v199_*__v173action_r121server.csv",
        "submission_v198_*__v173action_r121server.csv",
        "submission_v202*.csv",
        "submission_v203*.csv",
        "submission_v206*.csv",
        "submission_v204*.csv",
        "submission_v205*.csv",
        "submission_v207*.csv",
        "submission_v208*.csv",
        "submission_v209*.csv",
        "submission_v210*.csv",
        "submission_v211*.csv",
        "submission_v214*.csv",
        "submission_v216*.csv",
        "submission_v217*.csv",
        "submission_v218*.csv",
        "submission_v219*.csv",
        "submission_v220*.csv",
        "submission_v221*.csv",
        "submission_v222*.csv",
        "submission_v223*.csv",
        "submission_v224*.csv",
        "submission_v225*.csv",
        "submission_v226*.csv",
        "submission_v230*.csv",
        "submission_v231*.csv",
        "submission_v232*.csv",
        "submission_v234*.csv",
        "submission_v235*.csv",
        "submission_v236*.csv",
        "submission_v237*.csv",
        "submission_v238*.csv",
        "submission_v239*.csv",
        "submission_v240*.csv",
        "submission_v241*.csv",
        "submission_v242*.csv",
        "submission_v243*.csv",
        "submission_v244*.csv",
        "submission_v245*.csv",
        "submission_v246*.csv",
        "submission_v247*.csv",
        "submission_v248*.csv",
        "submission_v250*.csv",
        "submission_v251*.csv",
        "submission_v252*.csv",
        "submission_v253*.csv",
        "submission_v255*.csv",
        "submission_v256*.csv",
        "submission_v257*.csv",
        "submission_v258*.csv",
        "submission_v260*.csv",
        "submission_v261*.csv",
        "submission_v262*.csv",
        "submission_v263*.csv",
        "submission_v264*.csv",
        "submission_v265*.csv",
        "submission_v266*.csv",
        "submission_v267*.csv",
        "submission_v268*.csv",
        "submission_v269*.csv",
        "submission_v270*.csv",
        "submission_v271*.csv",
        "submission_v272*.csv",
        "submission_v273*.csv",
        "submission_v276*.csv",
        "submission_v277*.csv",
        "submission_v280*.csv",
        "submission_v282*.csv",
        "submission_v283*.csv",
        "submission_v285*.csv",
        "submission_v286*.csv",
        "submission_v287*.csv",
        "submission_v288*.csv",
        "submission_v289*.csv",
        "submission_v290*.csv",
        "submission_v291*.csv",
        "submission_v292*.csv",
        "submission_v293*.csv",
        "submission_v295*.csv",
        "submission_v297*.csv",
        "submission_v298*.csv",
        "submission_v299*.csv",
        "submission_v300*.csv",
        "submission_v301*.csv",
        "submission_v302*.csv",
        "submission_v303*.csv",
        "submission_r201_*__v173_v188cap5.csv",
    ]:
        candidates.extend(sorted(UPLOAD_DIR.glob(pattern)))
    candidates.extend(sorted(Path("v263_questionnaire_baseline").glob("submission_v263*.csv")))
    candidates.extend(sorted(Path("v264_current_best_structure_packager").glob("submission_v264*.csv")))
    candidates.extend(sorted(Path("v265_ttmatch_diagnostic").glob("submission_v265*.csv")))
    candidates.extend(sorted(Path("v266_clean_autoresearch_loop").glob("submission_v266*.csv")))
    candidates.extend(sorted(Path("v267_macro_f1_action_teacher").glob("submission_v267*.csv")))
    candidates.extend(sorted(Path("v268_macro_f1_point_residual").glob("submission_v268*.csv")))
    candidates.extend(sorted(Path("v269_clean_server_value_ranker").glob("submission_v269*.csv")))
    candidates.extend(sorted(Path("v270_clean_candidate_packager").glob("submission_v270*.csv")))
    candidates.extend(sorted(Path("v271_server_microblend_probe").glob("submission_v271*.csv")))
    candidates.extend(sorted(Path("v272_action_conditioned_point_residual").glob("submission_v272*.csv")))
    candidates.extend(sorted(Path("v273_player_conditional_action_response").glob("submission_v273*.csv")))
    candidates.extend(sorted(Path("v276_clean_next_stage_packager").glob("submission_v276*.csv")))
    candidates.extend(sorted(Path("v277_v272b_point_refinement").glob("submission_v277*.csv")))
    candidates.extend(sorted(Path("v280_joint_action_point_optimizer").glob("submission_v280*.csv")))
    candidates.extend(sorted(Path("v282_joint_context_support_optimizer").glob("submission_v282*.csv")))
    candidates.extend(sorted(Path("v283_pair_level_selector").glob("submission_v283*.csv")))
    candidates.extend(sorted(Path("v285_mambalite_sequence_teacher").glob("submission_v285*.csv")))
    candidates.extend(sorted(Path("v286_weak_action_specialist_pretraining").glob("submission_v286*.csv")))
    candidates.extend(sorted(Path("v287_weak_action_gated_ensemble").glob("submission_v287*.csv")))
    candidates.extend(sorted(Path("v288_specialist_feature_discovery").glob("submission_v288*.csv")))
    candidates.extend(sorted(Path("v289_terminal03_specialist").glob("submission_v289*.csv")))
    candidates.extend(sorted(Path("v290_shortcontrol411_specialist").glob("submission_v290*.csv")))
    candidates.extend(sorted(Path("v291_weak_class_training_upgrade").glob("submission_v291*.csv")))
    candidates.extend(sorted(Path("v292_weak_class_pretraining_action_teacher").glob("submission_v292*.csv")))
    candidates.extend(sorted(Path("v293_point_weakclass_residual_lab").glob("submission_v293*.csv")))
    candidates.extend(sorted(Path("v295_true_oof_point_specialists").glob("submission_v295*.csv")))
    candidates.extend(sorted(Path("v297_multisource_point_agreement").glob("submission_v297*.csv")))
    candidates.extend(sorted(Path("v298_action_point_support_prior").glob("submission_v298*.csv")))
    candidates.extend(sorted(Path("v299_point_hybrid_selector").glob("submission_v299*.csv")))
    candidates.extend(sorted(Path("v300_clean_server_blend_recycler").glob("submission_v300*.csv")))
    candidates.extend(sorted(Path("v301_action_point_consistency_explorer").glob("submission_v301*.csv")))
    candidates.extend(sorted(Path("v302_clean_server_calibration_sweep").glob("submission_v302*.csv")))
    candidates.extend(sorted(Path("v303_point_server_packaging").glob("submission_v303*.csv")))
    candidates = sorted(set(candidates), key=lambda p: p.name)

    summary = []
    churn = []
    slices = []
    for path in candidates:
        rec, churn_rows, slice_rows = evaluate_candidate(path, anchor, rows)
        summary.append(rec)
        churn.extend(churn_rows)
        slices.extend(slice_rows)

    pd.DataFrame(summary).to_csv(OUTDIR / "r200_candidate_summary.csv", index=False)
    pd.DataFrame(churn).to_csv(OUTDIR / "r200_churn_report.csv", index=False)
    pd.DataFrame(slices).to_csv(OUTDIR / "r200_slice_metrics.csv", index=False)
    pd.DataFrame(summary)[["candidate", "decision", "action_tier", "point_tier", "server_tier"]].to_csv(OUTDIR / "r200_candidate_decision.csv", index=False)
    (OUTDIR / "r200_candidate_decision.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTDIR / "r200_report.md").write_text(
        "# R200 Local Validation Dashboard\n\n"
        f"- Anchor: `{ANCHOR}`\n"
        f"- Candidates evaluated: `{len(summary)}`\n"
        "- No model training. TTMATCH is not read.\n\n"
        "## Outputs\n\n"
        "- `r200_candidate_summary.csv`\n"
        "- `r200_slice_metrics.csv`\n"
        "- `r200_churn_report.csv`\n"
        "- `r200_candidate_decision.csv`\n"
        "- `r200_candidate_decision.json`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r200_local_validation_dashboard.py", SRC_DEST)
    print(json.dumps({"outdir": str(OUTDIR), "candidates": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
