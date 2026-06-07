"""V248 faithful V173 action curriculum decomposition.

This script rebuilds the original V173 action pipeline and audits which
component/schedule/phase/transition explains the public-positive V173 action
anchor.  It is no-old and action-only: generated submissions keep V188 cap5
point and R121 server fixed.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import analysis_v160_v163_task_pretrain_distill as v160
import analysis_v165_combined_external_pretrain_proxy as v165
import analysis_v173_external_curriculum_pretrain as v173
from analysis_r184_receiver_affordance_refiner import add_affordance_columns, load_pickle, load_sub
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from analysis_v238_v242_action_model_helpers import normalize_probability_rows
from analysis_v243_v247_action_augmentation_helpers import build_context_key_frame, clip_density_weights
from analysis_v248_v173_decomposition_helpers import acceptance_mask_by_phase, component_weight_grid, transition_counts
from baseline_lgbm import ACTION_CLASSES, validate_raw_data


OUTDIR = Path("v248_v173_curriculum_decomposition")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v248_v173_curriculum_decomposition.py")
R166_TARGETS = Path("r166_teacher_distillation/r166_teacher_targets.npz")
V173_SUBMITTED = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
WEAK_ACTIONS = [0, 3, 4, 5, 7, 8, 9, 12, 14]


def ensure_pickle_classes() -> None:
    __main__.V3Tuning = v173.V3Tuning
    __main__.GrUTuning = v173.GrUTuning
    __main__.TransformerTuning = v173.TransformerTuning


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def action_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def context_weights(rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
    return clip_density_weights(
        density_ratio_weights(
            build_context_key_frame(add_audit_columns(rows.copy())),
            build_context_key_frame(add_audit_columns(test_rows.copy())),
            ["prefix_bin", "phase", "lag0_family", "lag0_depth"],
        )
    )


def write_submission(name: str, rally_uids: np.ndarray, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": rally_uids.astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def evaluate_pred(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, v173_pred: np.ndarray, weights: np.ndarray, extra: dict | None = None) -> dict:
    score = action_f1(y, pred)
    base_score = action_f1(y, base)
    v173_score = action_f1(y, v173_pred)
    iw = weighted_macro_f1(y, pred, weights)
    v173_iw = weighted_macro_f1(y, v173_pred, weights)
    weak = float(f1_score(y, pred, labels=WEAK_ACTIONS, average="macro", zero_division=0))
    weak_v173 = float(f1_score(y, v173_pred, labels=WEAK_ACTIONS, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "action_macro_f1": score,
        "delta_vs_base": float(score - base_score),
        "delta_vs_v173": float(score - v173_score),
        "iw_delta_vs_v173": float(iw - v173_iw),
        "weak_delta_vs_v173": float(weak - weak_v173),
        "churn_vs_base": float(np.mean(pred != base)),
        "churn_vs_v173": float(np.mean(pred != v173_pred)),
        "changed_vs_base_rows": int(np.sum(pred != base)),
        "changed_vs_v173_rows": int(np.sum(pred != v173_pred)),
    }
    if extra:
        rec.update(extra)
    return rec


def compose(base: np.ndarray, teacher: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    out[np.asarray(mask, dtype=bool)] = np.asarray(teacher, dtype=int)[np.asarray(mask, dtype=bool)]
    return out


def load_v173_action_sources() -> dict:
    ensure_pickle_classes()
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)

    targets = np.load(R166_TARGETS)
    r111_oof = load_pickle(v165.R111_OOF)
    r111_test = load_pickle(v165.R111_TEST)
    r101_oof = load_pickle(v165.R101_OOF)
    r101_test = load_pickle(v165.R101_TEST)
    tuning = r111_oof["tuning"]

    _, _, prefix, test_prefix, _ = v165.prepare_prefix_features()
    rows = v165.align_prefix_meta(v160.ensure_fold(r111_oof["valid_meta"]), prefix).reset_index(drop=True)
    rows = add_affordance_columns(rows)
    test_rows = add_affordance_columns(test_prefix.reset_index(drop=True))
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_probability_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_probability_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])
    teacher_action_oof = normalize_probability_rows(targets["teacher_action_oof"])
    teacher_action_test = normalize_probability_rows(targets["teacher_action_test"])

    if v165.OPEN_EVENTS.exists():
        opentt_priors, _ = v165.estimate_opentt_aux_priors(pd.read_csv(v165.OPEN_EVENTS))
        opentt_action_oof = v165.opentt_action_prior(rows, base_action_oof, opentt_priors)
        opentt_action_test = v165.opentt_action_prior(test_rows, base_action_test, opentt_priors)
    else:
        opentt_action_oof = base_action_oof.copy()
        opentt_action_test = base_action_test.copy()

    coachai_data = v165.load_coachai_sequences()
    coachai_priors, _ = v165.build_coachai_transition_priors(coachai_data)
    coachai_action_oof = v165.coachai_family_prior_for_rows(rows, coachai_priors, prefix["next_actionId"])
    coachai_action_test = v165.coachai_family_prior_for_rows(test_rows, coachai_priors, prefix["next_actionId"])
    external_action_oof = v173.weighted_mix([(0.55, opentt_action_oof), (0.45, coachai_action_oof)])
    external_action_test = v173.weighted_mix([(0.55, opentt_action_test), (0.45, coachai_action_test)])

    internal_action_oof, _ = v160.foldsafe_internal_priors(prefix, rows)
    test_internal = v160.build_test_internal_prefixes(test_raw)
    internal_action_test, _ = v160.full_internal_priors(prefix, test_rows, test_internal)

    best = pd.read_csv(OUTDIR.parent / "v173_external_curriculum_pretrain" / "v173_action_curriculum_search.csv").iloc[0]
    best_schedule = str(best["schedule"])
    best_alpha = float(best["alpha"])
    best_spec = next(rec for rec in component_weight_grid() if rec["name"] == best_schedule)
    best_teacher_oof = v173.weighted_mix(
        [
            (best_spec["external"], external_action_oof),
            (best_spec["internal"], internal_action_oof),
            (best_spec["teacher"], teacher_action_oof),
        ]
    )
    best_teacher_test = v173.weighted_mix(
        [
            (best_spec["external"], external_action_test),
            (best_spec["internal"], internal_action_test),
            (best_spec["teacher"], teacher_action_test),
        ]
    )
    v173_prob_oof = v173.row_log_blend(base_action_oof, best_teacher_oof, best_alpha)
    v173_prob_test = v173.row_log_blend(base_action_test, best_teacher_test, best_alpha)

    return {
        "rows": rows,
        "test_rows": test_rows,
        "rally_uids": rally_uids,
        "tuning": tuning,
        "prefix": prefix,
        "base_oof": base_action_oof,
        "base_test": base_action_test,
        "opentt_oof": normalize_probability_rows(opentt_action_oof),
        "opentt_test": normalize_probability_rows(opentt_action_test),
        "coachai_oof": normalize_probability_rows(coachai_action_oof),
        "coachai_test": normalize_probability_rows(coachai_action_test),
        "external_oof": normalize_probability_rows(external_action_oof),
        "external_test": normalize_probability_rows(external_action_test),
        "internal_oof": normalize_probability_rows(internal_action_oof),
        "internal_test": normalize_probability_rows(internal_action_test),
        "teacher_oof": teacher_action_oof,
        "teacher_test": teacher_action_test,
        "v173_prob_oof": v173_prob_oof,
        "v173_prob_test": v173_prob_test,
        "best_schedule": best_schedule,
        "best_alpha": best_alpha,
        "best_candidate": str(best["candidate"]),
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_v173_action_sources()
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = rows["next_actionId"].astype(int).to_numpy()
    weights = context_weights(rows, test_rows)
    tuning = ctx["tuning"]

    base_pred = v173.action_pred(rows, ctx["base_oof"], tuning)
    base_test = v173.action_pred(test_rows, ctx["base_test"], tuning)
    v173_pred = v173.action_pred(rows, ctx["v173_prob_oof"], tuning)
    v173_test = v173.action_pred(test_rows, ctx["v173_prob_test"], tuning)
    submitted_v173 = load_sub(V173_SUBMITTED, ctx["rally_uids"])["actionId"].astype(int).to_numpy() if V173_SUBMITTED.exists() else v173_test
    submitted_churn = float(np.mean(submitted_v173 != v173_test))

    records = [evaluate_pred("v173_best_rebuild", y, v173_pred, base_pred, v173_pred, weights, {"kind": "anchor", "schedule": ctx["best_schedule"], "alpha": ctx["best_alpha"]})]
    prob_variants: dict[str, tuple[np.ndarray, np.ndarray]] = {"v173_best_rebuild": (ctx["v173_prob_oof"], ctx["v173_prob_test"])}

    component_sources = {
        "opentt_only": ("opentt_oof", "opentt_test"),
        "coachai_only": ("coachai_oof", "coachai_test"),
        "external_only": ("external_oof", "external_test"),
        "internal_only": ("internal_oof", "internal_test"),
        "r166_teacher_only": ("teacher_oof", "teacher_test"),
    }
    for name, (oof_key, test_key) in component_sources.items():
        for alpha in [0.05, 0.10, 0.15]:
            prob_oof = v173.row_log_blend(ctx["base_oof"], ctx[oof_key], alpha)
            prob_test = v173.row_log_blend(ctx["base_test"], ctx[test_key], alpha)
            pred = v173.action_pred(rows, prob_oof, tuning)
            test_pred = v173.action_pred(test_rows, prob_test, tuning)
            rec_name = f"v248_{name}_a{clean_float(alpha)}"
            records.append(
                evaluate_pred(
                    rec_name,
                    y,
                    pred,
                    base_pred,
                    v173_pred,
                    weights,
                    {
                        "kind": "component",
                        "component": name,
                        "alpha": alpha,
                        "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
                        "test_changed_vs_v173_rows": int(np.sum(test_pred != v173_test)),
                    },
                )
            )
            prob_variants[rec_name] = (prob_oof, prob_test)

    schedule_rows = []
    for spec in component_weight_grid():
        teacher_oof = v173.weighted_mix([(spec["external"], ctx["external_oof"]), (spec["internal"], ctx["internal_oof"]), (spec["teacher"], ctx["teacher_oof"])])
        teacher_test = v173.weighted_mix([(spec["external"], ctx["external_test"]), (spec["internal"], ctx["internal_test"]), (spec["teacher"], ctx["teacher_test"])])
        for alpha in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
            prob_oof = v173.row_log_blend(ctx["base_oof"], teacher_oof, alpha)
            prob_test = v173.row_log_blend(ctx["base_test"], teacher_test, alpha)
            pred = v173.action_pred(rows, prob_oof, tuning)
            test_pred = v173.action_pred(test_rows, prob_test, tuning)
            name = f"v248_schedule_{spec['name']}_a{clean_float(alpha)}"
            rec = evaluate_pred(
                name,
                y,
                pred,
                base_pred,
                v173_pred,
                weights,
                {
                    "kind": "schedule",
                    "schedule": spec["name"],
                    "alpha": alpha,
                    "external": spec["external"],
                    "internal": spec["internal"],
                    "teacher": spec["teacher"],
                    "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
                    "test_changed_vs_v173_rows": int(np.sum(test_pred != v173_test)),
                },
            )
            schedule_rows.append(rec)
            records.append(rec)
            prob_variants[name] = (prob_oof, prob_test)
    pd.DataFrame(schedule_rows).sort_values(["delta_vs_v173", "iw_delta_vs_v173"], ascending=[False, False]).to_csv(OUTDIR / "v248_schedule_surface.csv", index=False)

    changed = v173_pred != base_pred
    phase_specs = {
        "receive": ["receive"],
        "third_ball": ["third_ball"],
        "fourth_ball": ["fourth_ball"],
        "rally": ["rally"],
        "receive_third": ["receive", "third_ball"],
        "early": ["receive", "third_ball", "fourth_ball"],
    }
    phase_rows = []
    phase_test_actions = {}
    for name, phases in phase_specs.items():
        mask = acceptance_mask_by_phase(rows, changed, phases, phase_col="r184_phase")
        pred = compose(base_pred, v173_pred, mask)
        test_mask = acceptance_mask_by_phase(test_rows, v173_test != base_test, phases, phase_col="r184_phase")
        test_action = compose(base_test, v173_test, test_mask)
        rec_name = f"v248_phase_accept_{name}"
        rec = evaluate_pred(
            rec_name,
            y,
            pred,
            base_pred,
            v173_pred,
            weights,
            {
                "kind": "phase_accept",
                "accepted_phases": ",".join(phases),
                "test_churn_vs_v173": float(np.mean(test_action != v173_test)),
                "test_changed_vs_v173_rows": int(np.sum(test_action != v173_test)),
            },
        )
        phase_rows.append(rec)
        records.append(rec)
        phase_test_actions[rec_name] = test_action
    pd.DataFrame(phase_rows).sort_values(["delta_vs_v173", "iw_delta_vs_v173"], ascending=[False, False]).to_csv(OUTDIR / "v248_phase_ablation.csv", index=False)

    frame = pd.DataFrame(
        {
            "phase": rows["r184_phase"].astype(str),
            "base_action": base_pred.astype(int),
            "v173_action": v173_pred.astype(int),
            "y": y.astype(int),
            "changed": changed,
        }
    )
    changed_frame = frame[frame["changed"]].copy()
    transition = transition_counts(changed_frame, "phase", "base_action", "v173_action")
    detail = []
    for row in transition.itertuples(index=False):
        mask = changed_frame["phase"].eq(row.phase) & changed_frame["base_action"].eq(row.base_action) & changed_frame["v173_action"].eq(row.v173_action)
        part = changed_frame.loc[mask]
        base_correct = part["base_action"].eq(part["y"]).mean()
        v173_correct = part["v173_action"].eq(part["y"]).mean()
        detail.append(
            {
                "phase": row.phase,
                "base_action": int(row.base_action),
                "v173_action": int(row.v173_action),
                "rows": int(row.rows),
                "base_correct_rate": float(base_correct),
                "v173_correct_rate": float(v173_correct),
                "delta_correct_rate": float(v173_correct - base_correct),
                "correct_delta_rows": float((v173_correct - base_correct) * row.rows),
            }
        )
    transition_df = pd.DataFrame(detail).sort_values(["correct_delta_rows", "rows"], ascending=[False, False])
    transition_df.to_csv(OUTDIR / "v248_transition_attribution.csv", index=False)

    positive_pairs = transition_df[(transition_df["rows"].ge(3)) & (transition_df["delta_correct_rate"].gt(0))]
    pair_mask = np.zeros(len(rows), dtype=bool)
    pair_test_mask = np.zeros(len(test_rows), dtype=bool)
    for rec in positive_pairs.head(20).itertuples(index=False):
        pair_mask |= rows["r184_phase"].astype(str).eq(str(rec.phase)).to_numpy() & (base_pred == int(rec.base_action)) & (v173_pred == int(rec.v173_action))
        pair_test_mask |= test_rows["r184_phase"].astype(str).eq(str(rec.phase)).to_numpy() & (base_test == int(rec.base_action)) & (v173_test == int(rec.v173_action))
    pos_pair_pred = compose(base_pred, v173_pred, pair_mask)
    pos_pair_test = compose(base_test, v173_test, pair_test_mask)
    rec = evaluate_pred(
        "v248_transition_positive_pairs_top20",
        y,
        pos_pair_pred,
        base_pred,
        v173_pred,
        weights,
        {
            "kind": "transition_positive",
            "test_churn_vs_v173": float(np.mean(pos_pair_test != v173_test)),
            "test_changed_vs_v173_rows": int(np.sum(pos_pair_test != v173_test)),
        },
    )
    records.append(rec)

    search = pd.DataFrame(records).sort_values(["delta_vs_v173", "iw_delta_vs_v173", "action_macro_f1"], ascending=[False, False, False]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v248_candidate_search.csv", index=False)

    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    generated = []
    export_actions = {
        "v248_rebuilt_v173": v173_test,
        "v248_transition_positive_pairs_top20": pos_pair_test,
    }
    best_phase = pd.DataFrame(phase_rows).sort_values(["delta_vs_v173", "iw_delta_vs_v173"], ascending=[False, False]).iloc[0]
    best_phase_name = str(best_phase["candidate"])
    if best_phase_name in phase_test_actions:
        export_actions[best_phase_name] = phase_test_actions[best_phase_name]
    for name in search[search["kind"].isin(["schedule", "component"])].head(3)["candidate"]:
        prob_oof, prob_test = prob_variants[name]
        export_actions[name] = v173.action_pred(test_rows, prob_test, tuning)
    for name, action in export_actions.items():
        generated.append(write_submission(f"submission_{name}__pv188cap5__sr121.csv", ctx["rally_uids"], action, point, server))

    report = {
        "verdict": "DIAGNOSTIC_COMPLETE",
        "submitted_v173_churn_vs_rebuild": submitted_churn,
        "best_schedule": ctx["best_schedule"],
        "best_alpha": ctx["best_alpha"],
        "best_candidate": ctx["best_candidate"],
        "best_rows": search.head(12).to_dict(orient="records"),
        "top_positive_transitions": transition_df.head(15).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "v248_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "v248_report.md").write_text(
        "# V248 V173 Curriculum Decomposition\n\n"
        f"- Submitted V173 churn vs rebuilt V173 test action: `{submitted_churn:.6f}`\n"
        f"- Best original V173 schedule: `{ctx['best_schedule']}`, alpha `{ctx['best_alpha']}`\n"
        f"- Best search candidate: `{search.iloc[0]['candidate']}` delta_vs_v173 `{search.iloc[0]['delta_vs_v173']:.6f}`, IW `{search.iloc[0]['iw_delta_vs_v173']:.6f}`\n"
        "- Generated candidates are diagnostic unless delta_vs_v173 and IW are both clearly positive.\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v248_v173_curriculum_decomposition.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR), "submitted_churn": submitted_churn}, indent=2))


if __name__ == "__main__":
    main()
