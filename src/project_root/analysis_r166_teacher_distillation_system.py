"""R166 task-aligned teacher distillation system.

R166 turns the small but useful prior/proxy signals from earlier runs into
reusable teacher targets and tests conservative post-hoc distillation weights.

The script intentionally separates:
- safe/no-old action and point teacher distillation;
- sensitive old-server submission variants.

It does not append external rows to AICUP train and does not use hidden
test_new targets.  The saved teacher target arrays can be used by a later
PyTorch GRU/Transformer training script for true KL distillation.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parent
SRC_ANALYSIS = ROOT_DIR / "src" / "analysis"
for p in [ROOT_DIR, SRC_ANALYSIS]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import analysis_v165_combined_external_pretrain_proxy as v165  # noqa: E402
from analysis_r48_action_meta_stacker import (  # noqa: E402,F401
    GrUTuning,
    TransformerTuning,
    V3Tuning,
    build_current_oof_action,
)
from analysis_r67_r70_meta_priors import ARTIFACT_PATH, R63_OOF_PATH  # noqa: E402
from analysis_r151b_r154_physics_prior_integration import blend_nonterminal_point  # noqa: E402
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, validate_raw_data  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


OUTDIR = Path("r166_teacher_distillation")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_r166_teacher_distillation_system.py")

R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R121_MIN = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R119_POINT = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R154_POINT = UPLOAD_DIR / "submission_r154_md_mirror_a0p01_r67_anchor.csv"
R142_OLDHARD = UPLOAD_DIR / "submission_r142_r67_anchor_oldhard.csv"
R142_OLDSHARPEN = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"
R143_OLDSHARPEN_NEWSCORE = (
    UPLOAD_DIR / "submission_r143_r124_r67_r119_r121_oldsharpen005095_newscore_gapcal.csv"
)


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def onehot_smooth(labels: np.ndarray, classes: list[int], peak: float = 0.90) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.full((len(labels), len(classes)), (1.0 - peak) / (len(classes) - 1), dtype=float)
    idx = {c: i for i, c in enumerate(classes)}
    for i, value in enumerate(labels):
        if int(value) in idx:
            out[i, idx[int(value)]] = peak
    return normalize_rows(out)


def align_prob_by_meta(source_meta: pd.DataFrame, prob: np.ndarray, target_meta: pd.DataFrame) -> np.ndarray:
    key_cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId"]
    src = source_meta[key_cols].copy().reset_index().rename(columns={"index": "_src_idx"})
    dst = target_meta[key_cols].copy().reset_index().rename(columns={"index": "_dst_idx"})
    merged = dst.merge(src, on=key_cols, how="left", validate="one_to_one")
    if merged["_src_idx"].isna().any():
        missing = int(merged["_src_idx"].isna().sum())
        raise ValueError(f"Could not align {missing} probability rows by {key_cols}.")
    order = merged.sort_values("_dst_idx")["_src_idx"].astype(int).to_numpy()
    return prob[order]


def action_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return v165.action_pred(meta, prob, tuning)


def point_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return v165.point_pred(meta, prob, tuning)


def eval_action(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict) -> dict:
    y = meta["next_actionId"].astype(int).to_numpy()
    pred = action_pred(meta, prob, tuning)
    base = action_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)),
        "action_churn_vs_student": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=ACTION_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 5, 8, 9, 12, 14]:
        rec[f"f1_action_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_action_{k}"] = int(np.sum(pred == k))
    rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    base = point_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_student": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 1, 3, 5, 7, 8, 9]:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_point_{k}"] = int(np.sum(pred == k))
    rec.update(extra)
    return rec


def eval_combo(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    student_action: np.ndarray,
    student_point: np.ndarray,
    tuning,
    name: str,
    extra: dict,
) -> dict:
    action_y = meta["next_actionId"].astype(int).to_numpy()
    point_y = meta["next_pointId"].astype(int).to_numpy()
    server_y = meta["serverGetPoint"].astype(int).to_numpy()
    a_pred = action_pred(meta, action_prob, tuning)
    p_pred = point_pred(meta, point_prob, tuning)
    a_base = action_pred(meta, student_action, tuning)
    p_base = point_pred(meta, student_point, tuning)
    action_f1 = float(f1_score(action_y, a_pred, labels=ACTION_CLASSES, average="macro", zero_division=0))
    point_f1 = float(f1_score(point_y, p_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    server_auc = float(roc_auc_score(server_y, server_prob))
    rec = {
        "candidate": name,
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "server_auc": server_auc,
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
        "action_churn_vs_student": float(np.mean(a_pred != a_base)),
        "point_churn_vs_student": float(np.mean(p_pred != p_base)),
    }
    rec.update(extra)
    return rec


def load_submission(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(
        sub, on="rally_uid", how="left", validate="one_to_one"
    )
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"Submission did not align: {path}")
    return out


def write_submission(name: str, rally_uids: np.ndarray, action: np.ndarray, point: np.ndarray, server: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(
        {
            "rally_uid": rally_uids.astype(int),
            "actionId": action.astype(int),
            "pointId": point.astype(int),
            "serverGetPoint": np.round(np.clip(server.astype(float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"candidate": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_r67_style_action_teacher(rows: pd.DataFrame) -> np.ndarray:
    art = load_pickle(ARTIFACT_PATH)
    source_meta = art["valid_meta"].copy().reset_index(drop=True)
    current_oof = build_current_oof_action()
    v64_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r63_oof = np.load(R63_OOF_PATH)
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * v64_oof)
    r67_oof = normalize_rows(0.80 * r42_oof + 0.20 * r63_oof)
    return align_prob_by_meta(source_meta, r67_oof, rows)


def build_v165_best_components(rows: pd.DataFrame, test_rows: pd.DataFrame, base: dict) -> dict:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    _, _, prefix, _, _ = v165.prepare_prefix_features()

    internal_action_oof, internal_point_oof = v165.foldsafe_internal_priors(prefix, rows)
    test_internal = v165.build_test_internal_prefixes(test_raw)
    internal_action_test, internal_point_test = v165.full_internal_priors(prefix, test_rows, test_internal)

    if v165.OPEN_EVENTS.exists():
        opentt_priors, opentt_table = v165.estimate_opentt_aux_priors(pd.read_csv(v165.OPEN_EVENTS))
        opentt_table.to_csv(OUTDIR / "r166_opentt_prior_table.csv", index=False)
        opentt_action_oof = v165.opentt_action_prior(rows, base["action_oof"], opentt_priors)
        opentt_action_test = v165.opentt_action_prior(test_rows, base["action_test"], opentt_priors)
    else:
        opentt_priors = {"segments_count": 0, "transition_rows": 0}
        opentt_action_oof = base["action_oof"].copy()
        opentt_action_test = base["action_test"].copy()

    dm, md = v165.load_external_states()
    mpm_tables, mpm_global, _ = v165.build_mpm_tables(dm, mirror=True)
    mpm_oof = v165.mpm_prior_for_rows(rows, mpm_tables, mpm_global)
    mpm_test = v165.mpm_prior_for_rows(test_rows, mpm_tables, mpm_global)
    proto = v165.build_external_prototypes(dm, md, mirror=False)
    proto_oof = v165.prototype_prior_for_rows(rows, proto, tau=0.35, k=250)
    proto_test = v165.prototype_prior_for_rows(test_rows, proto, tau=0.35, k=250)

    coachai_data = v165.load_coachai_sequences()
    coachai_priors, coachai_stats = v165.build_coachai_transition_priors(coachai_data)
    coachai_stats.to_csv(OUTDIR / "r166_coachai_transition_family_stats.csv", index=False)
    coachai_family_oof = v165.coachai_family_prior_for_rows(rows, coachai_priors, prefix["next_actionId"])
    coachai_family_test = v165.coachai_family_prior_for_rows(test_rows, coachai_priors, prefix["next_actionId"])
    coachai_point_oof = {mode: v165.coachai_grid_prior_for_rows(rows, coachai_priors, mode) for mode in ["direct", "mirror", "avg"]}
    coachai_point_test = {mode: v165.coachai_grid_prior_for_rows(test_rows, coachai_priors, mode) for mode in ["direct", "mirror", "avg"]}

    action_rows = []
    action_probs: dict[str, dict[str, np.ndarray]] = {}
    for ai in [0.0, 0.0025, 0.005, 0.01, 0.02]:
        a_oof = base["action_oof"] if ai == 0 else v165.log_blend(base["action_oof"], internal_action_oof, ai)
        a_test = base["action_test"] if ai == 0 else v165.log_blend(base["action_test"], internal_action_test, ai)
        for ao in [0.0, 0.005, 0.01, 0.02, 0.03]:
            ao_oof = a_oof if ao == 0 else v165.blend_action(a_oof, opentt_action_oof, ao)
            ao_test = a_test if ao == 0 else v165.blend_action(a_test, opentt_action_test, ao)
            for ac in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03]:
                if ai == 0 and ao == 0 and ac == 0:
                    continue
                prob = ao_oof if ac == 0 else v165.blend_action(ao_oof, coachai_family_oof, ac)
                test_prob = ao_test if ac == 0 else v165.blend_action(ao_test, coachai_family_test, ac)
                name = f"v165_action_i{clean_float(ai)}_op{clean_float(ao)}_ca{clean_float(ac)}"
                rec = eval_action(
                    rows,
                    prob,
                    base["action_oof"],
                    base["tuning"],
                    name,
                    {"alpha_internal": ai, "alpha_opentt": ao, "alpha_coachai": ac},
                )
                action_rows.append(rec)
                action_probs[name] = {"oof": prob, "test": test_prob}
    action_search = pd.DataFrame(action_rows).sort_values("action_macro_f1", ascending=False).reset_index(drop=True)
    action_search.to_csv(OUTDIR / "r166_v165_component_action_search.csv", index=False)

    point_rows = []
    point_probs: dict[str, dict[str, np.ndarray]] = {}
    for ai in [0.0, 0.0025, 0.005, 0.01]:
        p_oof = base["point_oof"] if ai == 0 else blend_nonterminal_point(base["point_oof"], internal_point_oof[:, 1:10], ai)
        p_test = base["point_test"] if ai == 0 else blend_nonterminal_point(base["point_test"], internal_point_test[:, 1:10], ai)
        for am in [0.0, 0.0025, 0.005, 0.01]:
            pm_oof = p_oof if am == 0 else blend_nonterminal_point(p_oof, mpm_oof, am)
            pm_test = p_test if am == 0 else blend_nonterminal_point(p_test, mpm_test, am)
            for ap in [0.0, 0.001, 0.0025, 0.005]:
                pp_oof = pm_oof if ap == 0 else blend_nonterminal_point(pm_oof, proto_oof, ap)
                pp_test = pm_test if ap == 0 else blend_nonterminal_point(pm_test, proto_test, ap)
                for mode in ["direct", "mirror", "avg"]:
                    for ac in [0.0, 0.001, 0.0025, 0.005, 0.01]:
                        if ai == 0 and am == 0 and ap == 0 and ac == 0:
                            continue
                        prob = pp_oof if ac == 0 else blend_nonterminal_point(pp_oof, coachai_point_oof[mode], ac)
                        test_prob = pp_test if ac == 0 else blend_nonterminal_point(pp_test, coachai_point_test[mode], ac)
                        name = (
                            f"v165_point_i{clean_float(ai)}_m{clean_float(am)}_p{clean_float(ap)}"
                            f"_ca{mode}{clean_float(ac)}"
                        )
                        rec = eval_point(
                            rows,
                            prob,
                            base["point_oof"],
                            base["tuning"],
                            name,
                            {"alpha_internal": ai, "alpha_mpm": am, "alpha_proto": ap, "coachai_mode": mode, "alpha_coachai": ac},
                        )
                        point_rows.append(rec)
                        point_probs[name] = {"oof": prob, "test": test_prob}
    point_search = pd.DataFrame(point_rows).sort_values("point_macro_f1", ascending=False).reset_index(drop=True)
    point_search.to_csv(OUTDIR / "r166_v165_component_point_search.csv", index=False)

    return {
        "opentt": opentt_priors,
        "coachai": {
            "canonical_rows": int(len(coachai_data)),
            "transition_rows": int(coachai_priors["rows"]),
            "sequences": int(coachai_priors["sequences"]),
        },
        "best_action_name": str(action_search.iloc[0]["candidate"]),
        "best_point_name": str(point_search.iloc[0]["candidate"]),
        "best_action_oof": action_probs[str(action_search.iloc[0]["candidate"])]["oof"],
        "best_action_test": action_probs[str(action_search.iloc[0]["candidate"])]["test"],
        "best_point_oof": point_probs[str(point_search.iloc[0]["candidate"])]["oof"],
        "best_point_test": point_probs[str(point_search.iloc[0]["candidate"])]["test"],
        "action_search": action_search,
        "point_search": point_search,
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    _, _, prefix, test_prefix, _ = v165.prepare_prefix_features()

    r111_oof = load_pickle(v165.R111_OOF)
    r111_test = load_pickle(v165.R111_TEST)
    r101_oof = load_pickle(v165.R101_OOF)
    r101_test = load_pickle(v165.R101_TEST)
    tuning = r111_oof["tuning"]

    valid_meta = v165.ensure_fold(r111_oof["valid_meta"])
    rows = v165.align_prefix_meta(valid_meta, prefix).reset_index(drop=True)
    test_rows = test_prefix.reset_index(drop=True)
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])
    base_point_oof = normalize_rows(0.50 * r111_oof["gru_point"] + 0.50 * r101_oof["gru_point"])
    base_point_test = normalize_rows(0.50 * r111_test["gru_point"] + 0.50 * r101_test["gru_point"])
    base_server_oof = 0.5 * r111_oof["gru_server"] + 0.5 * r101_oof["gru_server"]
    base_server_test = 0.5 * r111_test["gru_server"] + 0.5 * r101_test["gru_server"]

    base = {
        "action_oof": base_action_oof,
        "action_test": base_action_test,
        "point_oof": base_point_oof,
        "point_test": base_point_test,
        "server_oof": base_server_oof,
        "server_test": base_server_test,
        "tuning": tuning,
    }
    v165_components = build_v165_best_components(rows, test_rows, base)

    r67_action_oof = build_r67_style_action_teacher(rows)
    r67_sub = load_submission(R67_ANCHOR, rally_uids)
    r121_sub = load_submission(R121_MIN, rally_uids) if R121_MIN.exists() else r67_sub
    r119_sub = load_submission(R119_POINT, rally_uids) if R119_POINT.exists() else r67_sub
    r154_sub = load_submission(R154_POINT, rally_uids) if R154_POINT.exists() else r67_sub

    r67_action_test = onehot_smooth(r67_sub["actionId"].to_numpy(), ACTION_CLASSES, peak=0.90)
    r119_point_test = onehot_smooth(r119_sub["pointId"].to_numpy(), POINT_CLASSES, peak=0.88)
    r154_point_test = onehot_smooth(r154_sub["pointId"].to_numpy(), POINT_CLASSES, peak=0.88)

    # R166 teacher targets.  Action is style/public-anchor heavy; point is
    # deliberately low-churn and keeps neural/base probability dominant.
    teacher_action_oof = normalize_rows(
        0.52 * r67_action_oof
        + 0.28 * v165_components["best_action_oof"]
        + 0.12 * r111_oof["gru_action"]
        + 0.08 * r101_oof["gru_action"]
    )
    teacher_action_test = normalize_rows(
        0.52 * r67_action_test
        + 0.28 * v165_components["best_action_test"]
        + 0.12 * r111_test["gru_action"]
        + 0.08 * r101_test["gru_action"]
    )
    teacher_point_oof = normalize_rows(
        0.58 * base_point_oof
        + 0.34 * v165_components["best_point_oof"]
        + 0.04 * r111_oof["gru_point"]
        + 0.04 * r101_oof["gru_point"]
    )
    teacher_point_test = normalize_rows(
        0.52 * base_point_test
        + 0.30 * v165_components["best_point_test"]
        + 0.09 * r119_point_test
        + 0.09 * r154_point_test
    )
    teacher_server_oof = np.clip(0.60 * base_server_oof + 0.25 * r111_oof["gru_server"] + 0.15 * r101_oof["gru_server"], 1e-6, 1 - 1e-6)
    teacher_server_test = np.clip(0.50 * base_server_test + 0.50 * r121_sub["serverGetPoint"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)

    np.savez_compressed(
        OUTDIR / "r166_teacher_targets.npz",
        rally_uid_oof=rows["rally_uid"].astype(int).to_numpy(),
        prefix_len_oof=rows["prefix_len"].astype(int).to_numpy(),
        rally_uid_test=rally_uids.astype(int),
        teacher_action_oof=teacher_action_oof,
        teacher_action_test=teacher_action_test,
        teacher_point_oof=teacher_point_oof,
        teacher_point_test=teacher_point_test,
        teacher_server_oof=teacher_server_oof,
        teacher_server_test=teacher_server_test,
    )
    rows[["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint", "match", "fold"]].to_csv(
        OUTDIR / "r166_teacher_oof_meta.csv", index=False
    )

    student_sources = {
        "r101": {
            "action_oof": r101_oof["gru_action"],
            "action_test": r101_test["gru_action"],
            "point_oof": r101_oof["gru_point"],
            "point_test": r101_test["gru_point"],
            "server_oof": r101_oof["gru_server"],
            "server_test": r101_test["gru_server"],
        },
        "r111": {
            "action_oof": r111_oof["gru_action"],
            "action_test": r111_test["gru_action"],
            "point_oof": r111_oof["gru_point"],
            "point_test": r111_test["gru_point"],
            "server_oof": r111_oof["gru_server"],
            "server_test": r111_test["gru_server"],
        },
        "r111_r101_base": {
            "action_oof": base_action_oof,
            "action_test": base_action_test,
            "point_oof": base_point_oof,
            "point_test": base_point_test,
            "server_oof": base_server_oof,
            "server_test": base_server_test,
        },
        "v165_best": {
            "action_oof": v165_components["best_action_oof"],
            "action_test": v165_components["best_action_test"],
            "point_oof": v165_components["best_point_oof"],
            "point_test": v165_components["best_point_test"],
            "server_oof": base_server_oof,
            "server_test": base_server_test,
        },
    }

    action_rows = []
    point_rows = []
    combo_rows = []
    action_candidates: dict[str, dict[str, np.ndarray]] = {}
    point_candidates: dict[str, dict[str, np.ndarray]] = {}
    server_candidates: dict[str, dict[str, np.ndarray]] = {}
    for student_name, src in student_sources.items():
        for wa in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
            prob = src["action_oof"] if wa == 0 else v165.log_blend(src["action_oof"], teacher_action_oof, wa)
            test_prob = src["action_test"] if wa == 0 else v165.log_blend(src["action_test"], teacher_action_test, wa)
            name = f"r166_action_{student_name}_wa{clean_float(wa)}"
            action_rows.append(eval_action(rows, prob, src["action_oof"], tuning, name, {"student": student_name, "wa": wa}))
            action_candidates[name] = {"oof": prob, "test": test_prob, "student": student_name, "weight": wa}
        for wp in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
            prob = src["point_oof"] if wp == 0 else blend_nonterminal_point(src["point_oof"], teacher_point_oof[:, 1:10], wp)
            test_prob = src["point_test"] if wp == 0 else blend_nonterminal_point(src["point_test"], teacher_point_test[:, 1:10], wp)
            name = f"r166_point_{student_name}_wp{clean_float(wp)}"
            point_rows.append(eval_point(rows, prob, src["point_oof"], tuning, name, {"student": student_name, "wp": wp}))
            point_candidates[name] = {"oof": prob, "test": test_prob, "student": student_name, "weight": wp}
        for ws in [0.0, 0.02, 0.05, 0.10]:
            prob = src["server_oof"] if ws == 0 else np.clip((1 - ws) * src["server_oof"] + ws * teacher_server_oof, 1e-6, 1 - 1e-6)
            test_prob = src["server_test"] if ws == 0 else np.clip((1 - ws) * src["server_test"] + ws * teacher_server_test, 1e-6, 1 - 1e-6)
            name = f"r166_server_{student_name}_ws{clean_float(ws)}"
            server_candidates[name] = {"oof": prob, "test": test_prob, "student": student_name, "weight": ws}

    action_search = pd.DataFrame(action_rows).sort_values(["action_macro_f1", "action_churn_vs_student"], ascending=[False, True]).reset_index(drop=True)
    point_search = pd.DataFrame(point_rows).sort_values(["point_macro_f1", "point_churn_vs_student"], ascending=[False, True]).reset_index(drop=True)
    action_search.to_csv(OUTDIR / "r166_action_distill_search.csv", index=False)
    point_search.to_csv(OUTDIR / "r166_point_distill_search.csv", index=False)

    top_action_names = list(action_search.head(8)["candidate"])
    top_point_names = list(point_search.head(8)["candidate"])
    top_server_names = [k for k in server_candidates if k.endswith("ws0p0")]
    for a_name in top_action_names:
        for p_name in top_point_names:
            for s_name in top_server_names:
                combo_rows.append(
                    eval_combo(
                        rows,
                        action_candidates[a_name]["oof"],
                        point_candidates[p_name]["oof"],
                        server_candidates[s_name]["oof"],
                        student_sources[action_candidates[a_name]["student"]]["action_oof"],
                        student_sources[point_candidates[p_name]["student"]]["point_oof"],
                        tuning,
                        f"{a_name}__{p_name}__{s_name}",
                        {
                            "action_candidate": a_name,
                            "point_candidate": p_name,
                            "server_candidate": s_name,
                        },
                    )
                )
    combo_search = pd.DataFrame(combo_rows).sort_values(["overall", "point_churn_vs_student"], ascending=[False, True]).reset_index(drop=True)
    combo_search.to_csv(OUTDIR / "r166_combo_search.csv", index=False)

    best_action_name = str(action_search.iloc[0]["candidate"])
    best_point_name = str(point_search.iloc[0]["candidate"])
    best_combo = combo_search.iloc[0]
    best_combo_action = str(best_combo["action_candidate"])
    best_combo_point = str(best_combo["point_candidate"])
    best_combo_server = str(best_combo["server_candidate"])

    action_sources = {
        "r67_public": r67_sub["actionId"].astype(int).to_numpy(),
        "r166_best_action": action_pred(test_rows, action_candidates[best_action_name]["test"], tuning),
        "r166_combo_action": action_pred(test_rows, action_candidates[best_combo_action]["test"], tuning),
    }
    point_sources = {
        "r67_v3": r67_sub["pointId"].astype(int).to_numpy(),
        "r119_public_point": r119_sub["pointId"].astype(int).to_numpy(),
        "r154_safe_physics": r154_sub["pointId"].astype(int).to_numpy(),
        "r166_best_point": point_pred(test_rows, point_candidates[best_point_name]["test"], tuning),
        "r166_combo_point": point_pred(test_rows, point_candidates[best_combo_point]["test"], tuning),
    }
    server_sources = {
        "r121_min_w0p2": r121_sub["serverGetPoint"].astype(float).to_numpy(),
        "r67_current_server": r67_sub["serverGetPoint"].astype(float).to_numpy(),
        "r166_safe_teacher": server_candidates[best_combo_server]["test"],
    }
    sensitive_sources = {
        "oldhard": R142_OLDHARD,
        "oldsharpen005095": R142_OLDSHARPEN,
        "oldsharpen005095_newscore_gapcal": R143_OLDSHARPEN_NEWSCORE,
    }
    for key, path in sensitive_sources.items():
        if path.exists():
            server_sources[key] = load_submission(path, rally_uids)["serverGetPoint"].astype(float).to_numpy()

    generated = []
    combos = [
        ("r166_best_action", "r166_best_point", "r121_min_w0p2"),
        ("r166_combo_action", "r166_combo_point", "r121_min_w0p2"),
        ("r67_public", "r166_best_point", "r121_min_w0p2"),
        ("r166_best_action", "r119_public_point", "r121_min_w0p2"),
        ("r67_public", "r119_public_point", "r121_min_w0p2"),
        ("r166_combo_action", "r166_combo_point", "oldsharpen005095"),
        ("r166_combo_action", "r166_combo_point", "oldhard"),
        ("r166_combo_action", "r166_combo_point", "oldsharpen005095_newscore_gapcal"),
    ]
    for a_key, p_key, s_key in combos:
        if s_key not in server_sources:
            continue
        name = f"submission_r166__a{a_key}__p{p_key}__s{s_key}.csv"
        info = write_submission(name, rally_uids, action_sources[a_key], point_sources[p_key], server_sources[s_key])
        info.update({"action_source": a_key, "point_source": p_key, "server_source": s_key})
        generated.append(info)

    base_action_metric = float(
        f1_score(rows["next_actionId"].astype(int), action_pred(rows, base_action_oof, tuning), labels=ACTION_CLASSES, average="macro", zero_division=0)
    )
    base_point_metric = float(
        f1_score(rows["next_pointId"].astype(int), point_pred(rows, base_point_oof, tuning), labels=POINT_CLASSES, average="macro", zero_division=0)
    )
    base_server_auc = float(roc_auc_score(rows["serverGetPoint"].astype(int), base_server_oof))
    summary = {
        "safety": {
            "main_teacher_uses_old_test_server_labels": False,
            "main_teacher_uses_test_new_hidden_targets": False,
            "external_rows_appended_to_train": False,
            "sensitive_submissions_use_r142_server": True,
        },
        "base_metrics": {
            "base_action_macro_f1": base_action_metric,
            "base_point_macro_f1": base_point_metric,
            "base_server_auc": base_server_auc,
            "base_overall": float(0.4 * base_action_metric + 0.4 * base_point_metric + 0.2 * base_server_auc),
        },
        "v165_components": {
            "best_action_name": v165_components["best_action_name"],
            "best_point_name": v165_components["best_point_name"],
            "coachai": v165_components["coachai"],
            "opentt": {
                "segments_count": int(v165_components["opentt"].get("segments_count", 0)),
                "transition_rows": int(v165_components["opentt"].get("transition_rows", 0)),
            },
        },
        "best_action": action_search.head(10).to_dict(orient="records"),
        "best_point": point_search.head(10).to_dict(orient="records"),
        "best_combo": combo_search.head(10).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "r166_report.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r166_report.md").write_text(
        "# R166 Teacher Distillation System\n\n"
        "## Scope\n\n"
        "- Safe teacher targets do not use old-test server labels.\n"
        "- External datasets are used as priors/teachers only; no external rows are appended to AICUP train.\n"
        "- Sensitive submissions only swap in existing R142/R143 server columns.\n\n"
        "## Base Metrics\n\n"
        f"- Base action Macro-F1: `{base_action_metric:.6f}`\n"
        f"- Base point Macro-F1: `{base_point_metric:.6f}`\n"
        f"- Base server AUC: `{base_server_auc:.6f}`\n\n"
        "## Best Distillation Results\n\n"
        f"- Best action: `{best_action_name}` = `{action_search.iloc[0]['action_macro_f1']:.6f}`\n"
        f"- Best point: `{best_point_name}` = `{point_search.iloc[0]['point_macro_f1']:.6f}`\n"
        f"- Best combo: `{best_combo['candidate']}` = overall `{best_combo['overall']:.6f}`\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )

    with open("experiments_log.md", "a", encoding="utf-8") as f:
        f.write(
            "\n\n## R166 teacher distillation system\n\n"
            "- Implemented reusable teacher targets: "
            "`r166_teacher_distillation/r166_teacher_targets.npz`.\n"
            f"- Base action/point/server: {base_action_metric:.6f} / {base_point_metric:.6f} / {base_server_auc:.6f}.\n"
            f"- Best action: `{best_action_name}` = {action_search.iloc[0]['action_macro_f1']:.6f}, "
            f"churn {action_search.iloc[0]['action_churn_vs_student']:.4%}.\n"
            f"- Best point: `{best_point_name}` = {point_search.iloc[0]['point_macro_f1']:.6f}, "
            f"churn {point_search.iloc[0]['point_churn_vs_student']:.4%}.\n"
            f"- Best combo overall: `{best_combo['candidate']}` = {best_combo['overall']:.6f}.\n"
            "- Generated safe/no-old and sensitive/R142 server candidates under "
            "`upload_candidates_20260519/` and `submissions/selected/`.\n"
        )

    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
