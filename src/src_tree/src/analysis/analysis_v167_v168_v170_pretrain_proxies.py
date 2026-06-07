"""V167/V168/V170 task-aligned pretraining proxies.

These experiments consume R166 teacher targets and test three follow-up ideas:

V167: conditional-style action absorption with row-wise trust gates.
V168: physics/nonterminal point absorption with point0 preserved.
V170: rare/control action residual for classes 8/9/12/14.

This is a proxy layer, not full GPU retraining.  It is deliberately designed to
answer whether each pretraining direction has enough OOF signal to justify a
full GRU/Transformer KL training run.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from dataclasses import dataclass
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
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, validate_raw_data  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


OUTDIR = Path("v167_v168_v170_pretrain_proxies")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v167_v168_v170_pretrain_proxies.py")

R166_TARGETS = Path("r166_teacher_distillation/r166_teacher_targets.npz")
R166_META = Path("r166_teacher_distillation/r166_teacher_oof_meta.csv")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R119_POINT = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R121_MIN = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R154_POINT = UPLOAD_DIR / "submission_r154_md_mirror_a0p01_r67_anchor.csv"
R142_OLDSHARPEN = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"

RARE_CLASSES = [8, 9, 12, 14]


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def top_margin(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(prob, axis=1)
    top = order[:, -1]
    top1 = prob[np.arange(len(prob)), top]
    top2 = prob[np.arange(len(prob)), order[:, -2]]
    return top.astype(int), top1, top1 - top2


def row_log_blend(base: np.ndarray, prior: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    base = normalize_rows(np.clip(base, 1e-9, None))
    prior = normalize_rows(np.clip(prior, 1e-9, None))
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        a = np.full(len(base), float(a), dtype=float)
    a = np.clip(a, 0.0, 1.0)[:, None]
    out = np.exp((1.0 - a) * np.log(base) + a * np.log(prior))
    return normalize_rows(out)


def row_nonterminal_blend(base: np.ndarray, prior: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    base = normalize_rows(np.clip(base, 1e-9, None))
    prior = normalize_rows(np.clip(prior, 1e-9, None))
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        a = np.full(len(base), float(a), dtype=float)
    a = np.clip(a, 0.0, 1.0)[:, None]
    out = base.copy()
    mass = np.clip(1.0 - out[:, 0], 1e-9, 1.0)
    b9 = normalize_rows(out[:, 1:10] / mass[:, None])
    p9 = normalize_rows(prior[:, 1:10])
    q = np.exp((1.0 - a) * np.log(np.clip(b9, 1e-9, None)) + a * np.log(np.clip(p9, 1e-9, None)))
    q = normalize_rows(q)
    out[:, 1:10] = mass[:, None] * q
    return normalize_rows(out)


def rare_residual(base: np.ndarray, teacher: np.ndarray, alpha: float, mode: str, gate: np.ndarray | None = None) -> np.ndarray:
    base = normalize_rows(np.clip(base, 1e-9, None))
    teacher = normalize_rows(np.clip(teacher, 1e-9, None))
    logit = np.log(base)
    residual = np.log(teacher) - np.log(base)
    if mode == "positive_only":
        residual = np.maximum(residual, 0.0)
    elif mode == "signed":
        residual = np.clip(residual, -2.0, 2.0)
    elif mode == "teacher_ranked":
        residual = np.where(teacher > base, np.maximum(residual, 0.0), 0.0)
    else:
        raise ValueError(mode)
    boost = np.zeros_like(logit)
    for cls in RARE_CLASSES:
        boost[:, ACTION_CLASSES.index(cls)] = residual[:, ACTION_CLASSES.index(cls)]
    if gate is not None:
        boost *= gate[:, None]
    return normalize_rows(np.exp(logit + alpha * boost))


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
        "action_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=ACTION_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 5, 8, 9, 10, 11, 12, 13, 14]:
        rec[f"f1_action_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_action_{k}"] = int(np.sum(pred == k))
    rare_f1 = [float(report[str(k)]["f1-score"]) for k in RARE_CLASSES]
    rec["rare_action_mean_f1"] = float(np.mean(rare_f1))
    rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    base = point_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 1, 3, 5, 7, 8, 9]:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_point_{k}"] = int(np.sum(pred == k))
    rec.update(extra)
    return rec


def eval_combo(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, server: np.ndarray, tuning, name: str) -> dict:
    action_f1 = float(
        f1_score(
            meta["next_actionId"].astype(int),
            action_pred(meta, action_prob, tuning),
            labels=ACTION_CLASSES,
            average="macro",
            zero_division=0,
        )
    )
    point_f1 = float(
        f1_score(
            meta["next_pointId"].astype(int),
            point_pred(meta, point_prob, tuning),
            labels=POINT_CLASSES,
            average="macro",
            zero_division=0,
        )
    )
    server_auc = float(roc_auc_score(meta["serverGetPoint"].astype(int), server))
    return {
        "candidate": name,
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "server_auc": server_auc,
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def load_submission(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
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


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)

    targets = np.load(R166_TARGETS)
    rows = pd.read_csv(R166_META)
    r111_oof = load_pickle(v165.R111_OOF)
    r111_test = load_pickle(v165.R111_TEST)
    r101_oof = load_pickle(v165.R101_OOF)
    r101_test = load_pickle(v165.R101_TEST)
    tuning = r111_oof["tuning"]
    test_rows = pd.read_csv("test_new.csv").groupby("rally_uid", sort=False).tail(1).reset_index(drop=True)
    # Existing neural test_meta is already the safest alignment source.
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()
    test_rows = r111_test["test_meta"].copy().reset_index(drop=True)

    teacher_action_oof = targets["teacher_action_oof"]
    teacher_action_test = targets["teacher_action_test"]
    teacher_point_oof = targets["teacher_point_oof"]
    teacher_point_test = targets["teacher_point_test"]

    students = {
        "r111": {
            "action_oof": r111_oof["gru_action"],
            "action_test": r111_test["gru_action"],
            "point_oof": r111_oof["gru_point"],
            "point_test": r111_test["gru_point"],
            "server_oof": r111_oof["gru_server"],
            "server_test": r111_test["gru_server"],
        },
        "r101": {
            "action_oof": r101_oof["gru_action"],
            "action_test": r101_test["gru_action"],
            "point_oof": r101_oof["gru_point"],
            "point_test": r101_test["gru_point"],
            "server_oof": r101_oof["gru_server"],
            "server_test": r101_test["gru_server"],
        },
        "r111_r101_base": {
            "action_oof": normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"]),
            "action_test": normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"]),
            "point_oof": normalize_rows(0.50 * r111_oof["gru_point"] + 0.50 * r101_oof["gru_point"]),
            "point_test": normalize_rows(0.50 * r111_test["gru_point"] + 0.50 * r101_test["gru_point"]),
            "server_oof": 0.5 * r111_oof["gru_server"] + 0.5 * r101_oof["gru_server"],
            "server_test": 0.5 * r111_test["gru_server"] + 0.5 * r101_test["gru_server"],
        },
    }

    # V167: conditional-style row-wise teacher absorption.
    v167_rows = []
    v167_probs: dict[str, dict[str, np.ndarray]] = {}
    teacher_top, teacher_conf, teacher_margin = top_margin(teacher_action_oof)
    for student_name, src in students.items():
        base_top, base_conf, base_margin = top_margin(src["action_oof"])
        _, teacher_conf_test, teacher_margin_test = top_margin(teacher_action_test)
        _, base_conf_test, base_margin_test = top_margin(src["action_test"])
        for w in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
            gates = {
                "global": np.full(len(rows), w),
                "margin_gt0p08": w * (teacher_margin > 0.08).astype(float),
                "margin_gt0p12": w * (teacher_margin > 0.12).astype(float),
                "teacher_conf_gt_base": w * (teacher_conf > base_conf).astype(float),
                "same_top_or_high_margin": w * ((teacher_top == base_top) | (teacher_margin > 0.18)).astype(float),
                "low_churn_style": w * ((teacher_margin > 0.10) & (teacher_conf >= base_conf - 0.02)).astype(float),
            }
            test_gates = {
                "global": np.full(len(test_rows), w),
                "margin_gt0p08": w * (teacher_margin_test > 0.08).astype(float),
                "margin_gt0p12": w * (teacher_margin_test > 0.12).astype(float),
                "teacher_conf_gt_base": w * (teacher_conf_test > base_conf_test).astype(float),
                "same_top_or_high_margin": np.full(len(test_rows), w),  # hard R67-smoothed teacher makes same-top unavailable.
                "low_churn_style": w * ((teacher_margin_test > 0.10) & (teacher_conf_test >= base_conf_test - 0.02)).astype(float),
            }
            for mode, alpha in gates.items():
                prob = row_log_blend(src["action_oof"], teacher_action_oof, alpha)
                test_prob = row_log_blend(src["action_test"], teacher_action_test, test_gates[mode])
                name = f"v167_{student_name}_{mode}_w{clean_float(w)}"
                v167_rows.append(eval_action(rows, prob, src["action_oof"], tuning, name, {"student": student_name, "mode": mode, "w": w}))
                v167_probs[name] = {"oof": prob, "test": test_prob, "student": student_name}
    v167_search = pd.DataFrame(v167_rows).sort_values(["action_macro_f1", "action_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    v167_search.to_csv(OUTDIR / "v167_style_absorption_search.csv", index=False)

    # V168: conservative nonterminal point teacher absorption.
    v168_rows = []
    v168_probs: dict[str, dict[str, np.ndarray]] = {}
    for student_name, src in students.items():
        p0 = src["point_oof"][:, 0]
        long_mass = src["point_oof"][:, 7:10].sum(axis=1)
        p0_test = src["point_test"][:, 0]
        long_mass_test = src["point_test"][:, 7:10].sum(axis=1)
        for w in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]:
            gates = {
                "global": np.full(len(rows), w),
                "nonterminal": w * (p0 < 0.45).astype(float),
                "longside": w * ((p0 < 0.45) & (long_mass > 0.35)).astype(float),
                "longside_strict": w * ((p0 < 0.40) & (long_mass > 0.45)).astype(float),
            }
            test_gates = {
                "global": np.full(len(test_rows), w),
                "nonterminal": w * (p0_test < 0.45).astype(float),
                "longside": w * ((p0_test < 0.45) & (long_mass_test > 0.35)).astype(float),
                "longside_strict": w * ((p0_test < 0.40) & (long_mass_test > 0.45)).astype(float),
            }
            for mode, alpha in gates.items():
                prob = row_nonterminal_blend(src["point_oof"], teacher_point_oof, alpha)
                test_prob = row_nonterminal_blend(src["point_test"], teacher_point_test, test_gates[mode])
                name = f"v168_{student_name}_{mode}_w{clean_float(w)}"
                v168_rows.append(eval_point(rows, prob, src["point_oof"], tuning, name, {"student": student_name, "mode": mode, "w": w}))
                v168_probs[name] = {"oof": prob, "test": test_prob, "student": student_name}
    v168_search = pd.DataFrame(v168_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    v168_search.to_csv(OUTDIR / "v168_physics_point_search.csv", index=False)

    # V170: rare/control class-specific residual.
    v170_rows = []
    v170_probs: dict[str, dict[str, np.ndarray]] = {}
    for student_name, src in students.items():
        _, base_conf, _ = top_margin(src["action_oof"])
        _, teacher_conf, teacher_margin = top_margin(teacher_action_oof)
        _, base_conf_test, _ = top_margin(src["action_test"])
        _, teacher_conf_test, teacher_margin_test = top_margin(teacher_action_test)
        gates = {
            "none": np.ones(len(rows)),
            "teacher_conf_gt_base": (teacher_conf > base_conf).astype(float),
            "teacher_margin_gt0p1": (teacher_margin > 0.10).astype(float),
            "rare_teacher_top": np.isin(teacher_top, [ACTION_CLASSES.index(k) for k in RARE_CLASSES]).astype(float),
        }
        test_gates = {
            "none": np.ones(len(test_rows)),
            "teacher_conf_gt_base": (teacher_conf_test > base_conf_test).astype(float),
            "teacher_margin_gt0p1": (teacher_margin_test > 0.10).astype(float),
            "rare_teacher_top": np.ones(len(test_rows)),
        }
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            for mode in ["positive_only", "signed", "teacher_ranked"]:
                for gate_name, gate in gates.items():
                    prob = rare_residual(src["action_oof"], teacher_action_oof, alpha, mode, gate)
                    test_prob = rare_residual(src["action_test"], teacher_action_test, alpha, mode, test_gates[gate_name])
                    name = f"v170_{student_name}_{mode}_{gate_name}_a{clean_float(alpha)}"
                    v170_rows.append(
                        eval_action(
                            rows,
                            prob,
                            src["action_oof"],
                            tuning,
                            name,
                            {"student": student_name, "mode": mode, "gate": gate_name, "alpha": alpha},
                        )
                    )
                    v170_probs[name] = {"oof": prob, "test": test_prob, "student": student_name}
    v170_search = pd.DataFrame(v170_rows).sort_values(["action_macro_f1", "rare_action_mean_f1"], ascending=[False, False]).reset_index(drop=True)
    v170_search.to_csv(OUTDIR / "v170_rare_control_residual_search.csv", index=False)

    server = students["r111_r101_base"]["server_oof"]
    combo_rows = []
    for a_name in list(v167_search.head(5)["candidate"]) + list(v170_search.head(5)["candidate"]):
        aprob = v167_probs[a_name]["oof"] if a_name in v167_probs else v170_probs[a_name]["oof"]
        for p_name in list(v168_search.head(5)["candidate"]):
            combo_rows.append(eval_combo(rows, aprob, v168_probs[p_name]["oof"], server, tuning, f"{a_name}__{p_name}"))
    combo_search = pd.DataFrame(combo_rows).sort_values("overall", ascending=False).reset_index(drop=True)
    combo_search.to_csv(OUTDIR / "v167_v168_v170_combo_search.csv", index=False)

    r67_sub = load_submission(R67_ANCHOR, rally_uids)
    r119_sub = load_submission(R119_POINT, rally_uids)
    r154_sub = load_submission(R154_POINT, rally_uids)
    r121_sub = load_submission(R121_MIN, rally_uids)
    oldsharp_sub = load_submission(R142_OLDSHARPEN, rally_uids) if R142_OLDSHARPEN.exists() else None

    best_v167 = str(v167_search.iloc[0]["candidate"])
    best_v168 = str(v168_search.iloc[0]["candidate"])
    best_v170 = str(v170_search.iloc[0]["candidate"])
    best_combo = combo_search.iloc[0]
    combo_action_name = str(best_combo["candidate"]).split("__")[0]
    combo_point_name = str(best_combo["candidate"]).split("__")[1]

    def action_from(name: str) -> np.ndarray:
        if name in v167_probs:
            return action_pred(test_rows, v167_probs[name]["test"], tuning)
        return action_pred(test_rows, v170_probs[name]["test"], tuning)

    def point_from(name: str) -> np.ndarray:
        return point_pred(test_rows, v168_probs[name]["test"], tuning)

    generated = []
    candidates = [
        ("v167_best", action_from(best_v167), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("v168_point_probe", r67_sub["actionId"].astype(int).to_numpy(), "v168_best", point_from(best_v168), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("v170_best", action_from(best_v170), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("combo_best", action_from(combo_action_name), "combo_point", point_from(combo_point_name), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("v167_best", action_from(best_v167), "r154_safe_physics", r154_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
    ]
    if oldsharp_sub is not None:
        candidates.append(("combo_best", action_from(combo_action_name), "combo_point", point_from(combo_point_name), "oldsharpen005095", oldsharp_sub["serverGetPoint"].astype(float).to_numpy()))

    for a_key, action, p_key, point, s_key, sprob in candidates:
        name = f"submission_v167_v168_v170__a{a_key}__p{p_key}__s{s_key}.csv"
        info = write_submission(name, rally_uids, action, point, sprob)
        info.update({"action_source": a_key, "point_source": p_key, "server_source": s_key})
        generated.append(info)

    summary = {
        "base": {
            "r111_action": float(v167_search[v167_search["candidate"].eq("v167_r111_global_w0p01")]["action_macro_f1"].iloc[0])
            if "v167_r111_global_w0p01" in set(v167_search["candidate"])
            else None,
            "safe_server_auc": float(roc_auc_score(rows["serverGetPoint"].astype(int), server)),
        },
        "best_v167": v167_search.head(10).to_dict(orient="records"),
        "best_v168": v168_search.head(10).to_dict(orient="records"),
        "best_v170": v170_search.head(10).to_dict(orient="records"),
        "best_combo": combo_search.head(10).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "v167_v168_v170_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "v167_v168_v170_report.md").write_text(
        "# V167/V168/V170 Pretraining Proxies\n\n"
        "## Best Results\n\n"
        f"- V167 best: `{best_v167}` action `{v167_search.iloc[0]['action_macro_f1']:.6f}`, churn `{v167_search.iloc[0]['action_churn_vs_base']:.4%}`.\n"
        f"- V168 best: `{best_v168}` point `{v168_search.iloc[0]['point_macro_f1']:.6f}`, churn `{v168_search.iloc[0]['point_churn_vs_base']:.4%}`.\n"
        f"- V170 best: `{best_v170}` action `{v170_search.iloc[0]['action_macro_f1']:.6f}`, rare mean `{v170_search.iloc[0]['rare_action_mean_f1']:.6f}`.\n"
        f"- Best combo: `{best_combo['candidate']}` overall `{best_combo['overall']:.6f}`.\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )
    with open("experiments_log.md", "a", encoding="utf-8") as f:
        f.write(
            "\n\n## V167/V168/V170 pretraining proxies\n\n"
            f"- V167 best: `{best_v167}` action {v167_search.iloc[0]['action_macro_f1']:.6f}, "
            f"churn {v167_search.iloc[0]['action_churn_vs_base']:.4%}.\n"
            f"- V168 best: `{best_v168}` point {v168_search.iloc[0]['point_macro_f1']:.6f}, "
            f"churn {v168_search.iloc[0]['point_churn_vs_base']:.4%}.\n"
            f"- V170 best: `{best_v170}` action {v170_search.iloc[0]['action_macro_f1']:.6f}, "
            f"rare mean F1 {v170_search.iloc[0]['rare_action_mean_f1']:.6f}.\n"
            f"- Best combo: `{best_combo['candidate']}` overall {best_combo['overall']:.6f}.\n"
            "- Generated V167/V168/V170 safe candidates plus one oldsharpen diagnostic candidate.\n"
        )
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
