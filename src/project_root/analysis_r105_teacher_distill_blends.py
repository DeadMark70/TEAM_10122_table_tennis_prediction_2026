"""R105 post-hoc teacher distillation blends.

This is the conservative/distillation-safe version of the proposed R105:
instead of retraining the GRU with potentially leaky full-test teachers, it
evaluates soft probability imitation on OOF:
  student GRU prob -> blend toward tabular/golden teachers.

If the OOF blend helps, the same blend is applied to full-test probabilities.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import compose_v3_full_point, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r105_teacher_distill")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def eval_action(meta: pd.DataFrame, prob: np.ndarray, tuning: GrUTuning) -> float:
    pred = apply_segmented_multipliers(meta, prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    return float(f1_score(meta["next_actionId"].astype(int), pred, average="macro", labels=ACTION_CLASSES, zero_division=0))


def eval_point(meta: pd.DataFrame, prob: np.ndarray, tuning: GrUTuning) -> float:
    pred = apply_segmented_multipliers(meta, prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    return float(f1_score(meta["next_pointId"].astype(int), pred, average="macro", labels=POINT_CLASSES, zero_division=0))


def write_submission(test_meta, action_prob, point_prob, server_prob, tuning: GrUTuning, name: str, extra=None):
    action_pred = apply_segmented_multipliers(test_meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(test_meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    if extra:
        info.update(extra)
    return info


def run_student(name: str, oof_path: Path, test_path: Path, teacher_action_oof, teacher_action_test, teacher_point_oof, teacher_point_test):
    oof = load_pickle(oof_path)
    test = load_pickle(test_path)
    meta = normalize_meta(oof["valid_meta"]).reset_index(drop=True)
    test_meta = test["test_meta"].reset_index(drop=True)
    tuning = oof["tuning"]
    rows = []
    best_action = {"w": 0.0, "score": eval_action(meta, oof["gru_action"], tuning)}
    best_point = {"w": 0.0, "score": eval_point(meta, oof["gru_point"], tuning)}
    for w in [0.0, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]:
        a = normalize_rows((1 - w) * oof["gru_action"] + w * teacher_action_oof)
        p = normalize_rows((1 - w) * oof["gru_point"] + w * teacher_point_oof)
        af = eval_action(meta, a, tuning)
        pf = eval_point(meta, p, tuning)
        rows.append({"student": name, "weight": w, "action_macro_f1": af, "point_macro_f1": pf})
        if af > best_action["score"]:
            best_action = {"w": w, "score": af}
        if pf > best_point["score"]:
            best_point = {"w": w, "score": pf}
    action_test = normalize_rows((1 - best_action["w"]) * test["gru_action"] + best_action["w"] * teacher_action_test)
    point_test = normalize_rows((1 - best_point["w"]) * test["gru_point"] + best_point["w"] * teacher_point_test)
    info = write_submission(
        test_meta,
        action_test,
        point_test,
        test["gru_server"],
        tuning,
        f"submission_r105_{name}_distill_aw{clean_float(best_action['w'])}_pw{clean_float(best_point['w'])}.csv",
        {"student": name, "best_action_weight": best_action["w"], "best_action_oof": best_action["score"], "best_point_weight": best_point["w"], "best_point_oof": best_point["score"]},
    )
    return pd.DataFrame(rows), info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, *_ = prepare_prefix_features()
    v3_oof = load_pickle("oof_proba_v3.pkl")
    meta = art["valid_meta"].copy().reset_index(drop=True)
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(
        meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]
    ):
        raise ValueError("V3 OOF does not align.")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    test_prefix_v3, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    current_action_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    teacher_action_oof = normalize_rows(0.80 * current_action_oof + 0.20 * golden_oof)
    teacher_action_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])

    reports = []
    generated = []
    for name, oof_path, test_path in [
        ("r101", Path("r101_r103_destiny_gru/oof_proba_r101_r103.pkl"), Path("r101_r103_destiny_gru/test_proba_r101_r103.pkl")),
        ("r106", Path("r106_remaining_gate_gru/oof_proba_r106.pkl"), Path("r106_remaining_gate_gru/test_proba_r106.pkl")),
        ("r111", Path("r111_remaining_moe_gru/oof_proba_r111.pkl"), Path("r111_remaining_moe_gru/test_proba_r111.pkl")),
    ]:
        report, info = run_student(name, oof_path, test_path, teacher_action_oof, teacher_action_test, v3_point_oof, v3_point_test)
        reports.append(report)
        generated.append(info)
    search = pd.concat(reports, ignore_index=True)
    search.to_csv(OUTDIR / "r105_distill_search.csv", index=False)
    out = {"generated": generated, "best": search.sort_values(["action_macro_f1", "point_macro_f1"], ascending=False).head(20).to_dict(orient="records")}
    (OUTDIR / "r105_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r105_teacher_distill_blends.py", "src/analysis/analysis_r105_teacher_distill_blends.py")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
