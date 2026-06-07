"""Export V188 raw GRU point diagnostic submission.

This is intentionally high risk: pointId is the raw argmax from the V188 GRU
point model, while action/server stay on the no-old V173/R121 anchor.

Use only as a point ceiling diagnostic.  TTMATCH is not read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot
from analysis_r187_point_intent_student import add_r186_priors
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, prepare_prefix_features
from analysis_v188_point_intent_gru import (
    DEVICE,
    LOSS_SETTINGS,
    MAX_SEQ_LEN,
    OUTDIR as V188_OUTDIR,
    R186_TEST,
    R186_TRAIN,
    StrokeDataset,
    build_padded_stroke_tensor,
    load_pickle,
    predict_proba,
    raw_groups,
    sequences_for_rows,
    static_matrix,
    teacher_matrix,
    train_model,
)


OUTDIR = Path("v188_point_intent_gru")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v188_raw_gru_diagnostic_export.py")
RAW_FULL_NAME = "submission_v188_raw_gru_r186_w002_full_point__v173action_r121server.csv"
RAW_FOLDENS_NAME = "submission_v188_raw_gru_r186_w002_foldens_point__v173action_r121server.csv"


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


def write_submission(base_sub: pd.DataFrame, point: np.ndarray, name: str) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    rows = add_r186_priors(rows, pd.read_csv(R186_TRAIN))
    test_rows = add_r186_priors(test_rows, pd.read_csv(R186_TEST))

    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    v3_oof = load_pickle("oof_proba_v3.pkl")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)

    prefix_train = add_r185_columns(prefix, None, pool=True)
    v173_action_oof_prob = one_hot(state["v173_pred_oof"], 19)
    v173_action_test_prob = one_hot(state["v173_pred_test"], 19)
    r119_oof = r119_oof_prior(rows, prefix_train, v173_action_oof_prob)
    r119_test = action_conditioned_point_prior(test_rows, prefix_train, v173_action_test_prob)
    local_base_prob_oof = normalize_rows_safe(0.95 * base_point_oof + 0.05 * r119_oof)
    local_base_prob_test = normalize_rows_safe(0.95 * base_point_test + 0.05 * r119_test)

    train_seq, train_len = build_padded_stroke_tensor(sequences_for_rows(rows, raw_groups("train.csv")), MAX_SEQ_LEN, 0)
    test_seq, test_len = build_padded_stroke_tensor(sequences_for_rows(test_rows, raw_groups("test_new.csv")), MAX_SEQ_LEN, 0)
    vocab_sizes = [int(max(train_seq[:, :, i].max(), test_seq[:, :, i].max()) + 1) for i in range(train_seq.shape[2])]
    x_static, stats = static_matrix(rows, v173_action_oof_prob, local_base_prob_oof)
    x_test_static, _ = static_matrix(test_rows, v173_action_test_prob, local_base_prob_test, stats)
    teacher = teacher_matrix(rows)
    teacher_test = teacher_matrix(test_rows)
    y = rows["next_pointId"].astype(int).to_numpy()

    weights = dict(LOSS_SETTINGS)["r186_w002"]
    full_ds = StrokeDataset(train_seq, train_len, x_static, y, teacher)
    hold = max(1, len(train_seq) // 10)
    hold_ds = StrokeDataset(train_seq[:hold], train_len[:hold], x_static[:hold], y[:hold], teacher[:hold])
    test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
    full_model, _ = train_model(full_ds, hold_ds, vocab_sizes, x_static.shape[1], weights, 1988)
    full_prob = predict_proba(full_model, test_ds)
    raw_full_point = full_prob.argmax(axis=1).astype(int)

    fold_probs = []
    for fold in sorted(rows["fold"].unique()):
        valid = rows["fold"].eq(fold).to_numpy()
        train = ~valid
        train_ds = StrokeDataset(train_seq[train], train_len[train], x_static[train], y[train], teacher[train])
        valid_ds = StrokeDataset(train_seq[valid], train_len[valid], x_static[valid], y[valid], teacher[valid])
        model, _ = train_model(train_ds, valid_ds, vocab_sizes, x_static.shape[1], weights, 1880 + int(fold))
        fold_probs.append(predict_proba(model, test_ds))
    foldens_prob = normalize_rows_safe(np.mean(fold_probs, axis=0))
    raw_foldens_point = foldens_prob.argmax(axis=1).astype(int)

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    base_point = base_sub["pointId"].astype(int).to_numpy()
    full_info = write_submission(base_sub, raw_full_point, RAW_FULL_NAME)
    foldens_info = write_submission(base_sub, raw_foldens_point, RAW_FOLDENS_NAME)

    report = {
        "verdict": "HIGH_RISK_DIAGNOSTIC_EXPORTED",
        "recommended_raw_diagnostic": foldens_info,
        "full_model_diagnostic": full_info,
        "source": "v188_r186_w002_raw_argmax",
        "device": DEVICE,
        "foldens_test_churn_vs_v173_r119": float(np.mean(raw_foldens_point != base_point)),
        "foldens_test_changed_rows": int(np.sum(raw_foldens_point != base_point)),
        "foldens_point_counts": {str(k): int(v) for k, v in pd.Series(raw_foldens_point).value_counts().sort_index().items()},
        "full_model_test_churn_vs_v173_r119": float(np.mean(raw_full_point != base_point)),
        "full_model_test_changed_rows": int(np.sum(raw_full_point != base_point)),
        "full_model_point_counts": {str(k): int(v) for k, v in pd.Series(raw_full_point).value_counts().sort_index().items()},
        "known_oof": {
            "point_macro_f1": 0.26550930434142234,
            "delta_vs_base": 0.052903685053858895,
            "point_churn_vs_base": 0.6057352450816939,
        },
        "notes": [
            "High-risk point ceiling diagnostic only.",
            "Action is V173, server is R121 no-old.",
            "Recommended raw diagnostic uses a five-fold model ensemble on test probabilities.",
            "Full-train raw model is also exported, but may be less stable.",
            "Point is raw GRU argmax without residual/churn cap.",
            "Do not treat as private-safe final unless public validates it.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v188_raw_gru_diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v188_raw_gru_diagnostic_report.md").write_text(
        "# V188 Raw GRU Diagnostic\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Recommended submission: `{foldens_info['upload_path']}`\n"
        f"- Full-model submission: `{full_info['upload_path']}`\n"
        f"- Known OOF point Macro-F1: `{report['known_oof']['point_macro_f1']:.6f}`\n"
        f"- Known OOF churn: `{report['known_oof']['point_churn_vs_base']:.6f}`\n"
        f"- Fold-ensemble test churn vs V173/R119/R121: `{report['foldens_test_churn_vs_v173_r119']:.6f}`\n"
        f"- Full-model test churn vs V173/R119/R121: `{report['full_model_test_churn_vs_v173_r119']:.6f}`\n"
        f"- Fold-ensemble point counts: `{report['foldens_point_counts']}`\n"
        f"- Full-model point counts: `{report['full_model_point_counts']}`\n\n"
        "## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v188_raw_gru_diagnostic_export.py", SRC_DEST)
    print(json.dumps({"recommended_submission": RAW_FOLDENS_NAME, "foldens_test_churn": report["foldens_test_churn_vs_v173_r119"]}, indent=2))


if __name__ == "__main__":
    main()
