"""Export clean same-teacher V188 r186_w005 5% residual candidate.

This keeps the public-positive V188 2% configuration fixed and changes only the
residual churn cap from 0.02 to 0.05:

  action = V173
  point  = V188 GRU r186_w005, alpha=0.05, cap=0.05
  server = R121

TTMATCH is not read.
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
    LOSS_SETTINGS,
    MAX_SEQ_LEN,
    R186_TEST,
    R186_TRAIN,
    StrokeDataset,
    build_padded_stroke_tensor,
    capped_residual_labels,
    load_pickle,
    predict_proba,
    raw_groups,
    row_log_blend,
    sequences_for_rows,
    static_matrix,
    teacher_matrix,
    train_model,
)


OUTDIR = Path("v188_point_intent_gru")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v188_export_r186_w005_cap005.py")
SUBMISSION = "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"


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


def write_submission(base_sub: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / SUBMISSION
    upload = UPLOAD_DIR / SUBMISSION
    selected = SELECTED_DIR / SUBMISSION
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": SUBMISSION, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


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

    weights = dict(LOSS_SETTINGS)["r186_w005"]
    full_ds = StrokeDataset(train_seq, train_len, x_static, y, teacher)
    hold = max(1, len(train_seq) // 10)
    hold_ds = StrokeDataset(train_seq[:hold], train_len[:hold], x_static[:hold], y[:hold], teacher[:hold])
    test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
    full_model, _ = train_model(full_ds, hold_ds, vocab_sizes, x_static.shape[1], weights, 1988)
    test_prob = predict_proba(full_model, test_ds)

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    base_point = base_sub["pointId"].astype(int).to_numpy()
    blended = row_log_blend(local_base_prob_test, test_prob, 0.05)
    point, changed = capped_residual_labels(base_point, blended, 0.05)
    info = write_submission(base_sub, point)

    search = pd.read_csv(OUTDIR / "v188_search.csv")
    row = search[search["candidate"].eq("v188_r186_w005_a0p05_cap0p05")].iloc[0].to_dict()
    report = {
        "verdict": "EXPORTED",
        "submission": info,
        "candidate": "v188_r186_w005_a0p05_cap0p05",
        "known_oof": row,
        "test_churn_vs_v173_r119": float(np.mean(point != base_point)),
        "test_changed_rows": int(np.sum(changed)),
        "notes": [
            "Same teacher/config as public-positive V188 2% residual.",
            "Only cap changes from 0.02 to 0.05.",
            "Action is V173 and server is R121 no-old.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v188_r186_w005_cap005_export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v188_r186_w005_cap005_export_report.md").write_text(
        "# V188 r186_w005 5% Export\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Submission: `{info['upload_path']}`\n"
        f"- OOF point Macro-F1: `{float(row['point_macro_f1']):.6f}`\n"
        f"- OOF delta vs base: `{float(row['delta_vs_base']):.6f}`\n"
        f"- OOF churn: `{float(row['point_churn_vs_base']):.6f}`\n"
        f"- Test churn: `{report['test_churn_vs_v173_r119']:.6f}`\n"
        f"- Test changed rows: `{report['test_changed_rows']}`\n\n"
        "## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v188_export_r186_w005_cap005.py", SRC_DEST)
    print(json.dumps({"submission": SUBMISSION, "test_churn": report["test_churn_vs_v173_r119"]}, indent=2))


if __name__ == "__main__":
    main()
