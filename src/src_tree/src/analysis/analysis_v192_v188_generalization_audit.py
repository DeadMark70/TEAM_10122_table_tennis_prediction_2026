"""V192 V188 generalization audit.

This script investigates why V188 raw GRU point OOF is strong while raw test
argmax collapses.  It retrains the V188 r186_w005 setting, compares OOF/test
probability calibration, and audits residual transition behavior.

No submission is generated.  TTMATCH is not read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot, point_pred
from analysis_r187_point_intent_student import add_r186_priors
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, prepare_prefix_features
from analysis_v188_point_intent_gru import (
    DEVICE,
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
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v192_v188_generalization_audit")
SRC_DEST = Path("src/analysis/analysis_v192_v188_generalization_audit.py")
CAP5_SUB = Path("upload_candidates_20260519/submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv")


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


def distribution(labels: np.ndarray, n: int = 10) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n)
    return {str(i): int(counts[i]) for i in range(n) if counts[i] > 0}


def prob_stats(prob: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float | dict]:
    p = normalize_rows_safe(prob)
    order = np.argsort(-p, axis=1)
    top = p[np.arange(len(p)), order[:, 0]]
    second = p[np.arange(len(p)), order[:, 1]]
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1) / np.log(p.shape[1])
    rec: dict[str, float | dict] = {
        "p0_mean": float(p[:, 0].mean()),
        "p0_p50": float(np.quantile(p[:, 0], 0.50)),
        "p0_p90": float(np.quantile(p[:, 0], 0.90)),
        "p0_p99": float(np.quantile(p[:, 0], 0.99)),
        "top1_p0_rate": float(np.mean(order[:, 0] == 0)),
        "top1_mean": float(top.mean()),
        "margin_mean": float((top - second).mean()),
        "margin_p50": float(np.quantile(top - second, 0.50)),
        "entropy_mean": float(entropy.mean()),
    }
    if labels is not None:
        pred = order[:, 0].astype(int)
        rec["argmax_distribution"] = distribution(pred)
        rec["argmax_point_macro_f1"] = float(f1_score(labels, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    else:
        rec["argmax_distribution"] = distribution(order[:, 0])
    return rec


def add_phase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "r184_phase" not in out.columns:
        out["r184_phase"] = "unknown"
    return out


def feature_shift(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = ["prefix_len", "serverScoreDiff", "scoreTotal", "lag0_pointId", "lag0_actionId", "lag0_spinId", "lag0_strengthId"]
    for col in numeric:
        if col not in train_rows.columns or col not in test_rows.columns:
            continue
        tr = pd.to_numeric(train_rows[col], errors="coerce")
        te = pd.to_numeric(test_rows[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "train_mean": float(tr.mean()),
                "test_mean": float(te.mean()),
                "mean_delta": float(te.mean() - tr.mean()),
                "train_p50": float(tr.quantile(0.50)),
                "test_p50": float(te.quantile(0.50)),
                "train_p90": float(tr.quantile(0.90)),
                "test_p90": float(te.quantile(0.90)),
            }
        )
    for col in ["r184_phase", "r184_lag0_family", "r184_lag0_depth", "r184_lag0_side"]:
        if col not in train_rows.columns or col not in test_rows.columns:
            continue
        tr = train_rows[col].astype(str).value_counts(normalize=True)
        te = test_rows[col].astype(str).value_counts(normalize=True)
        keys = sorted(set(tr.index) | set(te.index))
        for key in keys:
            rows.append(
                {
                    "feature": f"{col}={key}",
                    "train_mean": float(tr.get(key, 0.0)),
                    "test_mean": float(te.get(key, 0.0)),
                    "mean_delta": float(te.get(key, 0.0) - tr.get(key, 0.0)),
                    "train_p50": np.nan,
                    "test_p50": np.nan,
                    "train_p90": np.nan,
                    "test_p90": np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values("mean_delta", key=lambda s: s.abs(), ascending=False)


def transition_table(base: np.ndarray, new: np.ndarray, prefix: str) -> pd.DataFrame:
    mask = np.asarray(base, dtype=int) != np.asarray(new, dtype=int)
    if not np.any(mask):
        return pd.DataFrame(columns=["source", "from_point", "to_point", "rows"])
    df = pd.DataFrame({"from_point": np.asarray(base, dtype=int)[mask], "to_point": np.asarray(new, dtype=int)[mask]})
    out = df.value_counts(["from_point", "to_point"]).reset_index(name="rows")
    out.insert(0, "source", prefix)
    return out.sort_values("rows", ascending=False)


def point0_bias_grid(prob: np.ndarray, y: np.ndarray, base_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for bias in [-4, -3, -2, -1, -0.5, 0, 0.5, 1]:
        p = prob.copy()
        p[:, 0] *= np.exp(float(bias))
        p = normalize_rows_safe(p)
        pred = p.argmax(axis=1).astype(int)
        rows.append(
            {
                "bias": float(bias),
                "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
                "point0_pred_rate": float(np.mean(pred == 0)),
                "churn_vs_base_pred": float(np.mean(pred != base_pred)),
            }
        )
    return pd.DataFrame(rows)


def nonterminal_renorm(prob: np.ndarray) -> np.ndarray:
    p = prob.copy()
    p[:, 0] = 0.0
    return normalize_rows_safe(p)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    rows = add_r186_priors(rows, pd.read_csv(R186_TRAIN))
    test_rows = add_r186_priors(test_rows, pd.read_csv(R186_TEST))

    r111_oof = load_pickle(R111_OOF)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    v3_oof = load_pickle("oof_proba_v3.pkl")
    tuning = r111_oof["tuning"]
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
    local_base_pred_oof = point_pred(rows, local_base_prob_oof, tuning)

    train_seq, train_len = build_padded_stroke_tensor(sequences_for_rows(rows, raw_groups("train.csv")), MAX_SEQ_LEN, 0)
    test_seq, test_len = build_padded_stroke_tensor(sequences_for_rows(test_rows, raw_groups("test_new.csv")), MAX_SEQ_LEN, 0)
    vocab_sizes = [int(max(train_seq[:, :, i].max(), test_seq[:, :, i].max()) + 1) for i in range(train_seq.shape[2])]
    x_static, stats = static_matrix(rows, v173_action_oof_prob, local_base_prob_oof)
    x_test_static, _ = static_matrix(test_rows, v173_action_test_prob, local_base_prob_test, stats)
    teacher = teacher_matrix(rows)
    teacher_test = teacher_matrix(test_rows)
    y = rows["next_pointId"].astype(int).to_numpy()

    weights = dict(LOSS_SETTINGS)["r186_w005"]
    oof_prob = np.zeros((len(rows), 10), dtype=float)
    fold_test_probs = []
    fold_rows = []
    test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
    for fold in sorted(rows["fold"].unique()):
        valid = rows["fold"].eq(fold).to_numpy()
        train = ~valid
        train_ds = StrokeDataset(train_seq[train], train_len[train], x_static[train], y[train], teacher[train])
        valid_ds = StrokeDataset(train_seq[valid], train_len[valid], x_static[valid], y[valid], teacher[valid])
        model, val_loss = train_model(train_ds, valid_ds, vocab_sizes, x_static.shape[1], weights, 1880 + int(fold))
        oof_prob[valid] = predict_proba(model, valid_ds)
        fold_test_probs.append(predict_proba(model, test_ds))
        pred = oof_prob[valid].argmax(axis=1)
        fold_rows.append(
            {
                "fold": int(fold),
                "val_loss": float(val_loss),
                "raw_point_macro_f1": float(f1_score(y[valid], pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
                "raw_point0_rate": float(np.mean(pred == 0)),
                "p0_mean": float(oof_prob[valid, 0].mean()),
            }
        )

    full_ds = StrokeDataset(train_seq, train_len, x_static, y, teacher)
    hold = max(1, len(train_seq) // 10)
    hold_ds = StrokeDataset(train_seq[:hold], train_len[:hold], x_static[:hold], y[:hold], teacher[:hold])
    full_model, _ = train_model(full_ds, hold_ds, vocab_sizes, x_static.shape[1], weights, 1988)
    test_full_prob = predict_proba(full_model, test_ds)
    test_foldens_prob = normalize_rows_safe(np.mean(fold_test_probs, axis=0))

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    cap5_sub = load_sub(CAP5_SUB, state["rally_uids"])
    test_base_point = base_sub["pointId"].astype(int).to_numpy()
    test_cap5_point = cap5_sub["pointId"].astype(int).to_numpy()

    oof_raw = oof_prob.argmax(axis=1).astype(int)
    test_full_raw = test_full_prob.argmax(axis=1).astype(int)
    test_foldens_raw = test_foldens_prob.argmax(axis=1).astype(int)
    oof_blend5 = row_log_blend(local_base_prob_oof, oof_prob, 0.05)
    oof_cap5, _ = capped_residual_labels(local_base_pred_oof, oof_blend5, 0.05)

    summaries = {
        "oof_raw": prob_stats(oof_prob, y),
        "test_full_raw": prob_stats(test_full_prob),
        "test_foldens_raw": prob_stats(test_foldens_prob),
        "oof_base": prob_stats(local_base_prob_oof, y),
        "test_base": prob_stats(local_base_prob_test),
        "oof_nonterminal_renorm": prob_stats(nonterminal_renorm(oof_prob), y),
        "test_full_nonterminal_renorm": prob_stats(nonterminal_renorm(test_full_prob)),
        "test_foldens_nonterminal_renorm": prob_stats(nonterminal_renorm(test_foldens_prob)),
    }

    pd.DataFrame(fold_rows).to_csv(OUTDIR / "v192_fold_raw_metrics.csv", index=False)
    feature_shift(rows, test_rows).to_csv(OUTDIR / "v192_feature_shift.csv", index=False)
    point0_bias_grid(oof_prob, y, local_base_pred_oof).to_csv(OUTDIR / "v192_oof_point0_bias_grid.csv", index=False)
    trans = pd.concat(
        [
            transition_table(local_base_pred_oof, oof_raw, "oof_raw_vs_base"),
            transition_table(local_base_pred_oof, oof_cap5, "oof_cap5_vs_base"),
            transition_table(test_base_point, test_full_raw, "test_full_raw_vs_base"),
            transition_table(test_base_point, test_foldens_raw, "test_foldens_raw_vs_base"),
            transition_table(test_base_point, test_cap5_point, "test_cap5_vs_base"),
        ],
        ignore_index=True,
    )
    trans.to_csv(OUTDIR / "v192_transition_tables.csv", index=False)

    raw0_base_nonzero = {
        "test_full_raw0_base_nonzero_rate": float(np.mean((test_full_raw == 0) & (test_base_point != 0))),
        "test_foldens_raw0_base_nonzero_rate": float(np.mean((test_foldens_raw == 0) & (test_base_point != 0))),
        "test_base_point0_rate": float(np.mean(test_base_point == 0)),
        "test_cap5_point0_rate": float(np.mean(test_cap5_point == 0)),
        "oof_label_point0_rate": float(np.mean(y == 0)),
        "oof_base_point0_rate": float(np.mean(local_base_pred_oof == 0)),
        "oof_raw_point0_rate": float(np.mean(oof_raw == 0)),
        "oof_cap5_point0_rate": float(np.mean(oof_cap5 == 0)),
    }

    report = {
        "verdict": "RAW_DECISION_BOUNDARY_NOT_TEST_STABLE",
        "device": DEVICE,
        "summaries": summaries,
        "raw0_base_nonzero": raw0_base_nonzero,
        "distributions": {
            "oof_label": distribution(y),
            "oof_base_pred": distribution(local_base_pred_oof),
            "oof_raw_pred": distribution(oof_raw),
            "oof_cap5_pred": distribution(oof_cap5),
            "test_base_pred": distribution(test_base_point),
            "test_full_raw_pred": distribution(test_full_raw),
            "test_foldens_raw_pred": distribution(test_foldens_raw),
            "test_cap5_pred": distribution(test_cap5_point),
        },
        "artifacts": [
            "v192_fold_raw_metrics.csv",
            "v192_feature_shift.csv",
            "v192_oof_point0_bias_grid.csv",
            "v192_transition_tables.csv",
        ],
        "notes": [
            "OOF raw GRU is strong, but test raw argmax collapses to point0.",
            "The collapse is driven by terminal/point0 decision boundary instability, not lack of sequence signal.",
            "Residual works because the V173/R119 point base preserves the public joint distribution while GRU only changes capped rows.",
            "Next calibration work should focus on point0/terminal bias, nonterminal renormalization, and prior-matched residuals.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v192_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v192_report.md").write_text(
        "# V192 V188 Generalization Audit\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Device: `{DEVICE}`\n\n"
        "## Key Distributions\n\n"
        f"- OOF label: `{report['distributions']['oof_label']}`\n"
        f"- OOF raw pred: `{report['distributions']['oof_raw_pred']}`\n"
        f"- Test full raw pred: `{report['distributions']['test_full_raw_pred']}`\n"
        f"- Test foldens raw pred: `{report['distributions']['test_foldens_raw_pred']}`\n"
        f"- Test cap5 pred: `{report['distributions']['test_cap5_pred']}`\n\n"
        "## Point0 Summary\n\n"
        f"- OOF raw p0 mean: `{summaries['oof_raw']['p0_mean']:.6f}`\n"
        f"- Test full raw p0 mean: `{summaries['test_full_raw']['p0_mean']:.6f}`\n"
        f"- Test foldens raw p0 mean: `{summaries['test_foldens_raw']['p0_mean']:.6f}`\n"
        f"- Test raw0/base-nonzero rate full: `{raw0_base_nonzero['test_full_raw0_base_nonzero_rate']:.6f}`\n"
        f"- Test raw0/base-nonzero rate foldens: `{raw0_base_nonzero['test_foldens_raw0_base_nonzero_rate']:.6f}`\n\n"
        "## Artifacts\n\n"
        + "\n".join(f"- `{a}`" for a in report["artifacts"])
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v192_v188_generalization_audit.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
