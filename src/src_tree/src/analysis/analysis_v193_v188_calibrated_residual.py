"""V193 calibrated V188 neural point residuals.

V192 showed that V188 raw GRU has useful OOF point structure but test raw
argmax collapses through the point0/terminal boundary.  V193 does not train a
larger model.  It calibrates the V188 r186_w005 neural probabilities before
residual inference:

  - point0 probability capping
  - target point0-rate matching
  - phase/prefix/depth domain gates
  - nonterminal-only residuals

Submissions keep action=V173 and server=R121.  TTMATCH is not read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot, point_pred
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
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v193_v188_calibrated_residual")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v193_v188_calibrated_residual.py")

ALPHAS = [0.03, 0.05, 0.075]
CAPS = [0.02, 0.03, 0.05]
P0_CAPS = [0.35, 0.45, 0.55, 0.65]
POINT0_TARGETS = [0.23, 0.26, 0.29]


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


def point0_cap(prob: np.ndarray, cap: float) -> np.ndarray:
    p = normalize_rows_safe(prob)
    out = p.copy()
    old0 = out[:, 0].copy()
    new0 = np.minimum(old0, float(cap))
    old_non = np.clip(1.0 - old0, 1e-12, 1.0)
    new_non = np.clip(1.0 - new0, 1e-12, 1.0)
    out[:, 0] = new0
    out[:, 1:] *= (new_non / old_non)[:, None]
    return normalize_rows_safe(out)


def point0_prior_match(prob: np.ndarray, target_rate: float) -> np.ndarray:
    p = normalize_rows_safe(prob)
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        q = p.copy()
        q[:, 0] *= np.exp(mid)
        q = normalize_rows_safe(q)
        rate = float(np.mean(q.argmax(axis=1) == 0))
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    q = p.copy()
    q[:, 0] *= np.exp((lo + hi) / 2.0)
    return normalize_rows_safe(q)


def nonterminal_only(prob: np.ndarray, base_prob: np.ndarray) -> np.ndarray:
    p = normalize_rows_safe(prob)
    out = p.copy()
    out[:, 0] = np.clip(base_prob[:, 0], 1e-12, 1.0)
    old_non = np.clip(p[:, 1:].sum(axis=1), 1e-12, 1.0)
    new_non = np.clip(1.0 - out[:, 0], 1e-12, 1.0)
    out[:, 1:] = p[:, 1:] * (new_non / old_non)[:, None]
    return normalize_rows_safe(out)


def gate_rows(rows: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(len(rows), dtype=bool)
    phase = rows["r184_phase"].astype(str)
    depth = rows["r184_lag0_depth"].astype(str)
    prefix = pd.to_numeric(rows["prefix_len"], errors="coerce").fillna(0)
    if mode == "domain_shift":
        return phase.eq("rally").to_numpy() | depth.eq("long").to_numpy() | prefix.ge(3).to_numpy()
    if mode == "long_rally":
        return phase.eq("rally").to_numpy() & depth.eq("long").to_numpy()
    if mode == "not_receive":
        return ~phase.eq("receive").to_numpy()
    raise ValueError(mode)


def apply_gate(base_prob: np.ndarray, candidate_prob: np.ndarray, rows: pd.DataFrame, mode: str) -> np.ndarray:
    mask = gate_rows(rows, mode)
    out = np.asarray(base_prob, dtype=float).copy()
    out[mask] = candidate_prob[mask]
    return normalize_rows_safe(out)


def distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=10)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    point_f1 = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_f1 = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    rec = {
        "candidate": name,
        "point_macro_f1": point_f1,
        "delta_vs_base": point_f1 - base_f1,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "point0_rate": float(np.mean(pred == 0)),
    }
    for k in [0, 1, 3, 4, 7, 8, 9]:
        rec[f"point{k}_f1"] = float(rep[str(k)]["f1-score"])
    rec.update(meta)
    return rec


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray) -> dict:
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


def build_v188_probs() -> dict:
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
    test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
    for fold in sorted(rows["fold"].unique()):
        valid = rows["fold"].eq(fold).to_numpy()
        train = ~valid
        train_ds = StrokeDataset(train_seq[train], train_len[train], x_static[train], y[train], teacher[train])
        valid_ds = StrokeDataset(train_seq[valid], train_len[valid], x_static[valid], y[valid], teacher[valid])
        model, _ = train_model(train_ds, valid_ds, vocab_sizes, x_static.shape[1], weights, 1880 + int(fold))
        oof_prob[valid] = predict_proba(model, valid_ds)
        fold_test_probs.append(predict_proba(model, test_ds))
    test_prob = normalize_rows_safe(np.mean(fold_test_probs, axis=0))

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    test_base_point = base_sub["pointId"].astype(int).to_numpy()
    return {
        "rows": rows,
        "test_rows": test_rows,
        "y": y,
        "oof_prob": oof_prob,
        "test_prob": test_prob,
        "base_prob_oof": local_base_prob_oof,
        "base_prob_test": local_base_prob_test,
        "base_pred_oof": local_base_pred_oof,
        "test_base_point": test_base_point,
        "base_sub": base_sub,
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    data = build_v188_probs()
    y = data["y"]
    base = data["base_pred_oof"]
    test_base = data["test_base_point"]
    search_rows = [eval_candidate("base", y, base, base, {"mode": "base"})]
    pred_store: dict[str, np.ndarray] = {}

    prob_variants: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    for p0cap in P0_CAPS:
        prob_variants.append((f"p0cap{str(p0cap).replace('.', 'p')}", point0_cap(data["oof_prob"], p0cap), point0_cap(data["test_prob"], p0cap), {"calibration": "p0cap", "p0_cap": p0cap}))
    for target in POINT0_TARGETS:
        prob_variants.append((f"p0match{str(target).replace('.', 'p')}", point0_prior_match(data["oof_prob"], target), point0_prior_match(data["test_prob"], target), {"calibration": "p0match", "point0_target": target}))
    prob_variants.append(("nonterminal_only", nonterminal_only(data["oof_prob"], data["base_prob_oof"]), nonterminal_only(data["test_prob"], data["base_prob_test"]), {"calibration": "nonterminal_only"}))

    for tag, oof_cal, test_cal, meta in prob_variants:
        for gate in ["all", "domain_shift", "long_rally", "not_receive"]:
            oof_gated = apply_gate(data["base_prob_oof"], oof_cal, data["rows"], gate)
            test_gated = apply_gate(data["base_prob_test"], test_cal, data["test_rows"], gate)
            for alpha in ALPHAS:
                blended = row_log_blend(data["base_prob_oof"], oof_gated, alpha)
                blended_test = row_log_blend(data["base_prob_test"], test_gated, alpha)
                for cap in CAPS:
                    pred, _ = capped_residual_labels(base, blended, cap)
                    test_pred, test_changed = capped_residual_labels(test_base, blended_test, cap)
                    name = f"v193_{tag}_{gate}_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                    rec = eval_candidate(name, y, pred, base, {**meta, "gate": gate, "alpha": alpha, "cap": cap})
                    rec["test_churn_vs_v173_r119_base"] = float(np.mean(test_pred != test_base))
                    rec["test_changed_rows"] = int(np.sum(test_changed))
                    rec["test_point_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
                    search_rows.append(rec)
                    pred_store[name] = test_pred

    search = pd.DataFrame(search_rows)
    search["tier"] = np.select(
        [search["point_churn_vs_base"].le(0.02), search["point_churn_vs_base"].le(0.05)],
        ["clean", "probe"],
        default="high_churn",
    )
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v193_search.csv", index=False)

    generated = []
    positive = search[(search["tier"].isin(["clean", "probe"])) & search["delta_vs_base"].gt(0) & search["candidate"].str.startswith("v193_")]
    emitted: set[str] = set()
    for tier in ["clean", "probe"]:
        part = positive[positive["tier"].eq(tier)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        if name in emitted:
            continue
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        info.update(rec)
        info["submission"] = sub_name
        generated.append(info)
        emitted.add(name)

    report = {
        "verdict": "CANDIDATES_GENERATED" if generated else "NO_POSITIVE_CANDIDATE",
        "base": search[search["candidate"].eq("base")].iloc[0].to_dict(),
        "best_clean": search[search["tier"].eq("clean")].head(12).to_dict(orient="records"),
        "best_probe": search[search["tier"].eq("probe")].head(12).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "V193 calibrates V188 neural probabilities instead of training a bigger raw model.",
            "Point0 capping, prior matching, and nonterminal-only variants are evaluated.",
            "Submissions are residual/churn-capped on the V173/R119 point base with V173 action and R121 server.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v193_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v193_report.md").write_text(
        "# V193 V188 Calibrated Residual\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + ("\n".join(f"- `{g['upload_path']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119_base']:.6f}`" for g in generated) or "- none")
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v193_v188_calibrated_residual.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated_count": len(generated), "search": str(OUTDIR / "v193_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
