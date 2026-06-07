"""V169/V171/V172 task-aligned sequence pretraining proxies.

V169:
  Task-adaptive transition LM proxy.  Use AICUP train prefixes plus
  test_new observed internal transitions for test-time priors.  This uses no
  hidden target and no old-test server label.

V171:
  Hierarchical causal destiny proxy.  Use action distributions to produce a
  fold-safe P(point | action) prior, then lightly refine point/server.

V172:
  Remaining-length / survival-aware routing proxy.  Estimate remaining bucket
  from public prefix context, then route action/point priors through
  short/mid/long survival experts.

These are intentionally proxy experiments rather than full GPU pretraining.
They answer whether each objective has enough OOF signal to justify a heavier
GRU/Transformer training run.
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
import analysis_v160_v163_task_pretrain_distill as v160  # noqa: E402
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


OUTDIR = Path("v169_v171_v172_sequence_causal_survival")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v169_v171_v172_sequence_causal_survival.py")

R166_TARGETS = Path("r166_teacher_distillation/r166_teacher_targets.npz")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R119_POINT = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R121_MIN = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R154_POINT = UPLOAD_DIR / "submission_r154_md_mirror_a0p01_r67_anchor.csv"
R142_OLDSHARPEN = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


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
    base9 = normalize_rows(out[:, 1:10] / mass[:, None])
    prior9 = normalize_rows(prior[:, 1:10])
    q = np.exp((1.0 - a) * np.log(np.clip(base9, 1e-9, None)) + a * np.log(np.clip(prior9, 1e-9, None)))
    out[:, 1:10] = mass[:, None] * normalize_rows(q)
    return normalize_rows(out)


def action_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return v165.action_pred(meta, prob, tuning)


def point_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return v165.point_pred(meta, prob, tuning)


def eval_action(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict | None = None) -> dict:
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
    if extra:
        rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict | None = None) -> dict:
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
    if extra:
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


def action_point_matrix(df: pd.DataFrame, alpha: float = 30.0) -> np.ndarray:
    y_action = df["next_actionId"].astype(int).to_numpy()
    y_point = df["next_pointId"].astype(int).to_numpy()
    global_counts = np.array([np.sum(y_point == p) + 1.0 for p in POINT_CLASSES], dtype=float)
    global_prior = global_counts / global_counts.sum()
    mat = np.zeros((len(ACTION_CLASSES), len(POINT_CLASSES)), dtype=float)
    for i, a in enumerate(ACTION_CLASSES):
        vals = y_point[y_action == a]
        counts = np.array([np.sum(vals == p) for p in POINT_CLASSES], dtype=float)
        mat[i] = (counts + alpha * global_prior) / (counts.sum() + alpha)
    return normalize_rows(mat)


def foldsafe_action_to_point(prefix: pd.DataFrame, rows: pd.DataFrame, action_prob: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    for fold in sorted(rows["fold"].astype(int).unique()):
        matches = set(rows.loc[rows["fold"].astype(int).eq(fold), "match"].astype(int).unique())
        train_part = prefix[~prefix["match"].astype(int).isin(matches)].copy()
        valid_idx = rows.index[rows["fold"].astype(int).eq(fold)].to_numpy()
        out[valid_idx] = normalize_rows(action_prob[valid_idx] @ action_point_matrix(train_part))
    return normalize_rows(out)


def full_action_to_point(prefix: pd.DataFrame, action_prob: np.ndarray) -> np.ndarray:
    return normalize_rows(action_prob @ action_point_matrix(prefix))


def route_from_remaining_prob(rem: np.ndarray) -> np.ndarray:
    """Map 7 remaining buckets to short/mid/long route probabilities."""
    rem = normalize_rows(rem)
    out = np.zeros((len(rem), 3), dtype=float)
    out[:, 0] = rem[:, 0]  # remaining == 1
    out[:, 1] = rem[:, 1:3].sum(axis=1)  # remaining 2-3
    out[:, 2] = rem[:, 3:].sum(axis=1)  # remaining >= 4
    return normalize_rows(out)


def route_dists(df: pd.DataFrame, target_col: str, classes: list[int], alpha: float = 30.0) -> np.ndarray:
    y = df[target_col].astype(int).to_numpy()
    rem = df["remaining_len_bucket"].clip(1, 7).astype(int).to_numpy()
    global_counts = np.array([np.sum(y == c) + 1.0 for c in classes], dtype=float)
    global_prior = global_counts / global_counts.sum()
    out = np.zeros((3, len(classes)), dtype=float)
    masks = [rem == 1, np.isin(rem, [2, 3]), rem >= 4]
    for i, mask in enumerate(masks):
        vals = y[mask]
        counts = np.array([np.sum(vals == c) for c in classes], dtype=float)
        out[i] = (counts + alpha * global_prior) / (counts.sum() + alpha)
    return normalize_rows(out)


def foldsafe_survival_priors(prefix: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rem_prior = np.zeros((len(rows), 7), dtype=float)
    action_prior = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    point_prior = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    for fold in sorted(rows["fold"].astype(int).unique()):
        matches = set(rows.loc[rows["fold"].astype(int).eq(fold), "match"].astype(int).unique())
        train_part = prefix[~prefix["match"].astype(int).isin(matches)].copy()
        valid_part = rows[rows["fold"].astype(int).eq(fold)].copy()
        fit_rem = v160.fit_prior_tables(train_part, "remaining_len_bucket", list(range(1, 8)), alpha=35.0)
        idx = valid_part.index.to_numpy()
        rem = v160.predict_prior(valid_part, fit_rem, min_support=8)
        routes = route_from_remaining_prob(rem)
        action_prior[idx] = normalize_rows(routes @ route_dists(train_part, "next_actionId", ACTION_CLASSES))
        point_prior[idx] = normalize_rows(routes @ route_dists(train_part, "next_pointId", POINT_CLASSES))
        rem_prior[idx] = rem
    return normalize_rows(rem_prior), normalize_rows(action_prior), normalize_rows(point_prior)


def full_survival_priors(prefix: pd.DataFrame, test_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_rem = v160.fit_prior_tables(prefix, "remaining_len_bucket", list(range(1, 8)), alpha=35.0)
    rem = v160.predict_prior(test_rows, fit_rem, min_support=8)
    routes = route_from_remaining_prob(rem)
    action_prior = normalize_rows(routes @ route_dists(prefix, "next_actionId", ACTION_CLASSES))
    point_prior = normalize_rows(routes @ route_dists(prefix, "next_pointId", POINT_CLASSES))
    return normalize_rows(rem), action_prior, point_prior


def survival_server_proxy(rows: pd.DataFrame, rem_prob: np.ndarray, base_server: np.ndarray, beta: float) -> np.ndarray:
    terminal_p = np.clip(rem_prob[:, 0], 0.0, 1.0)
    next_strike = rows["prefix_len"].astype(int).to_numpy() + 1
    terminal_server_win = (next_strike % 2 == 0).astype(float)
    terminal_score_proxy = terminal_p * terminal_server_win + (1.0 - terminal_p) * base_server
    return np.clip((1.0 - beta) * base_server + beta * terminal_score_proxy, 1e-6, 1.0 - 1e-6)


def causal_server_proxy(rows: pd.DataFrame, point_prob: np.ndarray, base_server: np.ndarray, beta: float) -> np.ndarray:
    terminal_p = np.clip(point_prob[:, 0], 0.0, 1.0)
    next_strike = rows["prefix_len"].astype(int).to_numpy() + 1
    terminal_server_win = (next_strike % 2 == 0).astype(float)
    terminal_score_proxy = terminal_p * terminal_server_win + (1.0 - terminal_p) * base_server
    return np.clip((1.0 - beta) * base_server + beta * terminal_score_proxy, 1e-6, 1.0 - 1e-6)


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
    r111_oof = load_pickle(v165.R111_OOF)
    r111_test = load_pickle(v165.R111_TEST)
    r101_oof = load_pickle(v165.R101_OOF)
    r101_test = load_pickle(v165.R101_TEST)
    tuning = r111_oof["tuning"]

    _, _, prefix, test_prefix, _ = v165.prepare_prefix_features()
    rows = v165.align_prefix_meta(v160.ensure_fold(r111_oof["valid_meta"]), prefix).reset_index(drop=True)
    test_rows = test_prefix.reset_index(drop=True)
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])
    base_point_oof = normalize_rows(0.50 * r111_oof["gru_point"] + 0.50 * r101_oof["gru_point"])
    base_point_test = normalize_rows(0.50 * r111_test["gru_point"] + 0.50 * r101_test["gru_point"])
    base_server_oof = 0.5 * r111_oof["gru_server"] + 0.5 * r101_oof["gru_server"]
    base_server_test = 0.5 * r111_test["gru_server"] + 0.5 * r101_test["gru_server"]

    teacher_action_oof = targets["teacher_action_oof"]
    teacher_action_test = targets["teacher_action_test"]

    base_metrics = eval_combo(rows, base_action_oof, base_point_oof, base_server_oof, tuning, "r111_r101_base")

    # R169: transition LM priors with observed test internal transitions.
    internal_action_oof, internal_point_oof = v160.foldsafe_internal_priors(prefix, rows)
    test_internal = v160.build_test_internal_prefixes(test_raw)
    internal_action_test, internal_point_test = v160.full_internal_priors(prefix, test_rows, test_internal)

    r169_action_rows, r169_action_probs = [], {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075]:
        for source_name, source_oof, source_test in [
            ("internal", internal_action_oof, internal_action_test),
            ("teacher_internal", normalize_rows(0.75 * teacher_action_oof + 0.25 * internal_action_oof), normalize_rows(0.75 * teacher_action_test + 0.25 * internal_action_test)),
        ]:
            prob = row_log_blend(base_action_oof, source_oof, alpha)
            test_prob = row_log_blend(base_action_test, source_test, alpha)
            name = f"r169_action_{source_name}_a{clean_float(alpha)}"
            r169_action_rows.append(eval_action(rows, prob, base_action_oof, tuning, name, {"alpha": alpha, "source": source_name}))
            r169_action_probs[name] = {"oof": prob, "test": test_prob}
    r169_action_search = pd.DataFrame(r169_action_rows).sort_values(["action_macro_f1", "action_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r169_action_search.to_csv(OUTDIR / "r169_transition_action_search.csv", index=False)

    r169_point_rows, r169_point_probs = [], {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075]:
        prob = row_nonterminal_blend(base_point_oof, internal_point_oof, alpha)
        test_prob = row_nonterminal_blend(base_point_test, internal_point_test, alpha)
        name = f"r169_point_internal_a{clean_float(alpha)}"
        r169_point_rows.append(eval_point(rows, prob, base_point_oof, tuning, name, {"alpha": alpha}))
        r169_point_probs[name] = {"oof": prob, "test": test_prob}
    r169_point_search = pd.DataFrame(r169_point_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r169_point_search.to_csv(OUTDIR / "r169_transition_point_search.csv", index=False)

    # R171: action-guided point and destiny server.
    ap_teacher_oof = foldsafe_action_to_point(prefix, rows, teacher_action_oof)
    ap_teacher_test = full_action_to_point(prefix, teacher_action_test)
    ap_base_oof = foldsafe_action_to_point(prefix, rows, base_action_oof)
    ap_base_test = full_action_to_point(prefix, base_action_test)
    r171_point_rows, r171_point_probs = [], {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        for source_name, source_oof, source_test in [
            ("base_action2point", ap_base_oof, ap_base_test),
            ("teacher_action2point", ap_teacher_oof, ap_teacher_test),
            ("mixed_action2point", normalize_rows(0.5 * ap_base_oof + 0.5 * ap_teacher_oof), normalize_rows(0.5 * ap_base_test + 0.5 * ap_teacher_test)),
        ]:
            prob = row_nonterminal_blend(base_point_oof, source_oof, alpha)
            test_prob = row_nonterminal_blend(base_point_test, source_test, alpha)
            name = f"r171_point_{source_name}_a{clean_float(alpha)}"
            r171_point_rows.append(eval_point(rows, prob, base_point_oof, tuning, name, {"alpha": alpha, "source": source_name}))
            r171_point_probs[name] = {"oof": prob, "test": test_prob}
    r171_point_search = pd.DataFrame(r171_point_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r171_point_search.to_csv(OUTDIR / "r171_action_guided_point_search.csv", index=False)

    r171_server_rows, r171_server_probs = [], {}
    best_r171_point_name = str(r171_point_search.iloc[0]["candidate"])
    for beta in [0.02, 0.05, 0.10, 0.15, 0.20]:
        server_oof = causal_server_proxy(rows, r171_point_probs[best_r171_point_name]["oof"], base_server_oof, beta)
        server_test = causal_server_proxy(test_rows, r171_point_probs[best_r171_point_name]["test"], base_server_test, beta)
        name = f"r171_server_destiny_b{clean_float(beta)}"
        r171_server_rows.append(
            {
                "candidate": name,
                "server_auc": float(roc_auc_score(rows["serverGetPoint"].astype(int), server_oof)),
                "server_mad_vs_base": float(np.mean(np.abs(server_oof - base_server_oof))),
                "beta": beta,
            }
        )
        r171_server_probs[name] = {"oof": server_oof, "test": server_test}
    r171_server_search = pd.DataFrame(r171_server_rows).sort_values(["server_auc", "server_mad_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r171_server_search.to_csv(OUTDIR / "r171_destiny_server_search.csv", index=False)

    # R172: survival routing.
    rem_oof, survival_action_oof, survival_point_oof = foldsafe_survival_priors(prefix, rows)
    rem_test, survival_action_test, survival_point_test = full_survival_priors(prefix, test_rows)
    pd.DataFrame(rem_oof, columns=[f"rem_p_{i}" for i in range(1, 8)]).assign(
        rally_uid=rows["rally_uid"].to_numpy(), true_remaining=rows["remaining_len_bucket"].to_numpy()
    ).to_csv(OUTDIR / "r172_remaining_oof_prior.csv", index=False)

    r172_action_rows, r172_action_probs = [], {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        prob = row_log_blend(base_action_oof, survival_action_oof, alpha)
        test_prob = row_log_blend(base_action_test, survival_action_test, alpha)
        name = f"r172_action_survival_a{clean_float(alpha)}"
        r172_action_rows.append(eval_action(rows, prob, base_action_oof, tuning, name, {"alpha": alpha}))
        r172_action_probs[name] = {"oof": prob, "test": test_prob}
    r172_action_search = pd.DataFrame(r172_action_rows).sort_values(["action_macro_f1", "action_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r172_action_search.to_csv(OUTDIR / "r172_survival_action_search.csv", index=False)

    r172_point_rows, r172_point_probs = [], {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        prob = row_nonterminal_blend(base_point_oof, survival_point_oof, alpha)
        test_prob = row_nonterminal_blend(base_point_test, survival_point_test, alpha)
        name = f"r172_point_survival_a{clean_float(alpha)}"
        r172_point_rows.append(eval_point(rows, prob, base_point_oof, tuning, name, {"alpha": alpha}))
        r172_point_probs[name] = {"oof": prob, "test": test_prob}
    r172_point_search = pd.DataFrame(r172_point_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r172_point_search.to_csv(OUTDIR / "r172_survival_point_search.csv", index=False)

    r172_server_rows, r172_server_probs = [], {}
    for beta in [0.02, 0.05, 0.10, 0.15, 0.20]:
        server_oof = survival_server_proxy(rows, rem_oof, base_server_oof, beta)
        server_test = survival_server_proxy(test_rows, rem_test, base_server_test, beta)
        name = f"r172_server_survival_b{clean_float(beta)}"
        r172_server_rows.append(
            {
                "candidate": name,
                "server_auc": float(roc_auc_score(rows["serverGetPoint"].astype(int), server_oof)),
                "server_mad_vs_base": float(np.mean(np.abs(server_oof - base_server_oof))),
                "beta": beta,
            }
        )
        r172_server_probs[name] = {"oof": server_oof, "test": server_test}
    r172_server_search = pd.DataFrame(r172_server_rows).sort_values(["server_auc", "server_mad_vs_base"], ascending=[False, True]).reset_index(drop=True)
    r172_server_search.to_csv(OUTDIR / "r172_survival_server_search.csv", index=False)

    safe_server_oof = base_server_oof
    combo_rows = []
    action_sources = {
        str(r169_action_search.iloc[0]["candidate"]): r169_action_probs[str(r169_action_search.iloc[0]["candidate"])]["oof"],
        str(r172_action_search.iloc[0]["candidate"]): r172_action_probs[str(r172_action_search.iloc[0]["candidate"])]["oof"],
    }
    point_sources = {
        str(r169_point_search.iloc[0]["candidate"]): r169_point_probs[str(r169_point_search.iloc[0]["candidate"])]["oof"],
        str(r171_point_search.iloc[0]["candidate"]): r171_point_probs[str(r171_point_search.iloc[0]["candidate"])]["oof"],
        str(r172_point_search.iloc[0]["candidate"]): r172_point_probs[str(r172_point_search.iloc[0]["candidate"])]["oof"],
    }
    server_sources = {
        "base_server": safe_server_oof,
        str(r171_server_search.iloc[0]["candidate"]): r171_server_probs[str(r171_server_search.iloc[0]["candidate"])]["oof"],
        str(r172_server_search.iloc[0]["candidate"]): r172_server_probs[str(r172_server_search.iloc[0]["candidate"])]["oof"],
    }
    for an, ap in action_sources.items():
        for pn, pp in point_sources.items():
            for sn, sp in server_sources.items():
                combo_rows.append(eval_combo(rows, ap, pp, sp, tuning, f"{an}__{pn}__{sn}"))
    combo_search = pd.DataFrame(combo_rows).sort_values("overall", ascending=False).reset_index(drop=True)
    combo_search.to_csv(OUTDIR / "r169_r171_r172_combo_search.csv", index=False)

    # Submission candidates: keep public-validated point/server anchors where useful.
    r67_sub = load_submission(R67_ANCHOR, rally_uids)
    r119_sub = load_submission(R119_POINT, rally_uids)
    r154_sub = load_submission(R154_POINT, rally_uids)
    r121_sub = load_submission(R121_MIN, rally_uids)
    oldsharp_sub = load_submission(R142_OLDSHARPEN, rally_uids) if R142_OLDSHARPEN.exists() else None

    best_r169_action = str(r169_action_search.iloc[0]["candidate"])
    best_r169_point = str(r169_point_search.iloc[0]["candidate"])
    best_r171_point = str(r171_point_search.iloc[0]["candidate"])
    best_r172_action = str(r172_action_search.iloc[0]["candidate"])
    best_r172_point = str(r172_point_search.iloc[0]["candidate"])
    best_r171_server = str(r171_server_search.iloc[0]["candidate"])
    best_r172_server = str(r172_server_search.iloc[0]["candidate"])
    best_combo = combo_search.iloc[0]

    def action_from(name: str) -> np.ndarray:
        if name in r169_action_probs:
            return action_pred(test_rows, r169_action_probs[name]["test"], tuning)
        return action_pred(test_rows, r172_action_probs[name]["test"], tuning)

    def point_from(name: str) -> np.ndarray:
        if name in r169_point_probs:
            return point_pred(test_rows, r169_point_probs[name]["test"], tuning)
        if name in r171_point_probs:
            return point_pred(test_rows, r171_point_probs[name]["test"], tuning)
        return point_pred(test_rows, r172_point_probs[name]["test"], tuning)

    def server_from(name: str) -> np.ndarray:
        if name in r171_server_probs:
            return r171_server_probs[name]["test"]
        if name in r172_server_probs:
            return r172_server_probs[name]["test"]
        return base_server_test

    generated = []
    candidates = [
        ("r169_best", action_from(best_r169_action), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("r169_point_probe", r67_sub["actionId"].astype(int).to_numpy(), "r169_best_point", point_from(best_r169_point), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("r171_point_probe", r67_sub["actionId"].astype(int).to_numpy(), "r171_best_point", point_from(best_r171_point), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("r172_best", action_from(best_r172_action), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("r172_point_probe", r67_sub["actionId"].astype(int).to_numpy(), "r172_best_point", point_from(best_r172_point), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("r171_server_probe", r67_sub["actionId"].astype(int).to_numpy(), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r171_best_server", server_from(best_r171_server)),
        ("r172_server_probe", r67_sub["actionId"].astype(int).to_numpy(), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r172_best_server", server_from(best_r172_server)),
        ("r172_best", action_from(best_r172_action), "r154_safe_physics", r154_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
    ]
    if oldsharp_sub is not None:
        candidates.append(("combo_best", action_from(str(best_combo["candidate"]).split("__")[0]), "combo_point", point_from(str(best_combo["candidate"]).split("__")[1]), "oldsharpen005095", oldsharp_sub["serverGetPoint"].astype(float).to_numpy()))

    for a_key, action, p_key, point, s_key, server in candidates:
        info = write_submission(f"submission_v169_v171_v172__a{a_key}__p{p_key}__s{s_key}.csv", rally_uids, action, point, server)
        info.update({"action_source": a_key, "point_source": p_key, "server_source": s_key})
        generated.append(info)

    summary = {
        "base_metrics": base_metrics,
        "test_internal_summary": {
            "test_internal_transition_rows": int(len(test_internal)),
            "unique_test_internal_rallies": int(test_internal["rally_uid"].nunique()) if len(test_internal) else 0,
        },
        "best_r169_action": r169_action_search.head(10).to_dict(orient="records"),
        "best_r169_point": r169_point_search.head(10).to_dict(orient="records"),
        "best_r171_point": r171_point_search.head(10).to_dict(orient="records"),
        "best_r171_server": r171_server_search.head(10).to_dict(orient="records"),
        "best_r172_action": r172_action_search.head(10).to_dict(orient="records"),
        "best_r172_point": r172_point_search.head(10).to_dict(orient="records"),
        "best_r172_server": r172_server_search.head(10).to_dict(orient="records"),
        "best_combo": combo_search.head(10).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "v169_v171_v172_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "v169_v171_v172_report.md").write_text(
        "# V169/V171/V172 Sequence-Causal-Survival Proxies\n\n"
        "## Best Results\n\n"
        f"- Base R111/R101 blend overall `{base_metrics['overall']:.6f}` "
        f"(action `{base_metrics['action_macro_f1']:.6f}`, point `{base_metrics['point_macro_f1']:.6f}`, server `{base_metrics['server_auc']:.6f}`).\n"
        f"- R169 action best: `{best_r169_action}` action `{r169_action_search.iloc[0]['action_macro_f1']:.6f}`, churn `{r169_action_search.iloc[0]['action_churn_vs_base']:.4%}`.\n"
        f"- R169 point best: `{best_r169_point}` point `{r169_point_search.iloc[0]['point_macro_f1']:.6f}`, churn `{r169_point_search.iloc[0]['point_churn_vs_base']:.4%}`.\n"
        f"- R171 point best: `{best_r171_point}` point `{r171_point_search.iloc[0]['point_macro_f1']:.6f}`, churn `{r171_point_search.iloc[0]['point_churn_vs_base']:.4%}`.\n"
        f"- R172 action best: `{best_r172_action}` action `{r172_action_search.iloc[0]['action_macro_f1']:.6f}`, churn `{r172_action_search.iloc[0]['action_churn_vs_base']:.4%}`.\n"
        f"- R172 point best: `{best_r172_point}` point `{r172_point_search.iloc[0]['point_macro_f1']:.6f}`, churn `{r172_point_search.iloc[0]['point_churn_vs_base']:.4%}`.\n"
        f"- Best combo: `{best_combo['candidate']}` overall `{best_combo['overall']:.6f}`.\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )
    with open("experiments_log.md", "a", encoding="utf-8") as f:
        f.write(
            "\n\n## V169/V171/V172 sequence-causal-survival proxies\n\n"
            f"- Base R111/R101 blend: overall {base_metrics['overall']:.6f}, "
            f"action {base_metrics['action_macro_f1']:.6f}, point {base_metrics['point_macro_f1']:.6f}, "
            f"server {base_metrics['server_auc']:.6f}.\n"
            f"- R169 action best: `{best_r169_action}` action {r169_action_search.iloc[0]['action_macro_f1']:.6f}, "
            f"churn {r169_action_search.iloc[0]['action_churn_vs_base']:.4%}.\n"
            f"- R169 point best: `{best_r169_point}` point {r169_point_search.iloc[0]['point_macro_f1']:.6f}, "
            f"churn {r169_point_search.iloc[0]['point_churn_vs_base']:.4%}.\n"
            f"- R171 point best: `{best_r171_point}` point {r171_point_search.iloc[0]['point_macro_f1']:.6f}, "
            f"churn {r171_point_search.iloc[0]['point_churn_vs_base']:.4%}; server best `{best_r171_server}` "
            f"AUC {r171_server_search.iloc[0]['server_auc']:.6f}.\n"
            f"- R172 action best: `{best_r172_action}` action {r172_action_search.iloc[0]['action_macro_f1']:.6f}, "
            f"churn {r172_action_search.iloc[0]['action_churn_vs_base']:.4%}; point best `{best_r172_point}` "
            f"{r172_point_search.iloc[0]['point_macro_f1']:.6f}; server best `{best_r172_server}` "
            f"AUC {r172_server_search.iloc[0]['server_auc']:.6f}.\n"
            f"- Best combo: `{best_combo['candidate']}` overall {best_combo['overall']:.6f}.\n"
            "- Generated V169/V171/V172 safe candidates plus optional oldsharpen diagnostic candidate.\n"
        )
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
