"""V222-V225 conservative action improvement suite.

This suite combines four action directions into one aligned selector:

* V222 soft teacher probabilities from V173 distillation, R166, and V208.
* V223 weak-class one-vs-rest specialists.
* V224 player/style latent support for rare/style actions.
* V225 action-point compatibility with the fixed V188 cap5 point anchor.

The suite keeps point fixed at V188 r186_w005 cap5 and server fixed at R121.
It uses V220-style support as a filter/feature, not as a direct generator.
No external rows and no TTMATCH are read.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import analysis_v217_macro_f1_utility_reranker as v217
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    V3Tuning,
    GrUTuning,
    TransformerTuning,
    add_probability_features,
    align_columns,
    best_non_anchor_by_score,
    build_action_candidate_frame,
    distill_v173_soft_anchor,
    load_point_anchor_labels,
    rebuild_r166_best_action,
    rebuild_r184_sources,
    select_capped_action_changes,
    source_probs_for_selector,
    topk_labels,
)
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR, build_terminal_action_candidate
from analysis_v220_action_backoff_support_filter import (
    backoff_support_score,
    build_next_action_examples,
    phase_name,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v222_v225_action_improvement_suite")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v222_v225_action_improvement_suite.py")

WEAK_CLASSES = {0, 3, 5, 7, 8, 9, 12, 14}
STYLE_CLASSES = {8, 9, 12, 14}
SERVE_CLASSES = {15, 16, 17, 18}
CAP_SCHEMES = [
    {
        "name": "v222_soft_selector_cap0p002",
        "cap": 0.002,
        "score_col": "score_soft",
        "allowed": WEAK_CLASSES | {1, 2, 4, 6, 10, 11, 13},
        "per_class_cap": {0: 2, 3: 3, 5: 3, 7: 2, 8: 1, 9: 2, 12: 2, 14: 1},
    },
    {
        "name": "v222_soft_selector_cap0p005",
        "cap": 0.005,
        "score_col": "score_soft",
        "allowed": WEAK_CLASSES | {1, 2, 4, 6, 10, 11, 13},
        "per_class_cap": {0: 3, 3: 4, 5: 4, 7: 3, 8: 2, 9: 3, 12: 3, 14: 1},
    },
    {
        "name": "v223_ovr_weak_cap0p002",
        "cap": 0.002,
        "score_col": "score_ovr",
        "allowed": WEAK_CLASSES,
        "per_class_cap": {0: 2, 3: 2, 5: 3, 7: 2, 8: 1, 9: 2, 12: 2, 14: 1},
    },
    {
        "name": "v223_ovr_weak_cap0p005",
        "cap": 0.005,
        "score_col": "score_ovr",
        "allowed": WEAK_CLASSES,
        "per_class_cap": {0: 3, 3: 4, 5: 4, 7: 3, 8: 2, 9: 3, 12: 3, 14: 1},
    },
    {
        "name": "v224_style_rare_cap0p003",
        "cap": 0.003,
        "score_col": "score_style",
        "allowed": STYLE_CLASSES,
        "per_class_cap": {8: 1, 9: 2, 12: 2, 14: 1},
    },
    {
        "name": "v225_compat_filter_cap0p005",
        "cap": 0.005,
        "score_col": "score_compat",
        "allowed": WEAK_CLASSES | {1, 2, 4, 5, 6, 10, 11, 13},
        "per_class_cap": {0: 3, 3: 4, 5: 4, 7: 3, 8: 2, 9: 3, 12: 3, 14: 1},
    },
    {
        "name": "v226_combined_selector_cap0p002",
        "cap": 0.002,
        "score_col": "score_combined",
        "allowed": WEAK_CLASSES | {1, 2, 4, 5, 6, 10, 11, 13},
        "per_class_cap": {0: 2, 3: 3, 5: 3, 7: 2, 8: 1, 9: 2, 12: 2, 14: 1},
    },
    {
        "name": "v226_combined_selector_cap0p005",
        "cap": 0.005,
        "score_col": "score_combined",
        "allowed": WEAK_CLASSES | {1, 2, 4, 5, 6, 10, 11, 13},
        "per_class_cap": {0: 3, 3: 4, 5: 4, 7: 3, 8: 2, 9: 3, 12: 3, 14: 1},
    },
]


def point_depth(point_id: int) -> int:
    p = int(point_id)
    if p == 0:
        return 0
    if p in (1, 2, 3):
        return 1
    if p in (4, 5, 6):
        return 2
    if p in (7, 8, 9):
        return 3
    return -1


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    row_sum = x.sum(axis=1, keepdims=True)
    return np.divide(x, row_sum, out=np.full_like(x, 1.0 / x.shape[1]), where=row_sum > 0)


def macro_f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def build_style_tables(rows: pd.DataFrame, smoothing: float = 1.0) -> dict:
    """Build smoothed P(action | player) style tables."""
    if rows.empty:
        global_counts = np.full(19, float(smoothing))
        return {"global": normalize_rows_safe(global_counts.reshape(1, -1))[0], "players": {}, "support": {}}
    global_counts = np.full(19, float(smoothing), dtype=float)
    for a in rows["next_action"].astype(int).to_numpy():
        if 0 <= int(a) < 19:
            global_counts[int(a)] += 1.0
    global_rate = normalize_rows_safe(global_counts.reshape(1, -1))[0]
    players = {}
    support = {}
    for player, g in rows.groupby("player", sort=False):
        counts = np.full(19, float(smoothing), dtype=float)
        for a in g["next_action"].astype(int).to_numpy():
            if 0 <= int(a) < 19:
                counts[int(a)] += 1.0
        players[int(player)] = normalize_rows_safe(counts.reshape(1, -1))[0]
        support[int(player)] = int(len(g))
    return {"global": global_rate, "players": players, "support": support}


def style_lift_for_candidate(tables: dict, player: int, candidate_action: int) -> float:
    rate = tables.get("players", {}).get(int(player), tables["global"])
    global_rate = tables["global"]
    cand = int(candidate_action)
    return float(rate[cand] / max(global_rate[cand], 1e-9))


def style_rate_for_candidate(tables: dict, player: int, candidate_action: int) -> tuple[float, int, float]:
    rate = tables.get("players", {}).get(int(player), tables["global"])
    support = int(tables.get("support", {}).get(int(player), 0))
    cand = int(candidate_action)
    return float(rate[cand]), support, style_lift_for_candidate(tables, int(player), cand)


def build_compat_table(action_labels: np.ndarray, point_labels: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    table = np.full((19, 10), float(smoothing), dtype=float)
    for a, p in zip(np.asarray(action_labels, dtype=int), np.asarray(point_labels, dtype=int)):
        if 0 <= int(a) < 19 and 0 <= int(p) < 10:
            table[int(a), int(p)] += 1.0
    return normalize_rows_safe(table)


def compat_delta(table: np.ndarray, candidate_action: int, anchor_action: int, point_id: int) -> float:
    p = int(point_id)
    c = int(candidate_action)
    a = int(anchor_action)
    return float(table[c, p] - table[a, p])


def select_budgeted_changes(
    anchor_labels: np.ndarray,
    candidate_frame: pd.DataFrame,
    score_col: str,
    total_cap: float,
    per_class_cap: dict[int, int] | None = None,
    allowed_classes: set[int] | None = None,
    min_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor_labels, dtype=int)
    out = anchor.copy()
    selected = np.zeros(len(anchor), dtype=bool)
    max_rows = int(np.floor(len(anchor) * float(total_cap)))
    if max_rows <= 0 or candidate_frame.empty or score_col not in candidate_frame.columns:
        return out, selected
    frame = candidate_frame.copy()
    frame = frame[np.isfinite(frame[score_col].astype(float))]
    frame = frame[frame[score_col].astype(float) > float(min_score)]
    frame = frame[frame["candidate_action"].astype(int) != frame["anchor_action"].astype(int)]
    frame = frame[~frame["candidate_action"].astype(int).isin(SERVE_CLASSES)]
    if allowed_classes is not None:
        frame = frame[frame["candidate_action"].astype(int).isin({int(x) for x in allowed_classes})]
    if frame.empty:
        return out, selected
    per_class_cap = {int(k): int(v) for k, v in (per_class_cap or {}).items()}
    frame = frame.sort_values([score_col, "score_combined" if "score_combined" in frame.columns else score_col], ascending=[False, False])
    class_counts: dict[int, int] = {}
    for row in frame.itertuples(index=False):
        rid = int(row.row_id)
        cand = int(row.candidate_action)
        if selected[rid]:
            continue
        if int(getattr(row, "terminal_mismatch", 0)) == 1:
            continue
        if per_class_cap and class_counts.get(cand, 0) >= per_class_cap.get(cand, max_rows):
            continue
        selected[rid] = True
        out[rid] = cand
        class_counts[cand] = class_counts.get(cand, 0) + 1
        if int(selected.sum()) >= max_rows:
            break
    return out, selected


def last_stroke_context(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rid, g in df.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        last = g.sort_values("strikeNumber").iloc[-1]
        records.append(
            {
                "rally_uid": int(rid),
                "prefix_len": int(last["strikeNumber"]),
                "phase": phase_name(int(last["strikeNumber"])),
                "player": int(last["gamePlayerOtherId"]),
                "lag0_action": int(last["actionId"]),
                "lag0_point": int(last["pointId"]),
                "lag0_depth": point_depth(int(last["pointId"])),
                "lag0_spin": int(last["spinId"]),
                "lag0_strength": int(last["strengthId"]),
            }
        )
    return pd.DataFrame(records)


def train_next_examples(train: pd.DataFrame, match_to_fold: dict[int, int] | None = None) -> pd.DataFrame:
    records = []
    for _, g in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) < 2:
            continue
        match = int(g.iloc[0]["match"])
        fold = int(match_to_fold.get(match, -1)) if match_to_fold is not None else -1
        for i in range(len(g) - 1):
            lag = g.iloc[i]
            nxt = g.iloc[i + 1]
            records.append(
                {
                    "match": match,
                    "fold": fold,
                    "player": int(lag["gamePlayerOtherId"]),
                    "phase": phase_name(int(lag["strikeNumber"])),
                    "lag0_action": int(lag["actionId"]),
                    "lag0_point": int(lag["pointId"]),
                    "lag0_depth": point_depth(int(lag["pointId"])),
                    "lag0_spin": int(lag["spinId"]),
                    "lag0_strength": int(lag["strengthId"]),
                    "next_action": int(nxt["actionId"]),
                }
            )
    return pd.DataFrame(records)


def ovr_feature_frame(rows: pd.DataFrame, anchor: np.ndarray, point: np.ndarray, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    cols = [
        "prefix_len",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
    ]
    x = rows[[c for c in cols if c in rows.columns]].reset_index(drop=True).copy()
    for cat in ["audit_phase", "audit_lag0_action_family", "audit_lag0_depth"]:
        if cat in rows.columns:
            x = pd.concat([x, pd.get_dummies(rows[cat].astype(str), prefix=cat, dtype=float).reset_index(drop=True)], axis=1)
    x["anchor_action"] = np.asarray(anchor, dtype=int)
    x["point_anchor"] = np.asarray(point, dtype=int)
    x["point_depth"] = [point_depth(p) for p in point]
    for name, prob in probs.items():
        p = normalize_rows_safe(prob)
        x[f"{name}_max"] = p.max(axis=1)
        x[f"{name}_entropy"] = -np.sum(np.clip(p, 1e-9, 1.0) * np.log(np.clip(p, 1e-9, 1.0)), axis=1)
        x[f"{name}_anchor_p"] = p[np.arange(len(anchor)), np.asarray(anchor, dtype=int)]
        for cls in sorted(WEAK_CLASSES):
            x[f"{name}_p{cls}"] = p[:, int(cls)]
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    x.columns = x.columns.astype(str)
    return x


def fit_ovr_scores(data: dict, y: np.ndarray, anchor: np.ndarray, point_oof: np.ndarray, point_test: np.ndarray, probs_oof: dict[str, np.ndarray], probs_test: dict[str, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict]]:
    rows = data["rows"]
    x_oof = ovr_feature_frame(rows, anchor, point_oof, probs_oof)
    test_anchor = normalize_rows_safe(probs_test["v173_anchor"]).argmax(axis=1)
    x_test = ovr_feature_frame(data["test_rows"], test_anchor, point_test, probs_test)
    metrics = []
    oof_scores = {cls: np.zeros(len(rows), dtype=float) for cls in WEAK_CLASSES}
    test_scores = {}
    for cls in sorted(WEAK_CLASSES):
        for fold in sorted(rows["fold"].astype(int).unique()):
            valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
            train = ~valid
            y_train = (y[train] == int(cls)).astype(int)
            if len(np.unique(y_train)) < 2:
                oof_scores[cls][valid] = 0.0
                continue
            clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.15, max_iter=1000, random_state=222 + cls * 10 + int(fold))
            clf.fit(x_oof.loc[train].to_numpy(), y_train)
            oof_scores[cls][valid] = clf.predict_proba(x_oof.loc[valid].to_numpy())[:, 1]
        y_bin = (y == int(cls)).astype(int)
        auc = float(roc_auc_score(y_bin, oof_scores[cls])) if len(np.unique(y_bin)) > 1 else np.nan
        ap = float(average_precision_score(y_bin, oof_scores[cls])) if len(np.unique(y_bin)) > 1 else np.nan
        metrics.append({"action": int(cls), "support": int(y_bin.sum()), "auc": auc, "average_precision": ap})
        if len(np.unique(y_bin)) < 2:
            test_scores[cls] = np.zeros(len(point_test), dtype=float)
        else:
            clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.15, max_iter=1000, random_state=322 + cls)
            clf.fit(x_oof.to_numpy(), y_bin)
            test_scores[cls] = clf.predict_proba(x_test.to_numpy())[:, 1]
    return oof_scores, test_scores, metrics


def add_support_columns(frame: pd.DataFrame, context: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        rid = int(row.row_id)
        ctx = context.iloc[rid]
        score, details = backoff_support_score(
            examples,
            phase=str(ctx["phase"]),
            lag0_action=int(ctx["lag0_action"]),
            lag0_point=int(ctx["lag0_point"]),
            lag0_depth=int(ctx["lag0_depth"]),
            lag0_spin=int(ctx["lag0_spin"]),
            lag0_strength=int(ctx["lag0_strength"]),
            base_action=int(row.anchor_action),
            cand_action=int(row.candidate_action),
            min_support=20,
        )
        margins = [float(d["margin"]) for d in details if int(d["n"]) >= 20]
        best_n = max([int(d["n"]) for d in details], default=0)
        records.append({"support_score": int(score), "support_margin": float(np.mean(margins)) if margins else 0.0, "support_best_n": int(best_n)})
    out = frame.copy()
    if records:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    else:
        out["support_score"] = 0
        out["support_margin"] = 0.0
        out["support_best_n"] = 0
    return out


def add_style_columns(frame: pd.DataFrame, context: pd.DataFrame, tables: dict) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        rid = int(row.row_id)
        player = int(context.iloc[rid]["player"])
        cand_rate, support, cand_lift = style_rate_for_candidate(tables, player, int(row.candidate_action))
        anchor_rate, _, anchor_lift = style_rate_for_candidate(tables, player, int(row.anchor_action))
        records.append(
            {
                "style_candidate_rate": cand_rate,
                "style_anchor_rate": anchor_rate,
                "style_lift": cand_lift,
                "style_anchor_lift": anchor_lift,
                "style_delta": cand_rate - anchor_rate,
                "style_support": support,
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def add_compat_columns(frame: pd.DataFrame, point_labels: np.ndarray, table: np.ndarray) -> pd.DataFrame:
    records = []
    points = np.asarray(point_labels, dtype=int)
    for row in frame.itertuples(index=False):
        rid = int(row.row_id)
        point = int(points[rid])
        cand = int(row.candidate_action)
        anchor = int(row.anchor_action)
        cd = compat_delta(table, cand, anchor, point)
        terminal_mismatch = int((cand == 0 and point != 0) or (point == 0 and cand in {10, 11, 15, 16, 17, 18}))
        records.append(
            {
                "compat_candidate": float(table[cand, point]),
                "compat_anchor": float(table[anchor, point]),
                "compat_delta": cd,
                "terminal_mismatch": terminal_mismatch,
                "serve_forbidden": int(cand in SERVE_CLASSES),
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def add_ovr_columns(frame: pd.DataFrame, ovr_scores: dict[int, np.ndarray]) -> pd.DataFrame:
    out = frame.copy()
    scores = []
    deltas = []
    for row in out.itertuples(index=False):
        cand = int(row.candidate_action)
        rid = int(row.row_id)
        cand_score = float(ovr_scores.get(cand, np.zeros(len(out)))[rid]) if cand in ovr_scores else 0.0
        anchor_score = float(ovr_scores.get(int(row.anchor_action), np.zeros(len(out)))[rid]) if int(row.anchor_action) in ovr_scores else 0.0
        scores.append(cand_score)
        deltas.append(cand_score - anchor_score)
    out["ovr_p_candidate"] = scores
    out["ovr_delta"] = deltas
    return out


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    teacher_delta = out.get("teacher_mean_p_candidate", 0.0) - out.get("anchor_prob_on_candidate", 0.0)
    utility = out.get("utility", 0.0)
    ovr = out.get("ovr_delta", 0.0)
    style = out.get("style_delta", 0.0)
    compat = out.get("compat_delta", 0.0)
    support = out.get("support_score", 0.0)
    support_margin = out.get("support_margin", 0.0)
    veto = 0.05 * out.get("serve_forbidden", 0.0) + 0.04 * out.get("terminal_mismatch", 0.0)
    rare = out["candidate_action"].astype(int).isin(STYLE_CLASSES).astype(float)
    out["score_soft"] = utility + 0.45 * teacher_delta + 0.05 * support_margin - veto
    out["score_ovr"] = utility + 0.55 * ovr + 0.08 * support + 0.05 * support_margin - veto
    out["score_style"] = utility + 0.45 * style + 0.12 * np.log1p(out.get("style_lift", 0.0)) + 0.08 * support + 0.08 * compat - veto
    out["score_compat"] = utility + 0.55 * compat + 0.12 * support + 0.05 * support_margin - veto
    out["score_combined"] = (
        utility
        + 0.25 * teacher_delta
        + 0.30 * ovr
        + 0.15 * style
        + 0.20 * compat
        + 0.08 * support
        + 0.05 * support_margin
        + 0.02 * rare
        - veto
    )
    # Rare/style rows need at least one physical or style support signal.
    unsupported_rare = out["candidate_action"].astype(int).isin(STYLE_CLASSES) & (out["support_score"].astype(float) < 0) & (out["style_delta"].astype(float) <= 0)
    for col in ["score_style", "score_combined"]:
        out.loc[unsupported_rare, col] -= 0.10
    return out


def build_candidate_sources(
    data: dict,
    state: dict,
    point: pd.DataFrame,
    point_oof: np.ndarray,
    point_test: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    r166_oof, r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    r184_oof, r184_test = rebuild_r184_sources(state, point)
    v216_oof, _ = build_terminal_action_candidate(v173_oof, point_oof, v173_prob_oof)
    v216_test, _ = build_terminal_action_candidate(v173_test, point_test, v173_prob_test)
    v208_oof_path = Path("v209_action_selector_reranker/v209_v208_action_point_aux_oof.npy")
    v208_test_path = Path("v209_action_selector_reranker/v209_v208_action_point_aux_test.npy")
    v208_oof = np.load(v208_oof_path) if v208_oof_path.exists() else v173_prob_oof
    v208_test = np.load(v208_test_path) if v208_test_path.exists() else v173_prob_test
    sources_oof = {
        "v173": v173_oof,
        "r166": r166_oof,
        "r166_top2": topk_labels(r166_prob_oof, 2),
        "v208_top1": normalize_rows_safe(v208_oof).argmax(axis=1),
        "v208_top2": topk_labels(v208_oof, 2),
        **r184_oof,
        "v216_terminal": v216_oof,
    }
    sources_test = {
        "v173": v173_test,
        "r166": r166_test,
        "r166_top2": topk_labels(r166_prob_test, 2),
        "v208_top1": normalize_rows_safe(v208_test).argmax(axis=1),
        "v208_top2": topk_labels(v208_test, 2),
        **r184_test,
        "v216_terminal": v216_test,
    }
    for name in [
        "submission_v217_macro_utility_churn0p005__pv188cap5__sr121.csv",
        "submission_v218_weak_all_cap0p005__pv188cap5__sr121.csv",
        "submission_v219_class_budget_s1p0__pv188cap5__sr121.csv",
        "submission_v220_backoff_balanced_weakonly__pv188cap5__sr121.csv",
    ]:
        path = UPLOAD_DIR / name
        if path.exists():
            tag = name.replace("submission_", "").split("__")[0]
            sources_test[tag] = load_sub(path, point["rally_uid"].astype(int).to_numpy())["actionId"].astype(int).to_numpy()
    probs_oof = source_probs_for_selector(v173_prob_oof, r166_prob_oof, v208_oof)
    probs_test = source_probs_for_selector(v173_prob_test, r166_prob_test, v208_test)
    probs_oof["teacher_mean"] = normalize_rows_safe(0.45 * probs_oof["v173_anchor"] + 0.35 * probs_oof["r166"] + 0.20 * probs_oof["v208"])
    probs_test["teacher_mean"] = normalize_rows_safe(0.45 * probs_test["v173_anchor"] + 0.35 * probs_test["r166"] + 0.20 * probs_test["v208"])
    return sources_oof, sources_test, probs_oof, probs_test, distill_metrics


def utility_candidate_frames(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    sources_oof: dict[str, np.ndarray],
    sources_test: dict[str, np.ndarray],
    probs_oof: dict[str, np.ndarray],
    probs_test: dict[str, np.ndarray],
    point_oof: np.ndarray,
    point_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    base_frame = build_action_candidate_frame(rows, sources_oof, truth=y, anchor_name="v173")
    test_frame = build_action_candidate_frame(test_rows, sources_test, truth=None, anchor_name="v173")
    parts = []
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_rows = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows = ~valid_rows
        train_ids = set(np.where(train_rows)[0])
        valid_ids = set(np.where(valid_rows)[0])
        compat = build_compat_table(y[train_rows], point_oof[train_rows], smoothing=1.0)
        train = add_probability_features(base_frame[base_frame["row_id"].isin(train_ids)].copy(), probs_oof, "v173_anchor", "v208", point_oof, compat)
        valid = add_probability_features(base_frame[base_frame["row_id"].isin(valid_ids)].copy(), probs_oof, "v173_anchor", "v208", point_oof, compat)
        x_train = v217.selector_features(train)
        clf = v217.train_correctness_model(x_train, train["is_correct"].astype(int).to_numpy())
        p_valid = clf.predict_proba(align_columns(v217.selector_features(valid), list(x_train.columns)))[:, 1]
        gain, loss = v217.row_delta_tables(
            y,
            sources_oof["v173"],
            valid["row_id"].astype(int).to_numpy(),
            valid["candidate_action"].astype(int).to_numpy(),
        )
        util = v217.expected_macro_f1_delta(p_valid, gain, loss)
        part = valid.copy()
        part["p_correct"] = p_valid
        part["utility"] = util
        parts.append(part)
        y_valid = valid["is_correct"].astype(int).to_numpy()
        metrics.append({"fold": int(fold), "auc": float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) > 1 else np.nan, "valid_candidate_rows": int(len(valid))})
    compat_full = build_compat_table(y, point_oof, smoothing=1.0)
    full_train = add_probability_features(base_frame.copy(), probs_oof, "v173_anchor", "v208", point_oof, compat_full)
    full_test = add_probability_features(test_frame.copy(), probs_test, "v173_anchor", "v208", point_test, compat_full)
    x_train = v217.selector_features(full_train)
    clf = v217.train_correctness_model(x_train, full_train["is_correct"].astype(int).to_numpy())
    p_test = clf.predict_proba(align_columns(v217.selector_features(full_test), list(x_train.columns)))[:, 1]
    full_test["p_correct"] = p_test
    full_test["utility"] = p_test / max(len(ACTION_CLASSES), 1)
    return pd.concat(parts, ignore_index=True), full_test, metrics


def enrich_frames(
    oof_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    data: dict,
    y: np.ndarray,
    v173_oof: np.ndarray,
    v173_test: np.ndarray,
    point_oof: np.ndarray,
    point_test: np.ndarray,
    probs_oof: dict[str, np.ndarray],
    probs_test: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows = data["rows"]
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    match_to_fold = rows.groupby("match")["fold"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
    train_examples = train_next_examples(train, match_to_fold=match_to_fold)
    test_context = last_stroke_context(test).set_index("rally_uid")
    point_sub = pd.read_csv(POINT_ANCHOR)
    test_context = test_context.loc[point_sub["rally_uid"].astype(int).to_numpy()].reset_index()
    oof_context = pd.DataFrame(
        {
            "rally_uid": rows["rally_uid"].astype(int).to_numpy(),
            "prefix_len": rows["prefix_len"].astype(int).to_numpy(),
            "phase": rows["audit_phase"].astype(str).to_numpy(),
            "player": rows["rally_uid"].astype(int).to_numpy(),
            "lag0_action": rows["lag0_actionId"].astype(int).to_numpy(),
            "lag0_point": rows["lag0_pointId"].astype(int).to_numpy(),
            "lag0_depth": rows["lag0_pointId"].astype(int).map(point_depth).to_numpy(),
            "lag0_spin": rows["lag0_spinId"].astype(int).to_numpy(),
            "lag0_strength": rows["lag0_strengthId"].astype(int).to_numpy(),
        }
    )
    # Exact player ids are not in prepare_data rows. Use full train examples for test style;
    # OOF falls back to rally_uid pseudo-player, so style only becomes active when support exists.
    full_style = build_style_tables(train_examples, smoothing=1.0)
    oof_frame = add_style_columns(oof_frame, oof_context, full_style)
    test_frame = add_style_columns(test_frame, test_context, full_style)
    full_examples_simple = build_next_action_examples(train)
    oof_frame = add_support_columns(oof_frame, oof_context, full_examples_simple)
    test_frame = add_support_columns(test_frame, test_context, full_examples_simple)
    compat_full = build_compat_table(y, point_oof, smoothing=1.0)
    oof_frame = add_compat_columns(oof_frame, point_oof, compat_full)
    test_frame = add_compat_columns(test_frame, point_test, compat_full)
    ovr_oof, ovr_test, ovr_metrics = fit_ovr_scores(data, y, v173_oof, point_oof, point_test, probs_oof, probs_test)
    oof_frame = add_ovr_columns(oof_frame, ovr_oof)
    test_frame = add_ovr_columns(test_frame, ovr_test)
    return score_frame(oof_frame), score_frame(test_frame), ovr_metrics


def class_f1_table(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for label in ACTION_CLASSES:
        f_anchor = f1_score(y, anchor, labels=[label], average="macro", zero_division=0)
        f_pred = f1_score(y, pred, labels=[label], average="macro", zero_division=0)
        rows.append({"action": int(label), "support": int((y == int(label)).sum()), "anchor_f1": float(f_anchor), "candidate_f1": float(f_pred), "delta": float(f_pred - f_anchor)})
    return pd.DataFrame(rows)


def write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
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


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    rows = data["rows"].copy()
    y = rows["next_actionId"].astype(int).to_numpy()
    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    point_oof, point_test = load_point_anchor_labels(data, point)
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    base_score = macro_f1_score(y, v173_oof)

    sources_oof, sources_test, probs_oof, probs_test, distill_metrics = build_candidate_sources(data, state, point, point_oof, point_test)
    oof_frame, test_frame, selector_metrics = utility_candidate_frames(rows, state["test_rows"], y, sources_oof, sources_test, probs_oof, probs_test, point_oof, point_test)
    oof_frame, test_frame, ovr_metrics = enrich_frames(oof_frame, test_frame, data, y, v173_oof, v173_test, point_oof, point_test, probs_oof, probs_test)

    oof_frame.to_csv(OUTDIR / "v222_v225_oof_candidate_frame.csv", index=False)
    test_out = test_frame.copy()
    test_out["rally_uid"] = rally_uids[test_out["row_id"].astype(int).to_numpy()]
    test_out.to_csv(OUTDIR / "v222_v225_test_candidate_frame.csv", index=False)
    pd.DataFrame(distill_metrics).to_csv(OUTDIR / "v222_soft_teacher_metrics.csv", index=False)
    pd.DataFrame(selector_metrics).to_csv(OUTDIR / "v222_v225_selector_fold_metrics.csv", index=False)
    pd.DataFrame(ovr_metrics).to_csv(OUTDIR / "v223_ovr_class_metrics.csv", index=False)

    records = [
        {
            "candidate": "v173_anchor",
            "action_macro_f1": base_score,
            "delta_vs_v173_anchor": 0.0,
            "action_churn_vs_v173_anchor": 0.0,
            "changed_rows": 0,
            "test_changed_rows": 0,
        }
    ]
    generated = []
    class_tables = []
    for scheme in CAP_SCHEMES:
        allowed = {int(x) for x in scheme["allowed"]}
        oof_pred, oof_changed = select_budgeted_changes(
            v173_oof,
            oof_frame,
            score_col=scheme["score_col"],
            total_cap=float(scheme["cap"]),
            per_class_cap=scheme["per_class_cap"],
            allowed_classes=allowed,
            min_score=0.0,
        )
        test_pred, test_changed = select_budgeted_changes(
            v173_test,
            test_frame,
            score_col=scheme["score_col"],
            total_cap=float(scheme["cap"]),
            per_class_cap=scheme["per_class_cap"],
            allowed_classes=allowed,
            min_score=0.0,
        )
        score = macro_f1_score(y, oof_pred)
        rec = {
            "candidate": scheme["name"],
            "action_macro_f1": score,
            "delta_vs_v173_anchor": score - base_score,
            "action_churn_vs_v173_anchor": float(np.mean(oof_pred != v173_oof)),
            "changed_rows": int(oof_changed.sum()),
            "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
            "test_changed_rows": int(test_changed.sum()),
            "changed_actions": json.dumps(pd.Series(test_pred[test_changed]).value_counts().sort_index().to_dict()) if test_changed.any() else "{}",
        }
        records.append(rec)
        class_tab = class_f1_table(y, v173_oof, oof_pred)
        class_tab["candidate"] = scheme["name"]
        class_tables.append(class_tab)
        info = write_submission(f"submission_{scheme['name']}__pv188cap5__sr121.csv", test_pred, point, server)
        info.update(rec)
        generated.append(info)

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True])
    search.to_csv(OUTDIR / "v222_v225_action_search.csv", index=False)
    if class_tables:
        pd.concat(class_tables, ignore_index=True).to_csv(OUTDIR / "v222_v225_class_f1_delta.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": [
            "V222-V225 combines soft teacher, weak OVR, style latent, and action-point compatibility.",
            "Point is fixed at V188 r186_w005 cap5 and server is fixed at R121.",
            "V220 support is used as a feature/filter; V221 direct generation is not used as a source.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v222_v225_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v222_v225_report.md").write_text(
        "# V222-V225 Action Improvement Suite\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v222_v225_action_improvement_suite.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
