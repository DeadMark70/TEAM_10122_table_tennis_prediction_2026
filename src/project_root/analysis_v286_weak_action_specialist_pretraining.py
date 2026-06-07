"""V286 weak-action specialist pretraining and gated export.

External table-tennis rows are used only to build coarse response priors.
Exact AICUP action labels come only from AICUP training rows, and generated
submissions keep pointId/serverGetPoint fixed to the V261/V173/R121 anchor.
"""

from __future__ import annotations

import __main__
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
V255_CANONICAL = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"

WEAK_ACTIONS = np.array([0, 3, 5, 7, 8, 9, 14], dtype=int)
PROTECTED_ACTIONS = np.array([1, 10, 12, 13], dtype=int)
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
FAMILIES = ["Attack", "Control", "Defensive", "Zero"]
ALLOWED_EXTERNAL = {"openttgames", "sonytabletennis", "TT3D", "DeepMindrobottabletennis", "TT-MatchDynamics"}
BANNED_EXTERNAL_PATTERNS = ["TTMATCH", "CoachAI", "ShuttleSet"]


def action_family(action_id: int) -> str:
    action_id = int(action_id)
    if action_id == 0:
        return "Zero"
    if 1 <= action_id <= 7:
        return "Attack"
    if 8 <= action_id <= 11:
        return "Control"
    if 12 <= action_id <= 14:
        return "Defensive"
    if 15 <= action_id <= 18:
        return "Serve"
    return "Zero"


def point_depth(point_id: int) -> str:
    point_id = int(point_id)
    if point_id <= 0:
        return "zero"
    if point_id in {1, 4, 7}:
        return "short"
    if point_id in {2, 5, 8}:
        return "half"
    return "long"


def phase_bin(prefix_len: int) -> str:
    prefix_len = int(prefix_len)
    if prefix_len <= 1:
        return "receive"
    if prefix_len == 2:
        return "third"
    if prefix_len == 3:
        return "fourth"
    return "rally"


def score_pressure_bin(score_total: float, score_diff: float) -> str:
    total = float(score_total)
    diff = abs(float(score_diff))
    if total >= 18 or diff <= 1:
        return "high"
    if total >= 12 or diff <= 2:
        return "medium"
    return "low"


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    sums = arr.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= 0
    if bad.any():
        arr[bad] = 1.0 / arr.shape[1]
        sums = arr.sum(axis=1, keepdims=True)
    return arr / sums


def weak_action_targets(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    out = np.zeros((len(labels), len(WEAK_ACTIONS)), dtype=np.int8)
    for j, action in enumerate(WEAK_ACTIONS):
        out[:, j] = (labels == int(action)).astype(np.int8)
    return out


def build_candidate_gate(frame: pd.DataFrame, min_score: float, min_support: int) -> pd.Series:
    score = pd.to_numeric(frame["specialist_score"], errors="coerce").fillna(-np.inf)
    support = pd.to_numeric(frame["support_count"], errors="coerce").fillna(0)
    candidate = pd.to_numeric(frame["candidate_action"], errors="coerce").fillna(-1).astype(int)
    anchor = pd.to_numeric(frame["anchor_action"], errors="coerce").fillna(-1).astype(int)
    return (
        score.ge(float(min_score))
        & support.ge(int(min_support))
        & candidate.ne(anchor)
        & candidate.isin(WEAK_ACTIONS.tolist())
        & ~anchor.isin(SERVE_ACTIONS.tolist())
    )


def annotate_prefix_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["phase_bin"] = out["prefix_len"].map(phase_bin)
    out["lag0_action_family"] = out["lag0_actionId"].map(action_family)
    out["lag0_point_depth"] = out["lag0_pointId"].map(point_depth)
    out["lag0_spin"] = pd.to_numeric(out.get("lag0_spinId", 0), errors="coerce").fillna(0).astype(int)
    out["lag0_strength"] = pd.to_numeric(out.get("lag0_strengthId", 0), errors="coerce").fillna(0).astype(int)
    out["score_pressure_bin"] = [
        score_pressure_bin(total, diff)
        for total, diff in zip(
            pd.to_numeric(out.get("scoreTotal", 0), errors="coerce").fillna(0),
            pd.to_numeric(out.get("serverScoreDiff", 0), errors="coerce").fillna(0),
        )
    ]
    return out


def build_aicup_prefix_frames(max_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test_new.csv")
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    train_rows = annotate_prefix_context(build_train_prefix_table(train, max_lag=max_lag))
    test_rows = annotate_prefix_context(build_test_prefix_table(test, max_lag=max_lag))
    return train_rows, test_rows


def _empty_external_audit(raw_rows: int = 0) -> dict[str, Any]:
    return {
        "raw_rows": int(raw_rows),
        "used_rows": 0,
        "source_counts": {},
        "ttmatch_rows_used": 0,
        "coachai_rows_used": 0,
        "shuttleset_rows_used": 0,
    }


def filter_clean_external_corpus(corpus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if corpus.empty:
        return corpus.copy(), _empty_external_audit(0)
    df = corpus.copy()
    source = df.get("source_dataset", pd.Series("", index=df.index)).astype(str)
    risk = df.get("risk_tier", pd.Series("", index=df.index)).astype(str).str.upper()
    allowed = source.isin(ALLOWED_EXTERNAL)
    allowed &= source.ne("TT-MatchDynamics") | risk.eq("YELLOW")
    for pattern in BANNED_EXTERNAL_PATTERNS:
        allowed &= ~source.str.contains(pattern, case=False, regex=False)
    out = df.loc[allowed].copy()
    audit = _empty_external_audit(len(df))
    audit["used_rows"] = int(len(out))
    audit["source_counts"] = {str(k): int(v) for k, v in out["source_dataset"].value_counts().sort_index().items()}
    used_source = out.get("source_dataset", pd.Series("", index=out.index)).astype(str)
    audit["ttmatch_rows_used"] = int(used_source.str.contains("TTMATCH", case=False, regex=False).sum())
    audit["coachai_rows_used"] = int(used_source.str.contains("CoachAI", case=False, regex=False).sum())
    audit["shuttleset_rows_used"] = int(used_source.str.contains("ShuttleSet", case=False, regex=False).sum())
    return out, audit


def load_clean_external_family_corpus() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not V255_CANONICAL.exists():
        return pd.DataFrame(), _empty_external_audit(0)
    raw = pd.read_csv(V255_CANONICAL, low_memory=False)
    clean, audit = filter_clean_external_corpus(raw)
    return clean, audit


def _external_phase(value: Any, event_index: int) -> str:
    text = str(value).lower()
    if "serve" in text or "receive" in text:
        return "receive"
    if "third" in text:
        return "third"
    if "fourth" in text:
        return "fourth"
    if "rally" in text:
        return "rally"
    return phase_bin(event_index + 1)


def _external_depth(row: pd.Series) -> str:
    y = pd.to_numeric(row.get("landing_y", np.nan), errors="coerce")
    if not np.isfinite(y):
        return "unknown"
    ay = abs(float(y))
    if ay < 0.25:
        return "short"
    if ay < 0.75:
        return "half"
    return "long"


def prepare_external_prior_rows(corpus: pd.DataFrame) -> pd.DataFrame:
    if corpus.empty:
        return pd.DataFrame(columns=["phase_bin", "prev_family", "depth_bin", "coarse_family", "terminal_like"])
    out = corpus.copy()
    if "phase_bin" not in out:
        out["phase_bin"] = [
            _external_phase(p, i)
            for p, i in zip(
                out.get("phase", pd.Series("", index=out.index)),
                pd.to_numeric(out.get("event_index", pd.Series(range(len(out)), index=out.index)), errors="coerce").fillna(0).astype(int),
            )
        ]
    if "depth_bin" not in out:
        out["depth_bin"] = out.apply(_external_depth, axis=1)
    if "prev_family" not in out:
        seq = out.get("sequence_id", pd.Series("global", index=out.index)).astype(str)
        out["prev_family"] = out.groupby(seq, sort=False)["coarse_family"].shift(1).fillna("Zero")
    out["coarse_family"] = out.get("coarse_family", "Zero").astype(str).where(lambda s: s.isin(FAMILIES + ["Serve"]), "Zero")
    out["coarse_family"] = out["coarse_family"].replace({"Serve": "Attack"})
    out["terminal_like"] = out.get("terminal_like", False).astype(bool)
    return out[["phase_bin", "prev_family", "depth_bin", "coarse_family", "terminal_like"]].copy()


def build_external_response_prior(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = prepare_external_prior_rows(corpus)
    columns = ["phase_bin", "prev_family", "depth_bin"] + [f"v286_ext_family_{f}" for f in FAMILIES] + ["v286_ext_terminal_rate"]
    if rows.empty:
        return pd.DataFrame([["rally", "Zero", "unknown", 0.25, 0.25, 0.25, 0.25, 0.0]], columns=columns)
    records = []
    for key, group in rows.groupby(["phase_bin", "prev_family", "depth_bin"], dropna=False):
        counts = group["coarse_family"].value_counts()
        total = float(len(group) + len(FAMILIES))
        rec = {"phase_bin": str(key[0]), "prev_family": str(key[1]), "depth_bin": str(key[2])}
        for family in FAMILIES:
            rec[f"v286_ext_family_{family}"] = float((counts.get(family, 0) + 1.0) / total)
        rec["v286_ext_terminal_rate"] = float((group["terminal_like"].sum() + 1.0) / (len(group) + 2.0))
        records.append(rec)
    prior = pd.DataFrame(records, columns=columns)
    family_cols = [c for c in prior.columns if c.startswith("v286_ext_family_")]
    prior[family_cols] = normalize_rows_safe(prior[family_cols].to_numpy())
    return prior


def add_external_response_features(aicup_rows: pd.DataFrame, external_prior: pd.DataFrame) -> pd.DataFrame:
    out = aicup_rows.copy()
    if "phase_bin" not in out or "lag0_action_family" not in out or "lag0_point_depth" not in out:
        out = annotate_prefix_context(out)
    family_cols = [f"v286_ext_family_{f}" for f in FAMILIES]
    use_prior = external_prior.copy()
    if use_prior.empty:
        use_prior = build_external_response_prior(pd.DataFrame())
    global_defaults = {c: float(use_prior[c].mean()) for c in family_cols + ["v286_ext_terminal_rate"] if c in use_prior}
    global_family = np.array([global_defaults.get(c, 0.25) for c in family_cols], dtype=float).reshape(1, -1)
    global_family = normalize_rows_safe(global_family)[0]
    for c, v in zip(family_cols, global_family):
        global_defaults[c] = float(v)
    left = out.assign(prev_family=out["lag0_action_family"].astype(str), depth_bin=out["lag0_point_depth"].astype(str))
    merged = left.merge(use_prior, on=["phase_bin", "prev_family", "depth_bin"], how="left", validate="many_to_one")
    for col in family_cols + ["v286_ext_terminal_rate"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(global_defaults.get(col, 0.0))
    return merged.drop(columns=["prev_family", "depth_bin"], errors="ignore")


def build_support_tables(rows: pd.DataFrame, y: np.ndarray) -> dict[str, pd.DataFrame]:
    base = rows[["phase_bin", "lag0_actionId", "lag0_pointId", "lag0_action_family", "lag0_point_depth"]].copy()
    base["candidate_action"] = np.asarray(y, dtype=int)
    base = base[base["candidate_action"].isin(WEAK_ACTIONS.tolist())].copy()

    def table(cols: list[str]) -> pd.DataFrame:
        if base.empty:
            return pd.DataFrame(columns=cols + ["support_count"])
        return base.groupby(cols, dropna=False).size().reset_index(name="support_count")

    return {
        "exact": table(["phase_bin", "lag0_actionId", "lag0_pointId", "candidate_action"]),
        "family_depth": table(["phase_bin", "lag0_action_family", "lag0_point_depth", "candidate_action"]),
        "phase": table(["phase_bin", "candidate_action"]),
        "global": table(["candidate_action"]),
    }


def add_support_counts(frame: pd.DataFrame, support_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = frame.copy()
    out["_row_id"] = np.arange(len(out))
    counts = np.zeros(len(out), dtype=float)
    specs = [
        ("exact", ["phase_bin", "lag0_actionId", "lag0_pointId", "candidate_action"]),
        ("family_depth", ["phase_bin", "lag0_action_family", "lag0_point_depth", "candidate_action"]),
        ("phase", ["phase_bin", "candidate_action"]),
        ("global", ["candidate_action"]),
    ]
    for name, keys in specs:
        tab = support_tables.get(name)
        if tab is None or tab.empty:
            continue
        merged = out[["_row_id"] + keys].merge(tab[keys + ["support_count"]], on=keys, how="left")
        vals = pd.to_numeric(merged["support_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        counts = np.maximum(counts, vals)
    out["support_count"] = counts.astype(int)
    return out.drop(columns=["_row_id"])


def feature_columns(rows: pd.DataFrame) -> list[str]:
    blocked = {
        "rally_uid",
        "match",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "fold",
        "phase_bin",
        "lag0_action_family",
        "lag0_point_depth",
        "score_pressure_bin",
    }
    return [c for c in rows.columns if c not in blocked and pd.api.types.is_numeric_dtype(rows[c])]


def specialist_features(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    num = rows[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cats = [c for c in ["phase_bin", "lag0_action_family", "lag0_point_depth", "score_pressure_bin"] if c in rows.columns]
    if cats:
        num = pd.concat([num, pd.get_dummies(rows[cats].astype(str), prefix=cats, dtype=float)], axis=1)
    return num.astype(float)


def align_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out:
            out[col] = 0.0
    return out[columns].astype(float)


def margin_entropy(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = normalize_rows_safe(prob)
    part = np.partition(p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    entropy = -np.sum(np.clip(p, 1e-9, 1.0) * np.log(np.clip(p, 1e-9, 1.0)), axis=1)
    return margin, entropy


def attach_teacher_features(rows: pd.DataFrame, v173_prob: np.ndarray, r166_prob: np.ndarray) -> pd.DataFrame:
    out = rows.copy()
    v173_margin, v173_entropy = margin_entropy(v173_prob)
    r166_margin, r166_entropy = margin_entropy(r166_prob)
    out["v173_margin"] = v173_margin
    out["v173_entropy"] = v173_entropy
    out["r166_margin"] = r166_margin
    out["r166_entropy"] = r166_entropy
    for action in WEAK_ACTIONS:
        out[f"v173_p_{action}"] = v173_prob[:, int(action)]
        out[f"r166_p_{action}"] = r166_prob[:, int(action)]
    return out


def train_specialists(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_all = specialist_features(rows, feature_cols)
    x_test_base = specialist_features(test_rows, feature_cols)
    columns = list(x_all.columns)
    x_test = align_columns(x_test_base, columns)
    folds = rows["fold"].astype(int).to_numpy() if "fold" in rows else None
    if folds is None:
        groups = rows["match"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        fold_pairs = list(splitter.split(x_all, y, groups))
    else:
        fold_pairs = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(np.unique(folds))]

    oof = np.zeros((len(rows), len(WEAK_ACTIONS)), dtype=float)
    test_sum = np.zeros((len(test_rows), len(WEAK_ACTIONS)), dtype=float)
    metrics: list[dict[str, Any]] = []
    for j, action in enumerate(WEAK_ACTIONS):
        fitted = 0
        for fold_id, (train_idx, valid_idx) in enumerate(fold_pairs):
            target = (np.asarray(y)[train_idx] == int(action)).astype(int)
            if len(np.unique(target)) < 2:
                continue
            clf = ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=286 + int(action) * 10 + int(fold_id),
                n_jobs=-1,
            )
            clf.fit(x_all.iloc[train_idx][columns], target)
            oof[valid_idx, j] = clf.predict_proba(x_all.iloc[valid_idx][columns])[:, 1]
            test_sum[:, j] += clf.predict_proba(x_test)[:, 1]
            fitted += 1
        if fitted:
            test_sum[:, j] /= float(fitted)
        metrics.append(
            {
                "action": int(action),
                "positive_rows": int(np.sum(np.asarray(y) == int(action))),
                "mean_oof_score": float(oof[:, j].mean()),
                "mean_test_score": float(test_sum[:, j].mean()),
                "fitted_folds": int(fitted),
            }
        )
    return oof, test_sum, metrics


def oof_specialist_frame(
    rows: pd.DataFrame,
    y: np.ndarray,
    anchor: np.ndarray,
    scores: np.ndarray,
    support_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "rally_uid": rows["rally_uid"].astype(int).to_numpy(),
            "fold": rows.get("fold", pd.Series(0, index=rows.index)).astype(int).to_numpy(),
            "y_true_action": np.asarray(y, dtype=int),
            "anchor_action": np.asarray(anchor, dtype=int),
        }
    )
    for j, action in enumerate(WEAK_ACTIONS):
        frame[f"specialist_p_{action}"] = scores[:, j]
        cand = rows[["phase_bin", "lag0_actionId", "lag0_pointId", "lag0_action_family", "lag0_point_depth"]].copy()
        cand["candidate_action"] = int(action)
        frame[f"support_{action}"] = add_support_counts(cand, support_tables)["support_count"].to_numpy()
    return frame


def candidate_frame_for_scores(
    rows: pd.DataFrame,
    anchor: np.ndarray,
    scores: np.ndarray,
    support_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces = []
    for j, action in enumerate(WEAK_ACTIONS):
        frame = rows[["phase_bin", "lag0_actionId", "lag0_pointId", "lag0_action_family", "lag0_point_depth"]].copy()
        frame["row_id"] = np.arange(len(rows))
        frame["anchor_action"] = np.asarray(anchor, dtype=int)
        frame["candidate_action"] = int(action)
        frame["specialist_score"] = scores[:, j]
        frame = add_support_counts(frame, support_tables)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True, sort=False)


def choose_row_candidates(candidate_frame: pd.DataFrame, threshold: float, min_support: int) -> pd.DataFrame:
    gated = candidate_frame.loc[build_candidate_gate(candidate_frame, threshold, min_support)].copy()
    if gated.empty:
        return pd.DataFrame(columns=["row_id", "candidate_action", "specialist_score", "support_count"])
    preferred = {0, 3, 8, 9}
    records = []
    for row_id, group in gated.groupby("row_id", sort=False):
        group = group.sort_values(["specialist_score", "support_count"], ascending=[False, False]).copy()
        best = group.iloc[0].copy()
        if int(best["candidate_action"]) not in preferred:
            pref = group[group["candidate_action"].isin(preferred)]
            if not pref.empty:
                pref_best = pref.sort_values(["specialist_score", "support_count"], ascending=[False, False]).iloc[0]
                if float(best["specialist_score"]) - float(pref_best["specialist_score"]) <= 0.03:
                    best = pref_best.copy()
        records.append(best)
    return pd.DataFrame(records)


def apply_candidate_cap(
    anchor: np.ndarray,
    row_candidates: pd.DataFrame,
    max_churn: float,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(anchor, dtype=int).copy()
    selected = np.zeros(n_rows, dtype=bool)
    if row_candidates.empty:
        return pred, selected
    max_rows = int(math.floor(n_rows * float(max_churn)))
    if max_rows <= 0:
        return pred, selected
    ranked = row_candidates.sort_values(["specialist_score", "support_count"], ascending=[False, False]).head(max_rows)
    ids = ranked["row_id"].astype(int).to_numpy()
    selected[ids] = True
    pred[ids] = ranked["candidate_action"].astype(int).to_numpy()
    return pred, selected


def class_f1(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict[int, float]:
    values = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {int(label): float(value) for label, value in zip(labels, values)}


def public_like_weights(rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
    try:
        from analysis_v233_public_like_validation_lab import density_ratio_weights

        cols = ["phase_bin", "lag0_action_family", "lag0_point_depth"]
        return density_ratio_weights(rows[cols].astype(str), test_rows[cols].astype(str), cols)
    except Exception:
        return np.ones(len(rows), dtype=float)


def weighted_macro_f1_local(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    scores = []
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    for label in ACTION_CLASSES:
        yt = y == label
        yp = pred == label
        tp = float(weights[yt & yp].sum())
        fp = float(weights[~yt & yp].sum())
        fn = float(weights[yt & ~yp].sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom <= 0 else 2 * tp / denom)
    return float(np.mean(scores))


def evaluate_predictions(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    anchor: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
    test_selected: np.ndarray,
    max_churn: float,
    threshold: float,
    min_support: int,
) -> dict[str, Any]:
    base_macro = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    macro = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    weak_base = f1_score(y, anchor, labels=WEAK_ACTIONS.tolist(), average="macro", zero_division=0)
    weak_score = f1_score(y, pred, labels=WEAK_ACTIONS.tolist(), average="macro", zero_division=0)
    prot_base = f1_score(y, anchor, labels=PROTECTED_ACTIONS.tolist(), average="macro", zero_division=0)
    prot_score = f1_score(y, pred, labels=PROTECTED_ACTIONS.tolist(), average="macro", zero_division=0)
    public_base = weighted_macro_f1_local(y, anchor, weights)
    public_score = weighted_macro_f1_local(y, pred, weights)
    changed = pred != anchor
    changed_rows = int(changed.sum())
    changed_precision = float(np.mean(pred[changed] == y[changed])) if changed_rows else 0.0
    deltas = {str(k): float(class_f1(y, pred, ACTION_CLASSES)[k] - class_f1(y, anchor, ACTION_CLASSES)[k]) for k in ACTION_CLASSES}
    serve_unchanged = bool(np.all(pred[np.isin(anchor, SERVE_ACTIONS)] == anchor[np.isin(anchor, SERVE_ACTIONS)]))
    test_churn = float(test_selected.mean()) if len(test_selected) else 0.0
    clean_probe = (
        macro - base_macro > 0
        and weak_score - weak_base > 0
        and prot_score - prot_base >= -0.001
        and public_score - public_base >= 0
        and changed_rows > 0
        and test_churn <= float(max_churn) + 1e-12
        and serve_unchanged
    )
    return {
        "candidate": name,
        "threshold": float(threshold),
        "min_support": int(min_support),
        "max_churn": float(max_churn),
        "action_macro_f1": float(macro),
        "delta_vs_v173": float(macro - base_macro),
        "weak_mean_f1": float(weak_score),
        "weak_mean_delta": float(weak_score - weak_base),
        "protected_mean_f1": float(prot_score),
        "protected_mean_delta": float(prot_score - prot_base),
        "public_like_action_macro_f1": float(public_score),
        "public_like_delta_vs_v173": float(public_score - public_base),
        "changed_rows": changed_rows,
        "changed_precision": changed_precision,
        "test_changed_rows": int(test_selected.sum()),
        "test_churn": test_churn,
        "serve_15_18_unchanged": serve_unchanged,
        "candidate_tier": "clean_probe" if clean_probe else "diagnostic_only",
        "class_f1_delta_json": json.dumps(deltas, sort_keys=True),
    }


def write_submission(name: str, action: np.ndarray, anchor_sub: pd.DataFrame) -> dict[str, Any]:
    out = pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return {
        "submission": name,
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(out)),
        "action_churn_vs_anchor": float(np.mean(out["actionId"].to_numpy() != anchor_sub["actionId"].to_numpy())),
    }


def candidate_filename(max_churn: float) -> str:
    fixed = {0.0025: "0p0025", 0.005: "0p005", 0.010: "0p010", 0.020: "0p020"}
    token = fixed.get(round(float(max_churn), 4), f"{max_churn:.4f}".rstrip("0").replace(".", "p"))
    return f"submission_v286_weak_spec_churn{token}__pv261cap1__sr121.csv"


def build_class_report(y: np.ndarray, anchor: np.ndarray, best_pred: np.ndarray) -> pd.DataFrame:
    anchor_f1 = class_f1(y, anchor, ACTION_CLASSES)
    best_f1 = class_f1(y, best_pred, ACTION_CLASSES)
    return pd.DataFrame(
        [
            {
                "action": int(action),
                "family": action_family(action),
                "is_weak": int(action in WEAK_ACTIONS),
                "is_protected": int(action in PROTECTED_ACTIONS),
                "v173_f1": float(anchor_f1[action]),
                "v286_f1": float(best_f1[action]),
                "delta": float(best_f1[action] - anchor_f1[action]),
            }
            for action in ACTION_CLASSES
        ]
    )


def _set_pickle_dataclasses() -> None:
    from analysis_v209_action_selector_reranker import GrUTuning, TransformerTuning, V3Tuning

    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v286*.csv"):
        stale.unlink()

    stage1_train, _ = build_aicup_prefix_frames(max_lag=6)
    external, external_audit = load_clean_external_family_corpus()
    external_prior = build_external_response_prior(external)
    stage1 = add_external_response_features(stage1_train, external_prior)
    (OUTDIR / "v286_external_audit.json").write_text(json.dumps(external_audit, indent=2, sort_keys=True), encoding="utf-8")
    stage1.to_csv(OUTDIR / "v286_stage1_pretrain_table.csv", index=False)

    _set_pickle_dataclasses()
    from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
    from analysis_v195_distribution_matched_point_gru import prepare_data
    from analysis_v209_action_selector_reranker import distill_v173_soft_anchor, rebuild_r166_best_action

    data = prepare_data()
    state = rebuild_v173_best_actions()
    rows = annotate_prefix_context(state["rows"].reset_index(drop=True))
    test_rows = annotate_prefix_context(state["test_rows"].reset_index(drop=True))
    rows = add_external_response_features(rows, external_prior)
    test_rows = add_external_response_features(test_rows, external_prior)
    y = rows["next_actionId"].astype(int).to_numpy()
    v173_oof = np.asarray(state["v173_pred_oof"], dtype=int)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    if len(anchor_sub) != len(test_rows):
        raise ValueError(f"Anchor rows {len(anchor_sub)} do not match test rows {len(test_rows)}")
    v173_test = anchor_sub["actionId"].astype(int).to_numpy()

    v173_prob_oof, v173_prob_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    _r166_pred_oof, _r166_pred_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(rows, test_rows)
    teacher_oof = normalize_rows_safe(0.80 * v173_prob_oof + 0.20 * normalize_rows_safe(r166_prob_oof))
    teacher_test = normalize_rows_safe(0.80 * v173_prob_test + 0.20 * normalize_rows_safe(r166_prob_test))
    rows = attach_teacher_features(rows, teacher_oof, normalize_rows_safe(r166_prob_oof))
    test_rows = attach_teacher_features(test_rows, teacher_test, normalize_rows_safe(r166_prob_test))

    cols = feature_columns(rows)
    for col in cols:
        if col not in test_rows:
            test_rows[col] = 0
    scores_oof, scores_test, specialist_metrics = train_specialists(rows, test_rows, y, cols)
    support_tables = build_support_tables(rows, y)
    oof_frame = oof_specialist_frame(rows, y, v173_oof, scores_oof, support_tables)
    oof_frame.to_csv(OUTDIR / "v286_specialist_oof.csv", index=False)

    weights = public_like_weights(rows, test_rows)
    oof_candidates = candidate_frame_for_scores(rows, v173_oof, scores_oof, support_tables)
    test_candidates = candidate_frame_for_scores(test_rows, v173_test, scores_test, support_tables)

    records: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    exported_by_churn: dict[float, dict[str, Any]] = {}
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        for min_support in [10, 20, 40]:
            oof_rows = choose_row_candidates(oof_candidates, threshold, min_support)
            test_rows_cand = choose_row_candidates(test_candidates, threshold, min_support)
            for max_churn in [0.0025, 0.005, 0.010, 0.020]:
                pred, selected = apply_candidate_cap(v173_oof, oof_rows, max_churn, len(v173_oof))
                test_pred, test_selected = apply_candidate_cap(v173_test, test_rows_cand, max_churn, len(v173_test))
                name = f"v286_t{str(threshold).replace('.', 'p')}_s{min_support}_c{str(max_churn).replace('.', 'p')}"
                rec = evaluate_predictions(name, y, pred, v173_oof, selected, weights, test_selected, max_churn, threshold, min_support)
                records.append(rec)
                predictions[name] = pred
                test_predictions[name] = test_pred
                if max_churn not in exported_by_churn:
                    exported_by_churn[max_churn] = write_submission(candidate_filename(max_churn), test_pred, anchor_sub)

    # Diagnostic thematic local submissions from the least strict candidate pool.
    base_test_rows = choose_row_candidates(test_candidates, 0.55, 10)
    for suffix, allowed in [("action0_3", {0, 3}), ("style8_9", {8, 9})]:
        thematic = base_test_rows[base_test_rows["candidate_action"].isin(allowed)].copy()
        test_pred, _selected = apply_candidate_cap(v173_test, thematic, 0.020, len(v173_test))
        write_submission(f"submission_v286_weak_spec_{suffix}__pv261cap1__sr121.csv", test_pred, anchor_sub)

    search = pd.DataFrame(records).sort_values(
        ["candidate_tier", "delta_vs_v173", "public_like_delta_vs_v173", "weak_mean_delta"],
        ascending=[True, False, False, False],
    )
    search.to_csv(OUTDIR / "v286_action_search.csv", index=False)
    best_name = str(search.iloc[0]["candidate"]) if len(search) else ""
    best_pred = predictions.get(best_name, v173_oof)
    class_report = build_class_report(y, v173_oof, best_pred)
    class_report.to_csv(OUTDIR / "v286_class_report.csv", index=False)

    clean = search[search["candidate_tier"].eq("clean_probe")].copy()
    upload_recommendation = "DO_NOT_UPLOAD"
    if not clean.empty:
        candidate = clean.sort_values(["test_churn", "weak_mean_delta"], ascending=[True, False]).iloc[0]
        if (
            float(candidate["delta_vs_v173"]) >= 0.001
            and float(candidate["weak_mean_delta"]) > 0
            and float(candidate["public_like_delta_vs_v173"]) >= 0.0005
            and float(candidate["protected_mean_delta"]) >= -0.001
            and 0.0025 <= float(candidate["test_churn"]) <= 0.02
        ):
            upload_recommendation = "REVIEW_LOWEST_CHURN_CLEAN_PROBE"

    generated = sorted(str(p.relative_to(ROOT)) for p in OUTDIR.glob("submission_v286*.csv"))
    best = search.iloc[0].to_dict() if len(search) else {}
    report = {
        "version": "V286",
        "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
        "weak_actions": WEAK_ACTIONS.tolist(),
        "protected_actions": PROTECTED_ACTIONS.tolist(),
        "external_audit": external_audit,
        "distill_metrics": distill_metrics,
        "specialist_metrics": specialist_metrics,
        "best_candidate": best,
        "generated_submissions": generated,
        "upload_recommendation": upload_recommendation,
        "copied_to_upload_or_selected": False,
    }
    (OUTDIR / "v286_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# V286 weak-action specialist pretraining",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        f"Weak actions: {WEAK_ACTIONS.tolist()}",
        f"Protected actions: {PROTECTED_ACTIONS.tolist()}",
        f"External rows used: {external_audit['used_rows']} / {external_audit['raw_rows']}",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Public-like delta vs V173: {float(best.get('public_like_delta_vs_v173', 0.0)):.6f}",
        f"Weak mean delta: {float(best.get('weak_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Upload recommendation: {upload_recommendation}",
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v286_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "best_candidate": report["best_candidate"].get("candidate", ""),
                "best_delta_vs_v173": report["best_candidate"].get("delta_vs_v173", 0.0),
                "upload_recommendation": report["upload_recommendation"],
                "generated_submissions": len(report["generated_submissions"]),
                "outdir": str(OUTDIR.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
