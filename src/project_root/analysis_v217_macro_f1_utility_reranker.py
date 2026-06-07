"""V217 macro-F1 utility action reranker.

V217 scores candidate action changes by expected macro-F1 delta instead of raw
candidate correctness.  It uses existing action candidate sources plus V216
terminal-aware candidates, then exports ultra-low-churn action-only submissions.

Point remains V188 cap5 and server remains R121.  No external rows and no
TTMATCH are read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    V3Tuning,
    GrUTuning,
    TransformerTuning,
    add_probability_features,
    action_point_compatibility,
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
from analysis_v216_terminal_action_tuner import (
    POINT_ANCHOR,
    SERVER_ANCHOR,
    build_terminal_action_candidate,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v217_macro_f1_utility_reranker")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v217_macro_f1_utility_reranker.py")
CAPS = [0.002, 0.005, 0.01]


def macro_f1_score(y: np.ndarray, pred: np.ndarray, labels=ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def class_f1_delta_for_change(y: np.ndarray, pred: np.ndarray, row: int, new_label: int, labels=ACTION_CLASSES) -> float:
    before = macro_f1_score(y, pred, labels)
    after_pred = np.asarray(pred, dtype=int).copy()
    after_pred[int(row)] = int(new_label)
    return macro_f1_score(y, after_pred, labels) - before


def expected_macro_f1_delta(p_correct: np.ndarray, gain_if_correct: np.ndarray, loss_if_wrong: np.ndarray) -> np.ndarray:
    p = np.asarray(p_correct, dtype=float)
    return p * np.asarray(gain_if_correct, dtype=float) - (1.0 - p) * np.asarray(loss_if_wrong, dtype=float)


def selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "candidate_action",
        "candidate_family",
        "anchor_action",
        "anchor_family",
        "differs_anchor",
        "is_anchor",
        "agreement_count",
        "prefix_len",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "anchor_prob_on_anchor",
        "anchor_prob_on_candidate",
        "v208_prob_on_anchor",
        "v208_prob_on_candidate",
        "v208_minus_anchor_candidate_prob",
        "action_point_compat",
        "anchor_point_compat",
        "compat_delta_vs_anchor",
    ]
    prob_cols = [c for c in frame.columns if c.endswith("_p_candidate") or c.endswith("_rank_candidate") or c.endswith("_margin") or c.endswith("_entropy")]
    cols = [c for c in numeric + prob_cols if c in frame.columns]
    x = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cats = [c for c in ["source", "audit_phase", "audit_lag0_action_family", "audit_lag0_depth"] if c in frame.columns]
    if cats:
        x = pd.concat([x, pd.get_dummies(frame[cats].astype(str), prefix=cats, dtype=float)], axis=1)
    return x.astype(float)


def align_columns(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[cols].astype(float)


def train_correctness_model(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.20, max_iter=1000, random_state=217)
    clf.fit(x, y)
    return clf


def add_features(frame: pd.DataFrame, probs: dict[str, np.ndarray], point_labels: np.ndarray, compat: np.ndarray | None) -> pd.DataFrame:
    return add_probability_features(frame, probs, "v173_anchor", "utility_model", point_labels, compat)


def row_delta_tables(y: np.ndarray, anchor: np.ndarray, candidate_row_ids: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_ids = np.asarray(candidate_row_ids, dtype=int)
    gain = np.zeros(len(candidates), dtype=float)
    loss = np.zeros(len(candidates), dtype=float)
    for i, (row_id, cand) in enumerate(zip(row_ids, candidates)):
        if int(cand) == int(anchor[row_id]):
            continue
        correct_delta = class_f1_delta_for_change(y, anchor, int(row_id), y[row_id])
        wrong_delta = class_f1_delta_for_change(y, anchor, int(row_id), cand)
        gain[i] = max(0.0, correct_delta)
        loss[i] = max(0.0, -wrong_delta)
    return gain, loss


def fit_expected_utility(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    sources_oof: dict[str, np.ndarray],
    sources_test: dict[str, np.ndarray],
    probs_oof: dict[str, np.ndarray],
    probs_test: dict[str, np.ndarray],
    point_oof: np.ndarray,
    point_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    base_frame = build_action_candidate_frame(rows, sources_oof, truth=y, anchor_name="v173")
    test_frame = build_action_candidate_frame(test_rows, sources_test, truth=None, anchor_name="v173")
    oof_best = np.zeros(len(rows), dtype=int)
    oof_utility = np.full(len(rows), -np.inf, dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_rows = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows_mask = ~valid_rows
        train_ids = set(np.where(train_rows_mask)[0])
        valid_ids = set(np.where(valid_rows)[0])
        compat = action_point_compatibility(y[train_rows_mask], point_oof[train_rows_mask], smoothing=1.0)
        train = add_features(base_frame[base_frame["row_id"].isin(train_ids)].copy(), probs_oof, point_oof, compat)
        valid = add_features(base_frame[base_frame["row_id"].isin(valid_ids)].copy(), probs_oof, point_oof, compat)
        x_train = selector_features(train)
        clf = train_correctness_model(x_train, train["is_correct"].astype(int).to_numpy())
        x_valid = align_columns(selector_features(valid), list(x_train.columns))
        p_valid = clf.predict_proba(x_valid)[:, 1]
        valid_actions = valid["candidate_action"].astype(int).to_numpy()
        valid_rows_idx = valid["row_id"].astype(int).to_numpy()
        gain, loss = row_delta_tables(y, sources_oof["v173"], valid_rows_idx, valid_actions)
        util = expected_macro_f1_delta(p_valid, gain, loss)
        best_action, delta, _ = best_non_anchor_by_score(valid, util)
        order = valid.drop_duplicates("row_id").sort_values("row_id")["row_id"].astype(int).to_numpy()
        oof_best[order] = best_action[order]
        oof_utility[order] = delta[order]
        y_valid = valid["is_correct"].astype(int).to_numpy()
        metrics.append({"fold": int(fold), "auc": float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) > 1 else np.nan, "valid_candidate_rows": int(len(valid))})

    compat_full = action_point_compatibility(y, point_oof, smoothing=1.0)
    train = add_features(base_frame.copy(), probs_oof, point_oof, compat_full)
    test = add_features(test_frame.copy(), probs_test, point_test, compat_full)
    x_train = selector_features(train)
    clf = train_correctness_model(x_train, train["is_correct"].astype(int).to_numpy())
    p_test = clf.predict_proba(align_columns(selector_features(test), list(x_train.columns)))[:, 1]
    test_actions = test["candidate_action"].astype(int).to_numpy()
    test_rows_idx = test["row_id"].astype(int).to_numpy()
    class_gain = np.full(len(point_test), 1.0 / len(ACTION_CLASSES), dtype=float)
    class_loss = np.full(len(point_test), 0.5 / len(ACTION_CLASSES), dtype=float)
    util_test = expected_macro_f1_delta(p_test, class_gain[test_rows_idx], class_loss[test_rows_idx])
    test_best, test_utility, _ = best_non_anchor_by_score(test, util_test)
    return oof_best, oof_utility, test_best, test_utility, metrics


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
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    point_oof, point_test = load_point_anchor_labels(data, point)
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    r166_oof, r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    r184_oof, r184_test = rebuild_r184_sources(state, point)
    v216_cand_oof, _ = build_terminal_action_candidate(v173_oof, point_oof, v173_prob_oof)
    v216_cand_test, _ = build_terminal_action_candidate(v173_test, point["pointId"].astype(int).to_numpy(), v173_prob_test)

    extra_sources_oof = {}
    extra_sources_test = {}
    for name in [
        "submission_v197_v166_r184_agree_attack_control__pv188_r186_w005_cap5__sr121.csv",
        "submission_v197_v166_r184_agree_state_pair__pv188_r186_w005_cap5__sr121.csv",
        "submission_v209_selector_churn0p005__pv188cap5__sr121.csv",
        "submission_v214_full_selector_churn0p005__pv188cap5__sr121.csv",
    ]:
        path = UPLOAD_DIR / name
        if path.exists():
            tag = name.replace("submission_", "").split("__")[0]
            extra_sources_test[tag] = load_sub(path, rally_uids)["actionId"].astype(int).to_numpy()
            # For OOF, only low-churn sources without OOF row labels are used as test-only signals via agreement;
            # skip them to avoid unsafe synthetic OOF labels.

    sources_oof = {"v173": v173_oof, "r166": r166_oof, **r184_oof, "v216_terminal": v216_cand_oof}
    sources_test = {"v173": v173_test, "r166": r166_test, **r184_test, "v216_terminal": v216_cand_test, **extra_sources_test}
    probs_oof = source_probs_for_selector(v173_prob_oof, r166_prob_oof, v173_prob_oof)
    probs_oof["utility_model"] = probs_oof.pop("v208")
    probs_test = source_probs_for_selector(v173_prob_test, r166_prob_test, v173_prob_test)
    probs_test["utility_model"] = probs_test.pop("v208")

    best_oof, utility_oof, best_test, utility_test, fold_metrics = fit_expected_utility(data["rows"], state["test_rows"], y, sources_oof, sources_test, probs_oof, probs_test, point_oof, point_test)
    records = [{"candidate": "v173_anchor", "action_macro_f1": macro_f1_score(y, v173_oof), "delta_vs_v173_anchor": 0.0, "action_churn_vs_v173_anchor": 0.0, "changed_rows": 0}]
    pred_store = {}
    for cap in CAPS:
        pred, changed = select_capped_action_changes(v173_oof, best_oof, utility_oof, cap, min_delta=0.0)
        test_pred, test_changed = select_capped_action_changes(v173_test, best_test, utility_test, cap, min_delta=0.0)
        name = f"v217_macro_utility_churn{str(cap).replace('.', 'p')}"
        score = macro_f1_score(y, pred)
        rec = {
            "candidate": name,
            "action_macro_f1": score,
            "delta_vs_v173_anchor": score - macro_f1_score(y, v173_oof),
            "action_churn_vs_v173_anchor": float(np.mean(pred != v173_oof)),
            "changed_rows": int(changed.sum()),
            "cap": cap,
            "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
            "test_changed_rows": int(test_changed.sum()),
            "mean_utility_changed_oof": float(utility_oof[changed].mean()) if changed.any() else 0.0,
        }
        records.append(rec)
        pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v217_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v217_selector_fold_metrics.csv", index=False)
    generated = []
    for name in [n for n in search["candidate"] if str(n).startswith("v217_")]:
        info = write_submission(f"submission_{name}__pv188cap5__sr121.csv", pred_store[name], point, server)
        info.update(search[search["candidate"].eq(name)].iloc[0].to_dict())
        generated.append(info)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(8).to_dict(orient="records"),
        "notes": [
            "V217 scores action changes by expected macro-F1 utility.",
            "Candidate pool includes V173, R166, R184, and V216 terminal-aware candidates.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v217_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v217_report.md").write_text(
        "# V217 Macro-F1 Utility Reranker\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v217_macro_f1_utility_reranker.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
