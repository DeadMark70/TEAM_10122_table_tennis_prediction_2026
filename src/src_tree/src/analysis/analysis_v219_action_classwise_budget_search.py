"""V219 class-wise weak action budget search.

V218 uses fixed class weights for weak action classes.  V219 searches an OOF
budget independently for each weak class, then exports scaled class-budget
candidates.  This keeps the current no-old anchor intact except for a small
number of class-targeted action changes.

Point remains V188 cap5 and server remains R121.  No external rows and no
TTMATCH are read.
"""

from __future__ import annotations

import __main__
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import analysis_v217_macro_f1_utility_reranker as v217
import analysis_v218_action_weakclass_booster as v218
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    distill_v173_soft_anchor,
    load_point_anchor_labels,
    rebuild_r166_best_action,
    rebuild_r184_sources,
    source_probs_for_selector,
)
from analysis_v216_terminal_action_tuner import (
    POINT_ANCHOR,
    SERVER_ANCHOR,
    build_terminal_action_candidate,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v219_action_classwise_budget")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v219_action_classwise_budget_search.py")

TARGET_CLASSES = [0, 3, 5, 7, 8, 9, 12, 14]
RARE_CLASSES = [8, 9, 12, 14]
MAX_K = {0: 18, 3: 24, 5: 20, 7: 20, 8: 12, 9: 18, 12: 18, 14: 12}
SCALES = [0.50, 0.75, 1.00]


def macro_f1_score(y: np.ndarray, pred: np.ndarray, labels=ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def top_candidates_for_class(frame: pd.DataFrame, target_class: int, score_col: str = "utility") -> pd.DataFrame:
    """Return best positive candidate per row for one target class."""
    target = int(target_class)
    cols = ["row_id", "candidate_action", "anchor_action", score_col]
    work = frame[frame["candidate_action"].astype(int).eq(target)].copy()
    work = work[work["candidate_action"].astype(int) != work["anchor_action"].astype(int)]
    work = work[work[score_col].astype(float) > 0.0]
    if work.empty:
        return pd.DataFrame(columns=cols)
    work = work.sort_values(score_col, ascending=False)
    work = work.drop_duplicates("row_id", keep="first")
    return work[cols].sort_values(score_col, ascending=False).reset_index(drop=True)


def best_budget_for_class(
    y: np.ndarray,
    anchor: np.ndarray,
    frame: pd.DataFrame,
    target_class: int,
    max_k: int,
    labels=ACTION_CLASSES,
) -> dict:
    """Search the best top-k count for one class using OOF macro-F1."""
    base_score = macro_f1_score(y, anchor, labels=labels)
    cand = top_candidates_for_class(frame, target_class)
    if cand.empty:
        return {"action": int(target_class), "best_k": 0, "best_score": base_score, "best_delta": 0.0, "available": 0}
    max_try = min(int(max_k), len(cand))
    best = {"action": int(target_class), "best_k": 0, "best_score": base_score, "best_delta": 0.0, "available": int(len(cand))}
    pred = np.asarray(anchor, dtype=int).copy()
    row_ids = cand["row_id"].astype(int).to_numpy()
    for k in range(1, max_try + 1):
        pred[row_ids[k - 1]] = int(target_class)
        score = macro_f1_score(y, pred, labels=labels)
        delta = score - base_score
        if delta > float(best["best_delta"]):
            best.update({"best_k": int(k), "best_score": float(score), "best_delta": float(delta)})
    return best


def scale_budgets(budgets: dict[int, int], scale: float) -> dict[int, int]:
    out = {}
    for cls, count in budgets.items():
        if int(count) <= 0:
            out[int(cls)] = 0
        else:
            out[int(cls)] = max(1, int(math.floor(float(count) * float(scale))))
    return out


def select_classwise_budget_changes(
    anchor_labels: np.ndarray,
    frame: pd.DataFrame,
    budgets: dict[int, int],
    score_col: str = "utility",
) -> tuple[np.ndarray, np.ndarray]:
    """Select top candidates per class, resolving row conflicts by score."""
    anchor = np.asarray(anchor_labels, dtype=int)
    selected = np.zeros(len(anchor), dtype=bool)
    out = anchor.copy()
    pieces = []
    for cls, budget in budgets.items():
        if int(budget) <= 0:
            continue
        top = top_candidates_for_class(frame, int(cls), score_col=score_col).head(int(budget)).copy()
        if not top.empty:
            pieces.append(top)
    if not pieces:
        return out, selected
    candidates = pd.concat(pieces, ignore_index=True).sort_values(score_col, ascending=False)
    for row in candidates.itertuples(index=False):
        rid = int(row.row_id)
        if selected[rid]:
            continue
        selected[rid] = True
        out[rid] = int(row.candidate_action)
    return out, selected


def class_f1_table(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for label in ACTION_CLASSES:
        f_anchor = f1_score(y, anchor, labels=[label], average="macro", zero_division=0)
        f_pred = f1_score(y, pred, labels=[label], average="macro", zero_division=0)
        rows.append(
            {
                "action": int(label),
                "support": int((np.asarray(y) == int(label)).sum()),
                "anchor_f1": float(f_anchor),
                "candidate_f1": float(f_pred),
                "delta": float(f_pred - f_anchor),
            }
        )
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
    __main__.V3Tuning = v217.V3Tuning
    __main__.GrUTuning = v217.GrUTuning
    __main__.TransformerTuning = v217.TransformerTuning

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
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)
    r166_oof, r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    r184_oof, r184_test = rebuild_r184_sources(state, point)
    v216_oof, _ = build_terminal_action_candidate(v173_oof, point_oof, v173_prob_oof)
    v216_test, _ = build_terminal_action_candidate(v173_test, point["pointId"].astype(int).to_numpy(), v173_prob_test)

    sources_oof = {"v173": v173_oof, "r166": r166_oof, **r184_oof, "v216_terminal": v216_oof}
    sources_test = {"v173": v173_test, "r166": r166_test, **r184_test, "v216_terminal": v216_test}
    probs_oof = source_probs_for_selector(v173_prob_oof, r166_prob_oof, v173_prob_oof)
    probs_oof["utility_model"] = probs_oof.pop("v208")
    probs_test = source_probs_for_selector(v173_prob_test, r166_prob_test, v173_prob_test)
    probs_test["utility_model"] = probs_test.pop("v208")

    oof_frame, test_frame, fold_metrics = v218.candidate_utility_frames(
        data["rows"],
        state["test_rows"],
        y,
        sources_oof,
        sources_test,
        probs_oof,
        probs_test,
        point_oof,
        point_test,
    )
    budget_rows = [best_budget_for_class(y, v173_oof, oof_frame, cls, MAX_K[cls]) for cls in TARGET_CLASSES]
    budget_table = pd.DataFrame(budget_rows)
    budget_table.to_csv(OUTDIR / "v219_class_budget_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v219_selector_fold_metrics.csv", index=False)

    best_budgets = {int(r.action): int(r.best_k) for r in budget_table.itertuples(index=False)}
    rare_budgets = {cls: best_budgets.get(cls, 0) for cls in RARE_CLASSES}
    base_score = macro_f1_score(y, v173_oof)
    records = [
        {
            "candidate": "v173_anchor",
            "action_macro_f1": base_score,
            "delta_vs_v173_anchor": 0.0,
            "action_churn_vs_v173_anchor": 0.0,
            "changed_rows": 0,
            "test_changed_rows": 0,
            "budgets": "{}",
        }
    ]
    generated = []
    class_tables = []
    schemes = []
    for scale in SCALES:
        schemes.append((f"v219_class_budget_s{str(scale).replace('.', 'p')}", scale_budgets(best_budgets, scale)))
    for scale in [0.75, 1.00]:
        schemes.append((f"v219_rare_budget_s{str(scale).replace('.', 'p')}", scale_budgets(rare_budgets, scale)))

    for name, budgets in schemes:
        pred, changed = select_classwise_budget_changes(v173_oof, oof_frame, budgets)
        test_pred, test_changed = select_classwise_budget_changes(v173_test, test_frame, budgets)
        score = macro_f1_score(y, pred)
        rec = {
            "candidate": name,
            "action_macro_f1": score,
            "delta_vs_v173_anchor": score - base_score,
            "action_churn_vs_v173_anchor": float(np.mean(pred != v173_oof)),
            "changed_rows": int(changed.sum()),
            "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
            "test_changed_rows": int(test_changed.sum()),
            "budgets": json.dumps({int(k): int(v) for k, v in budgets.items()}),
            "changed_target_classes": json.dumps(pd.Series(pred[changed]).value_counts().sort_index().to_dict()),
        }
        records.append(rec)
        class_table = class_f1_table(y, v173_oof, pred)
        class_table.insert(0, "candidate", name)
        class_tables.append(class_table)
        info = write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server)
        info.update(rec)
        generated.append(info)

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v219_action_search.csv", index=False)
    pd.concat(class_tables, ignore_index=True).to_csv(OUTDIR / "v219_class_f1_delta.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "budgets": budget_table.to_dict(orient="records"),
        "generated": generated,
        "best": search.head(8).to_dict(orient="records"),
        "notes": [
            "V219 searches OOF top-k budgets independently per weak action class.",
            "Budget-scaled candidates are generated to avoid jumping straight to the OOF optimum.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v219_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v219_report.md").write_text(
        "# V219 Class-Wise Action Budget Search\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v219_action_classwise_budget_search.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
