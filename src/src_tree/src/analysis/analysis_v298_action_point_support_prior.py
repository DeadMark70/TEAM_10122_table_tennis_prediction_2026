"""V298 action-point support prior explorer.

Point-only clean diagnostic over the V261 anchor.  Uses a fold-safe-ish
rebuilt V294 action-conditioned base plus empirical P(point | anchor_action)
support to propose very low-churn point edits.  Action/server remain fixed.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v298_action_point_support_prior"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v298_action_point_support_prior.py"
V294_DIR = ROOT / "v294_point_oof_artifact_builder"
ANCHOR_PATH = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
POINT_CLASSES = list(range(10))
LONG = {7, 8, 9}
RARE = {1, 3, 4}


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= 1e-12
    if np.any(bad):
        arr[bad] = 1.0 / arr.shape[1]
        denom = arr.sum(axis=1, keepdims=True)
    return arr / denom


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def build_action_point_prior(actions: np.ndarray, points: np.ndarray, smoothing: float = 5.0) -> np.ndarray:
    table = np.full((19, 10), float(smoothing), dtype=float)
    for a, p in zip(np.asarray(actions, dtype=int), np.asarray(points, dtype=int)):
        if 0 <= a < 19 and 0 <= p < 10:
            table[a, p] += 1.0
    return table / table.sum(axis=1, keepdims=True)


def support_candidates(base: np.ndarray, actions: np.ndarray, proba: np.ndarray, prior: np.ndarray, mode: str) -> pd.DataFrame:
    base = np.asarray(base, dtype=int)
    actions = np.asarray(actions, dtype=int)
    proba = normalize_rows_safe(proba)
    rows: list[dict[str, Any]] = []
    for row_id, (b, a) in enumerate(zip(base, actions)):
        if not (0 <= int(a) < 19):
            continue
        score_vec = proba[row_id] * prior[int(a)]
        cand = int(np.argmax(score_vec))
        if cand == int(b):
            continue
        if mode == "long789" and not (int(b) in LONG and cand in LONG):
            continue
        if mode == "no_point0" and cand == 0:
            continue
        if mode == "no_rare134" and cand in RARE:
            continue
        score = float(score_vec[cand] - score_vec[int(b)])
        if score <= 0.0:
            continue
        rows.append({"row_id": row_id, "candidate_point": cand, "score": score, "mode": mode})
    if not rows:
        return pd.DataFrame(columns=["row_id", "candidate_point", "score", "mode"])
    return pd.DataFrame(rows)


def apply_candidates(base: np.ndarray, candidates: pd.DataFrame, cap: float) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.asarray(base, dtype=int).copy()
    max_rows = int(math.floor(len(pred) * cap))
    if max_rows <= 0 or candidates.empty:
        return pred, candidates.head(0).copy()
    selected = candidates.sort_values(["score", "row_id"], ascending=[False, True]).head(max_rows).copy()
    rows = selected["row_id"].astype(int).to_numpy()
    pred[rows] = selected["candidate_point"].astype(int).to_numpy()
    return pred, selected


def write_submission(path: Path, points: np.ndarray, anchor: pd.DataFrame) -> None:
    out = anchor.copy()
    out["pointId"] = np.asarray(points, dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if len(out) != EXPECTED_ROWS:
        raise ValueError("bad submission rows")
    out.to_csv(path, index=False)


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_base = pd.read_csv(V294_DIR / "v294_point_oof_base.csv")
    oof_proba = normalize_rows_safe(np.load(V294_DIR / "v294_point_oof_proba.npy"))
    test_proba = normalize_rows_safe(np.load(V294_DIR / "v294_point_test_proba.npy"))
    anchor = pd.read_csv(ANCHOR_PATH)
    y_true = oof_base["y_true_point"].astype(int).to_numpy()
    base_oof = oof_base["base_point_oof"].astype(int).to_numpy()
    action_oof = oof_base["anchor_action"].astype(int).to_numpy()
    base_test = anchor["pointId"].astype(int).to_numpy()
    action_test = anchor["actionId"].astype(int).to_numpy()
    prior = build_action_point_prior(action_oof, y_true)

    variants = [(mode, cap) for mode in ["all", "long789", "no_point0", "no_rare134"] for cap in [0.0025, 0.005, 0.01]]
    base_score = macro_f1(y_true, base_oof)
    records: list[dict[str, Any]] = []
    audits: list[pd.DataFrame] = []
    generated: list[str] = []
    for mode, cap in variants:
        name = f"v298_support_{mode}_cap{str(cap).replace('.', 'p')}"
        train_c = support_candidates(base_oof, action_oof, oof_proba, prior, mode)
        test_c = support_candidates(base_test, action_test, test_proba, prior, mode)
        pred_oof, selected_oof = apply_candidates(base_oof, train_c, cap)
        pred_test, selected_test = apply_candidates(base_test, test_c, cap)
        delta = macro_f1(y_true, pred_oof) - base_score
        point0_delta = float(np.mean(pred_test == 0) - np.mean(base_test == 0))
        rec = {
            "candidate": name,
            "mode": mode,
            "cap": cap,
            "point_macro_f1": macro_f1(y_true, pred_oof),
            "base_point_macro_f1": base_score,
            "delta_vs_v294_base": delta,
            "oof_changed_rows": int(len(selected_oof)),
            "test_changed_rows": int(len(selected_test)),
            "point_churn": float(len(selected_test) / len(base_test)),
            "point0_rate_delta": point0_delta,
            "upload_recommendation": "REVIEW_UPLOAD" if delta >= 0.0015 and 3 <= len(selected_test) <= 20 and point0_delta <= 0.003 else "DO_NOT_UPLOAD",
        }
        path = OUT_DIR / f"submission_{name}__v173action_r121server.csv"
        write_submission(path, pred_test, anchor)
        rec["path"] = str(path.relative_to(ROOT))
        records.append(rec)
        generated.append(str(path.relative_to(ROOT)))
        if not selected_test.empty:
            audit = selected_test.copy()
            audit["candidate"] = name
            audit["rally_uid"] = anchor.iloc[audit["row_id"].astype(int).to_numpy()]["rally_uid"].to_numpy()
            audit["base_point"] = base_test[audit["row_id"].astype(int).to_numpy()]
            audits.append(audit)

    search = pd.DataFrame(records).sort_values(["upload_recommendation", "delta_vs_v294_base"], ascending=[False, False])
    search.to_csv(OUT_DIR / "v298_candidate_search.csv", index=False)
    if audits:
        pd.concat(audits, ignore_index=True).to_csv(OUT_DIR / "v298_changed_row_audit.csv", index=False)
    best = search.iloc[0].to_dict()
    report = {
        "version": "V298",
        "anchor_submission": str(ANCHOR_PATH.relative_to(ROOT)),
        "best_candidate": best,
        "generated_submissions": generated,
        "upload_recommendation": "REVIEW_UPLOAD" if search["upload_recommendation"].eq("REVIEW_UPLOAD").any() else "DO_NOT_UPLOAD",
        "no_ttmatch_no_old_server": True,
        "note": "Uses V294 rebuilt action-conditioned OOF/proba, not literal V261 OOF.",
    }
    (OUT_DIR / "v298_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "v298_report.md").write_text(
        "# V298 action-point support prior\n\n"
        f"- Best: `{best['candidate']}`\n"
        f"- Delta vs V294 base: `{float(best['delta_vs_v294_base']):.6f}`\n"
        f"- Test changed rows: `{int(best['test_changed_rows'])}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n",
        encoding="utf-8",
    )
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), SRC_DEST)
    return report


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(json.dumps({
        "outdir": str(OUT_DIR.relative_to(ROOT)),
        "best_candidate": best["candidate"],
        "best_delta_vs_v294_base": best["delta_vs_v294_base"],
        "best_test_changed_rows": best["test_changed_rows"],
        "generated_submissions": len(report["generated_submissions"]),
        "upload_recommendation": report["upload_recommendation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
