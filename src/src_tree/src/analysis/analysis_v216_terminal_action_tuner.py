"""V216 terminal-aware action consistency tuner.

The current no-old anchor keeps point fixed at V188 cap5.  V216 uses that fixed
point, especially pointId=0 terminal structure, to propose ultra-low-churn
action fixes over V173.  Point and server are never changed.

No external rows and no TTMATCH are read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    V3Tuning,
    GrUTuning,
    TransformerTuning,
    distill_v173_soft_anchor,
    evaluate_candidate,
    load_point_anchor_labels,
    select_capped_action_changes,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v216_terminal_action_tuner")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v216_terminal_action_tuner.py")

POINT_ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SERVER_ANCHOR = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
CAPS = [0.002, 0.005, 0.01]
SERVE_CLASSES = [15, 16, 17, 18]
TERMINAL_FRIENDLY = [0, 1, 2, 3, 13]
NONTERMINAL_FALLBACK = [10, 13, 1, 5, 2]


def terminal_action_scores(anchor: np.ndarray, point: np.ndarray) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=int)
    point = np.asarray(point, dtype=int)
    score = np.zeros((len(anchor), 19), dtype=float)
    terminal = point == 0
    score[terminal, :] = -0.01
    terminal_bonus = np.array([0.35, 0.10, 0.08, 0.08, 0.06])
    for action_id, bonus in zip(TERMINAL_FRIENDLY, terminal_bonus):
        score[terminal, action_id] += bonus
    score[~terminal, 0] -= 0.45
    score[:, SERVE_CLASSES] -= 0.35
    return score


def build_terminal_action_candidate(anchor: np.ndarray, point: np.ndarray, prior_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor, dtype=int)
    point = np.asarray(point, dtype=int)
    prior = normalize_rows_safe(prior_prob)
    scores = terminal_action_scores(anchor, point)
    utility = scores + 0.12 * np.log(np.clip(prior, 1e-6, 1.0))
    candidate = anchor.copy()
    gain = np.zeros(len(anchor), dtype=float)
    for i in range(len(anchor)):
        if point[i] == 0:
            allowed = TERMINAL_FRIENDLY
        elif anchor[i] == 0 or anchor[i] in SERVE_CLASSES:
            allowed = NONTERMINAL_FALLBACK
        else:
            allowed = [anchor[i]]
        best = max(allowed, key=lambda c: utility[i, c])
        candidate[i] = int(best)
        gain[i] = float(utility[i, best] - utility[i, anchor[i]])
    return candidate, gain


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
    cand_oof, gain_oof = build_terminal_action_candidate(v173_oof, point_oof, v173_prob_oof)
    cand_test, gain_test = build_terminal_action_candidate(v173_test, point["pointId"].astype(int).to_numpy(), v173_prob_test)

    records = [evaluate_candidate("v173_anchor", y, v173_oof, v173_oof, {"scheme": "anchor"})]
    pred_store = {}
    for cap in CAPS:
        pred, changed = select_capped_action_changes(v173_oof, cand_oof, gain_oof, cap, min_delta=0.0)
        test_pred, test_changed = select_capped_action_changes(v173_test, cand_test, gain_test, cap, min_delta=0.0)
        name = f"v216_terminal_action_tune_churn{str(cap).replace('.', 'p')}"
        rec = evaluate_candidate(name, y, pred, v173_oof, {"scheme": "terminal_action_tune", "cap": cap})
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(test_changed.sum())
        rec["changed_to_action0_oof"] = int(np.sum(changed & (pred == 0)))
        rec["changed_from_action0_oof"] = int(np.sum(changed & (v173_oof == 0)))
        rec["changed_from_serve_oof"] = int(np.sum(changed & np.isin(v173_oof, SERVE_CLASSES)))
        records.append(rec)
        pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v216_action_search.csv", index=False)
    pd.DataFrame(distill_metrics).to_csv(OUTDIR / "v216_v173_distill_metrics.csv", index=False)
    generated = []
    for name in [n for n in search["candidate"] if str(n).startswith("v216_")]:
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
            "V216 changes action only.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "Terminal point0 structure is used to tune action0/serve-like classes.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v216_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v216_report.md").write_text(
        "# V216 Terminal-Aware Action Tuner\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v216_terminal_action_tuner.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
