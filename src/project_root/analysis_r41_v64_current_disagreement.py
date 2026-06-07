"""R41: V64-current disagreement and gating analysis.

This is an OOF diagnostic. It aligns the historical V64 OOF predictions with
the current sampled-validation OOF rows, then measures where each branch wins.
It also evaluates a few low-degree gating rules for action/point.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers


OUT_DIR = Path("r41_v64_current_disagreement")
V64_DIR = Path(r"C:\aicup\tenis_new\artifacts\cv_v64_ultimate_knowledge_v1")


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


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def prefix_bin(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    if length == 3:
        return "3"
    if 4 <= length <= 6:
        return "4-6"
    return "7+"


def metrics(meta: pd.DataFrame, action_pred: np.ndarray, point_pred: np.ndarray, server_prob: np.ndarray | None = None) -> dict:
    out = {
        "action_macro_f1": float(f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "point_macro_f1": float(f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)),
    }
    if server_prob is not None:
        out["server_auc"] = float(roc_auc_score(meta["serverGetPoint"], server_prob))
        out["overall"] = float(0.4 * out["action_macro_f1"] + 0.4 * out["point_macro_f1"] + 0.2 * out["server_auc"])
    return out


def build_current_oof() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    v3 = load_pickle("oof_proba_v3.pkl")
    v5 = load_pickle("oof_proba_v5.pkl")
    v7 = load_pickle("oof_proba_v7.pkl")
    v10 = load_pickle("oof_proba_v10b.pkl")
    r7 = load_pickle("oof_proba_r7.pkl")
    meta = normalize_meta(v3["valid_meta"]).reset_index(drop=True)

    v3_action, v3_point, v3_server = compose_v3(v3)
    r7_action, _, r7_server = compose_v3(r7)
    r1_action = normalize_rows(0.4 * v5["gru_action"] + 0.6 * v7["tr_action"])
    r1_server = np.clip(0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"], 1e-6, 1 - 1e-6)
    r33_action = normalize_rows(0.85 * r1_action + 0.05 * r7_action + 0.10 * v5["gru_action"])
    r33_server = np.clip(0.70 * r1_server + 0.15 * v10["v10_server"] + 0.15 * r7_server, 1e-6, 1 - 1e-6)

    action_mult = tune_segmented_multipliers(meta, r33_action, ACTION_CLASSES, "action", "two")
    # Use the safer current production point policy: V3 point + V3 multipliers.
    current_action = apply_segmented_multipliers(meta, r33_action, action_mult, ACTION_CLASSES, "two")
    current_point = apply_segmented_multipliers(meta, v3_point, v3["tuning"].point_multipliers, POINT_CLASSES, v3["tuning"].bins_mode)
    return meta, current_action, current_point, r33_server


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def load_v64_oof_for_meta(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token = pd.read_csv(V64_DIR / "oof_token_predictions.csv")
    rally = pd.read_csv(V64_DIR / "oof_rally_predictions.csv")
    key = ["rally_uid", "prefix_len"]
    want = meta[key].reset_index()
    tok = want.merge(token[key + ["action_pred", "point_pred", "action_true", "point_true"]], on=key, how="left")
    ral = want.merge(rally[key + ["server_prob", "server_true"]], on=key, how="left")
    if tok["action_pred"].isna().any() or ral["server_prob"].isna().any():
        raise ValueError("Could not align V64 OOF with current OOF meta.")
    return (
        tok.sort_values("index")["action_pred"].astype(int).to_numpy(),
        tok.sort_values("index")["point_pred"].astype(int).to_numpy(),
        ral.sort_values("index")["server_prob"].astype(float).to_numpy(),
    )


def win_table(meta: pd.DataFrame, current_pred: np.ndarray, v64_pred: np.ndarray, target_col: str, task: str) -> pd.DataFrame:
    true = meta[target_col].to_numpy()
    cur_ok = current_pred == true
    v64_ok = v64_pred == true
    rows = []
    groups = [("all", np.ones(len(meta), dtype=bool))]
    groups += [(f"prefix_{b}", meta["prefix_bin"].eq(b).to_numpy()) for b in ["1", "2", "3", "4-6", "7+"]]
    if task == "action":
        groups += [(f"true_action_{c}", meta["next_actionId"].eq(c).to_numpy()) for c in ACTION_CLASSES]
        groups += [("v64_predicts_8or9", np.isin(v64_pred, [8, 9])), ("current_predicts_8or9", np.isin(current_pred, [8, 9]))]
    else:
        groups += [(f"true_point_{c}", meta["next_pointId"].eq(c).to_numpy()) for c in POINT_CLASSES]
        groups += [
            ("current_point0_v64_nonzero", (current_pred == 0) & (v64_pred != 0)),
            ("v64_point0_current_nonzero", (v64_pred == 0) & (current_pred != 0)),
            ("both_nonzero_disagree", (current_pred != 0) & (v64_pred != 0) & (current_pred != v64_pred)),
        ]
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "task": task,
                "group": name,
                "count": int(mask.sum()),
                "current_correct_rate": float(cur_ok[mask].mean()),
                "v64_correct_rate": float(v64_ok[mask].mean()),
                "both_correct_rate": float((cur_ok & v64_ok)[mask].mean()),
                "current_only_rate": float((cur_ok & ~v64_ok)[mask].mean()),
                "v64_only_rate": float((~cur_ok & v64_ok)[mask].mean()),
                "both_wrong_rate": float((~cur_ok & ~v64_ok)[mask].mean()),
                "disagree_rate": float((current_pred != v64_pred)[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_gates(meta: pd.DataFrame, cur_a: np.ndarray, cur_p: np.ndarray, cur_s: np.ndarray, v64_a: np.ndarray, v64_p: np.ndarray, v64_s: np.ndarray) -> pd.DataFrame:
    candidates: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    bins = meta["prefix_bin"].to_numpy()
    short = np.isin(bins, ["1", "2"])
    very_short = bins == "1"
    longish = np.isin(bins, ["4-6", "7+"])
    v64_rare = np.isin(v64_a, [8, 9])
    current_rare = np.isin(cur_a, [8, 9])
    current_v64_disagree = cur_a != v64_a
    current_point0_v64_nonzero = (cur_p == 0) & (v64_p != 0)
    v64_point0_current_nonzero = (v64_p == 0) & (cur_p != 0)

    def mix(base: np.ndarray, alt: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = base.copy()
        out[mask] = alt[mask]
        return out

    candidates.append(("current", cur_a, cur_p, cur_s))
    candidates.append(("v64", v64_a, v64_p, v64_s))
    candidates.append(("action_v64_short", mix(cur_a, v64_a, short), cur_p, cur_s))
    candidates.append(("action_v64_len1", mix(cur_a, v64_a, very_short), cur_p, cur_s))
    candidates.append(("action_v64_rare89_only", mix(cur_a, v64_a, v64_rare), cur_p, cur_s))
    candidates.append(("action_v64_when_current_rare89", mix(cur_a, v64_a, current_rare), cur_p, cur_s))
    candidates.append(("action_v64_when_current_rare89_disagree", mix(cur_a, v64_a, current_rare & current_v64_disagree), cur_p, cur_s))
    candidates.append(("action_v64_len1_or_current_rare89", mix(cur_a, v64_a, very_short | current_rare), cur_p, cur_s))
    candidates.append(("action_v64_short_or_current_rare89", mix(cur_a, v64_a, short | current_rare), cur_p, cur_s))
    candidates.append(("action_v64_longish", mix(cur_a, v64_a, longish), cur_p, cur_s))
    candidates.append(("point_v64_short", cur_a, mix(cur_p, v64_p, short), cur_s))
    candidates.append(("point_v64_len1", cur_a, mix(cur_p, v64_p, very_short), cur_s))
    candidates.append(("point_v64_when_current0", cur_a, mix(cur_p, v64_p, current_point0_v64_nonzero), cur_s))
    candidates.append(("point_v64_when_v640", cur_a, mix(cur_p, v64_p, v64_point0_current_nonzero), cur_s))
    candidates.append(("action_short_point_current0", mix(cur_a, v64_a, short), mix(cur_p, v64_p, current_point0_v64_nonzero), cur_s))
    candidates.append(("action_rare89_point_current0", mix(cur_a, v64_a, v64_rare), mix(cur_p, v64_p, current_point0_v64_nonzero), cur_s))
    candidates.append(("server_v64_only", cur_a, cur_p, v64_s))

    rows = []
    for name, a, p, s in candidates:
        m = metrics(meta, a, p, s)
        m["candidate"] = name
        m["action_diff_vs_current"] = float(np.mean(a != cur_a))
        m["point_diff_vs_current"] = float(np.mean(p != cur_p))
        m["server_mae_vs_current"] = float(np.mean(np.abs(s - cur_s)))
        rows.append(m)
    return pd.DataFrame(rows).sort_values("overall", ascending=False)


def oracle_upper_bound(meta: pd.DataFrame, cur_a: np.ndarray, cur_p: np.ndarray, v64_a: np.ndarray, v64_p: np.ndarray) -> dict:
    true_a = meta["next_actionId"].to_numpy()
    true_p = meta["next_pointId"].to_numpy()
    oracle_a = cur_a.copy()
    oracle_p = cur_p.copy()
    oracle_a[(cur_a != true_a) & (v64_a == true_a)] = v64_a[(cur_a != true_a) & (v64_a == true_a)]
    oracle_p[(cur_p != true_p) & (v64_p == true_p)] = v64_p[(cur_p != true_p) & (v64_p == true_p)]
    return {
        "action_oracle_macro_f1": float(f1_score(true_a, oracle_a, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "point_oracle_macro_f1": float(f1_score(true_p, oracle_p, average="macro", labels=POINT_CLASSES, zero_division=0)),
        "action_v64_only_rows": int(((cur_a != true_a) & (v64_a == true_a)).sum()),
        "point_v64_only_rows": int(((cur_p != true_p) & (v64_p == true_p)).sum()),
        "action_current_only_rows": int(((cur_a == true_a) & (v64_a != true_a)).sum()),
        "point_current_only_rows": int(((cur_p == true_p) & (v64_p != true_p)).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    meta, cur_a, cur_p, cur_s = build_current_oof()
    meta = meta.copy()
    meta["prefix_bin"] = meta["prefix_len"].map(prefix_bin)
    v64_a, v64_p, v64_s = load_v64_oof_for_meta(meta)

    current_metrics = metrics(meta, cur_a, cur_p, cur_s)
    v64_metrics = metrics(meta, v64_a, v64_p, v64_s)
    gates = evaluate_gates(meta, cur_a, cur_p, cur_s, v64_a, v64_p, v64_s)
    action_wins = win_table(meta, cur_a, v64_a, "next_actionId", "action")
    point_wins = win_table(meta, cur_p, v64_p, "next_pointId", "point")
    oracle = oracle_upper_bound(meta, cur_a, cur_p, v64_a, v64_p)

    aligned = meta[["rally_uid", "prefix_len", "prefix_bin", "next_actionId", "next_pointId", "serverGetPoint"]].copy()
    aligned["current_action"] = cur_a
    aligned["current_point"] = cur_p
    aligned["current_server"] = cur_s
    aligned["v64_action"] = v64_a
    aligned["v64_point"] = v64_p
    aligned["v64_server"] = v64_s
    aligned["action_current_ok"] = aligned["current_action"].eq(aligned["next_actionId"])
    aligned["action_v64_ok"] = aligned["v64_action"].eq(aligned["next_actionId"])
    aligned["point_current_ok"] = aligned["current_point"].eq(aligned["next_pointId"])
    aligned["point_v64_ok"] = aligned["v64_point"].eq(aligned["next_pointId"])

    aligned.to_csv(OUT_DIR / "r41_aligned_oof_predictions.csv", index=False)
    gates.to_csv(OUT_DIR / "r41_gate_candidates_oof.csv", index=False)
    action_wins.to_csv(OUT_DIR / "r41_action_win_table.csv", index=False)
    point_wins.to_csv(OUT_DIR / "r41_point_win_table.csv", index=False)

    report = {
        "current_metrics": current_metrics,
        "v64_metrics": v64_metrics,
        "oracle_upper_bound": oracle,
        "best_gate": gates.iloc[0].to_dict(),
        "outputs": {
            "aligned": str(OUT_DIR / "r41_aligned_oof_predictions.csv"),
            "gates": str(OUT_DIR / "r41_gate_candidates_oof.csv"),
            "action_wins": str(OUT_DIR / "r41_action_win_table.csv"),
            "point_wins": str(OUT_DIR / "r41_point_win_table.csv"),
        },
    }
    (OUT_DIR / "r41_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# R41 V64-Current Disagreement Analysis",
        "",
        "## Current OOF",
        f"- action: {current_metrics['action_macro_f1']:.6f}",
        f"- point: {current_metrics['point_macro_f1']:.6f}",
        f"- server: {current_metrics['server_auc']:.6f}",
        f"- overall: {current_metrics['overall']:.6f}",
        "",
        "## V64 OOF On Same Rows",
        f"- action: {v64_metrics['action_macro_f1']:.6f}",
        f"- point: {v64_metrics['point_macro_f1']:.6f}",
        f"- server: {v64_metrics['server_auc']:.6f}",
        f"- overall: {v64_metrics['overall']:.6f}",
        "",
        "## Oracle Upper Bound",
        f"- action oracle: {oracle['action_oracle_macro_f1']:.6f}",
        f"- point oracle: {oracle['point_oracle_macro_f1']:.6f}",
        f"- V64-only correct action rows: {oracle['action_v64_only_rows']}",
        f"- V64-only correct point rows: {oracle['point_v64_only_rows']}",
        f"- current-only correct action rows: {oracle['action_current_only_rows']}",
        f"- current-only correct point rows: {oracle['point_current_only_rows']}",
        "",
        "## Best Simple Gate",
        f"- candidate: `{report['best_gate']['candidate']}`",
        f"- overall: {report['best_gate']['overall']:.6f}",
        f"- action: {report['best_gate']['action_macro_f1']:.6f}",
        f"- point: {report['best_gate']['point_macro_f1']:.6f}",
        f"- server: {report['best_gate']['server_auc']:.6f}",
    ]
    (OUT_DIR / "r41_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
