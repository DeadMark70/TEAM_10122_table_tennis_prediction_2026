"""R33 safe OOF ensemble search.

Searches low-churn OOF blends around the known R1 policy:
- action: R1 base plus small weights from V10B/R7/R30/V3/V5/V7
- point: fixed V3, with only tiny point probe weights
- server: R1 base plus small weights from V10B/R7/R30/V3/V5/V7

No new model is trained and no submission is written here.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta, prefix_report
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R33 safe OOF ensemble search.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--r7-oof", default="oof_proba_r7.pkl")
    parser.add_argument("--r30-oof", default="oof_proba_r30.pkl")
    parser.add_argument("--out-dir", default="r33_safe_oof_ensemble")
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def assert_aligned(base: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    other = normalize_meta(other)
    cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    mismatch = (base[cols].to_numpy() != other[cols].to_numpy()).any(axis=1)
    if mismatch.any():
        first = int(np.where(mismatch)[0][0])
        raise ValueError(f"{name} not aligned at row {first}.")


def score_action(meta: pd.DataFrame, prob: np.ndarray, mode: str) -> dict:
    mult = tune_segmented_multipliers(meta, prob, ACTION_CLASSES, "action", mode)
    pred = apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, mode)
    f1 = f1_score(meta["next_actionId"], pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    return {"score": float(f1), "mult": mult, "pred": pred, "prob": prob}


def score_point(meta: pd.DataFrame, prob: np.ndarray, mode: str) -> dict:
    mult = tune_segmented_multipliers(meta, prob, POINT_CLASSES, "point", mode)
    pred = apply_segmented_multipliers(meta, prob, mult, POINT_CLASSES, mode)
    f1 = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    return {"score": float(f1), "mult": mult, "pred": pred, "prob": prob}


def candidate_small_blends(base: np.ndarray, components: dict[str, np.ndarray], weights: list[float]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = [("base", base)]
    for name, comp in components.items():
        for w in weights:
            if base.ndim == 1:
                prob = (1.0 - w) * base + w * comp
                prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
            else:
                prob = blend_probs(base, comp, w)
            out.append((f"base+{w:g}*{name}", prob))
    for a, b in combinations(list(components), 2):
        for wa in [0.05, 0.1, 0.15]:
            for wb in [0.05, 0.1, 0.15]:
                if wa + wb > 0.3:
                    continue
                prob = (1.0 - wa - wb) * base + wa * components[a] + wb * components[b]
                if prob.ndim == 2:
                    prob = prob / prob.sum(axis=1, keepdims=True)
                out.append((f"base+{wa:g}*{a}+{wb:g}*{b}", prob))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    r7 = load_pickle(args.r7_oof)
    r30 = load_pickle(args.r30_oof)

    meta = normalize_meta(v3["valid_meta"])
    for name, obj in [("V5", v5), ("V7", v7), ("V10B", v10), ("R7", r7), ("R30", r30)]:
        assert_aligned(meta, obj["valid_meta"], name)

    v3_action, v3_point, v3_server = compose_v3(v3)
    r7_action, r7_point, r7_server = compose_v3(r7)
    r30_action, r30_point, r30_server = compose_v3(r30)

    r1_action = 0.4 * v5["gru_action"] + 0.6 * v7["tr_action"]
    r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
    r1_point = v3_point
    r1_server = np.clip(0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"], 1e-6, 1.0 - 1e-6)

    action_components = {
        "v10b": v10["v10_action"],
        "r7": r7_action,
        "r30": r30_action,
        "v3": v3_action,
        "v5": v5["gru_action"],
        "v7": v7["tr_action"],
    }
    server_components = {
        "v10b": v10["v10_server"],
        "r7": r7_server,
        "r30": r30_server,
        "v3": v3_server,
        "v5": v5["gru_server"],
        "v7": v7["tr_server"],
    }
    point_components = {
        "v10b": v10["v10_point"],
        "r7": r7_point,
        "r30": r30_point,
        "v5": v5["gru_point"],
        "v7": v7["tr_point"],
    }

    action_rows: list[dict] = []
    best_action = None
    for name, prob in candidate_small_blends(r1_action, action_components, [0.05, 0.1, 0.15, 0.2, 0.3]):
        result = score_action(meta, prob, args.multiplier_bins)
        action_rows.append({"candidate": name, "action_macro_f1": result["score"]})
        if best_action is None or result["score"] > best_action["score"]:
            best_action = {"name": name, **result}

    point_rows: list[dict] = []
    best_point = None
    point_candidates = [("v3_fixed", r1_point)]
    for name, comp in point_components.items():
        for w in [0.02, 0.05, 0.1]:
            point_candidates.append((f"v3+{w:g}*{name}", blend_probs(r1_point, comp, w)))
    for name, prob in point_candidates:
        result = score_point(meta, prob, args.multiplier_bins)
        point_rows.append({"candidate": name, "point_macro_f1": result["score"]})
        if best_point is None or result["score"] > best_point["score"]:
            best_point = {"name": name, **result}

    server_rows: list[dict] = []
    best_server = None
    for name, prob in candidate_small_blends(r1_server, server_components, [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]):
        auc = roc_auc_score(meta["serverGetPoint"], prob)
        server_rows.append({"candidate": name, "server_auc": float(auc)})
        if best_server is None or auc > best_server["score"]:
            best_server = {"name": name, "score": float(auc), "prob": prob}

    overall = 0.4 * best_action["score"] + 0.4 * best_point["score"] + 0.2 * best_server["score"]
    selected = {
        "action_candidate": best_action["name"],
        "point_candidate": best_point["name"],
        "server_candidate": best_server["name"],
        "action_macro_f1": best_action["score"],
        "point_macro_f1": best_point["score"],
        "server_auc": best_server["score"],
        "overall": float(overall),
        "action_multipliers": best_action["mult"],
        "point_multipliers": best_point["mult"],
        "multiplier_bins": args.multiplier_bins,
    }

    pd.DataFrame(action_rows).sort_values("action_macro_f1", ascending=False).to_csv(
        out_dir / "r33_action_search.csv", index=False
    )
    pd.DataFrame(point_rows).sort_values("point_macro_f1", ascending=False).to_csv(
        out_dir / "r33_point_search.csv", index=False
    )
    pd.DataFrame(server_rows).sort_values("server_auc", ascending=False).to_csv(
        out_dir / "r33_server_search.csv", index=False
    )
    prefix = prefix_report(meta, best_action["pred"], best_point["pred"], best_server["prob"])
    prefix.to_csv(out_dir / "r33_prefix_report.csv", index=False)
    (out_dir / "r33_selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    lines = [
        "# R33 Safe OOF Ensemble",
        "",
        f"- action: `{selected['action_candidate']}` => {selected['action_macro_f1']:.6f}",
        f"- point: `{selected['point_candidate']}` => {selected['point_macro_f1']:.6f}",
        f"- server: `{selected['server_candidate']}` => {selected['server_auc']:.6f}",
        f"- overall: `{selected['overall']:.6f}`",
        "",
        "This is an OOF-only search. It does not train new models or write a submission.",
    ]
    (out_dir / "r33_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in selected.items() if not k.endswith("multipliers")}, indent=2))


if __name__ == "__main__":
    main()
