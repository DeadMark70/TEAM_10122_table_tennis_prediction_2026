"""R1 OOF diagnostics and best-per-task ensemble search.

Uses existing OOF probabilities only. No new model is trained here.
The goal is to estimate the value of combining V3/V5/V7 by task and by
prefix-length bins before spending time on full-test sequence predictions.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, evaluate_v3, tune_segmented_multipliers


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


PREFIX_BINS = [
    ("1", lambda s: s.eq(1)),
    ("2", lambda s: s.eq(2)),
    ("3", lambda s: s.eq(3)),
    ("4-6", lambda s: s.between(4, 6)),
    ("7+", lambda s: s.ge(7)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R1 OOF ensemble diagnostics.")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    return meta[cols].reset_index(drop=True).astype(
        {
            "rally_uid": int,
            "match": int,
            "prefix_len": int,
            "next_actionId": int,
            "next_pointId": int,
            "serverGetPoint": int,
        }
    )


def assert_aligned(base: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    other = normalize_meta(other)
    check_cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    mismatch = (base[check_cols].to_numpy() != other[check_cols].to_numpy()).any(axis=1)
    if mismatch.any():
        first = int(np.where(mismatch)[0][0])
        raise ValueError(f"{name} OOF rows are not aligned at row {first}.")


def compose_v3(oof: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tuning = oof["tuning"]
    action = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)
    point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )
    return action, point, np.clip(server, 1e-6, 1.0 - 1e-6)


def weight_grid(n: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    out: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            out.append(tuple((prefix + [remaining])[i] / units for i in range(n)))
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, n)
    return out


def blend_many(probs: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    out = np.zeros_like(probs[0], dtype=float)
    for prob, weight in zip(probs, weights):
        out += float(weight) * prob
    if out.ndim == 2:
        out = out / out.sum(axis=1, keepdims=True)
    return out


def score_action(meta: pd.DataFrame, prob: np.ndarray, mode: str) -> tuple[float, dict[str, list[float]], np.ndarray]:
    mult = tune_segmented_multipliers(meta, prob, ACTION_CLASSES, "action", mode)
    pred = apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, mode)
    score = f1_score(meta["next_actionId"], pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    return float(score), mult, pred


def score_point_with_existing(meta: pd.DataFrame, prob: np.ndarray, tuning: V3Tuning) -> tuple[float, np.ndarray]:
    pred = apply_segmented_multipliers(
        meta, prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode
    )
    score = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    return float(score), pred


def prefix_report(
    meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, fn in PREFIX_BINS:
        mask = fn(meta["prefix_len"]).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        action_f1 = f1_score(
            meta.iloc[idx]["next_actionId"], action_pred[idx], average="macro", labels=ACTION_CLASSES, zero_division=0
        )
        point_f1 = f1_score(
            meta.iloc[idx]["next_pointId"], point_pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0
        )
        try:
            server_auc = roc_auc_score(meta.iloc[idx]["serverGetPoint"], server_prob[idx])
        except ValueError:
            server_auc = float("nan")
        overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc
        rows.append(
            {
                "prefix_len_bin": label,
                "count": int(len(idx)),
                "action_macro_f1": float(action_f1),
                "point_macro_f1": float(point_f1),
                "server_auc": float(server_auc),
                "overall": float(overall),
            }
        )
    return pd.DataFrame(rows)


def evaluate_named(
    name: str,
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    server_prob: np.ndarray,
    action_mult: dict[str, list[float]],
    point_mult: dict[str, list[float]],
    mode: str,
) -> dict[str, object]:
    metrics = evaluate_v3(meta, action_prob, point_prob, server_prob, action_mult, point_mult, mode)
    return {"variant": name, **metrics}


def search_task_level(
    meta: pd.DataFrame,
    action_components: dict[str, np.ndarray],
    point_prob: np.ndarray,
    point_tuning: V3Tuning,
    server_components: dict[str, np.ndarray],
    step: float,
    mode: str,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    action_names = list(action_components)
    action_probs = [action_components[k] for k in action_names]
    server_names = list(server_components)
    server_probs = [server_components[k] for k in server_names]
    point_score, point_pred = score_point_with_existing(meta, point_prob, point_tuning)

    action_candidates: list[dict[str, object]] = []
    for weights in weight_grid(len(action_probs), step):
        prob = blend_many(action_probs, weights)
        f1, mult, pred = score_action(meta, prob, mode)
        action_candidates.append({"weights": weights, "prob": prob, "f1": f1, "mult": mult, "pred": pred})
        rows.append(
            {
                "search": "action_task",
                "weights": json.dumps(dict(zip(action_names, weights))),
                "action_macro_f1": f1,
                "point_macro_f1": point_score,
                "server_auc": np.nan,
                "overall_partial": np.nan,
            }
        )
    best_action = max(action_candidates, key=lambda x: float(x["f1"]))

    server_candidates: list[dict[str, object]] = []
    for weights in weight_grid(len(server_probs), step):
        prob = blend_many(server_probs, weights)
        auc = roc_auc_score(meta["serverGetPoint"], prob)
        server_candidates.append({"weights": weights, "prob": prob, "auc": float(auc)})
        rows.append(
            {
                "search": "server_task",
                "weights": json.dumps(dict(zip(server_names, weights))),
                "action_macro_f1": np.nan,
                "point_macro_f1": point_score,
                "server_auc": float(auc),
                "overall_partial": np.nan,
            }
        )
    best_server = max(server_candidates, key=lambda x: float(x["auc"]))
    overall = 0.4 * float(best_action["f1"]) + 0.4 * point_score + 0.2 * float(best_server["auc"])
    selected = {
        "name": "task_level",
        "action_names": action_names,
        "action_weights": dict(zip(action_names, best_action["weights"])),
        "server_names": server_names,
        "server_weights": dict(zip(server_names, best_server["weights"])),
        "action_prob": best_action["prob"],
        "point_prob": point_prob,
        "server_prob": best_server["prob"],
        "action_mult": best_action["mult"],
        "point_mult": point_tuning.point_multipliers,
        "action_pred": best_action["pred"],
        "point_pred": point_pred,
        "metrics": {
            "action_macro_f1": float(best_action["f1"]),
            "point_macro_f1": point_score,
            "server_auc": float(best_server["auc"]),
            "overall": float(overall),
        },
    }
    return selected, pd.DataFrame(rows)


def search_prefix_level(
    meta: pd.DataFrame,
    action_components: dict[str, np.ndarray],
    point_prob: np.ndarray,
    point_tuning: V3Tuning,
    server_components: dict[str, np.ndarray],
    step: float,
    mode: str,
) -> tuple[dict, pd.DataFrame]:
    action_names = list(action_components)
    action_probs = [action_components[k] for k in action_names]
    server_names = list(server_components)
    server_probs = [server_components[k] for k in server_names]
    rows: list[dict[str, object]] = []

    action_out = np.zeros_like(action_probs[0])
    server_out = np.zeros_like(server_probs[0])
    action_bin_weights: dict[str, dict[str, float]] = {}
    server_bin_weights: dict[str, dict[str, float]] = {}

    for label, fn in PREFIX_BINS:
        mask = fn(meta["prefix_len"]).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        local_meta = meta.iloc[idx].reset_index(drop=True)
        best_action = None
        for weights in weight_grid(len(action_probs), step):
            local_prob = blend_many([p[idx] for p in action_probs], weights)
            pred = np.asarray(ACTION_CLASSES)[np.argmax(local_prob, axis=1)]
            f1 = f1_score(
                local_meta["next_actionId"], pred, average="macro", labels=ACTION_CLASSES, zero_division=0
            )
            if best_action is None or f1 > best_action["f1"]:
                best_action = {"weights": weights, "f1": float(f1)}
        action_out[idx] = blend_many([p[idx] for p in action_probs], best_action["weights"])
        action_bin_weights[label] = dict(zip(action_names, best_action["weights"]))

        best_server = None
        for weights in weight_grid(len(server_probs), step):
            local_prob = blend_many([p[idx] for p in server_probs], weights)
            try:
                auc = roc_auc_score(local_meta["serverGetPoint"], local_prob)
            except ValueError:
                auc = float("nan")
            if best_server is None or (not np.isnan(auc) and auc > best_server["auc"]):
                best_server = {"weights": weights, "auc": float(auc)}
        server_out[idx] = blend_many([p[idx] for p in server_probs], best_server["weights"])
        server_bin_weights[label] = dict(zip(server_names, best_server["weights"]))
        rows.append(
            {
                "prefix_len_bin": label,
                "count": int(len(idx)),
                "action_search_f1_no_mult": best_action["f1"],
                "action_weights": json.dumps(action_bin_weights[label]),
                "server_search_auc": best_server["auc"],
                "server_weights": json.dumps(server_bin_weights[label]),
            }
        )

    action_score, action_mult, action_pred = score_action(meta, action_out, mode)
    point_score, point_pred = score_point_with_existing(meta, point_prob, point_tuning)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_out)
    overall = 0.4 * action_score + 0.4 * point_score + 0.2 * server_auc
    selected = {
        "name": "prefix_level",
        "action_bin_weights": action_bin_weights,
        "server_bin_weights": server_bin_weights,
        "action_prob": action_out,
        "point_prob": point_prob,
        "server_prob": server_out,
        "action_mult": action_mult,
        "point_mult": point_tuning.point_multipliers,
        "action_pred": action_pred,
        "point_pred": point_pred,
        "metrics": {
            "action_macro_f1": float(action_score),
            "point_macro_f1": point_score,
            "server_auc": float(server_auc),
            "overall": float(overall),
        },
    }
    return selected, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    meta = normalize_meta(v3["valid_meta"])
    assert_aligned(meta, v5["valid_meta"], "V5")
    assert_aligned(meta, v7["valid_meta"], "V7")

    v3_action, v3_point, v3_server = compose_v3(v3)
    action_components = {
        "v3": v3_action,
        "v5_gru": v5["gru_action"],
        "v7_transformer": v7["tr_action"],
    }
    point_components = {
        "v3": v3_point,
        "v5_gru": v5["gru_point"],
        "v7_transformer": v7["tr_point"],
    }
    server_components = {
        "v3": v3_server,
        "v5_gru": v5["gru_server"],
        "v7_transformer": v7["tr_server"],
    }

    rows: list[dict[str, object]] = []
    for name in action_components:
        action_prob = action_components[name]
        point_prob = v3_point if name != "point_probe" else point_components[name]
        action_f1, action_mult, _ = score_action(meta, action_prob, args.multiplier_bins)
        point_f1, _ = score_point_with_existing(meta, v3_point, v3["tuning"])
        server_auc = roc_auc_score(meta["serverGetPoint"], server_components.get(name, v3_server))
        rows.append(
            {
                "variant": f"{name}_action_v3_point_{name}_server",
                "action_macro_f1": action_f1,
                "point_macro_f1": point_f1,
                "server_auc": float(server_auc),
                "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
            }
        )
    rows.append(
        evaluate_named(
            "v3_selected",
            meta,
            v3_action,
            v3_point,
            v3_server,
            v3["tuning"].action_multipliers,
            v3["tuning"].point_multipliers,
            v3["tuning"].bins_mode,
        )
    )

    task_selected, task_search = search_task_level(
        meta, action_components, v3_point, v3["tuning"], server_components, args.weight_step, args.multiplier_bins
    )
    prefix_selected, prefix_search = search_prefix_level(
        meta, action_components, v3_point, v3["tuning"], server_components, args.weight_step, args.multiplier_bins
    )
    rows.append({"variant": "r1_task_level", **task_selected["metrics"]})
    rows.append({"variant": "r1_prefix_level", **prefix_selected["metrics"]})
    summary = pd.DataFrame(rows).sort_values("overall", ascending=False)
    summary.to_csv(out_dir / "r1_oof_ensemble_summary.csv", index=False)
    task_search.to_csv(out_dir / "r1_task_weight_search.csv", index=False)
    prefix_search.to_csv(out_dir / "r1_prefix_weight_search.csv", index=False)

    best = task_selected if task_selected["metrics"]["overall"] >= prefix_selected["metrics"]["overall"] else prefix_selected
    best_prefix = prefix_report(meta, best["action_pred"], best["point_pred"], best["server_prob"])
    best_prefix.to_csv(out_dir / "r1_best_prefix_report.csv", index=False)

    selected = {
        "best_name": best["name"],
        "best_metrics": best["metrics"],
        "task_level": {
            "metrics": task_selected["metrics"],
            "action_weights": task_selected["action_weights"],
            "server_weights": task_selected["server_weights"],
        },
        "prefix_level": {
            "metrics": prefix_selected["metrics"],
            "action_bin_weights": prefix_selected["action_bin_weights"],
            "server_bin_weights": prefix_selected["server_bin_weights"],
        },
        "point_policy": "fixed_v3_point_probabilities_and_v3_point_multipliers",
        "weight_step": args.weight_step,
        "multiplier_bins": args.multiplier_bins,
    }
    (out_dir / "r1_selected_ensemble.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    lines = [
        "# R1 OOF Best-Per-Task Ensemble",
        "",
        "## Summary",
        summary.to_csv(index=False),
        "",
        "## Selected",
        f"- Best ensemble: `{best['name']}`",
        f"- Overall: `{best['metrics']['overall']:.6f}`",
        f"- Action Macro-F1: `{best['metrics']['action_macro_f1']:.6f}`",
        f"- Point Macro-F1: `{best['metrics']['point_macro_f1']:.6f}`",
        f"- Server AUC: `{best['metrics']['server_auc']:.6f}`",
        "- Point policy: fixed V3 point probabilities and V3 point multipliers.",
        "",
        "## Decision",
    ]
    overall = float(best["metrics"]["overall"])
    if overall >= 0.313:
        lines.append("- CV is above the submit threshold. Next step: generate full-test sequence probabilities for the selected sequence components and write `submission_r1.csv`.")
    elif overall >= 0.3125:
        lines.append("- CV is borderline. Check stability/folds before spending a submission.")
    else:
        lines.append("- CV is below the submit threshold. Prefer V5 as the next submission candidate if using quota.")
    (out_dir / "r1_recommendation.md").write_text("\n".join(lines), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"best={best['name']} overall={best['metrics']['overall']:.6f}")
    print("wrote R1 reports")


if __name__ == "__main__":
    main()
