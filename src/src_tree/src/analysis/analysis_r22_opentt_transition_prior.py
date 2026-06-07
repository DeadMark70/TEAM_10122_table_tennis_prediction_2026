"""R22 OpenTTGames canonical transition-prior audit.

This experiment does not train a new model. It maps AICUP and Extended
OpenTTGames labels into a small canonical table-tennis event schema, estimates
external transition priors from OpenTTGames, and tests whether those priors can
serve as a low-weight action probability bias on existing OOF predictions.

The point and server branches are kept fixed to V3 to avoid contaminating the
stable branches with external taxonomy mismatch.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta, prefix_report
from baseline_lgbm import (
    ACTION_CLASSES,
    POINT_CLASSES,
    add_role_and_score_features,
    build_train_prefix_table,
)
from baseline_v3 import (
    add_remaining_bucket,
    apply_segmented_multipliers,
    tune_segmented_multipliers,
)


TECHNIQUES = ["serve", "loop", "flick", "smash", "push", "block", "chop", "lob", "unknown"]
FAMILIES = ["serve", "attack", "control", "defensive", "unknown"]


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


ACTION_TO_FAMILY = {
    0: "unknown",
    1: "attack",
    2: "attack",
    3: "attack",
    4: "attack",
    5: "attack",
    6: "attack",
    7: "attack",
    8: "control",
    9: "control",
    10: "control",
    11: "control",
    12: "defensive",
    13: "defensive",
    14: "defensive",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

ACTION_TO_TECHNIQUE = {
    0: "unknown",
    1: "loop",
    2: "loop",
    3: "smash",
    4: "flick",
    5: "loop",
    6: "push",
    7: "flick",
    8: "push",
    9: "push",
    10: "push",
    11: "push",
    12: "chop",
    13: "block",
    14: "lob",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

TECHNIQUE_TO_ACTIONS = {
    "serve": [15, 16, 17, 18],
    "loop": [1, 2, 5],
    "flick": [4, 7],
    "smash": [3],
    "push": [6, 8, 9, 10, 11],
    "block": [13],
    "chop": [12],
    "lob": [14],
    "unknown": [0],
}

FAMILY_TO_ACTIONS = {
    "serve": [15, 16, 17, 18],
    "attack": [1, 2, 3, 4, 5, 6, 7],
    "control": [8, 9, 10, 11],
    "defensive": [12, 13, 14],
    "unknown": [0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R22 OpenTTGames transition-prior audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--events", default="external_data/openttgames/processed/openttgames_events.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--summary", default="r22_prior_blend_report.csv")
    parser.add_argument("--prefix-report", default="r22_prefix_report.csv")
    parser.add_argument("--mapping-report", default="r22_canonical_mapping_report.csv")
    parser.add_argument("--transition-report", default="r22_external_transition_counts.csv")
    parser.add_argument("--selected", default="r22_selected.json")
    parser.add_argument("--recommendation", default="r22_recommendation.md")
    parser.add_argument("--feature-report", default="feature_report_r22.json")
    return parser.parse_args()


def phase_from_prefix_len(prefix_len: int) -> str:
    if prefix_len == 1:
        return "receive"
    if prefix_len == 2:
        return "third_ball"
    if prefix_len == 3:
        return "fourth_ball"
    return "rally"


def opentt_family(technique: str) -> str:
    if technique == "serve":
        return "serve"
    if technique in {"loop", "flick", "smash"}:
        return "attack"
    if technique == "push":
        return "control"
    if technique in {"block", "chop", "lob"}:
        return "defensive"
    return "unknown"


def action_family(action_id: int) -> str:
    return ACTION_TO_FAMILY.get(int(action_id), "unknown")


def action_technique(action_id: int) -> str:
    return ACTION_TO_TECHNIQUE.get(int(action_id), "unknown")


def normalize_prob(arr: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.maximum(arr, floor)
    return arr / arr.sum(axis=-1, keepdims=True)


def map_dist_to_actions(dist: dict[str, float], mapping: dict[str, list[int]]) -> np.ndarray:
    out = np.zeros(len(ACTION_CLASSES), dtype=float)
    for label, prob in dist.items():
        actions = mapping.get(label, mapping["unknown"])
        share = float(prob) / len(actions)
        for action in actions:
            out[ACTION_CLASSES.index(action)] += share
    return normalize_prob(out)


def segment_opentt_strokes(events: pd.DataFrame) -> list[list[dict[str, str]]]:
    segments: list[list[dict[str, str]]] = []
    for _, video in events.sort_values(["video_id", "frame"]).groupby("video_id", sort=False):
        current: list[dict[str, str]] = []
        for row in video.itertuples(index=False):
            event_type = str(row.event_type)
            if event_type == "empty_event":
                if current:
                    segments.append(current)
                    current = []
                continue
            if event_type == "rally_ending":
                if current:
                    segments.append(current)
                    current = []
                continue
            if event_type == "stroke":
                technique = str(row.technique) if str(row.technique) else "unknown"
                if technique == "serve" and current:
                    segments.append(current)
                    current = []
                current.append(
                    {
                        "technique": technique if technique in TECHNIQUES else "unknown",
                        "family": opentt_family(technique),
                        "hand": str(row.stroke_hand) if str(row.stroke_hand) else "unknown",
                    }
                )
        if current:
            segments.append(current)
    return [seg for seg in segments if len(seg) >= 2]


def estimate_external_priors(events: pd.DataFrame, alpha: float) -> dict[str, object]:
    segments = segment_opentt_strokes(events)
    tech_global = Counter()
    fam_global = Counter()
    tech_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    fam_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    transition_rows: list[dict[str, object]] = []

    for seg in segments:
        for idx in range(len(seg) - 1):
            cur = seg[idx]
            nxt = seg[idx + 1]
            phase = phase_from_prefix_len(idx + 1)
            tech_counts[(phase, cur["technique"])][nxt["technique"]] += 1
            fam_counts[(phase, cur["family"])][nxt["family"]] += 1
            tech_global[nxt["technique"]] += 1
            fam_global[nxt["family"]] += 1
            transition_rows.append(
                {
                    "phase": phase,
                    "current_technique": cur["technique"],
                    "next_technique": nxt["technique"],
                    "current_family": cur["family"],
                    "next_family": nxt["family"],
                }
            )

    tech_labels = TECHNIQUES
    fam_labels = FAMILIES
    tech_global_prob = {
        label: (tech_global[label] + 1.0) / (sum(tech_global.values()) + len(tech_labels)) for label in tech_labels
    }
    fam_global_prob = {
        label: (fam_global[label] + 1.0) / (sum(fam_global.values()) + len(fam_labels)) for label in fam_labels
    }

    def smooth_counter(counter: Counter, labels: list[str], global_prob: dict[str, float]) -> dict[str, float]:
        total = float(sum(counter.values()))
        denom = total + alpha
        return {label: (counter[label] + alpha * global_prob[label]) / denom for label in labels}

    return {
        "segments": segments,
        "transition_rows": pd.DataFrame(transition_rows),
        "tech_counts": tech_counts,
        "fam_counts": fam_counts,
        "tech_global_prob": tech_global_prob,
        "fam_global_prob": fam_global_prob,
        "smooth_counter": smooth_counter,
        "alpha": alpha,
    }


def external_action_prior_for_rows(prefix: pd.DataFrame, priors: dict[str, object], mix_tech: float) -> np.ndarray:
    out = np.zeros((len(prefix), len(ACTION_CLASSES)), dtype=float)
    tech_counts = priors["tech_counts"]
    fam_counts = priors["fam_counts"]
    tech_global_prob = priors["tech_global_prob"]
    fam_global_prob = priors["fam_global_prob"]
    smooth_counter = priors["smooth_counter"]

    for i, row in enumerate(prefix.itertuples(index=False)):
        phase = phase_from_prefix_len(int(row.prefix_len))
        last_action = int(row.lag0_actionId)
        last_tech = action_technique(last_action)
        last_fam = action_family(last_action)
        tech_dist = smooth_counter(tech_counts.get((phase, last_tech), Counter()), TECHNIQUES, tech_global_prob)
        fam_dist = smooth_counter(fam_counts.get((phase, last_fam), Counter()), FAMILIES, fam_global_prob)
        tech_prior = map_dist_to_actions(tech_dist, TECHNIQUE_TO_ACTIONS)
        fam_prior = map_dist_to_actions(fam_dist, FAMILY_TO_ACTIONS)
        out[i] = normalize_prob(mix_tech * tech_prior + (1.0 - mix_tech) * fam_prior)
    return normalize_prob(out)


def arithmetic_blend(base: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    return normalize_prob((1.0 - weight) * base + weight * prior)


def geometric_blend(base: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    logp = (1.0 - weight) * np.log(np.clip(base, 1e-12, 1.0)) + weight * np.log(np.clip(prior, 1e-12, 1.0))
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_prob(np.exp(logp))


def evaluate_variant(
    meta: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    point_mult: dict[str, list[float]],
    server_prob: np.ndarray,
    mode: str,
) -> dict[str, object]:
    action_mult = tune_segmented_multipliers(meta, action_prob, ACTION_CLASSES, "action", mode)
    action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
        "action_multipliers": action_mult,
        "action_pred": action_pred,
        "point_pred": point_pred,
    }


def write_mapping_report(output: str) -> None:
    rows = []
    for action in ACTION_CLASSES:
        rows.append(
            {
                "source": "AICUP",
                "label": action,
                "canonical_family": action_family(action),
                "canonical_technique": action_technique(action),
                "notes": "coarse mapping; not an exact OpenTTGames taxonomy",
            }
        )
    for technique in TECHNIQUES:
        rows.append(
            {
                "source": "OpenTTGames",
                "label": technique,
                "canonical_family": opentt_family(technique),
                "canonical_technique": technique,
                "notes": "external technique; maps only to coarse AICUP groups",
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    train = add_role_and_score_features(pd.read_csv(args.train))
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, args.max_lag))
    features_for_merge = [
        "rally_uid",
        "prefix_len",
        "next_actionId",
        "next_pointId",
        "serverGetPoint",
        "lag0_actionId",
    ]
    prefix_small = prefix_df[features_for_merge].copy()

    with open(args.v3_oof, "rb") as f:
        v3 = pickle.load(f)
    meta = normalize_meta(v3["valid_meta"])
    merged = meta.merge(
        prefix_small,
        on=["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"],
        how="left",
        validate="one_to_one",
    )
    if merged["lag0_actionId"].isna().any():
        raise ValueError("Could not recover lag0_actionId for all OOF rows.")
    merged["lag0_actionId"] = merged["lag0_actionId"].astype(int)

    events = pd.read_csv(args.events)
    write_mapping_report(args.mapping_report)

    v3_action, v3_point, v3_server = compose_v3(v3)
    base_eval = evaluate_variant(
        meta,
        v3_action,
        v3_point,
        v3["tuning"].point_multipliers,
        v3_server,
        v3["tuning"].bins_mode,
    )

    report_rows: list[dict[str, object]] = [
        {
            "variant": "v3_base",
            "alpha": np.nan,
            "mix_tech": np.nan,
            "method": "none",
            "prior_weight": 0.0,
            "action_macro_f1": base_eval["action_macro_f1"],
            "point_macro_f1": base_eval["point_macro_f1"],
            "server_auc": base_eval["server_auc"],
            "overall": base_eval["overall"],
            "action_churn_vs_base": 0.0,
        }
    ]
    candidates: list[dict[str, object]] = [
        {
            "variant": "v3_base",
            "eval": base_eval,
            "action_prob": v3_action,
            "alpha": None,
            "mix_tech": None,
            "method": "none",
            "prior_weight": 0.0,
        }
    ]

    transition_report_written = False
    for alpha in [1.0, 5.0, 20.0, 50.0]:
        priors = estimate_external_priors(events, alpha)
        if not transition_report_written:
            transition_df = priors["transition_rows"]
            if len(transition_df):
                counts = (
                    transition_df.groupby(
                        ["phase", "current_family", "next_family", "current_technique", "next_technique"],
                        dropna=False,
                    )
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
            else:
                counts = pd.DataFrame()
            counts.to_csv(args.transition_report, index=False)
            transition_report_written = True

        for mix_tech in [0.0, 0.5, 1.0]:
            prior_action = external_action_prior_for_rows(merged, priors, mix_tech=mix_tech)
            for method in ["arith", "geom"]:
                for weight in [0.01, 0.02, 0.05, 0.1, 0.2]:
                    if method == "arith":
                        action_prob = arithmetic_blend(v3_action, prior_action, weight)
                    else:
                        action_prob = geometric_blend(v3_action, prior_action, weight)
                    ev = evaluate_variant(
                        meta,
                        action_prob,
                        v3_point,
                        v3["tuning"].point_multipliers,
                        v3_server,
                        v3["tuning"].bins_mode,
                    )
                    churn = float((ev["action_pred"] != base_eval["action_pred"]).mean())
                    variant = f"alpha{alpha:g}_mix{mix_tech:g}_{method}_w{weight:g}"
                    report_rows.append(
                        {
                            "variant": variant,
                            "alpha": alpha,
                            "mix_tech": mix_tech,
                            "method": method,
                            "prior_weight": weight,
                            "action_macro_f1": ev["action_macro_f1"],
                            "point_macro_f1": ev["point_macro_f1"],
                            "server_auc": ev["server_auc"],
                            "overall": ev["overall"],
                            "action_churn_vs_base": churn,
                        }
                    )
                    candidates.append(
                        {
                            "variant": variant,
                            "eval": ev,
                            "action_prob": action_prob,
                            "alpha": alpha,
                            "mix_tech": mix_tech,
                            "method": method,
                            "prior_weight": weight,
                            "action_churn_vs_base": churn,
                        }
                    )

    summary = pd.DataFrame(report_rows).sort_values("overall", ascending=False).reset_index(drop=True)
    summary.to_csv(args.summary, index=False)

    best_name = str(summary.iloc[0]["variant"])
    best = next(c for c in candidates if c["variant"] == best_name)
    prefix = prefix_report(meta, best["eval"]["action_pred"], best["eval"]["point_pred"], v3_server)
    prefix.to_csv(args.prefix_report, index=False)

    selected = {
        "best_variant": best_name,
        "metrics": {
            "action_macro_f1": best["eval"]["action_macro_f1"],
            "point_macro_f1": best["eval"]["point_macro_f1"],
            "server_auc": best["eval"]["server_auc"],
            "overall": best["eval"]["overall"],
        },
        "alpha": best["alpha"],
        "mix_tech": best["mix_tech"],
        "method": best["method"],
        "prior_weight": best["prior_weight"],
        "action_churn_vs_base": best.get("action_churn_vs_base", 0.0),
        "action_multipliers": best["eval"]["action_multipliers"],
        "point_policy": "fixed_v3_point_probabilities_and_multipliers",
        "server_policy": "fixed_v3_server",
        "submit_recommendation": bool(
            best["eval"]["overall"] >= base_eval["overall"] + 0.0015
            and best.get("action_churn_vs_base", 0.0) <= 0.08
        ),
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")

    metadata = {
        "external_events": args.events,
        "v3_oof": args.v3_oof,
        "prefix_rows_used_for_prior": int(len(merged)),
        "mapping_policy": {
            "AICUP_action_to_family": ACTION_TO_FAMILY,
            "AICUP_action_to_technique": ACTION_TO_TECHNIQUE,
            "OpenTT_technique_to_actions": TECHNIQUE_TO_ACTIONS,
            "OpenTT_family_to_actions": FAMILY_TO_ACTIONS,
        },
        "selected": selected,
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rec_lines = [
        "# R22 OpenTTGames Transition-Prior Recommendation",
        "",
        "Status: completed OOF diagnostic. No submission generated.",
        "",
        "## Best Result",
        "",
        "```text",
        f"variant = {best_name}",
        f"action = {selected['metrics']['action_macro_f1']:.6f}",
        f"point  = {selected['metrics']['point_macro_f1']:.6f}",
        f"server = {selected['metrics']['server_auc']:.6f}",
        f"overall = {selected['metrics']['overall']:.6f}",
        f"action churn vs V3 = {selected['action_churn_vs_base']:.4%}",
        "```",
        "",
        "## Baseline",
        "",
        "```text",
        f"V3 action = {base_eval['action_macro_f1']:.6f}",
        f"V3 point  = {base_eval['point_macro_f1']:.6f}",
        f"V3 server = {base_eval['server_auc']:.6f}",
        f"V3 overall = {base_eval['overall']:.6f}",
        "```",
        "",
        "## Decision",
        "",
        "- Submit only if `submit_recommendation` is true in `r22_selected.json`.",
        "- This experiment only biases action. Point/server are fixed to V3.",
        "- If selected weight is zero/base or gain is below threshold, treat R22 as a documentation/prior feature negative result.",
    ]
    Path(args.recommendation).write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    print(summary.head(15).to_string(index=False))
    print("selected", json.dumps(selected, indent=2))
    print(f"wrote {args.summary}")
    print(f"wrote {args.prefix_report}")
    print(f"wrote {args.mapping_report}")
    print(f"wrote {args.transition_report}")
    print(f"wrote {args.selected}")
    print(f"wrote {args.feature_report}")
    print(f"wrote {args.recommendation}")


if __name__ == "__main__":
    main()
