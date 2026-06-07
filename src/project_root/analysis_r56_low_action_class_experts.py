"""R56 low-action-class expert blending.

Use OOF evidence to decide which expert columns to trust for weak action
classes, while keeping point/server fixed to the current safe branch.

This is different from R49:
- R49 blended whole expert distributions.
- R56 blends only selected action class columns, e.g. 8/9/12/14.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from baseline_lgbm import ACTION_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows
from analysis_r48_action_meta_stacker import build_current_oof_action


OUTDIR = Path("r56_low_action_class_experts")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
LOW_ACTION_CLASSES = [0, 3, 4, 7, 8, 9, 11, 12, 14]
RARE_CLASSES = [8, 9, 12, 14]
SAFE_MAX_CHURN = 0.10


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


def load_artifact() -> dict:
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


def class_report(y: np.ndarray, pred: np.ndarray, name: str) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=ACTION_CLASSES, zero_division=0)
    return pd.DataFrame(
        {
            "model": name,
            "actionId": ACTION_CLASSES,
            "support": s,
            "pred_count": [(pred == c).sum() for c in ACTION_CLASSES],
            "precision": p,
            "recall": r,
            "f1": f,
        }
    )


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


def blend_columns(base: np.ndarray, expert_probs: dict[str, np.ndarray], choices: dict[int, tuple[str, float]]) -> np.ndarray:
    out = base.copy()
    for cls, (expert_name, weight) in choices.items():
        out[:, cls] = (1.0 - weight) * base[:, cls] + weight * expert_probs[expert_name][:, cls]
    return normalize_rows(out)


def describe_candidate(
    name: str,
    prob: np.ndarray,
    meta: pd.DataFrame,
    y: np.ndarray,
    base_pred: np.ndarray,
    mult: dict,
    choices: dict[int, tuple[str, float]],
) -> dict:
    pred = apply_action(prob, meta, mult)
    return {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "churn_vs_r42_base": float(np.mean(pred != base_pred)),
        "choices": {str(k): {"expert": v[0], "weight": v[1]} for k, v in sorted(choices.items())},
        "pred8_count": int((pred == 8).sum()),
        "pred9_count": int((pred == 9).sum()),
        "pred12_count": int((pred == 12).sum()),
        "pred14_count": int((pred == 14).sum()),
        "pred0_count": int((pred == 0).sum()),
    }


def greedy_class_search(
    base: np.ndarray,
    experts: dict[str, np.ndarray],
    meta: pd.DataFrame,
    y: np.ndarray,
    mult: dict,
    classes: list[int],
    max_weight: float,
) -> tuple[dict[int, tuple[str, float]], list[dict]]:
    choices: dict[int, tuple[str, float]] = {}
    history: list[dict] = []
    current = base.copy()
    current_pred = apply_action(current, meta, mult)
    current_f1 = float(f1_score(y, current_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    weights = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    weights = [w for w in weights if w <= max_weight]
    for _ in range(2):
        improved = False
        for cls in classes:
            best = (current_f1, None, None, current)
            for expert_name, expert_prob in experts.items():
                if expert_name == "r42_base":
                    continue
                for w in weights:
                    trial_choices = dict(choices)
                    trial_choices[cls] = (expert_name, w)
                    trial = blend_columns(base, experts, trial_choices)
                    pred = apply_action(trial, meta, mult)
                    score = float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0))
                    if score > best[0] + 1e-8:
                        best = (score, expert_name, w, trial)
            if best[1] is not None:
                current_f1 = float(best[0])
                choices[cls] = (str(best[1]), float(best[2]))
                current = best[3]
                history.append({"class": cls, "expert": best[1], "weight": best[2], "action_macro_f1": current_f1})
                improved = True
        if not improved:
            break
    return choices, history


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "path": str(path),
        "upload_path": str(UPLOAD_DIR / name),
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
        "action0_count": int((pred == 0).sum()),
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    art = load_artifact()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    y = meta["next_actionId"].to_numpy(dtype=int)
    mult = art["selected"]["action_multipliers"]

    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    experts_oof = {
        "r42_base": r42_oof,
        "current": current_oof,
        "golden": golden_oof,
        "v49_familiar": art["experts_oof"]["v49_familiar_player"],
        "v50_short": art["experts_oof"]["v50_short_prefix"],
        "v49_robust": art["experts_oof"]["v49_robust_unseen"],
        "v48_macro": art["experts_oof"]["v48_macro_f1_weighted"],
        "v48_rare": art["experts_oof"]["v48_rare_control"],
    }
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)
    experts_test = {
        "r42_base": r42_test,
        "current": current_test,
        "golden": golden_test,
        "v49_familiar": art["experts_test"]["v49_familiar_player"],
        "v50_short": art["experts_test"]["v50_short_prefix"],
        "v49_robust": art["experts_test"]["v49_robust_unseen"],
        "v48_macro": art["experts_test"]["v48_macro_f1_weighted"],
        "v48_rare": art["experts_test"]["v48_rare_control"],
    }

    base_pred = apply_action(r42_oof, meta, mult)
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    reports = []
    for name, prob in experts_oof.items():
        pred = apply_action(prob, meta, mult)
        reports.append(class_report(y, pred, name))
    per_class = pd.concat(reports, ignore_index=True)
    per_class.to_csv(OUTDIR / "r56_action_class_report_by_expert.csv", index=False)

    # Greedy searches with different risk budgets.
    candidates = [
        ("conservative_low", LOW_ACTION_CLASSES, 0.15),
        ("moderate_low", LOW_ACTION_CLASSES, 0.30),
        ("aggressive_low", LOW_ACTION_CLASSES, 0.50),
        ("rare_only", RARE_CLASSES, 0.40),
        ("zero_control_defense", [0, 8, 9, 11, 12, 14], 0.35),
    ]
    rows = [
        {
            "candidate": "r42_base",
            "action_macro_f1": base_f1,
            "churn_vs_r42_base": 0.0,
            "choices": {},
            "pred8_count": int((base_pred == 8).sum()),
            "pred9_count": int((base_pred == 9).sum()),
            "pred12_count": int((base_pred == 12).sum()),
            "pred14_count": int((base_pred == 14).sum()),
            "pred0_count": int((base_pred == 0).sum()),
        }
    ]
    histories = {}
    prob_by_candidate = {"r42_base": r42_oof}
    test_prob_by_candidate = {"r42_base": r42_test}
    for label, classes, max_w in candidates:
        choices, hist = greedy_class_search(r42_oof, experts_oof, meta, y, mult, classes, max_w)
        histories[label] = hist
        prob = blend_columns(r42_oof, experts_oof, choices)
        test_prob = blend_columns(r42_test, experts_test, choices)
        rows.append(describe_candidate(label, prob, meta, y, base_pred, mult, choices))
        prob_by_candidate[label] = prob
        test_prob_by_candidate[label] = test_prob

    # Handcrafted low-risk probes based on R41: golden for classes where V64 wins,
    # familiar-player memory for classes where it tends to improve OOF globally.
    hand_choices = {
        "golden_rare_control_w0p2": {8: ("golden", 0.20), 9: ("golden", 0.20), 12: ("golden", 0.20), 14: ("golden", 0.20)},
        "golden_attack_control_w0p15": {
            3: ("golden", 0.15),
            4: ("golden", 0.15),
            7: ("golden", 0.15),
            8: ("golden", 0.15),
            9: ("golden", 0.15),
            11: ("golden", 0.15),
            12: ("golden", 0.15),
            14: ("golden", 0.15),
        },
        "familiar_low_w0p1": {c: ("v49_familiar", 0.10) for c in LOW_ACTION_CLASSES},
        "mixed_golden_familiar": {
            4: ("golden", 0.20),
            7: ("golden", 0.20),
            8: ("golden", 0.25),
            9: ("golden", 0.25),
            12: ("golden", 0.25),
            14: ("golden", 0.20),
            0: ("v49_familiar", 0.10),
            11: ("v49_familiar", 0.10),
        },
    }
    for label, choices in hand_choices.items():
        prob = blend_columns(r42_oof, experts_oof, choices)
        test_prob = blend_columns(r42_test, experts_test, choices)
        rows.append(describe_candidate(label, prob, meta, y, base_pred, mult, choices))
        prob_by_candidate[label] = prob
        test_prob_by_candidate[label] = test_prob

    search = pd.DataFrame(rows).sort_values("action_macro_f1", ascending=False)
    search.to_csv(OUTDIR / "r56_oof_class_blend_search.csv", index=False)
    (OUTDIR / "r56_greedy_histories.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    generated = []
    for row in search.to_dict(orient="records"):
        label = row["candidate"]
        if label == "r42_base":
            continue
        if float(row["churn_vs_r42_base"]) > SAFE_MAX_CHURN:
            continue
        pred = apply_action(test_prob_by_candidate[label], test_meta, mult)
        name = f"submission_r56_{label}_current_point_server.csv"
        info = write_submission(test_meta, pred, current_sub, name)
        info.update(
            {
                "source_oof_action_f1": row["action_macro_f1"],
                "source_oof_churn": row["churn_vs_r42_base"],
                "source_choices": row["choices"],
            }
        )
        generated.append(info)
        if len(generated) >= 6:
            break
    pd.DataFrame(generated).to_csv(OUTDIR / "r56_generated_candidates.csv", index=False)

    report = {
        "base_action_f1": base_f1,
        "low_action_classes": LOW_ACTION_CLASSES,
        "rare_classes": RARE_CLASSES,
        "best_oof": search.head(20).to_dict(orient="records"),
        "generated": generated,
        "note": "point/server fixed to current R34; R42 public anchor remains 0.3342886.",
    }
    (OUTDIR / "r56_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(20).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
