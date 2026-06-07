"""R49 conservative probability blends over V47-V50 action experts.

R48 meta-stacker was high-churn. R49 searches small probability blends around
the public-positive R42 action base, then writes lower-risk submissions.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import ACTION_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows
from analysis_r48_action_meta_stacker import build_current_oof_action


OUT_DIR = Path("r49_expert_blends")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


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


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_PATH, "rb") as f:
        art = pickle.load(f)
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    selected = art["selected"]

    current_oof = build_current_oof_action()
    v64_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * v64_oof)
    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    expert_pairs = {
        "v49_familiar": ("v49_familiar_player", "v49_familiar_player"),
        "v50_short": ("v50_short_prefix", "v50_short_prefix"),
        "v49_robust": ("v49_robust_unseen", "v49_robust_unseen"),
        "v48_macro": ("v48_macro_f1_weighted", "v48_macro_f1_weighted"),
        "v48_rare": ("v48_rare_control", "v48_rare_control"),
    }
    y = meta["next_actionId"].to_numpy(dtype=int)
    base_pred = apply_segmented_multipliers(meta, r42_oof, selected["action_multipliers"], ACTION_CLASSES, "two")
    rows = [
        {
            "candidate": "r42_base",
            "weights": {},
            "action_macro_f1": float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
            "churn_vs_base": 0.0,
        }
    ]
    scored = []
    for label, (oof_name, test_name) in expert_pairs.items():
        for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            prob = normalize_rows((1.0 - w) * r42_oof + w * art["experts_oof"][oof_name])
            pred = apply_segmented_multipliers(meta, prob, selected["action_multipliers"], ACTION_CLASSES, "two")
            row = {
                "candidate": f"r42+{w:g}*{label}",
                "weights": {label: w},
                "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "churn_vs_base": float(np.mean(pred != base_pred)),
                "pred8_count": int((pred == 8).sum()),
                "pred9_count": int((pred == 9).sum()),
                "pred12_count": int((pred == 12).sum()),
                "pred14_count": int((pred == 14).sum()),
            }
            rows.append(row)
            scored.append((row, normalize_rows((1.0 - w) * r42_test + w * art["experts_test"][test_name])))

    # Small two-expert blends around the strongest OOF expert.
    for wf in [0.05, 0.10, 0.15]:
        for ws in [0.03, 0.05, 0.10]:
            if wf + ws > 0.22:
                continue
            prob = normalize_rows(
                (1.0 - wf - ws) * r42_oof
                + wf * art["experts_oof"]["v49_familiar_player"]
                + ws * art["experts_oof"]["v50_short_prefix"]
            )
            pred = apply_segmented_multipliers(meta, prob, selected["action_multipliers"], ACTION_CLASSES, "two")
            row = {
                "candidate": f"r42+{wf:g}*v49_familiar+{ws:g}*v50_short",
                "weights": {"v49_familiar": wf, "v50_short": ws},
                "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
                "churn_vs_base": float(np.mean(pred != base_pred)),
                "pred8_count": int((pred == 8).sum()),
                "pred9_count": int((pred == 9).sum()),
                "pred12_count": int((pred == 12).sum()),
                "pred14_count": int((pred == 14).sum()),
            }
            rows.append(row)
            test_prob = normalize_rows(
                (1.0 - wf - ws) * r42_test
                + wf * art["experts_test"]["v49_familiar_player"]
                + ws * art["experts_test"]["v50_short_prefix"]
            )
            scored.append((row, test_prob))

    search = pd.DataFrame(rows).sort_values("action_macro_f1", ascending=False)
    search.to_csv(OUT_DIR / "r49_oof_blend_search.csv", index=False)

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")
    generated = []
    used = set()
    for row, test_prob in sorted(scored, key=lambda x: (x[0]["action_macro_f1"], -x[0]["churn_vs_base"]), reverse=True):
        if row["churn_vs_base"] > 0.10:
            continue
        key = row["candidate"]
        if key in used:
            continue
        used.add(key)
        pred = apply_segmented_multipliers(test_meta, test_prob, selected["action_multipliers"], ACTION_CLASSES, "two")
        safe_name = (
            "submission_r49_"
            + key.replace("r42+", "").replace("*", "x").replace("+", "_").replace(".", "p")
            + "_current_point_server.csv"
        )
        info = write_submission(test_meta, pred, current_sub, safe_name)
        info.update({"source_candidate": key, "source_oof_action_f1": row["action_macro_f1"], "source_oof_churn": row["churn_vs_base"]})
        generated.append(info)
        if len(generated) >= 8:
            break

    pd.DataFrame(generated).to_csv(OUT_DIR / "r49_generated_candidates.csv", index=False)
    report = {
        "base_oof": rows[0],
        "best_oof": search.head(10).to_dict(orient="records"),
        "generated": generated,
    }
    (OUT_DIR / "r49_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(15).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
