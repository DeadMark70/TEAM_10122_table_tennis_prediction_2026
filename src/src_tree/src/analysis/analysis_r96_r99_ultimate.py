"""R96/R98/R99 final-push action/point post-processing experiments.

R96:
  Ultimate blend sweep combining R92 style-injected proxy action and R93/R88
  legality-aware action, with optional R83 point-style upgrade.

R98:
  High-confidence pseudo-label diagnostic. This does not overwrite labels or
  train a final risky model; it reports how many test rows would qualify under
  strict confidence rules and emits a conservative action-prior calibration
  candidate if enough rows exist.

R99:
  Terminal soft corrector. Use V3 point0 probability as a terminal proxy and
  softly reweight terminal/nonterminal action groups. No hard forcing.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r82_r86_point_style import (
    ConditionalStyleEncoder,
    add_conditional_style_features,
    align_prefix_meta,
    aligned_multiclass_proba,
    blend_point,
    clean_float,
    compose_v3_full_point,
    prepare_prefix_features,
)
from analysis_r67_r70_meta_priors import R63_OOF_PATH, apply_action
from analysis_r87_r90_point_action_meta import style_gated_multiplier
from analysis_r91_r95_phase_mask_style import apply_soft_legality, build_legality_lookup, legality_prior_for_rows, legality_prior_oof
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r96_r99_ultimate")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R91_DIR = Path("r91_r95_phase_mask_style")


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_submission(
    test_meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
    name: str,
    extra: dict | None = None,
) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path = SELECTED_DIR / name
    selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    if extra:
        info.update(extra)
    return info


def terminal_soft_corrector(action_prob: np.ndarray, point_prob: np.ndarray, beta: float) -> np.ndarray:
    """Softly reweight terminal/finalizing vs transitional actions.

    Uses pointId=0 probability as a terminal proxy. This is intentionally soft:
    no class is forced to zero.
    """
    out = action_prob.copy()
    p_term = np.clip(point_prob[:, 0], 0.0, 1.0)
    finalizing = [0, 3, 12, 14]
    transitional = [6, 8, 9, 10, 11, 13]
    attack = [1, 2, 4, 5, 7]
    out[:, finalizing] *= np.exp(beta * (p_term[:, None] - 0.5))
    out[:, transitional] *= np.exp(-0.55 * beta * (p_term[:, None] - 0.5))
    out[:, attack] *= np.exp(0.25 * beta * (0.5 - p_term[:, None]))
    return normalize_rows(out)


def pseudo_calibrate_from_test(base: np.ndarray, action_prob: np.ndarray, thresholds: list[float]) -> tuple[dict, dict[str, np.ndarray]]:
    """Conservative transductive pseudo-label calibration.

    Instead of appending pseudo rows and retraining, estimate a tiny class-prior
    multiplier from very high confidence test rows. This is lower risk and
    still reveals whether R98 has enough signal.
    """
    out = {}
    probs = {}
    maxp = action_prob.max(axis=1)
    pred = action_prob.argmax(axis=1)
    global_prior = np.bincount(pred, minlength=len(ACTION_CLASSES)).astype(float)
    global_prior = (global_prior + 1.0) / (global_prior.sum() + len(ACTION_CLASSES))
    for th in thresholds:
        mask = maxp >= th
        counts = np.bincount(pred[mask], minlength=len(ACTION_CLASSES)).astype(float)
        if mask.sum() == 0:
            mult = np.ones(len(ACTION_CLASSES), dtype=float)
        else:
            prior = (counts + 10.0 * global_prior) / (counts.sum() + 10.0)
            mult = np.clip((prior / global_prior) ** 0.08, 0.85, 1.18)
        calibrated = normalize_rows(base * mult[None, :])
        key = f"r98_th{clean_float(th)}"
        probs[key] = calibrated
        out[key] = {
            "threshold": th,
            "n_pseudo": int(mask.sum()),
            "pseudo_rate": float(mask.mean()),
            "mult_min": float(mult.min()),
            "mult_max": float(mult.max()),
        }
    return out, probs


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    train_raw, test_raw, prefix, test_prefix, _features = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)

    v3_oof = load_pickle("oof_proba_v3.pkl")
    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]):
        raise ValueError("V3 OOF does not align.")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    test_prefix_v3, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    if not test_prefix_v3["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 test rows do not align.")

    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    r63_oof = np.load(R63_OOF_PATH)
    r67_oof = normalize_rows(0.80 * r42_oof + 0.20 * r63_oof)

    current_test = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test + 0.20 * golden_test)

    # Exact R63 full-test reconstruction, matching R91/R95.
    encoder = ConditionalStyleEncoder(k=8, alpha=35.0, beta=35.0, seed=7350).fit(pd.concat([train_raw, test_raw], ignore_index=True), train_raw)
    train_cond = add_conditional_style_features(prefix, encoder)
    test_cond = add_conditional_style_features(test_prefix, encoder)
    cond_cols = [c for c in train_cond.columns if c.startswith("cond_")]
    cond_features = [c for c in _features if c in train_cond.columns] + cond_cols
    r63_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=7350,
        n_jobs=-1,
        verbosity=-1,
    )
    r63_model.fit(train_cond[cond_features], train_cond["next_actionId"], sample_weight=class_weight_sample(train_cond["next_actionId"]))
    r63_test = aligned_multiclass_proba(r63_model, test_cond[cond_features], ACTION_CLASSES)
    r67_test = normalize_rows(0.80 * r42_test + 0.20 * r63_test)

    # R92 arrays generated by R91-R95.
    r92_oof = np.load(R91_DIR / "r92_style_injected_oof_action.npy")
    r92_test = np.load(R91_DIR / "r92_style_injected_test_action.npy")
    if len(r92_oof) != len(meta) or len(r92_test) != len(test_meta):
        raise ValueError("R92 arrays are not aligned.")

    # R93 best branch: R88 anchor + soft legality.
    target_classes = [0, 3, 7, 8, 9, 11, 12, 14]
    r88_oof = style_gated_multiplier(r67_oof, r63_oof, alpha=0.10, beta=0.10, cap=3.0, target_classes=target_classes)
    r88_test = style_gated_multiplier(r67_test, r63_test, alpha=0.10, beta=0.10, cap=3.0, target_classes=target_classes)
    legality_oof, support_oof, _ = legality_prior_oof(rows, prefix)
    legality_test, support_test, _ = legality_prior_for_rows(test_prefix, build_legality_lookup(prefix))
    r93_oof = apply_soft_legality(r88_oof, legality_oof, support_oof, gamma=0.05, floor=0.65, cap=2.0, min_support=100)
    r93_test = apply_soft_legality(r88_test, legality_test, support_test, gamma=0.05, floor=0.65, cap=2.0, min_support=100)

    # R83 point variants: use generated labels for test and OOF metrics from R91 logs.
    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    server_test = current_sub["serverGetPoint"].to_numpy(dtype=float)
    v3_point_test_pred = apply_segmented_multipliers(test_meta, v3_point_test, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
    r83_w075_point_pred = pd.read_csv(R91_DIR / "submission_r95_r67_w0p2_r83point_w0p075_current_server.csv")["pointId"].to_numpy(dtype=int)
    r83_w15_point_pred = pd.read_csv(R91_DIR / "submission_r95_r67_w0p2_r83point_w0p15_current_server.csv")["pointId"].to_numpy(dtype=int)

    y_action = meta["next_actionId"].to_numpy(dtype=int)
    y_point = meta["next_pointId"].to_numpy(dtype=int)
    base_action_pred = apply_action(r67_oof, meta, art["selected"]["action_multipliers"])
    base_action_f1 = f1_score(y_action, base_action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    base_point_pred = apply_segmented_multipliers(meta, v3_point_oof, art["selected"]["point_multipliers"], POINT_CLASSES, "two")
    base_point_f1 = f1_score(y_point, base_point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)

    rows_out: list[dict] = []
    action_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {"r67": (r67_oof, r67_test)}

    # R96: ultimate blend sweep.
    for w92 in [0.10, 0.15, 0.20, 0.25]:
        r92_blend_oof = normalize_rows((1.0 - w92) * r67_oof + w92 * r92_oof)
        # Test counterpart available exactly.
        r92_blend_test = normalize_rows((1.0 - w92) * r67_test + w92 * r92_test)
        for w93 in [0.10, 0.20, 0.30, 0.40, 0.50]:
            name = f"r96_r92w{clean_float(w92)}_r93w{clean_float(w93)}"
            prob = normalize_rows((1.0 - w93) * r92_blend_oof + w93 * r93_oof)
            pred = apply_action(prob, meta, art["selected"]["action_multipliers"])
            f1 = f1_score(y_action, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
            churn = float(np.mean(pred != base_action_pred))
            rows_out.append({"variant": name, "kind": "r96_ultimate", "action_macro_f1": float(f1), "action_churn": churn, "w92": w92, "w93": w93})
            if f1 > base_action_f1 and churn <= 0.14:
                test_prob = normalize_rows((1.0 - w93) * r92_blend_test + w93 * r93_test)
                action_probs[name] = (prob, test_prob)

    # R98: conservative pseudo-label prior calibration over best R96 base.
    best_r96_row = sorted([r for r in rows_out if r["kind"] == "r96_ultimate"], key=lambda r: r["action_macro_f1"], reverse=True)[0]
    best_r96_name = best_r96_row["variant"]
    best_oof, best_test = action_probs.get(best_r96_name, action_probs["r67"])
    r98_report, r98_probs = pseudo_calibrate_from_test(best_oof, best_test, thresholds=[0.90, 0.95, 0.97, 0.99])
    for key, prob in r98_probs.items():
        pred = apply_action(prob, meta, art["selected"]["action_multipliers"])
        f1 = f1_score(y_action, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
        churn = float(np.mean(pred != base_action_pred))
        rows_out.append({"variant": key, "kind": "r98_pseudo_prior_diag", "action_macro_f1": float(f1), "action_churn": churn, **r98_report[key]})

    # R99: terminal soft corrector on the best R96 variants.
    r99_candidates = list(action_probs.items())[:]
    for source, (prob_oof, prob_test) in r99_candidates:
        if source == "r67":
            continue
        for beta in [0.10, 0.20, 0.35, 0.50, 0.75]:
            name = f"r99_{source}_tb{clean_float(beta)}"
            p = terminal_soft_corrector(prob_oof, v3_point_oof, beta)
            pred = apply_action(p, meta, art["selected"]["action_multipliers"])
            f1 = f1_score(y_action, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
            churn = float(np.mean(pred != base_action_pred))
            rows_out.append({"variant": name, "kind": "r99_terminal_soft", "action_macro_f1": float(f1), "action_churn": churn, "source": source, "beta": beta})
            if f1 > base_action_f1 and churn <= 0.14:
                action_probs[name] = (p, terminal_soft_corrector(prob_test, v3_point_test, beta))

    search = pd.DataFrame(rows_out).sort_values("action_macro_f1", ascending=False)
    search.to_csv(OUTDIR / "r96_r99_action_search.csv", index=False)
    (OUTDIR / "r98_pseudo_report.json").write_text(json.dumps(r98_report, indent=2), encoding="utf-8")

    point_variants = {
        "v3point": (v3_point_test_pred, {"point_variant": "v3point", "oof_point_f1": float(base_point_f1), "point_churn": 0.0}),
        "r83point_w0p075": (r83_w075_point_pred, {"point_variant": "r83point_w0p075", "oof_point_f1": 0.20697186904345172, "point_churn": 0.04754918306102034}),
        "r83point_w0p15": (r83_w15_point_pred, {"point_variant": "r83point_w0p15", "oof_point_f1": 0.20753702352330533, "point_churn": 0.09656552184061354}),
    }

    generated: list[dict] = []
    top = search[(search["action_macro_f1"] > base_action_f1) & (search["action_churn"] <= 0.14)].head(8)
    for variant in top["variant"].tolist():
        if variant not in action_probs:
            continue
        prob_oof, prob_test = action_probs[variant]
        action_pred = apply_action(prob_test, test_meta, art["selected"]["action_multipliers"])
        srow = search[search["variant"].eq(variant)].head(1)
        action_info = {
            "action_variant": variant,
            "oof_action_f1": float(srow["action_macro_f1"].iloc[0]),
            "action_churn": float(srow["action_churn"].iloc[0]),
        }
        for point_key, (point_pred, point_info) in point_variants.items():
            if point_key == "r83point_w0p15" and action_info["action_churn"] > 0.10:
                continue
            name = f"submission_{variant}_{point_key}_current_server.csv"
            generated.append(write_submission(test_meta, action_pred, point_pred, server_test, name, {**action_info, **point_info}))

    pd.DataFrame(generated).to_csv(OUTDIR / "r96_r99_generated_candidates.csv", index=False)
    report = {
        "base": {"r67_action_f1": float(base_action_f1), "v3_point_f1": float(base_point_f1)},
        "best_action": search.head(30).to_dict(orient="records"),
        "generated": generated,
        "r98": r98_report,
    }
    (OUTDIR / "r96_r99_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(35).to_string(index=False))
    print(pd.DataFrame(generated).head(50).to_string(index=False))


if __name__ == "__main__":
    main()
