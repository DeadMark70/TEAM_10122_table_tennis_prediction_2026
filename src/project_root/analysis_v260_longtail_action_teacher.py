"""V260 long-tail calibrated action teachers around V173.

This script is action-only: point/server anchors are loaded only to preserve
test row ordering context, and no submissions are copied to upload locations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from analysis_v243_v247_action_experiment_common import (
    context_weights,
    evaluate_action,
    feature_columns,
    load_action_context,
    predict_full,
)

try:
    from analysis_v259_v262_breakthrough_helpers import normalize_rows_safe, verdict_from_deltas
except ImportError:
    def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=float)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, 0.0, None)
        row_sum = arr.sum(axis=1, keepdims=True)
        zero = row_sum[:, 0] <= 0.0
        if zero.any():
            arr[zero, :] = 1.0 / arr.shape[1]
            row_sum = arr.sum(axis=1, keepdims=True)
        return arr / row_sum

    def verdict_from_deltas(
        delta: float,
        public_like_delta: float,
        strong_delta: float = 0.003,
        strong_public: float = 0.001,
    ) -> str:
        if float(delta) >= strong_delta and float(public_like_delta) >= strong_public:
            return "CANDIDATE_FOR_PUBLIC_PROBE"
        if float(delta) > 0 and float(public_like_delta) >= 0:
            return "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
        return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


OUTDIR = Path("v260_longtail_action_teacher")
WEAK_CLASSES = np.array([0, 3, 4, 5, 7, 8, 9, 12, 14], dtype=int)
N_CLASSES = 19


def align_test_columns(test_rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = test_rows.copy()
    for col in cols:
        if col not in out:
            out[col] = 0
    return out


def class_weights(y: np.ndarray, base_weight: np.ndarray, power: float, cap: float, weak_boost: float = 1.0) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    factors = np.power(counts.mean() / counts, float(power))
    weights = np.asarray(base_weight, dtype=float) * factors[labels]
    if weak_boost != 1.0:
        weights = weights * np.where(np.isin(labels, WEAK_CLASSES), float(weak_boost), 1.0)
    weights = np.clip(weights, 0.05, float(cap))
    return weights / max(float(weights.mean()), 1e-12)


def train_extratrees(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    cols: list[str],
    weights: np.ndarray,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    class_weight: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof = np.zeros((len(rows), N_CLASSES), dtype=float)
    test_sum = np.zeros((len(test_rows), N_CLASSES), dtype=float)
    fold_metrics: list[dict] = []
    folds = sorted(rows["fold"].astype(int).unique())
    x_test = test_rows.loc[:, cols].fillna(0)
    for fold in folds:
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        model = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            class_weight=class_weight,
            random_state=int(seed) + int(fold),
            n_jobs=1,
        )
        model.fit(rows.loc[train, cols].fillna(0), y[train], sample_weight=weights[train])
        oof[valid] = predict_full(model, rows.loc[valid, cols].fillna(0))
        test_sum += predict_full(model, x_test)
        fold_metrics.append(
            {
                "fold": int(fold),
                "valid_rows": int(valid.sum()),
                "train_rows": int(train.sum()),
                "features": int(len(cols)),
                "seed": int(seed) + int(fold),
            }
        )
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / max(len(folds), 1)), fold_metrics


def logit_adjust_probability(prob: np.ndarray, counts: np.ndarray, tau: float) -> np.ndarray:
    p = np.clip(normalize_rows_safe(prob), 1e-8, 1.0)
    prior = (np.asarray(counts, dtype=float) + 1.0)
    prior = prior / prior.sum()
    logits = np.log(p) - float(tau) * np.log(np.clip(prior, 1e-8, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logits))


def weak_probability_boost(prob: np.ndarray, multiplier: float) -> np.ndarray:
    out = normalize_rows_safe(prob).copy()
    out[:, WEAK_CLASSES] *= float(multiplier)
    return normalize_rows_safe(out)


def blend_probabilities(anchor: np.ndarray, teacher: np.ndarray, weight: float) -> np.ndarray:
    a = np.clip(normalize_rows_safe(anchor), 1e-8, 1.0)
    t = np.clip(normalize_rows_safe(teacher), 1e-8, 1.0)
    logp = (1.0 - float(weight)) * np.log(a) + float(weight) * np.log(t)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


def classgate_probability(anchor_prob: np.ndarray, teacher_prob: np.ndarray) -> np.ndarray:
    anchor = normalize_rows_safe(anchor_prob)
    teacher = normalize_rows_safe(teacher_prob)
    teacher_pred = teacher.argmax(axis=1).astype(int)
    use_teacher = np.isin(teacher_pred, WEAK_CLASSES)
    out = anchor.copy()
    out[use_teacher] = teacher[use_teacher]
    return normalize_rows_safe(out)


def build_variants(
    teacher_name: str,
    oof_prob: np.ndarray,
    test_prob: np.ndarray,
    v173_prob_oof: np.ndarray,
    v173_prob_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    variants = {f"v260_{teacher_name}_raw": (normalize_rows_safe(oof_prob), normalize_rows_safe(test_prob))}
    for weight in [0.05, 0.10, 0.20]:
        suffix = f"0p{int(round(weight * 100)):02d}"
        variants[f"v260_{teacher_name}_v173blend_w{suffix}"] = (
            blend_probabilities(v173_prob_oof, oof_prob, weight),
            blend_probabilities(v173_prob_test, test_prob, weight),
        )
    variants[f"v260_{teacher_name}_classgate"] = (
        classgate_probability(v173_prob_oof, oof_prob),
        classgate_probability(v173_prob_test, test_prob),
    )
    return variants


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report_md(report: dict) -> None:
    best = report["best"]
    text = (
        "# V260 Long-Tail Action Teacher\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Best candidate: `{best['candidate']}`\n"
        f"- Best delta vs V173: `{report['best_delta_vs_v173_anchor']:.8f}`\n"
        f"- Best IW delta vs V173: `{best['iw_delta_vs_v173']:.8f}`\n"
        f"- Best weak-class delta vs V173: `{best['weak_delta_vs_v173']:.8f}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n\n"
        "Point and server are fixed to the current anchors. Outputs are local-only under "
        "`v260_longtail_action_teacher/`.\n"
    )
    (OUTDIR / "v260_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    v173_oof = ctx["v173_oof"]
    v173_test = ctx["v173_test"]
    v173_prob_oof = ctx["v173_prob_oof"]
    v173_prob_test = ctx["v173_prob_test"]

    cols = feature_columns(rows)
    test_rows = align_test_columns(test_rows, cols)
    density_weight = context_weights(rows, test_rows, low=0.20, high=5.0)
    counts = np.bincount(y, minlength=N_CLASSES)

    teacher_specs = [
        {
            "name": "balanced_extratrees",
            "weights": class_weights(y, density_weight, power=0.50, cap=5.0),
            "seed": 2601,
            "n_estimators": 220,
            "min_samples_leaf": 3,
            "class_weight": "balanced_subsample",
            "post": None,
        },
        {
            "name": "logit_adjusted_extratrees",
            "weights": class_weights(y, density_weight, power=0.40, cap=4.5),
            "seed": 2602,
            "n_estimators": 200,
            "min_samples_leaf": 4,
            "class_weight": "balanced_subsample",
            "post": "logit",
        },
        {
            "name": "weak_ovr_boosted_extratrees",
            "weights": class_weights(y, density_weight, power=0.55, cap=6.0, weak_boost=1.65),
            "seed": 2603,
            "n_estimators": 240,
            "min_samples_leaf": 3,
            "class_weight": None,
            "post": "weak_boost",
        },
    ]

    records = [evaluate_action("v173_anchor", y, v173_oof, v173_oof, density_weight)]
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fold_metrics: list[dict] = []
    for spec in teacher_specs:
        oof_prob, test_prob, metrics = train_extratrees(
            rows,
            test_rows,
            y,
            cols,
            spec["weights"],
            seed=spec["seed"],
            n_estimators=spec["n_estimators"],
            min_samples_leaf=spec["min_samples_leaf"],
            class_weight=spec["class_weight"],
        )
        if spec["post"] == "logit":
            oof_prob = logit_adjust_probability(oof_prob, counts, tau=0.35)
            test_prob = logit_adjust_probability(test_prob, counts, tau=0.35)
        elif spec["post"] == "weak_boost":
            oof_prob = weak_probability_boost(oof_prob, multiplier=1.18)
            test_prob = weak_probability_boost(test_prob, multiplier=1.18)
        for metric in metrics:
            metric["teacher"] = spec["name"]
        fold_metrics.extend(metrics)
        variants.update(build_variants(spec["name"], oof_prob, test_prob, v173_prob_oof, v173_prob_test))

    test_action_map: dict[str, np.ndarray] = {}
    for name, (prob_oof, prob_test) in variants.items():
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = evaluate_action(name, y, pred, v173_oof, density_weight)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        rec["weak_class_test_rows"] = int(np.isin(test_pred, WEAK_CLASSES).sum())
        records.append(rec)
        test_action_map[name] = test_pred

    search = pd.DataFrame(records).sort_values(
        ["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    search.to_csv(OUTDIR / "v260_action_search.csv", index=False)

    non_anchor = search[search["candidate"].ne("v173_anchor")]
    best = non_anchor.iloc[0].to_dict()
    best_name = str(best["candidate"])
    best_delta = float(best["delta_vs_v173_anchor"])
    best_iw_delta = float(best["iw_delta_vs_v173"])
    verdict = verdict_from_deltas(best_delta, best_iw_delta)
    if verdict == "CANDIDATE_FOR_PUBLIC_PROBE":
        upload_recommendation = "REVIEW_FOR_PUBLIC_PROBE_DO_NOT_UPLOAD_AUTOMATICALLY"
    elif verdict == "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW":
        upload_recommendation = "HOLD_FOR_CONTROLLER_REVIEW"
    else:
        upload_recommendation = "DO_NOT_UPLOAD_KEEP_V173_ACTION"

    best_oof_prob, best_test_prob = variants[best_name]
    np.save(OUTDIR / "v260_oof_action_prob.npy", normalize_rows_safe(best_oof_prob))
    np.save(OUTDIR / "v260_test_action_prob.npy", normalize_rows_safe(best_test_prob))

    names = np.array(list(test_action_map.keys()), dtype=object)
    actions = np.vstack([test_action_map[name] for name in test_action_map]).astype(np.int16)
    np.savez_compressed(
        OUTDIR / "v260_candidate_test_actions.npz",
        candidate_names=names,
        test_actions=actions,
        best_candidate=np.array(best_name, dtype=object),
        v173_test_action=np.asarray(v173_test, dtype=np.int16),
    )

    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "best": best,
        "upload_recommendation": upload_recommendation,
        "weak_classes": WEAK_CLASSES.astype(int).tolist(),
        "teacher_count": len(teacher_specs),
        "variant_count": len(variants),
        "fold_metrics": fold_metrics,
        "best_table": search.head(12).to_dict(orient="records"),
        "outputs": {
            "search_csv": str(OUTDIR / "v260_action_search.csv"),
            "oof_probability": str(OUTDIR / "v260_oof_action_prob.npy"),
            "test_probability": str(OUTDIR / "v260_test_action_prob.npy"),
            "candidate_test_actions": str(OUTDIR / "v260_candidate_test_actions.npz"),
        },
        "notes": [
            "V173 remains the center; variants are raw teacher, low-weight V173 blends, and weak-class gates.",
            "Classgate only admits teacher rows whose teacher argmax is in the required weak class set.",
            "Point/server anchors are unchanged and no upload/submissions directories are written.",
        ],
    }
    (OUTDIR / "v260_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    write_report_md(report)

    summary = {
        "worker": "V260",
        "outdir": str(OUTDIR),
        "verdict": verdict,
        "best_candidate": best_name,
        "best_delta_vs_v173_anchor": best_delta,
        "best_iw_delta_vs_v173": best_iw_delta,
        "upload_recommendation": upload_recommendation,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
