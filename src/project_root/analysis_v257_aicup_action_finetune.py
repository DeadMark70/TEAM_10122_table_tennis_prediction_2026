from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import (
    context_weights,
    evaluate_action,
    feature_columns,
    load_action_context,
    predict_full,
)


OUTDIR = Path("v257_aicup_action_finetune")
ENCODER_CONFIG = Path("v257_shuttlenet_repretrain") / "v257_encoder_config.json"
RNG = 257


def _action_family_id(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=int)
    out = np.full(action.shape, 4, dtype=int)
    out[np.isin(action, [15, 16, 17, 18])] = 0
    out[np.isin(action, [1, 2, 6, 10, 11, 13])] = 1
    out[np.isin(action, [3, 4, 5, 7, 8, 9, 12, 14])] = 2
    out[np.isin(action, [0])] = 3
    return out


def v257_transfer_features(rows: pd.DataFrame, config: dict | None) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    prefix = pd.to_numeric(rows.get("prefix_len", 0), errors="coerce").fillna(0).astype(float)
    out["v257_prefix_norm"] = np.clip(prefix / 16.0, 0.0, 1.0)
    for col in ("phase", "lag0_family", "lag1_family", "lag0_actionId", "lag1_actionId"):
        if col in rows:
            vals = rows[col].astype(str)
            top = vals.value_counts().head(20).index
            out = pd.concat([out, pd.get_dummies(vals.where(vals.isin(top), "__other__"), prefix=f"v257_{col}", dtype=float)], axis=1)
    if config:
        out["v257_external_family_vocab_size"] = float(len(config.get("vocabs", {}).get("family", {})))
        out["v257_external_shot_vocab_size"] = float(len(config.get("vocabs", {}).get("shot", {})))
    else:
        out["v257_external_family_vocab_size"] = 0.0
        out["v257_external_shot_vocab_size"] = 0.0
    return out.fillna(0.0)


def align_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(train.columns) | set(test.columns))
    return train.reindex(columns=cols, fill_value=0.0), test.reindex(columns=cols, fill_value=0.0)


def load_encoder_config() -> dict | None:
    if not ENCODER_CONFIG.exists():
        return None
    return json.loads(ENCODER_CONFIG.read_text(encoding="utf-8"))


def train_transfer_action(ctx: dict, config: dict | None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    base_cols = feature_columns(rows, drop_keywords=("next_",))
    base_train = rows.loc[:, base_cols].copy().fillna(0.0)
    base_test = test_rows.reindex(columns=base_cols, fill_value=0.0).fillna(0.0)
    x_train, x_test = align_frames(
        pd.concat([base_train.reset_index(drop=True), v257_transfer_features(rows, config).reset_index(drop=True)], axis=1),
        pd.concat([base_test.reset_index(drop=True), v257_transfer_features(test_rows, config).reset_index(drop=True)], axis=1),
    )
    weights = context_weights(rows, test_rows)
    oof = np.zeros((len(rows), 19), dtype=float)
    test_sum = np.zeros((len(test_rows), 19), dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        model = ExtraTreesClassifier(
            n_estimators=80,
            min_samples_leaf=5,
            random_state=RNG + int(fold),
            n_jobs=1,
        )
        model.fit(x_train.loc[train].fillna(0), y[train], sample_weight=weights[train])
        oof[valid] = predict_full(model, x_train.loc[valid].fillna(0))
        test_sum += predict_full(model, x_test.fillna(0))
        metrics.append({"fold": int(fold), "valid_rows": int(valid.sum()), "features": int(x_train.shape[1])})
    return normalize_probability_rows(oof), normalize_probability_rows(test_sum / max(len(metrics), 1)), metrics


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    config = load_encoder_config()
    ctx = load_action_context()
    weights = context_weights(ctx["rows"], ctx["test_rows"])
    model_oof, model_test, fold_metrics = train_transfer_action(ctx, config)
    anchor_oof = np.eye(19, dtype=float)[ctx["v173_oof"]]
    candidates = {
        "v173_anchor": ctx["v173_oof"],
        "v257_raw_action": model_oof.argmax(axis=1),
        "v257_v173blend_w0p05": blend_probabilities(anchor_oof, model_oof, 0.05).argmax(axis=1),
        "v257_v173blend_w0p10": blend_probabilities(anchor_oof, model_oof, 0.10).argmax(axis=1),
        "v257_v173blend_w0p20": blend_probabilities(anchor_oof, model_oof, 0.20).argmax(axis=1),
    }
    gate = candidates["v257_v173blend_w0p10"].copy()
    anchor_family = _action_family_id(ctx["v173_oof"])
    gate_family = _action_family_id(gate)
    gate[anchor_family != gate_family] = ctx["v173_oof"][anchor_family != gate_family]
    candidates["v257_v173blend_classgate"] = gate

    records = [evaluate_action(name, ctx["y"], pred, ctx["v173_oof"], weights) for name, pred in candidates.items()]
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"], ascending=[False, False, False])
    search.to_csv(OUTDIR / "v257_action_search.csv", index=False)
    non_anchor = search[search["candidate"].ne("v173_anchor")]
    best = non_anchor.iloc[0].to_dict() if len(non_anchor) else {}
    best_delta = float(best.get("delta_vs_v173_anchor", 0.0))
    public_delta = float(best.get("iw_delta_vs_v173", 0.0))
    if best_delta >= 0.003 and public_delta >= 0.001:
        verdict = "CANDIDATE_FOR_PUBLIC_PROBE"
    elif best_delta > 0 and public_delta >= 0:
        verdict = "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    else:
        verdict = "LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "status": "DONE_WITH_CONCERNS",
        "concern": "CPU-safe smoke fine-tune uses transferred corpus/config features; full PyTorch encoder fine-tune remains future work.",
        "encoder_config_found": config is not None,
        "best_candidate": best,
        "verdict": verdict,
        "fold_metrics": fold_metrics,
        "test_probability_shape": list(model_test.shape),
        "submissions_written": False,
        "submission_note": "Worker B avoided controller-owned upload_candidates_20260519 and submissions/selected paths.",
    }
    (OUTDIR / "v257_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v257_report.md").write_text(
        "# V257 AICUP Action Fine-Tune\n\n"
        f"- Encoder config found: {config is not None}\n"
        f"- Best candidate: {best.get('candidate', 'none')}\n"
        f"- OOF delta vs V173: {best_delta:.6f}\n"
        f"- Public-like delta: {public_delta:.6f}\n"
        f"- Verdict: {verdict}\n"
        "- Submissions written: no (controller-owned integration paths avoided)\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": verdict, "best_delta": best_delta, "public_like_delta": public_delta}))


if __name__ == "__main__":
    main()
