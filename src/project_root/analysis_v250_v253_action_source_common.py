"""Common utilities for V250-V253 action source experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows, select_top_changes
from analysis_v243_v247_action_experiment_common import (
    context_weights,
    evaluate_action,
    feature_columns,
    load_action_context,
    write_submission,
)
from analysis_v250_v253_action_source_helpers import phase_family_one_hot
from baseline_lgbm import ACTION_CLASSES


def response_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_candidates = [
        "prefix_len",
        "phase_id",
        "lag0_exists",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "lag0_handId",
        "lag0_positionId",
        "lag0_point_depth",
        "lag0_point_side",
        "scoreSelf",
        "scoreOther",
        "scoreDiff",
        "scoreTotal",
        "numberGame",
        "next_hitter_is_server",
        "is_server_hitter",
        "next_hitter_id",
        "next_receiver_id",
    ]
    parts = []
    numeric = pd.DataFrame(index=rows.index)
    for col in numeric_candidates:
        if col in rows:
            numeric[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    parts.append(numeric)
    parts.append(phase_family_one_hot(rows))
    out = pd.concat(parts, axis=1).fillna(0.0)
    return out.astype(float)


def align_feature_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(train.columns) | set(test.columns))
    return train.reindex(columns=cols, fill_value=0.0), test.reindex(columns=cols, fill_value=0.0)


def evaluate_and_export_variants(
    outdir: Path,
    prefix: str,
    teacher_oof: np.ndarray,
    teacher_test: np.ndarray,
    ctx: dict,
    extra_variants: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    y = ctx["y"]
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    v173_oof = ctx["v173_oof"]
    v173_test = ctx["v173_test"]
    weights = context_weights(rows, test_rows)
    variants = {
        f"{prefix}_raw": (teacher_oof, teacher_test),
        f"{prefix}_v173blend_w0p15": (blend_probabilities(ctx["v173_prob_oof"], teacher_oof, 0.15), blend_probabilities(ctx["v173_prob_test"], teacher_test, 0.15)),
        f"{prefix}_v173blend_w0p30": (blend_probabilities(ctx["v173_prob_oof"], teacher_oof, 0.30), blend_probabilities(ctx["v173_prob_test"], teacher_test, 0.30)),
        f"{prefix}_v173blend_w0p50": (blend_probabilities(ctx["v173_prob_oof"], teacher_oof, 0.50), blend_probabilities(ctx["v173_prob_test"], teacher_test, 0.50)),
    }
    if extra_variants:
        variants.update(extra_variants)
    records = [evaluate_action("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    outdir.mkdir(exist_ok=True)
    for name, (prob_oof, prob_test) in variants.items():
        pred = normalize_probability_rows(prob_oof).argmax(axis=1).astype(int)
        test_pred = normalize_probability_rows(prob_test).argmax(axis=1).astype(int)
        rec = evaluate_action(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(outdir / f"{name}_oof_action_prob.npy", normalize_probability_rows(prob_oof))
        np.save(outdir / f"{name}_test_action_prob.npy", normalize_probability_rows(prob_test))
        generated.append(write_submission(outdir, f"submission_{name}__pv188cap5__sr121.csv", test_pred, ctx["point"], ctx["server"]))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"], ascending=[False, False, False])
    return search, generated


def report_json(outdir: Path, name: str, search: pd.DataFrame, generated: list[dict]) -> None:
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max()) if (search["candidate"].ne("v173_anchor")).any() else 0.0
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    (outdir / f"{name}_report.json").write_text(
        json.dumps({"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(outdir)}, indent=2))
