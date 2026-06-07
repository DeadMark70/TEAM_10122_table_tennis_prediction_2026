"""Retest point residual specialists against literal V188/V261 base.

This suite consumes V305A literal V188 artifacts and tests conservative point
specialist families. It is local-only and writes no upload/selected copies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import POINT_CLASSES
from analysis_v261_action_conditioned_point_residual import (
    EXPECTED_COLUMNS,
    add_foldsafe_proxy_columns,
    build_frames,
    distribution,
    normalize_rows_safe,
    numeric_feature_columns,
    train_oof_prob,
)
from analysis_v305_rebuild_v261_from_literal_v188 import (
    ARTIFACT_DIR,
    V300_SUBMISSION,
    apply_probability_residual,
    align_train_to_literal_meta,
    cap_token,
    load_artifacts,
    load_submission,
    point_column,
    point0_stats,
    select_top_margin_changes,
)


OUTDIR = Path("v305_point_residual_retest_suite")
CAPS = [0.0025, 0.005, 0.01, 0.02]


def classify_candidate(delta: float, churn: float, point0_add: int) -> str:
    if float(delta) >= 0.0015 and float(churn) <= 0.03 and int(point0_add) <= 2:
        return "REVIEW"
    return "DO_NOT_UPLOAD"


def class_f1_rows(y: np.ndarray, pred: np.ndarray, candidate: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cls in POINT_CLASSES:
        score = f1_score((y == cls).astype(int), (pred == cls).astype(int), zero_division=0)
        rows.append({"candidate": candidate, "point_class": int(cls), "f1": float(score)})
    return rows


def build_prior_probs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    key_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    global_counts = np.bincount(y, minlength=10).astype(float) + 1.0
    global_prob = global_counts / global_counts.sum()
    oof = np.zeros((len(train_df), 10), dtype=float)
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(fold).to_numpy()
        fit = train_df.loc[~valid, key_cols].copy()
        fit["target"] = y[~valid]
        table = fit.groupby(key_cols + ["target"]).size().unstack(fill_value=0)
        for cls in range(10):
            if cls not in table.columns:
                table[cls] = 0
        table = table[list(range(10))].astype(float) + 1.0
        table = table.div(table.sum(axis=1), axis=0)
        lookup = train_df.loc[valid, key_cols].merge(table.reset_index(), on=key_cols, how="left")
        arr = lookup[list(range(10))].to_numpy(dtype=float).copy()
        missing = ~np.isfinite(arr).all(axis=1)
        arr[missing] = global_prob
        oof[valid] = arr

    fit_all = train_df[key_cols].copy()
    fit_all["target"] = y
    table = fit_all.groupby(key_cols + ["target"]).size().unstack(fill_value=0)
    for cls in range(10):
        if cls not in table.columns:
            table[cls] = 0
    table = table[list(range(10))].astype(float) + 1.0
    table = table.div(table.sum(axis=1), axis=0)
    lookup_test = test_df[key_cols].merge(table.reset_index(), on=key_cols, how="left")
    test_arr = lookup_test[list(range(10))].to_numpy(dtype=float).copy()
    missing = ~np.isfinite(test_arr).all(axis=1)
    test_arr[missing] = global_prob
    return normalize_rows_safe(oof), normalize_rows_safe(test_arr)


def gated_residual(
    base: np.ndarray,
    prob: np.ndarray,
    cap: float,
    gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = normalize_rows_safe(prob)
    top = p.argmax(axis=1).astype(int)
    margin = p[np.arange(len(p)), top] - p[np.arange(len(p)), np.clip(base, 0, p.shape[1] - 1)]
    changed = select_top_margin_changes(base, top, np.where(gate, margin, -np.inf), cap)
    out = np.asarray(base, dtype=int).copy()
    out[changed] = top[changed]
    return out, changed, margin


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[EXPECTED_COLUMNS]
    if len(out) != 1845:
        raise ValueError(f"{name} has {len(out)} rows")
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    if not ARTIFACT_DIR.exists():
        raise FileNotFoundError("Run analysis_v305_export_literal_v188_point_artifact.py first.")
    artifacts = load_artifacts()
    train_df, test_df, _ = build_frames()
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns(train_df, test_df)
    train_df = align_train_to_literal_meta(train_df, artifacts["meta"])
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0

    y = train_df["next_pointId"].astype(int).to_numpy()
    features = [c for c in numeric_feature_columns(train_df, include_proxy=True) if c in test_df]
    model_oof, model_test, model_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        features,
        seed=30540,
        n_estimators=280,
        min_samples_leaf=3,
    )
    key_cols = ["lag0_action_family", "lag0_point_depth", "v261_action_family"]
    prior_oof, prior_test = build_prior_probs(train_df, test_df, y, key_cols)
    mix_oof = normalize_rows_safe(0.70 * model_oof + 0.30 * prior_oof)
    mix_test = normalize_rows_safe(0.70 * model_test + 0.30 * prior_test)

    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    if len(oof_base) != len(y):
        raise ValueError("Literal V188 OOF length mismatch")
    base_score = float(f1_score(y, oof_base, labels=POINT_CLASSES, average="macro", zero_division=0))
    anchor = load_submission(V300_SUBMISSION)

    model_top_oof = model_oof.argmax(axis=1).astype(int)
    prior_top_oof = prior_oof.argmax(axis=1).astype(int)
    model_top_test = model_test.argmax(axis=1).astype(int)
    prior_top_test = prior_test.argmax(axis=1).astype(int)
    mix_top_oof = mix_oof.argmax(axis=1).astype(int)
    mix_top_test = mix_test.argmax(axis=1).astype(int)

    families: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
        (
            "rare134",
            mix_oof,
            mix_test,
            np.isin(mix_top_oof, [1, 3, 4]) & (mix_top_oof == prior_top_oof),
            np.isin(mix_top_test, [1, 3, 4]) & (mix_top_test == prior_top_test),
        ),
        (
            "long789",
            mix_oof,
            mix_test,
            np.isin(mix_top_oof, [7, 8, 9]) & (mix_top_oof == prior_top_oof),
            np.isin(mix_top_test, [7, 8, 9]) & (mix_top_test == prior_top_test),
        ),
        (
            "agreement_only",
            mix_oof,
            mix_test,
            (model_top_oof == prior_top_oof) & (model_top_oof != oof_base),
            (model_top_test == prior_top_test) & (model_top_test != test_base),
        ),
        (
            "nonterminal_highconf",
            model_oof,
            model_test,
            (model_top_oof != 0) & (oof_base != 0) & (model_oof.max(axis=1) >= 0.34),
            (model_top_test != 0) & (test_base != 0) & (model_test.max(axis=1) >= 0.34),
        ),
        (
            "point0_conservative",
            model_oof,
            model_test,
            (model_top_oof == 0) & (model_oof[:, 0] >= 0.62),
            (model_top_test == 0) & (model_test[:, 0] >= 0.62),
        ),
    ]

    records: list[dict[str, object]] = []
    class_rows = class_f1_rows(y, oof_base, "v188_literal_cap5_base")
    submissions: list[dict[str, object]] = []
    for family, oof_prob, test_prob, oof_gate, test_gate in families:
        for cap in CAPS:
            oof_pred, oof_changed, _ = gated_residual(oof_base, oof_prob, cap, oof_gate)
            test_pred, test_changed, _ = gated_residual(test_base, test_prob, cap, test_gate)
            score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
            p0_add, p0_remove = point0_stats(test_base, test_pred)
            name = f"submission_v305_{family}_cap{cap_token(cap)}__v173action_v300server.csv"
            path = write_submission(anchor, test_pred, name)
            dec = classify_candidate(score - base_score, float(np.mean(test_changed)), p0_add)
            rec = {
                "candidate": f"v305_{family}_cap{cap_token(cap)}",
                "family": family,
                "cap": cap,
                "point_macro_f1": score,
                "literal_delta": score - base_score,
                "test_churn": float(np.mean(test_changed)),
                "test_changed_rows": int(test_changed.sum()),
                "point0_additions": p0_add,
                "point0_removals": p0_remove,
                "test_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                "submission": name,
                "path": path,
                "decision": dec,
            }
            records.append(rec)
            submissions.append({"candidate": rec["candidate"], "path": path})
            class_rows.extend(class_f1_rows(y, oof_pred, rec["candidate"]))

    search = pd.DataFrame(records)
    search = search.sort_values(["decision", "literal_delta", "test_churn"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v305_point_retest_search.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTDIR / "v305_point_retest_class_f1.csv", index=False)
    review = search[search["decision"].eq("REVIEW")]
    best = search.sort_values(["literal_delta", "test_churn"], ascending=[False, True]).head(1).iloc[0].to_dict() if not search.empty else {}
    report = {
        "verdict": "HAS_REVIEW_CANDIDATE" if not review.empty else "NO_UPLOAD_WORTHY_CANDIDATE",
        "literal_v188_cap5_point_macro_f1": base_score,
        "best_candidate": best,
        "review_candidates": review.head(5).to_dict(orient="records"),
        "submissions": submissions,
        "folds": proxy_folds + [{"stage": "specialist_model", **r} for r in model_folds],
        "notes": [
            "Retests point specialists against literal V188 cap5 OOF labels.",
            "No point0 additions are allowed past the decision gate except point0-specific candidates.",
            "Outputs are local-only and use V300 server packaging for current-best comparability.",
        ],
    }
    (OUTDIR / "v305_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v305_report.md").write_text(
        "# V305 Point Residual Retest Suite\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Literal V188 cap5 OOF point Macro-F1: `{base_score:.6f}`\n"
        f"- Best candidate: `{best.get('candidate', 'none')}`\n"
        f"- Best delta: `{float(best.get('literal_delta', 0.0)):.6f}`\n"
        f"- Decision: `{best.get('decision', 'none')}`\n\n"
        "## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "best": best.get("candidate")}, indent=2))


if __name__ == "__main__":
    main()
