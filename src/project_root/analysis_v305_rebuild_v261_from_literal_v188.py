"""Rebuild V261-style point residuals on literal V188 OOF artifacts.

This script consumes V305A artifacts. It keeps the action fixed at V173 and
rebuilds point residual candidates over literal V188 cap5 labels, avoiding the
proxy-base issue in the original V261 run.

Outputs are local-only under v305_rebuild_v261_from_literal_v188. This script
does not copy files to upload_candidates_20260519 or submissions/selected.
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


OUTDIR = Path("v305_rebuild_v261_from_literal_v188")
ARTIFACT_DIR = Path("v305_literal_v188_point_artifact")
V188_R121_SUBMISSION = Path("v188_point_intent_gru/submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv")
V300_SUBMISSION = Path("v300_clean_server_blend_recycler/submission_v300_best_safe_repack__v173action_v261point_server.csv")
CAPS = [0.005, 0.01, 0.015, 0.02, 0.03]


def cap_token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def select_top_margin_changes(base: np.ndarray, cand: np.ndarray, margin: np.ndarray, cap: float) -> np.ndarray:
    base = np.asarray(base, dtype=int)
    cand = np.asarray(cand, dtype=int)
    margin = np.asarray(margin, dtype=float)
    if not (len(base) == len(cand) == len(margin)):
        raise ValueError("base, cand, and margin must have the same length")
    eligible = (cand != base) & np.isfinite(margin) & (margin > 0)
    budget = int(np.floor(len(base) * float(cap)))
    out = np.zeros(len(base), dtype=bool)
    if budget <= 0 or not eligible.any():
        return out
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-margin[idx])]
    out[order[: min(budget, len(order))]] = True
    return out


def apply_probability_residual(base_labels: np.ndarray, prob: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    p = normalize_rows_safe(prob)
    top = p.argmax(axis=1).astype(int)
    base_prob = p[np.arange(len(p)), np.clip(base, 0, p.shape[1] - 1)]
    margin = p[np.arange(len(p)), top] - base_prob
    changed = select_top_margin_changes(base, top, margin, cap)
    out = base.copy()
    out[changed] = top[changed]
    return out, changed, margin


def point0_stats(base: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    base = np.asarray(base, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return int(np.sum((base != 0) & (pred == 0))), int(np.sum((base == 0) & (pred != 0)))


def load_artifacts() -> dict[str, object]:
    required = {
        "v188_oof_prob": ARTIFACT_DIR / "v305_v188_r186_w005_oof_proba.npy",
        "v188_test_prob": ARTIFACT_DIR / "v305_v188_r186_w005_test_proba.npy",
        "cap5_oof": ARTIFACT_DIR / "v305_v188_cap5_oof_pred.csv",
        "cap5_test": ARTIFACT_DIR / "v305_v188_cap5_test_pred.csv",
        "meta": ARTIFACT_DIR / "v305_v188_oof_meta.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run analysis_v305_export_literal_v188_point_artifact.py first. Missing: " + ", ".join(missing))
    return {
        "v188_oof_prob": np.load(required["v188_oof_prob"]),
        "v188_test_prob": np.load(required["v188_test_prob"]),
        "cap5_oof": pd.read_csv(required["cap5_oof"]),
        "cap5_test": pd.read_csv(required["cap5_test"]),
        "meta": pd.read_csv(required["meta"]),
    }


def align_train_to_literal_meta(train_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    key = ["rally_uid", "prefix_len", "next_actionId", "next_pointId"]
    missing = [c for c in key if c not in train_df.columns or c not in meta.columns]
    if missing:
        raise KeyError(f"Cannot align literal meta; missing columns: {missing}")
    indexed = train_df.reset_index(drop=True).copy()
    indexed["_v305_train_idx"] = np.arange(len(indexed))
    merged = meta[key].merge(indexed[key + ["_v305_train_idx"]], on=key, how="left", validate="one_to_one")
    if merged["_v305_train_idx"].isna().any():
        raise ValueError("Literal V188 meta rows did not align to generated train prefix table.")
    order = merged["_v305_train_idx"].astype(int).to_numpy()
    out = indexed.iloc[order].drop(columns=["_v305_train_idx"]).reset_index(drop=True)
    if len(out) != len(meta):
        raise ValueError(f"Aligned rows {len(out)} != meta rows {len(meta)}")
    return out


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission: {path}")
    sub = pd.read_csv(path)
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns {list(sub.columns)} != {EXPECTED_COLUMNS}")
    if len(sub) != 1845:
        raise ValueError(f"{path} row count {len(sub)} != 1845")
    return sub


def point_column(frame: pd.DataFrame) -> str:
    for col in ("pointId", "cap0p05_point_pred", "point_pred"):
        if col in frame.columns:
            return col
    raise KeyError(f"No point prediction column found. Columns: {list(frame.columns)}")


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[EXPECTED_COLUMNS]
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def class_f1_rows(y: np.ndarray, pred: np.ndarray, candidate: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cls in POINT_CLASSES:
        score = f1_score((y == cls).astype(int), (pred == cls).astype(int), zero_division=0)
        rows.append({"candidate": candidate, "point_class": int(cls), "f1": float(score)})
    return rows


def decision(delta_vs_v188: float, churn: float, p0_add: int) -> str:
    if delta_vs_v188 >= 0.0015 and churn <= 0.03 and p0_add <= 2:
        return "REVIEW"
    return "DO_NOT_UPLOAD"


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    artifacts = load_artifacts()
    train_df, test_df, _ = build_frames()
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns(train_df, test_df)
    train_df = align_train_to_literal_meta(train_df, artifacts["meta"])
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0

    point_features = numeric_feature_columns(train_df, include_proxy=True)
    point_features = [c for c in point_features if c in test_df]
    y = train_df["next_pointId"].astype(int).to_numpy()

    model_oof_prob, model_test_prob, point_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        point_features,
        seed=30510,
        n_estimators=260,
        min_samples_leaf=4,
    )

    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    if len(oof_base) != len(y):
        raise ValueError(f"OOF base length {len(oof_base)} != y length {len(y)}")

    v188_score = float(f1_score(y, oof_base, labels=POINT_CLASSES, average="macro", zero_division=0))
    raw_pred = model_oof_prob.argmax(axis=1).astype(int)
    raw_score = float(f1_score(y, raw_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    r121_anchor = load_submission(V188_R121_SUBMISSION)
    v300_anchor = load_submission(V300_SUBMISSION)
    current_best_point = v300_anchor["pointId"].astype(int).to_numpy()

    records: list[dict[str, object]] = [
        {
            "candidate": "v188_literal_cap5_base",
            "cap": 0.0,
            "server_source": "r121",
            "point_macro_f1": v188_score,
            "delta_vs_v188_cap5": 0.0,
            "raw_model_score": raw_score,
            "test_churn_vs_v188_cap5": 0.0,
            "test_churn_vs_current_best_v300": float(np.mean(test_base != current_best_point)),
            "test_changed_rows": 0,
            "point0_additions": 0,
            "point0_removals": 0,
            "test_point_distribution": json.dumps(distribution(test_base), sort_keys=True),
            "risk_tier": "baseline",
            "decision": "BASELINE",
        }
    ]
    class_rows = class_f1_rows(y, oof_base, "v188_literal_cap5_base")
    submissions: list[dict[str, object]] = []

    for cap in CAPS:
        oof_pred, oof_changed, _ = apply_probability_residual(oof_base, model_oof_prob, cap)
        test_pred, test_changed, _ = apply_probability_residual(test_base, model_test_prob, cap)
        score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
        p0_add, p0_remove = point0_stats(test_base, test_pred)
        tier = "safe" if cap <= 0.01 else "normal" if cap <= 0.02 else "probe"
        dec = decision(score - v188_score, float(np.mean(test_changed)), p0_add)
        for server_source, anchor in [("r121", r121_anchor), ("v300", v300_anchor)]:
            name = f"submission_v305_v261_literal_cap{cap_token(cap)}__v173action_{server_source}server.csv"
            path = write_submission(anchor, test_pred, name)
            submissions.append({"candidate": name, "path": path, "server_source": server_source})
            records.append(
                {
                    "candidate": f"v305_v261_literal_cap{cap_token(cap)}",
                    "cap": cap,
                    "server_source": server_source,
                    "point_macro_f1": score,
                    "delta_vs_v188_cap5": score - v188_score,
                    "raw_model_score": raw_score,
                    "test_churn_vs_v188_cap5": float(np.mean(test_changed)),
                    "test_churn_vs_current_best_v300": float(np.mean(test_pred != current_best_point)),
                    "test_changed_rows": int(test_changed.sum()),
                    "point0_additions": p0_add,
                    "point0_removals": p0_remove,
                    "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                    "submission": name,
                    "path": path,
                    "risk_tier": tier,
                    "decision": dec,
                }
            )
        class_rows.extend(class_f1_rows(y, oof_pred, f"v305_v261_literal_cap{cap_token(cap)}"))

    search = pd.DataFrame(records)
    search = search.sort_values(["decision", "delta_vs_v188_cap5", "test_churn_vs_v188_cap5"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v305_v261_literal_search.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTDIR / "v305_v261_literal_class_f1.csv", index=False)

    candidate_rows = search[search["candidate"].astype(str).str.startswith("v305_v261_literal")]
    best = candidate_rows.sort_values(["delta_vs_v188_cap5", "test_churn_vs_v188_cap5"], ascending=[False, True]).head(1)
    best_dict = best.iloc[0].to_dict() if not best.empty else {}
    report = {
        "verdict": "HAS_REVIEW_CANDIDATE" if (candidate_rows["decision"].eq("REVIEW").any() if not candidate_rows.empty else False) else "NO_UPLOAD_WORTHY_CANDIDATE",
        "v188_literal_cap5_point_macro_f1": v188_score,
        "raw_action_conditioned_model_point_macro_f1": raw_score,
        "best_candidate": best_dict,
        "submissions": submissions,
        "folds": proxy_folds + [{"stage": "action_conditioned_point", **r} for r in point_folds],
        "notes": [
            "Uses literal V188 cap5 OOF/test labels from V305A as the residual base.",
            "Outputs are local-only under v305_rebuild_v261_from_literal_v188.",
            "No TTMATCH, no old-server, no upload folder copies.",
            "R121 variants preserve clean comparability; V300 variants compare against current public best packaging.",
        ],
    }
    (OUTDIR / "v305_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v305_report.md").write_text(
        "# V305 Rebuild V261 From Literal V188\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- V188 literal cap5 OOF point Macro-F1: `{v188_score:.6f}`\n"
        f"- Raw action-conditioned model OOF point Macro-F1: `{raw_score:.6f}`\n"
        f"- Best candidate: `{best_dict.get('candidate', 'none')}`\n"
        f"- Best delta vs V188 cap5: `{float(best_dict.get('delta_vs_v188_cap5', 0.0)):.6f}`\n"
        f"- Decision: `{best_dict.get('decision', 'none')}`\n\n"
        "## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "best": best_dict.get("candidate")}, indent=2))


if __name__ == "__main__":
    main()
