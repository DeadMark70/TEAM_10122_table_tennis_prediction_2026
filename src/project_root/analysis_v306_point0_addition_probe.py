"""V306 point0-addition focused validation from V305 literal artifacts.

This probe isolates residual moves where the literal V188 cap5 point prediction
is nonzero and the V261-style residual model prefers point 0. Outputs stay
local under v306_point0_addition_probe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column


OUTDIR = Path("v306_point0_addition_probe")
ARTIFACT_DIR = Path("v305_literal_v188_point_artifact")
V300_SUBMISSION = Path("v300_clean_server_blend_recycler/submission_v300_best_safe_repack__v173action_v261point_server.csv")
ROW_BUDGETS = [4, 9, 14, 18]
CAPS = [0.0025, 0.005, 0.0075, 0.01]
CURRENT_BEST_PL = 0.3576975


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    selector_type: str
    cap: float
    test_budget: int
    oof_budget: int


def cap_token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def select_point0_additions(base: np.ndarray, prob: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the top positive-margin nonzero->0 changes up to an exact budget."""
    base = np.asarray(base, dtype=int)
    p = normalize_rows_safe(prob)
    if p.ndim != 2 or len(base) != len(p):
        raise ValueError("base and prob must have matching row counts")
    if p.shape[1] <= 0:
        raise ValueError("prob must include point0 probability in column 0")
    clipped_base = np.clip(base, 0, p.shape[1] - 1)
    margin = p[:, 0] - p[np.arange(len(p)), clipped_base]
    eligible = (base != 0) & np.isfinite(margin) & (margin > 0)
    selected = np.zeros(len(base), dtype=bool)
    if budget <= 0 or not eligible.any():
        return selected, margin
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-margin[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected, margin


def apply_point0_additions(base: np.ndarray, prob: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected, margin = select_point0_additions(base, prob, budget)
    pred = np.asarray(base, dtype=int).copy()
    pred[selected] = 0
    return pred, selected, margin


def decision_label(literal_oof_delta: float, test_changed_rows: int) -> str:
    if float(literal_oof_delta) >= 0.0015 and int(test_changed_rows) <= 18:
        return "REVIEW_P0"
    return "DO_NOT_UPLOAD"


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission: {path}")
    sub = pd.read_csv(path)
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns {list(sub.columns)} != {EXPECTED_COLUMNS}")
    return sub


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[EXPECTED_COLUMNS]
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


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
        raise FileNotFoundError("Missing V305 literal artifacts: " + ", ".join(missing))
    return {
        "v188_oof_prob": np.load(required["v188_oof_prob"]),
        "v188_test_prob": np.load(required["v188_test_prob"]),
        "cap5_oof": pd.read_csv(required["cap5_oof"]),
        "cap5_test": pd.read_csv(required["cap5_test"]),
        "meta": pd.read_csv(required["meta"]),
    }


def build_v261_literal_probabilities(artifacts: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
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
    folds = proxy_folds + [{"stage": "action_conditioned_point", **r} for r in point_folds]
    return y, model_oof_prob, model_test_prob, folds


def candidate_specs(oof_rows: int, test_rows: int) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for budget in ROW_BUDGETS:
        cap = budget / test_rows
        specs.append(
            CandidateSpec(
                name=f"v306_p0_budget{budget}",
                selector_type="row_budget",
                cap=cap,
                test_budget=budget,
                oof_budget=int(np.floor(oof_rows * cap)),
            )
        )
    for cap in CAPS:
        specs.append(
            CandidateSpec(
                name=f"v306_p0_cap{cap_token(cap)}",
                selector_type="cap",
                cap=cap,
                test_budget=int(np.floor(test_rows * cap)),
                oof_budget=int(np.floor(oof_rows * cap)),
            )
        )
    return specs


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    artifacts = load_artifacts()
    y, model_oof_prob, model_test_prob, folds = build_v261_literal_probabilities(artifacts)

    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    if len(oof_base) != len(y):
        raise ValueError(f"OOF base length {len(oof_base)} != y length {len(y)}")

    v188_prior_oof = normalize_rows_safe(artifacts["v188_oof_prob"])
    v188_prior_test = normalize_rows_safe(artifacts["v188_test_prob"])
    v188_prior_oof_p0_margin = v188_prior_oof[:, 0] - v188_prior_oof[np.arange(len(oof_base)), np.clip(oof_base, 0, v188_prior_oof.shape[1] - 1)]
    v188_prior_test_p0_margin = v188_prior_test[:, 0] - v188_prior_test[np.arange(len(test_base)), np.clip(test_base, 0, v188_prior_test.shape[1] - 1)]

    base_score = float(f1_score(y, oof_base, labels=POINT_CLASSES, average="macro", zero_division=0))
    v300_anchor = load_submission(V300_SUBMISSION)
    current_best_point = v300_anchor["pointId"].astype(int).to_numpy()
    records: list[dict[str, object]] = [
        {
            "candidate": "v188_literal_cap5_base",
            "selector_type": "baseline",
            "cap": 0.0,
            "oof_budget": 0,
            "test_budget": 0,
            "point_macro_f1": base_score,
            "literal_oof_delta": 0.0,
            "test_changed_rows": 0,
            "point0_additions": 0,
            "point0_removals": 0,
            "test_churn_vs_v188_cap5": 0.0,
            "test_churn_vs_current_best_v300": float(np.mean(test_base != current_best_point)),
            "decision": "BASELINE",
            "server_source": "v300",
        }
    ]
    submissions: list[dict[str, object]] = []

    for spec in candidate_specs(len(oof_base), len(test_base)):
        oof_pred, oof_changed, oof_margin = apply_point0_additions(oof_base, model_oof_prob, spec.oof_budget)
        test_pred, test_changed, test_margin = apply_point0_additions(test_base, model_test_prob, spec.test_budget)
        score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
        delta = score - base_score
        changed_rows = int(test_changed.sum())
        dec = decision_label(delta, changed_rows)
        name = f"submission_{spec.name}__v173action_v300server.csv"
        path = write_submission(v300_anchor, test_pred, name)
        submissions.append({"candidate": spec.name, "submission": name, "path": path})
        records.append(
            {
                "candidate": spec.name,
                "selector_type": spec.selector_type,
                "cap": spec.cap,
                "oof_budget": spec.oof_budget,
                "test_budget": spec.test_budget,
                "point_macro_f1": score,
                "literal_oof_delta": delta,
                "test_changed_rows": changed_rows,
                "oof_changed_rows": int(oof_changed.sum()),
                "point0_additions": changed_rows,
                "point0_removals": 0,
                "test_churn_vs_v188_cap5": float(np.mean(test_changed)),
                "test_churn_vs_current_best_v300": float(np.mean(test_pred != current_best_point)),
                "model_p0_margin_min_changed": float(test_margin[test_changed].min()) if changed_rows else 0.0,
                "model_p0_margin_mean_changed": float(test_margin[test_changed].mean()) if changed_rows else 0.0,
                "v188_prior_p0_margin_mean_changed": float(v188_prior_test_p0_margin[test_changed].mean()) if changed_rows else 0.0,
                "oof_v188_prior_p0_margin_mean_changed": float(v188_prior_oof_p0_margin[oof_changed].mean()) if int(oof_changed.sum()) else 0.0,
                "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                "server_source": "v300",
                "submission": name,
                "path": path,
                "decision": dec,
            }
        )

    search = pd.DataFrame(records)
    search = search.sort_values(
        ["decision", "literal_oof_delta", "test_changed_rows"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    search.to_csv(OUTDIR / "v306_point0_search.csv", index=False)

    candidates = search[search["candidate"].astype(str).str.startswith("v306_p0_")]
    best = candidates.sort_values(["literal_oof_delta", "test_changed_rows"], ascending=[False, True]).head(1)
    best_dict = best.iloc[0].to_dict() if not best.empty else {}
    review = candidates[candidates["decision"].eq("REVIEW_P0")]
    report = {
        "verdict": "HAS_REVIEW_P0_CANDIDATE" if not review.empty else "NO_UPLOAD_WORTHY_CANDIDATE",
        "upload_recommendation": "review_top_v306_point0_candidate_before_upload" if not review.empty else "keep_current_v300_best",
        "current_clean_best": V300_SUBMISSION.name,
        "current_clean_best_pl": CURRENT_BEST_PL,
        "v188_literal_cap5_point_macro_f1": base_score,
        "best_candidate": best_dict,
        "top_review_candidates": review.head(4).to_dict(orient="records"),
        "submissions": submissions,
        "folds": folds,
        "notes": [
            "Only nonzero base point predictions are eligible, and selected residuals always add point0.",
            "Candidates are packaged with V173 action unchanged and V300 server only.",
            "Decision gate is REVIEW_P0 only when literal OOF delta >= 0.0015 and test_changed_rows <= 18.",
            "Outputs are local-only under v306_point0_addition_probe.",
        ],
    }
    (OUTDIR / "v306_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v306_report.md").write_text(
        "# V306 Point0 Addition Probe\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n"
        f"- Current clean best: `{V300_SUBMISSION.name}` PL `{CURRENT_BEST_PL:.7f}`\n"
        f"- Baseline literal cap5 point Macro-F1: `{base_score:.6f}`\n"
        f"- Best candidate: `{best_dict.get('candidate', 'none')}`\n"
        f"- Best literal OOF delta: `{float(best_dict.get('literal_oof_delta', 0.0)):.6f}`\n"
        f"- Best test changed rows: `{int(best_dict.get('test_changed_rows', 0))}`\n"
        f"- Best decision: `{best_dict.get('decision', 'none')}`\n\n"
        "## Review Candidates\n\n"
        + (
            "\n".join(
                f"- `{r.get('candidate')}` delta `{float(r.get('literal_oof_delta', 0.0)):.6f}` rows `{int(r.get('test_changed_rows', 0))}`"
                for r in report["top_review_candidates"]
            )
            if report["top_review_candidates"]
            else "- None"
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "best": best_dict.get("candidate")}, indent=2))


if __name__ == "__main__":
    main()
