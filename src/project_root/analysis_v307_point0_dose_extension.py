"""V307 point0 dose extension around the public-positive V306 structure.

This extends the V306 nonzero->point0 residual ranking to a wider set of
row-count and probability-cap doses. It consumes the V306 outputs as the
public-positive reference and the V305 literal artifacts as the source point
structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import POINT_CLASSES
from analysis_v261_action_conditioned_point_residual import EXPECTED_COLUMNS, distribution, normalize_rows_safe
from analysis_v305_rebuild_v261_from_literal_v188 import point_column
from analysis_v306_point0_addition_probe import (
    V300_SUBMISSION,
    build_v261_literal_probabilities,
    cap_token,
    load_artifacts,
    load_submission,
    write_submission as write_v306_submission,
)


OUTDIR = Path("v307_point0_dose_extension")
V306_DIR = Path("v306_point0_addition_probe")
V306_SEARCH = V306_DIR / "v306_point0_search.csv"
V306_REPORT = V306_DIR / "v306_report.json"
V306_PUBLIC_BEST = V306_DIR / "submission_v306_p0_cap0p01__v173action_v300server.csv"
CURRENT_PUBLIC_BEST_PL = 0.3577905
ROW_BUDGETS = [6, 9, 12, 14, 16, 18, 20, 22, 24, 27, 30, 36]
CAPS = [0.006, 0.008, 0.010, 0.012, 0.015, 0.020]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    selector_type: str
    cap: float
    test_budget: int
    oof_budget: int


def select_exact_point0_additions(base: np.ndarray, prob: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly budget nonzero->0 positive-margin rows when available."""
    base = np.asarray(base, dtype=int)
    p = normalize_rows_safe(prob)
    if p.ndim != 2 or len(base) != len(p):
        raise ValueError("base and prob must have matching row counts")
    if p.shape[1] <= 0:
        raise ValueError("prob must include point0 probability in column 0")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    clipped_base = np.clip(base, 0, p.shape[1] - 1)
    margin = p[:, 0] - p[np.arange(len(p)), clipped_base]
    eligible = (base != 0) & np.isfinite(margin) & (margin > 0)
    selected = np.zeros(len(base), dtype=bool)
    if budget == 0:
        return selected, margin

    idx = np.where(eligible)[0]
    if len(idx) < budget:
        raise ValueError(f"requested budget {budget} but only {len(idx)} eligible point0 additions")
    order = idx[np.argsort(-margin[idx], kind="mergesort")]
    selected[order[: int(budget)]] = True
    return selected, margin


def apply_point0_additions(base: np.ndarray, prob: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected, margin = select_exact_point0_additions(base, prob, budget)
    pred = np.asarray(base, dtype=int).copy()
    pred[selected] = 0
    return pred, selected, margin


def decision_label(literal_oof_delta: float, test_changed_rows: int) -> str:
    delta = float(literal_oof_delta)
    rows = int(test_changed_rows)
    if delta >= 0.003 and rows <= 24:
        return "REVIEW_STRONG"
    if delta >= 0.004 and rows <= 36:
        return "REVIEW_EXPLORE"
    return "DO_NOT_UPLOAD"


def candidate_specs(oof_rows: int, test_rows: int) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for budget in ROW_BUDGETS:
        cap = budget / test_rows
        specs.append(
            CandidateSpec(
                name=f"v307_p0_budget{budget}",
                selector_type="row_budget",
                cap=cap,
                test_budget=budget,
                oof_budget=int(np.floor(oof_rows * cap)),
            )
        )
    for cap in CAPS:
        specs.append(
            CandidateSpec(
                name=f"v307_p0_cap{cap_token(cap)}",
                selector_type="cap",
                cap=cap,
                test_budget=int(np.floor(test_rows * cap)),
                oof_budget=int(np.floor(oof_rows * cap)),
            )
        )
    return specs


def load_v306_reference() -> dict[str, Any]:
    missing = [str(path) for path in [V306_SEARCH, V306_REPORT, V306_PUBLIC_BEST] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V306 artifacts: " + ", ".join(missing))
    return {
        "search": pd.read_csv(V306_SEARCH),
        "report": json.loads(V306_REPORT.read_text(encoding="utf-8")),
        "submission": load_submission(V306_PUBLIC_BEST),
    }


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    OUTDIR.mkdir(exist_ok=True)
    original_outdir = write_v306_submission.__globals__["OUTDIR"]
    write_v306_submission.__globals__["OUTDIR"] = OUTDIR
    try:
        return write_v306_submission(anchor, point, name)
    finally:
        write_v306_submission.__globals__["OUTDIR"] = original_outdir


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:
        return None
    return value


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    v306 = load_v306_reference()
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
    v188_prior_oof_p0_margin = v188_prior_oof[:, 0] - v188_prior_oof[
        np.arange(len(oof_base)), np.clip(oof_base, 0, v188_prior_oof.shape[1] - 1)
    ]
    v188_prior_test_p0_margin = v188_prior_test[:, 0] - v188_prior_test[
        np.arange(len(test_base)), np.clip(test_base, 0, v188_prior_test.shape[1] - 1)
    ]

    base_score = float(f1_score(y, oof_base, labels=POINT_CLASSES, average="macro", zero_division=0))
    v300_anchor = load_submission(V300_SUBMISSION)
    public_best_point = v306["submission"]["pointId"].astype(int).to_numpy()
    current_v300_point = v300_anchor["pointId"].astype(int).to_numpy()
    records: list[dict[str, Any]] = [
        {
            "candidate": "v188_literal_cap5_base",
            "selector_type": "baseline",
            "cap": 0.0,
            "oof_budget": 0,
            "test_budget": 0,
            "point_macro_f1": base_score,
            "literal_oof_delta": 0.0,
            "test_changed_rows": 0,
            "oof_changed_rows": 0,
            "point0_additions": 0,
            "point0_removals": 0,
            "test_churn_vs_v188_cap5": 0.0,
            "test_churn_vs_public_v306": float(np.mean(test_base != public_best_point)),
            "test_churn_vs_current_best_v300": float(np.mean(test_base != current_v300_point)),
            "decision": "BASELINE",
            "server_source": "v300",
        }
    ]
    submissions: list[dict[str, Any]] = []

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
                "test_churn_vs_public_v306": float(np.mean(test_pred != public_best_point)),
                "test_churn_vs_current_best_v300": float(np.mean(test_pred != current_v300_point)),
                "model_p0_margin_min_changed": float(test_margin[test_changed].min()) if changed_rows else 0.0,
                "model_p0_margin_mean_changed": float(test_margin[test_changed].mean()) if changed_rows else 0.0,
                "oof_model_p0_margin_mean_changed": float(oof_margin[oof_changed].mean()) if int(oof_changed.sum()) else 0.0,
                "v188_prior_p0_margin_mean_changed": float(v188_prior_test_p0_margin[test_changed].mean()) if changed_rows else 0.0,
                "oof_v188_prior_p0_margin_mean_changed": float(v188_prior_oof_p0_margin[oof_changed].mean()) if int(oof_changed.sum()) else 0.0,
                "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                "server_source": "v300",
                "submission": name,
                "path": path,
                "decision": dec,
            }
        )

    search = pd.DataFrame(records).sort_values(
        ["literal_oof_delta", "test_changed_rows"],
        ascending=[False, True],
    ).reset_index(drop=True)
    search.to_csv(OUTDIR / "v307_point0_dose_search.csv", index=False)

    candidates = search[search["candidate"].astype(str).str.startswith("v307_p0_")]
    top3 = candidates.head(3).to_dict(orient="records")
    review = candidates[candidates["decision"].isin(["REVIEW_STRONG", "REVIEW_EXPLORE"])]
    best_dict = top3[0] if top3 else {}
    v306_best = v306["report"].get("best_candidate", {})
    v306_cap_row = v306["search"][v306["search"]["candidate"].eq("v306_p0_cap0p01")]
    report = {
        "verdict": "HAS_REVIEW_CANDIDATE" if not review.empty else "NO_UPLOAD_WORTHY_CANDIDATE",
        "upload_recommendation": "review_v307_dose_candidates" if not review.empty else "keep_public_v306_best",
        "current_public_best": V306_PUBLIC_BEST.name,
        "current_public_best_pl": CURRENT_PUBLIC_BEST_PL,
        "packaging": "V173 action with V300 server only",
        "v188_literal_cap5_point_macro_f1": base_score,
        "v306_best_candidate": v306_best,
        "v306_cap0p01_literal_oof_delta": float(v306_cap_row.iloc[0]["literal_oof_delta"]) if not v306_cap_row.empty else None,
        "best_candidate": best_dict,
        "top3_candidates": top3,
        "top_review_candidates": review.head(6).to_dict(orient="records"),
        "submissions": submissions,
        "folds": folds,
        "notes": [
            "Only nonzero base point predictions are eligible, and selected residuals always add point0.",
            "V306 artifacts are consumed as the public-positive reference; V305 literal artifacts provide the base point structure.",
            "Row-budget candidates use exact test changed-row budgets.",
            "Decision gates: REVIEW_STRONG for delta >= 0.003 and rows <= 24; REVIEW_EXPLORE for delta >= 0.004 and rows <= 36.",
            "Outputs are local-only under v307_point0_dose_extension.",
        ],
    }
    (OUTDIR / "v307_report.json").write_text(json.dumps(sanitize_json(report), indent=2), encoding="utf-8")

    review_lines = [
        f"- `{r.get('candidate')}` delta `{float(r.get('literal_oof_delta', 0.0)):.6f}` rows `{int(r.get('test_changed_rows', 0))}` decision `{r.get('decision')}`"
        for r in top3
    ]
    (OUTDIR / "v307_report.md").write_text(
        "# V307 Point0 Dose Extension\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n"
        f"- Current public best: `{V306_PUBLIC_BEST.name}` PL `{CURRENT_PUBLIC_BEST_PL:.7f}`\n"
        f"- Packaging: `{report['packaging']}`\n"
        f"- Baseline literal cap5 point Macro-F1: `{base_score:.6f}`\n"
        f"- Best candidate: `{best_dict.get('candidate', 'none')}`\n"
        f"- Best literal OOF delta: `{float(best_dict.get('literal_oof_delta', 0.0)):.6f}`\n"
        f"- Best test changed rows: `{int(best_dict.get('test_changed_rows', 0))}`\n"
        f"- Best decision: `{best_dict.get('decision', 'none')}`\n\n"
        "## Top 3 Candidates\n\n"
        + ("\n".join(review_lines) if review_lines else "- None")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "best": best_dict.get("candidate")}, indent=2))


if __name__ == "__main__":
    main()
