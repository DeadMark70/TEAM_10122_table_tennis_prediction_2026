"""V313 joint point0/action-terminal consistency research.

This keeps action labels unchanged, but only allows point0 additions through
when the action probability source is terminal-compatible and not strongly
nonterminal. Outputs are local-only under v313_joint_terminal_consistency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from analysis_v261_action_conditioned_point_residual import EXPECTED_COLUMNS, distribution, normalize_rows_safe
from analysis_v305_rebuild_v261_from_literal_v188 import point_column
from analysis_v306_point0_addition_probe import (
    V300_SUBMISSION,
    build_v261_literal_probabilities,
    cap_token,
    load_artifacts,
    load_submission,
)
from analysis_v307_point0_dose_extension import sanitize_json


OUTDIR = Path("v313_joint_terminal_consistency")
V306_DIR = Path("v306_point0_addition_probe")
V307_DIR = Path("v307_point0_dose_extension")
V306_SEARCH = V306_DIR / "v306_point0_search.csv"
V307_SEARCH = V307_DIR / "v307_point0_dose_search.csv"
V307_REPORT = V307_DIR / "v307_report.json"
ACTION_OOF_PROB = Path("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_oof_action_prob.npy")
ACTION_TEST_PROB = Path("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_test_action_prob.npy")
V173_ACTION_SUBMISSION = Path(
    "v173_external_curriculum_pretrain/submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
)
ROW_BUDGETS = [18, 24, 30, 36]
V307_BUDGET24_DELTA = 0.00469246629968606
V307_BUDGET24_ROWS = 24
TERMINAL_PROB_MIN = 0.12
TERMINAL_MARGIN_FLOOR = -0.10
STRONG_NONTERMINAL_PROB_MAX = 0.05
STRONG_NONTERMINAL_TOP_MIN = 0.30
STRONG_NONTERMINAL_MARGIN_MIN = 0.20


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    requested_budget: int
    cap: float
    oof_budget: int


def validate_submission_schema(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"submission columns {list(frame.columns)} != {EXPECTED_COLUMNS}")
    return frame.loc[:, EXPECTED_COLUMNS].copy()


def action_terminal_masks(action_prob: np.ndarray, base_action: np.ndarray) -> dict[str, np.ndarray]:
    prob = normalize_rows_safe(action_prob)
    base_action = np.asarray(base_action, dtype=int)
    if prob.ndim != 2 or len(prob) != len(base_action):
        raise ValueError("action_prob and base_action must have matching row counts")
    if prob.shape[1] <= 0:
        raise ValueError("action_prob must include action0 probability")

    clipped_action = np.clip(base_action, 0, prob.shape[1] - 1)
    base_prob = prob[np.arange(len(prob)), clipped_action]
    terminal_prob = prob[:, 0]
    nonterminal_top = prob[:, 1:].max(axis=1) if prob.shape[1] > 1 else np.zeros(len(prob))
    terminal_margin = terminal_prob - base_prob
    nonterminal_margin = nonterminal_top - terminal_prob
    terminal_compatible = (base_action == 0) | (terminal_prob >= TERMINAL_PROB_MIN) | (terminal_margin >= TERMINAL_MARGIN_FLOOR)
    strong_nonterminal = (
        (base_action != 0)
        & (terminal_prob <= STRONG_NONTERMINAL_PROB_MAX)
        & (nonterminal_top >= STRONG_NONTERMINAL_TOP_MIN)
        & (nonterminal_margin >= STRONG_NONTERMINAL_MARGIN_MIN)
    )
    return {
        "terminal_prob": terminal_prob,
        "base_action_prob": base_prob,
        "nonterminal_top_prob": nonterminal_top,
        "terminal_margin": terminal_margin,
        "nonterminal_margin": nonterminal_margin,
        "terminal_compatible": terminal_compatible,
        "strong_nonterminal": strong_nonterminal,
        "eligible_action_source": terminal_compatible & ~strong_nonterminal,
    }


def select_joint_point0_additions(
    base_point: np.ndarray,
    point_prob: np.ndarray,
    action_prob: np.ndarray,
    base_action: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    base_point = np.asarray(base_point, dtype=int)
    p_point = normalize_rows_safe(point_prob)
    if p_point.ndim != 2 or len(base_point) != len(p_point):
        raise ValueError("base_point and point_prob must have matching row counts")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    masks = action_terminal_masks(action_prob, base_action)
    clipped_point = np.clip(base_point, 0, p_point.shape[1] - 1)
    point0_margin = p_point[:, 0] - p_point[np.arange(len(p_point)), clipped_point]
    positive_margin = (base_point != 0) & np.isfinite(point0_margin) & (point0_margin > 0)
    eligible = positive_margin & masks["eligible_action_source"]
    selected = np.zeros(len(base_point), dtype=bool)
    if budget > 0 and eligible.any():
        idx = np.where(eligible)[0]
        order = idx[np.argsort(-point0_margin[idx], kind="mergesort")]
        selected[order[: min(int(budget), len(order))]] = True
    pred = base_point.copy()
    pred[selected] = 0
    audit = pd.DataFrame(
        {
            "row_id": np.arange(len(base_point), dtype=int),
            "base_point": base_point,
            "base_action": np.asarray(base_action, dtype=int),
            "point0_margin": point0_margin,
            "positive_point0_margin": positive_margin,
            "terminal_prob": masks["terminal_prob"],
            "terminal_margin": masks["terminal_margin"],
            "nonterminal_top_prob": masks["nonterminal_top_prob"],
            "nonterminal_margin": masks["nonterminal_margin"],
            "terminal_compatible": masks["terminal_compatible"],
            "strong_nonterminal": masks["strong_nonterminal"],
            "eligible_action_source": masks["eligible_action_source"],
            "selected": selected,
        }
    )
    return pred, selected, audit


def decision_label(
    literal_oof_delta: float,
    test_changed_rows: int,
    v307_budget24_delta: float = V307_BUDGET24_DELTA,
    v307_budget24_rows: int = V307_BUDGET24_ROWS,
) -> str:
    delta = float(literal_oof_delta)
    rows = int(test_changed_rows)
    if delta > float(v307_budget24_delta):
        return "REVIEW"
    if delta >= float(v307_budget24_delta) * 0.98 and rows < int(v307_budget24_rows):
        return "REVIEW"
    return "DO_NOT_UPLOAD"


def _candidate_specs(oof_rows: int, test_rows: int) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for budget in ROW_BUDGETS:
        cap = budget / test_rows
        specs.append(CandidateSpec(f"v313_joint_budget{budget}", budget, cap, int(np.floor(oof_rows * cap))))
    return specs


def _load_action_prob(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing action probability artifact: {path}")
    prob = normalize_rows_safe(np.load(path))
    if prob.shape != (expected_rows, len(ACTION_CLASSES)):
        raise ValueError(f"{path} shape {prob.shape} != {(expected_rows, len(ACTION_CLASSES))}")
    return prob


def _write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = validate_submission_schema(out)
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def _load_v307_budget24_reference() -> dict[str, float]:
    if not V307_SEARCH.exists():
        return {"delta": V307_BUDGET24_DELTA, "rows": float(V307_BUDGET24_ROWS)}
    search = pd.read_csv(V307_SEARCH)
    row = search[search["candidate"].eq("v307_p0_budget24")]
    if row.empty:
        return {"delta": V307_BUDGET24_DELTA, "rows": float(V307_BUDGET24_ROWS)}
    return {"delta": float(row.iloc[0]["literal_oof_delta"]), "rows": float(row.iloc[0]["test_changed_rows"])}


def _write_report_md(report: dict[str, Any]) -> None:
    best = report["best_candidate"]
    top_lines = [
        f"- `{row['candidate']}` delta `{float(row['literal_oof_delta']):.6f}` rows `{int(row['test_changed_rows'])}` "
        f"vetoed `{int(row['test_vetoed_strong_nonterminal'])}` decision `{row['decision']}`"
        for row in report["top_candidates"]
    ]
    (OUTDIR / "v313_report.md").write_text(
        "# V313 Joint Terminal Consistency\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n"
        f"- Best candidate: `{best.get('candidate', 'none')}`\n"
        f"- Best literal OOF delta: `{float(best.get('literal_oof_delta', 0.0)):.6f}`\n"
        f"- Best changed rows: `{int(best.get('test_changed_rows', 0))}`\n"
        f"- V307 budget24 delta: `{float(report['v307_budget24_delta']):.6f}`\n"
        f"- Joint consistency improves over pure point0 dose: `{report['joint_improves_over_pure_point0_dose']}`\n\n"
        "## Top Candidates\n\n"
        + ("\n".join(top_lines) if top_lines else "- None")
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    missing = [str(path) for path in [V306_SEARCH, V307_SEARCH, V307_REPORT] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V306/V307 artifacts: " + ", ".join(missing))

    artifacts = load_artifacts()
    y_point, model_oof_prob, model_test_prob, folds = build_v261_literal_probabilities(artifacts)
    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_base_point = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base_point = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    if len(oof_base_point) != len(y_point):
        raise ValueError(f"OOF base length {len(oof_base_point)} != y length {len(y_point)}")

    action_oof_prob = _load_action_prob(ACTION_OOF_PROB, len(oof_base_point))
    action_test_prob = _load_action_prob(ACTION_TEST_PROB, len(test_base_point))
    oof_base_action = action_oof_prob.argmax(axis=1).astype(int)
    v300_anchor = load_submission(V300_SUBMISSION)
    test_base_action = v300_anchor["actionId"].astype(int).to_numpy()
    if V173_ACTION_SUBMISSION.exists():
        v173_action = load_submission(V173_ACTION_SUBMISSION)["actionId"].astype(int).to_numpy()
        if len(v173_action) == len(test_base_action):
            test_base_action = v173_action

    base_score = float(f1_score(y_point, oof_base_point, labels=POINT_CLASSES, average="macro", zero_division=0))
    current_v300_point = v300_anchor["pointId"].astype(int).to_numpy()
    v306_search = pd.read_csv(V306_SEARCH)
    v307_search = pd.read_csv(V307_SEARCH)
    v307_ref = _load_v307_budget24_reference()

    records: list[dict[str, Any]] = [
        {
            "candidate": "v188_literal_cap5_base",
            "requested_budget": 0,
            "cap": 0.0,
            "oof_budget": 0,
            "point_macro_f1": base_score,
            "literal_oof_delta": 0.0,
            "test_changed_rows": 0,
            "oof_changed_rows": 0,
            "test_terminal_compatible_candidates": 0,
            "test_vetoed_strong_nonterminal": 0,
            "test_incompatible_action_source": 0,
            "test_churn_vs_current_best_v300": float(np.mean(test_base_point != current_v300_point)),
            "decision": "BASELINE",
            "server_source": "v300",
        }
    ]
    changed_rows: list[pd.DataFrame] = []
    submissions: list[dict[str, str]] = []

    for spec in _candidate_specs(len(oof_base_point), len(test_base_point)):
        oof_pred, oof_selected, oof_audit = select_joint_point0_additions(
            oof_base_point, model_oof_prob, action_oof_prob, oof_base_action, spec.oof_budget
        )
        test_pred, test_selected, test_audit = select_joint_point0_additions(
            test_base_point, model_test_prob, action_test_prob, test_base_action, spec.requested_budget
        )
        score = float(f1_score(y_point, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
        delta = score - base_score
        changed_count = int(test_selected.sum())
        decision = decision_label(delta, changed_count, v307_ref["delta"], int(v307_ref["rows"]))
        name = f"submission_{spec.name}__v173action_v300server.csv"
        path = _write_submission(v300_anchor, test_pred, name)
        submissions.append({"candidate": spec.name, "submission": name, "path": path})

        candidate_mask = test_audit["positive_point0_margin"].to_numpy()
        selected_audit = test_audit[test_audit["selected"]].copy()
        if not selected_audit.empty:
            selected_audit["candidate"] = spec.name
            selected_audit["rally_uid"] = v300_anchor.loc[selected_audit["row_id"], "rally_uid"].astype(int).to_numpy()
            selected_audit["old_pointId"] = test_base_point[selected_audit["row_id"].to_numpy()]
            selected_audit["new_pointId"] = 0
            selected_audit["actionId"] = v300_anchor.loc[selected_audit["row_id"], "actionId"].astype(int).to_numpy()
            selected_audit["serverGetPoint"] = v300_anchor.loc[selected_audit["row_id"], "serverGetPoint"].astype(int).to_numpy()
            changed_rows.append(selected_audit)

        pure_row = v307_search[v307_search["candidate"].eq(f"v307_p0_budget{spec.requested_budget}")]
        pure_delta = float(pure_row.iloc[0]["literal_oof_delta"]) if not pure_row.empty else np.nan
        records.append(
            {
                "candidate": spec.name,
                "requested_budget": spec.requested_budget,
                "cap": spec.cap,
                "oof_budget": spec.oof_budget,
                "point_macro_f1": score,
                "literal_oof_delta": delta,
                "delta_vs_v307_budget24": delta - float(v307_ref["delta"]),
                "delta_vs_pure_v307_same_budget": delta - pure_delta if np.isfinite(pure_delta) else np.nan,
                "test_changed_rows": changed_count,
                "oof_changed_rows": int(oof_selected.sum()),
                "point0_additions": changed_count,
                "point0_removals": 0,
                "test_terminal_compatible_candidates": int((candidate_mask & test_audit["terminal_compatible"].to_numpy()).sum()),
                "test_vetoed_strong_nonterminal": int((candidate_mask & test_audit["strong_nonterminal"].to_numpy()).sum()),
                "test_incompatible_action_source": int((candidate_mask & ~test_audit["eligible_action_source"].to_numpy()).sum()),
                "oof_vetoed_strong_nonterminal": int((oof_audit["positive_point0_margin"] & oof_audit["strong_nonterminal"]).sum()),
                "test_churn_vs_v188_cap5": float(np.mean(test_selected)),
                "test_churn_vs_current_best_v300": float(np.mean(test_pred != current_v300_point)),
                "model_p0_margin_min_changed": float(test_audit.loc[test_selected, "point0_margin"].min()) if changed_count else 0.0,
                "model_p0_margin_mean_changed": float(test_audit.loc[test_selected, "point0_margin"].mean()) if changed_count else 0.0,
                "terminal_prob_mean_changed": float(test_audit.loc[test_selected, "terminal_prob"].mean()) if changed_count else 0.0,
                "terminal_margin_mean_changed": float(test_audit.loc[test_selected, "terminal_margin"].mean()) if changed_count else 0.0,
                "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                "server_source": "v300",
                "action_policy": "unchanged",
                "submission": name,
                "path": path,
                "decision": decision,
            }
        )

    search = pd.DataFrame(records).sort_values(["literal_oof_delta", "test_changed_rows"], ascending=[False, True])
    search.to_csv(OUTDIR / "v313_joint_search.csv", index=False)
    if changed_rows:
        changed = pd.concat(changed_rows, ignore_index=True)
    else:
        changed = pd.DataFrame(
            columns=[
                "candidate",
                "row_id",
                "rally_uid",
                "old_pointId",
                "new_pointId",
                "actionId",
                "serverGetPoint",
                "point0_margin",
                "terminal_prob",
                "terminal_margin",
                "strong_nonterminal",
            ]
        )
    changed.to_csv(OUTDIR / "v313_changed_rows.csv", index=False)

    candidates = search[search["candidate"].astype(str).str.startswith("v313_joint_")]
    best = candidates.head(1).iloc[0].to_dict() if not candidates.empty else {}
    review = candidates[candidates["decision"].eq("REVIEW")]
    improves_over_pure = bool(
        best and float(best.get("literal_oof_delta", 0.0)) > float(v307_ref["delta"]) and int(best.get("test_changed_rows", 0)) <= 36
    )
    report = {
        "version": "V313",
        "verdict": "HAS_REVIEW_CANDIDATE" if not review.empty else "NO_UPLOAD_WORTHY_CANDIDATE",
        "upload_recommendation": "REVIEW" if not review.empty else "DO_NOT_UPLOAD",
        "packaging": "V173/V300 action unchanged with V300 server",
        "v188_literal_cap5_point_macro_f1": base_score,
        "v307_budget24_delta": float(v307_ref["delta"]),
        "v307_budget24_rows": int(v307_ref["rows"]),
        "v306_candidates_consumed": int(len(v306_search)),
        "v307_candidates_consumed": int(len(v307_search)),
        "best_candidate": best,
        "top_candidates": candidates.head(4).to_dict(orient="records"),
        "top_review_candidates": review.head(4).to_dict(orient="records"),
        "joint_improves_over_pure_point0_dose": improves_over_pure,
        "submissions": submissions,
        "folds": folds,
        "action_probability_source": {"oof": str(ACTION_OOF_PROB), "test": str(ACTION_TEST_PROB)},
        "thresholds": {
            "terminal_prob_min": TERMINAL_PROB_MIN,
            "terminal_margin_floor": TERMINAL_MARGIN_FLOOR,
            "strong_nonterminal_prob_max": STRONG_NONTERMINAL_PROB_MAX,
            "strong_nonterminal_top_min": STRONG_NONTERMINAL_TOP_MIN,
            "strong_nonterminal_margin_min": STRONG_NONTERMINAL_MARGIN_MIN,
        },
        "notes": [
            "Consumes V306/V307 point candidate tables and V305 literal OOF artifacts.",
            "Point0 additions are allowed only for terminal-compatible action sources.",
            "Strong nonterminal action-source rows are vetoed.",
            "Action edits are not forced; diagnostics only.",
            "Decision is REVIEW only when point delta beats V307 budget24 or has similar delta with fewer rows.",
        ],
    }
    (OUTDIR / "v313_report.json").write_text(json.dumps(sanitize_json(report), indent=2), encoding="utf-8")
    _write_report_md(report)
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "verdict": report["verdict"],
                "upload_recommendation": report["upload_recommendation"],
                "best": best.get("candidate"),
                "joint_improves_over_pure_point0_dose": improves_over_pure,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
