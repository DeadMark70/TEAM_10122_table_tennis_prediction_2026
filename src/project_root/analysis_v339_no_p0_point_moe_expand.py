"""V339 no-p0 point MoE expansion.

This experiment keeps the V306/V338 action and server columns fixed and only
tests nonterminal point corrections. It reuses the V333 hierarchical point
heads and V337 selection/export conventions, while reporting churn against the
current V338 public-positive point anchor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import POINT_CLASSES
from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v339_no_p0_point_moe_expand"
BASE_ANCHOR_PATH = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
PUBLIC_ANCHOR_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
MIN_POINT_DELTA = 0.003


@dataclass(frozen=True)
class VariantSpec:
    name: str
    budget: int
    selector: str
    filename: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("v339_no_p0_hier_b18", 18, "no_p0_add", "submission_v339_no_p0_hier_b18__v173action_v300server.csv"),
    VariantSpec("v339_no_p0_hier_b24", 24, "no_p0_add", "submission_v339_no_p0_hier_b24__v173action_v300server.csv"),
    VariantSpec("v339_no_p0_hier_b30", 30, "no_p0_add", "submission_v339_no_p0_hier_b30__v173action_v300server.csv"),
    VariantSpec("v339_no_p0_hier_b36", 36, "no_p0_add", "submission_v339_no_p0_hier_b36__v173action_v300server.csv"),
    VariantSpec("v339_no_p0_hier_b48", 48, "no_p0_add", "submission_v339_no_p0_hier_b48__v173action_v300server.csv"),
    VariantSpec("v339_no_p0_margin_hi_b24", 24, "margin_hi", "submission_v339_no_p0_margin_hi_b24__v173action_v300server.csv"),
    VariantSpec("v339_no_p0_margin_hi_b36", 36, "margin_hi", "submission_v339_no_p0_margin_hi_b36__v173action_v300server.csv"),
    VariantSpec(
        "v339_no_p0_nonterminal_longside_b24",
        24,
        "nonterminal_longside",
        "submission_v339_no_p0_nonterminal_longside_b24__v173action_v300server.csv",
    ),
    VariantSpec(
        "v339_no_p0_nonterminal_longside_b36",
        36,
        "nonterminal_longside",
        "submission_v339_no_p0_nonterminal_longside_b36__v173action_v300server.csv",
    ),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def enforce_no_p0_add(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Block nonzero point anchors from being changed to point 0."""
    base_arr = np.asarray(base, dtype=int)
    out = np.asarray(cand, dtype=int).copy()
    if len(base_arr) != len(out):
        raise ValueError("base and cand must have the same length")
    blocked = (base_arr != 0) & (out == 0)
    out[blocked] = base_arr[blocked]
    return out


def select_budget(base: np.ndarray, cand: np.ndarray, score: np.ndarray, budget: int) -> np.ndarray:
    base_arr = np.asarray(base, dtype=int)
    cand_arr = np.asarray(cand, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    if not (len(base_arr) == len(cand_arr) == len(score_arr)):
        raise ValueError("base, cand, and score must have the same length")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    cand_arr = enforce_no_p0_add(base_arr, cand_arr)
    out = base_arr.copy()
    eligible = (base_arr != cand_arr) & np.isfinite(score_arr) & (score_arr > 0)
    if budget == 0 or not eligible.any():
        return out
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score_arr[idx], kind="mergesort")]
    chosen = order[: min(int(budget), len(order))]
    out[chosen] = cand_arr[chosen]
    return out


def build_export_frame(anchor: pd.DataFrame, point: np.ndarray) -> pd.DataFrame:
    out = anchor.copy()
    out["pointId"] = enforce_no_p0_add(anchor["pointId"].to_numpy(dtype=int), np.asarray(point, dtype=int))
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("V339 export changed actionId")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("V339 export changed serverGetPoint")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y_true, pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def _point_distribution(values: np.ndarray) -> str:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=10)
    return json.dumps({str(i): int(v) for i, v in enumerate(counts) if v > 0}, sort_keys=True)


def _load_public_anchor(expected_rows: int) -> pd.DataFrame | None:
    if not PUBLIC_ANCHOR_PATH.exists():
        return None
    frame = pd.read_csv(PUBLIC_ANCHOR_PATH)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def _selector_candidate(base: np.ndarray, prob: np.ndarray, selector: str) -> tuple[np.ndarray, np.ndarray]:
    from analysis_v333_hierarchical_point_model import point_depth3
    from analysis_v261_action_conditioned_point_residual import normalize_rows_safe

    p = normalize_rows_safe(prob)
    base_arr = np.asarray(base, dtype=int)
    top = p.argmax(axis=1).astype(int)
    base_prob = p[np.arange(len(p)), np.clip(base_arr, 0, p.shape[1] - 1)]
    margin = p[np.arange(len(p)), top] - base_prob
    eligible = (top != base_arr) & np.isfinite(margin) & (margin > 0.0)

    if selector == "no_p0_add":
        eligible &= ~((base_arr != 0) & (top == 0))
    elif selector == "margin_hi":
        eligible &= ~((base_arr != 0) & (top == 0))
        positive = margin[eligible]
        if len(positive):
            eligible &= margin >= max(0.0, float(np.quantile(positive, 0.50)))
    elif selector == "nonterminal_longside":
        base_depth = np.array([point_depth3(v) for v in base_arr], dtype=int)
        top_depth = np.array([point_depth3(v) for v in top], dtype=int)
        eligible &= (base_arr != 0) & (top != 0) & ((base_depth == 2) | (top_depth == 2))
    else:
        raise ValueError(f"unknown V339 selector: {selector}")

    cand = base_arr.copy()
    cand[eligible] = top[eligible]
    cand = enforce_no_p0_add(base_arr, cand)
    score = np.where(eligible, margin, 0.0)
    return cand, score


def _evaluate_variant(
    spec: VariantSpec,
    state: dict[str, Any],
    hier_oof_prob: np.ndarray,
    hier_test_prob: np.ndarray,
    base_score: float,
    public_anchor: pd.DataFrame | None,
) -> tuple[dict[str, Any], np.ndarray]:
    y = state["y"]
    base_oof = state["v306_oof_point"]
    base_test = state["v306_test_point"]
    cap = spec.budget / len(base_test)
    oof_budget = int(np.floor(len(base_oof) * cap))

    oof_cand, oof_score = _selector_candidate(base_oof, hier_oof_prob, spec.selector)
    test_cand, test_score = _selector_candidate(base_test, hier_test_prob, spec.selector)
    oof_pred = select_budget(base_oof, oof_cand, oof_score, oof_budget)
    test_pred = select_budget(base_test, test_cand, test_score, spec.budget)

    score = macro_f1(y, oof_pred)
    delta = score - base_score
    changed_vs_v306 = test_pred != base_test
    churn_v306 = point_distribution_report(base_test, test_pred)
    public_point = None if public_anchor is None else public_anchor["pointId"].to_numpy(dtype=int)
    duplicate_public = bool(public_point is not None and np.array_equal(test_pred, public_point))
    churn_v338 = point_distribution_report(public_point, test_pred) if public_point is not None else {}
    evidence_pass = bool(delta >= MIN_POINT_DELTA)
    decision = "EXPORT_LOCAL" if evidence_pass and not duplicate_public else "DO_NOT_EXPORT"
    if duplicate_public:
        decision = "DUPLICATE_PUBLIC_ANCHOR"

    record = {
        "candidate": spec.name,
        "selector": spec.selector,
        "budget": spec.budget,
        "oof_budget": oof_budget,
        "base_point_macro_f1": base_score,
        "point_macro_f1": score,
        "point_oof_delta_vs_v306": delta,
        "test_changed_rows_vs_v306": int(changed_vs_v306.sum()),
        "test_changed_rows_vs_v338": int(churn_v338.get("changed_rows", -1)),
        "oof_changed_rows": int(np.sum(oof_pred != base_oof)),
        "test_margin_mean_changed": float(test_score[changed_vs_v306].mean()) if changed_vs_v306.any() else 0.0,
        "oof_margin_mean_changed": float(oof_score[oof_pred != base_oof].mean()) if np.any(oof_pred != base_oof) else 0.0,
        "test_point0_additions": int(churn_v306["point0_additions"]),
        "test_point0_removals": int(churn_v306["point0_removals"]),
        "anchor_point0_total": int(churn_v306["point0_base"]),
        "test_point0_total": int(churn_v306["point0_candidate"]),
        "duplicate_public_anchor": duplicate_public,
        "test_distribution": _point_distribution(test_pred),
        "anchor_distribution": _point_distribution(base_test),
        "decision": decision,
    }
    return record, test_pred


def _write_blocked(exc: Exception) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "V339",
        "decision": "BLOCKED_MISSING_ANCHOR",
        "verdict": "BLOCKED_MISSING_ANCHOR",
        "generated_submissions": [],
        "anchor_error": f"{type(exc).__name__}: {exc}",
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
            "fixed_action": "V173 via V306 package anchor",
            "fixed_server": "V300 via V306 package anchor",
            "no_point0_additions": True,
        },
    }
    pd.DataFrame().to_csv(safe_output_path(OUTDIR, "candidate_summary.csv"), index=False)
    pd.DataFrame().to_csv(safe_output_path(OUTDIR, "point_churn_report.csv"), index=False)
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    try:
        from analysis_v333_hierarchical_point_model import (
            numeric_feature_columns,
            reconstruct_v306_point_anchor,
            train_hierarchical_point_probabilities,
        )

        state = reconstruct_v306_point_anchor()
    except Exception as exc:
        return _write_blocked(exc)

    train_df = state["train_df"]
    test_df = state["test_df"]
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0
    features = [c for c in numeric_feature_columns(train_df, include_proxy=True) if c in test_df]
    features = [c for c in features if c != "v333_v306_point_anchor"]
    hier_oof_prob, hier_test_prob, hier_folds = train_hierarchical_point_probabilities(train_df, test_df, features)
    base_score = macro_f1(state["y"], state["v306_oof_point"])
    public_anchor = _load_public_anchor(len(state["package_anchor"]))

    records: list[dict[str, Any]] = []
    churn_rows: list[dict[str, Any]] = []
    generated: list[dict[str, str]] = []
    best_pred_by_name: dict[str, np.ndarray] = {}
    base_test = state["v306_test_point"]
    public_point = None if public_anchor is None else public_anchor["pointId"].to_numpy(dtype=int)

    for spec in VARIANTS:
        record, test_pred = _evaluate_variant(spec, state, hier_oof_prob, hier_test_prob, base_score, public_anchor)
        if record["decision"] == "EXPORT_LOCAL":
            out = build_export_frame(state["package_anchor"], test_pred)
            path = safe_output_path(OUTDIR, spec.filename)
            out.to_csv(path, index=False, float_format="%.8f")
            record["submission"] = spec.filename
            record["path"] = relative_path(path)
            generated.append({"candidate": spec.name, "submission": spec.filename, "path": relative_path(path)})
        records.append(record)
        best_pred_by_name[spec.name] = test_pred

        changed = np.where((test_pred != base_test) | ((public_point is not None) & (test_pred != public_point)))[0]
        for idx in changed:
            churn_rows.append(
                {
                    "candidate": spec.name,
                    "row_id": int(idx),
                    "rally_uid": state["package_anchor"].iloc[idx]["rally_uid"],
                    "v306_pointId": int(base_test[idx]),
                    "candidate_pointId": int(test_pred[idx]),
                    "v338_pointId": None if public_point is None else int(public_point[idx]),
                    "changed_vs_v306": bool(test_pred[idx] != base_test[idx]),
                    "changed_vs_v338": None if public_point is None else bool(test_pred[idx] != public_point[idx]),
                    "point0_addition_vs_v306": bool(base_test[idx] != 0 and test_pred[idx] == 0),
                }
            )

    summary = pd.DataFrame(records).sort_values(
        ["decision", "point_oof_delta_vs_v306", "test_changed_rows_vs_v306"],
        ascending=[True, False, True],
    )
    summary.to_csv(safe_output_path(OUTDIR, "candidate_summary.csv"), index=False)
    pd.DataFrame(churn_rows).to_csv(safe_output_path(OUTDIR, "point_churn_report.csv"), index=False)
    eligible = summary[summary["decision"].eq("EXPORT_LOCAL")]
    best_source = eligible if not eligible.empty else summary
    best = best_source.sort_values(["point_oof_delta_vs_v306", "test_changed_rows_vs_v306"], ascending=[False, True]).head(1)
    best_dict = best.iloc[0].to_dict() if not best.empty else {}
    report = {
        "version": "V339",
        "decision": "HAS_EXPORT" if generated else "DO_NOT_UPLOAD",
        "verdict": "HAS_EVIDENCE_CANDIDATE" if generated else "NO_EXPORT_NO_EVIDENCE",
        "anchor_status": state.get("status"),
        "anchor_source": state.get("anchor_source"),
        "public_anchor": relative_path(PUBLIC_ANCHOR_PATH),
        "base_anchor": relative_path(BASE_ANCHOR_PATH),
        "base_point_macro_f1": base_score,
        "raw_hier_point_macro_f1": macro_f1(state["y"], hier_oof_prob.argmax(axis=1).astype(int)),
        "best_candidate": best_dict,
        "generated_submissions": generated,
        "candidate_summary": relative_path(OUTDIR / "candidate_summary.csv"),
        "point_churn_report": relative_path(OUTDIR / "point_churn_report.csv"),
        "features_count": len(features),
        "folds": state.get("folds", []) + hier_folds,
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
            "fixed_action": "V173 via V306 package anchor",
            "fixed_server": "V300 via V306 package anchor",
            "no_point0_additions": True,
            "export_gate": f"point_oof_delta_vs_v306 >= {MIN_POINT_DELTA}",
        },
        "notes": [
            "All selectors enforce no nonzero-to-zero point changes before export.",
            "Submissions preserve actionId and serverGetPoint exactly from V306.",
            "Candidates identical to the V338 public-positive anchor are marked DUPLICATE_PUBLIC_ANCHOR.",
        ],
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            _json_safe(
                {
                    "outdir": relative_path(OUTDIR),
                    "decision": report.get("decision"),
                    "verdict": report.get("verdict"),
                    "best": (report.get("best_candidate") or {}).get("candidate"),
                    "generated": len(report.get("generated_submissions", [])),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
