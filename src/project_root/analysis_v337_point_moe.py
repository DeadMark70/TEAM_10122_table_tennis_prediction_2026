"""V337 lightweight point MoE.

This branch keeps V173 action and V300/R121 server fixed through the strict
V306 package anchor, then tests small point-router variants. It does not read
TTMATCH, old-server labels, or write upload-candidate directories.
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

try:
    from analysis_v335_moe_anchor_contract import (
        point_distribution_report,
        safe_output_path,
        validate_submission_schema,
        write_json,
    )
except Exception:  # pragma: no cover - temporary fallback until Worker A lands V335.
    def safe_output_path(outdir: Path, filename: str) -> Path:
        root = Path(outdir).resolve()
        path = (root / filename).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"unsafe output path: {path}")
        return path

    def validate_submission_schema(frame: pd.DataFrame, expected_rows: int | None = 1845) -> None:
        expected = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
        if list(frame.columns) != expected:
            raise ValueError(f"bad columns: {list(frame.columns)}")
        if expected_rows is not None and len(frame) != expected_rows:
            raise ValueError(f"bad row count: {len(frame)}")

    def point_distribution_report(base_point: Any, cand_point: Any) -> dict[str, Any]:
        base = np.asarray(base_point, dtype=int)
        cand = np.asarray(cand_point, dtype=int)
        return {
            "changed_rows": int(np.sum(base != cand)),
            "point0_additions": int(np.sum((base != 0) & (cand == 0))),
            "point0_removals": int(np.sum((base == 0) & (cand != 0))),
            "base_point0_total": int(np.sum(base == 0)),
            "cand_point0_total": int(np.sum(cand == 0)),
        }

    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v337_point_moe"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
MIN_POINT_DELTA = 0.003


@dataclass(frozen=True)
class VariantSpec:
    name: str
    budget: int
    selector: str
    expert: str
    allow_p0_add: bool
    filename: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("point_moe_no_p0_add_b12", 12, "no_p0_add", "hierarchical", False, "submission_v337_point_moe_no_p0_add_b12__v173action_v300server.csv"),
    VariantSpec("point_moe_no_p0_add_b24", 24, "no_p0_add", "hierarchical", False, "submission_v337_point_moe_no_p0_add_b24__v173action_v300server.csv"),
    VariantSpec("point_moe_p0_cap12", 12, "soft", "hierarchical", True, "submission_v337_point_moe_p0_cap12__v173action_v300server.csv"),
    VariantSpec("point_moe_p0_cap18", 18, "soft", "hierarchical", True, "submission_v337_point_moe_p0_cap18__v173action_v300server.csv"),
    VariantSpec("point_moe_depthside_cap24", 24, "depth_confident", "hierarchical", False, "submission_v337_point_moe_depthside_cap24__v173action_v300server.csv"),
    VariantSpec("point_moe_actioncond_cap24", 24, "action_conditioned_table", "rule", False, "submission_v337_point_moe_actioncond_cap24__v173action_v300server.csv"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
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


def point_to_depth_side(point_id: int) -> tuple[int, int]:
    point = int(point_id)
    if point == 0:
        return -1, -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    z = point - 1
    return z // 3, z % 3


def depth_side_to_point(depth: int, side: int) -> int:
    d = int(np.clip(depth, 0, 2))
    s = int(np.clip(side, 0, 2))
    return d * 3 + s + 1


def apply_point0_policy(base: np.ndarray, cand: np.ndarray, *, allow_p0_add: bool) -> np.ndarray:
    base_arr = np.asarray(base, dtype=int)
    out = np.asarray(cand, dtype=int).copy()
    if len(base_arr) != len(out):
        raise ValueError("base and cand must have the same length")
    if not allow_p0_add:
        blocked = (base_arr != 0) & (out == 0)
        out[blocked] = base_arr[blocked]
    return out


def select_by_budget(base: np.ndarray, cand: np.ndarray, utility: np.ndarray, budget: int) -> np.ndarray:
    base_arr = np.asarray(base, dtype=int)
    cand_arr = np.asarray(cand, dtype=int)
    util = np.asarray(utility, dtype=float)
    if not (len(base_arr) == len(cand_arr) == len(util)):
        raise ValueError("base, cand, and utility must have the same length")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    out = base_arr.copy()
    eligible = (base_arr != cand_arr) & np.isfinite(util) & (util > 0)
    if budget == 0 or not eligible.any():
        return out
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-util[idx], kind="mergesort")]
    chosen = order[: min(int(budget), len(order))]
    out[chosen] = cand_arr[chosen]
    return out


def build_point_experts(test_meta: pd.DataFrame, base_point: np.ndarray, action_anchor: np.ndarray) -> dict[str, np.ndarray]:
    base = np.asarray(base_point, dtype=int)
    action = np.asarray(action_anchor, dtype=int)
    if len(base) != len(test_meta) or len(action) != len(base):
        raise ValueError("test_meta, base_point, and action_anchor must align")

    lag_point = (
        pd.to_numeric(test_meta["lag0_pointId"], errors="coerce").fillna(pd.Series(base)).to_numpy(dtype=int)
        if "lag0_pointId" in test_meta
        else base.copy()
    )
    prefix = (
        pd.to_numeric(test_meta["prefix_len"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if "prefix_len" in test_meta
        else np.zeros(len(base), dtype=int)
    )

    terminal = base.copy()
    terminal[np.isin(action, [10, 11, 12, 13, 14])] = 0

    depth = base.copy()
    side = base.copy()
    long_side = base.copy()
    action_cond = base.copy()
    for i, old in enumerate(base):
        if old == 0:
            continue
        d, s = point_to_depth_side(lag_point[i] if 1 <= int(lag_point[i]) <= 9 else old)
        depth[i] = depth_side_to_point((d + (1 if prefix[i] >= 4 else 0)) % 3, s)
        side[i] = depth_side_to_point(d, (s + 1) % 3)
        if old in (7, 8, 9):
            long_side[i] = depth_side_to_point(2, (s + 1) % 3)
        family = 0 if action[i] == 0 else 1 if action[i] <= 7 else 2 if action[i] <= 11 else 3
        action_cond[i] = depth_side_to_point((d + family) % 3, s)

    no_p0_add = apply_point0_policy(base, depth, allow_p0_add=False)
    return {
        "terminal_p0": terminal,
        "depth_short_half_long": depth,
        "side_fh_mid_bh": side,
        "long_side_789": long_side,
        "no_p0_add_depthside": no_p0_add,
        "action_conditioned_table": action_cond,
    }


def macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y_true, pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def build_export_frame(anchor: pd.DataFrame, point: np.ndarray) -> pd.DataFrame:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("V337 export changed actionId")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("V337 export changed serverGetPoint")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def _point_distribution(values: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=10)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def _point0_totals(report: dict[str, Any]) -> dict[str, int]:
    return {
        "anchor_point0_total": int(report.get("point0_base", report.get("base_point0_total", 0))),
        "test_point0_total": int(report.get("point0_candidate", report.get("cand_point0_total", 0))),
        "test_point0_additions": int(report.get("point0_additions", 0)),
        "test_point0_removals": int(report.get("point0_removals", 0)),
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _select_from_prob(base: np.ndarray, prob: np.ndarray, budget: int, selector: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from analysis_v333_hierarchical_point_model import select_variant_predictions

    pred, selected, margin = select_variant_predictions(base, prob, budget, selector)
    return pred.astype(int), selected.astype(bool), margin.astype(float)


def _rule_candidate(base: np.ndarray, meta: pd.DataFrame, action: np.ndarray, expert: str) -> tuple[np.ndarray, np.ndarray]:
    experts = build_point_experts(meta, base, action)
    cand = experts[expert]
    utility = np.zeros(len(base), dtype=float)
    for i, (old, new) in enumerate(zip(base, cand)):
        if old == new:
            continue
        old_d, old_s = point_to_depth_side(old)
        new_d, new_s = point_to_depth_side(new)
        utility[i] = 0.05
        if old != 0 and new != 0 and old_d == new_d and old_s != new_s:
            utility[i] += 0.20
        if expert == "action_conditioned_table" and int(action[i]) in (3, 4, 5, 10, 11):
            utility[i] += 0.15
    return cand, utility


def _evaluate_variant(
    spec: VariantSpec,
    state: dict[str, Any],
    hier_oof_prob: np.ndarray,
    hier_test_prob: np.ndarray,
    base_score: float,
) -> tuple[dict[str, Any], np.ndarray]:
    y = state["y"]
    base_oof = state["v306_oof_point"]
    base_test = state["v306_test_point"]

    if spec.expert == "hierarchical":
        oof_budget = int(np.floor(len(base_oof) * (spec.budget / len(base_test))))
        oof_pred, oof_selected, oof_utility = _select_from_prob(base_oof, hier_oof_prob, oof_budget, spec.selector)
        test_pred, test_selected, test_utility = _select_from_prob(base_test, hier_test_prob, spec.budget, spec.selector)
    else:
        oof_cand, oof_utility = _rule_candidate(base_oof, state["train_df"], state["v333_v173_train_action"], spec.selector)
        test_cand, test_utility = _rule_candidate(base_test, state["test_df"], state["v333_v173_test_action"], spec.selector)
        oof_budget = int(np.floor(len(base_oof) * (spec.budget / len(base_test))))
        oof_pred = select_by_budget(base_oof, apply_point0_policy(base_oof, oof_cand, allow_p0_add=spec.allow_p0_add), oof_utility, oof_budget)
        test_pred = select_by_budget(base_test, apply_point0_policy(base_test, test_cand, allow_p0_add=spec.allow_p0_add), test_utility, spec.budget)
        oof_selected = oof_pred != base_oof
        test_selected = test_pred != base_test

    test_pred = apply_point0_policy(base_test, test_pred, allow_p0_add=spec.allow_p0_add)
    oof_pred = apply_point0_policy(base_oof, oof_pred, allow_p0_add=spec.allow_p0_add)
    test_selected = test_pred != base_test
    oof_selected = oof_pred != base_oof

    score = macro_f1(y, oof_pred)
    churn = point_distribution_report(base_test, test_pred)
    p0 = _point0_totals(churn)
    duplicate_v333 = spec.name in {"point_moe_p0_cap12", "point_moe_p0_cap18", "point_moe_no_p0_add_b24"}
    evidence_pass = bool((score - base_score) >= MIN_POINT_DELTA and (not duplicate_v333 or "no_p0_add" in spec.name))
    decision = "EXPORT_LOCAL" if evidence_pass else "DO_NOT_UPLOAD"
    record = {
        "candidate": spec.name,
        "expert": spec.expert,
        "selector": spec.selector,
        "budget": spec.budget,
        "oof_budget": int(oof_budget),
        "point_macro_f1": score,
        "base_point_macro_f1": base_score,
        "point_oof_delta_vs_v306": score - base_score,
        "test_changed_rows": int(test_selected.sum()),
        "oof_changed_rows": int(oof_selected.sum()),
        "test_churn": float(test_selected.sum() / len(base_test)),
        "test_margin_mean_changed": float(np.asarray(test_utility)[test_selected].mean()) if test_selected.any() else 0.0,
        "oof_margin_mean_changed": float(np.asarray(oof_utility)[oof_selected].mean()) if oof_selected.any() else 0.0,
        "test_distribution": json.dumps(_point_distribution(test_pred), sort_keys=True),
        "anchor_distribution": json.dumps(_point_distribution(base_test), sort_keys=True),
        **p0,
        "duplicate_v333_point0_dose": bool(duplicate_v333 and spec.allow_p0_add),
        "exploratory": bool(duplicate_v333 and spec.allow_p0_add),
        "decision": decision,
    }
    return record, test_pred


def _write_blocked(exc: Exception) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "V337",
        "decision": "BLOCKED_MISSING_ANCHOR",
        "verdict": "BLOCKED_MISSING_ANCHOR",
        "generated_submissions": [],
        "anchor_error": f"{type(exc).__name__}: {exc}",
        "policy": {
            "no_ttm": True,
            "no_old_server": True,
            "no_upload_dir_writes": True,
            "fixed_action": "V173",
            "fixed_server": "V300/R121",
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
            reconstruct_v306_point_anchor,
            train_hierarchical_point_probabilities,
            numeric_feature_columns,
        )

        state = reconstruct_v306_point_anchor()
    except Exception as exc:
        return _write_blocked(exc)

    state["v333_v173_train_action"] = state["train_df"]["v333_v173_action_anchor"].to_numpy(dtype=int)
    state["v333_v173_test_action"] = state["test_df"]["v333_v173_action_anchor"].to_numpy(dtype=int)
    for col in state["train_df"].columns:
        if col not in state["test_df"] and pd.api.types.is_numeric_dtype(state["train_df"][col]):
            state["test_df"][col] = 0
    features = [c for c in numeric_feature_columns(state["train_df"], include_proxy=True) if c in state["test_df"]]
    features = [c for c in features if c != "v333_v306_point_anchor"]
    hier_oof_prob, hier_test_prob, hier_folds = train_hierarchical_point_probabilities(
        state["train_df"], state["test_df"], features
    )
    base_score = macro_f1(state["y"], state["v306_oof_point"])

    records: list[dict[str, Any]] = []
    churn_rows: list[dict[str, Any]] = []
    generated: list[dict[str, str]] = []
    best_pred_by_name: dict[str, np.ndarray] = {}
    for spec in VARIANTS:
        record, test_pred = _evaluate_variant(spec, state, hier_oof_prob, hier_test_prob, base_score)
        if record["decision"] == "EXPORT_LOCAL":
            out = build_export_frame(state["package_anchor"], test_pred)
            path = safe_output_path(OUTDIR, spec.filename)
            out.to_csv(path, index=False, float_format="%.8f")
            record["submission"] = spec.filename
            record["path"] = _relative(path)
            generated.append({"candidate": spec.name, "submission": spec.filename, "path": _relative(path)})
        records.append(record)
        best_pred_by_name[spec.name] = test_pred

        changed = np.where(test_pred != state["v306_test_point"])[0]
        for idx in changed:
            churn_rows.append(
                {
                    "candidate": spec.name,
                    "row_id": int(idx),
                    "rally_uid": state["package_anchor"].iloc[idx]["rally_uid"],
                    "old_pointId": int(state["v306_test_point"][idx]),
                    "new_pointId": int(test_pred[idx]),
                    "point0_addition": bool(state["v306_test_point"][idx] != 0 and test_pred[idx] == 0),
                    "point0_removal": bool(state["v306_test_point"][idx] == 0 and test_pred[idx] != 0),
                }
            )

    summary = pd.DataFrame(records).sort_values(
        ["decision", "point_oof_delta_vs_v306", "test_changed_rows"],
        ascending=[True, False, True],
    )
    summary.to_csv(safe_output_path(OUTDIR, "candidate_summary.csv"), index=False)
    pd.DataFrame(churn_rows).to_csv(safe_output_path(OUTDIR, "point_churn_report.csv"), index=False)
    best = summary.sort_values(["point_oof_delta_vs_v306", "test_changed_rows"], ascending=[False, True]).head(1)
    best_dict = best.iloc[0].to_dict() if not best.empty else {}
    report = {
        "version": "V337",
        "decision": "HAS_EXPORT" if generated else "DO_NOT_UPLOAD",
        "verdict": "HAS_EVIDENCE_CANDIDATE" if generated else "NO_EXPORT_NO_EVIDENCE",
        "anchor_status": state.get("status"),
        "anchor_source": state.get("anchor_source"),
        "action_anchor_source": state.get("action_anchor_source"),
        "v173_rebuild_status": state.get("v173_rebuild_status"),
        "base_point_macro_f1": base_score,
        "raw_hier_point_macro_f1": macro_f1(state["y"], hier_oof_prob.argmax(axis=1).astype(int)),
        "best_candidate": best_dict,
        "generated_submissions": generated,
        "candidate_summary": _relative(OUTDIR / "candidate_summary.csv"),
        "point_churn_report": _relative(OUTDIR / "point_churn_report.csv"),
        "features_count": len(features),
        "folds": state.get("folds", []) + hier_folds,
        "policy": {
            "fixed_action": "V173 action via V306 package anchor",
            "fixed_server": "V300/R121 server via V306 package anchor",
            "base_point_anchor": "V306/V300/V261 point family",
            "no_ttm": True,
            "no_old_server": True,
            "no_upload_dir_writes": True,
            "manual_row_edits": False,
        },
        "notes": [
            "P0-add variants are marked exploratory when they overlap V333 point0-dose behavior.",
            "No-p0-add variants block nonzero-to-zero point changes.",
            "Submissions preserve actionId and serverGetPoint exactly.",
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
                    "outdir": _relative(OUTDIR),
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
