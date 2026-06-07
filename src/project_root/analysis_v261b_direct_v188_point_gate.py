"""V261B direct V188 point-anchor audit.

V261 showed a positive point-residual signal, but only against a fold-safe
tabular proxy base.  This script is intentionally conservative: it first looks
for a direct V188 cap5 OOF point prediction/probability artifact.  If that
artifact is not present, V261B writes an explicit blocked report and does not
package upload candidates.

No TTMATCH, old-server labels, action changes, or upload-directory writes are
used here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


OUTDIR = Path("v261b_direct_v188_point_gate")
ANCHOR_SUBMISSION = Path("upload_candidates_20260519/submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv")
V188_SEARCH = Path("v188_point_intent_gru/v188_search.csv")
V261_SEARCH = Path("v261_action_conditioned_point_residual/v261_point_search.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
POINT_CLASSES = list(range(10))
SEARCH_ROOTS = [
    Path("v188_point_intent_gru"),
    Path("v193_v188_calibrated_residual"),
    Path("v192_v188_generalization_audit"),
    Path("r180_point_physics_calibration"),
    Path("r177_generalization_finalizer"),
]


def cap_changed_rows(scores: np.ndarray, cap: float) -> np.ndarray:
    """Select the highest-scored rows under a fractional cap."""
    arr = np.asarray(scores, dtype=float)
    budget = int(np.floor(len(arr) * float(cap)))
    mask = np.zeros(len(arr), dtype=bool)
    if budget <= 0 or len(arr) == 0:
        return mask
    clean = np.where(np.isfinite(arr), arr, -np.inf)
    order = np.argsort(-clean, kind="mergesort")[:budget]
    mask[order] = True
    return mask


def point0_rate(labels: np.ndarray) -> float:
    arr = np.asarray(labels, dtype=int)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr == 0))


def per_class_f1_delta(y_true: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray, classes: list[int]) -> pd.DataFrame:
    rows = []
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(base_pred, dtype=int)
    cand = np.asarray(cand_pred, dtype=int)
    for cls in classes:
        base_f1 = f1_score(y == cls, base == cls, zero_division=0)
        cand_f1 = f1_score(y == cls, cand == cls, zero_division=0)
        rows.append({"class_id": int(cls), "base_f1": float(base_f1), "candidate_f1": float(cand_f1), "delta_f1": float(cand_f1 - base_f1)})
    return pd.DataFrame(rows)


def is_direct_v188_oof_artifact(path: str | Path) -> bool:
    """True only for row-level V188 OOF point prediction/probability artifacts."""
    p = Path(path)
    lower = str(p).replace("\\", "/").lower()
    if p.suffix.lower() not in {".npy", ".npz", ".pkl", ".parquet"}:
        return False
    if "v188" not in lower or "oof" not in lower or "point" not in lower:
        return False
    diagnostic_tokens = ["bias_grid", "search", "report", "fold_metrics"]
    return not any(token in lower for token in diagnostic_tokens)


def _iter_files(root: Path):
    if not root.exists():
        return
    for current, dirs, files in os.walk(root, topdown=True, onerror=lambda _: None):
        dirs[:] = [d for d in dirs if not d.startswith("pytest-cache-files") and d != "__pycache__"]
        for name in files:
            yield Path(current) / name


def _csv_metric(path: Path, candidate: str) -> dict:
    if not path.exists():
        return {"found": False, "path": str(path), "candidate": candidate}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive audit branch
        return {"found": False, "path": str(path), "candidate": candidate, "error": str(exc)}
    if "candidate" not in df.columns:
        return {"found": False, "path": str(path), "candidate": candidate, "reason": "no_candidate_column"}
    hit = df[df["candidate"].astype(str).eq(candidate)]
    if hit.empty:
        return {"found": False, "path": str(path), "candidate": candidate, "reason": "candidate_missing"}
    return {"found": True, "path": str(path), "candidate": candidate, "row": hit.iloc[0].to_dict()}


def discover_v188_oof_anchor() -> dict:
    """Return direct OOF audit state.

    We only accept artifacts whose path/name explicitly identifies V188 and OOF
    point predictions/probabilities.  Aggregate search CSV metrics are useful
    context but are not enough for same-row V261B validation.
    """
    candidate = "v188_r186_w005_a0p05_cap0p05"
    accepted: list[dict] = []
    nearby: list[dict] = []
    for root in SEARCH_ROOTS:
        for path in _iter_files(root) or []:
            lower = str(path).replace("\\", "/").lower()
            suffix_ok = path.suffix.lower() in {".npy", ".npz", ".pkl", ".parquet", ".csv"}
            if not suffix_ok:
                continue
            has_oof = "oof" in lower
            has_point = "point" in lower
            has_v188 = "v188" in lower
            if has_oof and has_point:
                nearby.append({"path": str(path), "has_v188": has_v188})
            if is_direct_v188_oof_artifact(path):
                accepted.append({"path": str(path), "reason": "path_contains_v188_oof_point"})
    metric = _csv_metric(V188_SEARCH, candidate)
    direct_found = len(accepted) > 0
    return {
        "direct_v188_oof_found": direct_found,
        "verdict": "DIRECT_V188_OOF_FOUND" if direct_found else "BLOCKED_NEEDS_V188_OOF",
        "accepted_direct_oof_artifacts": accepted,
        "nearby_oof_point_artifacts": nearby,
        "known_v188_search_metric": metric,
        "search_roots": [str(p) for p in SEARCH_ROOTS],
    }


def load_anchor_submission() -> pd.DataFrame:
    sub = pd.read_csv(ANCHOR_SUBMISSION)
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected anchor columns: {list(sub.columns)}")
    if len(sub) != 1845:
        raise ValueError(f"Unexpected anchor row count: {len(sub)}")
    return sub


def write_blocked_outputs(audit: dict) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    anchor_info = {}
    if ANCHOR_SUBMISSION.exists():
        anchor = load_anchor_submission()
        anchor_info = {
            "path": str(ANCHOR_SUBMISSION),
            "rows": int(len(anchor)),
            "point0_rate": point0_rate(anchor["pointId"].to_numpy()),
            "point_distribution": json.dumps({str(k): int(v) for k, v in enumerate(np.bincount(anchor["pointId"].astype(int), minlength=10)) if v > 0}, sort_keys=True),
        }
    v261_context = {}
    if V261_SEARCH.exists():
        df = pd.read_csv(V261_SEARCH)
        if "candidate" in df.columns and not df.empty:
            hit = df[df["candidate"].astype(str).str.startswith("v261_action_conditioned_cap")]
            if not hit.empty:
                best = hit.sort_values(["delta_vs_base_proxy", "point_macro_f1"], ascending=[False, False]).iloc[0]
                v261_context = {
                    "proxy_best_candidate": str(best.get("candidate", "")),
                    "proxy_delta_vs_base": float(best.get("delta_vs_base_proxy", np.nan)),
                    "proxy_test_changed_rows": int(best.get("test_changed_rows", 0)),
                    "proxy_test_churn_vs_current_anchor": float(best.get("test_churn_vs_current_anchor", np.nan)),
                }
    search_rows = [
        {
            "candidate": "v261b_direct_v188_oof_gate",
            "verdict": "BLOCKED_NEEDS_V188_OOF",
            "point_macro_f1": np.nan,
            "delta_vs_direct_v188": np.nan,
            "delta_vs_base": np.nan,
            "point_churn_vs_base": np.nan,
            "test_churn_vs_current_anchor": 0.0,
            "test_changed_rows": 0,
            "point0_rate": anchor_info.get("point0_rate", np.nan),
            "reason": "Direct V188 cap5 OOF prediction/probability artifact was not found; V261 proxy-positive result cannot be promoted.",
        }
    ]
    pd.DataFrame(search_rows).to_csv(OUTDIR / "v261b_point_search.csv", index=False)
    pd.DataFrame(columns=["class_id", "base_f1", "candidate_f1", "delta_f1"]).to_csv(OUTDIR / "v261b_class_f1_delta.csv", index=False)

    report = {
        "verdict": "BLOCKED_NEEDS_V188_OOF",
        "upload_recommendation": "do_not_upload",
        "audit": audit,
        "anchor_submission": anchor_info,
        "v261_proxy_context": v261_context,
        "questionnaire_constraints": [
            "Treat task 1/2 as sequential multi-class Macro-F1 problems.",
            "Do not optimize majority-class accuracy.",
            "Use class imbalance and per-class F1 checks before any point residual upload.",
            "Keep server probability unchanged for AUC; V261B changes point only if directly validated.",
        ],
        "next_step": "Re-run or modify V188 to persist OOF point probabilities/predictions for v188_r186_w005_a0p05_cap0p05, then rerun V261B.",
    }
    (OUTDIR / "v261b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v261b_report.md").write_text(
        "# V261B Direct V188 Point Gate\n\n"
        "- Verdict: `BLOCKED_NEEDS_V188_OOF`\n"
        "- Upload recommendation: `do_not_upload`\n"
        f"- Anchor point0 rate: `{anchor_info.get('point0_rate', float('nan')):.6f}`\n"
        f"- V261 proxy context: `{json.dumps(v261_context, sort_keys=True)}`\n\n"
        "## Audit\n\n"
        f"- Direct V188 OOF found: `{audit['direct_v188_oof_found']}`\n"
        f"- Nearby OOF point artifacts: `{len(audit['nearby_oof_point_artifacts'])}`\n"
        "- Known V188 search metric exists: "
        f"`{audit['known_v188_search_metric'].get('found', False)}`\n\n"
        "## Interpretation\n\n"
        "V261 remains a proxy-positive point residual, but V261B cannot validate it against the current V188 cap5 OOF anchor from existing artifacts. "
        "No submission files were generated.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    OUTDIR.mkdir(exist_ok=True)
    audit = discover_v188_oof_anchor()
    (OUTDIR / "v261b_oof_anchor_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    if not audit["direct_v188_oof_found"]:
        report = write_blocked_outputs(audit)
        print(json.dumps({"verdict": report["verdict"], "outdir": str(OUTDIR), "upload_recommendation": report["upload_recommendation"]}, indent=2))
        return
    raise SystemExit("Direct V188 OOF artifacts were found, but row-level validation wiring is intentionally not automatic yet.")


if __name__ == "__main__":
    main()
