"""V287 weak-action gated ensemble over the V173/V261 clean anchor.

V286 found a small local advantage on weak actions, mainly 5/7 and a little
0/3, but it also hurt protected classes. This script keeps V173 as the anchor
and only lets V286 contribute on explicitly allowed weak-action sets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import ACTION_CLASSES
from analysis_v286_weak_action_specialist_pretraining import (
    PROTECTED_ACTIONS,
    WEAK_ACTIONS,
    class_f1,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v287_weak_action_gated_ensemble"
V286_OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"

V286_OOF = V286_OUTDIR / "v286_specialist_oof.csv"
V286_SUB_BY_CHURN = {
    0.0025: V286_OUTDIR / "submission_v286_weak_spec_churn0p0025__pv261cap1__sr121.csv",
    0.005: V286_OUTDIR / "submission_v286_weak_spec_churn0p005__pv261cap1__sr121.csv",
    0.010: V286_OUTDIR / "submission_v286_weak_spec_churn0p010__pv261cap1__sr121.csv",
    0.020: V286_OUTDIR / "submission_v286_weak_spec_churn0p020__pv261cap1__sr121.csv",
}

ALLOW_SETS = {
    "safe57": {5, 7},
    "medium0357": {0, 3, 5, 7},
    "broad03578914": {0, 3, 5, 7, 8, 9, 14},
}


def apply_allowed_action_filter(frame: pd.DataFrame, allowed_actions: Iterable[int]) -> pd.DataFrame:
    allowed = {int(x) for x in allowed_actions}
    out = frame[frame["candidate_action"].astype(int).isin(allowed)].copy()
    return out.reset_index(drop=True)


def select_best_candidate_per_row(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)
    ranked = frame.sort_values(["row_id", "specialist_score", "support_count"], ascending=[True, False, False])
    return ranked.groupby("row_id", as_index=False, sort=False).head(1).reset_index(drop=True)


def apply_row_cap(anchor: np.ndarray, row_candidates: pd.DataFrame, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(anchor, dtype=int).copy()
    selected = np.zeros(len(pred), dtype=bool)
    if row_candidates.empty or max_rows <= 0:
        return pred, selected
    ranked = row_candidates.sort_values(["specialist_score", "support_count"], ascending=[False, False]).head(int(max_rows))
    ids = ranked["row_id"].astype(int).to_numpy()
    selected[ids] = True
    pred[ids] = ranked["candidate_action"].astype(int).to_numpy()
    return pred, selected


def changed_row_report(anchor: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    anchor = np.asarray(anchor, dtype=int)
    pred = np.asarray(pred, dtype=int)
    changed = pred != anchor
    report: dict[str, int] = {"changed_rows": int(changed.sum())}
    for action in sorted(set(pred[changed].tolist())):
        report[f"changed_to_{int(action)}"] = int(np.sum(changed & (pred == int(action))))
    return report


def _cap_token(churn: float) -> str:
    return f"{float(churn):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_oof_candidate_frame(oof: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for action in WEAK_ACTIONS.tolist():
        score_col = f"specialist_p_{action}"
        support_col = f"support_{action}"
        frame = pd.DataFrame(
            {
                "row_id": np.arange(len(oof), dtype=int),
                "anchor_action": oof["anchor_action"].astype(int).to_numpy(),
                "candidate_action": int(action),
                "specialist_score": pd.to_numeric(oof[score_col], errors="coerce").fillna(0.0).to_numpy(),
                "support_count": pd.to_numeric(oof[support_col], errors="coerce").fillna(0).to_numpy(),
            }
        )
        pieces.append(frame)
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["candidate_action"].astype(int).ne(out["anchor_action"].astype(int))]
    return out.reset_index(drop=True)


def _f1_macro(y: np.ndarray, pred: np.ndarray, labels: list[int] | np.ndarray = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def evaluate_variant(
    name: str,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    pred_oof: np.ndarray,
    anchor_test: np.ndarray,
    pred_test: np.ndarray,
    allowed: set[int],
    max_churn: float,
) -> dict[str, Any]:
    base_macro = _f1_macro(y, anchor_oof)
    macro = _f1_macro(y, pred_oof)
    weak_base = _f1_macro(y, anchor_oof, WEAK_ACTIONS)
    weak = _f1_macro(y, pred_oof, WEAK_ACTIONS)
    prot_base = _f1_macro(y, anchor_oof, PROTECTED_ACTIONS)
    prot = _f1_macro(y, pred_oof, PROTECTED_ACTIONS)
    class_delta = {
        str(k): float(class_f1(y, pred_oof, ACTION_CLASSES)[k] - class_f1(y, anchor_oof, ACTION_CLASSES)[k])
        for k in ACTION_CLASSES
    }
    changed_test = pred_test != anchor_test
    rec = {
        "candidate": name,
        "allowed_actions": "/".join(str(x) for x in sorted(allowed)),
        "max_churn": float(max_churn),
        "action_macro_f1": float(macro),
        "delta_vs_v173": float(macro - base_macro),
        "weak_mean_delta": float(weak - weak_base),
        "protected_mean_delta": float(prot - prot_base),
        "test_changed_rows": int(changed_test.sum()),
        "test_churn": float(changed_test.mean()),
        "class_f1_delta_json": json.dumps(class_delta, sort_keys=True),
        **changed_row_report(anchor_test, pred_test),
    }
    rec["candidate_tier"] = (
        "clean_probe"
        if rec["delta_vs_v173"] > 0 and rec["weak_mean_delta"] > 0 and rec["protected_mean_delta"] >= -0.0005
        else "diagnostic_only"
    )
    return rec


def write_submission(name: str, action: np.ndarray, anchor_sub: pd.DataFrame) -> Path:
    out = pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return path


def filtered_test_prediction(anchor_test: np.ndarray, source_test: np.ndarray, allowed: set[int]) -> np.ndarray:
    pred = np.asarray(anchor_test, dtype=int).copy()
    source = np.asarray(source_test, dtype=int)
    changed = source != anchor_test
    keep = changed & np.isin(source, sorted(allowed))
    pred[keep] = source[keep]
    return pred


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v287*.csv"):
        stale.unlink()

    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF file: {V286_OOF}")
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    oof = pd.read_csv(V286_OOF)
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()

    oof_candidates = build_oof_candidate_frame(oof)
    rows = []
    generated = []
    for allow_name, allowed in ALLOW_SETS.items():
        allowed_oof = apply_allowed_action_filter(oof_candidates, allowed)
        allowed_oof = allowed_oof[
            (allowed_oof["specialist_score"].astype(float) >= 0.55)
            & (allowed_oof["support_count"].astype(float) >= 10)
        ].copy()
        best_per_row = select_best_candidate_per_row(allowed_oof)
        for max_churn, source_path in V286_SUB_BY_CHURN.items():
            max_rows = int(math.floor(len(anchor_oof) * max_churn))
            pred_oof, _selected = apply_row_cap(anchor_oof, best_per_row, max_rows)
            source_sub = pd.read_csv(source_path)
            pred_test = filtered_test_prediction(anchor_test, source_sub["actionId"].astype(int).to_numpy(), allowed)
            name = f"v287_{allow_name}_c{_cap_token(max_churn)}"
            rows.append(evaluate_variant(name, y, anchor_oof, pred_oof, anchor_test, pred_test, allowed, max_churn))
            sub_name = f"submission_{name}__pv261cap1__sr121.csv"
            generated.append(str(write_submission(sub_name, pred_test, anchor_sub).relative_to(ROOT)))

    search = pd.DataFrame(rows).sort_values(
        ["candidate_tier", "delta_vs_v173", "weak_mean_delta", "test_changed_rows"],
        ascending=[True, False, False, True],
    )
    search.to_csv(OUTDIR / "v287_action_search.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    upload_recommendation = "DO_NOT_UPLOAD"
    clean = search[search["candidate_tier"].eq("clean_probe")].copy()
    if not clean.empty:
        candidate = clean.sort_values(["test_changed_rows", "delta_vs_v173"], ascending=[True, False]).iloc[0]
        if (
            float(candidate["delta_vs_v173"]) >= 0.001
            and float(candidate["weak_mean_delta"]) > 0
            and float(candidate["protected_mean_delta"]) >= 0
            and 1 <= int(candidate["test_changed_rows"]) <= 20
        ):
            upload_recommendation = "REVIEW_LOW_CHURN_WEAK_ACTION_PROBE"

    report = {
        "version": "V287",
        "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
        "best_candidate": best,
        "generated_submissions": generated,
        "upload_recommendation": upload_recommendation,
    }
    (OUTDIR / "v287_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# V287 weak-action gated ensemble",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        f"Best candidate: `{best.get('candidate', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Weak mean delta: {float(best.get('weak_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Upload recommendation: {upload_recommendation}",
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v287_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "best_candidate": report["best_candidate"].get("candidate", ""),
                "best_delta_vs_v173": report["best_candidate"].get("delta_vs_v173", 0.0),
                "generated_submissions": len(report["generated_submissions"]),
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
