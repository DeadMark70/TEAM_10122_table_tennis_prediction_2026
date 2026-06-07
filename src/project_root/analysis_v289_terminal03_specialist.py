"""V289 terminal action 0/3 specialist diagnostics.

Generated submissions keep pointId and serverGetPoint fixed to the V261/R121
anchor and are local diagnostic candidates only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_v286_weak_action_specialist_pretraining import class_f1
from analysis_v287_weak_action_gated_ensemble import apply_row_cap, changed_row_report
from analysis_v288_specialist_feature_discovery import PROTECTED_ACTIONS, SERVE_ACTIONS, build_basic_feature_frame
from baseline_lgbm import ACTION_CLASSES


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    if here.name == "analysis" and here.parent.name == "src":
        return here.parent.parent
    return here


ROOT = _repo_root()
OUTDIR = ROOT / "v289_terminal03_specialist"
V286_OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
V286_OOF = V286_OUTDIR / "v286_specialist_oof.csv"
V286_STAGE1 = V286_OUTDIR / "v286_stage1_pretrain_table.csv"
V286_SUB_BY_CHURN = {
    0.0025: V286_OUTDIR / "submission_v286_weak_spec_churn0p0025__pv261cap1__sr121.csv",
    0.005: V286_OUTDIR / "submission_v286_weak_spec_churn0p005__pv261cap1__sr121.csv",
    0.010: V286_OUTDIR / "submission_v286_weak_spec_churn0p010__pv261cap1__sr121.csv",
}

TERMINAL_ACTIONS = [0, 3]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_terminal_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows.copy()
    n_rows = len(base)
    for col in ["prefix_len", "lag0_pointId", "lag0_actionId", "lag0_spinId", "lag0_strengthId", "lag0_positionId"]:
        if col not in base:
            base[col] = pd.Series(np.zeros(n_rows, dtype=int), index=base.index)
    if "scoreTotal" not in base:
        base["scoreTotal"] = pd.Series(np.zeros(n_rows, dtype=float), index=base.index)
    if "serverScoreDiff" not in base:
        base["serverScoreDiff"] = pd.Series(np.zeros(n_rows, dtype=float), index=base.index)
    if "scoreSelf" not in base:
        total = pd.to_numeric(base["scoreTotal"], errors="coerce").fillna(0)
        diff = pd.to_numeric(base["serverScoreDiff"], errors="coerce").fillna(0)
        base["scoreSelf"] = (total + diff) / 2.0
    if "scoreOther" not in base:
        total = pd.to_numeric(base["scoreTotal"], errors="coerce").fillna(0)
        diff = pd.to_numeric(base["serverScoreDiff"], errors="coerce").fillna(0)
        base["scoreOther"] = (total - diff) / 2.0
    out = build_basic_feature_frame(base)
    out["lag0_point_is_zero"] = out["lag0_pointId"].eq(0).astype(int)
    out["lag0_action_is_finisher_like"] = out["lag0_actionId"].isin([0, 3, 13]).astype(int)
    out["is_late_pressure"] = ((out["scoreTotal"] >= 18) & (out["serverScoreDiff"].abs() <= 1)).astype(int)
    out["is_receive_or_rally"] = (out["is_receive"].eq(1) | out["is_rally"].eq(1)).astype(int)
    out["terminal_context_score"] = (
        0.34 * out["lag0_point_is_zero"].astype(float)
        + 0.22 * out["is_late_pressure"].astype(float)
        + 0.18 * out["lag0_action_is_finisher_like"].astype(float)
        + 0.14 * out["is_receive_or_rally"].astype(float)
        + 0.12 * out["v286_ext_terminal_rate"].astype(float).clip(0.0, 1.0)
        if "v286_ext_terminal_rate" in out
        else 0.34 * out["lag0_point_is_zero"].astype(float)
        + 0.22 * out["is_late_pressure"].astype(float)
        + 0.18 * out["lag0_action_is_finisher_like"].astype(float)
        + 0.14 * out["is_receive_or_rally"].astype(float)
    )
    return out


def filter_terminal_candidates(frame: pd.DataFrame, min_score: float, min_support: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)
    filtered = frame[
        frame["candidate_action"].astype(int).isin(TERMINAL_ACTIONS)
        & (pd.to_numeric(frame["terminal_score"], errors="coerce").fillna(0.0) >= float(min_score))
        & (pd.to_numeric(frame["support_count"], errors="coerce").fillna(0) >= int(min_support))
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=frame.columns)
    ranked = filtered.sort_values(["row_id", "terminal_score", "support_count"], ascending=[True, False, False])
    return ranked.groupby("row_id", as_index=False, sort=False).head(1).reset_index(drop=True)


def _load_terminal_features(oof: pd.DataFrame) -> pd.DataFrame:
    if V286_STAGE1.exists():
        stage = pd.read_csv(V286_STAGE1)
        stage = stage[stage["next_actionId"].notna()].reset_index(drop=True)
        if "rally_uid" in stage and "rally_uid" in oof:
            keyed = stage.copy()
            keyed["next_actionId"] = keyed["next_actionId"].astype(int)
            wanted = oof[["rally_uid", "y_true_action"]].copy()
            wanted["_order"] = np.arange(len(wanted), dtype=int)
            wanted["y_true_action"] = wanted["y_true_action"].astype(int)
            matched = wanted.merge(
                keyed,
                left_on=["rally_uid", "y_true_action"],
                right_on=["rally_uid", "next_actionId"],
                how="left",
            )
            matched = matched.sort_values(["_order", "prefix_len"]).groupby("_order", as_index=False).tail(1)
            matched = matched.sort_values("_order").reset_index(drop=True)
            if len(matched) == len(oof) and matched["prefix_len"].notna().all():
                frame = build_terminal_feature_frame(matched)
                keep = [
                    "prefix_len_bin",
                    "phase_bin",
                    "lag0_point_depth",
                    "lag0_pointId",
                    "lag0_action_family",
                    "scoreTotal",
                    "serverScoreDiff",
                    "v286_ext_terminal_rate",
                    "terminal_context_score",
                    "lag0_point_is_zero",
                    "is_late_pressure",
                ]
                return frame[[col for col in keep if col in frame.columns]].reset_index(drop=True)
        if len(stage) == len(oof):
            frame = build_terminal_feature_frame(stage)
            keep = [
                "prefix_len_bin",
                "phase_bin",
                "lag0_point_depth",
                "lag0_pointId",
                "lag0_action_family",
                "scoreTotal",
                "serverScoreDiff",
                "v286_ext_terminal_rate",
                "terminal_context_score",
                "lag0_point_is_zero",
                "is_late_pressure",
            ]
            return frame[[col for col in keep if col in frame.columns]].reset_index(drop=True)
    fallback = build_terminal_feature_frame(oof)
    return fallback.reset_index(drop=True)


def build_terminal_candidate_frame(oof: pd.DataFrame, terminal_features: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    context = pd.to_numeric(terminal_features["terminal_context_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    anchor = oof["anchor_action"].astype(int).to_numpy()
    for action in TERMINAL_ACTIONS:
        prob = pd.to_numeric(oof[f"specialist_p_{action}"], errors="coerce").fillna(0.0)
        support = pd.to_numeric(oof[f"support_{action}"], errors="coerce").fillna(0)
        support_score = np.log1p(support.to_numpy(dtype=float)) / np.log1p(max(float(support.max()), 1.0))
        score = 0.62 * prob.to_numpy(dtype=float) + 0.26 * context.to_numpy(dtype=float) + 0.12 * support_score
        pieces.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(oof), dtype=int),
                    "anchor_action": anchor,
                    "candidate_action": int(action),
                    "terminal_score": score,
                    "support_count": support.to_numpy(dtype=int),
                    "specialist_p": prob.to_numpy(dtype=float),
                    "terminal_context_score": context.to_numpy(dtype=float),
                }
            )
        )
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["candidate_action"].astype(int).ne(out["anchor_action"].astype(int))]
    out = out[~out["anchor_action"].astype(int).isin(PROTECTED_ACTIONS + SERVE_ACTIONS)]
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
    threshold: float,
    min_support: int,
    max_churn: float,
) -> dict[str, Any]:
    base_macro = _f1_macro(y, anchor_oof)
    macro = _f1_macro(y, pred_oof)
    terminal_base = _f1_macro(y, anchor_oof, TERMINAL_ACTIONS)
    terminal = _f1_macro(y, pred_oof, TERMINAL_ACTIONS)
    prot_base = _f1_macro(y, anchor_oof, PROTECTED_ACTIONS)
    prot = _f1_macro(y, pred_oof, PROTECTED_ACTIONS)
    class_delta = {
        str(k): float(class_f1(y, pred_oof, ACTION_CLASSES)[k] - class_f1(y, anchor_oof, ACTION_CLASSES)[k])
        for k in ACTION_CLASSES
    }
    changed_test = pred_test != anchor_test
    rec = {
        "candidate": name,
        "allowed_actions": "0/3",
        "threshold": float(threshold),
        "min_support": int(min_support),
        "max_churn": float(max_churn),
        "action_macro_f1": float(macro),
        "delta_vs_v173": float(macro - base_macro),
        "terminal_mean_delta": float(terminal - terminal_base),
        "protected_mean_delta": float(prot - prot_base),
        "test_changed_rows": int(changed_test.sum()),
        "test_churn": float(changed_test.mean()),
        "class_f1_delta_json": json.dumps(class_delta, sort_keys=True),
        **changed_row_report(anchor_test, pred_test),
    }
    rec["candidate_tier"] = (
        "clean_probe"
        if rec["delta_vs_v173"] >= 0.001
        and rec["terminal_mean_delta"] > 0
        and rec["protected_mean_delta"] >= 0
        and 3 <= rec["test_changed_rows"] <= 25
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


def build_terminal_class_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    anchor_f1 = class_f1(y, anchor, ACTION_CLASSES)
    pred_f1 = class_f1(y, pred, ACTION_CLASSES)
    return pd.DataFrame(
        [
            {
                "action": int(action),
                "is_terminal03": int(action in TERMINAL_ACTIONS),
                "is_protected": int(action in PROTECTED_ACTIONS),
                "anchor_f1": float(anchor_f1[action]),
                "v289_f1": float(pred_f1[action]),
                "delta": float(pred_f1[action] - anchor_f1[action]),
            }
            for action in ACTION_CLASSES
        ]
    )


def _submission_name(max_churn: float) -> str:
    token = {0.0025: "0p0025", 0.005: "0p005", 0.010: "0p010"}[float(max_churn)]
    return f"submission_v289_terminal03_c{token}__pv261cap1__sr121.csv"


def _filtered_test_prediction(anchor_test: np.ndarray, source_test: np.ndarray) -> np.ndarray:
    pred = np.asarray(anchor_test, dtype=int).copy()
    source = np.asarray(source_test, dtype=int)
    changed = source != anchor_test
    keep = changed & np.isin(source, TERMINAL_ACTIONS) & ~np.isin(anchor_test, PROTECTED_ACTIONS + SERVE_ACTIONS)
    pred[keep] = source[keep]
    return pred


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v289*.csv"):
        stale.unlink()
    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF file: {V286_OOF}")
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing anchor submission: {ANCHOR_SUBMISSION}")

    oof = pd.read_csv(V286_OOF)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    terminal_features = _load_terminal_features(oof)
    candidate_pool = build_terminal_candidate_frame(oof, terminal_features)

    rows = []
    predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[float, np.ndarray] = {}
    generated = []
    for max_churn, source_path in V286_SUB_BY_CHURN.items():
        source_sub = pd.read_csv(source_path)
        pred_test = _filtered_test_prediction(anchor_test, source_sub["actionId"].astype(int).to_numpy())
        test_predictions[max_churn] = pred_test
        generated.append(str(write_submission(_submission_name(max_churn), pred_test, anchor_sub).relative_to(ROOT)))

    for threshold in [0.45, 0.50, 0.55, 0.60]:
        for min_support in [10, 20, 40]:
            selected = filter_terminal_candidates(candidate_pool, threshold, min_support)
            capped_input = selected.rename(columns={"terminal_score": "specialist_score"})
            for max_churn in [0.0025, 0.005, 0.010]:
                max_rows = int(math.floor(len(anchor_oof) * max_churn))
                pred_oof, _mask = apply_row_cap(anchor_oof, capped_input, max_rows)
                name = f"v289_terminal03_t{str(threshold).replace('.', 'p')}_s{min_support}_c{str(max_churn).replace('.', 'p')}"
                rows.append(
                    evaluate_variant(
                        name,
                        y,
                        anchor_oof,
                        pred_oof,
                        anchor_test,
                        test_predictions[max_churn],
                        threshold,
                        min_support,
                        max_churn,
                    )
                )
                predictions[name] = pred_oof

    search = pd.DataFrame(rows).sort_values(
        ["candidate_tier", "delta_vs_v173", "terminal_mean_delta", "protected_mean_delta", "test_changed_rows"],
        ascending=[True, False, False, False, True],
    )
    search.to_csv(OUTDIR / "v289_terminal03_search.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    best_pred = predictions.get(str(best.get("candidate", "")), anchor_oof)
    build_terminal_class_report(y, anchor_oof, best_pred).to_csv(OUTDIR / "v289_terminal03_class_report.csv", index=False)

    upload_recommendation = "DO_NOT_UPLOAD"
    clean = search[search["candidate_tier"].eq("clean_probe")].copy()
    if not clean.empty:
        candidate = clean.sort_values(["test_changed_rows", "delta_vs_v173"], ascending=[True, False]).iloc[0]
        if (
            float(candidate["delta_vs_v173"]) >= 0.001
            and float(candidate["protected_mean_delta"]) >= 0
            and 3 <= int(candidate["test_changed_rows"]) <= 25
        ):
            upload_recommendation = "REVIEW_LOW_CHURN_V289_TERMINAL03"

    report = _json_safe(
        {
            "version": "V289",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "allowed_actions": TERMINAL_ACTIONS,
            "best_candidate": best,
            "generated_submissions": generated,
            "upload_recommendation": upload_recommendation,
            "copied_to_upload_or_selected": False,
        }
    )
    (OUTDIR / "v289_terminal03_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    md = [
        "# V289 terminal action 0/3 specialist",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        f"Best candidate: `{best.get('candidate', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Terminal mean delta: {float(best.get('terminal_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Upload recommendation: {upload_recommendation}",
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v289_terminal03_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
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
