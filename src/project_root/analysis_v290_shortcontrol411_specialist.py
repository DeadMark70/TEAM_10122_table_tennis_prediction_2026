"""V290 short-control specialist for actions 4/11.

This is a local diagnostic line over the V261/V173/R121 anchor. Generated
submissions only change actionId; pointId and serverGetPoint are copied from
the V261 anchor.
"""

from __future__ import annotations

import __main__
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from baseline_lgbm import ACTION_CLASSES
from analysis_v286_weak_action_specialist_pretraining import (
    class_f1,
    point_depth,
)
from analysis_v288_specialist_feature_discovery import build_basic_feature_frame


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v290_shortcontrol411_specialist"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
V286_OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
V286_OOF = V286_OUTDIR / "v286_specialist_oof.csv"

SHORTCONTROL_ACTIONS = np.array([4, 11], dtype=int)
PROTECTED_ACTIONS = np.array([1, 10, 12, 13], dtype=int)
DEFAULT_FILTER_BLOCKED_ANCHORS = np.array([10, 12, 13], dtype=int)
SERVE_ACTIONS = np.array([15, 16, 17, 18], dtype=int)
MIN_SCORES = [0.50, 0.55, 0.60, 0.65]
MIN_SUPPORTS = [10, 20, 40]
CAPS = [0.0025, 0.005, 0.010]


def build_shortcontrol_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows.copy()
    for col, value in {
        "lag0_positionId": 0,
        "scoreSelf": 0,
        "scoreOther": 0,
        "scoreTotal": 0,
        "serverScoreDiff": 0,
    }.items():
        if col not in base:
            base[col] = value
    out = build_basic_feature_frame(base)
    out["is_short_depth"] = out["lag0_point_depth"].eq("short").astype(int)
    out["is_receive_short"] = ((out["is_receive"] == 1) & (out["is_short_depth"] == 1)).astype(int)
    out["is_control_incoming"] = out["lag0_action_family"].eq("Control").astype(int)
    out["is_serve_incoming"] = out["lag0_action_family"].eq("Serve").astype(int)
    out["is_backspin_like"] = out["lag0_spin"].isin([2, 3]).astype(int)
    out["lag0_point_depth_code"] = out["lag0_pointId"].map(lambda value: {"zero": 0, "short": 1, "half": 2, "long": 3}[point_depth(value)])
    score = (
        0.35 * out["is_receive_short"]
        + 0.18 * out["is_short_depth"]
        + 0.14 * out["is_third"]
        + 0.12 * out["is_control_incoming"]
        + 0.10 * out["is_backspin_like"]
        + 0.06 * out["is_serve_incoming"]
        + 0.05 * out["is_receive"]
    )
    out["shortcontrol_context_score"] = score.clip(0.0, 1.0)
    return out


def filter_shortcontrol_candidates(
    frame: pd.DataFrame,
    min_score: float,
    min_support: int,
    protected_anchor_actions: Iterable[int] = DEFAULT_FILTER_BLOCKED_ANCHORS,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)
    blocked = {int(x) for x in protected_anchor_actions}
    filtered = frame[
        pd.to_numeric(frame["shortcontrol_score"], errors="coerce").fillna(0.0).ge(float(min_score))
        & pd.to_numeric(frame["support_count"], errors="coerce").fillna(0).ge(int(min_support))
        & frame["candidate_action"].astype(int).isin(SHORTCONTROL_ACTIONS.tolist())
        & frame["candidate_action"].astype(int).ne(frame["anchor_action"].astype(int))
        & ~frame["anchor_action"].astype(int).isin(blocked)
        & ~frame["anchor_action"].astype(int).isin(SERVE_ACTIONS.tolist())
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=frame.columns)
    ranked = filtered.sort_values(["row_id", "shortcontrol_score", "support_count"], ascending=[True, False, False])
    return ranked.groupby("row_id", as_index=False, sort=False).head(1).reset_index(drop=True)


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


def _set_pickle_dataclasses() -> None:
    from analysis_v209_action_selector_reranker import GrUTuning, TransformerTuning, V3Tuning

    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "prefix_len",
        "is_receive",
        "is_third",
        "lag0_pointId",
        "lag0_spin",
        "lag0_strength",
        "lag0_point_depth_code",
        "is_receive_short",
        "shortcontrol_context_score",
    ]
    num = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cats = pd.get_dummies(
        frame[["phase_bin", "lag0_action_family", "lag0_point_depth", "lag0_action_point_pair"]].astype(str),
        dtype=float,
    )
    return pd.concat([num, cats], axis=1).astype(float)


def _align(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out:
            out[col] = 0.0
    return out[columns].astype(float)


def train_shortcontrol_scores(rows: pd.DataFrame, test_rows: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_frame = build_shortcontrol_feature_frame(rows)
    test_frame = build_shortcontrol_feature_frame(test_rows)
    x_all = _feature_matrix(train_frame)
    columns = list(x_all.columns)
    x_test = _align(_feature_matrix(test_frame), columns)
    target = np.isin(np.asarray(y, dtype=int), SHORTCONTROL_ACTIONS).astype(int)
    folds = rows["fold"].astype(int).to_numpy() if "fold" in rows else np.arange(len(rows)) % 5
    oof = np.zeros(len(rows), dtype=float)
    test_sum = np.zeros(len(test_rows), dtype=float)
    fitted = 0
    for fold in sorted(np.unique(folds)):
        valid = folds == fold
        train = ~valid
        if len(np.unique(target[train])) < 2:
            continue
        clf = ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=290 + int(fold),
            n_jobs=1,
        )
        clf.fit(x_all.loc[train, columns], target[train])
        oof[valid] = clf.predict_proba(x_all.loc[valid, columns])[:, 1]
        test_sum += clf.predict_proba(x_test)[:, 1]
        fitted += 1
    if fitted:
        test_sum /= float(fitted)
    else:
        base = float(target.mean()) if len(target) else 0.0
        oof[:] = base
        test_sum[:] = base
    return oof, test_sum


def _support_table(rows: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    frame = build_shortcontrol_feature_frame(rows)
    frame["candidate_action"] = np.asarray(y, dtype=int)
    frame = frame[frame["candidate_action"].isin(SHORTCONTROL_ACTIONS.tolist())].copy()
    if frame.empty:
        return pd.DataFrame(columns=["phase_bin", "lag0_action_family", "lag0_point_depth", "candidate_action", "support_count"])
    return (
        frame.groupby(["phase_bin", "lag0_action_family", "lag0_point_depth", "candidate_action"], dropna=False)
        .size()
        .reset_index(name="support_count")
    )


def _add_support(frame: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    if support.empty:
        out = frame.copy()
        out["support_count"] = 0
        return out
    merged = frame.merge(
        support,
        on=["phase_bin", "lag0_action_family", "lag0_point_depth", "candidate_action"],
        how="left",
    )
    merged["support_count"] = pd.to_numeric(merged["support_count"], errors="coerce").fillna(0).astype(int)
    return merged


def _choose_action(features: pd.DataFrame) -> np.ndarray:
    prefer_11 = (features["is_receive_short"] == 1) | (features["lag0_point_depth"].eq("short"))
    prefer_4 = (features["is_backspin_like"] == 1) & ((features["is_receive"] == 1) | (features["is_third"] == 1))
    return np.where(prefer_4 & ~prefer_11, 4, 11).astype(int)


def build_candidate_frame(rows: pd.DataFrame, anchor: np.ndarray, scores: np.ndarray, support: pd.DataFrame) -> pd.DataFrame:
    features = build_shortcontrol_feature_frame(rows)
    candidates = features[["phase_bin", "lag0_action_family", "lag0_point_depth"]].copy()
    candidates["row_id"] = np.arange(len(features), dtype=int)
    candidates["anchor_action"] = np.asarray(anchor, dtype=int)
    candidates["candidate_action"] = _choose_action(features)
    candidates["model_score"] = np.asarray(scores, dtype=float)
    candidates["context_score"] = features["shortcontrol_context_score"].to_numpy(dtype=float)
    candidates = _add_support(candidates, support)
    max_support = max(float(candidates["support_count"].max()), 1.0)
    support_score = np.log1p(candidates["support_count"].astype(float)) / math.log1p(max_support)
    candidates["shortcontrol_score"] = (
        0.65 * candidates["model_score"].astype(float)
        + 0.25 * candidates["context_score"].astype(float)
        + 0.10 * support_score
    )
    return candidates


def apply_row_cap(anchor: np.ndarray, row_candidates: pd.DataFrame, max_churn: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(anchor, dtype=int).copy()
    selected = np.zeros(len(pred), dtype=bool)
    max_rows = int(math.floor(len(pred) * float(max_churn)))
    if row_candidates.empty or max_rows <= 0:
        return pred, selected
    ranked = row_candidates.sort_values(["shortcontrol_score", "support_count"], ascending=[False, False]).head(max_rows)
    ids = ranked["row_id"].astype(int).to_numpy()
    selected[ids] = True
    pred[ids] = ranked["candidate_action"].astype(int).to_numpy()
    return pred, selected


def _macro(y: np.ndarray, pred: np.ndarray, labels: Iterable[int] = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def _changed_row_report(anchor: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    changed = np.asarray(anchor, dtype=int) != np.asarray(pred, dtype=int)
    report: dict[str, int] = {"changed_rows": int(changed.sum())}
    for action in sorted(set(np.asarray(pred, dtype=int)[changed].tolist())):
        report[f"changed_to_{int(action)}"] = int(np.sum(changed & (np.asarray(pred, dtype=int) == int(action))))
    return report


def evaluate_variant(
    name: str,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    pred_oof: np.ndarray,
    anchor_test: np.ndarray,
    pred_test: np.ndarray,
    max_churn: float,
    min_score: float,
    min_support: int,
) -> dict[str, Any]:
    base = _macro(y, anchor_oof)
    score = _macro(y, pred_oof)
    short_base = _macro(y, anchor_oof, SHORTCONTROL_ACTIONS)
    short_score = _macro(y, pred_oof, SHORTCONTROL_ACTIONS)
    prot_base = _macro(y, anchor_oof, PROTECTED_ACTIONS)
    prot_score = _macro(y, pred_oof, PROTECTED_ACTIONS)
    base_f1 = class_f1(y, anchor_oof, ACTION_CLASSES)
    pred_f1 = class_f1(y, pred_oof, ACTION_CLASSES)
    deltas = {str(k): float(pred_f1[k] - base_f1[k]) for k in ACTION_CLASSES}
    changed_test = pred_test != anchor_test
    d4 = deltas["4"]
    d11 = deltas["11"]
    clean = ((d4 > 0 and d11 > 0) or ((d4 > 0 or d11 > 0) and prot_score - prot_base >= 0)) and changed_test.sum() > 0
    return {
        "candidate": name,
        "threshold": float(min_score),
        "min_support": int(min_support),
        "max_churn": float(max_churn),
        "action_macro_f1": float(score),
        "delta_vs_v173": float(score - base),
        "short_control_mean_delta": float(short_score - short_base),
        "protected_mean_delta": float(prot_score - prot_base),
        "action4_delta": float(d4),
        "action11_delta": float(d11),
        "test_changed_rows": int(changed_test.sum()),
        "test_churn": float(changed_test.mean()),
        "candidate_tier": "clean_probe" if clean else "diagnostic_only",
        "class_f1_delta_json": json.dumps(deltas, sort_keys=True),
        **_changed_row_report(anchor_test, pred_test),
    }


def build_class_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    anchor_f1 = class_f1(y, anchor, ACTION_CLASSES)
    pred_f1 = class_f1(y, pred, ACTION_CLASSES)
    return pd.DataFrame(
        [
            {
                "action": int(action),
                "is_shortcontrol": int(action in SHORTCONTROL_ACTIONS.tolist()),
                "is_protected": int(action in PROTECTED_ACTIONS.tolist()),
                "anchor_f1": float(anchor_f1[action]),
                "v290_f1": float(pred_f1[action]),
                "delta": float(pred_f1[action] - anchor_f1[action]),
            }
            for action in ACTION_CLASSES
        ]
    )


def _cap_token(churn: float) -> str:
    fixed = {0.0025: "0p0025", 0.005: "0p005", 0.010: "0p010"}
    return fixed.get(round(float(churn), 4), f"{float(churn):.4f}".rstrip("0").rstrip(".").replace(".", "p"))


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


def load_anchor_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    _set_pickle_dataclasses()
    from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

    state = rebuild_v173_best_actions()
    rows = state["rows"].reset_index(drop=True).copy()
    test_rows = state["test_rows"].reset_index(drop=True).copy()
    y = rows["next_actionId"].astype(int).to_numpy()
    anchor_oof = np.asarray(state["v173_pred_oof"], dtype=int)
    if V286_OOF.exists():
        oof = pd.read_csv(V286_OOF)
        if len(oof) == len(rows):
            rows["fold"] = oof.get("fold", pd.Series(np.arange(len(rows)) % 5)).astype(int).to_numpy()
            anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    return rows, test_rows, y, anchor_oof


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v290*.csv"):
        stale.unlink()
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing anchor submission: {ANCHOR_SUBMISSION}")

    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    rows, test_rows, y, anchor_oof = load_anchor_frames()
    if len(anchor_sub) != len(test_rows):
        raise ValueError(f"Anchor rows {len(anchor_sub)} do not match test rows {len(test_rows)}")

    scores_oof, scores_test = train_shortcontrol_scores(rows, test_rows, y)
    support = _support_table(rows, y)
    oof_candidates = build_candidate_frame(rows, anchor_oof, scores_oof, support)
    test_candidates = build_candidate_frame(test_rows, anchor_test, scores_test, support)

    records: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for min_score in MIN_SCORES:
        for min_support in MIN_SUPPORTS:
            oof_pool = filter_shortcontrol_candidates(
                oof_candidates,
                min_score=min_score,
                min_support=min_support,
                protected_anchor_actions=PROTECTED_ACTIONS,
            )
            test_pool = filter_shortcontrol_candidates(
                test_candidates,
                min_score=min_score,
                min_support=min_support,
                protected_anchor_actions=PROTECTED_ACTIONS,
            )
            for cap in CAPS:
                pred_oof, _selected = apply_row_cap(anchor_oof, oof_pool, cap)
                pred_test, _test_selected = apply_row_cap(anchor_test, test_pool, cap)
                name = f"v290_shortcontrol_t{str(min_score).replace('.', 'p')}_s{min_support}_c{_cap_token(cap)}"
                records.append(evaluate_variant(name, y, anchor_oof, pred_oof, anchor_test, pred_test, cap, min_score, min_support))
                predictions[name] = pred_oof
                test_predictions[name] = pred_test

    search = pd.DataFrame(records).sort_values(
        ["candidate_tier", "delta_vs_v173", "short_control_mean_delta", "protected_mean_delta", "test_changed_rows"],
        ascending=[True, False, False, False, True],
    )
    search.to_csv(OUTDIR / "v290_shortcontrol_search.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    best_pred = predictions.get(str(best.get("candidate", "")), anchor_oof)
    build_class_report(y, anchor_oof, best_pred).to_csv(OUTDIR / "v290_shortcontrol_class_report.csv", index=False)

    generated = []
    for cap in CAPS:
        cap_rows = search[search["max_churn"].astype(float).eq(float(cap))].copy()
        if cap_rows.empty:
            pred_test = anchor_test.copy()
        else:
            chosen = cap_rows.iloc[0]
            pred_test = test_predictions[str(chosen["candidate"])]
        filename = f"submission_v290_shortcontrol_c{_cap_token(cap)}__pv261cap1__sr121.csv"
        generated.append(str(write_submission(filename, pred_test, anchor_sub).relative_to(ROOT)))

    upload_recommendation = "DO_NOT_UPLOAD"
    clean = search[search["candidate_tier"].eq("clean_probe")].copy()
    if not clean.empty:
        candidate = clean.sort_values(["test_changed_rows", "delta_vs_v173"], ascending=[True, False]).iloc[0]
        if (
            float(candidate["delta_vs_v173"]) >= 0.001
            and float(candidate["protected_mean_delta"]) >= 0
            and 3 <= int(candidate["test_changed_rows"]) <= 25
        ):
            upload_recommendation = "REVIEW_LOW_CHURN_V290_SHORTCONTROL"

    report = _json_safe(
        {
            "version": "V290",
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "allowed_actions": SHORTCONTROL_ACTIONS.tolist(),
            "protected_actions": PROTECTED_ACTIONS.tolist(),
            "best_candidate": best,
            "generated_submissions": generated,
            "upload_recommendation": upload_recommendation,
            "copied_to_upload_or_selected": False,
            "no_ttmatch_or_old_server": True,
        }
    )
    (OUTDIR / "v290_shortcontrol_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    md = [
        "# V290 short-control 4/11 specialist",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Point/server: fixed from V261 anchor",
        "TTMATCH/old-server: not used",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best.get('candidate', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Short-control mean delta: {float(best.get('short_control_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Action 4 delta: {float(best.get('action4_delta', 0.0)):.6f}",
        f"Action 11 delta: {float(best.get('action11_delta', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Upload recommendation: {upload_recommendation}",
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v290_shortcontrol_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
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
