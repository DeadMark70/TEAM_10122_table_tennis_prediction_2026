"""V442 intent-first point fine-tune and residual packager."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS, safe_output_path, validate_submission_schema
from analysis_v419_intent_first_point_finetune import build_test_rows, build_train_transition_rows
from analysis_v435_residual_packager import apply_ranked_candidates


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v442_intent_first_sequence_point"

NUMERIC_COLUMNS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "action_confidence",
]
LEAK_COLUMNS = {"target_actionId", "target_pointId", "target_serverGetPoint", "serverGetPoint"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def action_intent(action_id: Any) -> str:
    value = _int_or_none(action_id)
    if value is None:
        return "unknown"
    if value == 0:
        return "terminal"
    if value in {1, 2, 5, 6, 7}:
        return "drive"
    if value in {3, 14}:
        return "attack"
    if value in {4, 8, 9, 10, 11}:
        return "control"
    if value in {12, 13}:
        return "defense"
    if value in {15, 16, 17, 18}:
        return "serve"
    return "other"


def point_depth(point_id: Any) -> str:
    value = _int_or_none(point_id)
    if value is None or value == 0:
        return "terminal_or_unknown"
    if value in {1, 2, 3}:
        return "short"
    if value in {4, 5, 6}:
        return "half"
    if value in {7, 8, 9}:
        return "long"
    return "unknown"


def point_side(point_id: Any) -> str:
    value = _int_or_none(point_id)
    if value is None or value == 0:
        return "unknown"
    if value in {1, 4, 7}:
        return "left"
    if value in {2, 5, 8}:
        return "middle"
    if value in {3, 6, 9}:
        return "right"
    return "unknown"


def condition_point_features_on_action_intent(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "pred_action" not in data.columns:
        data["pred_action"] = data.get("actionId", 0)
    data["pred_intent"] = data["pred_action"].map(action_intent)
    data["current_depth"] = data.get("pointId", pd.Series([0] * len(data), index=data.index)).map(point_depth)
    data["current_side"] = data.get("pointId", pd.Series([0] * len(data), index=data.index)).map(point_side)
    keep_numeric = [col for col in NUMERIC_COLUMNS if col in data.columns and col not in LEAK_COLUMNS]
    features = data.loc[:, keep_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    categorical = pd.get_dummies(
        data[["pred_intent", "current_depth", "current_side"]].astype(str),
        prefix=["pred_intent", "point_depth", "point_side"],
        dtype=float,
    )
    out = pd.concat([features.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    return out.drop(columns=[col for col in LEAK_COLUMNS if col in out.columns], errors="ignore")


def package_point_residual_candidates(
    anchor: pd.DataFrame,
    proposals: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out, report = apply_ranked_candidates(
        anchor.loc[:, SUBMISSION_COLUMNS].copy(),
        proposals,
        target_col="pointId",
        candidate_col="candidate_pointId",
        max_changes=top_k,
        allow_point0_additions=False,
    )
    validate_submission_schema(out, expected_rows=None if len(out) != 1845 else 1845)
    return out, report


def _splitter(y: np.ndarray, groups: pd.Series | None, seed: int = 442):
    y = np.asarray(y, dtype=int)
    if groups is not None and groups.nunique(dropna=True) >= 3:
        return list(GroupKFold(n_splits=min(3, groups.nunique(dropna=True))).split(np.zeros(len(y)), y, groups))
    counts = pd.Series(y).value_counts()
    if len(counts) and counts.min() >= 2:
        return list(StratifiedKFold(n_splits=min(3, int(counts.min())), shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    idx = np.arange(len(y))
    return [(idx, idx)]


def _fit_classifier(x: pd.DataFrame, y: np.ndarray):
    if len(np.unique(y)) == 1:
        value = int(np.unique(y)[0])

        class _Const:
            classes_ = np.array([value])

            def predict_proba(self, x_new):
                return np.ones((len(x_new), 1), dtype=float)

        return _Const()
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=250, class_weight="balanced", random_state=442, n_jobs=-1),
    ).fit(x, y)


def _aligned_proba(model: Any, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(x)
    model_classes = getattr(model, "classes_", None)
    if model_classes is None and hasattr(model, "named_steps"):
        model_classes = model.named_steps["logisticregression"].classes_
    out = np.zeros((len(x), len(classes)), dtype=float)
    lookup = {int(label): idx for idx, label in enumerate(classes)}
    for j, label in enumerate(model_classes):
        if int(label) in lookup:
            out[:, lookup[int(label)]] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    out[row_sum.ravel() <= 0, :] = 1.0 / len(classes)
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


def _load_v432_pred_action(anchor: pd.DataFrame) -> pd.DataFrame:
    candidates = sorted((ROOT / "v432_aicup_exact_model_zoo_finetune").glob("test_action_probs_*.csv"))
    if not candidates:
        return pd.DataFrame({"rally_uid": anchor["rally_uid"], "pred_action": anchor["actionId"], "action_confidence": 0.55})
    frame = pd.read_csv(candidates[0])
    if "pred_action" not in frame.columns and "pred_actionId" in frame.columns:
        frame["pred_action"] = frame["pred_actionId"]
    aligned = anchor[["rally_uid"]].merge(frame, on="rally_uid", how="left")
    aligned["pred_action"] = pd.to_numeric(aligned.get("pred_action", anchor["actionId"]), errors="coerce").fillna(anchor["actionId"]).astype(int)
    aligned["action_confidence"] = pd.to_numeric(aligned.get("action_confidence", 0.55), errors="coerce").fillna(0.55)
    return aligned[["rally_uid", "pred_action", "action_confidence"]]


def run_pipeline(*, quick: bool = False) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_schema(anchor)
    transitions = build_train_transition_rows(train)
    if quick and len(transitions) > 8000:
        transitions = transitions.sample(n=8000, random_state=442).sort_index().reset_index(drop=True)
    # Train action intent proxy from observed current row -> next action.
    action_x = condition_point_features_on_action_intent(transitions.assign(pred_action=transitions["actionId"], action_confidence=0.75))
    y_action = transitions["target_actionId"].astype(int).to_numpy()
    action_classes = sorted(np.unique(y_action).astype(int).tolist())
    action_model = _fit_classifier(action_x, y_action)
    action_proba = _aligned_proba(action_model, action_x, action_classes)
    pred_action = np.array([action_classes[i] for i in action_proba.argmax(axis=1)], dtype=int)
    action_conf = action_proba.max(axis=1)

    point_rows = transitions.assign(pred_action=pred_action, action_confidence=action_conf)
    point_x = condition_point_features_on_action_intent(point_rows)
    y_point = transitions["target_pointId"].astype(int).to_numpy()
    point_classes = sorted(np.unique(y_point).astype(int).tolist())
    splits = _splitter(y_point, transitions.get("match"))
    oof = np.zeros((len(point_x), len(point_classes)), dtype=float)
    for train_idx, val_idx in splits:
        model = _fit_classifier(point_x.iloc[train_idx], y_point[train_idx])
        oof[val_idx] = _aligned_proba(model, point_x.iloc[val_idx], point_classes)
    pd.DataFrame(oof, columns=[f"point_prob_{c}" for c in point_classes]).to_csv(OUTDIR / "oof_point_probs_intent_first.csv", index=False)
    np.save(OUTDIR / "oof_point_probs_intent_first.npy", oof)

    full_model = _fit_classifier(point_x, y_point)
    test_rows = build_test_rows(test, anchor)
    pred_action_test = _load_v432_pred_action(anchor)
    test_rows = test_rows.merge(pred_action_test, on="rally_uid", how="left")
    test_rows["pred_action"] = pd.to_numeric(test_rows["pred_action"], errors="coerce").fillna(test_rows["actionId"]).astype(int)
    test_rows["action_confidence"] = pd.to_numeric(test_rows["action_confidence"], errors="coerce").fillna(0.55)
    test_x = condition_point_features_on_action_intent(test_rows).reindex(columns=point_x.columns, fill_value=0.0)
    test_prob = _aligned_proba(full_model, test_x, point_classes)
    test_pred = np.array([point_classes[i] for i in test_prob.argmax(axis=1)], dtype=int)
    test_conf = test_prob.max(axis=1)
    test_margin = np.sort(test_prob, axis=1)[:, -1] - np.sort(test_prob, axis=1)[:, -2]
    test_prob_frame = pd.DataFrame(test_prob, columns=[f"point_prob_{c}" for c in point_classes])
    test_prob_frame.insert(0, "rally_uid", anchor["rally_uid"].values)
    test_prob_frame["candidate_pointId"] = test_pred
    test_prob_frame["point_confidence"] = test_conf
    test_prob_frame["point_margin"] = test_margin
    test_prob_frame.to_csv(OUTDIR / "test_point_probs_intent_first.csv", index=False)
    np.save(OUTDIR / "test_point_probs_intent_first.npy", test_prob)

    proposals = test_prob_frame[["rally_uid", "candidate_pointId", "point_confidence", "point_margin"]].copy()
    proposals["utility"] = proposals["point_confidence"] + proposals["point_margin"]
    proposals = proposals.merge(anchor[["rally_uid", "pointId"]], on="rally_uid", how="left")
    proposals = proposals.loc[proposals["candidate_pointId"].astype(int) != proposals["pointId"].astype(int)]
    proposals = proposals.sort_values("utility", ascending=False).drop(columns=["pointId"])
    proposals.to_csv(OUTDIR / "point_candidate_table.csv", index=False)
    reports = []
    for top_k in (5, 10):
        submission, report = package_point_residual_candidates(anchor, proposals, top_k=top_k)
        filename = f"submission_v442_point_top{top_k}__v362anchor.csv"
        submission.to_csv(safe_output_path(OUTDIR, filename), index=False)
        report["filename"] = filename
        reports.append(report)
    summary = {
        "version": "V442",
        "quick": quick,
        "train_rows": int(len(transitions)),
        "point_classes": point_classes,
        "candidate_rows": int(len(proposals)),
        "exports": [r["filename"] for r in reports],
        "reports": reports,
    }
    write_json(OUTDIR / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(quick=args.quick), indent=2))


if __name__ == "__main__":
    main()
