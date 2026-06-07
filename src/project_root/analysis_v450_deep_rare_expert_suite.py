"""V450 deeper rare-class expert suite.

Trains binary rare-class experts for selected action and point families. This
module writes score tables and diagnostics only; V450 intentionally exports no
submissions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v416_external_embedding_aicup_finetune import build_test_rows, build_train_transition_rows
from analysis_v433_weak_class_expert_bank import SUBMISSION_COLUMNS, build_feature_matrix


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OUTDIR = ROOT / "v450_deep_rare_expert_suite"
EXPECTED_TEST_ROWS = 1845

FORBIDDEN_EXACT_EXPORT_COLUMNS = {
    "target_actionId",
    "target_pointId",
    "target_serverGetPoint",
    "external_actionId",
    "external_pointId",
    "external_serverGetPoint",
    "old_server_actionId",
    "old_server_pointId",
    "old_server_serverGetPoint",
}


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    target: str
    positive_labels: tuple[int, ...]
    candidate_value: int
    description: str
    min_strike: int | None = None
    max_strike: int | None = None
    require_anchor_values: tuple[int, ...] | None = None


class ConstantBinaryModel:
    def __init__(self, positive_probability: float):
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(x), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
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
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def build_expert_specs() -> list[ExpertSpec]:
    """Return the fixed V450 action and point expert families."""

    return [
        ExpertSpec(
            "rare_action_8_9_12_14",
            "action",
            (8, 9, 12, 14),
            8,
            "rare control/defense action family",
        ),
        ExpertSpec(
            "terminal_action_0",
            "action",
            (0,),
            0,
            "terminal or unknown action detector",
        ),
        ExpertSpec(
            "control_attack_transition",
            "action",
            (8, 9, 10, 11, 12, 14),
            10,
            "control-to-attack transition detector",
            min_strike=3,
        ),
        ExpertSpec(
            "defense_recovery_13_14",
            "action",
            (13, 14),
            13,
            "defense recovery action detector",
        ),
        ExpertSpec(
            "long_point_7_8_9",
            "point",
            (7, 8, 9),
            8,
            "long point depth detector",
        ),
        ExpertSpec(
            "half_boundary_4_6",
            "point",
            (4, 6),
            4,
            "half-court boundary side detector",
        ),
        ExpertSpec(
            "short_point_1_2_3",
            "point",
            (1, 2, 3),
            2,
            "short point depth detector",
        ),
        ExpertSpec(
            "point0_removal_detector",
            "point",
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            8,
            "nonzero replacement detector for point-0 anchor rows",
            require_anchor_values=(0,),
        ),
    ]


def group_expert_specs(specs: list[ExpertSpec]) -> dict[str, list[ExpertSpec]]:
    grouped: dict[str, list[ExpertSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.target, []).append(spec)
    return grouped


def _target_column(frame: pd.DataFrame, target: str) -> str:
    for col in (f"target_{target}Id", f"{target}Id"):
        if col in frame.columns:
            return col
    raise ValueError(f"frame missing {target} target column")


def _anchor_column(frame: pd.DataFrame, target: str) -> str | None:
    for col in (f"anchor_{target}Id", f"v362_{target}Id", f"{target}Id"):
        if col in frame.columns:
            return col
    return None


def _spec_context_mask(frame: pd.DataFrame, spec: ExpertSpec) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    if spec.min_strike is not None and "strikeNumber" in frame.columns:
        strike = pd.to_numeric(frame["strikeNumber"], errors="coerce").fillna(0)
        mask &= strike.ge(spec.min_strike).to_numpy()
    if spec.max_strike is not None and "strikeNumber" in frame.columns:
        strike = pd.to_numeric(frame["strikeNumber"], errors="coerce").fillna(0)
        mask &= strike.le(spec.max_strike).to_numpy()
    if spec.require_anchor_values is not None:
        anchor_col = _anchor_column(frame, spec.target)
        if anchor_col is None:
            return np.zeros(len(frame), dtype=bool)
        anchor = pd.to_numeric(frame[anchor_col], errors="coerce").fillna(-9999).astype(int)
        mask &= anchor.isin(spec.require_anchor_values).to_numpy()
    return mask


def _splitter(y_binary: np.ndarray, groups: pd.Series | None, *, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y_binary = np.asarray(y_binary, dtype=int)
    indices = np.arange(len(y_binary), dtype=int)
    if len(indices) < 3:
        return [(indices, indices)]
    if groups is not None and groups.nunique(dropna=True) >= 2:
        splits = int(min(max(2, n_splits), groups.nunique(dropna=True)))
        return list(GroupKFold(n_splits=splits).split(np.zeros(len(y_binary)), y_binary, groups))
    counts = pd.Series(y_binary).value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        splits = int(min(max(2, n_splits), counts.min()))
        return list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state).split(np.zeros(len(y_binary)), y_binary))
    return [(indices, indices)]


def _mask_feature_columns(x: pd.DataFrame, *, probability: float, random_state: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    probability = float(np.clip(probability, 0.0, 0.85))
    if probability <= 0.0 or x.empty:
        return x.reset_index(drop=True).copy(), {"train_only": True, "mask_probability": probability, "masked_columns": 0}
    rng = np.random.default_rng(random_state)
    keep = rng.random(len(x.columns)) >= probability
    if not keep.any():
        keep[int(rng.integers(0, len(keep)))] = True
    out = x.reset_index(drop=True).copy()
    dropped = [col for col, keep_col in zip(out.columns, keep) if not keep_col]
    out.loc[:, dropped] = 0.0
    return out, {"train_only": True, "mask_probability": probability, "masked_columns": int(len(dropped))}


def _smote_like_bootstrap(
    x: pd.DataFrame,
    y_binary: np.ndarray,
    *,
    multiplier: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    x_base = x.reset_index(drop=True).copy()
    y_base = np.asarray(y_binary, dtype=int)
    multiplier = max(1, int(multiplier))
    positive_idx = np.flatnonzero(y_base == 1)
    if multiplier <= 1 or len(positive_idx) == 0:
        return x_base, y_base, {
            "train_only": True,
            "smote_like_bootstrap": True,
            "synthetic_rows": 0,
            "output_train_rows": int(len(x_base)),
        }
    rng = np.random.default_rng(random_state)
    synthetic_count = int(len(positive_idx) * (multiplier - 1))
    values = x_base.to_numpy(dtype=float, copy=True)
    synthetic = np.zeros((synthetic_count, values.shape[1]), dtype=float)
    for row_id in range(synthetic_count):
        left = int(rng.choice(positive_idx))
        right = int(rng.choice(positive_idx))
        lam = float(rng.uniform(0.1, 0.9))
        synthetic[row_id] = lam * values[left] + (1.0 - lam) * values[right]
    x_aug = pd.concat([x_base, pd.DataFrame(synthetic, columns=x_base.columns)], ignore_index=True)
    y_aug = np.concatenate([y_base, np.ones(synthetic_count, dtype=int)])
    order = rng.permutation(len(y_aug))
    return x_aug.iloc[order].reset_index(drop=True), y_aug[order], {
        "train_only": True,
        "smote_like_bootstrap": True,
        "synthetic_rows": int(synthetic_count),
        "output_train_rows": int(len(y_aug)),
    }


def _model_specs(random_state: int, *, quick: bool) -> list[tuple[str, Any]]:
    trees = 80 if quick else 160
    return [
        (
            "logistic",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state),
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=trees,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        ),
    ]


def _fit_model(model: Any, x: pd.DataFrame, y_binary: np.ndarray) -> Any:
    y_binary = np.asarray(y_binary, dtype=int)
    if len(np.unique(y_binary)) < 2:
        return ConstantBinaryModel(float(y_binary.mean()) if len(y_binary) else 0.0)
    return model.fit(x, y_binary)


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["logisticregression"].classes_
    classes = np.asarray(classes, dtype=int)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return np.asarray(prob[:, int(np.flatnonzero(classes == 1)[0])], dtype=float)


def _binary_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    pred = (score >= 0.5).astype(int)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )
    auc = 0.0
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, score))
        except ValueError:
            auc = 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "mean_score": float(np.mean(score)) if len(score) else 0.0,
        "auc": auc,
    }


def _candidate_values(y_target: pd.Series, labels: tuple[int, ...], fallback: int) -> int:
    observed = y_target[pd.to_numeric(y_target, errors="coerce").isin(labels)]
    if observed.empty:
        return int(fallback)
    return int(observed.value_counts().sort_values(ascending=False).index[0])


def _train_single_expert(
    spec: ExpertSpec,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    n_splits: int,
    quick: bool,
    random_state: int,
) -> dict[str, Any]:
    target_col = _target_column(train_rows, spec.target)
    y_target = pd.to_numeric(train_rows[target_col], errors="coerce").fillna(-9999).astype(int)
    train_context = _spec_context_mask(train_rows, spec)
    y_binary = (y_target.isin(spec.positive_labels).to_numpy() & train_context).astype(int)
    test_context = _spec_context_mask(test_rows, spec)
    groups = train_rows["match"] if "match" in train_rows.columns else train_rows.get("rally_uid")
    if groups is not None:
        groups = groups.reset_index(drop=True)

    model_defs = _model_specs(random_state, quick=quick)
    model_oof_scores: list[np.ndarray] = []
    model_test_scores: list[np.ndarray] = []
    report_rows: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []

    for model_id, (model_name, _model) in enumerate(model_defs, start=1):
        oof_score = np.zeros(len(x_train), dtype=float)
        fold_metrics: list[dict[str, Any]] = []
        for fold_id, (tr_idx, va_idx) in enumerate(_splitter(y_binary, groups, n_splits=n_splits, random_state=random_state), start=1):
            x_fold, y_fold, smote_report = _smote_like_bootstrap(
                x_train.iloc[tr_idx],
                y_binary[tr_idx],
                multiplier=2,
                random_state=random_state + model_id * 100 + fold_id,
            )
            x_fold, mask_report = _mask_feature_columns(
                x_fold,
                probability=0.10,
                random_state=random_state + model_id * 1000 + fold_id,
            )
            model = _model_specs(random_state + model_id * 100 + fold_id, quick=quick)[model_id - 1][1]
            model = _fit_model(model, x_fold.reindex(columns=x_train.columns, fill_value=0.0), y_fold)
            fold_score = _positive_probability(model, x_train.iloc[va_idx].reindex(columns=x_train.columns, fill_value=0.0))
            oof_score[va_idx] = np.clip(fold_score, 0.0, 1.0)
            fold_metrics.append(_binary_metrics(y_binary[va_idx], oof_score[va_idx]))
            prep_rows.append(
                {
                    "expert": spec.name,
                    "model": model_name,
                    "fold": fold_id,
                    "train_only": True,
                    "original_train_rows": int(len(tr_idx)),
                    "validation_rows_unchanged": int(len(va_idx)),
                    "test_rows_unchanged": int(len(x_test)),
                    **smote_report,
                    **mask_report,
                }
            )

        x_full, y_full, smote_report = _smote_like_bootstrap(
            x_train,
            y_binary,
            multiplier=2,
            random_state=random_state + model_id * 7000,
        )
        x_full, mask_report = _mask_feature_columns(
            x_full,
            probability=0.10,
            random_state=random_state + model_id * 9000,
        )
        final_model = _fit_model(_model, x_full.reindex(columns=x_train.columns, fill_value=0.0), y_full)
        test_score = _positive_probability(final_model, x_test.reindex(columns=x_train.columns, fill_value=0.0))
        test_score = np.clip(test_score, 0.0, 1.0)
        test_score[~test_context] = 0.0
        model_oof_scores.append(np.clip(oof_score, 0.0, 1.0))
        model_test_scores.append(test_score)
        metrics = _binary_metrics(y_binary, oof_score)
        report_rows.append(
            {
                "expert": spec.name,
                "target": spec.target,
                "model": model_name,
                "labels": " ".join(str(v) for v in spec.positive_labels),
                "description": spec.description,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "support": metrics["support"],
                "positive_rate": metrics["positive_rate"],
                "mean_oof_score": metrics["mean_score"],
                "mean_test_score": float(np.mean(test_score)) if len(test_score) else 0.0,
                "folds": int(n_splits),
                "train_context_rows": int(train_context.sum()),
                "test_context_rows": int(test_context.sum()),
                "train_only_class_weight": True,
                "train_only_smote_like_bootstrap": True,
                "train_only_feature_masking": True,
            }
        )
        prep_rows.append(
            {
                "expert": spec.name,
                "model": model_name,
                "fold": "full_train",
                "train_only": True,
                "original_train_rows": int(len(x_train)),
                "validation_rows_unchanged": 0,
                "test_rows_unchanged": int(len(x_test)),
                **smote_report,
                **mask_report,
            }
        )

    oof_ensemble = np.mean(model_oof_scores, axis=0) if model_oof_scores else np.zeros(len(x_train), dtype=float)
    test_ensemble = np.mean(model_test_scores, axis=0) if model_test_scores else np.zeros(len(x_test), dtype=float)
    test_ensemble[~test_context] = 0.0
    candidate = _candidate_values(y_target, spec.positive_labels, spec.candidate_value)
    return {
        "spec": spec,
        "oof_score": np.clip(oof_ensemble, 0.0, 1.0),
        "test_score": np.clip(test_ensemble, 0.0, 1.0),
        "candidate": candidate,
        "report_rows": report_rows,
        "prep_rows": prep_rows,
    }


def _prepare_real_frames(train_path: Path, test_path: Path, anchor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_raw = pd.read_csv(train_path, low_memory=False)
    test_raw = pd.read_csv(test_path, low_memory=False)
    anchor = pd.read_csv(anchor_path, low_memory=False).loc[:, SUBMISSION_COLUMNS].copy()
    train_rows = build_train_transition_rows(train_raw)
    test_rows = build_test_rows(test_raw, anchor)
    test_rows = test_rows.merge(
        anchor.rename(columns={"actionId": "anchor_actionId", "pointId": "anchor_pointId", "serverGetPoint": "anchor_serverGetPoint"}),
        on="rally_uid",
        how="left",
    )
    train_rows["anchor_actionId"] = train_rows["actionId"]
    train_rows["anchor_pointId"] = train_rows["pointId"]
    if "serverGetPoint" in train_rows.columns:
        train_rows["anchor_serverGetPoint"] = train_rows["serverGetPoint"]
    return train_rows, test_rows, {
        "train_rows_raw": int(len(train_raw)),
        "train_transition_rows": int(len(train_rows)),
        "test_rows_raw": int(len(test_raw)),
        "test_rows": int(len(test_rows)),
        "anchor_rows": int(len(anchor)),
        "anchor_source": str(anchor_path.resolve()),
    }


def _quick_training_sample(train_rows: pd.DataFrame, *, max_rows: int = 7000, random_state: int = 450) -> pd.DataFrame:
    if len(train_rows) <= max_rows:
        return train_rows.reset_index(drop=True)
    specs = build_expert_specs()
    action_col = _target_column(train_rows, "action")
    point_col = _target_column(train_rows, "point")
    rare_action = set().union(*(set(spec.positive_labels) for spec in specs if spec.target == "action"))
    rare_point = set().union(*(set(spec.positive_labels) for spec in specs if spec.target == "point"))
    rare_mask = (
        pd.to_numeric(train_rows[action_col], errors="coerce").isin(rare_action)
        | pd.to_numeric(train_rows[point_col], errors="coerce").isin(rare_point)
    )
    rare_rows = train_rows.loc[rare_mask]
    background_rows = train_rows.loc[~rare_mask]
    rare_take = min(len(rare_rows), max_rows // 2)
    background_take = max_rows - rare_take
    sampled = pd.concat(
        [
            rare_rows.sample(n=rare_take, random_state=random_state) if len(rare_rows) > rare_take else rare_rows,
            background_rows.sample(n=min(len(background_rows), background_take), random_state=random_state + 1),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=random_state + 2).reset_index(drop=True)


def write_expert_outputs(
    action_scores: pd.DataFrame,
    point_scores: pd.DataFrame,
    oof_report: pd.DataFrame,
    summary: dict[str, Any],
    *,
    outdir: Path = OUTDIR,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "action_expert_scores_test": outdir / "action_expert_scores_test.csv",
        "point_expert_scores_test": outdir / "point_expert_scores_test.csv",
        "expert_oof_report": outdir / "expert_oof_report.csv",
        "expert_summary": outdir / "expert_summary.json",
    }
    for frame_name, frame in (("action", action_scores), ("point", point_scores), ("oof", oof_report)):
        forbidden = FORBIDDEN_EXACT_EXPORT_COLUMNS.intersection(frame.columns)
        if forbidden:
            raise ValueError(f"{frame_name} export contains forbidden exact columns: {sorted(forbidden)}")
    action_scores.to_csv(paths["action_expert_scores_test"], index=False)
    point_scores.to_csv(paths["point_expert_scores_test"], index=False)
    oof_report.to_csv(paths["expert_oof_report"], index=False)
    write_json(paths["expert_summary"], summary)
    return paths


def run_pipeline(
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    anchor_path: Path = ANCHOR_PATH,
    outdir: Path = OUTDIR,
    quick: bool = False,
) -> dict[str, Any]:
    train_rows, test_rows, metadata = _prepare_real_frames(train_path, test_path, anchor_path)
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"test score table rows would be {len(test_rows)}, expected {EXPECTED_TEST_ROWS}")
    if quick:
        available = len(train_rows)
        train_rows = _quick_training_sample(train_rows, max_rows=7000, random_state=450)
        metadata["quick_train_transition_rows_used"] = int(len(train_rows))
        metadata["quick_train_transition_rows_available"] = int(available)

    x_train, x_test, feature_columns = build_feature_matrix(train_rows, test_rows)
    specs = build_expert_specs()
    grouped = group_expert_specs(specs)
    n_splits = 2 if quick else 5
    action_scores = pd.DataFrame({"row_id": np.arange(len(test_rows), dtype=int), "rally_uid": test_rows["rally_uid"].to_numpy()})
    point_scores = action_scores.copy()
    report_rows: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []

    for spec_id, spec in enumerate(specs, start=1):
        result = _train_single_expert(
            spec,
            x_train,
            x_test,
            train_rows,
            test_rows,
            n_splits=n_splits,
            quick=quick,
            random_state=450 + spec_id * 17,
        )
        destination = action_scores if spec.target == "action" else point_scores
        destination[f"{spec.name}_score"] = result["test_score"]
        destination[f"{spec.name}_candidate_{spec.target}Id"] = int(result["candidate"])
        destination[f"{spec.name}_positive_labels"] = " ".join(str(v) for v in spec.positive_labels)
        report_rows.extend(result["report_rows"])
        prep_rows.extend(result["prep_rows"])

    oof_report = pd.DataFrame(report_rows).sort_values(["target", "expert", "f1"], ascending=[True, True, False]).reset_index(drop=True)
    summary = {
        "version": "V450",
        "quick": bool(quick),
        "expert_count": int(len(specs)),
        "action_experts": [spec.name for spec in grouped.get("action", [])],
        "point_experts": [spec.name for spec in grouped.get("point", [])],
        "metadata": metadata,
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "action_score_rows": int(len(action_scores)),
        "point_score_rows": int(len(point_scores)),
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "score_tables_row_check": {
            "action": int(len(action_scores) == EXPECTED_TEST_ROWS),
            "point": int(len(point_scores) == EXPECTED_TEST_ROWS),
        },
        "train_only_controls": {
            "class_weight": True,
            "smote_like_bootstrap": True,
            "feature_masking": True,
            "synthetic_test_rows": 0,
        },
        "forbidden_exact_export_columns": sorted(FORBIDDEN_EXACT_EXPORT_COLUMNS),
        "forbidden_exact_columns_exported": [],
        "submission_exports": 0,
        "prep_reports": prep_rows,
    }
    paths = write_expert_outputs(action_scores, point_scores, oof_report, summary, outdir=outdir)
    summary["outputs"] = {key: str(path.resolve()) for key, path in paths.items()}
    write_json(outdir / "expert_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use a bounded train sample and two folds")
    args = parser.parse_args()
    summary = run_pipeline(quick=args.quick)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
