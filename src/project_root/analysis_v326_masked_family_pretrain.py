"""V326 masked family pretraining.

Builds a clean canonical external event table, trains a lightweight masked
coarse-family reconstruction model, and projects AICUP train/test prefixes into
coarse family representation features. It does not read TTMATCH contents, map
external labels to exact AICUP actionId values, or write submission files.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
V255_CANONICAL = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test_new.csv"
OUTDIR = ROOT / "v326_masked_family_pretrain"

CANONICAL_COLUMNS = [
    "corpus",
    "sequence_id",
    "step_idx",
    "phase_code",
    "coarse_family",
    "landing_depth",
    "landing_side",
    "has_spin",
    "has_speed",
    "source_weight",
]

MODEL_FEATURE_COLUMNS = [
    "prev_family",
    "next_family",
    "phase_code",
    "step_bin",
    "landing_depth",
    "landing_side",
    "has_spin",
    "has_speed",
    "corpus",
]

FAMILY_ORDER = ["Zero", "Serve", "Attack", "Control", "Defensive", "Unknown"]
MASK_TOKEN = "__masked__"

SOURCE_WEIGHTS = {
    "openttgames": 1.0,
    "CoachAI-Projects-main": 0.65,
    "DeepMindrobottabletennis": 0.45,
    "sonytabletennis": 0.45,
    "TT3D": 0.45,
    "TT-MatchDynamics": 0.40,
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def contains_ttmatch(value: Any) -> bool:
    return "TTMATCH" in str(value).upper().replace("\\", "/").split("/")


def phase_code_from_step(step_number: int | float) -> str:
    try:
        step = int(step_number)
    except Exception:
        return "unknown"
    if step <= 1:
        return "serve_like"
    if step == 2:
        return "receive_like"
    if step == 3:
        return "third_ball_like"
    if step == 4:
        return "fourth_ball_like"
    return "rally_like"


def step_bin(step_idx: int | float) -> str:
    try:
        step = int(step_idx)
    except Exception:
        return "unknown"
    if step <= 0:
        return "0"
    if step == 1:
        return "1"
    if step == 2:
        return "2"
    if step == 3:
        return "3"
    if step <= 6:
        return "4_6"
    return "7_plus"


def normalize_phase(value: Any, step_idx: int | float) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    if text and text not in {"nan", "none", "<na>"}:
        return text
    return phase_code_from_step(int(step_idx) + 1)


def normalize_family(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>", "unknown"}:
        return "Unknown"
    if any(key in text for key in ["serve", "service"]):
        return "Serve"
    if any(key in text for key in ["zero", "terminal", "net", "error"]):
        return "Zero"
    if any(key in text for key in ["defen", "clear", "lob", "chop"]):
        return "Defensive"
    if any(key in text for key in ["attack", "smash", "drive", "topspin", "loop", "kill"]):
        return "Attack"
    if any(key in text for key in ["control", "push", "drop", "short", "bounce", "receive", "block"]):
        return "Control"
    title = str(value).strip().title()
    return title if title in FAMILY_ORDER else "Unknown"


def action_id_to_family(action_id: Any) -> str:
    try:
        action = int(action_id)
    except Exception:
        return "Unknown"
    if action in {15, 16, 17, 18}:
        return "Serve"
    if action in {1, 2, 3, 4, 5, 6, 7, 10, 11, 13}:
        return "Attack"
    if action in {8, 9}:
        return "Control"
    if action in {12, 14}:
        return "Defensive"
    if action == 0:
        return "Zero"
    return "Unknown"


def _bucket_existing(value: Any, allowed: set[str]) -> str | None:
    text = str(value).strip().lower()
    if text in allowed:
        return text
    return None


def _numeric_buckets(values: pd.Series, labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(["unknown"] * len(values), index=values.index, dtype=object)
    valid = numeric.dropna()
    if valid.empty:
        return out
    q1, q2 = valid.quantile([1 / 3, 2 / 3]).tolist()
    if not (math.isfinite(float(q1)) and math.isfinite(float(q2))) or q1 == q2:
        out.loc[numeric.notna()] = labels[1]
        return out
    out.loc[numeric <= q1] = labels[0]
    out.loc[(numeric > q1) & (numeric <= q2)] = labels[1]
    out.loc[numeric > q2] = labels[2]
    return out


def landing_depth_series(frame: pd.DataFrame) -> pd.Series:
    if "landing_depth" in frame:
        existing = frame["landing_depth"].map(lambda v: _bucket_existing(v, {"near", "mid", "far", "unknown"}))
        if existing.notna().any():
            return existing.fillna("unknown")
    source = frame["landing_y"] if "landing_y" in frame else pd.Series(np.nan, index=frame.index)
    return _numeric_buckets(source, ["near", "mid", "far"])


def landing_side_series(frame: pd.DataFrame) -> pd.Series:
    if "landing_side" in frame:
        existing = frame["landing_side"].map(lambda v: _bucket_existing(v, {"left", "middle", "right", "unknown"}))
        if existing.notna().any():
            return existing.fillna("unknown")
    source = frame["landing_x"] if "landing_x" in frame else pd.Series(np.nan, index=frame.index)
    return _numeric_buckets(source, ["left", "middle", "right"])


def source_weight(corpus: Any) -> float:
    return float(SOURCE_WEIGHTS.get(str(corpus), 0.25))


def build_canonical_event_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    frame = raw.copy()
    corpus = frame.get("source_dataset", frame.get("source", pd.Series("unknown", index=frame.index))).astype(str)
    source_path = frame.get("source_path", pd.Series("", index=frame.index)).astype(str)
    allowed = ~(corpus.map(contains_ttmatch) | source_path.map(contains_ttmatch))
    frame = frame.loc[allowed].copy()
    corpus = corpus.loc[allowed]
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    if "event_index" in frame:
        step = pd.to_numeric(frame["event_index"], errors="coerce")
    elif "step_idx" in frame:
        step = pd.to_numeric(frame["step_idx"], errors="coerce")
    else:
        seq_source = frame.get("sequence_id", pd.Series("sequence", index=frame.index)).astype(str)
        step = frame.groupby(seq_source, sort=False).cumcount()
    step = step.fillna(0).astype(int)

    sequence_id = frame.get("sequence_id", pd.Series(np.arange(len(frame)), index=frame.index)).astype(str)
    phase_source = frame.get("phase", frame.get("phase_code", pd.Series("", index=frame.index)))
    family_source = frame.get("coarse_family", frame.get("action_family", pd.Series("Unknown", index=frame.index)))

    out = pd.DataFrame(
        {
            "corpus": corpus.to_numpy(dtype=object),
            "sequence_id": sequence_id.to_numpy(dtype=object),
            "step_idx": step.to_numpy(dtype=int),
            "phase_code": [normalize_phase(v, s) for v, s in zip(phase_source, step)],
            "coarse_family": [normalize_family(v) for v in family_source],
            "landing_depth": landing_depth_series(frame).to_numpy(dtype=object),
            "landing_side": landing_side_series(frame).to_numpy(dtype=object),
            "has_spin": pd.to_numeric(frame.get("spin", pd.Series(np.nan, index=frame.index)), errors="coerce")
            .notna()
            .to_numpy(dtype=bool),
            "has_speed": pd.to_numeric(frame.get("speed", pd.Series(np.nan, index=frame.index)), errors="coerce")
            .notna()
            .to_numpy(dtype=bool),
            "source_weight": [source_weight(c) for c in corpus],
        },
        columns=CANONICAL_COLUMNS,
    )
    out = out.sort_values(["corpus", "sequence_id", "step_idx"], kind="mergesort").reset_index(drop=True)
    return out


def load_external_canonical(canonical_path: Path = V255_CANONICAL) -> pd.DataFrame:
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing clean external canonical corpus: {canonical_path}")
    return pd.read_csv(canonical_path, low_memory=False)


def add_sequence_context(events: pd.DataFrame) -> pd.DataFrame:
    out = events.sort_values(["corpus", "sequence_id", "step_idx"], kind="mergesort").reset_index(drop=True).copy()
    grouped = out.groupby(["corpus", "sequence_id"], sort=False)["coarse_family"]
    out["prev_family"] = grouped.shift(1).fillna(MASK_TOKEN)
    out["next_family"] = grouped.shift(-1).fillna(MASK_TOKEN)
    out["step_bin"] = out["step_idx"].map(step_bin)
    return out


def masked_training_frame(events: pd.DataFrame) -> pd.DataFrame:
    base = add_sequence_context(events)
    views = []
    for view_name, mask_prev, mask_next in [
        ("full", False, False),
        ("mask_prev", True, False),
        ("mask_next", False, True),
        ("mask_both", True, True),
    ]:
        part = base.copy()
        if mask_prev:
            part["prev_family"] = MASK_TOKEN
        if mask_next:
            part["next_family"] = MASK_TOKEN
        part["mask_view"] = view_name
        views.append(part)
    return pd.concat(views, ignore_index=True)


def make_family_estimator(random_state: int = 326) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer([("cat", encoder, MODEL_FEATURE_COLUMNS)], remainder="drop")
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _fit_estimator(estimator: Any, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> Any:
    if isinstance(estimator, Pipeline):
        estimator.fit(x, y, clf__sample_weight=sample_weight)
    else:
        estimator.fit(x, y, sample_weight=sample_weight)
    return estimator


def _predict(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(estimator.predict(x), dtype=object)


def _classes(estimator: Any) -> list[str]:
    if hasattr(estimator, "classes_"):
        return [str(v) for v in estimator.classes_]
    if isinstance(estimator, Pipeline) and hasattr(estimator.named_steps.get("clf"), "classes_"):
        return [str(v) for v in estimator.named_steps["clf"].classes_]
    return []


def _predict_proba_aligned(estimator: Any, x: pd.DataFrame, family_order: list[str] = FAMILY_ORDER) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(x), dtype=float)
    classes = _classes(estimator)
    out = np.zeros((len(x), len(family_order)), dtype=float)
    for idx, cls in enumerate(classes):
        if cls in family_order:
            out[:, family_order.index(cls)] = raw[:, idx]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum[:, 0] <= 0
    if missing.any():
        out[missing, family_order.index("Unknown")] = 1.0
        row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


def _metric_payload(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def train_masked_family_model(
    events: pd.DataFrame,
    *,
    random_state: int = 326,
    min_folds: int = 3,
) -> tuple[Any, dict[str, Any], list[str]]:
    if events.empty:
        raise ValueError("Cannot train V326 model on an empty event table")
    train = masked_training_frame(events)
    train = train[train["coarse_family"].notna()].reset_index(drop=True)
    x = train[MODEL_FEATURE_COLUMNS].astype(str)
    y = train["coarse_family"].astype(str)
    sample_weight = pd.to_numeric(train["source_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    labels = sorted(y.unique().tolist())

    if len(labels) < 2:
        final_model: Any = DummyClassifier(strategy="most_frequent")
        _fit_estimator(final_model, x, y, sample_weight)
        oof_pred = _predict(final_model, x)
        metric_mode = "dummy_single_class"
    else:
        min_class_count = int(y.value_counts().min())
        n_splits = min(5, min_class_count, len(y))
        if n_splits >= int(min_folds):
            oof_pred = np.empty(len(y), dtype=object)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_idx, valid_idx in splitter.split(x, y):
                fold_model = make_family_estimator(random_state)
                _fit_estimator(fold_model, x.iloc[train_idx], y.iloc[train_idx], sample_weight[train_idx])
                oof_pred[valid_idx] = _predict(fold_model, x.iloc[valid_idx])
            metric_mode = f"{n_splits}_fold_stratified_cv"
        else:
            fit_model = make_family_estimator(random_state)
            _fit_estimator(fit_model, x, y, sample_weight)
            oof_pred = _predict(fit_model, x)
            metric_mode = "insample_small_data_fallback"
        final_model = make_family_estimator(random_state)
        _fit_estimator(final_model, x, y, sample_weight)

    metrics: dict[str, Any] = {
        "metric_mode": metric_mode,
        "training_rows": int(len(train)),
        "external_event_rows": int(len(events)),
        "overall_accuracy": _metric_payload(y, oof_pred, labels)["accuracy"],
        "overall_macro_f1": _metric_payload(y, oof_pred, labels)["macro_f1"],
        "by_corpus": {},
    }
    for corpus, idx in train.groupby("corpus", sort=True).groups.items():
        indexer = np.asarray(list(idx), dtype=int)
        corpus_labels = sorted(y.iloc[indexer].unique().tolist())
        corpus_metrics = _metric_payload(y.iloc[indexer], oof_pred[indexer], corpus_labels)
        corpus_metrics["rows"] = int(len(indexer))
        metrics["by_corpus"][str(corpus)] = corpus_metrics
    return final_model, metrics, MODEL_FEATURE_COLUMNS.copy()


def _aicup_model_input(prefix: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame(index=prefix.index)
    out["prev_family"] = prefix.get("lag0_actionId", pd.Series(-1, index=prefix.index)).map(action_id_to_family)
    out["next_family"] = MASK_TOKEN
    next_step_number = pd.to_numeric(prefix["prefix_len"], errors="coerce").fillna(0).astype(int) + 1
    out["phase_code"] = next_step_number.map(phase_code_from_step)
    out["step_bin"] = pd.to_numeric(prefix["prefix_len"], errors="coerce").fillna(0).astype(int).map(step_bin)
    out["landing_depth"] = "unknown"
    out["landing_side"] = "unknown"
    spin = pd.to_numeric(prefix.get("lag0_spinId", pd.Series(-1, index=prefix.index)), errors="coerce").fillna(-1)
    out["has_spin"] = spin.gt(0).astype(str)
    out["has_speed"] = "False"
    out["corpus"] = f"AICUP_{split}"
    return out[MODEL_FEATURE_COLUMNS].astype(str)


def build_aicup_prefix_tables(train_raw: pd.DataFrame, test_raw: pd.DataFrame, max_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    from baseline_lgbm import add_role_and_score_features, build_test_prefix_table, build_train_prefix_table, validate_raw_data

    validate_raw_data(train_raw, test_raw)
    train = add_role_and_score_features(train_raw)
    test = add_role_and_score_features(test_raw)
    return build_train_prefix_table(train, max_lag), build_test_prefix_table(test, max_lag)


def _entropy(prob: np.ndarray) -> np.ndarray:
    safe = np.clip(prob, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)


def _feature_frame_for_prefix(prefix: pd.DataFrame, split: str, model: Any) -> pd.DataFrame:
    x = _aicup_model_input(prefix, split)
    prob = _predict_proba_aligned(model, x, FAMILY_ORDER)
    pred_idx = prob.argmax(axis=1)
    out = pd.DataFrame(
        {
            "split": split,
            "rally_uid": prefix["rally_uid"].astype(int).to_numpy(),
            "match": prefix["match"].astype(int).to_numpy(),
            "prefix_len": prefix["prefix_len"].astype(int).to_numpy(),
            "row_key": [
                f"{split}:{int(r)}:{int(m)}:{int(p)}"
                for r, m, p in zip(prefix["rally_uid"], prefix["match"], prefix["prefix_len"])
            ],
            "v326_phase_code": x["phase_code"].to_numpy(dtype=object),
            "v326_step_bin": x["step_bin"].to_numpy(dtype=object),
            "v326_prev_family": x["prev_family"].to_numpy(dtype=object),
            "v326_pred_family": [FAMILY_ORDER[i] for i in pred_idx],
            "v326_pred_family_conf": prob.max(axis=1),
            "v326_family_entropy": _entropy(prob),
        }
    )
    for i, family in enumerate(FAMILY_ORDER):
        out[f"v326_family_p_{family.lower()}"] = prob[:, i]
    return out


def build_aicup_prefix_family_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, model: Any) -> pd.DataFrame:
    train_prefix, test_prefix = build_aicup_prefix_tables(train_raw, test_raw)
    train_features = _feature_frame_for_prefix(train_prefix, "train", model)
    test_features = _feature_frame_for_prefix(test_prefix, "test", model)
    return pd.concat([train_features, test_features], ignore_index=True)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# V326 Masked Family Pretraining",
        "",
        f"Decision: `{report['decision']}`",
        f"External event rows: `{report['external_rows']}`",
        f"Masked training rows: `{report['metrics']['training_rows']}`",
        f"AICUP prefix feature rows: `{report['aicup_feature_rows']}`",
        f"Submissions written: `{report['submissions_written']}`",
        f"TTMATCH content rows read: `{report['ttmatch_content_rows_read']}`",
        "",
        "## Masked Family CV",
        "",
        f"- Mode: `{report['metrics']['metric_mode']}`",
        f"- Accuracy: `{report['metrics']['overall_accuracy']:.6f}`",
        f"- Macro-F1: `{report['metrics']['overall_macro_f1']:.6f}`",
        "",
        "## By Corpus",
        "",
    ]
    for corpus, rec in sorted(report["metrics"]["by_corpus"].items()):
        lines.append(
            f"- `{corpus}`: rows={int(rec['rows'])}, "
            f"accuracy={float(rec['accuracy']):.6f}, macro_f1={float(rec['macro_f1']):.6f}"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- No exact external-to-AICUP actionId mapping is trained.",
            "- AICUP feature export excludes exact `actionId` and `next_actionId` columns.",
            "- No submission, upload candidate, or selected submission files are written.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    outdir: Path,
    events: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    aicup_features: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    events.to_csv(outdir / "v326_external_event_table.csv", index=False)
    with (outdir / "v326_family_model.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_cols,
                "family_order": FAMILY_ORDER,
                "mask_token": MASK_TOKEN,
            },
            f,
        )
    aicup_features.to_csv(outdir / "v326_aicup_prefix_family_features.csv", index=False)
    (outdir / "v326_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    (outdir / "v326_report.md").write_text(markdown_report(report), encoding="utf-8")


def run_pipeline(
    *,
    canonical_path: Path = V255_CANONICAL,
    train_path: Path = TRAIN_CSV,
    test_path: Path = TEST_CSV,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    raw_external = load_external_canonical(canonical_path)
    ttmatch_rows = int(
        raw_external.get("source_dataset", pd.Series("", index=raw_external.index)).map(contains_ttmatch).sum()
        + raw_external.get("source_path", pd.Series("", index=raw_external.index)).map(contains_ttmatch).sum()
    )
    events = build_canonical_event_table(raw_external)
    if events.empty:
        raise ValueError("V326 canonical event table is empty after clean-policy filtering")
    model, metrics, feature_cols = train_masked_family_model(events)
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    aicup_features = build_aicup_prefix_family_features(train_raw, test_raw, model)
    report: dict[str, Any] = {
        "version": "V326",
        "decision": "REPRESENTATION_ONLY_DO_NOT_UPLOAD",
        "canonical_source": rel(Path(canonical_path)),
        "external_rows": int(len(events)),
        "external_corpora": sorted(events["corpus"].astype(str).unique().tolist()),
        "ttmatch_rows": int(ttmatch_rows),
        "ttmatch_content_rows_read": 0,
        "metrics": metrics,
        "aicup_feature_rows": int(len(aicup_features)),
        "aicup_train_feature_rows": int(aicup_features["split"].eq("train").sum()),
        "aicup_test_feature_rows": int(aicup_features["split"].eq("test").sum()),
        "feature_probability_columns": [c for c in aicup_features.columns if c.startswith("v326_family_p_")],
        "submissions_written": 0,
        "upload_or_selected_writes": 0,
        "artifacts": [
            rel(Path(outdir) / "v326_external_event_table.csv"),
            rel(Path(outdir) / "v326_family_model.pkl"),
            rel(Path(outdir) / "v326_aicup_prefix_family_features.csv"),
            rel(Path(outdir) / "v326_report.json"),
            rel(Path(outdir) / "v326_report.md"),
        ],
    }
    write_outputs(Path(outdir), events, model, feature_cols, aicup_features, report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": rel(OUTDIR),
                "external_rows": report["external_rows"],
                "aicup_feature_rows": report["aicup_feature_rows"],
                "accuracy": report["metrics"]["overall_accuracy"],
                "macro_f1": report["metrics"]["overall_macro_f1"],
                "submissions_written": report["submissions_written"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
