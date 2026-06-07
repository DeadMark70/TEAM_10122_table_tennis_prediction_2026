from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


OUTDIR = Path("v390_synthetic_augmented_scorer")
V385_GRAMMAR = Path("v385_expanded_synthetic_grammar") / "expanded_synthetic_grammar.csv"
V388_DIR = Path("v388_large_synthetic_candidate_pool")
POINT_POOL = V388_DIR / "point_change_pool.csv"
ACTION_POOL = V388_DIR / "action_change_pool.csv"

FEATURE_COLUMNS = [
    "target_action_family",
    "target_point_depth",
    "target_point_side",
    "phase",
    "prefix_len_bin",
    "last_action_family",
    "last_spin",
    "last_strength",
    "same_depth",
    "same_side",
    "same_family",
    "support_count",
    "source_family_count",
    "is_point0_addition",
    "is_serve_15_18_addition",
]

CATEGORICAL_FEATURES = [
    "target_action_family",
    "target_point_depth",
    "target_point_side",
    "phase",
    "prefix_len_bin",
    "last_action_family",
    "last_spin",
    "last_strength",
]

NUMERIC_FEATURES = [
    "same_depth",
    "same_side",
    "same_family",
    "support_count",
    "source_family_count",
    "is_point0_addition",
    "is_serve_15_18_addition",
]

ACTION_FAMILY_BY_ID = {
    1: "attack",
    2: "attack",
    3: "attack",
    4: "receive",
    5: "control",
    6: "control",
    7: "receive",
    8: "defensive",
    9: "defensive",
    10: "defensive",
    11: "control",
    12: "attack",
    13: "setup",
    14: "setup",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

POINT_DEPTH_BY_ID = {
    0: "terminal",
    1: "short",
    2: "short",
    3: "short",
    4: "half",
    5: "half",
    6: "half",
    7: "long",
    8: "long",
    9: "long",
}

POINT_SIDE_BY_ID = {
    0: "terminal",
    1: "left",
    2: "middle",
    3: "right",
    4: "left",
    5: "middle",
    6: "right",
    7: "left",
    8: "middle",
    9: "right",
}


def output_filenames() -> list[str]:
    return [
        "point_augmented_scores.csv",
        "action_augmented_scores.csv",
        "model_report.json",
        "search_report.json",
    ]


def _norm_text(value: Any, default: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().lower()
    return text if text else default


def _norm_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _norm_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _action_family(action_id: Any, fallback: Any = None) -> str:
    text = _norm_text(fallback, default="")
    if text:
        return text
    return ACTION_FAMILY_BY_ID.get(_norm_int(action_id, default=-1), "unknown")


def _point_depth(point_id: Any, fallback: Any = None) -> str:
    text = _norm_text(fallback, default="")
    if text:
        return text
    return POINT_DEPTH_BY_ID.get(_norm_int(point_id, default=-1), "unknown")


def _point_side(point_id: Any, fallback: Any = None) -> str:
    text = _norm_text(fallback, default="")
    if text:
        return text
    return POINT_SIDE_BY_ID.get(_norm_int(point_id, default=-1), "unknown")


def _minimal_synthetic_grammar() -> pd.DataFrame:
    rows = [
        {
            "synthetic_id": "synthetic_v390_fallback_0001",
            "rally_uid": "synthetic_v390_fallback_0001",
            "phase": "rally",
            "prefix_len_bin": "mid_prefix",
            "last_action_family": "attack",
            "last_spin": "topspin",
            "last_strength": "medium",
            "terminal_context": False,
            "target_action_family": "attack",
            "target_action_id_optional": 3,
            "target_point_depth": "long",
            "target_point_side": "right",
            "target_point_id_optional": 9,
            "compatibility_label": "compatible",
            "weight": 1.0,
        },
        {
            "synthetic_id": "synthetic_v390_fallback_0002",
            "rally_uid": "synthetic_v390_fallback_0002",
            "phase": "rally",
            "prefix_len_bin": "short_prefix",
            "last_action_family": "control",
            "last_spin": "backspin",
            "last_strength": "soft",
            "terminal_context": False,
            "target_action_family": "control",
            "target_action_id_optional": 11,
            "target_point_depth": "terminal",
            "target_point_side": "terminal",
            "target_point_id_optional": 0,
            "compatibility_label": "incompatible",
            "weight": 0.35,
        },
        {
            "synthetic_id": "synthetic_v390_fallback_0003",
            "rally_uid": "synthetic_v390_fallback_0003",
            "phase": "receive",
            "prefix_len_bin": "short_prefix",
            "last_action_family": "receive",
            "last_spin": "topspin",
            "last_strength": "medium",
            "terminal_context": False,
            "target_action_family": "serve",
            "target_action_id_optional": 15,
            "target_point_depth": "short",
            "target_point_side": "left",
            "target_point_id_optional": 1,
            "compatibility_label": "incompatible",
            "weight": 0.35,
        },
    ]
    return pd.DataFrame(rows)


def load_synthetic_grammar(root: str | Path = Path(".")) -> tuple[pd.DataFrame, bool]:
    path = Path(root) / V385_GRAMMAR
    if not path.exists():
        return _minimal_synthetic_grammar(), True
    grammar = pd.read_csv(path)
    return grammar, False


def synthetic_features(grammar: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=grammar.index)
    out["row_id"] = grammar.get("synthetic_id", grammar.get("rally_uid", pd.Series(index=grammar.index))).map(
        lambda value: _norm_text(value, default="synthetic_unknown")
    )
    out["rally_uid"] = grammar.get("rally_uid", out["row_id"])
    out["target_action_family"] = grammar.get("target_action_family", "unknown").map(_norm_text)
    out["target_point_depth"] = grammar.get("target_point_depth", "unknown").map(_norm_text)
    out["target_point_side"] = grammar.get("target_point_side", "unknown").map(_norm_text)
    out["phase"] = grammar.get("phase", "unknown").map(_norm_text)
    out["prefix_len_bin"] = grammar.get("prefix_len_bin", "unknown").map(_norm_text)
    out["last_action_family"] = grammar.get("last_action_family", "unknown").map(_norm_text)
    out["last_spin"] = grammar.get("last_spin", "unknown").map(_norm_text)
    out["last_strength"] = grammar.get("last_strength", "unknown").map(_norm_text)
    out["target_action_id"] = grammar.get("target_action_id_optional", -1).map(_norm_int)
    out["target_point_id"] = grammar.get("target_point_id_optional", -1).map(_norm_int)
    out["same_depth"] = (
        out["target_point_depth"].eq("terminal") & grammar.get("terminal_context", False).map(_norm_bool)
    ).astype(int)
    out["same_side"] = out["target_point_side"].ne("unknown").astype(int)
    out["same_family"] = out["target_action_family"].eq(out["last_action_family"]).astype(int)
    out["support_count"] = (pd.to_numeric(grammar.get("weight", 1.0), errors="coerce").fillna(1.0) * 10).round(6)
    out["source_family_count"] = 1
    out["is_point0_addition"] = (
        out["target_point_id"].eq(0) & ~grammar.get("terminal_context", False).map(_norm_bool)
    ).astype(int)
    out["is_serve_15_18_addition"] = (
        out["target_action_id"].isin({15, 16, 17, 18}) & ~out["phase"].isin({"serve", "service"})
    ).astype(int)
    out["label"] = grammar.get("compatibility_label", "compatible").map(_norm_text).isin(
        {"compatible", "true", "1", "yes"}
    )
    return out


class DeterministicLinearModel:
    def __init__(self) -> None:
        self.intercept = 0.0
        self.categorical_log_odds: dict[str, dict[str, float]] = {}
        self.numeric_weights: dict[str, float] = {}

    def fit(self, features: pd.DataFrame, labels: pd.Series, sample_weight: pd.Series | None = None) -> "DeterministicLinearModel":
        labels = labels.astype(int).reset_index(drop=True)
        x = features.reset_index(drop=True)
        weights = (
            pd.Series(1.0, index=x.index)
            if sample_weight is None
            else pd.to_numeric(sample_weight.reset_index(drop=True), errors="coerce").fillna(1.0)
        )
        pos_w = float(weights[labels.eq(1)].sum())
        neg_w = float(weights[labels.eq(0)].sum())
        self.intercept = math.log((pos_w + 1.0) / (neg_w + 1.0))

        for column in CATEGORICAL_FEATURES:
            mapping: dict[str, float] = {}
            for value in sorted(x[column].map(_norm_text).unique()):
                mask = x[column].map(_norm_text).eq(value)
                p = float(weights[mask & labels.eq(1)].sum())
                n = float(weights[mask & labels.eq(0)].sum())
                mapping[value] = math.log((p + 0.5) / (n + 0.5)) * 0.45
            self.categorical_log_odds[column] = mapping

        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(x[column], errors="coerce").fillna(0.0)
            pos_mean = float((values[labels.eq(1)] * weights[labels.eq(1)]).sum() / max(pos_w, 1e-9))
            neg_mean = float((values[labels.eq(0)] * weights[labels.eq(0)]).sum() / max(neg_w, 1e-9))
            self.numeric_weights[column] = _bounded(pos_mean - neg_mean, -2.0, 2.0)
        self.numeric_weights["is_point0_addition"] = min(self.numeric_weights.get("is_point0_addition", 0.0), -0.8)
        self.numeric_weights["is_serve_15_18_addition"] = min(
            self.numeric_weights.get("is_serve_15_18_addition", 0.0), -0.8
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> list[float]:
        scores: list[float] = []
        for _, row in features.iterrows():
            raw = self.intercept
            for column in CATEGORICAL_FEATURES:
                raw += self.categorical_log_odds.get(column, {}).get(_norm_text(row.get(column)), 0.0)
            for column in NUMERIC_FEATURES:
                raw += self.numeric_weights.get(column, 0.0) * _norm_float(row.get(column))
            scores.append(_sigmoid(raw))
        return scores


class SklearnModel:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def predict_proba(self, features: pd.DataFrame) -> list[float]:
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(features[FEATURE_COLUMNS])
            return [float(value) for value in proba[:, 1]]
        decision = self.pipeline.decision_function(features[FEATURE_COLUMNS])
        return [_sigmoid(float(value)) for value in decision]


def fit_augmented_model(training_rows: pd.DataFrame, prefer_sklearn: bool = True) -> tuple[Any, str, str | None]:
    features = training_rows[FEATURE_COLUMNS].copy()
    labels = training_rows["label"].astype(int)
    weights = pd.to_numeric(training_rows.get("support_count", 1.0), errors="coerce").fillna(1.0)

    if prefer_sklearn:
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            preprocessor = ColumnTransformer(
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                    ("num", StandardScaler(), NUMERIC_FEATURES),
                ]
            )
            clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=390)
            pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
            pipeline.fit(features, labels, clf__sample_weight=weights)
            return SklearnModel(pipeline), "sklearn_logistic_regression", None
        except Exception as exc:  # pragma: no cover - exercised when sklearn is unavailable.
            fallback = DeterministicLinearModel().fit(features, labels, weights)
            return fallback, "deterministic_linear_fallback", str(exc)

    fallback = DeterministicLinearModel().fit(features, labels, weights)
    return fallback, "deterministic_linear_fallback", None


def _candidate_features(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(columns=["row_id", "rally_uid", *FEATURE_COLUMNS])

    def series(column: str, default: Any) -> pd.Series:
        if column in out.columns:
            return out[column]
        if isinstance(default, pd.Series):
            return default.reindex(out.index)
        return pd.Series(default, index=out.index)

    out["row_id"] = series("rally_uid", pd.Series(index=out.index)).map(lambda value: str(value))
    out["rally_uid"] = series("rally_uid", out["row_id"])
    candidate_action = series("candidate_action", series("base_action", -1))
    candidate_point = series("candidate_point", series("base_point", -1))
    base_action = series("base_action", candidate_action)
    base_point = series("base_point", candidate_point)

    features = pd.DataFrame(index=out.index)
    features["row_id"] = out["row_id"]
    features["rally_uid"] = out["rally_uid"]
    features["base_action"] = base_action
    features["candidate_action"] = candidate_action
    features["base_point"] = base_point
    features["candidate_point"] = candidate_point
    features["target_action_family"] = [
        _action_family(action, out.iloc[i].get("candidate_family", out.iloc[i].get("target_action_family")))
        for i, action in enumerate(candidate_action)
    ]
    features["target_point_depth"] = [
        _point_depth(point, out.iloc[i].get("candidate_depth", out.iloc[i].get("target_point_depth")))
        for i, point in enumerate(candidate_point)
    ]
    features["target_point_side"] = [
        _point_side(point, out.iloc[i].get("candidate_side", out.iloc[i].get("target_point_side")))
        for i, point in enumerate(candidate_point)
    ]
    features["phase"] = series("phase", "unknown").map(_norm_text)
    features["prefix_len_bin"] = series("prefix_len_bin", "unknown").map(_norm_text)
    features["last_action_family"] = series("last_action_family", series("base_family", "unknown")).map(_norm_text)
    features["last_spin"] = series("last_spin", "unknown").map(_norm_text)
    features["last_strength"] = series("last_strength", "unknown").map(_norm_text)
    base_depth = [_point_depth(value, None) for value in base_point]
    base_side = [_point_side(value, None) for value in base_point]
    base_family = [_action_family(value, None) for value in base_action]
    features["same_depth"] = [
        int(target == base and target != "unknown") for target, base in zip(features["target_point_depth"], base_depth)
    ]
    features["same_side"] = [
        int(target == base and target != "unknown") for target, base in zip(features["target_point_side"], base_side)
    ]
    features["same_family"] = [
        int(target == base and target != "unknown")
        for target, base in zip(features["target_action_family"], base_family)
    ]
    for column in ["same_depth", "same_side", "same_family"]:
        if column in out.columns:
            features[column] = out[column].map(lambda value: int(_norm_bool(value)))
    features["support_count"] = series("support_count", 0).map(_norm_float)
    features["source_family_count"] = series("source_family_count", 0).map(_norm_float)
    point0 = [
        int(_norm_int(candidate, -1) == 0 and _norm_int(base, -1) != 0)
        for candidate, base in zip(candidate_point, base_point)
    ]
    serve = [
        int(_norm_int(candidate, -1) in {15, 16, 17, 18})
        for candidate in candidate_action
    ]
    features["is_point0_addition"] = series("is_point0_addition", pd.Series(point0, index=out.index)).map(
        lambda value: int(_norm_bool(value))
    )
    features["is_serve_15_18_addition"] = series(
        "is_serve_15_18_addition", pd.Series(serve, index=out.index)
    ).map(lambda value: int(_norm_bool(value)))
    features["candidate_kind"] = kind
    return features


def synthetic_fallback_candidates(training_rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = training_rows.copy()
    out["row_id"] = out["row_id"].map(lambda value: value if str(value).startswith("synthetic_") else f"synthetic_{value}")
    out["candidate_kind"] = kind
    out["base_action"] = out["target_action_id"]
    out["candidate_action"] = out["target_action_id"]
    out["base_point"] = out["target_point_id"].where(out["target_point_id"].ne(0), 8)
    out["candidate_point"] = out["target_point_id"]
    if kind == "action":
        out["base_action"] = out["target_action_id"].where(~out["target_action_id"].isin({15, 16, 17, 18}), 3)
        out["candidate_action"] = out["target_action_id"]
        out["base_point"] = out["target_point_id"]
        out["candidate_point"] = out["target_point_id"]
    return out[["row_id", "rally_uid", "candidate_kind", "base_action", "candidate_action", "base_point", "candidate_point", *FEATURE_COLUMNS]]


def score_feature_frame(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["augmented_model_score"] = []
        out["risk_adjusted_score"] = []
        out["pass_augmented_gate"] = []
        return out

    raw_scores = model.predict_proba(out[FEATURE_COLUMNS])
    adjusted = []
    gates = []
    for raw, (_, row) in zip(raw_scores, out.iterrows()):
        score = float(raw)
        score += min(_norm_float(row.get("support_count")) / 120.0, 0.12)
        score += min(_norm_float(row.get("source_family_count")) / 30.0, 0.08)
        if _norm_bool(row.get("same_depth")):
            score += 0.03
        if _norm_bool(row.get("same_side")):
            score += 0.02
        if _norm_bool(row.get("same_family")):
            score += 0.03
        if _norm_bool(row.get("is_point0_addition")):
            score -= 0.35
        if _norm_bool(row.get("is_serve_15_18_addition")):
            score -= 0.35
        score = round(_bounded(score), 6)
        adjusted.append(score)
        gates.append(
            bool(
                score >= 0.55
                and not _norm_bool(row.get("is_point0_addition"))
                and not _norm_bool(row.get("is_serve_15_18_addition"))
            )
        )

    out["augmented_model_score"] = [round(float(value), 6) for value in raw_scores]
    out["risk_adjusted_score"] = adjusted
    out["pass_augmented_gate"] = pd.Series(gates, dtype=object)
    sort_cols = ["pass_augmented_gate", "risk_adjusted_score", "support_count", "source_family_count"]
    return out.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _missing_inputs(root: Path) -> list[str]:
    paths = [root / V385_GRAMMAR, root / POINT_POOL, root / ACTION_POOL]
    return [str(path) for path in paths if not path.exists()]


def run_pipeline(
    root: str | Path = Path("."),
    outdir: str | Path = OUTDIR,
    prefer_sklearn: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    output_dir = Path(outdir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grammar, missing_v385 = load_synthetic_grammar(root)
    training_rows = synthetic_features(grammar)
    model, model_used, sklearn_error = fit_augmented_model(training_rows, prefer_sklearn=prefer_sklearn)

    point_pool = _read_optional(root / POINT_POOL)
    action_pool = _read_optional(root / ACTION_POOL)
    missing_v388 = point_pool.empty and action_pool.empty

    if point_pool.empty:
        point_features = synthetic_fallback_candidates(training_rows, "point")
    else:
        point_features = _candidate_features(point_pool, "point")
    if action_pool.empty:
        action_features = synthetic_fallback_candidates(training_rows, "action")
    else:
        action_features = _candidate_features(action_pool, "action")

    point_scores = score_feature_frame(point_features, model)
    action_scores = score_feature_frame(action_features, model)

    point_scores.to_csv(output_dir / "point_augmented_scores.csv", index=False)
    action_scores.to_csv(output_dir / "action_augmented_scores.csv", index=False)

    label_counts = training_rows["label"].value_counts().to_dict()
    model_report = {
        "version": "v390_synthetic_augmented_scorer",
        "model_used": model_used,
        "sklearn_error": sklearn_error,
        "missing_v385": missing_v385,
        "missing_v388": missing_v388,
        "training_rows": int(len(training_rows)),
        "positive_synthetic_rows": int(label_counts.get(True, 0)),
        "negative_synthetic_rows": int(label_counts.get(False, 0)),
        "feature_columns": FEATURE_COLUMNS,
    }
    (output_dir / "model_report.json").write_text(json.dumps(model_report, indent=2), encoding="utf-8")

    emitted_submission_csvs = sorted(path.name for path in output_dir.glob("submission_*.csv"))
    report = {
        **model_report,
        "purpose": "Synthetic-augmented point/action scorer; no submission CSVs are generated.",
        "synthetic_source": "deterministic_fallback" if missing_v385 else "v385_expanded_synthetic_grammar",
        "candidate_source": "deterministic_synthetic_fallback" if missing_v388 else "v388_large_synthetic_candidate_pool",
        "point_candidates_scored": int(len(point_scores)),
        "action_candidates_scored": int(len(action_scores)),
        "point_pass_count": int(point_scores.get("pass_augmented_gate", pd.Series(dtype=bool)).sum()),
        "action_pass_count": int(action_scores.get("pass_augmented_gate", pd.Series(dtype=bool)).sum()),
        "missing_inputs": _missing_inputs(root),
        "outputs": output_filenames(),
        "emitted_submission_csvs": emitted_submission_csvs,
        "policy": [
            "No submission CSVs emitted by V390.",
            "No TTMATCH, old-server labels, hidden labels, or manual test-row edits used.",
            "Synthetic data is used only for scorer training and deterministic fallback scoring.",
        ],
    }
    (output_dir / "search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
