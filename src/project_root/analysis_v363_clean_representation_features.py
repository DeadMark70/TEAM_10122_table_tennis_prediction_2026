"""V363 clean representation features.

Builds reusable train/test context feature matrices for downstream specialist
experiments. The script does not export submissions, does not read TTMATCH, and
does not map external exact labels into AICUP exact action or point IDs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test_new.csv"
OUTDIR = ROOT / "v363_clean_representation_features"
V255_CANONICAL = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"
V326_AICUP_FEATURES = ROOT / "v326_masked_family_pretrain" / "v326_aicup_prefix_family_features.csv"
V327_AICUP_FEATURES = ROOT / "v327_response_style_contrastive" / "v327_aicup_response_style_features.csv"
V328_SEARCH = ROOT / "v328_coarse_to_exact_distillation" / "v328_action_search.csv"

FAMILIES = ("zero", "serve", "attack", "control", "defensive", "unknown")
LABEL_LIKE_EXACT = {
    "actionid",
    "pointid",
    "servergetpoint",
    "next_actionid",
    "next_pointid",
    "next_servergetpoint",
    "target_actionid",
    "target_pointid",
    "target_servergetpoint",
    "event_type",
    "raw_label",
    "allowed_target",
}
BANNED_SOURCE_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def contains_banned_source(value: Any) -> bool:
    upper = str(value).upper().replace("\\", "/")
    return any(token in upper for token in BANNED_SOURCE_TOKENS)


def is_label_like_column(column: Any) -> bool:
    name = str(column).strip()
    lower = name.lower()
    if lower in LABEL_LIKE_EXACT:
        return True
    if lower.endswith("_actionid") or lower.endswith("_pointid"):
        return True
    if "servergetpoint" in lower:
        return True
    if lower in {"label", "target", "y", "class"}:
        return True
    if lower.endswith("_label") or lower.endswith("_target"):
        return True
    return False


def drop_label_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-label and target-like columns from external/feature frames."""
    keep = [col for col in df.columns if not is_label_like_column(col)]
    return df.loc[:, keep].copy()


def prefix_len_bin(prefix_len: Any) -> str:
    try:
        value = int(prefix_len)
    except Exception:
        return "unknown"
    if value <= 0:
        return "unknown"
    if value <= 3:
        return str(value)
    if value <= 6:
        return "4_6"
    return "7p"


def phase_from_prefix_len(prefix_len: Any) -> str:
    try:
        value = int(prefix_len)
    except Exception:
        return "unknown"
    if value <= 0:
        return "unknown"
    if value == 1:
        return "serve"
    if value == 2:
        return "receive"
    if value == 3:
        return "third_ball"
    if value == 4:
        return "fourth_ball"
    return "rally"


def normalize_phase(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    if text in {"", "nan", "none", "<na>"}:
        return "unknown"
    if "serve" in text:
        return "serve"
    if "receive" in text:
        return "receive"
    if "third" in text:
        return "third_ball"
    if "fourth" in text:
        return "fourth_ball"
    if "terminal" in text:
        return "terminal"
    if "rally" in text:
        return "rally"
    return text


def action_to_family(action_id: Any) -> str:
    try:
        value = int(action_id)
    except Exception:
        return "unknown"
    if value == 0:
        return "zero"
    if value in {15, 16, 17, 18}:
        return "serve"
    if 1 <= value <= 7:
        return "attack"
    if 8 <= value <= 11:
        return "control"
    if 12 <= value <= 14:
        return "defensive"
    return "unknown"


def normalize_family(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return "unknown"
    if "serve" in text or "service" in text:
        return "serve"
    if "zero" in text or "terminal" in text or "net" in text or "error" in text:
        return "zero"
    if "defen" in text or "chop" in text or "lob" in text or "clear" in text:
        return "defensive"
    if "control" in text or "push" in text or "drop" in text or "receive" in text or "block" in text:
        return "control"
    if "attack" in text or "smash" in text or "drive" in text or "topspin" in text or "loop" in text:
        return "attack"
    return text if text in FAMILIES else "unknown"


def point_to_depth(point_id: Any) -> str:
    try:
        value = int(point_id)
    except Exception:
        return "missing"
    if value == 0:
        return "terminal"
    if value in {1, 2, 3}:
        return "short"
    if value in {4, 5, 6}:
        return "half"
    if value in {7, 8, 9}:
        return "long"
    return "missing"


def point_to_side(point_id: Any) -> str:
    try:
        value = int(point_id)
    except Exception:
        return "missing"
    if value == 0:
        return "terminal"
    if value in {1, 4, 7}:
        return "left"
    if value in {2, 5, 8}:
        return "middle"
    if value in {3, 6, 9}:
        return "right"
    return "missing"


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _prefix_len(frame: pd.DataFrame) -> pd.Series:
    if "prefix_len" in frame.columns:
        return _numeric(frame, "prefix_len", 0).astype(int)
    return _numeric(frame, "strikeNumber", 0).astype(int)


def _coarse_context(frame: pd.DataFrame) -> pd.DataFrame:
    prefix = _prefix_len(frame)
    action = frame["actionId"] if "actionId" in frame.columns else pd.Series(pd.NA, index=frame.index)
    point = frame["pointId"] if "pointId" in frame.columns else pd.Series(pd.NA, index=frame.index)
    out = pd.DataFrame(index=frame.index)
    out["prefix_len"] = prefix
    out["phase"] = prefix.map(phase_from_prefix_len)
    out["action_family"] = action.map(action_to_family)
    out["point_depth"] = point.map(point_to_depth)
    out["point_side"] = point.map(point_to_side)
    return out


def _prob_table(frame: pd.DataFrame, keys: list[str], value: str) -> tuple[dict[tuple[Any, ...], float], dict[tuple[Any, ...], int]]:
    counts = frame.groupby(keys + [value], dropna=False).size().rename("count").reset_index()
    totals = frame.groupby(keys, dropna=False).size().rename("total").reset_index()
    merged = counts.merge(totals, on=keys, how="left")
    merged["prob"] = merged["count"] / merged["total"].clip(lower=1)
    prob: dict[tuple[Any, ...], float] = {}
    count: dict[tuple[Any, ...], int] = {}
    for row in merged.itertuples(index=False):
        row_data = row._asdict()
        key = tuple(row_data[k] for k in keys + [value])
        prob[key] = float(row_data["prob"])
        count[key] = int(row_data["count"])
    return prob, count


def fit_aicup_priors(train: pd.DataFrame) -> dict[str, Any]:
    coarse = _coarse_context(train)
    family_depth_prob, family_depth_count = _prob_table(coarse, ["action_family"], "point_depth")
    family_side_prob, family_side_count = _prob_table(coarse, ["action_family"], "point_side")
    phase_family_prob, phase_family_count = _prob_table(coarse, ["phase"], "action_family")
    depth_global = coarse["point_depth"].value_counts(normalize=True).to_dict()
    side_global = coarse["point_side"].value_counts(normalize=True).to_dict()
    family_global = coarse["action_family"].value_counts(normalize=True).to_dict()
    return {
        "family_depth_prob": family_depth_prob,
        "family_depth_count": family_depth_count,
        "family_side_prob": family_side_prob,
        "family_side_count": family_side_count,
        "phase_family_prob": phase_family_prob,
        "phase_family_count": phase_family_count,
        "depth_global": {str(k): float(v) for k, v in depth_global.items()},
        "side_global": {str(k): float(v) for k, v in side_global.items()},
        "family_global": {str(k): float(v) for k, v in family_global.items()},
        "train_rows": int(len(train)),
    }


def build_context_features(
    frame: pd.DataFrame,
    split: str,
    priors: dict[str, Any],
    external_phase_family: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    coarse = _coarse_context(frame)
    score_self = _numeric(frame, "scoreSelf", 0)
    score_other = _numeric(frame, "scoreOther", 0)
    out = pd.DataFrame(index=frame.index)
    out["row_id"] = np.arange(len(frame), dtype=int)
    out["rally_uid"] = _numeric(frame, "rally_uid", -1).astype(int)
    out["match"] = _numeric(frame, "match", -1).astype(int)
    out["numberGame"] = _numeric(frame, "numberGame", -1).astype(int)
    out["rally_id"] = _numeric(frame, "rally_id", -1).astype(int)
    out["prefix_len"] = coarse["prefix_len"].astype(int)
    out["prefix_len_bin"] = coarse["prefix_len"].map(prefix_len_bin)
    out["phase"] = coarse["phase"]
    out["lag0_action_family"] = coarse["action_family"]
    out["lag0_point_depth"] = coarse["point_depth"]
    out["lag0_point_side"] = coarse["point_side"]
    out["lag0_spinId"] = _numeric(frame, "spinId", -1).astype(int)
    out["lag0_strengthId"] = _numeric(frame, "strengthId", -1).astype(int)
    out["lag0_positionId"] = _numeric(frame, "positionId", -1).astype(int)
    out["score_self"] = score_self.astype(float)
    out["score_other"] = score_other.astype(float)
    out["score_margin"] = (score_self - score_other).astype(float)
    out["score_total"] = (score_self + score_other).astype(float)
    out["score_tied"] = (score_self == score_other).astype(int)
    out["score_close"] = ((score_self - score_other).abs() <= 2).astype(int)
    out["score_deuce_like"] = ((score_self >= 10) & (score_other >= 10) & ((score_self - score_other).abs() <= 2)).astype(int)
    out["score_pressure"] = np.select(
        [
            out["score_deuce_like"].eq(1),
            out["score_total"].ge(18) & out["score_close"].eq(1),
        ],
        ["deuce_like", "late_close"],
        default="normal",
    )

    families = coarse["action_family"].to_numpy()
    depths = coarse["point_depth"].to_numpy()
    sides = coarse["point_side"].to_numpy()
    phases = coarse["phase"].to_numpy()
    out["aicup_family_point_depth_prior"] = [
        priors["family_depth_prob"].get((family, depth), priors["depth_global"].get(str(depth), 0.0))
        for family, depth in zip(families, depths)
    ]
    out["aicup_family_point_depth_count"] = [
        priors["family_depth_count"].get((family, depth), 0) for family, depth in zip(families, depths)
    ]
    out["aicup_family_point_side_prior"] = [
        priors["family_side_prob"].get((family, side), priors["side_global"].get(str(side), 0.0))
        for family, side in zip(families, sides)
    ]
    out["aicup_family_point_side_count"] = [
        priors["family_side_count"].get((family, side), 0) for family, side in zip(families, sides)
    ]
    out["aicup_phase_action_family_prior"] = [
        priors["phase_family_prob"].get((phase, family), priors["family_global"].get(str(family), 0.0))
        for phase, family in zip(phases, families)
    ]
    out["aicup_phase_action_family_count"] = [
        priors["phase_family_count"].get((phase, family), 0) for phase, family in zip(phases, families)
    ]

    if external_phase_family:
        for family in FAMILIES:
            out[f"external_phase_family_prior_{family}"] = [
                external_phase_family.get((phase, family), external_phase_family.get(("__global__", family), 0.0))
                for phase in phases
            ]

    return drop_label_like_columns(out)


def control_high_cardinality_objects(df: pd.DataFrame, max_unique: int = 64) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    report: list[dict[str, Any]] = []
    for column in list(out.columns):
        series = out[column]
        if not (is_object_dtype(series) or is_string_dtype(series)):
            continue
        unique = int(series.nunique(dropna=True))
        if unique <= max_unique:
            continue
        freq = series.map(series.value_counts(dropna=False)).fillna(0).astype(int)
        loc = out.columns.get_loc(column)
        out.insert(loc, f"{column}_freq", freq)
        out = out.drop(columns=[column])
        report.append(
            {
                "feature": column,
                "unique_values": unique,
                "action": "frequency_encoded",
                "replacement": f"{column}_freq",
            }
        )
    return out, report


def load_external_phase_family_priors(path: Path = V255_CANONICAL) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    if not path.exists():
        return {}, {"path": rel(path), "available": False, "rows": 0}
    usecols = ["source_dataset", "source_path", "coarse_family", "phase"]
    raw = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    raw = drop_label_like_columns(raw)
    if "source_dataset" in raw.columns:
        source_bad = raw["source_dataset"].map(contains_banned_source)
    else:
        source_bad = pd.Series(False, index=raw.index)
    if "source_path" in raw.columns:
        path_bad = raw["source_path"].map(contains_banned_source)
    else:
        path_bad = pd.Series(False, index=raw.index)
    kept = raw.loc[~(source_bad | path_bad)].copy()
    if kept.empty or "coarse_family" not in kept.columns:
        return {}, {"path": rel(path), "available": True, "rows": int(len(raw)), "kept_rows": 0}
    phase = kept["phase"].map(normalize_phase) if "phase" in kept.columns else pd.Series("unknown", index=kept.index)
    family = kept["coarse_family"].map(normalize_family)
    coarse = pd.DataFrame({"phase": phase, "family": family})
    by_phase = coarse.groupby(["phase", "family"]).size().rename("count").reset_index()
    phase_total = coarse.groupby("phase").size().rename("total").reset_index()
    merged = by_phase.merge(phase_total, on="phase", how="left")
    priors = {
        (str(row.phase), str(row.family)): float(row.count / max(row.total, 1))
        for row in merged.itertuples(index=False)
    }
    global_counts = coarse["family"].value_counts(normalize=True)
    for family_name, value in global_counts.items():
        priors[("__global__", str(family_name))] = float(value)
    return priors, {
        "path": rel(path),
        "available": True,
        "rows": int(len(raw)),
        "kept_rows": int(len(kept)),
        "banned_rows_dropped": int((source_bad | path_bad).sum()),
    }


def _optional_feature_table(path: Path, prefix: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not path.exists():
        return None, {"path": rel(path), "available": False}
    table = pd.read_csv(path, low_memory=False)
    table = drop_label_like_columns(table)
    keys = ["split", "rally_uid", "match", "prefix_len"]
    if not set(keys).issubset(table.columns):
        return None, {"path": rel(path), "available": True, "used": False, "reason": "missing merge keys"}
    keep_cols = keys + [
        col
        for col in table.columns
        if col not in keys and (col.startswith(prefix) or col in {"v326_pred_family", "v327_context_covered"})
    ]
    table = table.loc[:, keep_cols].drop_duplicates(keys)
    return table, {"path": rel(path), "available": True, "used": True, "rows": int(len(table)), "columns": keep_cols}


def merge_optional_representation_features(features: pd.DataFrame, split: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = features.copy()
    out.insert(0, "split", split)
    reports: list[dict[str, Any]] = []
    for path, prefix in ((V326_AICUP_FEATURES, "v326_"), (V327_AICUP_FEATURES, "v327_")):
        table, report = _optional_feature_table(path, prefix)
        reports.append(report)
        if table is None:
            continue
        subset = table.loc[table["split"].astype(str).eq(split)].copy()
        if subset.empty:
            continue
        out = out.merge(subset, on=["split", "rally_uid", "match", "prefix_len"], how="left")
    out = out.drop(columns=["split"])
    return out, reports


def feature_summary(df: pd.DataFrame, high_cardinality_report: list[dict[str, Any]]) -> pd.DataFrame:
    actions = {row["replacement"]: row["action"] for row in high_cardinality_report}
    rows = []
    for col in df.columns:
        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique(dropna=True)),
                "missing_values": int(df[col].isna().sum()),
                "handling": actions.get(col, "kept"),
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(
    train_path: Path = TRAIN_CSV,
    test_path: Path = TEST_CSV,
    outdir: Path = OUTDIR,
) -> dict[str, Any]:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    priors = fit_aicup_priors(train)
    external_priors, external_report = load_external_phase_family_priors()

    train_features = build_context_features(train, "train", priors, external_priors)
    test_features = build_context_features(test, "test", priors, external_priors)
    train_features, train_optional_reports = merge_optional_representation_features(train_features, "train")
    test_features, test_optional_reports = merge_optional_representation_features(test_features, "test")

    train_features = drop_label_like_columns(train_features)
    test_features = drop_label_like_columns(test_features)
    train_features, train_high_card = control_high_cardinality_objects(train_features, max_unique=64)
    test_features, test_high_card = control_high_cardinality_objects(test_features, max_unique=64)

    common_columns = [col for col in train_features.columns if col in test_features.columns]
    train_features = train_features.loc[:, common_columns]
    test_features = test_features.loc[:, common_columns]

    outdir.mkdir(parents=True, exist_ok=True)
    train_out = outdir / "train_context_features.csv"
    test_out = outdir / "test_context_features.csv"
    summary_out = outdir / "feature_summary.csv"
    report_out = outdir / "search_report.json"

    train_features.to_csv(train_out, index=False)
    test_features.to_csv(test_out, index=False)
    summary = feature_summary(
        pd.concat([train_features.head(5000), test_features.head(5000)], ignore_index=True),
        train_high_card + test_high_card,
    )
    summary.to_csv(summary_out, index=False)

    report = {
        "version": "v363",
        "submissions_written": 0,
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "feature_columns": int(len(common_columns)),
        "outputs": {
            "train_context_features": rel(train_out),
            "test_context_features": rel(test_out),
            "feature_summary": rel(summary_out),
            "search_report": rel(report_out),
        },
        "forbidden_inputs": {
            "ttmatch_rows_used": 0,
            "old_server_rows_used": 0,
            "test_hidden_labels_used": 0,
            "external_exact_action_labels_mapped": False,
        },
        "label_like_columns_dropped": sorted(LABEL_LIKE_EXACT),
        "high_cardinality_handling": train_high_card + test_high_card,
        "aicup_train_priors": {
            "train_rows": priors["train_rows"],
            "family_depth_keys": len(priors["family_depth_prob"]),
            "family_side_keys": len(priors["family_side_prob"]),
            "phase_family_keys": len(priors["phase_family_prob"]),
        },
        "external_sources": {
            "v255_phase_family_priors": external_report,
            "v326_train_merge": train_optional_reports[0],
            "v327_train_merge": train_optional_reports[1],
            "v326_test_merge": test_optional_reports[0],
            "v327_test_merge": test_optional_reports[1],
            "v328_search_present": V328_SEARCH.exists(),
        },
    }
    report_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
