"""V348 public-risk row gate for point candidate-bank rows.

This module scores row-level point candidates from the V343 candidate bank
against public evidence. It writes reports only under
v348_public_risk_row_gate and never exports submission files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    safe_output_path,
    validate_submission_schema,
    write_json,
)
from analysis_v343_row_candidate_bank import point_depth


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v348_public_risk_row_gate"
CANDIDATE_BANK = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
TEST_NEW = ROOT / "test_new.csv"
V306_ANCHOR = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V338_PUBLIC_POSITIVE = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V341_SUMMARY = ROOT / "v341_no_p0_point_pack" / "joint_summary.csv"
V307_REPORT = ROOT / "v307_point0_dose_extension" / "v307_report.json"


BOOL_COLUMNS = [
    "is_point0_addition",
    "is_point0_removal",
    "is_nonterminal_point_swap",
    "is_same_depth_swap",
    "changed_in_v338",
]
PUBLIC_TAGS = [
    "v338_public_positive",
    "v306_point0_probe",
    "v307_saturated_p0",
    "no_p0_expansion",
    "v338_family_support",
    "historical_point_model",
    "unknown",
]
SCORE_COLUMNS = [
    "row_id",
    "rally_uid",
    "anchor_value",
    "candidate_value",
    "transition",
    "same_depth",
    "point0_addition",
    "point0_removal",
    "nonterminal_point_swap",
    "changed_in_v338",
    "positive_label",
    "risk_label",
    "v341_extra_risk",
    "v307_extra_p0_risk",
    "source_count",
    "agreement_count",
    "source_family_count",
    "mean_source_local_delta",
    "support_score",
    "risk_score",
    "trust_score",
    "gate_decision",
    "prefix_len",
    "phase",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False).astype(bool)


def load_candidate_bank(path: Path = CANDIDATE_BANK) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing candidate bank: {relative_path(path)}")
    bank = pd.read_csv(path)
    required = {
        "row_id",
        "rally_uid",
        "anchor_value",
        "candidate_value",
        "transition",
        "source",
        "source_dir",
        "source_public_tag",
        "source_local_delta_if_known",
        *BOOL_COLUMNS,
    }
    missing = sorted(required - set(bank.columns))
    if missing:
        raise ValueError(f"candidate bank missing columns: {missing}")
    for col in BOOL_COLUMNS:
        bank[col] = _as_bool(bank[col])
    bank["row_id"] = bank["row_id"].astype(int)
    bank["anchor_value"] = bank["anchor_value"].astype(int)
    bank["candidate_value"] = bank["candidate_value"].astype(int)
    bank["source_local_delta_if_known"] = pd.to_numeric(bank["source_local_delta_if_known"], errors="coerce")
    return bank


def changed_row_keys(base: pd.DataFrame, candidate: pd.DataFrame) -> set[tuple[int, int]]:
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs from base")
    base_point = base["pointId"].to_numpy(dtype=int)
    cand_point = candidate["pointId"].to_numpy(dtype=int)
    return {
        (int(row_id), int(cand_point[row_id]))
        for row_id in np.where(base_point != cand_point)[0]
    }


def row_ids_changed(base: pd.DataFrame, candidate: pd.DataFrame) -> set[int]:
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("submission row order differs from base")
    return set(np.where(base["pointId"].to_numpy(dtype=int) != candidate["pointId"].to_numpy(dtype=int))[0].astype(int).tolist())


def v341_extra_keys(base: pd.DataFrame, public_positive: pd.DataFrame, summary_path: Path = V341_SUMMARY) -> set[tuple[int, int]]:
    if not summary_path.exists():
        return set()
    summary = pd.read_csv(summary_path)
    if "path" not in summary.columns:
        return set()
    public_positive_rows = row_ids_changed(base, public_positive)
    out: set[tuple[int, int]] = set()
    for raw_path in summary.loc[summary.get("decision", "EXPORT_LOCAL") == "EXPORT_LOCAL", "path"].dropna():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        candidate = load_submission(path, expected_rows=len(base))
        if not base["rally_uid"].equals(candidate["rally_uid"]):
            continue
        base_point = base["pointId"].to_numpy(dtype=int)
        cand_point = candidate["pointId"].to_numpy(dtype=int)
        for row_id in np.where(base_point != cand_point)[0]:
            if int(row_id) not in public_positive_rows:
                out.add((int(row_id), int(cand_point[row_id])))
    return out


def _report_candidate_path(report: dict[str, Any], keys: list[str]) -> Path | None:
    for key in keys:
        item = report.get(key)
        if isinstance(item, dict) and item.get("path"):
            path = Path(str(item["path"]))
            return path if path.is_absolute() else ROOT / path
    return None


def v307_extra_p0_keys(base: pd.DataFrame, report_path: Path = V307_REPORT) -> set[tuple[int, int]]:
    if not report_path.exists():
        return set()
    report = _json_load(report_path)
    v306_path = _report_candidate_path(report, ["v306_best_candidate"])
    v307_path = _report_candidate_path(report, ["best_candidate"])
    if v306_path is None or v307_path is None or not v306_path.exists() or not v307_path.exists():
        return set()
    v306 = load_submission(v306_path, expected_rows=len(base))
    v307 = load_submission(v307_path, expected_rows=len(base))
    if not base["rally_uid"].equals(v306["rally_uid"]) or not base["rally_uid"].equals(v307["rally_uid"]):
        raise ValueError("V306/V307 evidence row order differs from anchor")
    base_point = base["pointId"].to_numpy(dtype=int)
    v306_point = v306["pointId"].to_numpy(dtype=int)
    v307_point = v307["pointId"].to_numpy(dtype=int)
    out: set[tuple[int, int]] = set()
    for row_id in np.where(v306_point != v307_point)[0]:
        if base_point[row_id] != v307_point[row_id]:
            out.add((int(row_id), int(v307_point[row_id])))
    return out


def load_optional_row_context(path: Path, base: pd.DataFrame) -> pd.DataFrame:
    context = pd.DataFrame({"row_id": np.arange(len(base), dtype=int)})
    context["prefix_len"] = np.nan
    context["phase"] = "unknown"
    if not path.exists():
        return context
    raw = pd.read_csv(path)
    if len(raw) != len(base) or "rally_uid" not in raw or not raw["rally_uid"].equals(base["rally_uid"]):
        return context
    if "strikeNumber" in raw:
        prefix = pd.to_numeric(raw["strikeNumber"], errors="coerce")
        context["prefix_len"] = prefix
        context["phase"] = pd.cut(
            prefix.fillna(-1),
            bins=[-2, 2, 5, 10_000],
            labels=["early", "mid", "late"],
        ).astype(str)
    return context


def _family_count_columns(rows: pd.DataFrame) -> dict[str, int]:
    counts = {f"family_count_{tag}": 0 for tag in PUBLIC_TAGS}
    for tag, value in rows["source_public_tag"].value_counts(dropna=False).items():
        key = f"family_count_{tag if pd.notna(tag) else 'unknown'}"
        counts[key] = int(value)
    return counts


def aggregate_candidate_rows(bank: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = bank.groupby(["row_id", "anchor_value", "candidate_value", "transition"], sort=False, dropna=False)
    for (row_id, anchor_value, candidate_value, transition), group in grouped:
        families = group["source_public_tag"].fillna("unknown").astype(str)
        source_dirs = group["source_dir"].fillna("unknown").astype(str)
        rows.append(
            {
                "row_id": int(row_id),
                "rally_uid": group["rally_uid"].iloc[0],
                "anchor_value": int(anchor_value),
                "candidate_value": int(candidate_value),
                "transition": str(transition),
                "same_depth": bool(group["is_same_depth_swap"].any()),
                "point0_addition": bool(group["is_point0_addition"].any()),
                "point0_removal": bool(group["is_point0_removal"].any()),
                "nonterminal_point_swap": bool(group["is_nonterminal_point_swap"].any()),
                "changed_in_v338": bool(group["changed_in_v338"].any()),
                "source_count": int(group["source"].nunique()),
                "agreement_count": int(len(group)),
                "source_family_count": int(families.nunique()),
                "source_dir_count": int(source_dirs.nunique()),
                "mean_source_local_delta": float(group["source_local_delta_if_known"].mean())
                if group["source_local_delta_if_known"].notna().any()
                else np.nan,
                **_family_count_columns(group),
            }
        )
    scores = pd.DataFrame(rows)
    if scores.empty:
        scores = pd.DataFrame(columns=SCORE_COLUMNS)
    else:
        scores = scores.merge(context, on="row_id", how="left")
    return scores


def score_rows(
    rows: pd.DataFrame,
    positive_keys: set[tuple[int, int]],
    v341_keys: set[tuple[int, int]],
    v307_keys: set[tuple[int, int]],
) -> pd.DataFrame:
    out = rows.copy()
    keys = list(zip(out["row_id"].astype(int), out["candidate_value"].astype(int)))
    out["positive_label"] = [key in positive_keys for key in keys]
    out["v341_extra_risk"] = [key in v341_keys for key in keys]
    out["v307_extra_p0_risk"] = [key in v307_keys for key in keys]
    out["risk_label"] = out["v341_extra_risk"] | out["v307_extra_p0_risk"]

    support = (
        1.20 * out["positive_label"].astype(float)
        + 0.22 * np.log1p(out["agreement_count"].astype(float))
        + 0.18 * out["source_family_count"].astype(float)
        + 0.35 * out["changed_in_v338"].astype(float)
        + 0.12 * out["same_depth"].astype(float)
        + 0.08 * (out["mean_source_local_delta"].fillna(0.0).clip(lower=0.0) * 100.0)
    )
    risk = (
        1.45 * out["risk_label"].astype(float)
        + 0.95 * out["v341_extra_risk"].astype(float)
        + 0.85 * out["v307_extra_p0_risk"].astype(float)
        + 0.55 * out["point0_addition"].astype(float)
        + 0.30 * out["point0_removal"].astype(float)
        + 0.20 * (~out["same_depth"]).astype(float)
        + 0.12 * (out["source_family_count"].astype(float) <= 1).astype(float)
    )
    out["support_score"] = support.round(6)
    out["risk_score"] = risk.round(6)
    out["trust_score"] = (support - risk).round(6)
    out["gate_decision"] = np.select(
        [
            out["risk_label"],
            out["trust_score"] >= 0.75,
            out["trust_score"] >= 0.25,
        ],
        ["BLOCK_PUBLIC_RISK", "ALLOW_HIGH_TRUST", "REVIEW_MEDIUM_TRUST"],
        default="BLOCK_LOW_TRUST",
    )
    sort_cols = ["trust_score", "risk_score", "support_score", "row_id", "candidate_value"]
    return out.sort_values(sort_cols, ascending=[False, True, False, True, True]).reset_index(drop=True)


def build_feature_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scores.empty:
        return pd.DataFrame(columns=["feature", "group", "rows", "mean_risk_score", "mean_trust_score"])
    for feature in [
        "positive_label",
        "risk_label",
        "v341_extra_risk",
        "v307_extra_p0_risk",
        "same_depth",
        "point0_addition",
        "point0_removal",
        "changed_in_v338",
        "gate_decision",
        "phase",
    ]:
        if feature not in scores:
            continue
        for value, group in scores.groupby(feature, dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "group": str(value),
                    "rows": int(len(group)),
                    "mean_risk_score": float(group["risk_score"].mean()),
                    "mean_trust_score": float(group["trust_score"].mean()),
                    "positive_rows": int(group["positive_label"].sum()),
                    "risk_rows": int(group["risk_label"].sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "group"]).reset_index(drop=True)


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    candidate_bank_path: Path = CANDIDATE_BANK,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = safe_output_path(outdir, "search_report.json")

    try:
        base = load_submission(V306_ANCHOR, expected_rows=expected_rows)
        public_positive = load_submission(V338_PUBLIC_POSITIVE, expected_rows=expected_rows)
        bank = load_candidate_bank(candidate_bank_path)
    except Exception as exc:  # noqa: BLE001 - report blocked state as JSON.
        report = {"version": "V348", "decision": "BLOCKED_INPUT", "reason": f"{type(exc).__name__}: {exc}"}
        write_json(report_path, report)
        return report

    positive_keys = changed_row_keys(base, public_positive)
    v341_keys = v341_extra_keys(base, public_positive)
    v307_keys = v307_extra_p0_keys(base)
    context = load_optional_row_context(TEST_NEW, base)
    aggregated = aggregate_candidate_rows(bank, context)
    scores = score_rows(aggregated, positive_keys, v341_keys, v307_keys)
    feature_summary = build_feature_summary(scores)

    score_path = safe_output_path(outdir, "row_gate_scores.csv")
    feature_path = safe_output_path(outdir, "feature_summary.csv")
    scores.to_csv(score_path, index=False)
    feature_summary.to_csv(feature_path, index=False)

    decision_counts = scores["gate_decision"].value_counts().sort_index().to_dict() if not scores.empty else {}
    report = {
        "version": "V348",
        "decision": "GATE_SCORED",
        "row_gate_scores": relative_path(score_path),
        "feature_summary": relative_path(feature_path),
        "candidate_bank": relative_path(candidate_bank_path),
        "candidate_bank_rows": int(len(bank)),
        "scored_candidate_rows": int(len(scores)),
        "positive_evidence_rows": int(len(positive_keys)),
        "v341_extra_risk_rows": int(len(v341_keys)),
        "v307_extra_p0_risk_rows": int(len(v307_keys)),
        "scored_positive_rows": int(scores["positive_label"].sum()) if not scores.empty else 0,
        "scored_risk_rows": int(scores["risk_label"].sum()) if not scores.empty else 0,
        "gate_decision_counts": {str(k): int(v) for k, v in decision_counts.items()},
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "upload_candidates_writes": False,
            "submission_exports": False,
            "reports_only": True,
        },
        "scoring": {
            "risk_score": "weighted public-risk evidence plus point0/depth penalties",
            "trust_score": "support_score - risk_score",
            "positive_rows": "V338 changed rows vs V306",
            "risk_rows": "V341 rows beyond V338 plus V307 best-candidate extras beyond V306 cap0p01",
        },
    }
    write_json(report_path, _json_safe(report))
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
