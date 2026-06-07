"""V445 full professor MoE packager.

Combines the V442 point residuals, V444 rare-class sweep proposals, and V434
MoE candidates into conservative V362-anchored submissions.
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
    action_distribution_report,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
)
from analysis_v435_residual_packager import apply_ranked_candidates


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v445_full_professor_moe_packager"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V442_POINT_CANDIDATES = ROOT / "v442_intent_first_sequence_point" / "point_candidate_table.csv"
V443_STYLE_EMBEDDINGS = ROOT / "v443_response_style_contrastive" / "test_style_embeddings.csv"
V444_ACTION_SWEEP = ROOT / "v444_rare_aug_dropout_sweep" / "action_sweep_scores_test.csv"
V444_POINT_SWEEP = ROOT / "v444_rare_aug_dropout_sweep" / "point_sweep_scores_test.csv"
V444_OOF_REPORT = ROOT / "v444_rare_aug_dropout_sweep" / "oof_sweep_report.csv"
V434_ACTION_CANDIDATES = ROOT / "v434_anchor_aware_moe_gate" / "moe_action_candidates.csv"
V434_POINT_CANDIDATES = ROOT / "v434_anchor_aware_moe_gate" / "moe_point_candidates.csv"

RISKY_SOURCE_MARKERS = (
    "v436",
    "v438",
    "ttmatch_quarantined",
    "quarantined_contrastive",
    "sony_nd_audit_only",
    "sony",
)


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


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _source_text(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    parts = []
    for col in ("source", "sources", "source_family", "source_details"):
        if col in frame.columns:
            parts.append(frame[col].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    out = parts[0].copy()
    for part in parts[1:]:
        out = out.str.cat(part, sep="|")
    return out


def _risky_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    text = _source_text(frame).str.lower()
    return pd.Series(
        [any(marker in value for marker in RISKY_SOURCE_MARKERS) for value in text],
        index=frame.index,
        dtype=bool,
    )


def _source_family(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return "unknown"
    first = text.split("|")[0].split(";")[0]
    return first.split("/")[0] if "/" in first else first


def _source_family_counts(*frames: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "source_family" in frame.columns:
            values = frame["source_family"]
        elif "source" in frame.columns:
            values = frame["source"].map(_source_family)
        elif "sources" in frame.columns:
            values = frame["sources"].map(_source_family)
        else:
            values = pd.Series(["unknown"] * len(frame), index=frame.index)
        for family, count in values.value_counts(dropna=False).items():
            key = _source_family(family)
            counts[key] = counts.get(key, 0) + int(count)
    return dict(sorted(counts.items()))


def rank_professor_sources(rows: pd.DataFrame) -> pd.DataFrame:
    """Rank source families, preferring OOF evidence and safety over confidence."""

    if rows.empty:
        out = rows.copy()
        out["rank_score"] = []
        return out
    out = rows.copy()
    oof_delta = pd.to_numeric(out.get("oof_delta", 0.0), errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(out.get("confidence", 0.0), errors="coerce").fillna(0.0)
    changed = pd.to_numeric(out.get("changed_rows", 0), errors="coerce").fillna(0.0)
    point0 = pd.to_numeric(out.get("point0_additions", 0), errors="coerce").fillna(0.0)
    serve = pd.to_numeric(out.get("serve_additions", 0), errors="coerce").fillna(0.0)
    risky = _risky_mask(out).astype(float) if any(c in out.columns for c in ("source", "sources", "source_family")) else 0.0

    out["rank_score"] = (
        (80.0 * oof_delta)
        + (0.20 * confidence)
        - (0.002 * changed)
        - (4.0 * point0)
        - (4.0 * serve)
        - (100.0 * risky)
    )
    return out.sort_values(["rank_score", "oof_delta", "confidence"], ascending=[False, False, False]).reset_index(drop=True)


def _filter_risky_candidates(candidates: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    if candidates is None or candidates.empty:
        return pd.DataFrame(), 0
    frame = candidates.copy()
    mask = _risky_mask(frame)
    return frame.loc[~mask].copy(), int(mask.sum())


def _add_candidate_columns(frame: pd.DataFrame, *, target: str, source: str) -> pd.DataFrame:
    if frame.empty:
        candidate_col = f"candidate_{target}Id"
        return pd.DataFrame(columns=["rally_uid", candidate_col, "utility", "source", "source_family"])
    out = frame.copy()
    candidate_col = f"candidate_{target}Id"
    if candidate_col not in out.columns and "candidate_value" in out.columns:
        out[candidate_col] = out["candidate_value"]
    if "utility" not in out.columns:
        score_col = next((col for col in ("utility", "score", "confidence", f"{target}_confidence") if col in out.columns), None)
        out["utility"] = pd.to_numeric(out[score_col], errors="coerce") if score_col else 0.0
    if "source" not in out.columns:
        out["source"] = source
    if "source_family" not in out.columns:
        out["source_family"] = out["source"].map(_source_family)
    out["utility"] = pd.to_numeric(out["utility"], errors="coerce").fillna(-np.inf)
    out[candidate_col] = pd.to_numeric(out[candidate_col], errors="coerce")
    out = out.loc[out["rally_uid"].notna() & out[candidate_col].notna()].copy()
    out[candidate_col] = out[candidate_col].astype(int)
    return out


def _v444_source_scores(oof_report: pd.DataFrame, target: str) -> dict[str, float]:
    if oof_report.empty:
        return {}
    subset = oof_report.loc[oof_report["target"].astype(str).eq(target)].copy()
    if subset.empty:
        return {}
    rows = pd.DataFrame(
        {
            "source": "v444_rare_aug_dropout_sweep/" + subset["variant"].astype(str),
            "variant": subset["variant"].astype(str),
            "oof_delta": pd.to_numeric(subset.get("macro_f1", 0.0), errors="coerce").fillna(0.0)
            - pd.to_numeric(subset.get("macro_f1", 0.0), errors="coerce").fillna(0.0).mean(),
            "changed_rows": 0,
            "point0_additions": 0,
            "serve_additions": 0,
            "confidence": pd.to_numeric(subset.get("accuracy", 0.0), errors="coerce").fillna(0.0),
        }
    )
    ranked = rank_professor_sources(rows)
    return {str(row["variant"]): float(row["rank_score"]) for _, row in ranked.iterrows()}


def _candidates_from_v444_sweep(
    anchor: pd.DataFrame,
    scores: pd.DataFrame,
    oof_report: pd.DataFrame,
    *,
    target: str,
) -> pd.DataFrame:
    candidate_col = f"candidate_{target}Id"
    pred_suffix = f"_pred_{target}Id"
    if scores.empty:
        return pd.DataFrame(columns=["rally_uid", candidate_col, "utility", "source", "source_family"])
    source_scores = _v444_source_scores(oof_report, target)
    rows: list[pd.DataFrame] = []
    for col in scores.columns:
        if not col.endswith(pred_suffix):
            continue
        variant = col[: -len(pred_suffix)]
        confidence_col = f"{variant}_confidence"
        margin_col = f"{variant}_margin"
        weak_col = f"{variant}_weak_prob_sum"
        part = pd.DataFrame({"rally_uid": scores["rally_uid"], candidate_col: scores[col]})
        part["confidence"] = pd.to_numeric(scores.get(confidence_col, 0.0), errors="coerce").fillna(0.0)
        part["margin"] = pd.to_numeric(scores.get(margin_col, 0.0), errors="coerce").fillna(0.0)
        part["weak_prob_sum"] = pd.to_numeric(scores.get(weak_col, 0.0), errors="coerce").fillna(0.0)
        part["source"] = f"v444_rare_aug_dropout_sweep/{variant}"
        part["source_family"] = "v444_rare_aug_dropout_sweep"
        part["utility"] = (
            part["confidence"]
            + part["margin"]
            + (0.05 * part["weak_prob_sum"])
            + (0.10 * source_scores.get(variant, 0.0))
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["rally_uid", candidate_col, "utility", "source", "source_family"])
    out = pd.concat(rows, ignore_index=True)
    anchor_values = anchor[["rally_uid", f"{target}Id"]].copy()
    out = out.merge(anchor_values, on="rally_uid", how="left")
    out[candidate_col] = pd.to_numeric(out[candidate_col], errors="coerce")
    out = out.loc[out[candidate_col].notna()]
    out[candidate_col] = out[candidate_col].astype(int)
    out = out.loc[out[candidate_col] != pd.to_numeric(out[f"{target}Id"], errors="coerce")].copy()
    return out.drop(columns=[f"{target}Id"]).sort_values("utility", ascending=False).reset_index(drop=True)


def _dedupe_candidates(candidates: pd.DataFrame, *, target: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    candidate_col = f"candidate_{target}Id"
    out = candidates.copy()
    out["utility"] = pd.to_numeric(out["utility"], errors="coerce").fillna(-np.inf)
    out = out.sort_values(["utility", "rally_uid"], ascending=[False, True])
    return out.drop_duplicates(["rally_uid", candidate_col], keep="first").reset_index(drop=True)


def package_professor_candidates(
    anchor: pd.DataFrame,
    *,
    action_candidates: pd.DataFrame | None = None,
    point_candidates: pd.DataFrame | None = None,
    action_top: int = 0,
    point_top: int = 0,
    name: str = "professor_candidate",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply ranked professor candidates to the V362 anchor with safety guards."""

    submission = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    base = submission.copy()
    report: dict[str, Any] = {"name": name}
    safe_action, risky_action = _filter_risky_candidates(action_candidates)
    safe_point, risky_point = _filter_risky_candidates(point_candidates)
    report["risky_source_rows"] = int(risky_action + risky_point)
    report["source_families"] = _source_family_counts(safe_action, safe_point)

    if action_top > 0:
        safe_action = _add_candidate_columns(safe_action, target="action", source="unknown")
        submission, action_report = apply_ranked_candidates(
            submission,
            safe_action,
            target_col="actionId",
            candidate_col="candidate_actionId",
            max_changes=action_top,
            allow_serve_additions=False,
        )
        action_report["blocked_risky_sources"] = int(risky_action)
        report["action"] = action_report
    elif risky_action:
        report["action"] = {"blocked_risky_sources": int(risky_action)}

    if point_top > 0:
        safe_point = _add_candidate_columns(safe_point, target="point", source="unknown")
        submission, point_report = apply_ranked_candidates(
            submission,
            safe_point,
            target_col="pointId",
            candidate_col="candidate_pointId",
            max_changes=point_top,
            allow_point0_additions=False,
        )
        point_report["blocked_risky_sources"] = int(risky_point)
        report["point"] = point_report
    elif risky_point:
        report["point"] = {"blocked_risky_sources": int(risky_point)}

    changed_mask = (submission["actionId"].astype(int).to_numpy() != base["actionId"].astype(int).to_numpy()) | (
        submission["pointId"].astype(int).to_numpy() != base["pointId"].astype(int).to_numpy()
    )
    report["total_changed_rows"] = int(changed_mask.sum())
    report["server_preserved"] = bool(
        np.allclose(
            pd.to_numeric(submission["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(base["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
        )
    )
    report["action_distribution"] = action_distribution_report(base["actionId"], submission["actionId"])
    report["point_distribution"] = point_distribution_report(base["pointId"], submission["pointId"])
    validate_submission_schema(submission, expected_rows=None if len(submission) != 1845 else 1845)
    return submission, report


def _load_professor_candidate_tables(anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    v442_point = _add_candidate_columns(_read_optional_csv(V442_POINT_CANDIDATES), target="point", source="v442_intent_first_sequence_point")
    if not v442_point.empty:
        v442_point["source"] = "v442_intent_first_sequence_point"
        v442_point["source_family"] = "v442_intent_first_sequence_point"
        if "point_margin" in v442_point.columns and "point_confidence" in v442_point.columns:
            v442_point["utility"] = pd.to_numeric(v442_point["point_confidence"], errors="coerce").fillna(0.0) + pd.to_numeric(
                v442_point["point_margin"], errors="coerce"
            ).fillna(0.0)

    oof_report = _read_optional_csv(V444_OOF_REPORT)
    v444_action = _candidates_from_v444_sweep(anchor, _read_optional_csv(V444_ACTION_SWEEP), oof_report, target="action")
    v444_point = _candidates_from_v444_sweep(anchor, _read_optional_csv(V444_POINT_SWEEP), oof_report, target="point")

    v434_action = _add_candidate_columns(_read_optional_csv(V434_ACTION_CANDIDATES), target="action", source="v434_anchor_aware_moe_gate")
    v434_point = _add_candidate_columns(_read_optional_csv(V434_POINT_CANDIDATES), target="point", source="v434_anchor_aware_moe_gate")
    for frame in (v434_action, v434_point):
        if not frame.empty:
            frame["source"] = "v434_anchor_aware_moe_gate"
            frame["source_family"] = "v434_anchor_aware_moe_gate"

    action_candidates = _dedupe_candidates(pd.concat([v434_action, v444_action], ignore_index=True), target="action")
    point_candidates = _dedupe_candidates(pd.concat([v442_point, v434_point, v444_point], ignore_index=True), target="point")
    style_embeddings = _read_optional_csv(V443_STYLE_EMBEDDINGS)
    metadata = {
        "v442_point_rows": int(len(v442_point)),
        "v443_style_rows": int(len(style_embeddings)),
        "v443_style_feature_columns": int(len([c for c in style_embeddings.columns if c != "rally_uid"])),
        "v444_action_rows": int(len(v444_action)),
        "v444_point_rows": int(len(v444_point)),
        "v434_action_rows": int(len(v434_action)),
        "v434_point_rows": int(len(v434_point)),
        "combined_action_rows": int(len(action_candidates)),
        "combined_point_rows": int(len(point_candidates)),
        "source_families": _source_family_counts(action_candidates, point_candidates),
    }
    return action_candidates, point_candidates, metadata


def _report_row(report: dict[str, Any]) -> dict[str, Any]:
    action = report.get("action", {})
    point = report.get("point", {})
    return {
        "name": report.get("name", ""),
        "filename": report.get("filename", ""),
        "total_changed_rows": report.get("total_changed_rows", 0),
        "action_changed_rows": action.get("applied_changes", 0),
        "point_changed_rows": point.get("applied_changes", 0),
        "blocked_point0_additions": point.get("blocked_point0_additions", 0),
        "blocked_serve_additions": action.get("blocked_serve_additions", 0),
        "blocked_risky_sources": int(action.get("blocked_risky_sources", 0)) + int(point.get("blocked_risky_sources", 0)),
        "server_preserved": report.get("server_preserved", False),
        "source_families": json.dumps(report.get("source_families", {}), sort_keys=True),
    }


def run_packager() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR_PATH).loc[:, SUBMISSION_COLUMNS].copy()
    validate_submission_schema(anchor)
    action_candidates, point_candidates, input_metadata = _load_professor_candidate_tables(anchor)

    configs = [
        ("point_top5", 0, 5, "submission_v445_professor_point_top5__v362anchor.csv"),
        ("point_top10", 0, 10, "submission_v445_professor_point_top10__v362anchor.csv"),
        ("joint_top5", 5, 5, "submission_v445_professor_joint_top5__v362anchor.csv"),
        ("joint_top10", 10, 10, "submission_v445_professor_joint_top10__v362anchor.csv"),
        ("private_probe_top20", 20, 20, "submission_v445_professor_private_probe_top20__v362anchor.csv"),
    ]
    reports: list[dict[str, Any]] = []
    exports: list[str] = []
    for name, action_top, point_top, filename in configs:
        submission, report = package_professor_candidates(
            anchor,
            action_candidates=action_candidates,
            point_candidates=point_candidates,
            action_top=action_top,
            point_top=point_top,
            name=name,
        )
        path = safe_output_path(OUTDIR, filename)
        submission.to_csv(path, index=False)
        report["filename"] = filename
        reports.append(report)
        exports.append(str(path))

    packaging_report = pd.DataFrame([_report_row(report) for report in reports])
    packaging_report.to_csv(safe_output_path(OUTDIR, "packaging_report.csv"), index=False)
    summary = {
        "version": "V445",
        "anchor_path": str(ANCHOR_PATH),
        "anchor_rows": int(len(anchor)),
        "inputs": input_metadata,
        "exports": exports,
        "reports": reports,
        "risky_source_rows_filtered": int(sum(row["blocked_risky_sources"] for row in packaging_report.to_dict("records"))),
        "risky_source_families": [],
    }
    write_json(safe_output_path(OUTDIR, "summary.json"), summary)
    return summary


def main() -> None:
    summary = run_packager()
    print(json.dumps({"outdir": str(OUTDIR), "exports": len(summary["exports"])}, indent=2))


if __name__ == "__main__":
    main()
