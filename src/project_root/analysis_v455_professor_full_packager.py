"""V455 full professor-run residual packager.

The packager consumes optional V447-V450 artifacts plus the already packaged
V445 branch. It never uploads raw model predictions directly; it turns them
into low-churn candidates against the V362 public-best anchor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SERVE_ACTION_CLASSES,
    SUBMISSION_COLUMNS,
    action_distribution_report,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
)
from analysis_v435_residual_packager import apply_ranked_candidates


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v455_professor_full_packager"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
V445_DIR = ROOT / "v445_full_professor_moe_packager"
V449_DIR = ROOT / "v449_intent_gru_point_full"
V450_DIR = ROOT / "v450_deep_rare_expert_suite"
V448_STYLE = ROOT / "v448_neural_response_style_contrastive" / "test_neural_style_embeddings.csv"

RISKY_MARKERS = ("ttmatch_quarantined", "v436", "sony", "v438")


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


def _uid_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _is_risky_source(source: Any) -> bool:
    text = str(source or "").lower()
    return any(marker in text for marker in RISKY_MARKERS)


def filter_full_professor_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply clean-branch candidate safety gates."""

    if rows.empty:
        return pd.DataFrame(columns=["rally_uid", "target", "candidate_value", "anchor_value", "utility", "source"])
    out = rows.copy()
    for col in ["candidate_value", "anchor_value", "utility"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[out["rally_uid"].notna() & out["target"].notna() & out["candidate_value"].notna()].copy()
    out["target"] = out["target"].astype(str)
    out["source"] = out.get("source", "unknown").fillna("unknown").astype(str)
    out = out.loc[~out["source"].map(_is_risky_source)].copy()

    point0_add = (
        out["target"].eq("point")
        & out["candidate_value"].eq(0)
        & out["anchor_value"].fillna(-1).ne(0)
    )
    serve_add = (
        out["target"].eq("action")
        & out["candidate_value"].isin(SERVE_ACTION_CLASSES)
        & ~out["anchor_value"].fillna(-1).isin(SERVE_ACTION_CLASSES)
    )
    out = out.loc[~point0_add & ~serve_add].copy()
    out = out.loc[out["candidate_value"] != out["anchor_value"]].copy()
    out["candidate_value"] = out["candidate_value"].astype(int)
    out["utility"] = out["utility"].fillna(-np.inf)
    return out.sort_values(["utility", "rally_uid"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def summarize_submission_safety(anchor: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    validate_submission_schema(candidate, expected_rows=None if len(candidate) != 1845 else 1845)
    base = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    cand = candidate.loc[:, SUBMISSION_COLUMNS].copy()
    action_changed = base["actionId"].astype(int).to_numpy() != cand["actionId"].astype(int).to_numpy()
    point_changed = base["pointId"].astype(int).to_numpy() != cand["pointId"].astype(int).to_numpy()
    server_preserved = bool(
        np.allclose(
            pd.to_numeric(base["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(cand["serverGetPoint"], errors="coerce").to_numpy(dtype=float),
        )
    )
    action_report = action_distribution_report(base["actionId"], cand["actionId"])
    point_report = point_distribution_report(base["pointId"], cand["pointId"])
    return {
        "rows": int(len(cand)),
        "total_changed_rows": int(np.logical_or(action_changed, point_changed).sum()),
        "action_changed_rows": int(action_changed.sum()),
        "point_changed_rows": int(point_changed.sum()),
        "server_preserved": server_preserved,
        "point0_additions": int(point_report["point0_additions"]),
        "serve_additions": max(0, int(action_report["serve_15_18_delta"])),
        "action_distribution": action_report,
        "point_distribution": point_report,
    }


def _style_bonus() -> pd.DataFrame:
    style = _read_optional_csv(V448_STYLE)
    if style.empty or "rally_uid" not in style.columns:
        return pd.DataFrame(columns=["rally_uid", "style_bonus"])
    numeric = style.drop(columns=["rally_uid"], errors="ignore").select_dtypes(include=[np.number])
    if numeric.empty:
        bonus = np.zeros(len(style), dtype=float)
    elif "style_confidence" in numeric.columns:
        bonus = pd.to_numeric(style["style_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        values = numeric.to_numpy(dtype=float)
        bonus = np.linalg.norm(values, axis=1)
    if np.nanmax(bonus) > np.nanmin(bonus):
        bonus = (bonus - np.nanmin(bonus)) / max(np.nanmax(bonus) - np.nanmin(bonus), 1e-9)
    else:
        bonus = np.zeros_like(bonus)
    return pd.DataFrame({"rally_uid": style["rally_uid"], "style_bonus": bonus})


def _diff_candidates_from_submission(anchor: pd.DataFrame, path: Path, *, source: str, base_utility: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    sub = pd.read_csv(path)
    if list(sub.columns) != SUBMISSION_COLUMNS or len(sub) != len(anchor):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for idx, (base_row, cand_row) in enumerate(zip(anchor.to_dict("records"), sub.to_dict("records"))):
        uid = cand_row["rally_uid"]
        if int(base_row["actionId"]) != int(cand_row["actionId"]):
            rows.append(
                {
                    "rally_uid": uid,
                    "target": "action",
                    "candidate_value": int(cand_row["actionId"]),
                    "anchor_value": int(base_row["actionId"]),
                    "utility": base_utility - idx * 1e-6,
                    "source": source,
                }
            )
        if int(base_row["pointId"]) != int(cand_row["pointId"]):
            rows.append(
                {
                    "rally_uid": uid,
                    "target": "point",
                    "candidate_value": int(cand_row["pointId"]),
                    "anchor_value": int(base_row["pointId"]),
                    "utility": base_utility - idx * 1e-6,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def _v445_candidates(anchor: pd.DataFrame) -> pd.DataFrame:
    frames = []
    weights = {
        "submission_v445_professor_point_top5__v362anchor.csv": 3.0,
        "submission_v445_professor_point_top10__v362anchor.csv": 2.2,
        "submission_v445_professor_joint_top5__v362anchor.csv": 1.8,
        "submission_v445_professor_joint_top10__v362anchor.csv": 1.2,
    }
    for filename, utility in weights.items():
        frames.append(_diff_candidates_from_submission(anchor, V445_DIR / filename, source=f"v445/{filename}", base_utility=utility))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()


def _v449_candidates(anchor: pd.DataFrame) -> pd.DataFrame:
    table = _read_optional_csv(V449_DIR / "point_candidate_table.csv")
    if table.empty:
        return pd.DataFrame()
    candidate_col = "candidate_pointId" if "candidate_pointId" in table.columns else "candidate_value"
    if candidate_col not in table.columns:
        return pd.DataFrame()
    out = table.copy()
    out = out.merge(anchor[["rally_uid", "pointId"]].rename(columns={"pointId": "anchor_value"}), on="rally_uid", how="left")
    utility_col = next((c for c in ["utility", "confidence", "margin", "score"] if c in out.columns), None)
    utility = pd.to_numeric(out[utility_col], errors="coerce") if utility_col else pd.Series(0.5, index=out.index)
    return pd.DataFrame(
        {
            "rally_uid": out["rally_uid"],
            "target": "point",
            "candidate_value": pd.to_numeric(out[candidate_col], errors="coerce"),
            "anchor_value": pd.to_numeric(out["anchor_value"], errors="coerce"),
            "utility": utility.fillna(0.0) + 0.35,
            "source": "v449_intent_gru_point_full",
        }
    )


def expert_candidates_from_scores(anchor: pd.DataFrame, scores: pd.DataFrame, *, target: str) -> pd.DataFrame:
    if scores.empty or "rally_uid" not in scores.columns:
        return pd.DataFrame()
    target_col = "actionId" if target == "action" else "pointId"
    candidate_col_names = [c for c in scores.columns if c.endswith(f"_pred_{target_col}") or c.endswith(f"_candidate_{target_col}")]
    rows = []
    for col in candidate_col_names:
        prefix = col.rsplit("_", 2)[0]
        confidence_col = f"{prefix}_confidence"
        score_col = f"{prefix}_score"
        if confidence_col in scores.columns:
            utility_source = scores[confidence_col]
        elif score_col in scores.columns:
            utility_source = scores[score_col]
        else:
            utility_source = pd.Series([0.25] * len(scores), index=scores.index)
        utility = pd.to_numeric(utility_source, errors="coerce").fillna(0.25)
        part = pd.DataFrame(
            {
                "rally_uid": scores["rally_uid"],
                "target": target,
                "candidate_value": pd.to_numeric(scores[col], errors="coerce"),
                "utility": utility + 0.2,
                "source": f"v450_deep_rare_expert_suite/{prefix}",
            }
        )
        part = part.merge(anchor[["rally_uid", target_col]].rename(columns={target_col: "anchor_value"}), on="rally_uid", how="left")
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _expert_candidates(anchor: pd.DataFrame, *, target: str) -> pd.DataFrame:
    path = V450_DIR / ("action_expert_scores_test.csv" if target == "action" else "point_expert_scores_test.csv")
    scores = _read_optional_csv(path)
    return expert_candidates_from_scores(anchor, scores, target=target)


def collect_full_professor_candidates(anchor: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _v445_candidates(anchor),
        _v449_candidates(anchor),
        _expert_candidates(anchor, target="action"),
        _expert_candidates(anchor, target="point"),
    ]
    candidates = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if candidates.empty:
        return candidates
    style = _style_bonus()
    if not style.empty:
        candidates = candidates.merge(style, on="rally_uid", how="left")
        candidates["utility"] = pd.to_numeric(candidates["utility"], errors="coerce").fillna(0.0) + (
            0.05 * pd.to_numeric(candidates["style_bonus"], errors="coerce").fillna(0.0)
        )
    candidates = filter_full_professor_candidates(candidates)
    candidates["_uid"] = candidates["rally_uid"].map(_uid_key)
    # Keep the strongest proposal for each row/target/value.
    candidates = candidates.sort_values("utility", ascending=False).drop_duplicates(["_uid", "target", "candidate_value"])
    return candidates.drop(columns=["_uid"], errors="ignore").reset_index(drop=True)


def _apply_candidates(anchor: pd.DataFrame, candidates: pd.DataFrame, *, action_top: int, point_top: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    submission = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    report: dict[str, Any] = {}
    action_rows = candidates.loc[candidates["target"].eq("action")].rename(columns={"candidate_value": "candidate_actionId"})
    point_rows = candidates.loc[candidates["target"].eq("point")].rename(columns={"candidate_value": "candidate_pointId"})
    if action_top > 0:
        submission, action_report = apply_ranked_candidates(
            submission,
            action_rows,
            target_col="actionId",
            candidate_col="candidate_actionId",
            max_changes=action_top,
            allow_serve_additions=False,
        )
        report["action"] = action_report
    if point_top > 0:
        submission, point_report = apply_ranked_candidates(
            submission,
            point_rows,
            target_col="pointId",
            candidate_col="candidate_pointId",
            max_changes=point_top,
            allow_point0_additions=False,
        )
        report["point"] = point_report
    report.update(summarize_submission_safety(anchor, submission))
    return submission, report


def run_packager() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_schema(anchor)
    candidates = collect_full_professor_candidates(anchor)
    candidates.to_csv(OUTDIR / "full_candidate_table.csv", index=False)

    novel_candidates = candidates.loc[~candidates["source"].astype(str).str.startswith("v445/")].copy()
    configs = [
        ("full_point_top5", candidates, 0, 5),
        ("full_point_top10", candidates, 0, 10),
        ("full_joint_top5", candidates, 5, 5),
        ("full_joint_top10", candidates, 10, 10),
        ("full_private_probe_top20", candidates, 20, 20),
        ("full_novel_point_top5", novel_candidates, 0, 5),
        ("full_novel_joint_top5", novel_candidates, 5, 5),
    ]
    reports: list[dict[str, Any]] = []
    exports: list[str] = []
    for name, candidate_table, action_top, point_top in configs:
        submission, report = _apply_candidates(anchor, candidate_table, action_top=action_top, point_top=point_top)
        filename = f"submission_v455_{name}__v362anchor.csv"
        path = safe_output_path(OUTDIR, filename)
        submission.to_csv(path, index=False)
        report["name"] = name
        report["filename"] = filename
        report["action_top"] = int(action_top)
        report["point_top"] = int(point_top)
        report["candidate_rows_used"] = int(len(candidate_table))
        reports.append(report)
        exports.append(str(path))

    report_rows = []
    for report in reports:
        report_rows.append(
            {
                "name": report["name"],
                "filename": report["filename"],
                "total_changed_rows": report["total_changed_rows"],
                "action_changed_rows": report["action_changed_rows"],
                "point_changed_rows": report["point_changed_rows"],
                "point0_additions": report["point0_additions"],
                "serve_additions": report["serve_additions"],
                "server_preserved": report["server_preserved"],
            }
        )
    pd.DataFrame(report_rows).to_csv(OUTDIR / "packaging_report.csv", index=False)
    summary = {
        "version": "V455",
        "anchor_path": str(ANCHOR_PATH),
        "candidate_rows": int(len(candidates)),
        "novel_candidate_rows": int(len(novel_candidates)),
        "exports": exports,
        "reports": reports,
    }
    write_json(OUTDIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_packager()
    print(json.dumps(_json_safe({"outdir": str(OUTDIR), "candidate_rows": result["candidate_rows"], "exports": len(result["exports"])}), indent=2))
