"""V406 public-response meta model.

Report-only scorer for post-V400 candidate metadata. It parses historical
public PL records from experiments_log.md, reads ranked candidate CSVs from the
current clean research line, and scores transfer risk without writing
submissions.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v406_public_response_meta_model"
CURRENT_ANCHOR_PL = 0.3590124
TRAIN_MIN_LABELS = 6

RANKED_INPUTS = [
    ("v383", ROOT / "v383_synthetic_adjusted_packager" / "ranked_candidates.csv"),
    ("v387", ROOT / "v387_expanded_synthetic_packager" / "ranked_candidates.csv"),
    ("v391", ROOT / "v391_oof_gated_submission_packager" / "ranked_candidates.csv"),
    ("v400", ROOT / "v400_public_component_recombination" / "ranked_candidates.csv"),
    ("v401", ROOT / "v401_action_point_compatibility" / "ranked_candidates.csv"),
    ("v402", ROOT / "v402_rare_point_specialist_lab" / "ranked_candidates.csv"),
    ("v403", ROOT / "v403_neural_posterior_gate" / "ranked_candidates.csv"),
    ("v405", ROOT / "v405_v362_pruning_lab" / "ranked_candidates.csv"),
]

PUBLIC_FALLBACKS = [
    ("V300", "submission_v300_best_safe_repack__v173action_v261point_server.csv", 0.3576975),
    ("V306", "submission_v306_p0_cap0p01__v173action_v300server.csv", 0.3577905),
    ("V307", "submission_v307_budget24__v173action_v300server.csv", 0.3577789),
    ("V338", "submission_v338_joint_moe_no_p0.csv", 0.3590041),
    ("V341", "submission_v341_no_p0_expansion.csv", 0.3581101),
    ("V353", "submission_v353_v338_prune2.csv", 0.3590041),
    ("V362", "submission_v362_depth_agree_only__v173action_v300server.csv", 0.3590124),
    ("V391", "submission_v391_oof_point_top36__v173action_v300server.csv", 0.3578818),
]

FEATURE_COLUMNS = [
    "point_churn",
    "action_churn",
    "server_changed",
    "point0_additions",
    "point0_removals",
    "longside_transition_count",
    "half_boundary_transition_count",
    "short_control_transition_count",
    "synthetic_proxy_flag",
    "public_positive_component_agreement_flag",
    "posterior_model_flag",
    "specialist_flag",
]


def _norm(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def _to_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else float(parsed)


def _to_int(value: object, default: int = 0) -> int:
    return int(round(_to_float(value, float(default))))


def candidate_key(value: object) -> str:
    text = _norm(value).lower().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    text = re.sub(r"\.csv$", "", text)
    text = re.sub(r"^submission_", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _candidate_from_table(header: list[str], cells: list[str]) -> str | None:
    for wanted in ("id", "version", "candidate", "name"):
        if wanted in header and cells[header.index(wanted)].strip():
            return cells[header.index(wanted)].strip(" `")
    for wanted in ("file", "submission", "path"):
        if wanted in header and cells[header.index(wanted)].strip():
            return Path(cells[header.index(wanted)].strip(" `")).name
    return None


def parse_public_pl_records(log_text: str, *, include_fallback: bool = False) -> pd.DataFrame:
    """Parse historical public PL rows from markdown tables and inline notes."""

    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        lowered = [cell.lower() for cell in cells]
        if any(("public" in cell or cell == "pl") and ("pl" in cell or "lb" in cell) for cell in lowered):
            header = lowered
            continue
        if not header or len(cells) != len(header):
            continue
        pl_idx = next(
            (
                idx
                for idx, name in enumerate(header)
                if ("public" in name and ("pl" in name or "lb" in name)) or name == "pl"
            ),
            None,
        )
        if pl_idx is None:
            continue
        match = re.search(r"\b0\.\d{4,}\b", cells[pl_idx])
        candidate = _candidate_from_table(header, cells)
        if match and candidate:
            rows.append(
                {
                    "candidate": candidate,
                    "source": "experiments_log_table",
                    "public_pl": float(match.group(0)),
                }
            )

    inline_patterns = [
        re.compile(
            r"(?P<candidate>(?:submission_)?(?:v|V|r|R)\d+[A-Za-z0-9_.\-]*)"
            r"(?:(?!\n).){0,100}?\bPL\b\s*[:=]?\s*`?(?P<pl>0\.\d{4,})`?",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<candidate>(?:submission_)?(?:v|V|r|R)\d+[A-Za-z0-9_.\-]*)"
            r"(?:(?!\n).){0,100}?Public\s+(?:LB\s*/\s*)?PL\s*[:=]?\s*`?(?P<pl>0\.\d{4,})`?",
            re.IGNORECASE,
        ),
    ]
    for pattern in inline_patterns:
        for match in pattern.finditer(log_text):
            rows.append(
                {
                    "candidate": match.group("candidate").strip("`.,"),
                    "source": "experiments_log_inline",
                    "public_pl": float(match.group("pl")),
                }
            )

    if include_fallback:
        for version, candidate, pl in PUBLIC_FALLBACKS:
            rows.append({"candidate": candidate, "source": f"fallback_{version}", "public_pl": pl})

    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = candidate_key(row["candidate"])
        if not key:
            continue
        current = by_key.get(key)
        if current is None or not str(row["source"]).startswith("fallback_"):
            by_key[key] = row

    table = pd.DataFrame(by_key.values())
    if table.empty:
        table = pd.DataFrame(columns=["candidate", "source", "public_pl"])
    table["candidate_key"] = table["candidate"].map(candidate_key)
    table["version"] = table["candidate_key"].str.extract(r"\b(v\d+|r\d+)\b", expand=False).fillna("")
    table = table.sort_values(["public_pl", "candidate_key"], ascending=[False, True]).reset_index(drop=True)
    table["closest_anchor_pl"] = table["public_pl"].shift(-1).fillna(CURRENT_ANCHOR_PL)
    table["pl_delta_vs_closest_anchor"] = table["public_pl"].astype(float) - table["closest_anchor_pl"].astype(float)
    table["positive_transfer"] = table["pl_delta_vs_closest_anchor"] > 0
    return table


def _read_ranked(path: Path, source_label: str) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        return pd.DataFrame(), False
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(), True
    if frame.empty:
        return frame, True
    frame = frame.copy()
    frame["source_label"] = source_label
    frame["source_ranked_path"] = str(path)
    return frame, True


def _candidate_path(row: pd.Series) -> str:
    for col in ("path", "candidate_path", "submission_path"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _selected_rows_path(row: pd.Series) -> str:
    for col in ("selected_rows_path", "selected_rows", "selected_path"):
        value = row.get(col)
        if isinstance(value, str) and value.strip() and value.strip().lower().endswith(".csv"):
            return value.strip()
    return ""


def _resolve_path(root: Path, path_text: str, row: pd.Series | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    direct = root / path
    if direct.exists():
        return direct
    if row is not None:
        ranked_path = row.get("source_ranked_path")
        if isinstance(ranked_path, str) and ranked_path:
            relative_to_ranked = Path(ranked_path).parent / path
            if relative_to_ranked.exists():
                return relative_to_ranked
    return direct


def source_family(candidate: str, path: str, source_label: str) -> str:
    text = f"{candidate} {path} {source_label}".lower()
    if "v391" in text or "oof" in text or "proxy" in text:
        return "proxy_oof"
    if "synthetic" in text or "synth" in text or source_label in {"v383", "v387"}:
        return "synthetic"
    if "public_agree" in text or "public_component" in text:
        return "public_positive_agreement"
    if "posterior" in text or source_label == "v403":
        return "posterior"
    if "specialist" in text or source_label == "v402":
        return "specialist"
    if "compat" in text or source_label == "v401":
        return "compatibility"
    if source_label == "v405":
        return "v362_pruning"
    return source_label or "unknown"


def _transition_bucket(old_point: int, new_point: int) -> str:
    values = {old_point, new_point}
    if values <= {7, 8, 9}:
        return "longside"
    if values & {4, 5, 6}:
        return "half_boundary"
    if values & {1, 2, 3}:
        return "short_control"
    return "other"


def _parse_transition_counts(value: object) -> dict[str, int]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in parsed.items():
        if isinstance(key, str) and "->" in key:
            counts[key] = _to_int(count)
    return counts


def transition_features(row: pd.Series, *, root: Path) -> dict[str, int]:
    counts = _parse_transition_counts(row.get("transition_counts"))
    longside = half = short = point0_removals = 0
    if counts:
        for transition, count in counts.items():
            try:
                old_text, new_text = transition.split("->", 1)
                old_point, new_point = int(old_text), int(new_text)
            except ValueError:
                continue
            bucket = _transition_bucket(old_point, new_point)
            longside += count if bucket == "longside" else 0
            half += count if bucket == "half_boundary" else 0
            short += count if bucket == "short_control" else 0
            point0_removals += count if old_point == 0 and new_point != 0 else 0
        return {
            "longside_transition_count": longside,
            "half_boundary_transition_count": half,
            "short_control_transition_count": short,
            "point0_removals": point0_removals,
        }

    selected_path = _selected_rows_path(row)
    if selected_path:
        path = _resolve_path(root, selected_path, row)
        if path.exists():
            try:
                selected = pd.read_csv(path)
            except (OSError, EmptyDataError):
                selected = pd.DataFrame()
            old_col = next((c for c in ("anchor_point", "old_point", "base_point", "point_old") if c in selected), None)
            new_col = next((c for c in ("new_point", "candidate_point", "point_new") if c in selected), None)
            if old_col and new_col:
                for old_value, new_value in zip(selected[old_col], selected[new_col]):
                    old_point, new_point = _to_int(old_value), _to_int(new_value)
                    bucket = _transition_bucket(old_point, new_point)
                    longside += 1 if bucket == "longside" else 0
                    half += 1 if bucket == "half_boundary" else 0
                    short += 1 if bucket == "short_control" else 0
                    point0_removals += 1 if old_point == 0 and new_point != 0 else 0
    return {
        "longside_transition_count": longside,
        "half_boundary_transition_count": half,
        "short_control_transition_count": short,
        "point0_removals": point0_removals,
    }


def build_candidate_feature_table(
    *,
    root: Path = ROOT,
    ranked_inputs: list[tuple[str, Path]] | None = None,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    ranked_inputs = ranked_inputs or RANKED_INPUTS
    frames: list[pd.DataFrame] = []
    missing: dict[str, bool] = {}
    for source_label, path in ranked_inputs:
        frame, present = _read_ranked(path, source_label)
        missing[source_label] = not present or frame.empty
        if present and not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["candidate", "path", *FEATURE_COLUMNS]), missing

    raw = pd.concat(frames, ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        candidate = _norm(row.get("candidate")) or Path(_candidate_path(row)).stem
        path = _candidate_path(row)
        family = source_family(candidate, path, _norm(row.get("source_label")))
        features = transition_features(row, root=root)
        text = f"{candidate} {path} {row.get('evidence', '')} {family}".lower()
        rows.append(
            {
                "candidate": candidate,
                "path": path,
                "selected_rows": _selected_rows_path(row),
                "source_label": row.get("source_label", ""),
                "source_family": family,
                "candidate_key": candidate_key(candidate or path),
                "point_churn": _to_int(row.get("point_churn", row.get("selected_row_count", row.get("selected_rows", 0)))),
                "action_churn": _to_int(row.get("action_churn", 0)),
                "server_changed": _to_int(row.get("server_changed", 0)),
                "point0_additions": _to_int(row.get("point0_additions", 0)),
                "point0_removals": features["point0_removals"],
                "longside_transition_count": features["longside_transition_count"],
                "half_boundary_transition_count": features["half_boundary_transition_count"],
                "short_control_transition_count": features["short_control_transition_count"],
                "synthetic_proxy_flag": int(any(token in text for token in ("synthetic", "synth", "proxy", "oof", "v391"))),
                "public_positive_component_agreement_flag": int(
                    any(token in text for token in ("public_agree", "public-positive", "public_positive", "deterministic_public"))
                ),
                "posterior_model_flag": int("posterior" in text),
                "specialist_flag": int("specialist" in text),
                "input_risk": _norm(row.get("risk")),
                "evidence": _norm(row.get("evidence")),
                "rank": _to_int(row.get("rank", len(rows) + 1), len(rows) + 1),
            }
        )
    return pd.DataFrame(rows), missing


def deterministic_score(row: pd.Series) -> tuple[float, str, str]:
    score = 0.50
    reasons: list[str] = ["deterministic risk model"]
    risk = "normal"

    if _to_int(row.get("public_positive_component_agreement_flag")):
        score += 0.22
        reasons.append("public-positive component agreement")
    if _to_int(row.get("specialist_flag")):
        score += 0.05
        reasons.append("specialist source")
    if _to_int(row.get("posterior_model_flag")):
        score -= 0.06
        reasons.append("posterior source")
    if _to_int(row.get("synthetic_proxy_flag")):
        score -= 0.28
        risk = "high"
        reasons.append("synthetic/proxy-like source")
    if "v391" in f"{row.get('candidate', '')} {row.get('path', '')}".lower():
        score -= 0.35
        risk = "high"
        reasons.append("V391 public fail lineage")

    point_churn = _to_int(row.get("point_churn"))
    action_churn = _to_int(row.get("action_churn"))
    point0_additions = _to_int(row.get("point0_additions"))
    server_changed = _to_int(row.get("server_changed"))
    score -= min(point_churn, 80) * 0.006
    score -= action_churn * 0.035
    score -= point0_additions * 0.025
    score -= server_changed * 0.040
    if point_churn > 30:
        risk = "high" if risk == "high" else "medium"
        reasons.append("point churn above 30")
    if point0_additions > 0:
        risk = "high" if risk == "high" else "medium"
        reasons.append("point0 additions")
    if action_churn > 0:
        risk = "high"
        reasons.append("action churn")

    score += min(_to_int(row.get("longside_transition_count")), 12) * 0.010
    score += min(_to_int(row.get("half_boundary_transition_count")), 8) * 0.004
    score -= min(_to_int(row.get("short_control_transition_count")), 8) * 0.003
    if not math.isfinite(score):
        score = 0.0
    score = float(max(-1.0, min(1.0, score)))
    if risk == "normal" and score >= 0.60 and point_churn <= 15:
        risk = "low"
    return score, risk, "; ".join(reasons)


def _fit_or_choose_mode(history: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    labeled_count = int(history["public_pl"].notna().sum()) if "public_pl" in history else 0
    positives = int(history["positive_transfer"].sum()) if "positive_transfer" in history else 0
    negatives = labeled_count - positives
    if labeled_count < TRAIN_MIN_LABELS or positives == 0 or negatives == 0:
        return "deterministic_risk_model", {
            "trained": False,
            "labeled_public_examples": labeled_count,
            "reason": "fewer than 6 labeled examples or single-class labels",
        }
    return "deterministic_risk_model", {
        "trained": False,
        "labeled_public_examples": labeled_count,
        "reason": "candidate rows do not have enough exact public labels for reliable training",
    }


def score_candidates(candidates: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    mode, model_info = _fit_or_choose_mode(history)
    if candidates.empty:
        scored = pd.DataFrame(columns=[*candidates.columns, "response_score", "risk", "score_reason", "model_mode"])
        return scored, mode, model_info
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        score, risk, reason = deterministic_score(row)
        record = row.to_dict()
        record.update({"response_score": score, "risk": risk, "score_reason": reason, "model_mode": mode})
        rows.append(record)
    scored = pd.DataFrame(rows)
    risk_order = {"low": 0, "normal": 1, "medium": 2, "high": 3}
    scored["_risk_order"] = scored["risk"].map(risk_order).fillna(2)
    scored = scored.sort_values(
        ["_risk_order", "response_score", "point_churn", "rank"],
        ascending=[True, False, True, True],
    ).drop(columns=["_risk_order"]).reset_index(drop=True)
    scored["meta_rank"] = range(1, len(scored) + 1)
    return scored, mode, model_info


def run_pipeline(*, outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "experiments_log.md"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    history = parse_public_pl_records(log_text, include_fallback=True)
    candidates, missing_inputs = build_candidate_feature_table(root=ROOT)
    scored, mode, model_info = score_candidates(candidates, history)

    history_path = outdir / "historical_public_response_table.csv"
    scores_path = outdir / "candidate_response_scores.csv"
    ranked_path = outdir / "ranked_candidates.csv"
    report_path = outdir / "search_report.json"
    history.to_csv(history_path, index=False)
    scored.to_csv(scores_path, index=False)
    scored.to_csv(ranked_path, index=False)

    risk_counts = scored["risk"].value_counts().to_dict() if "risk" in scored else {}
    report = {
        "version": "V406",
        "model_mode": mode,
        "model_info": model_info,
        "historical_public_records": int(len(history)),
        "candidate_score_count": int(len(scored)),
        "risk_counts": {str(k): int(v) for k, v in risk_counts.items()},
        "missing_inputs": missing_inputs,
        "policy": {
            "generated_submissions": False,
            "wrote_upload_candidates": False,
            "used_ttmatch": False,
            "used_old_server_branch": False,
        },
        "outputs": {
            "historical_public_response_table": str(history_path),
            "candidate_response_scores": str(scores_path),
            "ranked_candidates": str(ranked_path),
            "search_report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
