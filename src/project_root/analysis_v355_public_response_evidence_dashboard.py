"""V355 public-response and row-evidence dashboard.

Integrates V352/V353/V354/V351 reports into one ranked list for quota
decisions. The dashboard is conservative: old-server and TTMATCH candidates are
blocked, point0 additions are penalized, and V338-subset candidates are
preferred over expansions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v355_public_response_evidence_dashboard"
V352_FAMILY = ROOT / "v352_public_response_lab" / "family_response_summary.csv"
V353_CANDIDATES = ROOT / "v353_v338_row_causal_audit" / "candidate_summary.csv"
V354_EVIDENCE = ROOT / "v354_independent_row_evidence" / "row_evidence.csv"
V351_CANDIDATES = ROOT / "v351_v338_pruning_trust_model" / "candidate_summary.csv"


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _to_float_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _to_bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(_to_bool).fillna(default).astype(bool)


def block_policy_violations(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    name_text = (
        out.get("name", pd.Series("", index=out.index)).astype(str)
        + " "
        + out.get("path", pd.Series("", index=out.index)).astype(str)
        + " "
        + out.get("family", pd.Series("", index=out.index)).astype(str)
    ).str.lower()
    out["policy_blocked"] = name_text.str.contains("ttmatch|oldhard|oldsharpen|oldrank|oldserver|old-server")
    return out


def _family_delta_lookup(family_response: pd.DataFrame) -> dict[str, float]:
    if family_response.empty or "family" not in family_response.columns:
        return {}
    delta_col = ""
    for candidate in ("best_public_delta_vs_v338", "best_delta_vs_v338", "public_delta_vs_v338"):
        if candidate in family_response.columns:
            delta_col = candidate
            break
    if not delta_col:
        numeric = [col for col in family_response.columns if col != "family" and pd.api.types.is_numeric_dtype(family_response[col])]
        delta_col = numeric[0] if numeric else ""
    if not delta_col:
        return {}
    return {
        str(row["family"]).lower(): float(row[delta_col])
        for _, row in family_response.iterrows()
        if pd.notna(row.get(delta_col))
    }


def _candidate_evidence_scores(evidence: pd.DataFrame) -> dict[str, float]:
    if evidence.empty:
        return {}
    key_col = "candidate_key" if "candidate_key" in evidence.columns else None
    score_col = "independent_evidence_score" if "independent_evidence_score" in evidence.columns else None
    if key_col is None or score_col is None:
        return {}
    grouped = pd.to_numeric(evidence[score_col], errors="coerce").fillna(0.0).groupby(evidence[key_col].astype(str)).mean()
    return {str(k): float(v) for k, v in grouped.items()}


def _row_evidence_lookup(evidence: pd.DataFrame) -> dict[int, float]:
    if evidence.empty or "row_id" not in evidence.columns or "independent_evidence_score" not in evidence.columns:
        return {}
    work = evidence.copy()
    work["row_id"] = pd.to_numeric(work["row_id"], errors="coerce")
    work["independent_evidence_score"] = pd.to_numeric(work["independent_evidence_score"], errors="coerce")
    work = work.dropna(subset=["row_id", "independent_evidence_score"])
    grouped = work.groupby(work["row_id"].astype(int))["independent_evidence_score"].mean()
    return {int(k): float(v) for k, v in grouped.items()}


def _parse_row_ids(value: Any) -> list[int]:
    if value is None:
        return []
    out: list[int] = []
    for part in str(value).replace(",", " ").split():
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _infer_family(row: pd.Series) -> str:
    text = f"{row.get('name', '')} {row.get('path', '')}".lower()
    if "v338" in text or "v351" in text or "v353" in text or "prune" in text or "trust_top" in text:
        return "v338_subset"
    if "point0" in text or "_p0" in text:
        return "point0_addition"
    if "v341" in text or "expand" in text:
        return "v341_expansion"
    return str(row.get("family", "unknown"))


def rank_candidates(candidates: pd.DataFrame, family_response: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out
    if "name" not in out.columns:
        out["name"] = out.get("candidate", out.get("path", pd.Series([f"candidate_{i}" for i in range(len(out))])))
    if "path" not in out.columns:
        out["path"] = ""
    out["family"] = out.apply(_infer_family, axis=1)
    out = block_policy_violations(out)

    family_delta = _family_delta_lookup(family_response)
    evidence_scores = _candidate_evidence_scores(evidence)
    out["family_public_delta"] = out["family"].astype(str).str.lower().map(family_delta).fillna(0.0)
    out["independent_evidence_score"] = out["name"].astype(str).map(evidence_scores).fillna(0.0)
    row_evidence = _row_evidence_lookup(evidence)
    if row_evidence and "reverted_row_ids" in out.columns:
        means: list[float] = []
        for value in out["reverted_row_ids"]:
            rows = _parse_row_ids(value)
            scores = [row_evidence[row] for row in rows if row in row_evidence]
            means.append(float(np.mean(scores)) if scores else 0.0)
        out["reverted_row_evidence_mean"] = means
        # For pruning candidates, lower evidence on reverted rows is better.
        out["independent_evidence_score"] = out["independent_evidence_score"].astype(float) - out[
            "reverted_row_evidence_mean"
        ].astype(float)
    else:
        out["reverted_row_evidence_mean"] = 0.0

    new_rows = _to_float_series(out, "new_rows_beyond_v338")
    point0 = _to_float_series(out, "point0_additions_vs_v306")
    churn = _to_float_series(out, "point_churn_vs_v338")
    action_ok = _to_bool_series(out, "action_preserved", default=True)
    server_ok = _to_bool_series(out, "server_preserved", default=True)
    subset_bonus = (new_rows.eq(0) & point0.eq(0)).astype(float) * 3.0
    preserve_bonus = (action_ok & server_ok).astype(float) * 1.0
    out["score"] = (
        10.0
        + subset_bonus
        + preserve_bonus
        + out["family_public_delta"].astype(float) * 100.0
        + out["independent_evidence_score"].astype(float) * 0.25
        - point0.astype(float) * 0.45
        - new_rows.astype(float) * 0.35
        - churn.astype(float) * 0.03
        - out["policy_blocked"].astype(float) * 1000.0
    )
    out["recommendation_tier"] = np.where(
        out["policy_blocked"],
        "blocked",
        np.where((new_rows.eq(0) & point0.eq(0)), "top_review", "hold_review"),
    )
    return out.sort_values(["policy_blocked", "score", "name"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)


def top_recommendations(ranked: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    allowed = ranked[~_to_bool_series(ranked, "policy_blocked")].copy()
    if "path" in allowed.columns:
        path = allowed["path"].astype(str).str.strip().str.lower()
        allowed = allowed[path.ne("") & path.ne("nan")].copy()
    if "reverted_row_ids" in allowed.columns:
        key = allowed["reverted_row_ids"].astype(str).str.strip()
        nonempty = key.ne("") & key.str.lower().ne("nan")
        allowed = pd.concat(
            [
                allowed[nonempty].drop_duplicates(subset=["reverted_row_ids"], keep="first"),
                allowed[~nonempty],
            ],
            ignore_index=True,
            sort=False,
        )
        allowed = allowed.sort_values(["score", "name"], ascending=[False, True], kind="mergesort")
    return allowed.head(limit).reset_index(drop=True)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_candidate_sources() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_name, path in [
        ("v353", V353_CANDIDATES),
        ("v351", V351_CANDIDATES),
    ]:
        frame = _read_optional_csv(path)
        if frame.empty:
            continue
        frame["source_report"] = source_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_recommendation(path: Path, top: pd.DataFrame, ranked: pd.DataFrame) -> None:
    lines = ["# V355 Public Response Evidence Dashboard", ""]
    if top.empty:
        lines.append("No clean recommendation is available.")
    else:
        lines.append("## Next Upload Priority")
        lines.append("")
        for _, row in top.iterrows():
            lines.append(f"- `{row.get('name')}`: `{row.get('path')}` score={float(row.get('score', 0.0)):.3f}")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append("- Blocks TTMATCH and old-server candidates.")
    lines.append("- Prefers V338 subset/pruning rows over point0 additions or V341-style expansion.")
    lines.append(f"- Ranked candidates: {len(ranked)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_sources()
    family_response = _read_optional_csv(V352_FAMILY)
    evidence = _read_optional_csv(V354_EVIDENCE)
    ranked = rank_candidates(candidates, family_response, evidence)
    top = top_recommendations(ranked, limit=5)
    ranked_path = OUTDIR / "ranked_candidates.csv"
    top_path = OUTDIR / "next_upload_priority.csv"
    rec_path = OUTDIR / "recommendation.md"
    ranked.to_csv(ranked_path, index=False)
    top.to_csv(top_path, index=False)
    write_recommendation(rec_path, top, ranked)
    report = {
        "outdir": relative_path(OUTDIR),
        "candidate_count": int(len(ranked)),
        "recommendation_count": int(len(top)),
        "top": top[["name", "path", "score", "recommendation_tier"]].to_dict("records") if not top.empty else [],
        "inputs": {
            "v352_family": relative_path(V352_FAMILY) if V352_FAMILY.exists() else None,
            "v353_candidates": relative_path(V353_CANDIDATES) if V353_CANDIDATES.exists() else None,
            "v354_evidence": relative_path(V354_EVIDENCE) if V354_EVIDENCE.exists() else None,
            "v351_candidates": relative_path(V351_CANDIDATES) if V351_CANDIDATES.exists() else None,
        },
        "files": {
            "ranked_candidates": relative_path(ranked_path),
            "next_upload_priority": relative_path(top_path),
            "recommendation": relative_path(rec_path),
        },
    }
    write_json(OUTDIR / "search_report.json", report)
    print(json.dumps(_json_safe({"outdir": OUTDIR, "candidates": len(ranked), "recommendations": len(top)}), indent=2))
    return report


if __name__ == "__main__":
    main()
