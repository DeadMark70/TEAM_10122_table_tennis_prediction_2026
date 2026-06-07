from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v325_meta_selector_round2"

CURRENT_PUBLIC_BEST_FILE = "submission_v306_p0_cap0p01__v173action_v300server.csv"
CURRENT_PUBLIC_BEST_PL = 0.3577905
V307_BUDGET24_FILE = "submission_v307_p0_budget24__v173action_v300server.csv"
V307_BUDGET24_PL = 0.3577789
V319_STRONGEST_SERVER_DELTA = 0.02966814279739871


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_file: str
    branch: str
    public_pl: float | None
    local_delta: float
    changed_rows: int
    point0_additions: int
    action_server_changed_rows: int
    risk_tier: str
    source_report: str
    already_uploaded: bool = False
    recommendation: str = "REVIEW"
    evidence_margin_vs_v319: float = 0.0
    score: float = 0.0
    rationale: str = ""
    upload_role: str = "UNASSIGNED"


KNOWN_PUBLIC_RESULTS = {
    CURRENT_PUBLIC_BEST_FILE: CURRENT_PUBLIC_BEST_PL,
    V307_BUDGET24_FILE: V307_BUDGET24_PL,
    "submission_v300_best_safe_repack__v173action_v261point_server.csv": 0.3576975,
    "submission_v261_cap0p01__v173action_r121server.csv": 0.3576720,
    "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv": 0.3573932,
    "submission_v291_fast57_modelbank_c0p010__pv261cap1__sr121.csv": 0.3559391,
    "submission_v220_weakonly_churn0p005__pv188cap5__sr121.csv": 0.3542440,
    "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv": 0.3509562,
}

REPORT_VERSION_MIN = 306
REPORT_VERSION_MAX = 324


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _as_name(value: str) -> str:
    return Path(str(value).strip().strip("`")).name


def parse_public_results_from_log(path: Path = ROOT / "experiments_log.md") -> dict[str, float]:
    results = dict(KNOWN_PUBLIC_RESULTS)
    if not path.exists():
        return results

    text = path.read_text(encoding="utf-8", errors="ignore")
    table_re = re.compile(r"^\|\s*[^|]+?\s*\|\s*[^|]*\|\s*([^|]+?\.csv)\s*\|\s*([0-9.]+)\s*\|", re.MULTILINE)
    for file_name, public_pl in table_re.findall(text):
        results[_as_name(file_name)] = _to_float(public_pl)

    pending: str | None = None
    file_re = re.compile(r"submission_[A-Za-z0-9_.,+\-=()]+[^`\s|/\\]*?\.csv")
    pl_re = re.compile(r"\bPL\s*=\s*([0-9.]+)|Public score:\s*([0-9.]+)|PL score:\s*([0-9.]+)", re.IGNORECASE)
    for line in text.splitlines():
        files = file_re.findall(line)
        if files:
            pending = _as_name(files[-1])
        pl_match = pl_re.search(line)
        if pending and pl_match:
            score = next(group for group in pl_match.groups() if group)
            results[pending] = _to_float(score)
            pending = None
    return results


def classify_branch(candidate_file: str, row: dict[str, Any] | None = None, source_report: str = "") -> str:
    row = row or {}
    name = _as_name(candidate_file).lower()
    source = source_report.lower()
    source_family = str(row.get("source_family", "")).lower()
    point_anchor = str(row.get("point_anchor", "")).lower()
    server_source = str(row.get("server_source", "")).lower()

    if name == CURRENT_PUBLIC_BEST_FILE.lower():
        return "V306 public best baseline"
    if "v306_p0" in name:
        return "V306 superseded point0 variant"
    if row.get("branch"):
        return str(row["branch"])
    if "v321" in name or "v321" in source:
        return "V321 server robust rankblend"
    if "v322" in name or "v322" in source:
        return "V322 nonterminal point model bank"
    if "v323" in name or "v323" in source:
        return "V323 action disagreement mining"
    if "v324" in name or "v324" in source:
        return "V324 clean external corpus audit"
    if "v307_p0" in name:
        return "V307 point0 dose extension"
    if "v311" in name or "v311" in source:
        return "V311 saturated point0 expansion"
    if "v312" in name or "v312" in source:
        return "V312 action micro-edit"
    if "v313" in name or "v313" in source:
        return "V313 joint terminal consistency"
    if "v314" in name or "v314" in source:
        if "v307_p0_budget24" in name or point_anchor == "v307_p0_budget24":
            return "V314 server-only on saturated V307 budget24"
        if "v307_p0_cap0p02" in name or point_anchor == "v307_p0_cap0p02":
            return "V314 server-only on saturated V307 cap0p02"
        return "V314 server-only on V306"
    if "v315" in source:
        return str(row.get("branch", "V315 historical selector"))
    if "v316" in name or "v316" in source:
        return "V316 nonterminal point correction"
    if "v317" in name or "v317" in source:
        return "V317 action specialist"
    if "v318" in name or "v318" in source:
        return "V318 joint nonterminal consistency"
    if "v319" in name or "v319" in source:
        return "V319 clean server value state"
    if "v309" in name or "v309" in source:
        return "V309 server-only on V306"
    if "v308" in name or "v308" in source:
        return "V308 point0 row ablation"
    if source_family.startswith("v300") or server_source.startswith("v300"):
        return "V300 public-proven server"
    return "unclassified"


def _is_server_candidate(branch: str, row: dict[str, Any]) -> bool:
    branch_l = branch.lower()
    if "server" in branch_l:
        return True
    return any(
        _to_float(row.get(key), 0.0) > 0.0
        for key in (
            "server_mad_vs_current_v306",
            "server_mad_vs_v306_best_server",
            "server_mad_vs_v306_server",
            "server_mad",
            "target_mad",
        )
    )


def _changed_rows(row: dict[str, Any]) -> int:
    return _to_int(
        row.get(
            "test_changed_rows",
            row.get(
                "changed_rows",
                row.get(
                    "changed_action_rows",
                    row.get(
                        "point_changed_rows_vs_point_anchor",
                        row.get("point_changed_rows_vs_v306_anchor", row.get("point_changed_rows_vs_anchor", 0)),
                    ),
                ),
            ),
        )
    )


def _local_delta(row: dict[str, Any]) -> float:
    return _to_float(
        row.get(
            "literal_oof_delta",
            row.get(
                "local_delta_vs_v306_point_anchor",
                row.get(
                    "action_oof_delta",
                    row.get(
                        "server_auc_delta",
                        row.get("oof_auc_delta_vs_anchor", row.get("mean_source_evidence_delta", row.get("local_delta", 0.0))),
                    ),
                ),
            ),
        )
    )


def _server_margin_vs_v319(row: dict[str, Any], local_delta: float) -> float:
    explicit = row.get("evidence_margin_vs_v319", row.get("delta_vs_v319", None))
    if explicit not in (None, ""):
        return _to_float(explicit)
    return local_delta - V319_STRONGEST_SERVER_DELTA


def infer_risk_tier(row: dict[str, Any], branch: str, changed_rows: int, point0_additions: int, action_server_rows: int) -> str:
    decision = str(row.get("decision", row.get("recommendation", ""))).upper()
    row_risk = str(row.get("risk", row.get("risk_tier", ""))).lower()
    branch_l = branch.lower()

    if "do_not_upload" in decision.lower() or "high" in row_risk:
        return "high"
    if "local-negative" in decision.lower():
        return "high"
    if "action micro" in branch_l or "action specialist" in branch_l or "action disagreement" in branch_l:
        return "high" if changed_rows >= 10 else "medium"
    if "saturated point0" in branch_l or ("point0 dose" in branch_l and point0_additions > 18):
        return "high"
    if "joint terminal" in branch_l:
        return "high"
    if "server" in branch_l:
        return "medium"
    if point0_additions > 18:
        return "high"
    if changed_rows <= 18 and action_server_rows == 0:
        return "low"
    return "medium"


def infer_recommendation(
    candidate_file: str,
    branch: str,
    public_pl: float | None,
    row: dict[str, Any],
    risk_tier: str,
    point0_additions: int,
    local_delta: float,
    changed_rows: int,
    server_candidate: bool,
    evidence_margin_vs_v319: float,
) -> str:
    name = _as_name(candidate_file)
    branch_l = branch.lower()
    decision = str(row.get("decision", row.get("recommendation", ""))).upper()
    has_server_mad = any(
        _to_float(row.get(key), 0.0) > 0.0
        for key in ("server_mad_vs_current_v306", "server_mad_vs_v306_best_server", "server_mad_vs_v306_server", "server_mad", "target_mad")
    )

    if name == CURRENT_PUBLIC_BEST_FILE:
        return "BASELINE"
    if "RESEARCH_CONTINUE" in decision:
        return "RESEARCH_CONTINUE"
    if changed_rows == 0 and not has_server_mad and not server_candidate:
        return "DO_NOT_UPLOAD_NOOP"
    if local_delta < 0.0:
        return "DO_NOT_UPLOAD_LOCAL_NEGATIVE"
    if "v306 superseded" in branch_l or "v306 low-churn point0" in branch_l or "v308 point0" in branch_l:
        return "DO_NOT_UPLOAD_SUPERSEDED"
    if public_pl is not None and public_pl < CURRENT_PUBLIC_BEST_PL:
        return "DO_NOT_UPLOAD_SATURATED"
    if "point0 dose" in branch_l or "saturated point0" in branch_l:
        if point0_additions > 18 or "v307" in branch_l or "v311" in branch_l:
            return "DO_NOT_UPLOAD_SATURATED"
    if "action micro" in branch_l or "action specialist" in branch_l or "action disagreement" in branch_l:
        if (
            "DO_NOT_UPLOAD" in decision
            or local_delta < 0.002
            or _to_float(row.get("changed_row_oof_precision", 0.0)) < 0.3
        ):
            return "DO_NOT_UPLOAD_ACTION_MICRO"
    if "joint terminal" in branch_l and "DO_NOT_UPLOAD" in decision:
        return "DO_NOT_UPLOAD_JOINT_TERMINAL"
    if "DO_NOT_UPLOAD" in decision and "server" not in branch_l:
        return "DO_NOT_UPLOAD"
    if server_candidate:
        if evidence_margin_vs_v319 > 0.0 and local_delta > V319_STRONGEST_SERVER_DELTA:
            return "REVIEW_SERVER_STRONGER_THAN_V319"
        return "CONDITIONAL_QUOTA"
    if risk_tier == "high":
        return "DO_NOT_UPLOAD"
    return "REVIEW"


def candidate_from_row(row: dict[str, Any], source_report: str, public_results: dict[str, float]) -> CandidateEvidence | None:
    candidate_file = str(
        row.get("candidate_file")
        or row.get("submission")
        or row.get("candidate")
        or row.get("path")
        or row.get("branch_id")
        or ""
    )
    if not candidate_file:
        return None
    candidate_file = _as_name(candidate_file)
    if not candidate_file.endswith(".csv") and str(row.get("recommendation", "")).upper() != "RESEARCH_CONTINUE":
        return None
    version_match = re.search(r"_v(\d+)", candidate_file.lower())
    if version_match and int(version_match.group(1)) < REPORT_VERSION_MIN and candidate_file not in public_results:
        return None

    branch = classify_branch(candidate_file, row, source_report)
    local_delta = _local_delta(row)
    changed_rows = _changed_rows(row)
    point0_additions = _to_int(row.get("point0_additions", row.get("test_point0_additions", 0)))
    action_rows = _to_int(row.get("changed_action_rows", row.get("action_changed_rows_vs_point_anchor", 0)))
    server_rows = _to_int(row.get("server_changed_rows", 0))
    server_candidate = _is_server_candidate(branch, row)
    if server_candidate:
        server_rows = max(server_rows, 1)
    action_server_rows = action_rows + server_rows

    public_pl = public_results.get(candidate_file)
    already_uploaded = public_pl is not None
    evidence_margin_vs_v319 = _server_margin_vs_v319(row, local_delta) if server_candidate else _to_float(row.get("evidence_margin_vs_v319"), 0.0)
    risk_tier = infer_risk_tier(row, branch, changed_rows, point0_additions, action_server_rows)
    recommendation = infer_recommendation(
        candidate_file,
        branch,
        public_pl,
        row,
        risk_tier,
        point0_additions,
        local_delta,
        changed_rows,
        server_candidate,
        evidence_margin_vs_v319,
    )
    if recommendation.startswith("DO_NOT_UPLOAD"):
        risk_tier = "high"

    return CandidateEvidence(
        candidate_file=candidate_file,
        branch=branch,
        public_pl=public_pl,
        local_delta=local_delta,
        changed_rows=changed_rows,
        point0_additions=point0_additions,
        action_server_changed_rows=action_server_rows,
        risk_tier=risk_tier,
        source_report=source_report,
        already_uploaded=already_uploaded,
        recommendation=recommendation,
        evidence_margin_vs_v319=evidence_margin_vs_v319,
    )


def _iter_candidate_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    keys = (
        "top_review_candidates",
        "review_candidates",
        "top5_candidates",
        "top_candidates",
        "top_server_variants",
        "top_server_point_combinations",
        "top3_candidates",
        "upload_queue",
        "candidate_evidence",
        "generated_submissions",
        "submissions",
    )
    for key in keys:
        rows = report.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                yield row
            elif isinstance(row, str):
                yield {"submission": row, "path": row}
    for key in ("best_candidate", "best_review_candidate", "auto_calibrated_candidate"):
        row = report.get(key)
        if isinstance(row, dict) and row:
            yield row


def _research_candidate_from_report(report: dict[str, Any], source: str) -> CandidateEvidence | None:
    version = str(report.get("version", "")).upper()
    if version != "V324" and "v324" not in source.lower():
        return None
    recommendation = report.get("recommendation") or report.get("verdict") or "RESEARCH_CONTINUE"
    summary = "V324 clean external corpus audit"
    return CandidateEvidence(
        candidate_file="v324_clean_external_corpus_audit",
        branch=summary,
        public_pl=None,
        local_delta=0.0,
        changed_rows=0,
        point0_additions=0,
        action_server_changed_rows=0,
        risk_tier="low",
        source_report=source,
        recommendation="RESEARCH_CONTINUE" if "do_not" not in str(recommendation).lower() else "DO_NOT_UPLOAD",
    )


def discover_report_jsons(root: Path = ROOT) -> list[Path]:
    reports: list[Path] = []
    for path in root.glob("v3*_*/v3*_report.json"):
        match = re.search(r"v(\d+)", path.name.lower())
        if not match:
            continue
        version = int(match.group(1))
        if REPORT_VERSION_MIN <= version <= REPORT_VERSION_MAX or version == 315:
            reports.append(path)
    v320_report = root / "v320_clean_candidate_meta_selector" / "v320_report.json"
    if v320_report.exists():
        reports.append(v320_report)
    return sorted(set(reports))


def build_candidate_evidence(root: Path = ROOT) -> list[CandidateEvidence]:
    public_results = parse_public_results_from_log(root / "experiments_log.md")
    candidates: dict[str, CandidateEvidence] = {}

    for report_path in discover_report_jsons(root):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = report_path.relative_to(root).as_posix()
        research_candidate = _research_candidate_from_report(report, source)
        if research_candidate is not None:
            candidates[research_candidate.candidate_file] = research_candidate
        for row in _iter_candidate_rows(report):
            candidate = candidate_from_row(row, source, public_results)
            if candidate is None:
                continue
            previous = candidates.get(candidate.candidate_file)
            if previous is None or _candidate_preferred(candidate, previous):
                candidates[candidate.candidate_file] = candidate

    if CURRENT_PUBLIC_BEST_FILE not in candidates:
        candidates[CURRENT_PUBLIC_BEST_FILE] = CandidateEvidence(
            candidate_file=CURRENT_PUBLIC_BEST_FILE,
            branch="V306 public best baseline",
            public_pl=CURRENT_PUBLIC_BEST_PL,
            local_delta=0.003578457165028276,
            changed_rows=18,
            point0_additions=18,
            action_server_changed_rows=0,
            risk_tier="low",
            source_report="known_public_best",
            already_uploaded=True,
            recommendation="BASELINE",
        )
    return list(candidates.values())


def _candidate_preferred(candidate: CandidateEvidence, previous: CandidateEvidence) -> bool:
    if previous.recommendation.startswith("DO_NOT_UPLOAD") and not candidate.recommendation.startswith("DO_NOT_UPLOAD"):
        return True
    if candidate.public_pl is not None and previous.public_pl is None:
        return True
    if candidate.recommendation == "REVIEW_SERVER_STRONGER_THAN_V319" and previous.recommendation != candidate.recommendation:
        return True
    return candidate.local_delta > previous.local_delta


def _history_adjustment(candidate: CandidateEvidence) -> tuple[float, list[str]]:
    branch = candidate.branch.lower()
    name = candidate.candidate_file.lower()
    score = 0.0
    reasons: list[str] = []

    if "v306 public best baseline" in branch:
        score += 0.40
        reasons.append("public-proven V306 family")
    if "v306 superseded" in branch:
        score -= 0.75
        reasons.append("superseded by public-best V306 cap0p01")
    if "v300" in branch or "v319" in branch:
        score += 0.08
        reasons.append("public-proven server family")
    if "v322" in branch:
        score += 0.50
        reasons.append("round-2 nonterminal branch")
    elif "nonterminal" in branch:
        score += 0.25
        reasons.append("nonterminal branch")
    if "v324" in branch:
        score += 0.65
        reasons.append("strongest clean external-corpus research branch")
    if "point0 dose" in branch or "saturated point0" in branch:
        score -= 1.10
        reasons.append("point0 expansion saturated after V307 public")
    if "v308 point0" in branch:
        score -= 0.75
        reasons.append("point0 ablation superseded by V306 cap0p01 public result")
    if "action micro" in branch or "action specialist" in branch or "action disagreement" in branch:
        score -= 0.65
        reasons.append("action micro-edits need strong changed-row precision")
    if "server" in branch:
        score -= 0.30
        reasons.append("server-only held for quota-scarce use")
    if candidate.recommendation == "REVIEW_SERVER_STRONGER_THAN_V319":
        score += 0.55
        reasons.append("server evidence beats V319")
    if "saturated v307" in branch or "v307_p0" in name:
        score -= 0.45
        reasons.append("inherits saturated V307 point anchor")
    if "value_consensus" in name:
        score += 0.08
        reasons.append("multi-source value consensus")
    if "mad0p001" in name or "mad0p002" in name:
        score += 0.04
        reasons.append("tiny server perturbation")
    return score, reasons


def score_candidate(candidate: CandidateEvidence) -> tuple[float, str]:
    risk_penalty = {"low": 0.0, "medium": 0.25, "high": 1.25}.get(candidate.risk_tier, 0.35)
    score = 1.0
    reasons: list[str] = []

    if candidate.public_pl is not None:
        public_delta = candidate.public_pl - CURRENT_PUBLIC_BEST_PL
        score += public_delta * 220.0
        if candidate.candidate_file == CURRENT_PUBLIC_BEST_FILE:
            score += 0.55
        reasons.append(f"public PL {candidate.public_pl:.7f} ({public_delta:+.7f} vs V306)")

    score += candidate.local_delta * 85.0
    if candidate.local_delta:
        reasons.append(f"local delta {candidate.local_delta:+.6f}")
    if candidate.evidence_margin_vs_v319:
        score += candidate.evidence_margin_vs_v319 * 120.0
        reasons.append(f"evidence margin vs V319 {candidate.evidence_margin_vs_v319:+.6f}")

    history_score, history_reasons = _history_adjustment(candidate)
    score += history_score
    reasons.extend(history_reasons)

    score -= candidate.changed_rows * 0.006
    score -= candidate.action_server_changed_rows * 0.09
    score -= risk_penalty
    if risk_penalty:
        reasons.append(f"{candidate.risk_tier} risk")

    if candidate.already_uploaded and candidate.candidate_file != CURRENT_PUBLIC_BEST_FILE:
        score -= 0.80
        reasons.append("already uploaded")
    if candidate.recommendation.startswith("DO_NOT_UPLOAD"):
        score -= 1.20
        reasons.append(candidate.recommendation)
    elif candidate.recommendation == "CONDITIONAL_QUOTA":
        score -= 0.45
        reasons.append("quota-only server candidate")
    elif candidate.recommendation == "RESEARCH_CONTINUE":
        score += 0.10
        reasons.append("research branch, not an upload")

    return score, "; ".join(reasons) or "ranked by default evidence"


def rank_candidates(candidates: Iterable[CandidateEvidence]) -> list[CandidateEvidence]:
    ranked: list[CandidateEvidence] = []
    for candidate in candidates:
        score, rationale = score_candidate(candidate)
        ranked.append(replace(candidate, score=score, rationale=rationale, upload_role="UNASSIGNED"))
    return sorted(ranked, key=lambda row: (-row.score, row.risk_tier, row.changed_rows, row.candidate_file))


def _with_role(candidate: CandidateEvidence, role: str) -> CandidateEvidence:
    return replace(candidate, upload_role=role)


def _uploadable(row: CandidateEvidence) -> bool:
    if row.already_uploaded or row.risk_tier == "high" or row.recommendation.startswith("DO_NOT_UPLOAD"):
        return False
    if row.recommendation == "RESEARCH_CONTINUE":
        return False
    if row.recommendation == "CONDITIONAL_QUOTA":
        return False
    if row.recommendation == "REVIEW_SERVER_STRONGER_THAN_V319":
        return True
    return row.local_delta > 0.0 and row.changed_rows > 0


def select_upload_queue(candidates: Iterable[CandidateEvidence], limit: int = 8) -> list[CandidateEvidence]:
    ranked = rank_candidates(candidates)
    queue: list[CandidateEvidence] = []
    seen: set[str] = set()

    def add(candidate: CandidateEvidence, role: str) -> None:
        if candidate.candidate_file in seen or len(queue) >= limit:
            return
        queue.append(_with_role(candidate, role))
        seen.add(candidate.candidate_file)

    baseline = next((row for row in ranked if row.candidate_file == CURRENT_PUBLIC_BEST_FILE), None)
    if baseline is not None:
        add(baseline, "BASELINE_PUBLIC_BEST")

    first_upload = next((row for row in ranked if _uploadable(row)), None)
    if first_upload is not None:
        add(first_upload, "FIRST_NEW_UPLOAD")

    research = next((row for row in ranked if row.recommendation == "RESEARCH_CONTINUE"), None)
    if research is not None:
        add(research, "RESEARCH_CONTINUE")
    else:
        continuation = next(
            (
                row
                for row in ranked
                if not row.recommendation.startswith("DO_NOT_UPLOAD")
                and row.recommendation != "BASELINE"
                and not row.already_uploaded
                and "server" not in row.branch.lower()
            ),
            None,
        )
        if continuation is not None:
            add(continuation, "RESEARCH_CONTINUE")

    quota_only = [row for row in ranked if row.recommendation == "CONDITIONAL_QUOTA" and not row.already_uploaded]
    if quota_only:
        add(quota_only[0], "QUOTA_SCARCE_ONLY")

    do_not_uploads = [
        row
        for row in ranked
        if row.recommendation.startswith("DO_NOT_UPLOAD") or (row.already_uploaded and row.candidate_file != CURRENT_PUBLIC_BEST_FILE)
    ]

    def add_first_do_not(predicate: Any) -> None:
        for row in do_not_uploads:
            if predicate(row):
                add(row, "DO_NOT_UPLOAD")
                return

    add_first_do_not(lambda row: row.candidate_file == V307_BUDGET24_FILE)
    add_first_do_not(lambda row: "v311" in row.candidate_file.lower() or "saturated point0" in row.branch.lower())
    add_first_do_not(lambda row: "local_negative" in row.recommendation.lower())
    add_first_do_not(lambda row: "noop" in row.recommendation.lower())
    add_first_do_not(lambda row: "action" in row.branch.lower())
    add_first_do_not(lambda row: "joint" in row.branch.lower())
    add_first_do_not(lambda row: "v306 superseded" in row.branch.lower())

    used_do_not_branches = {row.branch for row in queue if row.upload_role == "DO_NOT_UPLOAD"}
    for row in do_not_uploads:
        if len(queue) >= limit:
            break
        if row.branch in used_do_not_branches and row.public_pl is None:
            continue
        add(row, "DO_NOT_UPLOAD")
        used_do_not_branches.add(row.branch)

    for row in quota_only[1:]:
        add(row, "QUOTA_SCARCE_ONLY")

    for row in ranked:
        if len(queue) >= limit:
            break
        role = "WATCHLIST"
        if row.recommendation == "CONDITIONAL_QUOTA":
            role = "QUOTA_SCARCE_ONLY"
        add(row, role)

    return queue


def _next_actions(queue: list[CandidateEvidence]) -> dict[str, str | None]:
    return {
        "current_clean_best_baseline": next((row.candidate_file for row in queue if row.upload_role == "BASELINE_PUBLIC_BEST"), None),
        "first_new_upload_if_quota_used": next((row.candidate_file for row in queue if row.upload_role == "FIRST_NEW_UPLOAD"), None),
        "strongest_research_branch_to_continue": next((row.branch for row in queue if row.upload_role == "RESEARCH_CONTINUE"), None),
        "quota_only_server_candidate": next((row.candidate_file for row in queue if row.upload_role == "QUOTA_SCARCE_ONLY"), None),
    }


def write_outputs(candidates: list[CandidateEvidence], queue: list[CandidateEvidence], outdir: Path = OUTDIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fields = list(CandidateEvidence.__dataclass_fields__)

    with (outdir / "v325_upload_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in queue:
            writer.writerow(asdict(row))

    all_ranked = rank_candidates(candidates)
    next_actions = _next_actions(queue)
    do_not_upload = [row for row in queue if row.upload_role == "DO_NOT_UPLOAD"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_public_best": CURRENT_PUBLIC_BEST_FILE,
        "current_public_best_pl": CURRENT_PUBLIC_BEST_PL,
        "v307_budget24_public_pl": V307_BUDGET24_PL,
        "v307_budget24_delta_vs_v306": V307_BUDGET24_PL - CURRENT_PUBLIC_BEST_PL,
        "v319_strongest_server_delta": V319_STRONGEST_SERVER_DELTA,
        "candidate_count": len(candidates),
        "queue_count": len(queue),
        "next_actions": next_actions,
        "policy": {
            "exclude_uploaded_as_new_uploads": True,
            "no_noop_uploads": True,
            "no_local_negative_uploads": True,
            "point0_dose_beyond_v306": "DO_NOT_UPLOAD_SATURATED after V307 budget24 public negative",
            "action_micro_edits": "do-not-upload unless strong changed-row evidence clears threshold",
            "server_only_micro": "quota-only unless evidence margin beats V319",
            "no_upload_copy": True,
            "no_ttmatch": True,
            "no_old_server": True,
        },
        "upload_queue": [asdict(row) for row in queue],
        "do_not_upload": [asdict(row) for row in do_not_upload],
        "candidate_evidence": [asdict(row) for row in all_ranked],
    }
    (outdir / "v325_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# V325 Meta Selector Round 2",
        "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_FILE}` PL {CURRENT_PUBLIC_BEST_PL:.7f}",
        f"- V307 budget24 public result: `{V307_BUDGET24_FILE}` PL {V307_BUDGET24_PL:.7f} ({V307_BUDGET24_PL - CURRENT_PUBLIC_BEST_PL:+.7f})",
        f"- Parsed candidates: {len(candidates)}",
        "",
        "## Next Actions",
        "",
        f"- Current clean best baseline: `{next_actions['current_clean_best_baseline'] or ''}`",
        f"- First new upload if quota is used: `{next_actions['first_new_upload_if_quota_used'] or ''}`",
        f"- Strongest research branch to continue: {next_actions['strongest_research_branch_to_continue'] or ''}",
        f"- Quota-only server candidate: `{next_actions['quota_only_server_candidate'] or ''}`",
        "",
        "## Queue",
        "",
        "| Role | Candidate | Branch | Public PL | Local Delta | Rows | Point0 Adds | Risk | Recommendation | Rationale |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in queue:
        public = "" if row.public_pl is None else f"{row.public_pl:.7f}"
        lines.append(
            f"| {row.upload_role} | `{row.candidate_file}` | {row.branch} | {public} | "
            f"{row.local_delta:+.6f} | {row.changed_rows} | {row.point0_additions} | "
            f"{row.risk_tier} | {row.recommendation} | {row.rationale} |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "- No-op and local-negative rows are explicitly do-not-upload, not uploadable review rows.",
            "- Server-only rows remain quota-only unless their evidence margin beats the strongest V319 server evidence.",
            "- Saturated point0 expansion, action micro edits without strong precision, and local-negative nonterminal point rows are penalized.",
            "- This selector writes only V325 queue/report artifacts and does not copy anything to upload or selected folders.",
        ]
    )
    (outdir / "v325_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V325 round-2 meta selector and next-action queue.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    candidates = build_candidate_evidence(ROOT)
    queue = select_upload_queue(candidates, limit=args.limit)
    write_outputs(candidates, queue, OUTDIR)
    print(f"Wrote {len(queue)} queue rows to {OUTDIR / 'v325_upload_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
