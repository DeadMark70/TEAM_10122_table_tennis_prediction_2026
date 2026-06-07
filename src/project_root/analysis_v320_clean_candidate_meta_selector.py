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
OUTDIR = ROOT / "v320_clean_candidate_meta_selector"

CURRENT_PUBLIC_BEST_FILE = "submission_v306_p0_cap0p01__v173action_v300server.csv"
CURRENT_PUBLIC_BEST_PL = 0.3577905
V307_BUDGET24_FILE = "submission_v307_p0_budget24__v173action_v300server.csv"
V307_BUDGET24_PL = 0.3577789


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
REPORT_VERSION_MAX = 319


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
    if "v307_p0_budget24" in name:
        return "V307 point0 dose extension"
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
    return any(row.get(key) not in (None, "", 0, 0.0) for key in ("server_mad_vs_current_v306", "server_mad_vs_v306_best_server"))


def infer_risk_tier(row: dict[str, Any], branch: str, changed_rows: int, point0_additions: int, action_server_rows: int) -> str:
    decision = str(row.get("decision", row.get("recommendation", ""))).upper()
    row_risk = str(row.get("risk", row.get("risk_tier", ""))).lower()
    branch_l = branch.lower()

    if "do_not_upload" in decision.lower() or "high" in row_risk:
        return "high"
    if "action micro" in branch_l or "action specialist" in branch_l:
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
) -> str:
    name = _as_name(candidate_file)
    branch_l = branch.lower()
    decision = str(row.get("decision", row.get("recommendation", ""))).upper()

    if name == CURRENT_PUBLIC_BEST_FILE:
        return "BASELINE"
    changed_rows = _to_int(
        row.get(
            "test_changed_rows",
            row.get(
                "changed_rows",
                row.get(
                    "changed_action_rows",
                    row.get("point_changed_rows_vs_point_anchor", row.get("point_changed_rows_vs_v306_anchor", 0)),
                ),
            ),
        )
    )
    has_server_mad = any(
        _to_float(row.get(key), 0.0) > 0.0
        for key in ("server_mad_vs_current_v306", "server_mad_vs_v306_best_server", "server_mad_vs_v306_server", "server_mad")
    )
    if changed_rows == 0 and not has_server_mad:
        return "DO_NOT_UPLOAD_NOOP"
    if "v306 superseded" in branch_l or "v308 point0" in branch_l:
        return "DO_NOT_UPLOAD_SUPERSEDED"
    if public_pl is not None and public_pl < CURRENT_PUBLIC_BEST_PL:
        return "DO_NOT_UPLOAD_SATURATED"
    if "point0 dose" in branch_l or "saturated point0" in branch_l:
        if point0_additions > 18 or "v307" in branch_l or "v311" in branch_l:
            return "DO_NOT_UPLOAD_SATURATED"
    if "action micro" in branch_l or "action specialist" in branch_l:
        if "DO_NOT_UPLOAD" in decision or _to_float(row.get("action_oof_delta", row.get("local_delta", 0.0))) < 0.0015:
            return "DO_NOT_UPLOAD_ACTION_MICRO"
    if "joint terminal" in branch_l and "DO_NOT_UPLOAD" in decision:
        return "DO_NOT_UPLOAD_JOINT_TERMINAL"
    if "DO_NOT_UPLOAD" in decision and "server" not in branch_l:
        return "DO_NOT_UPLOAD"
    if "server" in branch_l:
        return "CONDITIONAL_QUOTA"
    if risk_tier == "high":
        return "DO_NOT_UPLOAD"
    return "REVIEW"


def candidate_from_row(row: dict[str, Any], source_report: str, public_results: dict[str, float]) -> CandidateEvidence | None:
    candidate_file = str(row.get("candidate_file") or row.get("submission") or row.get("candidate") or row.get("path") or "")
    if not candidate_file:
        return None
    candidate_file = _as_name(candidate_file)
    if not candidate_file.endswith(".csv"):
        return None
    version_match = re.search(r"_v(\d+)", candidate_file.lower())
    if version_match and int(version_match.group(1)) < REPORT_VERSION_MIN and candidate_file not in public_results:
        return None

    branch = classify_branch(candidate_file, row, source_report)
    local_delta = _to_float(
        row.get(
            "literal_oof_delta",
            row.get(
                "local_delta_vs_v306_point_anchor",
                row.get(
                    "action_oof_delta",
                    row.get("server_auc_delta", row.get("oof_auc_delta_vs_anchor", row.get("local_delta", 0.0))),
                ),
            ),
        )
    )
    changed_rows = _to_int(
        row.get(
            "test_changed_rows",
            row.get(
                "changed_rows",
                row.get(
                    "changed_action_rows",
                    row.get("point_changed_rows_vs_point_anchor", row.get("point_changed_rows_vs_v306_anchor", 0)),
                ),
            ),
        )
    )
    point0_additions = _to_int(row.get("point0_additions", row.get("test_point0_additions", 0)))
    action_rows = _to_int(row.get("changed_action_rows", row.get("action_changed_rows_vs_point_anchor", 0)))
    server_rows = _to_int(row.get("server_changed_rows", 0))
    if _is_server_candidate(branch, row):
        if _to_float(row.get("server_mad_vs_current_v306"), 0.0) > 0.0:
            server_rows = max(server_rows, 1)
        if _to_float(row.get("server_mad_vs_v306_best_server"), 0.0) > 0.0:
            server_rows = max(server_rows, 1)
        if _to_float(row.get("server_mad_vs_v306_server"), 0.0) > 0.0:
            server_rows = max(server_rows, 1)
    action_server_rows = action_rows + server_rows

    public_pl = public_results.get(candidate_file)
    already_uploaded = public_pl is not None
    risk_tier = infer_risk_tier(row, branch, changed_rows, point0_additions, action_server_rows)
    recommendation = infer_recommendation(candidate_file, branch, public_pl, row, risk_tier, point0_additions)
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


def discover_report_jsons(root: Path = ROOT) -> list[Path]:
    reports: list[Path] = []
    for path in root.glob("v3*_*/v3*_report.json"):
        match = re.search(r"v(\d+)", path.name.lower())
        if not match:
            continue
        version = int(match.group(1))
        if REPORT_VERSION_MIN <= version <= REPORT_VERSION_MAX or version == 315:
            reports.append(path)
    return sorted(set(reports))


def build_candidate_evidence(root: Path = ROOT) -> list[CandidateEvidence]:
    public_results = parse_public_results_from_log(root / "experiments_log.md")
    candidates: dict[str, CandidateEvidence] = {}

    for report_path in discover_report_jsons(root):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = str(report_path.relative_to(root))
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
    if "nonterminal" in branch:
        score += 0.35
        reasons.append("novel nonterminal branch")
    if "point0 dose" in branch or "saturated point0" in branch:
        score -= 1.10
        reasons.append("point0 expansion saturated after V307 public")
    if "v308 point0" in branch:
        score -= 0.75
        reasons.append("point0 ablation superseded by V306 cap0p01 public result")
    if "action micro" in branch or "action specialist" in branch:
        score -= 0.65
        reasons.append("action micro-edits failed after V220/V291")
    if "server-only" in branch or "clean server" in branch:
        score -= 0.18
        reasons.append("server-only micro held for quota-scarce use")
    if "saturated v307" in branch or "v307_p0" in name:
        score -= 0.45
        reasons.append("inherits saturated V307 point anchor")
    if "v302_meanmix_w_0p25" in name or "mad0p002" in name:
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
        score -= 0.20
        reasons.append("conditional quota-only candidate")

    return score, "; ".join(reasons) or "ranked by default evidence"


def rank_candidates(candidates: Iterable[CandidateEvidence]) -> list[CandidateEvidence]:
    ranked: list[CandidateEvidence] = []
    for candidate in candidates:
        score, rationale = score_candidate(candidate)
        ranked.append(replace(candidate, score=score, rationale=rationale, upload_role="UNASSIGNED"))
    return sorted(ranked, key=lambda row: (-row.score, row.risk_tier, row.changed_rows, row.candidate_file))


def _with_role(candidate: CandidateEvidence, role: str) -> CandidateEvidence:
    return replace(candidate, upload_role=role)


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

    new_uploads = [
        row
        for row in ranked
        if not row.already_uploaded
        and row.risk_tier != "high"
        and not row.recommendation.startswith("DO_NOT_UPLOAD")
        and (row.local_delta > 0.0 or row.recommendation == "CONDITIONAL_QUOTA")
    ]
    for role, row in zip(("FIRST_NEW_UPLOAD", "SECOND_UPLOAD"), new_uploads):
        add(row, role)

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
    add_first_do_not(lambda row: "v311" in row.candidate_file.lower())
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

    for row in ranked:
        if len(queue) >= limit:
            break
        role = "WATCHLIST"
        if row.recommendation == "CONDITIONAL_QUOTA":
            role = "QUOTA_SCARCE_ONLY"
        add(row, role)

    return queue


def write_outputs(candidates: list[CandidateEvidence], queue: list[CandidateEvidence], outdir: Path = OUTDIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    all_ranked = rank_candidates(candidates)
    fields = list(CandidateEvidence.__dataclass_fields__)

    for file_name, rows in (
        ("v320_upload_queue.csv", queue),
        ("v320_candidate_evidence.csv", all_ranked),
    ):
        with (outdir / file_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_public_best": CURRENT_PUBLIC_BEST_FILE,
        "current_public_best_pl": CURRENT_PUBLIC_BEST_PL,
        "v307_budget24_public_pl": V307_BUDGET24_PL,
        "v307_budget24_delta_vs_v306": V307_BUDGET24_PL - CURRENT_PUBLIC_BEST_PL,
        "candidate_count": len(candidates),
        "queue_count": len(queue),
        "policy": {
            "exclude_uploaded_as_new_uploads": True,
            "uploaded_files_kept_only_as_baseline_or_do_not_upload_evidence": True,
            "point0_dose_beyond_v306": "DO_NOT_UPLOAD_SATURATED after V307 budget24 public negative",
            "action_micro_edits_after_v220_v291": "penalized/high risk unless strong new evidence",
            "server_only_micro": "conditional quota-scarce only",
            "no_upload_copy": True,
            "no_ttmatch": True,
            "no_old_server": True,
        },
        "upload_queue": [asdict(row) for row in queue],
        "candidate_evidence": [asdict(row) for row in all_ranked],
    }
    (outdir / "v320_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# V320 Clean Candidate Meta Selector",
        "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_FILE}` PL {CURRENT_PUBLIC_BEST_PL:.7f}",
        f"- V307 budget24 public result: `{V307_BUDGET24_FILE}` PL {V307_BUDGET24_PL:.7f} ({V307_BUDGET24_PL - CURRENT_PUBLIC_BEST_PL:+.7f})",
        f"- Parsed candidates: {len(candidates)}",
        "",
        "## Upload Queue",
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
            "- Already uploaded files are not eligible as new uploads; V306 remains only the public-best baseline.",
            "- V307 budget24 is public-negative, so V307 cap0p02 and V311 budget36-style point0 expansion are marked do-not-upload.",
            "- Action micro-edits remain penalized after V220/V291 public failures unless a later report provides materially stronger evidence.",
            "- Server-only micro candidates are retained only as quota-scarce fallbacks.",
        ]
    )
    (outdir / "v320_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V320 clean candidate meta selector and historical backtest.")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(argv)

    candidates = build_candidate_evidence(ROOT)
    queue = select_upload_queue(candidates, limit=args.limit)
    write_outputs(candidates, queue, OUTDIR)
    print(f"Wrote {len(queue)} queue rows to {OUTDIR / 'v320_upload_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
