from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v315_public_response_meta_selector"
CURRENT_PUBLIC_BEST_FILE = "submission_v306_p0_cap0p01__v173action_v300server.csv"
CURRENT_PUBLIC_BEST_PL = 0.3577905


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
    score: float = 0.0
    rationale: str = ""


PUBLIC_POSITIVE_BRANCHES = {
    "v173 action": 0.20,
    "v188 low-churn point": 0.12,
    "v261 low-churn point": 0.18,
    "v306 low-churn point0": 0.30,
    "v300 server": 0.10,
}

PUBLIC_NEGATIVE_PATTERNS = {
    "weak action": -0.30,
    "v166 full action": -0.45,
    "v272 style microedit": -0.18,
    "v277 style microedit": -0.18,
    "v291 style microedit": -0.22,
    "v295 style microedit": -0.22,
    "v297 style microedit": -0.20,
    "v298 style microedit": -0.20,
    "v299 style microedit": -0.20,
    "v301 style microedit": -0.20,
}

KNOWN_PUBLIC_RESULTS = {
    "submission_v306_p0_cap0p01__v173action_v300server.csv": CURRENT_PUBLIC_BEST_PL,
    "submission_v300_best_safe_repack__v173action_v261point_server.csv": 0.3576975,
    "submission_v261_cap0p01__v173action_r121server.csv": 0.3576720,
    "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv": 0.3573932,
    "submission_v272_point_actioncond_cap0p010__v173action_r121server.csv": 0.3576159,
    "submission_v277_nonterminal_cap0p010__v173action_r121server.csv": 0.3574825,
    "submission_v220_weakonly_churn0p005__pv188cap5__sr121.csv": 0.3542440,
    "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv": 0.3509562,
}


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


def parse_public_results_from_log(path: Path = ROOT / "experiments_log.md") -> dict[str, float]:
    results = dict(KNOWN_PUBLIC_RESULTS)
    if not path.exists():
        return results
    row_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*[^|]*\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = row_re.match(line)
        if not match:
            continue
        _version, file_name, public_pl = match.groups()
        file_name = file_name.strip(" `")
        if file_name.lower() in {"file", "not generated"}:
            continue
        results[Path(file_name).name] = _to_float(public_pl)
    return results


def classify_branch(candidate_file: str, row: dict[str, Any] | None = None) -> str:
    row = row or {}
    name = Path(candidate_file).name.lower()
    source_family = str(row.get("source_family", "")).lower()
    server_source = str(row.get("server_source", "")).lower()
    if name == CURRENT_PUBLIC_BEST_FILE.lower():
        return "V306 low-churn point0"
    if "v309" in name and "v306p0cap0p01" in name:
        return "V309 server-only on V306"
    if "v314" in name and "v306_p0" in name:
        return "V314 clean server on V306"
    if "v314" in name and "v307_p0_budget24" in name:
        return "V314 clean server on V307 budget24"
    if "v314" in name and "v307_p0_cap0p02" in name:
        return "V314 clean server on V307 cap0p02"
    if "v306_p0_" in name:
        return "V306 low-churn point0"
    if "v307" in name:
        return "V307 point0 dose extension"
    if "v308" in name:
        return "V308 point0 row ablation"
    if "v300" in name or source_family == "v300" or server_source.startswith("v300"):
        return "V300 server"
    if "v302" in name or source_family == "v302":
        return "V302/V309 server calibration"
    if "v310" in name or "terminal" in name:
        return "V310 action consistency"
    if "v301" in name:
        return "V301 style microedit"
    if "v299" in name:
        return "V299 style microedit"
    if "v298" in name:
        return "V298 style microedit"
    if "v297" in name:
        return "V297 style microedit"
    if "v295" in name:
        return "V295 style microedit"
    if "v291" in name:
        return "V291 style microedit"
    if "v277" in name:
        return "V277 style microedit"
    if "v272" in name:
        return "V272 style microedit"
    return "unclassified"


def infer_risk_tier(row: dict[str, Any], branch: str, changed_rows: int, server_changed_rows: int) -> str:
    decision = str(row.get("decision", "")).upper()
    branch_l = branch.lower()
    if "do_not_upload" in decision.lower() or any(key in branch_l for key in ["weak action", "style microedit", "v310"]):
        return "high"
    if changed_rows <= 24 and server_changed_rows <= 1 and "point0" in branch_l:
        return "low"
    if "server" in branch_l and changed_rows == 0:
        return "medium"
    if changed_rows <= 36 and "review" in decision.lower():
        return "medium"
    return "medium"


def candidate_from_row(
    row: dict[str, Any],
    source_report: str,
    public_results: dict[str, float],
) -> CandidateEvidence | None:
    candidate_file = str(row.get("submission") or row.get("candidate") or row.get("path") or "")
    if not candidate_file:
        return None
    candidate_file = Path(candidate_file).name
    branch = classify_branch(candidate_file, row)
    local_delta = _to_float(
        row.get("literal_oof_delta", row.get("action_oof_delta", row.get("local_delta", 0.0)))
    )
    changed_rows = _to_int(
        row.get(
            "test_changed_rows",
            row.get(
                "changed_action_rows",
                row.get("point_changed_rows_vs_v306_anchor", row.get("point_changed_rows_vs_point_anchor", 0)),
            ),
        )
    )
    point0_additions = _to_int(row.get("point0_additions", row.get("test_point0_rows", 0)))
    action_rows = _to_int(row.get("changed_action_rows", row.get("action_changed_rows_vs_v306_anchor", 0)))
    server_rows = _to_int(row.get("server_changed_rows", 0))
    if row.get("server_mad_vs_v306_best_server") not in (None, "", 0):
        server_rows = max(server_rows, 1)
    if row.get("server_mad_vs_current_v306") not in (None, "", 0):
        server_rows = max(server_rows, 1)
    action_server_changed_rows = action_rows + server_rows
    public_pl = public_results.get(candidate_file)
    risk_tier = infer_risk_tier(row, branch, changed_rows, action_server_changed_rows)
    return CandidateEvidence(
        candidate_file=candidate_file,
        branch=branch,
        public_pl=public_pl,
        local_delta=local_delta,
        changed_rows=changed_rows,
        point0_additions=point0_additions,
        action_server_changed_rows=action_server_changed_rows,
        risk_tier=risk_tier,
        source_report=source_report,
    )


def _iter_candidate_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in (
        "top_review_candidates",
        "review_candidates",
        "v306_search_best",
        "top_server_variants",
        "top_server_point_combinations",
        "top3_candidates",
    ):
        rows = report.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                yield row
            elif isinstance(row, str):
                yield {"candidate": Path(row).name, "path": row}
    best = report.get("best_candidate")
    if isinstance(best, dict):
        yield best


def discover_report_jsons(root: Path = ROOT) -> list[Path]:
    reports: list[Path] = []
    for path in root.glob("v3*_*/v3*_report.json"):
        match = re.search(r"v(\d+)", path.name.lower())
        if match and 300 <= int(match.group(1)) <= 314:
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
        for row in _iter_candidate_rows(report):
            candidate = candidate_from_row(row, str(report_path.relative_to(root)), public_results)
            if candidate is None:
                continue
            previous = candidates.get(candidate.candidate_file)
            if previous is None or candidate.local_delta > previous.local_delta:
                candidates[candidate.candidate_file] = candidate
    if CURRENT_PUBLIC_BEST_FILE not in candidates:
        candidates[CURRENT_PUBLIC_BEST_FILE] = CandidateEvidence(
            candidate_file=CURRENT_PUBLIC_BEST_FILE,
            branch="V306 low-churn point0",
            public_pl=CURRENT_PUBLIC_BEST_PL,
            local_delta=0.003578457165028276,
            changed_rows=18,
            point0_additions=18,
            action_server_changed_rows=0,
            risk_tier="low",
            source_report="known_public_best",
        )
    return list(candidates.values())


def branch_adjustment(branch: str) -> float:
    branch_l = branch.lower()
    adjustment = 0.0
    for key, value in PUBLIC_POSITIVE_BRANCHES.items():
        if key in branch_l:
            adjustment += value
    for key, value in PUBLIC_NEGATIVE_PATTERNS.items():
        if key in branch_l:
            adjustment += value
    if "v307 point0" in branch_l:
        adjustment += 0.16
    if "v309 server-only" in branch_l or "server calibration" in branch_l:
        adjustment -= 0.06
    if "v314 clean server" in branch_l:
        adjustment += 0.04
    return adjustment


def score_candidate(candidate: CandidateEvidence, mode: str = "conservative") -> tuple[float, str]:
    mode = mode.lower()
    risk_penalty = {"low": 0.0, "medium": 0.12, "high": 0.42}.get(candidate.risk_tier, 0.18)
    local_weight = 55.0 if mode == "conservative" else 95.0
    churn_weight = 0.008 if mode == "conservative" else 0.003
    score = 1.0
    rationale_parts: list[str] = []
    if candidate.public_pl is not None:
        public_delta = candidate.public_pl - CURRENT_PUBLIC_BEST_PL
        score += 0.35 + public_delta * 180.0
        rationale_parts.append(f"public PL {candidate.public_pl:.7f}")
    score += candidate.local_delta * local_weight
    if candidate.local_delta:
        rationale_parts.append(f"local delta {candidate.local_delta:+.6f}")
    branch_adj = branch_adjustment(candidate.branch)
    score += branch_adj
    if branch_adj:
        rationale_parts.append(f"history branch adjustment {branch_adj:+.2f}")
    score -= candidate.changed_rows * churn_weight
    score -= candidate.action_server_changed_rows * (0.055 if mode == "conservative" else 0.025)
    score -= risk_penalty
    if candidate.risk_tier != "low":
        rationale_parts.append(f"{candidate.risk_tier} risk")
    if mode == "conservative" and "budget24" in candidate.candidate_file.lower():
        score += 0.08
        rationale_parts.append("conservative row budget")
    if mode == "aggressive" and "cap0p02" in candidate.candidate_file.lower():
        score += 0.16
        rationale_parts.append("aggressive higher local delta")
    name_l = candidate.candidate_file.lower()
    if "server" in candidate.branch.lower() and "best_safe_repack" in name_l:
        score -= 0.04
        rationale_parts.append("server anchor repack")
    if "v302_meanmix_w_0p25" in name_l:
        score += 0.04
        rationale_parts.append("small server calibration")
    return score, "; ".join(rationale_parts) or "ranked by default evidence"


def rank_candidates(candidates: Iterable[CandidateEvidence], mode: str = "conservative") -> list[CandidateEvidence]:
    ranked: list[CandidateEvidence] = []
    for candidate in candidates:
        score, rationale = score_candidate(candidate, mode)
        ranked.append(
            CandidateEvidence(
                **{**asdict(candidate), "score": score, "rationale": rationale}
            )
        )
    return sorted(ranked, key=lambda row: (-row.score, row.risk_tier, row.changed_rows, row.candidate_file))


def select_upload_queue(candidates: Iterable[CandidateEvidence], mode: str = "conservative", limit: int = 5) -> list[CandidateEvidence]:
    ranked = rank_candidates(candidates, mode=mode)
    selected: list[CandidateEvidence] = []

    def add_first(predicate: Any) -> None:
        if len(selected) >= limit:
            return
        for candidate in ranked:
            if candidate in selected or candidate.risk_tier == "high":
                continue
            if predicate(candidate):
                selected.append(candidate)
                return

    add_first(lambda row: row.candidate_file == CURRENT_PUBLIC_BEST_FILE)
    add_first(lambda row: "v307_p0_budget24__" in row.candidate_file.lower() and row.branch == "V307 point0 dose extension")
    add_first(lambda row: "v307_p0_cap0p02__" in row.candidate_file.lower() and row.branch == "V307 point0 dose extension")
    add_first(lambda row: row.branch == "V309 server-only on V306" and "best_safe_repack" not in row.candidate_file.lower())
    add_first(lambda row: row.branch.startswith("V314 clean server") and "v302_meanmix_w_0p25" in row.candidate_file.lower())

    if len(selected) >= limit:
        return sorted(selected[:limit], key=lambda row: -row.score)

    seen_branches: dict[str, int] = {}
    for row in selected:
        seen_branches[row.branch] = seen_branches.get(row.branch, 0) + 1
    for candidate in ranked:
        if candidate in selected:
            continue
        if candidate.risk_tier == "high":
            continue
        branch_count = seen_branches.get(candidate.branch, 0)
        if branch_count >= 2 and candidate.public_pl is None:
            continue
        selected.append(candidate)
        seen_branches[candidate.branch] = branch_count + 1
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda row: -row.score)


def write_outputs(candidates: list[CandidateEvidence], queue: list[CandidateEvidence], mode: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_ranked = rank_candidates(candidates, mode=mode)
    fieldnames = list(asdict(all_ranked[0]).keys()) if all_ranked else list(CandidateEvidence.__dataclass_fields__)
    for path, rows in (
        (OUTDIR / "v315_upload_queue.csv", queue),
        (OUTDIR / "v315_candidate_evidence.csv", all_ranked),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "current_public_best": CURRENT_PUBLIC_BEST_FILE,
        "current_public_best_pl": CURRENT_PUBLIC_BEST_PL,
        "candidate_count": len(candidates),
        "queue_limit": 5,
        "decision_rules": {
            "public_positive": PUBLIC_POSITIVE_BRANCHES,
            "public_negative": PUBLIC_NEGATIVE_PATTERNS,
            "conservative": "penalizes churn and server-only variants; favors V307 budget24 over cap0p02",
            "aggressive": "weights local delta more strongly; favors V307 cap0p02 over budget24",
        },
        "upload_queue": [asdict(row) for row in queue],
        "candidate_evidence": [asdict(row) for row in all_ranked],
    }
    (OUTDIR / "v315_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_lines = [
        "# V315 Public-Response Meta Selector",
        "",
        f"- Mode: `{mode}`",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_FILE}` PL {CURRENT_PUBLIC_BEST_PL:.7f}",
        f"- Parsed candidates: {len(candidates)}",
        "",
        "## Recommended Upload Queue",
        "",
        "| Rank | Candidate | Branch | Public PL | Local Delta | Rows | Point0 Adds | Action/Server Rows | Risk | Rationale |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(queue, 1):
        public = "" if row.public_pl is None else f"{row.public_pl:.7f}"
        md_lines.append(
            f"| {idx} | `{row.candidate_file}` | {row.branch} | {public} | "
            f"{row.local_delta:+.6f} | {row.changed_rows} | {row.point0_additions} | "
            f"{row.action_server_changed_rows} | {row.risk_tier} | {row.rationale} |"
        )
    md_lines.extend(
        [
            "",
            "## Encoded Public-Response Rules",
            "",
            "- Positive history: V173 action, V188/V261/V306 low-churn point changes, and V300 server.",
            "- Negative history: weak action edits, V166 full action, and V272/V277/V291/V295/V297/V298/V299/V301 style microedits.",
            "- Conservative mode limits churn exposure before unproven local-only upside.",
        ]
    )
    (OUTDIR / "v315_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["conservative", "aggressive"], default="conservative")
    args = parser.parse_args(argv)
    candidates = build_candidate_evidence(ROOT)
    queue = select_upload_queue(candidates, mode=args.mode, limit=5)
    write_outputs(candidates, queue, args.mode)
    print(f"Wrote {len(queue)} queue rows to {OUTDIR / 'v315_upload_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
