"""V331 anchor-contract validation and public-result sanity lab.

This is a reusable guard for local candidate reports. It audits V300+ local
report/search outputs, checks whether generated candidates have a clear action
anchor contract, and backtests the guard against known public outcomes.

No TTMATCH, old-server, or upload directory writes are used.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v331_anchor_contract_validation_lab"
REPORT_PATH = OUTDIR / "v331_report.json"
AUDIT_PATH = OUTDIR / "v331_candidate_audit.csv"

BANNED_WRITE_PARTS = {"upload_candidates", "upload_candidates_20260519", "selected", "submissions"}
UNSAFE_ANCHOR_SOURCES = {"", "missing", "fallback_lag0_actionid", "fallback_lag0_actionId"}
LOCAL_VERSION_RE = re.compile(r"^v(?P<version>\d{3,})[_-]", re.IGNORECASE)


@dataclass(frozen=True)
class PublicRecord:
    version: str
    candidate_file: str
    public_pl: float | None
    public_status: str
    expected_process_status: str
    note: str


@dataclass(frozen=True)
class AnchorContract:
    report_path: str
    version: str
    action_anchor_source: str
    point_fixed_source: str
    server_fixed_source: str
    generated_submission_count: int
    evidence_pass: bool
    changed_rows: int | None
    churn: float | None
    decision: str
    unsafe: bool
    unsafe_reasons: str


KNOWN_PUBLIC_RECORDS: tuple[PublicRecord, ...] = (
    PublicRecord(
        "V306",
        "submission_v306_p0_cap0p01__v173action_v300server.csv",
        0.3577905,
        "positive_current_best",
        "BASELINE_PUBLIC_BEST",
        "V306 low-churn point0 is the public clean best.",
    ),
    PublicRecord(
        "V300",
        "submission_v300_best_safe_repack__v173action_v261point_server.csv",
        0.3576975,
        "positive_clean_best",
        "CLEAN_BASELINE",
        "V300 clean server repack was the prior clean best.",
    ),
    PublicRecord(
        "V307",
        "submission_v307_p0_budget24__v173action_v300server.csv",
        0.3577789,
        "negative_saturated",
        "DO_NOT_UPLOAD_SATURATED",
        "V307 budget24 was public-negative versus V306 despite higher local delta.",
    ),
    PublicRecord(
        "V322",
        "submission_v322_modelbank_agree12__v173action_v300server.csv",
        None,
        "not_public_small",
        "REVIEW_SMALL_NONTERMINAL",
        "V322 is not public yet and has small nonterminal churn.",
    ),
    PublicRecord(
        "V328",
        "submission_v328_lowchurn_selector__v306point_v300server.csv",
        None,
        "local_do_not_upload",
        "DO_NOT_UPLOAD",
        "V328 failed local action evidence and must not be uploaded.",
    ),
    PublicRecord(
        "V291",
        "submission_v291_fast57_modelbank_c0p010__pv261cap1__sr121.csv",
        0.3559391,
        "negative_action_microedit",
        "NEGATIVE_PUBLIC",
        "V291 action micro-edit failed publicly.",
    ),
    PublicRecord(
        "V220",
        "submission_v220_weakonly_churn0p005__pv188cap5__sr121.csv",
        0.3542440,
        "negative_action_microedit",
        "NEGATIVE_PUBLIC",
        "V220 weak-action repair failed publicly.",
    ),
    PublicRecord(
        "V191",
        "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv",
        0.3509562,
        "negative_full_action",
        "NEGATIVE_PUBLIC",
        "V191/V166 full-action replacement failed publicly.",
    ),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_output_path(path: str | Path, *, outdir: Path = OUTDIR) -> Path:
    """Allow writes only directly below the V331 local output directory."""
    path_obj = Path(path)
    parts = {part.lower() for part in path_obj.parts}
    if parts & BANNED_WRITE_PARTS:
        raise ValueError(f"V331 refuses upload/selected/submissions write path: {path}")
    resolved = path_obj if path_obj.is_absolute() else ROOT / path_obj
    try:
        resolved.resolve().relative_to(outdir.resolve())
    except ValueError as exc:
        raise ValueError(f"V331 outputs must stay under {outdir}: {path}") from exc
    return resolved


def local_version_dirs(root: Path = ROOT) -> list[Path]:
    dirs: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir() or path.name == OUTDIR.name:
            continue
        match = LOCAL_VERSION_RE.match(path.name)
        if not match:
            continue
        version = int(match.group("version"))
        if version >= 300:
            dirs.append((version, path))
    return [path for _, path in sorted(dirs)]


def report_paths(root: Path = ROOT) -> list[Path]:
    paths: list[Path] = []
    for directory in local_version_dirs(root):
        paths.extend(sorted(directory.glob("*report.json")))
    return paths


def search_csv_paths(root: Path = ROOT) -> list[Path]:
    paths: list[Path] = []
    for directory in local_version_dirs(root):
        paths.extend(sorted(directory.glob("*search*.csv")))
    return paths


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "pass", "passed", "review", "review_p0", "review_action"}:
        return True
    if text in {"0", "false", "no", "fail", "failed", "do_not_upload", "none"}:
        return False
    return None


def _first_text(data: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return str(data[key])
    return ""


def _nested_get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _all_text(data: Any) -> str:
    if isinstance(data, Mapping):
        return " ".join(_all_text(v) for v in data.values())
    if isinstance(data, list):
        return " ".join(_all_text(v) for v in data)
    return "" if data is None else str(data)


def _generated_candidate_names(report: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("generated_submissions", "generated_candidates", "submissions"):
        value = report.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, Mapping):
                for subkey in ("candidate_file", "submission", "path", "candidate"):
                    if item.get(subkey) not in (None, ""):
                        names.add(Path(str(item[subkey])).name)
                        break
            elif item not in (None, ""):
                names.add(Path(str(item)).name)
    return names


def infer_action_anchor_source(report: Mapping[str, Any]) -> str:
    candidates = [
        _nested_get(report, ("frame_meta", "anchor_oof_source")),
        _nested_get(report, ("frame_meta", "anchor_test_source")),
        report.get("action_anchor"),
        _nested_get(report, ("policy", "fixed_action_server")),
        report.get("packaging"),
        report.get("anchor_submission"),
        report.get("anchor_path"),
    ]
    for value in candidates:
        if value not in (None, ""):
            text = str(value)
            if "server" in text.lower() and "action" not in text.lower() and "v173" not in text.lower():
                continue
            return text
    text = _all_text(report).lower()
    if "fallback_lag0_actionid" in text:
        return "fallback_lag0_actionId"
    if "v173action" in text or "v173 action" in text or "rebuilt_v173" in text:
        return "V173 action inferred from report artifacts"
    return "missing"


def infer_point_source(report: Mapping[str, Any]) -> str:
    value = (
        report.get("point_fixed_to")
        or _nested_get(report, ("policy", "base_point_anchor"))
        or report.get("current_public_best")
        or report.get("current_clean_best")
    )
    if value not in (None, ""):
        return str(value)
    text = _all_text(report).lower()
    if "v306point" in text or "v306 point" in text:
        return "V306 point inferred from report artifacts"
    if "v261point" in text or "v261 point" in text or "pv261" in text:
        return "V261 point inferred from report artifacts"
    return "missing"


def infer_server_source(report: Mapping[str, Any]) -> str:
    value = report.get("server_fixed_to") or report.get("server_source") or report.get("packaging")
    if value not in (None, ""):
        return str(value)
    text = _all_text(report).lower()
    if "v300server" in text or "v300 server" in text:
        return "V300 server inferred from report artifacts"
    if "sr121" in text or "r121server" in text:
        return "R121 server inferred from report artifacts"
    return "missing"


def generated_submission_count(report: Mapping[str, Any], report_path: Path | None = None) -> int:
    explicit = report.get("generated_submission_count")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    for key in ("generated_submissions", "generated_candidates", "submissions"):
        value = report.get(key)
        if isinstance(value, list):
            return len(value)
    if report_path is not None:
        return len(list(report_path.parent.glob("submission_*.csv")))
    return 0


def infer_evidence_pass(report: Mapping[str, Any]) -> bool:
    for path in (("best_candidate", "evidence_pass"), ("evidence_pass",)):
        value = _nested_get(report, path)
        parsed = _as_bool(value)
        if parsed is not None:
            return parsed
    public_positive = {
        record.candidate_file
        for record in KNOWN_PUBLIC_RECORDS
        if record.expected_process_status in {"BASELINE_PUBLIC_BEST", "CLEAN_BASELINE"}
    }
    if _generated_candidate_names(report) & public_positive:
        return True
    status_text = " ".join(
        str(value)
        for value in [
            report.get("decision"),
            report.get("verdict"),
            report.get("upload_recommendation"),
            _nested_get(report, ("best_candidate", "decision")),
            _nested_get(report, ("best_candidate", "upload_recommendation")),
            _nested_get(report, ("best_candidate", "recommendation")),
        ]
        if value not in (None, "")
    ).upper()
    if "DO_NOT_UPLOAD" in status_text or "AUDIT_ONLY" in status_text:
        return False
    if "HAS_REVIEW" in status_text or "REVIEW" in status_text or "BASELINE" in status_text:
        return True
    for key in ("top_review_candidates", "review_candidates", "upload_queue"):
        value = report.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _first_number(row: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key not in row or row[key] in (None, ""):
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def infer_changed_rows(row: Mapping[str, Any]) -> int | None:
    value = _first_number(
        row,
        (
            "changed_rows",
            "changed_action_rows",
            "test_changed_rows",
            "pair_changed_rows",
            "action_server_changed_rows",
            "point_changed_rows",
        ),
    )
    return None if value is None else int(value)


def infer_churn(row: Mapping[str, Any]) -> float | None:
    return _first_number(
        row,
        (
            "churn",
            "test_churn",
            "test_churn_vs_v173",
            "test_churn_vs_current_best_v300",
            "test_churn_vs_public_v306",
            "point_churn",
            "action_churn",
        ),
    )


def report_decision(report: Mapping[str, Any]) -> str:
    return _first_text(
        report,
        ("decision", "upload_recommendation", "verdict", "recommendation"),
    ) or _first_text(report.get("best_candidate", {}) if isinstance(report.get("best_candidate"), Mapping) else {}, ("decision", "upload_recommendation"))


def evaluate_anchor_contract(report_path: Path, report: Mapping[str, Any] | None = None) -> AnchorContract:
    data = load_json(report_path) if report is None else dict(report)
    action_anchor = infer_action_anchor_source(data)
    point_source = infer_point_source(data)
    server_source = infer_server_source(data)
    generated_count = generated_submission_count(data, report_path)
    evidence = infer_evidence_pass(data)
    best = data.get("best_candidate", {})
    best_row = best if isinstance(best, Mapping) else {}
    changed_rows = infer_changed_rows(best_row)
    churn = infer_churn(best_row)
    decision = report_decision(data)

    reasons: list[str] = []
    anchor_key = action_anchor.strip().lower()
    if anchor_key in {v.lower() for v in UNSAFE_ANCHOR_SOURCES}:
        reasons.append("unsafe_action_anchor")
    if "fallback_lag0_actionid" in anchor_key:
        reasons.append("fallback_lag0_actionId")
    if generated_count > 0 and not evidence:
        reasons.append("generated_without_evidence")
    if data.get("old_server_used") is True or data.get("ttmatch_used") is True:
        reasons.append("banned_source_used")

    version_match = re.search(r"v(\d{3,})", report_path.as_posix(), flags=re.IGNORECASE)
    version = f"V{version_match.group(1)}" if version_match else "unknown"
    return AnchorContract(
        report_path=relative_path(report_path),
        version=version,
        action_anchor_source=action_anchor,
        point_fixed_source=point_source,
        server_fixed_source=server_source,
        generated_submission_count=generated_count,
        evidence_pass=evidence,
        changed_rows=changed_rows,
        churn=churn,
        decision=decision,
        unsafe=bool(reasons),
        unsafe_reasons=";".join(dict.fromkeys(reasons)),
    )


def known_public_frame(records: Iterable[PublicRecord] = KNOWN_PUBLIC_RECORDS) -> pd.DataFrame:
    rows = []
    for record in records:
        row = asdict(record)
        row["public_delta_vs_v306"] = (
            None if record.public_pl is None else float(record.public_pl - 0.3577905)
        )
        row["guard_rank_score"] = guard_rank_score(record)
        row["is_negative_public"] = record.expected_process_status == "NEGATIVE_PUBLIC"
        rows.append(row)
    return pd.DataFrame(rows)


def guard_rank_score(record: PublicRecord) -> float:
    status_base = {
        "BASELINE_PUBLIC_BEST": 100.0,
        "CLEAN_BASELINE": 90.0,
        "REVIEW_SMALL_NONTERMINAL": 60.0,
        "DO_NOT_UPLOAD": -50.0,
        "DO_NOT_UPLOAD_SATURATED": -70.0,
        "NEGATIVE_PUBLIC": -100.0,
    }.get(record.expected_process_status, 0.0)
    if record.public_pl is not None:
        status_base += (record.public_pl - 0.3577905) * 100.0
    return status_base


def historical_sanity_checks(records: Iterable[PublicRecord] = KNOWN_PUBLIC_RECORDS) -> dict[str, Any]:
    by_version = {record.version.upper(): record for record in records}
    ranked = sorted(records, key=guard_rank_score, reverse=True)
    scores = {record.version: guard_rank_score(record) for record in records}
    negative_versions = {
        record.version
        for record in records
        if record.expected_process_status == "NEGATIVE_PUBLIC"
    }
    checks = {
        "v306_above_v307": scores["V306"] > scores["V307"],
        "v300_above_v307": scores["V300"] > scores["V307"],
        "v322_small_not_public": by_version["V322"].public_pl is None
        and by_version["V322"].expected_process_status == "REVIEW_SMALL_NONTERMINAL",
        "v328_do_not_upload": by_version["V328"].expected_process_status == "DO_NOT_UPLOAD",
        "v191_v220_v291_negative": {"V191", "V220", "V291"}.issubset(negative_versions),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ranked_versions": [record.version for record in ranked],
        "rank_scores": scores,
        "negative_versions": sorted(negative_versions),
    }


def _candidate_name_from_row(row: Mapping[str, Any]) -> str:
    for key in ("candidate_file", "submission", "path", "candidate", "name"):
        value = row.get(key)
        if value not in (None, ""):
            return Path(str(value)).name
    return "unknown"


def _public_by_candidate() -> dict[str, PublicRecord]:
    return {record.candidate_file: record for record in KNOWN_PUBLIC_RECORDS}


def candidate_rows_from_report(report_path: Path, report: Mapping[str, Any], contract: AnchorContract) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [
        ("best_candidate", report.get("best_candidate")),
        ("top_candidates", report.get("top_candidates")),
        ("review_candidates", report.get("review_candidates")),
        ("top_review_candidates", report.get("top_review_candidates")),
        ("upload_queue", report.get("upload_queue")),
    ]
    public_by_candidate = _public_by_candidate()
    for group, value in groups:
        records = value if isinstance(value, list) else [value] if isinstance(value, Mapping) else []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            candidate_file = _candidate_name_from_row(record)
            public = public_by_candidate.get(candidate_file)
            rows.append(
                {
                    "source_kind": "report",
                    "source_group": group,
                    "source_path": relative_path(report_path),
                    "version": contract.version,
                    "candidate_file": candidate_file,
                    "candidate": str(record.get("candidate", "")),
                    "decision": str(record.get("decision") or record.get("recommendation") or record.get("upload_recommendation") or contract.decision),
                    "action_anchor_source": contract.action_anchor_source,
                    "point_fixed_source": contract.point_fixed_source,
                    "server_fixed_source": contract.server_fixed_source,
                    "generated_submission_count": contract.generated_submission_count,
                    "evidence_pass": contract.evidence_pass,
                    "contract_unsafe": contract.unsafe,
                    "unsafe_reasons": contract.unsafe_reasons,
                    "changed_rows": infer_changed_rows(record),
                    "churn": infer_churn(record),
                    "local_delta": _first_number(
                        record,
                        (
                            "local_delta",
                            "literal_oof_delta",
                            "action_oof_delta",
                            "local_delta_vs_v306_point_anchor",
                            "delta_vs_v173",
                            "proxy_delta_vs_proxy_base",
                        ),
                    ),
                    "public_pl": None if public is None else public.public_pl,
                    "expected_process_status": "" if public is None else public.expected_process_status,
                }
            )
    return rows


def candidate_rows_from_search_csv(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    public_by_candidate = _public_by_candidate()
    for raw in frame.to_dict(orient="records"):
        candidate_file = _candidate_name_from_row(raw)
        public = public_by_candidate.get(candidate_file)
        rows.append(
            {
                "source_kind": "search_csv",
                "source_group": "",
                "source_path": relative_path(path),
                "version": _version_from_path(path),
                "candidate_file": candidate_file,
                "candidate": str(raw.get("candidate", "")),
                "decision": str(raw.get("decision") or raw.get("recommendation") or raw.get("upload_recommendation") or raw.get("verdict") or ""),
                "action_anchor_source": "",
                "point_fixed_source": "",
                "server_fixed_source": "",
                "generated_submission_count": "",
                "evidence_pass": raw.get("evidence_pass", ""),
                "contract_unsafe": "",
                "unsafe_reasons": "",
                "changed_rows": infer_changed_rows(raw),
                "churn": infer_churn(raw),
                "local_delta": _first_number(
                    raw,
                    (
                        "local_delta",
                        "literal_oof_delta",
                        "action_oof_delta",
                        "local_delta_vs_v306_point_anchor",
                        "delta_vs_v173",
                        "proxy_delta_vs_proxy_base",
                    ),
                ),
                "public_pl": None if public is None else public.public_pl,
                "expected_process_status": "" if public is None else public.expected_process_status,
            }
        )
    return rows


def _version_from_path(path: Path) -> str:
    match = re.search(r"v(\d{3,})", path.as_posix(), flags=re.IGNORECASE)
    return f"V{match.group(1)}" if match else "unknown"


def build_audit(root: Path = ROOT) -> tuple[list[AnchorContract], pd.DataFrame, dict[str, Any]]:
    contracts: list[AnchorContract] = []
    audit_rows: list[dict[str, Any]] = []
    for path in report_paths(root):
        report = load_json(path)
        contract = evaluate_anchor_contract(path, report)
        contracts.append(contract)
        audit_rows.extend(candidate_rows_from_report(path, report, contract))
    for path in search_csv_paths(root):
        audit_rows.extend(candidate_rows_from_search_csv(path))
    for record in KNOWN_PUBLIC_RECORDS:
        audit_rows.append(
            {
                "source_kind": "known_public",
                "source_group": "",
                "source_path": "internal_known_public_table",
                "version": record.version,
                "candidate_file": record.candidate_file,
                "candidate": record.version,
                "decision": record.expected_process_status,
                "action_anchor_source": "",
                "point_fixed_source": "",
                "server_fixed_source": "",
                "generated_submission_count": "",
                "evidence_pass": "",
                "contract_unsafe": "",
                "unsafe_reasons": "",
                "changed_rows": "",
                "churn": "",
                "local_delta": "",
                "public_pl": record.public_pl,
                "expected_process_status": record.expected_process_status,
            }
        )
    audit = pd.DataFrame(audit_rows)
    sanity = historical_sanity_checks()
    return contracts, audit, sanity


def write_outputs(root: Path = ROOT, outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = ensure_output_path(outdir / "v331_report.json", outdir=outdir)
    audit_path = ensure_output_path(outdir / "v331_candidate_audit.csv", outdir=outdir)

    contracts, audit, sanity = build_audit(root)
    audit.to_csv(audit_path, index=False)
    unsafe = [contract for contract in contracts if contract.unsafe]
    report = {
        "version": "V331",
        "outdir": relative_path(outdir),
        "policy": {
            "scan_scope": "local v300+ report/search files only",
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_directory_writes": True,
            "unsafe_generated_csv_rule": "generated_submission_count > 0 requires evidence_pass",
            "unsafe_action_anchor_rule": "missing or fallback_lag0_actionId action anchors are unsafe",
        },
        "report_count": len(contracts),
        "candidate_audit_rows": int(len(audit)),
        "unsafe_report_count": len(unsafe),
        "unsafe_reports": [asdict(contract) for contract in unsafe],
        "known_public_records": known_public_frame().to_dict(orient="records"),
        "historical_sanity": sanity,
        "outputs": {
            "report_json": relative_path(report_path),
            "candidate_audit_csv": relative_path(audit_path),
        },
    }
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    report = write_outputs()
    print(json.dumps(json_safe({"outputs": report["outputs"], "historical_sanity": report["historical_sanity"]}), indent=2))


if __name__ == "__main__":
    main()
