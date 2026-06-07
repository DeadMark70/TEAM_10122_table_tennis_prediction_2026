from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v281_ttmatch_diagnostic_validator"
DIAGNOSTIC_SUBMISSION_DIRS = [
    ROOT / "v280_joint_action_point_optimizer",
    ROOT / "v282_joint_context_support_optimizer",
    ROOT / "v283_pair_level_selector",
]
V265_DIR = ROOT / "v265_ttmatch_diagnostic"
ANCHOR_PATH = (
    ROOT
    / "v261_action_conditioned_point_residual"
    / "submission_v261_cap0p01__v173action_r121server.csv"
)

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
DIAGNOSTIC_COLUMNS = [
    "candidate",
    "changed_rows",
    "ttmatch_covered_changed_rows",
    "action_agree_with_ttmatch",
    "point_agree_with_ttmatch",
    "pair_agree_with_ttmatch",
    "ttmatch_disagreement_rate",
    "verdict",
    "diagnostic_source",
]
ALLOWED_VERDICTS = {
    "TTMATCH_DIAG_SUPPORT",
    "TTMATCH_DIAG_CONFLICT",
    "TTMATCH_DIAG_INCONCLUSIVE",
}


@dataclass(frozen=True)
class DiagnosticLookup:
    rows: pd.DataFrame
    source: str


def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def write_empty_diagnostic() -> None:
    pd.DataFrame(columns=DIAGNOSTIC_COLUMNS).to_csv(
        OUTDIR / "v281_ttmatch_diagnostic.csv", index=False
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for record in df.astype(str).to_dict("records"):
        rows.append("| " + " | ".join(record[col] for col in df.columns) + " |")
    return "\n".join([header, separator, *rows])


def write_report(status: str, details: Iterable[str], diagnostics: pd.DataFrame | None) -> None:
    lines = [
        "# V281 TTMATCH Diagnostic Validator",
        "",
        f"status: `{status}`",
        "",
        "TTMATCH is diagnostic-only. This report does not create clean labels, train from TTMATCH, or modify submissions.",
        "",
    ]
    detail_lines = list(details)
    if detail_lines:
        lines.extend(["## Details", ""])
        lines.extend(f"- {line}" for line in detail_lines)
        lines.append("")

    if diagnostics is not None and not diagnostics.empty:
        lines.extend(["## Candidate Diagnostics", ""])
        lines.append(dataframe_to_markdown(diagnostics))
        lines.append("")

    (OUTDIR / "v281_report.md").write_text("\n".join(lines), encoding="utf-8")


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in SUBMISSION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[SUBMISSION_COLUMNS].copy()


def discover_diagnostic_submissions() -> list[Path]:
    paths: list[Path] = []
    for directory in DIAGNOSTIC_SUBMISSION_DIRS:
        if not directory.exists():
            continue
        paths.extend(sorted(directory.glob("submission_v280*.csv")))
        paths.extend(sorted(directory.glob("submission_v282*.csv")))
        paths.extend(sorted(directory.glob("submission_v283*.csv")))
    return sorted(paths)


def normalize_lookup(df: pd.DataFrame, source: str) -> DiagnosticLookup | None:
    required = {"rally_uid", "pred_actionId", "pred_pointId"}
    if required.issubset(df.columns):
        lookup = df.copy()
        if "match_type" in lookup.columns:
            lookup = lookup[lookup["match_type"].astype(str).str.lower() != "none"]
        if "support" in lookup.columns:
            lookup = lookup[pd.to_numeric(lookup["support"], errors="coerce").fillna(0) > 0]
        lookup = lookup.rename(
            columns={"pred_actionId": "ttmatch_actionId", "pred_pointId": "ttmatch_pointId"}
        )
        lookup = lookup[["rally_uid", "ttmatch_actionId", "ttmatch_pointId"]].dropna()
        if not lookup.empty:
            return DiagnosticLookup(lookup.drop_duplicates("rally_uid"), source)
    return None


def load_row_level_diagnostic() -> DiagnosticLookup | None:
    coverage_path = V265_DIR / "v265_match_coverage.csv"
    if coverage_path.exists():
        loaded = normalize_lookup(pd.read_csv(coverage_path), str(coverage_path.relative_to(ROOT)))
        if loaded is not None:
            return loaded

    csv_paths = []
    if V265_DIR.exists():
        csv_paths.extend(sorted(V265_DIR.glob("*.csv")))
    for path in csv_paths:
        loaded = normalize_lookup(pd.read_csv(path), str(path.relative_to(ROOT)))
        if loaded is not None:
            return loaded
    return None


def load_v265_submission_fallback(anchor: pd.DataFrame) -> DiagnosticLookup | None:
    candidates = [
        V265_DIR / "submission_v265_ttmatch_action_point__r121.csv",
        V265_DIR / "submission_v265_ttmatch_action_point__oldsharpen005095.csv",
        V265_DIR / "submission_v265_ttmatch_action_only__v261point_r121.csv",
        V265_DIR / "submission_v265_ttmatch_point_only__v173action_r121.csv",
    ]
    anchor_cmp = anchor[["rally_uid", "actionId", "pointId"]].rename(
        columns={"actionId": "anchor_actionId", "pointId": "anchor_pointId"}
    )
    for path in candidates:
        if not path.exists():
            continue
        df = read_submission(path)
        merged = df.merge(anchor_cmp, on="rally_uid", how="inner")
        changed = merged[
            (merged["actionId"] != merged["anchor_actionId"])
            | (merged["pointId"] != merged["anchor_pointId"])
        ].copy()
        if changed.empty:
            continue
        lookup = changed.rename(
            columns={"actionId": "ttmatch_actionId", "pointId": "ttmatch_pointId"}
        )
        lookup = lookup[["rally_uid", "ttmatch_actionId", "ttmatch_pointId"]]
        return DiagnosticLookup(lookup.drop_duplicates("rally_uid"), str(path.relative_to(ROOT)))
    return None


def load_diagnostic_lookup(anchor: pd.DataFrame) -> DiagnosticLookup | None:
    return load_row_level_diagnostic() or load_v265_submission_fallback(anchor)


def diagnostic_verdict(covered_rows: int, pair_agree: int) -> str:
    if covered_rows <= 0:
        return "TTMATCH_DIAG_INCONCLUSIVE"
    pair_agreement_rate = pair_agree / covered_rows
    if covered_rows >= 5 and pair_agreement_rate >= 0.60:
        return "TTMATCH_DIAG_SUPPORT"
    if covered_rows >= 5 and pair_agreement_rate < 0.30:
        return "TTMATCH_DIAG_CONFLICT"
    return "TTMATCH_DIAG_INCONCLUSIVE"


def evaluate_candidate(
    path: Path, anchor: pd.DataFrame, diagnostic: DiagnosticLookup
) -> dict[str, object]:
    candidate = read_submission(path)
    merged = candidate.merge(
        anchor[["rally_uid", "actionId", "pointId"]].rename(
            columns={"actionId": "anchor_actionId", "pointId": "anchor_pointId"}
        ),
        on="rally_uid",
        how="inner",
        validate="one_to_one",
    )
    changed = merged[
        (merged["actionId"] != merged["anchor_actionId"])
        | (merged["pointId"] != merged["anchor_pointId"])
    ].copy()
    covered = changed.merge(diagnostic.rows, on="rally_uid", how="inner", validate="one_to_one")

    action_agree = int((covered["actionId"] == covered["ttmatch_actionId"]).sum())
    point_agree = int((covered["pointId"] == covered["ttmatch_pointId"]).sum())
    pair_agree = int(
        (
            (covered["actionId"] == covered["ttmatch_actionId"])
            & (covered["pointId"] == covered["ttmatch_pointId"])
        ).sum()
    )
    covered_rows = int(len(covered))
    disagreement_rate = 0.0 if covered_rows == 0 else 1.0 - (pair_agree / covered_rows)
    verdict = diagnostic_verdict(covered_rows, pair_agree)
    if verdict not in ALLOWED_VERDICTS:
        raise AssertionError(f"invalid diagnostic verdict: {verdict}")

    return {
        "candidate": path.name,
        "changed_rows": int(len(changed)),
        "ttmatch_covered_changed_rows": covered_rows,
        "action_agree_with_ttmatch": action_agree,
        "point_agree_with_ttmatch": point_agree,
        "pair_agree_with_ttmatch": pair_agree,
        "ttmatch_disagreement_rate": round(disagreement_rate, 6),
        "verdict": verdict,
        "diagnostic_source": diagnostic.source,
    }


def main() -> None:
    ensure_outdir()
    diagnostic_paths = discover_diagnostic_submissions()
    if not diagnostic_paths:
        write_empty_diagnostic()
        write_report(
            "waiting_for_joint_submissions",
            ["No V280/V282 submissions found under joint optimizer output directories."],
            None,
        )
        print(json.dumps({"outdir": OUTDIR.name, "status": "waiting_for_joint_submissions"}, indent=2))
        return

    if not ANCHOR_PATH.exists():
        write_empty_diagnostic()
        write_report(
            "ttmatch_diagnostic_unavailable",
            [f"Anchor submission not found: {ANCHOR_PATH.relative_to(ROOT)}"],
            None,
        )
        print(
            json.dumps(
                {"outdir": OUTDIR.name, "status": "ttmatch_diagnostic_unavailable"},
                indent=2,
            )
        )
        return

    anchor = read_submission(ANCHOR_PATH)
    diagnostic = load_diagnostic_lookup(anchor)
    if diagnostic is None:
        write_empty_diagnostic()
        write_report(
            "ttmatch_diagnostic_unavailable",
            ["No useful row-level V265/R178/TTMATCH diagnostic source was found."],
            None,
        )
        print(
            json.dumps(
                {"outdir": OUTDIR.name, "status": "ttmatch_diagnostic_unavailable"},
                indent=2,
            )
        )
        return

    rows = [evaluate_candidate(path, anchor, diagnostic) for path in diagnostic_paths]
    diagnostics = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    invalid = sorted(set(diagnostics["verdict"]) - ALLOWED_VERDICTS)
    if invalid:
        raise AssertionError(f"invalid diagnostic verdicts: {invalid}")

    diagnostics.to_csv(OUTDIR / "v281_ttmatch_diagnostic.csv", index=False)
    write_report(
        "ok",
        [
            f"V280/V282 submissions evaluated: {len(diagnostic_paths)}",
            f"Diagnostic source: {diagnostic.source}",
            "Verdicts are diagnostic only and are not clean-validation labels.",
        ],
        diagnostics,
    )
    print(json.dumps({"outdir": OUTDIR.name, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
