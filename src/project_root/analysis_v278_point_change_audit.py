"""V278 changed-row audit for V277 point refinement candidates.

The audit is intentionally independent of the V277 generator so it can run
while Worker A is still producing outputs.  If the V277 search table or
submission files are not present yet, this script writes a waiting report
instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v278_point_change_audit")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
TEST_PATH = Path("test_new.csv")
V277_DIR = Path("v277_v272b_point_refinement")
V277_SEARCH = V277_DIR / "v277_point_search.csv"
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845

CHANGED_ROW_COLUMNS = [
    "candidate",
    "rally_uid",
    "anchor_point",
    "candidate_point",
    "prefix_len",
    "prefix_bin",
    "phase",
    "lag0_actionId",
    "lag0_pointId",
    "lag0_spinId",
    "lag0_strengthId",
    "is_point0_add",
    "is_nonterminal_change",
]

AUDIT_COLUMNS = [
    "candidate",
    "path",
    "v277_verdict",
    "changed_rows",
    "point0_additions",
    "nonterminal_changes",
    "top_phase",
    "top_phase_count",
    "top_phase_share",
    "top_prefix_bin",
    "top_prefix_count",
    "top_prefix_share",
    "top_transition",
    "top_transition_count",
    "top_transition_share",
    "is_concentrated",
    "recommendation",
    "recommendation_reason",
]


def validate_submission_frame(df: pd.DataFrame, *, label: str, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{label} bad columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"{label} bad rows: {len(df)}")
    if df["rally_uid"].duplicated().any():
        raise ValueError(f"{label} duplicate rally_uid values")


def phase_from_prefix(prefix_len: int) -> int:
    val = int(prefix_len)
    if val <= 1:
        return 0
    if val == 2:
        return 1
    if val == 3:
        return 2
    return 3


def prefix_bin(prefix_len: int) -> str:
    val = int(prefix_len)
    if val <= 1:
        return "p1"
    if val == 2:
        return "p2"
    if val == 3:
        return "p3"
    if 4 <= val <= 6:
        return "p4_6"
    return "p7_plus"


def normalize_path(path_value: object) -> Path | None:
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw.replace("\\", "/"))
    return path if path.is_absolute() else Path(path)


def load_anchor() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"missing anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_frame(anchor, label="anchor")
    return anchor


def load_test_context() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"missing test data: {TEST_PATH}")
    test = pd.read_csv(TEST_PATH)
    required = {"rally_uid", "strikeNumber", "actionId", "pointId", "spinId", "strengthId"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError(f"test_new.csv missing columns: {missing}")

    rows: list[dict[str, int]] = []
    for rally_uid, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        last = group.iloc[-1]
        pref = int(last["strikeNumber"])
        rows.append(
            {
                "rally_uid": int(rally_uid),
                "prefix_len": pref,
                "prefix_bin": prefix_bin(pref),
                "phase": phase_from_prefix(pref),
                "lag0_actionId": int(last["actionId"]),
                "lag0_pointId": int(last["pointId"]),
                "lag0_spinId": int(last["spinId"]),
                "lag0_strengthId": int(last["strengthId"]),
            }
        )
    context = pd.DataFrame(rows)
    if len(context) != EXPECTED_ROWS:
        raise ValueError(f"test context rows differ from expected submissions: {len(context)}")
    return context


def discover_v277_candidates(search: pd.DataFrame) -> list[tuple[str, Path, str]]:
    by_path: dict[Path, tuple[str, Path, str]] = {}

    if "path" in search.columns:
        for _, row in search.iterrows():
            path = normalize_path(row.get("path"))
            if path is None or not path.name.startswith("submission_v277"):
                continue
            candidate = str(row.get("candidate", path.stem))
            verdict = str(row.get("verdict", ""))
            by_path[path] = (candidate, path, verdict)

    for path in sorted(V277_DIR.glob("submission_v277*.csv")):
        by_path.setdefault(path, (path.stem, path, ""))

    return sorted(by_path.values(), key=lambda item: item[0])


def v277_ready() -> tuple[bool, list[str], pd.DataFrame | None, list[tuple[str, Path, str]]]:
    missing: list[str] = []
    if not V277_DIR.exists():
        missing.append(str(V277_DIR))
    if not V277_SEARCH.exists():
        missing.append(str(V277_SEARCH))
    if missing:
        return False, missing, None, []

    search = pd.read_csv(V277_SEARCH)
    candidates = discover_v277_candidates(search)
    missing_candidate_files = [str(path) for _, path, _ in candidates if not path.exists()]
    if missing_candidate_files:
        missing.extend(missing_candidate_files)
    if not candidates:
        missing.append("submission_v277*.csv")
    return len(missing) == 0, missing, search, candidates


def write_waiting_report(missing: list[str]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CHANGED_ROW_COLUMNS).to_csv(OUTDIR / "v278_changed_rows.csv", index=False)
    pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(OUTDIR / "v278_candidate_audit.csv", index=False)
    missing_lines = "\n".join(f"- `{item}`" for item in missing)
    report = (
        "# V278 Changed-Row Audit\n\n"
        "status: waiting_for_v277\n\n"
        "V277 outputs are not complete yet, so no changed-row audit was run.\n\n"
        "Missing inputs:\n"
        f"{missing_lines}\n"
    )
    (OUTDIR / "v278_report.md").write_text(report, encoding="utf-8")


def top_share(series: pd.Series) -> tuple[str, int, float]:
    if series.empty:
        return "", 0, 0.0
    counts = series.astype(str).value_counts()
    label = str(counts.index[0])
    count = int(counts.iloc[0])
    return label, count, float(count / len(series))


def recommendation(verdict: str, point0_additions: int, changed_rows: int, max_concentration: float) -> tuple[str, str]:
    is_concentrated = changed_rows >= 5 and max_concentration > 0.70
    if point0_additions > 5:
        return "REJECT_POINT0", "point0 additions exceed 5"
    if point0_additions > 3:
        return "REJECT_POINT0", "point0 additions exceed clean-review threshold of 3"
    if is_concentrated:
        return "REJECT_CONCENTRATION", "phase/prefix/transition concentration exceeds 70%"
    if changed_rows == 0:
        return "KEEP_CLEAN_REVIEW", "zero changed rows; clean but no-op"
    if verdict == "CANDIDATE_FOR_PUBLIC_PROBE" and point0_additions <= 3:
        return "KEEP_CLEAN_REVIEW", "V277 public-probe verdict with clean point0/concentration audit"
    return "KEEP_CLEAN_REVIEW", f"clean concentration/point0 audit, but V277 verdict is {verdict}"


def audit_candidate(
    candidate: str,
    path: Path,
    verdict: str,
    anchor: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sub = pd.read_csv(path)
    validate_submission_frame(sub, label=candidate)

    merged = anchor[["rally_uid", "pointId"]].rename(columns={"pointId": "anchor_point"}).merge(
        sub[["rally_uid", "pointId"]].rename(columns={"pointId": "candidate_point"}),
        on="rally_uid",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(anchor):
        raise ValueError(f"{candidate} did not align one-to-one with anchor")
    changed = merged[merged["anchor_point"].astype(int) != merged["candidate_point"].astype(int)].copy()
    changed = changed.merge(context, on="rally_uid", how="left", validate="one_to_one")
    if changed[["prefix_len", "phase", "lag0_actionId", "lag0_pointId", "lag0_spinId", "lag0_strengthId"]].isna().any().any():
        raise ValueError(f"{candidate} changed rows missing test context")

    changed["candidate"] = candidate
    changed["anchor_point"] = changed["anchor_point"].astype(int)
    changed["candidate_point"] = changed["candidate_point"].astype(int)
    changed["is_point0_add"] = (changed["anchor_point"].ne(0) & changed["candidate_point"].eq(0)).astype(int)
    changed["is_nonterminal_change"] = (
        changed["anchor_point"].ne(0)
        & changed["candidate_point"].ne(0)
        & changed["anchor_point"].ne(changed["candidate_point"])
    ).astype(int)
    changed["transition"] = changed["anchor_point"].astype(str) + "->" + changed["candidate_point"].astype(str)

    phase_label, phase_count, phase_share = top_share(changed["phase"])
    prefix_label, prefix_count, prefix_share = top_share(changed["prefix_bin"])
    transition_label, transition_count, transition_share = top_share(changed["transition"])
    max_concentration = max(phase_share, prefix_share, transition_share)
    rec, reason = recommendation(verdict, int(changed["is_point0_add"].sum()), len(changed), max_concentration)

    row = {
        "candidate": candidate,
        "path": str(path),
        "v277_verdict": verdict,
        "changed_rows": int(len(changed)),
        "point0_additions": int(changed["is_point0_add"].sum()),
        "nonterminal_changes": int(changed["is_nonterminal_change"].sum()),
        "top_phase": phase_label,
        "top_phase_count": phase_count,
        "top_phase_share": phase_share,
        "top_prefix_bin": prefix_label,
        "top_prefix_count": prefix_count,
        "top_prefix_share": prefix_share,
        "top_transition": transition_label,
        "top_transition_count": transition_count,
        "top_transition_share": transition_share,
        "is_concentrated": bool(len(changed) >= 5 and max_concentration > 0.70),
        "recommendation": rec,
        "recommendation_reason": reason,
    }
    return changed[CHANGED_ROW_COLUMNS], row


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def section_counts(title: str, df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return f"## {title}\n\nNo changed rows.\n"
    counts = df.groupby(columns, dropna=False).size().reset_index(name="changed_rows")
    counts = counts.sort_values("changed_rows", ascending=False)
    return f"## {title}\n\n{markdown_table(counts)}\n"


def write_full_report(changed_rows: pd.DataFrame, audit: pd.DataFrame) -> None:
    lines = [
        "# V278 Changed-Row Audit",
        "",
        "status: complete",
        "",
        "## Candidate Audit",
        "",
        markdown_table(
            audit[
                [
                    "candidate",
                    "v277_verdict",
                    "changed_rows",
                    "point0_additions",
                    "nonterminal_changes",
                    "top_phase_share",
                    "top_prefix_share",
                    "top_transition_share",
                    "recommendation",
                ]
            ]
        ),
        "",
        section_counts("Changed Rows by Phase", changed_rows, ["candidate", "phase"]),
        section_counts("Changed Rows by Prefix Bin", changed_rows, ["candidate", "prefix_bin"]),
        section_counts("Changed Rows by Point Transition", changed_rows.assign(transition=changed_rows["anchor_point"].astype(str) + "->" + changed_rows["candidate_point"].astype(str)), ["candidate", "transition"]),
        section_counts("Point0 Additions", changed_rows[changed_rows["is_point0_add"].eq(1)], ["candidate", "anchor_point", "candidate_point"]),
        section_counts("Nonterminal Changes", changed_rows[changed_rows["is_nonterminal_change"].eq(1)], ["candidate", "anchor_point", "candidate_point"]),
    ]
    (OUTDIR / "v278_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit() -> pd.DataFrame:
    ready, missing, _search, candidates = v277_ready()
    if not ready:
        write_waiting_report(missing)
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor()
    context = load_test_context()

    changed_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for candidate, path, verdict in candidates:
        changed, row = audit_candidate(candidate, path, verdict, anchor, context)
        changed_frames.append(changed)
        audit_rows.append(row)

    changed_rows = pd.concat(changed_frames, ignore_index=True) if changed_frames else pd.DataFrame(columns=CHANGED_ROW_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    changed_rows.to_csv(OUTDIR / "v278_changed_rows.csv", index=False)
    audit.to_csv(OUTDIR / "v278_candidate_audit.csv", index=False)
    write_full_report(changed_rows, audit)
    return audit


def main() -> None:
    audit = run_audit()
    if audit.empty:
        print("V278 audit status: waiting_for_v277")
    else:
        print(audit[["candidate", "changed_rows", "point0_additions", "nonterminal_changes", "recommendation"]].to_string(index=False))


if __name__ == "__main__":
    main()
