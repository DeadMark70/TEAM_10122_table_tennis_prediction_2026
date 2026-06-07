"""V277/V272B point refinement from existing V272 submissions.

This script keeps the V261 clean anchor fixed for action/server and filters
existing V272 point edits into lower-risk point-only variants. It does not
retrain models or read raw V272 diagnostics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
POINT0_RATE_MIN = 0.24
POINT0_RATE_MAX = 0.31
PUBLIC_PROBE_MAX_CHURN = 0.010
PUBLIC_PROBE_MAX_POINT0_ADDED = 3
REVIEW_MAX_CHURN = 0.015
REVIEW_MAX_POINT0_ADDED = 5

OUTDIR = Path("v277_v272b_point_refinement")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
V272_DIR = Path("v272_action_conditioned_point_residual")
V272_MODEL_CAP005 = V272_DIR / "submission_v272_point_actioncond_cap0p005__v173action_r121server.csv"
V272_MODEL_CAP010 = V272_DIR / "submission_v272_point_actioncond_cap0p010__v173action_r121server.csv"
V272_MODEL_CAP015 = V272_DIR / "submission_v272_point_actioncond_cap0p015__v173action_r121server.csv"
V272_TABLE_CAP010 = V272_DIR / "submission_v272_point_actioncond_table_cap0p010__v173action_r121server.csv"


def validate_submission_frame(df: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"bad columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"bad rows: {len(df)}")


def changed_mask(anchor_point: np.ndarray, candidate_point: np.ndarray) -> np.ndarray:
    return np.asarray(anchor_point, dtype=int) != np.asarray(candidate_point, dtype=int)


def no_point0_add_mask(anchor_point: np.ndarray, candidate_point: np.ndarray) -> np.ndarray:
    anchor = np.asarray(anchor_point, dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    return ~((anchor != 0) & (cand == 0))


def nonterminal_change_mask(anchor_point: np.ndarray, candidate_point: np.ndarray) -> np.ndarray:
    anchor = np.asarray(anchor_point, dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    return (anchor != 0) & (cand != 0) & (anchor != cand)


def agreement_mask(anchor_point: np.ndarray, model_point: np.ndarray, table_point: np.ndarray) -> np.ndarray:
    anchor = np.asarray(anchor_point, dtype=int)
    model = np.asarray(model_point, dtype=int)
    table = np.asarray(table_point, dtype=int)
    return (model == table) & (model != anchor)


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing submission: {path}")
    df = pd.read_csv(path)
    validate_submission_frame(df)
    return df


def require_aligned(anchor: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if not anchor["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError(f"{label} rally_uid does not align with anchor")


def build_submission(anchor: pd.DataFrame, candidate_point: np.ndarray, keep_mask: np.ndarray) -> pd.DataFrame:
    anchor_point = anchor["pointId"].astype(int).to_numpy()
    cand = np.asarray(candidate_point, dtype=int)
    keep = np.asarray(keep_mask, dtype=bool)
    if len(cand) != len(anchor) or len(keep) != len(anchor):
        raise ValueError("candidate point and mask must match anchor length")
    out_point = anchor_point.copy()
    out_point[keep] = cand[keep]

    out = anchor.copy()
    out["pointId"] = out_point
    out = out[EXPECTED_COLUMNS]
    validate_submission_frame(out)
    if not out["actionId"].equals(anchor["actionId"]):
        raise ValueError("actionId changed; V277 must keep anchor action fixed")
    if not np.allclose(out["serverGetPoint"].astype(float), anchor["serverGetPoint"].astype(float)):
        raise ValueError("serverGetPoint changed; V277 must keep anchor server fixed")
    return out


def point_distribution(point: np.ndarray) -> str:
    labels = np.asarray(point, dtype=int)
    counts = np.bincount(labels, minlength=max(10, int(labels.max()) + 1))
    return json.dumps({str(i): int(v) for i, v in enumerate(counts) if v > 0}, separators=(",", ":"))


def verdict_for(point_churn: float, point0_added_rows: int, point0_rate_test: float) -> str:
    in_rate = POINT0_RATE_MIN <= point0_rate_test <= POINT0_RATE_MAX
    if point_churn <= PUBLIC_PROBE_MAX_CHURN and point0_added_rows <= PUBLIC_PROBE_MAX_POINT0_ADDED and in_rate:
        return "CANDIDATE_FOR_PUBLIC_PROBE"
    if point_churn <= REVIEW_MAX_CHURN and point0_added_rows <= REVIEW_MAX_POINT0_ADDED and in_rate:
        return "EXPLORATORY_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def metrics_record(
    *,
    candidate: str,
    path: Path,
    anchor: pd.DataFrame,
    submission: pd.DataFrame,
    source_candidate: str,
    filter_type: str,
) -> dict[str, object]:
    anchor_point = anchor["pointId"].astype(int).to_numpy()
    out_point = submission["pointId"].astype(int).to_numpy()
    changed = changed_mask(anchor_point, out_point)
    point_churn = float(changed.mean())
    point0_added_rows = int(np.sum((anchor_point != 0) & (out_point == 0)))
    point0_removed_rows = int(np.sum((anchor_point == 0) & (out_point != 0)))
    point0_rate_test = float(np.mean(out_point == 0))
    return {
        "candidate": candidate,
        "path": str(path),
        "changed_rows": int(changed.sum()),
        "point_churn": point_churn,
        "point0_added_rows": point0_added_rows,
        "point0_removed_rows": point0_removed_rows,
        "point0_rate_test": point0_rate_test,
        "point_distribution": point_distribution(out_point),
        "source_candidate": source_candidate,
        "filter_type": filter_type,
        "verdict": verdict_for(point_churn, point0_added_rows, point0_rate_test),
    }


def write_submission(path: Path, submission: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission_frame(submission)
    submission.to_csv(path, index=False, float_format="%.8f")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, UPLOAD_DIR / path.name)


def add_variant(
    *,
    records: list[dict[str, object]],
    anchor: pd.DataFrame,
    source: pd.DataFrame,
    keep_mask: np.ndarray,
    filename: str,
    candidate: str,
    source_candidate: str,
    filter_type: str,
) -> None:
    path = OUTDIR / filename
    submission = build_submission(anchor, source["pointId"].astype(int).to_numpy(), keep_mask)
    write_submission(path, submission)
    records.append(
        metrics_record(
            candidate=candidate,
            path=path,
            anchor=anchor,
            submission=submission,
            source_candidate=source_candidate,
            filter_type=filter_type,
        )
    )


def write_report(search: pd.DataFrame) -> None:
    lines = [
        "# V277/V272B Point Refinement",
        "",
        "Fixed clean anchor:",
        "",
        "```text",
        str(ANCHOR_PATH),
        "action = fixed V173 action anchor",
        "server = fixed R121 server anchor",
        "changed field = pointId only",
        "```",
        "",
        "## Verdict Gates",
        "",
        f"- `CANDIDATE_FOR_PUBLIC_PROBE`: point_churn <= {PUBLIC_PROBE_MAX_CHURN:.3f}, point0_added_rows <= {PUBLIC_PROBE_MAX_POINT0_ADDED}, point0_rate_test in [{POINT0_RATE_MIN:.2f}, {POINT0_RATE_MAX:.2f}]",
        f"- `EXPLORATORY_REVIEW`: point_churn <= {REVIEW_MAX_CHURN:.3f}, point0_added_rows <= {REVIEW_MAX_POINT0_ADDED}, point0_rate_test in [{POINT0_RATE_MIN:.2f}, {POINT0_RATE_MAX:.2f}]",
        "- `LOCAL_NEGATIVE_DO_NOT_SUBMIT`: otherwise",
        "",
        "## Candidates",
        "",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"- `{row['candidate']}`: changed_rows={int(row['changed_rows'])}, "
            f"churn={float(row['point_churn']):.6f}, "
            f"point0_added={int(row['point0_added_rows'])}, "
            f"point0_removed={int(row['point0_removed_rows'])}, "
            f"point0_rate={float(row['point0_rate_test']):.6f}, "
            f"verdict=`{row['verdict']}`"
        )
    lines.extend(
        [
            "",
            "## Policy Checks",
            "",
            "- Inputs are limited to the V261 anchor and existing V272 submission/search outputs.",
            "- No TTMATCH, old-server files, or raw V272 diagnostic rows are read.",
            "- Every emitted submission is rebuilt from the anchor for actionId/serverGetPoint.",
            f"- Submissions copied to `{UPLOAD_DIR}`.",
        ]
    )
    (OUTDIR / "v277_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_submission(ANCHOR_PATH)
    cap005 = load_submission(V272_MODEL_CAP005)
    cap010 = load_submission(V272_MODEL_CAP010)
    cap015 = load_submission(V272_MODEL_CAP015)
    table010 = load_submission(V272_TABLE_CAP010)

    for label, frame in [
        ("cap0p005", cap005),
        ("cap0p010", cap010),
        ("cap0p015", cap015),
        ("table_cap0p010", table010),
    ]:
        require_aligned(anchor, frame, label)

    anchor_point = anchor["pointId"].astype(int).to_numpy()
    cap005_point = cap005["pointId"].astype(int).to_numpy()
    cap010_point = cap010["pointId"].astype(int).to_numpy()
    cap015_point = cap015["pointId"].astype(int).to_numpy()
    table010_point = table010["pointId"].astype(int).to_numpy()

    records: list[dict[str, object]] = []
    add_variant(
        records=records,
        anchor=anchor,
        source=cap005,
        keep_mask=changed_mask(anchor_point, cap005_point),
        filename="submission_v277_v272_cap0p005_direct__v173action_r121server.csv",
        candidate="v277_v272_cap0p005_direct",
        source_candidate=V272_MODEL_CAP005.name,
        filter_type="direct",
    )
    add_variant(
        records=records,
        anchor=anchor,
        source=cap010,
        keep_mask=changed_mask(anchor_point, cap010_point) & no_point0_add_mask(anchor_point, cap010_point),
        filename="submission_v277_v272_cap0p010_no_point0_add__v173action_r121server.csv",
        candidate="v277_v272_cap0p010_no_point0_add",
        source_candidate=V272_MODEL_CAP010.name,
        filter_type="no_point0_add",
    )
    add_variant(
        records=records,
        anchor=anchor,
        source=cap015,
        keep_mask=changed_mask(anchor_point, cap015_point) & no_point0_add_mask(anchor_point, cap015_point),
        filename="submission_v277_v272_cap0p015_no_point0_add__v173action_r121server.csv",
        candidate="v277_v272_cap0p015_no_point0_add",
        source_candidate=V272_MODEL_CAP015.name,
        filter_type="no_point0_add",
    )
    add_variant(
        records=records,
        anchor=anchor,
        source=cap010,
        keep_mask=nonterminal_change_mask(anchor_point, cap010_point),
        filename="submission_v277_v272_cap0p010_nonterminal_only__v173action_r121server.csv",
        candidate="v277_v272_cap0p010_nonterminal_only",
        source_candidate=V272_MODEL_CAP010.name,
        filter_type="nonterminal_only",
    )
    add_variant(
        records=records,
        anchor=anchor,
        source=cap015,
        keep_mask=nonterminal_change_mask(anchor_point, cap015_point),
        filename="submission_v277_v272_cap0p015_nonterminal_only__v173action_r121server.csv",
        candidate="v277_v272_cap0p015_nonterminal_only",
        source_candidate=V272_MODEL_CAP015.name,
        filter_type="nonterminal_only",
    )
    agree = agreement_mask(anchor_point, cap010_point, table010_point)
    add_variant(
        records=records,
        anchor=anchor,
        source=cap010,
        keep_mask=agree,
        filename="submission_v277_v272_cap0p010_model_table_agreement__v173action_r121server.csv",
        candidate="v277_v272_cap0p010_model_table_agreement",
        source_candidate=f"{V272_MODEL_CAP010.name}+{V272_TABLE_CAP010.name}",
        filter_type="model_table_agreement",
    )
    add_variant(
        records=records,
        anchor=anchor,
        source=cap010,
        keep_mask=agree & nonterminal_change_mask(anchor_point, cap010_point),
        filename="submission_v277_v272_cap0p010_agreement_nonterminal__v173action_r121server.csv",
        candidate="v277_v272_cap0p010_agreement_nonterminal",
        source_candidate=f"{V272_MODEL_CAP010.name}+{V272_TABLE_CAP010.name}",
        filter_type="agreement_nonterminal_only",
    )

    search = pd.DataFrame(records)
    ordered_cols = [
        "candidate",
        "path",
        "changed_rows",
        "point_churn",
        "point0_added_rows",
        "point0_removed_rows",
        "point0_rate_test",
        "point_distribution",
        "source_candidate",
        "filter_type",
        "verdict",
    ]
    search = search[ordered_cols]
    search.to_csv(OUTDIR / "v277_point_search.csv", index=False)
    write_report(search)
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "generated_submissions": int(len(search)),
                "candidate_verdicts": search.set_index("candidate")["verdict"].to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
