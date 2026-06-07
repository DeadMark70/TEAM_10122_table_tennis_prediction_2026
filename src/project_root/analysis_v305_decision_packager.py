"""Package V305 literal point rebuild results into an upload decision report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTDIR = Path("v305_decision_packager")
V261_SEARCH = Path("v305_rebuild_v261_from_literal_v188/v305_v261_literal_search.csv")
RETEST_SEARCH = Path("v305_point_residual_retest_suite/v305_point_retest_search.csv")
CURRENT_BEST = "submission_v300_best_safe_repack__v173action_v261point_server.csv"
CURRENT_BEST_PL = 0.3576975


def _float(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return default


def rank_key(row: dict) -> tuple[int, float, float]:
    is_review = 1 if str(row.get("decision", "")).upper() == "REVIEW" else 0
    delta = _float(row, "literal_delta", "delta_vs_v188_cap5", default=0.0)
    churn = _float(row, "test_churn", "test_churn_vs_v188_cap5", default=1.0)
    return (is_review, delta, -churn)


def normalize_search(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    out = df.copy()
    out["source"] = source
    out["literal_delta"] = out.apply(lambda r: _float(r.to_dict(), "literal_delta", "delta_vs_v188_cap5", default=0.0), axis=1)
    out["test_churn"] = out.apply(lambda r: _float(r.to_dict(), "test_churn", "test_churn_vs_v188_cap5", default=1.0), axis=1)
    out["decision"] = out.get("decision", "DO_NOT_UPLOAD")
    if "submission" not in out:
        out["submission"] = ""
    if "path" not in out:
        out["path"] = ""
    return out


def append_log(report: dict) -> None:
    log_path = Path("experiments_log.md")
    best = report.get("best_candidate", {})
    text = (
        "\n\n## V305 literal V188/V261 point OOF rebuild\n\n"
        "- Motivation: V188/V261 lacked literal row-level OOF/proba, so later point decisions risked proxy-validation bias.\n"
        f"- Current clean public best entering V305: `{CURRENT_BEST}` PL `{CURRENT_BEST_PL:.7f}`.\n"
        f"- Verdict: `{report['verdict']}`.\n"
        f"- Best V305 candidate: `{best.get('candidate', 'none')}` delta `{float(best.get('literal_delta', 0.0)):.6f}` decision `{best.get('decision', 'none')}`.\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`.\n"
        f"- Current clean best after V305: `{report['current_clean_best']}`.\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    frames = [
        normalize_search(V261_SEARCH, "v305_v261_literal"),
        normalize_search(RETEST_SEARCH, "v305_point_retest"),
    ]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    baseline = {
        "candidate": "current_public_best_v300",
        "source": "baseline",
        "submission": CURRENT_BEST,
        "path": "v300_clean_server_blend_recycler/" + CURRENT_BEST,
        "literal_delta": 0.0,
        "test_churn": 0.0,
        "decision": "BASELINE",
        "public_pl": CURRENT_BEST_PL,
    }
    if df.empty:
        ranked = pd.DataFrame([baseline])
    else:
        df["rank_tuple"] = df.apply(lambda r: rank_key(r.to_dict()), axis=1)
        df = df.sort_values("rank_tuple", ascending=False).drop(columns=["rank_tuple"]).reset_index(drop=True)
        ranked = pd.concat([pd.DataFrame([baseline]), df], ignore_index=True)
    ranked.to_csv(OUTDIR / "v305_ranked_candidates.csv", index=False)

    review = ranked[ranked["decision"].astype(str).str.upper().eq("REVIEW")]
    best = review.head(1).iloc[0].to_dict() if not review.empty else {}
    report = {
        "verdict": "HAS_REVIEW_CANDIDATE" if best else "NO_UPLOAD_WORTHY_CANDIDATE",
        "upload_recommendation": "review_top_v305_candidate_before_upload" if best else "keep_current_v300_best",
        "current_clean_best": CURRENT_BEST,
        "current_clean_best_pl": CURRENT_BEST_PL,
        "best_candidate": best,
        "top_review_candidates": review.head(2).to_dict(orient="records"),
        "candidate_count": int(max(0, len(ranked) - 1)),
        "notes": [
            "V305 candidates must clear literal OOF gates before upload consideration.",
            "If no REVIEW candidate exists, current V300 best_safe_repack remains the clean recommendation.",
        ],
    }
    (OUTDIR / "v305_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v305_report.md").write_text(
        "# V305 Decision Packager\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n"
        f"- Current clean best: `{CURRENT_BEST}` PL `{CURRENT_BEST_PL:.7f}`\n"
        f"- Best V305 candidate: `{best.get('candidate', 'none')}`\n"
        f"- Candidate count: `{report['candidate_count']}`\n\n"
        "## Review Candidates\n\n"
        + ("\n".join(f"- `{r.get('candidate')}` delta `{float(r.get('literal_delta', 0.0)):.6f}` churn `{float(r.get('test_churn', 0.0)):.6f}`" for r in report["top_review_candidates"]) if report["top_review_candidates"] else "- None\n")
        + "\n",
        encoding="utf-8",
    )
    append_log(report)
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "recommendation": report["upload_recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
