"""Create a non-destructive workspace inventory.

This script classifies root files/directories so the project can be cleaned up
without breaking active relative paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def category(path: Path) -> str:
    name = path.name
    if path.is_dir():
        if name.startswith(("r", "v")) and any(ch.isdigit() for ch in name):
            return "experiment_dir"
        if name == "upload_candidates_20260519":
            return "upload_candidates"
        if name == "external_data":
            return "external_data"
        if name in {"docs", "tools"}:
            return "project_docs_tools"
        if name == "__pycache__":
            return "cache"
        return "directory"

    ext = path.suffix.lower()
    if name in {"train.csv", "test_new.csv", "test_old.csv", "sample_submission.csv"}:
        return "core_data"
    if name.startswith("submission_") and ext == ".csv":
        return "root_submission"
    if name.startswith("oof_proba_") and ext == ".pkl":
        return "oof_artifact"
    if name.startswith("analysis_") and ext == ".py":
        return "analysis_script"
    if name.startswith("baseline_") and ext == ".py":
        return "baseline_script"
    if name.startswith("generate_") and ext == ".py":
        return "submission_generator"
    if name.startswith("train_") and ext == ".py":
        return "training_script"
    if name.startswith(("cv_report_", "class_report_", "feature_report_", "prefix_len_report_")):
        return "report"
    if name.endswith(("_selected.json", "_recommendation.md", "_summary.csv")):
        return "experiment_summary"
    if ext == ".md":
        return "markdown_note"
    if ext == ".json":
        return "json_artifact"
    if ext == ".csv":
        return "csv_artifact"
    if ext == ".pkl":
        return "pickle_artifact"
    return "other"


def recommendation(cat: str) -> str:
    keep = {
        "core_data",
        "analysis_script",
        "baseline_script",
        "submission_generator",
        "training_script",
        "upload_candidates",
        "experiment_dir",
        "external_data",
        "project_docs_tools",
    }
    if cat in keep:
        return "keep_in_place_for_now"
    if cat == "root_submission":
        return "copy_to_submissions_archive_later"
    if cat == "oof_artifact":
        return "copy_to_artifacts_oof_later"
    if cat in {"report", "experiment_summary", "markdown_note", "json_artifact", "csv_artifact"}:
        return "copy_to_artifacts_reports_later"
    if cat == "cache":
        return "can_ignore_or_delete_later"
    return "review"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.astype(str)
    cols = list(text_df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in text_df.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    rows = []
    for path in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        stat = path.stat()
        cat = category(path)
        rows.append(
            {
                "name": path.name,
                "relative_path": str(path.relative_to(ROOT)),
                "is_dir": path.is_dir(),
                "extension": "" if path.is_dir() else path.suffix.lower(),
                "size_bytes": stat.st_size if path.is_file() else 0,
                "modified": pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "category": cat,
                "recommendation": recommendation(cat),
            }
        )
    df = pd.DataFrame(rows)
    out_csv = DOCS / "workspace_inventory_20260519.csv"
    out_md = DOCS / "workspace_inventory_20260519.md"
    df.to_csv(out_csv, index=False)

    counts = df.groupby(["category", "recommendation"]).size().reset_index(name="count")
    lines = ["# Workspace Inventory - 2026-05-19", "", "## Summary", ""]
    lines.append(markdown_table(counts))
    lines += ["", "## Largest Root Files", ""]
    lines.append(markdown_table(df[~df["is_dir"]].sort_values("size_bytes", ascending=False).head(40)))
    lines += ["", "## Root Items", ""]
    lines.append(markdown_table(df))
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
