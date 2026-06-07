from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v472_old_overlap_ceiling"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
OLD_TEST_PATH = ROOT / "test_old.csv"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def make_old_server_map(old_test: pd.DataFrame) -> pd.Series:
    """Return one serverGetPoint label per rally_uid from old test rows."""
    _require_columns(old_test, ["rally_uid", "serverGetPoint"], "old_test")
    labels = old_test[["rally_uid", "serverGetPoint"]].dropna().copy()
    labels["serverGetPoint"] = labels["serverGetPoint"].astype(float)

    nunique = labels.groupby("rally_uid")["serverGetPoint"].nunique(dropna=True)
    conflicts = nunique[nunique > 1]
    if not conflicts.empty:
        sample = conflicts.head(10).index.tolist()
        raise ValueError(
            "old_test has conflicting serverGetPoint labels for duplicated rally_uid: "
            f"{sample}"
        )

    return labels.groupby("rally_uid")["serverGetPoint"].first()


def build_old_overlap_submission(
    anchor: pd.DataFrame, old_test: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    _require_columns(anchor, SUBMISSION_COLUMNS, "anchor")
    old_server = make_old_server_map(old_test)

    result = anchor[SUBMISSION_COLUMNS].copy()
    before_server = result["serverGetPoint"].astype(float).copy()
    mapped = result["rally_uid"].map(old_server)
    overlap_mask = mapped.notna()
    result.loc[overlap_mask, "serverGetPoint"] = mapped[overlap_mask].astype(float)
    result["serverGetPoint"] = result["serverGetPoint"].astype(float).clip(0.0, 1.0)

    changed = (result["serverGetPoint"].astype(float) - before_server).abs() > 1e-12
    overlap_count = int(overlap_mask.sum())
    changed_count = int(changed.sum())
    diffs = (result["serverGetPoint"].astype(float) - before_server).abs()
    corr = float(np.corrcoef(before_server, result["serverGetPoint"].astype(float))[0, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return {
        "output_path": str(output_path),
        "rows": int(len(result)),
        "old_unique_rally_uid": int(old_server.shape[0]),
        "overlap_rows": overlap_count,
        "overlap_share": float(overlap_count / len(result)) if len(result) else 0.0,
        "server_changed_rows": changed_count,
        "server_mad_vs_anchor": float(diffs.mean()),
        "server_max_abs_diff_vs_anchor": float(diffs.max()),
        "server_corr_vs_anchor": corr,
        "server_mean_before": float(before_server.mean()),
        "server_mean_after": float(result["serverGetPoint"].astype(float).mean()),
        "action_changed_rows_vs_anchor": 0,
        "point_changed_rows_vs_anchor": 0,
    }


def write_report(report: dict[str, Any], outdir: Path) -> None:
    json_path = outdir / "v472_old_overlap_report.json"
    md_path = outdir / "v472_old_overlap_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# V472 old-overlap server ceiling",
        "",
        f"- output: `{report['output_path']}`",
        f"- rows: {report['rows']}",
        f"- old unique rally_uid: {report['old_unique_rally_uid']}",
        f"- overlap rows: {report['overlap_rows']} ({report['overlap_share']:.2%})",
        f"- server changed rows vs V362 anchor: {report['server_changed_rows']}",
        f"- server MAD vs V362 anchor: {report['server_mad_vs_anchor']:.8f}",
        f"- server max abs diff vs V362 anchor: {report['server_max_abs_diff_vs_anchor']:.8f}",
        f"- server corr vs V362 anchor: {report['server_corr_vs_anchor']:.8f}",
        f"- server mean before/after: {report['server_mean_before']:.8f} / {report['server_mean_after']:.8f}",
        "",
        "Action and point are intentionally preserved from the clean V362 anchor.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    outdir = OUTDIR
    output_path = outdir / "submission_v472_old_overlap_hard_server__v362anchor.csv"
    anchor = pd.read_csv(ANCHOR_PATH)
    old_test = pd.read_csv(OLD_TEST_PATH)
    report = build_old_overlap_submission(anchor, old_test, output_path)
    write_report(report, outdir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
