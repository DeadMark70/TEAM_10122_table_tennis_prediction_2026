"""R39: audit historical V64/V58/V56 high-LB teacher submissions.

This script does not train a model. It aligns historical submissions from
`C:/aicup/tenis_new` with the current `test_new.csv`, compares them to the
current best local candidate, and writes a few low-effort R40 diagnostic blends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
OLD_ROOT = Path(r"C:\aicup\tenis_new")
OUT_DIR = ROOT / "r39_v64_teacher_audit"
UPLOAD_DIR = ROOT / "upload_candidates_20260519"


SUBMISSIONS = {
    "current_r34": ROOT / "upload_candidates_20260519" / "submission_r34_r33action_v3point_r28server.csv",
    "v64_v58_v56_3way": OLD_ROOT / "submission_v64_v58_v56_3way_blend.csv",
    "v61_v58_v56_3way": OLD_ROOT / "submission_v61_v58_v56_3way_blend.csv",
    "v68_5way": OLD_ROOT / "submission_v68_v64_v61_v58_v56_5way_blend.csv",
    "v64_v79_transductive": OLD_ROOT / "submission_v64_v79_transductive_blend_v1.csv",
    "v58_tactical": OLD_ROOT / "submission_v58_tactical_final.csv",
    "v56_spatial_aux": OLD_ROOT / "submission_v56_spatial_aux_final.csv",
}


def prefix_bin(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    if length == 3:
        return "3"
    if 4 <= length <= 6:
        return "4-6"
    return "7+"


def read_test_meta() -> pd.DataFrame:
    test = pd.read_csv(ROOT / "test_new.csv")
    rows = []
    for rid, g in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        last = g.iloc[-1]
        rows.append(
            {
                "rally_uid": int(rid),
                "prefix_len": int(last["strikeNumber"]),
                "prefix_bin": prefix_bin(int(last["strikeNumber"])),
                "sex": int(last["sex"]),
                "match": int(last["match"]),
                "numberGame": int(last["numberGame"]),
                "last_actionId": int(last["actionId"]),
                "last_pointId": int(last["pointId"]),
                "last_spinId": int(last["spinId"]),
            }
        )
    return pd.DataFrame(rows)


def read_submission(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    out = df[cols].copy()
    out = out.rename(
        columns={
            "actionId": f"{name}_actionId",
            "pointId": f"{name}_pointId",
            "serverGetPoint": f"{name}_serverGetPoint",
        }
    )
    proba_cols = [c for c in df.columns if c.startswith("seq_action_prob_") or c.startswith("seq_point_prob_")]
    if proba_cols:
        out = pd.concat([out, df[proba_cols]], axis=1)
    return out


def distribution(df: pd.DataFrame, col: str, classes: list[int]) -> pd.DataFrame:
    vc = df[col].value_counts().to_dict()
    total = len(df)
    return pd.DataFrame(
        [{"class": c, "count": int(vc.get(c, 0)), "rate": float(vc.get(c, 0) / total)} for c in classes]
    )


def compare_model(base: pd.DataFrame, model: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    action_col = f"{model}_actionId"
    point_col = f"{model}_pointId"
    server_col = f"{model}_serverGetPoint"
    base_action = "current_r34_actionId"
    base_point = "current_r34_pointId"
    base_server = "current_r34_serverGetPoint"

    for label, part in [("all", base)] + [(b, g) for b, g in base.groupby("prefix_bin", sort=False)]:
        a_diff = part[action_col].ne(part[base_action])
        p_diff = part[point_col].ne(part[base_point])
        s1 = part[server_col].to_numpy(dtype=float)
        s0 = part[base_server].to_numpy(dtype=float)
        corr = np.corrcoef(s0, s1)[0, 1] if len(part) > 1 and np.std(s0) > 0 and np.std(s1) > 0 else np.nan
        rows.append(
            {
                "model": model,
                "prefix_bin": label,
                "count": int(len(part)),
                "action_diff_rate_vs_current": float(a_diff.mean()),
                "point_diff_rate_vs_current": float(p_diff.mean()),
                "both_action_point_diff_rate": float((a_diff & p_diff).mean()),
                "server_mae_vs_current": float(np.mean(np.abs(s1 - s0))),
                "server_corr_vs_current": float(corr) if not np.isnan(corr) else np.nan,
                "server_mean": float(np.mean(s1)),
                "server_std": float(np.std(s1)),
            }
        )
    summary = rows[0].copy()
    return pd.DataFrame(rows), summary


def component_diff(df: pd.DataFrame, left: str, right: str) -> dict:
    s0 = df[f"{left}_serverGetPoint"].to_numpy(dtype=float)
    s1 = df[f"{right}_serverGetPoint"].to_numpy(dtype=float)
    corr = np.corrcoef(s0, s1)[0, 1] if np.std(s0) > 0 and np.std(s1) > 0 else np.nan
    return {
        "left": left,
        "right": right,
        "action_diff_rate": float(df[f"{left}_actionId"].ne(df[f"{right}_actionId"]).mean()),
        "point_diff_rate": float(df[f"{left}_pointId"].ne(df[f"{right}_pointId"]).mean()),
        "server_mae": float(np.mean(np.abs(s0 - s1))),
        "server_corr": float(corr) if not np.isnan(corr) else np.nan,
    }


def write_candidate(df: pd.DataFrame, name: str, action_source: str, point_source: str, server_source: str) -> Path:
    sub = pd.DataFrame(
        {
            "rally_uid": df["rally_uid"].astype(int),
            "actionId": df[f"{action_source}_actionId"].astype(int),
            "pointId": df[f"{point_source}_pointId"].astype(int),
            "serverGetPoint": np.round(df[f"{server_source}_serverGetPoint"].clip(1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUT_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    meta = read_test_meta()
    merged = meta.copy()
    available = {}
    for name, path in SUBMISSIONS.items():
        if not path.exists():
            continue
        available[name] = str(path)
        sub = read_submission(path, name)
        merged = merged.merge(sub, on="rally_uid", how="left")
    missing = merged.filter(regex="_actionId$").isna().sum()
    if missing.any():
        raise ValueError(f"Some submissions did not align:\n{missing[missing > 0]}")
    merged.to_csv(OUT_DIR / "r39_aligned_predictions.csv", index=False)

    rows = []
    summaries = []
    for name in available:
        if name == "current_r34":
            continue
        table, summary = compare_model(merged, name)
        rows.append(table)
        summaries.append(summary)
        distribution(merged, f"{name}_actionId", list(range(19))).to_csv(OUT_DIR / f"r39_action_distribution_{name}.csv", index=False)
        distribution(merged, f"{name}_pointId", list(range(10))).to_csv(OUT_DIR / f"r39_point_distribution_{name}.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "r39_diff_by_prefix.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "r39_model_summary.csv", index=False)

    component_pairs = [
        ("v64_v58_v56_3way", "v58_tactical"),
        ("v64_v58_v56_3way", "v56_spatial_aux"),
        ("v64_v58_v56_3way", "v61_v58_v56_3way"),
        ("v68_5way", "v64_v58_v56_3way"),
        ("v64_v79_transductive", "v64_v58_v56_3way"),
    ]
    pd.DataFrame([component_diff(merged, a, b) for a, b in component_pairs if a in available and b in available]).to_csv(
        OUT_DIR / "r39_component_diff.csv", index=False
    )

    candidate_paths = []
    if "v64_v58_v56_3way" in available:
        candidate_paths.append(write_candidate(merged, "submission_r40_v64action_current_point_server.csv", "v64_v58_v56_3way", "current_r34", "current_r34"))
        candidate_paths.append(write_candidate(merged, "submission_r40_v64action_v64point_current_server.csv", "v64_v58_v56_3way", "v64_v58_v56_3way", "current_r34"))
        candidate_paths.append(write_candidate(merged, "submission_r40_v64_full_copy.csv", "v64_v58_v56_3way", "v64_v58_v56_3way", "v64_v58_v56_3way"))
    if "v68_5way" in available:
        candidate_paths.append(write_candidate(merged, "submission_r40_v68action_current_point_server.csv", "v68_5way", "current_r34", "current_r34"))
    for path in candidate_paths:
        (UPLOAD_DIR / path.name).write_bytes(path.read_bytes())

    report = {
        "available_sources": available,
        "outputs": {
            "aligned": str(OUT_DIR / "r39_aligned_predictions.csv"),
            "diff_by_prefix": str(OUT_DIR / "r39_diff_by_prefix.csv"),
            "model_summary": str(OUT_DIR / "r39_model_summary.csv"),
            "component_diff": str(OUT_DIR / "r39_component_diff.csv"),
            "candidates": [str(p) for p in candidate_paths],
        },
        "note": "R40 full-copy candidates are diagnostic historical-teacher candidates; action-only blends are lower risk.",
    }
    (OUT_DIR / "r39_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# R39 V64 Teacher Audit",
        "",
        "Compared historical V64/V58/V56 submissions against current R34 best candidate on current test_new rows.",
        "",
        "## Key Source Files",
        *[f"- `{name}`: `{path}`" for name, path in available.items()],
        "",
        "## Generated Candidates",
        *[f"- `{p.name}`" for p in candidate_paths],
        "",
        "See CSV reports in `r39_v64_teacher_audit/` for prefix and class distribution details.",
    ]
    (OUT_DIR / "r39_v64_vs_current_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
