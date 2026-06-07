"""V199 point calibration sweep over existing residual submissions.

This is a lightweight calibration/composition pass.  It does not retrain point
models and never exports raw neural argmax.  It selectively applies changed
rows from V193/V196/V188 residual candidates on top of the current V188 cap5
anchor.

Action remains V173 and server remains R121.  TTMATCH is not read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v165_combined_external_pretrain_proxy import prepare_prefix_features
from analysis_v194_train_test_split_distribution_audit import add_audit_columns


OUTDIR = Path("v199_point_calibration_sweep")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v199_point_calibration_sweep.py")

ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SOURCES = {
    "v193_cap5": UPLOAD_DIR / "submission_v193_p0match0p29_all_a0p075_cap0p05__v173action_r121server.csv",
    "v196_cap5": UPLOAD_DIR / "submission_v196a_r111_importance_p0t029_rw1_conf085_cw025_tc005_a0p075_cap0p05__v173action_r121server.csv",
    "v193_clean2": UPLOAD_DIR / "submission_v193_p0match0p29_not_receive_a0p075_cap0p02__v173action_r121server.csv",
}


def apply_point_changes(anchor: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(anchor, dtype=int).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = np.asarray(source, dtype=int)[m]
    return out


def point_gate(rows: pd.DataFrame, mode: str) -> np.ndarray:
    prefix = pd.to_numeric(rows["prefix_len"], errors="coerce").fillna(0)
    phase = rows["audit_phase"].astype(str)
    depth = rows["audit_lag0_depth"].astype(str)
    if mode == "all":
        return np.ones(len(rows), dtype=bool)
    if mode == "not_receive":
        return ~phase.eq("receive").to_numpy()
    if mode == "domain_shift":
        return phase.eq("rally").to_numpy() | depth.eq("long").to_numpy() | prefix.ge(3).to_numpy()
    if mode == "long_rally":
        return phase.eq("rally").to_numpy() & depth.eq("long").to_numpy()
    if mode == "nonzero_to_zero_only":
        return np.ones(len(rows), dtype=bool)
    raise ValueError(mode)


def load_sub(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, anchor: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = anchor[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def dist(values: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=10)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    _, _, _, test_prefix, _ = prepare_prefix_features()
    rows = add_audit_columns(test_prefix.reset_index(drop=True))
    rally_uids = rows["rally_uid"].astype(int).to_numpy()
    anchor = load_sub(ANCHOR, rally_uids)
    anchor_point = anchor["pointId"].astype(int).to_numpy()
    source_points = {k: load_sub(v, rally_uids)["pointId"].astype(int).to_numpy() for k, v in SOURCES.items() if v.exists()}

    candidates: dict[str, np.ndarray] = {}
    for source_name, source in source_points.items():
        changed = source != anchor_point
        for gate in ["all", "not_receive", "domain_shift", "long_rally"]:
            mask = changed & point_gate(rows, gate)
            if mask.sum() == 0:
                continue
            candidates[f"v199_{source_name}_{gate}"] = apply_point_changes(anchor_point, source, mask)
        nz0 = changed & (anchor_point != 0) & (source == 0)
        candidates[f"v199_{source_name}_nonzero_to_zero_only"] = apply_point_changes(anchor_point, source, nz0)

    if "v193_cap5" in source_points and "v196_cap5" in source_points:
        agree = (source_points["v193_cap5"] == source_points["v196_cap5"]) & (source_points["v193_cap5"] != anchor_point)
        candidates["v199_v193_v196_agree"] = apply_point_changes(anchor_point, source_points["v193_cap5"], agree)

    summary = []
    generated = []
    for name, point in candidates.items():
        changed = point != anchor_point
        if changed.sum() == 0:
            continue
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, anchor, point)
        rec = {
            "candidate": name,
            "submission": sub_name,
            "point_churn_vs_v188_cap5": float(changed.mean()),
            "changed_rows": int(changed.sum()),
            "point0_rate": float(np.mean(point == 0)),
            "point_distribution": json.dumps(dist(point), sort_keys=True),
        }
        rec.update(info)
        generated.append(info)
        summary.append(rec)
    pd.DataFrame(summary).sort_values(["point_churn_vs_v188_cap5", "candidate"]).to_csv(OUTDIR / "v199_candidate_summary.csv", index=False)
    report = {
        "verdict": "GENERATED",
        "generated": summary,
        "notes": [
            "V199 uses existing residual submissions only; no raw neural argmax.",
            "Action=V173 and server=R121 are preserved.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v199_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v199_report.md").write_text(
        "# V199 Point Calibration Sweep\n\n"
        f"- Generated candidates: `{len(summary)}`\n"
        "- Fixed action/server: V173 + R121.\n"
        "- Anchor point: V188 r186_w005 cap5.\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v199_point_calibration_sweep.py", SRC_DEST)
    print(json.dumps({"generated": len(summary), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
