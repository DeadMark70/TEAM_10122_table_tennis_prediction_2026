"""R201 no-old server microblend.

Small clean server-only blends over the current no-old anchor.  No old-server
labels and no TTMATCH are read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("r201_no_old_server_microblend")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_r201_no_old_server_microblend.py")

ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SERVER_SOURCES = {
    "r121_mean": UPLOAD_DIR / "submission_r121_traj_mean_w0p35.csv",
    "r121_last": UPLOAD_DIR / "submission_r121_traj_last_w0p35.csv",
    "r121_min035": UPLOAD_DIR / "submission_r121_traj_min_w0p35.csv",
    "r121_min02": UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv",
}


def blend_server(anchor: np.ndarray, sources: list[np.ndarray], weights: list[float]) -> np.ndarray:
    out = np.asarray(anchor, dtype=float).copy()
    total = 1.0
    for src, w in zip(sources, weights):
        out += float(w) * (np.asarray(src, dtype=float) - np.asarray(anchor, dtype=float))
        total += 0.0
    return np.clip(out, 0.0, 1.0)


def server_mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def load_sub(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, anchor: pd.DataFrame, server: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = anchor[["rally_uid", "actionId", "pointId"]].copy()
    out["serverGetPoint"] = np.asarray(server, dtype=float)
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    anchor = pd.read_csv(ANCHOR)
    rally_uids = anchor["rally_uid"].astype(int).to_numpy()
    anchor_server = anchor["serverGetPoint"].astype(float).to_numpy()
    src = {k: load_sub(v, rally_uids)["serverGetPoint"].astype(float).to_numpy() for k, v in SERVER_SOURCES.items() if v.exists()}
    candidates = {
        "r201_server_mean_last_w0p01": blend_server(anchor_server, [src["r121_mean"], src["r121_last"]], [0.01, 0.01]),
        "r201_server_mean_last_w0p02": blend_server(anchor_server, [src["r121_mean"], src["r121_last"]], [0.02, 0.02]),
        "r201_server_min035_w0p02": blend_server(anchor_server, [src["r121_min035"]], [0.02]),
        "r201_server_all_r121_w0p01": blend_server(anchor_server, [src["r121_mean"], src["r121_last"], src["r121_min035"]], [0.01, 0.01, 0.01]),
    }
    summary = []
    for name, server in candidates.items():
        sub_name = f"submission_{name}__v173_v188cap5.csv"
        info = write_submission(sub_name, anchor, server)
        rec = {
            "candidate": name,
            "submission": sub_name,
            "server_mad_vs_anchor": server_mad(server, anchor_server),
            "server_corr_vs_anchor": float(np.corrcoef(server, anchor_server)[0, 1]),
            "server_min": float(server.min()),
            "server_max": float(server.max()),
        }
        rec.update(info)
        summary.append(rec)
    pd.DataFrame(summary).sort_values("server_mad_vs_anchor").to_csv(OUTDIR / "r201_candidate_summary.csv", index=False)
    report = {"verdict": "GENERATED", "generated": summary, "notes": ["No old-server labels.", "TTMATCH is not read."]}
    (OUTDIR / "r201_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r201_report.md").write_text("# R201 No-Old Server Microblend\n\nGenerated clean server microblend candidates.\n", encoding="utf-8")
    shutil.copy2("analysis_r201_no_old_server_microblend.py", SRC_DEST)
    print(json.dumps({"generated": len(summary), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
