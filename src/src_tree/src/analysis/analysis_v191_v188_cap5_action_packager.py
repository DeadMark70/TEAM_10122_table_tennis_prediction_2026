"""V191 action probes on the V188 cap5 point anchor.

After V188 r186_w005 cap5 became the no-old point anchor, action probes should
be repackaged with:

  point  = V188 r186_w005 alpha=0.05 cap=0.05
  server = R121 no-old

This script creates clean no-old V166 and R184-lite action probes without
falling back to the older R119 point anchor.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v191_v188_cap5_action_packager")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v191_v188_cap5_action_packager.py")

POINT_ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SERVER_ANCHOR = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
V173_BASE = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"

ACTION_SOURCES = {
    "v166_best_action": UPLOAD_DIR / "submission_r166__ar166_best_action__pr119_public_point__sr121_min_w0p2.csv",
    "r184_attack_to_control": UPLOAD_DIR / "submission_r184_attack_to_control__pr119_sr121.csv",
    "r184_state_pair_supported": UPLOAD_DIR / "submission_r184_state_pair_supported__pr119_sr121.csv",
    "r184_receive_affordance_control": UPLOAD_DIR / "submission_r184_receive_affordance_control__pr119_sr121.csv",
}


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"Submission did not align: {path}")
    return out


def write_combo(name: str, action_src: pd.DataFrame, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": action_src["actionId"].astype(int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    point = load_sub(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    v173 = load_sub(V173_BASE, rally_uids)

    generated = []
    metrics = []
    for tag, path in ACTION_SOURCES.items():
        action = load_sub(path, rally_uids)
        name = f"submission_v191_{tag}__pv188_r186_w005_cap5__sr121.csv"
        info = write_combo(name, action, point, server)
        action_churn = float(np.mean(action["actionId"].astype(int).to_numpy() != v173["actionId"].astype(int).to_numpy()))
        point_churn = float(np.mean(point["pointId"].astype(int).to_numpy() != v173["pointId"].astype(int).to_numpy()))
        rec = {
            "submission": name,
            "action_source": tag,
            "action_source_path": str(path),
            "point_source": "v188_r186_w005_a0p05_cap0p05",
            "server_source": "r121",
            "action_churn_vs_v173": action_churn,
            "point_churn_vs_v173_r119": point_churn,
            "rows": int(len(point)),
        }
        rec.update(info)
        metrics.append(rec)
        generated.append(info)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTDIR / "v191_candidate_metrics.csv", index=False)
    report = {
        "verdict": "GENERATED",
        "point_anchor": str(POINT_ANCHOR),
        "server_anchor": str(SERVER_ANCHOR),
        "generated": metrics,
        "notes": [
            "All candidates use V188 r186_w005 cap5 point anchor.",
            "All candidates use R121 no-old server.",
            "Action is the only changed component.",
            "These replace older action probes that still used R119 point.",
        ],
    }
    (OUTDIR / "v191_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v191_report.md").write_text(
        "# V191 V188-cap5 Action Packager\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Generated submissions: `{len(generated)}`\n"
        f"- Point anchor: `{POINT_ANCHOR}`\n"
        f"- Server anchor: `{SERVER_ANCHOR}`\n\n"
        "## Generated\n\n"
        + "\n".join(
            f"- `{m['upload_path']}` action `{m['action_source']}`, action churn vs V173 `{m['action_churn_vs_v173']:.6f}`"
            for m in metrics
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v191_v188_cap5_action_packager.py", SRC_DEST)
    print(json.dumps({"generated_count": len(generated), "metrics": str(OUTDIR / "v191_candidate_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
