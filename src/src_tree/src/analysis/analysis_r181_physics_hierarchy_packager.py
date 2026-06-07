"""R181 physics hierarchy packager.

Composes R179/R180 outputs into risk-tiered candidates.  R177 remains
unchanged; old-server variants are emitted only as diagnostic/sensitive rows.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("r181_physics_hierarchy_packager")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")

R67 = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R119 = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R121 = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R142_SHARP = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"
R179_REPORT = Path("r179_action_physics_hierarchy/r179_report.json")
R180_REPORT = Path("r180_point_physics_calibration/r180_report.json")


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align with R67")
    return out


def report_first_generated(report_path: Path) -> Path:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    generated = data.get("generated") or []
    if not generated:
        raise ValueError(f"{report_path} has no generated candidates")
    return Path(generated[0]["upload_path"])


def compose(name: str, action_src: pd.DataFrame, point_src: pd.DataFrame, server_src: pd.DataFrame, rally_uids: np.ndarray) -> Path:
    OUTDIR.mkdir(exist_ok=True)
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)})
    out["actionId"] = action_src["actionId"].astype(int).to_numpy()
    out["pointId"] = point_src["pointId"].astype(int).to_numpy()
    out["serverGetPoint"] = np.round(np.clip(server_src["serverGetPoint"].astype(float).to_numpy(), 1e-6, 1 - 1e-6), 8)
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, UPLOAD_DIR / name)
    shutil.copy2(path, SELECTED_DIR / name)
    return path


def candidate_metrics(name: str, sub: pd.DataFrame, base: pd.DataFrame, tier: str, action_component: str, point_component: str, server_component: str) -> dict:
    server = sub["serverGetPoint"].astype(float).to_numpy()
    base_server = base["serverGetPoint"].astype(float).to_numpy()
    return {
        "candidate": name,
        "tier": tier,
        "action_component": action_component,
        "point_component": point_component,
        "server_component": server_component,
        "action_churn_vs_r67": float((sub["actionId"].to_numpy() != base["actionId"].to_numpy()).mean()),
        "point_churn_vs_r67": float((sub["pointId"].to_numpy() != base["pointId"].to_numpy()).mean()),
        "server_mad_vs_r67": float(np.mean(np.abs(server - base_server))),
        "server_corr_vs_r67": float(np.corrcoef(server, base_server)[0, 1]) if np.std(server) > 0 and np.std(base_server) > 0 else None,
    }


def append_experiment_log(report: dict) -> None:
    with open("experiments_log.md", "a", encoding="utf-8") as f:
        f.write(
            "\n\n## R179-R181 physics hierarchy experiments\n\n"
            "- R179 implemented action phase/family/style hierarchy priors as a new action-only experiment line.\n"
            "- R180 implemented point terminal/depth/side calibration and long-side local redistribution while preserving the direct point decoder anchor.\n"
            "- R181 packaged no-old clean candidates separately from old-server diagnostic candidates.\n"
            "- R119 remains a point component probe, not a low-churn safe default.\n"
            "- TTMATCH is not used by these scripts.\n"
            f"- Generated R181 candidates: `{len(report['generated'])}`. Metrics: `r181_physics_hierarchy_packager/r181_candidate_risk_metrics.csv`.\n"
        )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    base = load_sub(R67)
    rally_uids = base["rally_uid"].astype(int).to_numpy()
    r119 = load_sub(R119, rally_uids)
    r121 = load_sub(R121, rally_uids)
    sharp = load_sub(R142_SHARP, rally_uids)
    r179 = load_sub(report_first_generated(R179_REPORT), rally_uids)
    r180 = load_sub(report_first_generated(R180_REPORT), rally_uids)

    specs = [
        (
            "submission_r181_no_old_r179action_r67point_r121server.csv",
            "no-old clean",
            r179,
            base,
            r121,
            "r179",
            "r67",
            "r121",
        ),
        (
            "submission_r181_no_old_r67action_r180point_r121server.csv",
            "no-old clean",
            base,
            r180,
            r121,
            "r67",
            "r180",
            "r121",
        ),
        (
            "submission_r181_no_old_r179action_r180point_r121server.csv",
            "no-old clean",
            r179,
            r180,
            r121,
            "r179",
            "r180",
            "r121",
        ),
        (
            "submission_r181_no_old_r179action_r119point_r121server.csv",
            "no-old r119 point probe",
            r179,
            r119,
            r121,
            "r179",
            "r119_probe",
            "r121",
        ),
        (
            "submission_r181_diagnostic_r179action_r67point_oldsharpen005095.csv",
            "old-server diagnostic",
            r179,
            base,
            sharp,
            "r179",
            "r67",
            "oldsharpen005095",
        ),
        (
            "submission_r181_diagnostic_r67action_r180point_oldsharpen005095.csv",
            "old-server diagnostic",
            base,
            r180,
            sharp,
            "r67",
            "r180",
            "oldsharpen005095",
        ),
        (
            "submission_r181_diagnostic_r179action_r180point_oldsharpen005095.csv",
            "old-server diagnostic",
            r179,
            r180,
            sharp,
            "r179",
            "r180",
            "oldsharpen005095",
        ),
        (
            "submission_r181_diagnostic_r179action_r119point_oldsharpen005095.csv",
            "old-server diagnostic r119 point probe",
            r179,
            r119,
            sharp,
            "r179",
            "r119_probe",
            "oldsharpen005095",
        ),
    ]

    generated = []
    metrics = []
    for name, tier, action_src, point_src, server_src, action_component, point_component, server_component in specs:
        path = compose(name, action_src, point_src, server_src, rally_uids)
        sub = load_sub(path, rally_uids)
        generated.append({"candidate": name, "path": str(path), "upload_path": str(UPLOAD_DIR / name), "tier": tier})
        metrics.append(candidate_metrics(name, sub, base, tier, action_component, point_component, server_component))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTDIR / "r181_candidate_risk_metrics.csv", index=False)
    report = {
        "generated": generated,
        "metrics": metrics_df.to_dict(orient="records"),
        "notes": [
            "No-old clean candidates use R121 server, not old-server labels.",
            "R119 candidates are explicitly marked as point probes.",
            "Old-server candidates use oldsharpen005095 only and are diagnostic/sensitive.",
            "R177 is not modified by R181.",
        ],
    }
    (OUTDIR / "r181_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r181_report.md").write_text(
        "# R181 Physics Hierarchy Packager\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}` ({g['tier']})" for g in generated)
        + "\n\n## Candidate Metrics CSV\n\n```csv\n"
        + metrics_df.to_csv(index=False)
        + "```\n",
        encoding="utf-8",
    )
    append_experiment_log(report)
    shutil.copy2("analysis_r181_physics_hierarchy_packager.py", "src/analysis/analysis_r181_physics_hierarchy_packager.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
