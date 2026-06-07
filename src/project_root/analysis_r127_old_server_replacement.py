"""R127 old-test server direct/soft replacement diagnostics.

This is a diagnostic branch to quantify how much public score can be explained
by the old test serverGetPoint overlap.  It only modifies the serverGetPoint
column for rally_uid values covered by old test alignment; action/point are
kept from the chosen base submission.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("r127_old_server_replacement")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
ALIGN_PATHS = [
    Path("r27_old_server_alignment_report.csv"),
    Path("artifacts/summaries/r27_old_server_alignment_report.csv"),
]


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def load_alignment() -> pd.DataFrame:
    for path in ALIGN_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("Could not find R27 old server alignment report.")
    df = df.dropna(subset=["rally_uid", "old_serverGetPoint"]).copy()
    df["rally_uid"] = df["rally_uid"].astype(int)
    df["old_serverGetPoint"] = df["old_serverGetPoint"].astype(float)
    return df[["rally_uid", "old_serverGetPoint", "prefix_relation", "old_prefix_len", "new_prefix_len"]]


def apply_old_server(base_path: Path, align: pd.DataFrame, mode: str, weight: float) -> tuple[pd.DataFrame, dict]:
    sub = pd.read_csv(base_path)
    if "rally_uid" not in sub.columns or "serverGetPoint" not in sub.columns:
        raise ValueError(f"{base_path} is not a valid submission.")
    merged = sub.merge(align[["rally_uid", "old_serverGetPoint"]], on="rally_uid", how="left")
    covered = merged["old_serverGetPoint"].notna().to_numpy()
    base_server = merged["serverGetPoint"].astype(float).to_numpy()
    old_label = merged["old_serverGetPoint"].fillna(0.5).astype(float).to_numpy()
    if mode == "hard":
        new_server = np.where(covered, old_label, base_server)
    elif mode == "soft":
        new_server = np.where(covered, weight * old_label + (1.0 - weight) * base_server, base_server)
    elif mode == "sharpen":
        soft = weight * old_label + (1.0 - weight) * base_server
        new_server = np.where(covered, np.where(soft >= 0.5, 0.98, 0.02), base_server)
    else:
        raise ValueError(mode)
    out = sub.copy()
    out["serverGetPoint"] = np.round(np.clip(new_server, 1e-6, 1.0 - 1e-6), 8)
    info = {
        "base_path": str(base_path),
        "mode": mode,
        "weight": weight,
        "covered": int(covered.sum()),
        "total": int(len(sub)),
        "coverage": float(covered.mean()),
        "base_server_mean": float(base_server.mean()),
        "covered_base_mean": float(base_server[covered].mean()) if covered.any() else None,
        "covered_old_mean": float(old_label[covered].mean()) if covered.any() else None,
        "server_mad_vs_base": float(np.mean(np.abs(new_server - base_server))),
        "covered_server_mad_vs_base": float(np.mean(np.abs(new_server[covered] - base_server[covered]))) if covered.any() else None,
        "extreme_rate": float(np.mean((new_server <= 0.02) | (new_server >= 0.98))),
    }
    return out, info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    align = load_alignment()
    coverage_report = {
        "matched_rows": int(len(align)),
        "matched_old_mean": float(align["old_serverGetPoint"].mean()),
        "prefix_relation_counts": align["prefix_relation"].value_counts(dropna=False).to_dict(),
        "old_prefix_len_mean": float(align["old_prefix_len"].mean()),
        "new_prefix_len_mean": float(align["new_prefix_len"].mean()),
    }
    base_submissions = {
        "r67_anchor": UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv",
        "r124_r67_r120_r121": UPLOAD_DIR / "submission_r124_r67_public_anchor__r120_motif_point__r121_min_w0p2.csv",
        "r124_r67_r119_r121": UPLOAD_DIR / "submission_r124_r67_public_anchor__r119_point_w0p05__r121_min_w0p2.csv",
        "r124_local_best": UPLOAD_DIR / "submission_r124_r120_local_motif__r120_motif_point__r121_mean_w0p35.csv",
    }
    weights = [0.5, 0.7, 0.9]
    rows = []
    generated = []
    for base_name, base_path in base_submissions.items():
        if not base_path.exists():
            continue
        specs = [("hard", 1.0), ("sharpen", 0.9)] + [("soft", w) for w in weights]
        for mode, weight in specs:
            out, info = apply_old_server(base_path, align, mode, weight)
            if mode == "hard":
                name = f"submission_r127_{base_name}_oldserver_hard.csv"
            elif mode == "sharpen":
                name = f"submission_r127_{base_name}_oldserver_sharpen_w{clean_float(weight)}.csv"
            else:
                name = f"submission_r127_{base_name}_oldserver_soft_w{clean_float(weight)}.csv"
            path = OUTDIR / name
            out.to_csv(path, index=False, float_format="%.8f")
            upload_path = UPLOAD_DIR / name
            selected_path = SELECTED_DIR / name
            upload_path.write_bytes(path.read_bytes())
            selected_path.write_bytes(path.read_bytes())
            info.update({"base_name": base_name, "candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)})
            rows.append(info)
            generated.append(info)
    report = {"coverage": coverage_report, "generated": generated}
    pd.DataFrame(rows).to_csv(OUTDIR / "r127_old_server_replacement_report.csv", index=False)
    (OUTDIR / "r127_old_server_replacement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r127_old_server_replacement.py", "src/analysis/analysis_r127_old_server_replacement.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
