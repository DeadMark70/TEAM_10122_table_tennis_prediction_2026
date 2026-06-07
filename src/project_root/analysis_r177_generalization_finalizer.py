"""R177 generalization-oriented finalizer.

This script does not introduce a new model.  It packages the strongest existing
signals into risk-tiered candidates and reports:

- action/point churn vs the public-validated R67 anchor,
- server shift vs R67,
- old-covered proxy ranking quality,
- private coverage/noise risk summaries from R144 when available.

The goal is to improve selection discipline and generalization, not to chase a
single public leaderboard number.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


OUTDIR = Path("r177_generalization_finalizer")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")

R67 = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R119 = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R121 = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R142_HARD = UPLOAD_DIR / "submission_r142_r67_anchor_oldhard.csv"
R142_SHARP = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"
R142_RANK = UPLOAD_DIR / "submission_r142_r67_anchor_oldrankpreserve005095.csv"
R143_SHARP_NEWSCORE = UPLOAD_DIR / "submission_r143_r67_anchor_oldsharpen005095_newscore_gapcal.csv"
R143_RANK_NEWSCORE = UPLOAD_DIR / "submission_r143_r67_anchor_oldrankpreserve005095_newscore_gapcal.csv"
V173_ACTION = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
V166_ACTION = UPLOAD_DIR / "submission_r166__ar166_best_action__pr119_public_point__sr121_min_w0p2.csv"
R27_ALIGN = Path("r27_old_server_alignment_report.csv")
R144_RISK = Path("r144_r146_risk_action_sweep/r144_private_risk_simulator.csv")


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align")
    return out


def load_old_alignment(rally_uids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    align = pd.read_csv(R27_ALIGN).dropna(subset=["rally_uid", "old_serverGetPoint"]).copy()
    align["rally_uid"] = align["rally_uid"].astype(int)
    align["old_serverGetPoint"] = align["old_serverGetPoint"].astype(int)
    merged = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(
        align[["rally_uid", "old_serverGetPoint"]].drop_duplicates("rally_uid"),
        on="rally_uid",
        how="left",
    )
    covered = merged["old_serverGetPoint"].notna().to_numpy()
    old_label = merged["old_serverGetPoint"].fillna(0).astype(int).to_numpy()
    return covered, old_label


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


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


def candidate_metrics(name: str, sub: pd.DataFrame, base: pd.DataFrame, covered: np.ndarray, old_label: np.ndarray, tier: str) -> dict:
    server = sub["serverGetPoint"].astype(float).to_numpy()
    base_server = base["serverGetPoint"].astype(float).to_numpy()
    row = {
        "candidate": name,
        "tier": tier,
        "action_churn_vs_r67": float((sub["actionId"].to_numpy() != base["actionId"].to_numpy()).mean()),
        "point_churn_vs_r67": float((sub["pointId"].to_numpy() != base["pointId"].to_numpy()).mean()),
        "server_mad_vs_r67": float(np.mean(np.abs(server - base_server))),
        "server_corr_vs_r67": float(np.corrcoef(server, base_server)[0, 1]) if np.std(server) > 0 else None,
        "covered_rows": int(covered.sum()),
        "new_only_rows": int((~covered).sum()),
        "covered_old_auc_proxy": safe_auc(old_label[covered], server[covered]),
        "covered_old_pos_mean": float(server[covered & (old_label == 1)].mean()),
        "covered_old_neg_mean": float(server[covered & (old_label == 0)].mean()),
        "extreme_005_095_rate": float(np.mean((server <= 0.05) | (server >= 0.95))),
    }
    row["covered_old_separation"] = row["covered_old_pos_mean"] - row["covered_old_neg_mean"]
    return row


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    base = load_sub(R67)
    rally_uids = base["rally_uid"].astype(int).to_numpy()
    r119 = load_sub(R119, rally_uids)
    r121 = load_sub(R121, rally_uids)
    hard = load_sub(R142_HARD, rally_uids)
    sharp = load_sub(R142_SHARP, rally_uids)
    rank = load_sub(R142_RANK, rally_uids)
    sharp_newscore = load_sub(R143_SHARP_NEWSCORE, rally_uids)
    rank_newscore = load_sub(R143_RANK_NEWSCORE, rally_uids)
    v173 = load_sub(V173_ACTION, rally_uids) if V173_ACTION.exists() else None
    v166 = load_sub(V166_ACTION, rally_uids) if V166_ACTION.exists() else None

    covered, old_label = load_old_alignment(rally_uids)
    specs = [
        ("submission_r177_no_old_safe_r67_r119_r121.csv", "no-old fallback", base, r119, r121),
        ("submission_r177_public_max_r67_r119_oldhard.csv", "public-max", base, r119, hard),
        ("submission_r177_public_safer_r67_r119_oldsharpen005095.csv", "safer-public", base, r119, sharp),
        ("submission_r177_private_rank_r67_r119_oldrankpreserve005095.csv", "private-safer", base, r119, rank),
        ("submission_r177_private_score_r67_r119_oldsharpen_newscore.csv", "private-safer", base, r119, sharp_newscore),
        ("submission_r177_private_rankscore_r67_r119_oldrank_newscore.csv", "private-safer", base, r119, rank_newscore),
    ]
    if v173 is not None:
        specs.append(("submission_r177_v173action_r119_oldsharpen.csv", "action-probe", v173, r119, sharp))
    if v166 is not None:
        specs.append(("submission_r177_v166action_r119_oldsharpen.csv", "action-probe", v166, r119, sharp))

    rows = []
    generated = []
    for name, tier, action_src, point_src, server_src in specs:
        path = compose(name, action_src, point_src, server_src, rally_uids)
        sub = load_sub(path, rally_uids)
        rows.append(candidate_metrics(name, sub, base, covered, old_label, tier))
        generated.append({"candidate": name, "path": str(path), "upload_path": str(UPLOAD_DIR / name), "tier": tier})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTDIR / "r177_candidate_risk_metrics.csv", index=False)

    risk_summary = []
    if R144_RISK.exists():
        risk = pd.read_csv(R144_RISK)
        # Pull a compact reference for the server policies used above.
        keep = risk[
            risk["candidate"].isin(
                ["oldhard", "oldsharpen005095", "oldrankpreserve005095", "oldsharpen005095_newscore"]
            )
            & risk["coverage_rate"].isin([0.0, 0.3, 0.67, 1.0])
            & risk["noise_rate"].isin([0.0, 0.03])
        ].copy()
        keep.to_csv(OUTDIR / "r177_r144_risk_reference.csv", index=False)
        risk_summary = keep.to_dict(orient="records")

    report = {
        "generated": generated,
        "metrics": metrics.sort_values(["tier", "server_mad_vs_r67"]).to_dict(orient="records"),
        "risk_reference": risk_summary[:40],
        "notes": [
            "R177 packages existing methods into public-max / safer-public / private-safer / no-old tiers.",
            "No new model is trained here; this is a selection and generalization-control layer.",
            "Use public-max only for ceiling probes; private-safer candidates reduce hard old-server dependence.",
        ],
    }
    (OUTDIR / "r177_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_metrics = metrics.to_csv(index=False)
    (OUTDIR / "r177_report.md").write_text(
        "# R177 Generalization Finalizer\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}` ({g['tier']})" for g in generated)
        + "\n\n## Candidate Metrics CSV\n\n```csv\n"
        + md_metrics
        + "```\n"
        + "\n",
        encoding="utf-8",
    )
    with open("experiments_log.md", "a", encoding="utf-8") as f:
        f.write(
            "\n\n## R177 generalization finalizer\n\n"
            "- Packaged existing methods into no-old fallback, public-max, safer-public, private-safer, and action-probe candidates.\n"
            f"- Generated {len(generated)} candidates. Metrics saved to `r177_generalization_finalizer/r177_candidate_risk_metrics.csv`.\n"
            "- This layer is for final selection/generalization control; it does not train a new model.\n"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
