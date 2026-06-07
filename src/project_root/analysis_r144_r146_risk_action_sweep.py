"""R144 private-risk simulator and R146 action-under-R142 sweep.

R144 is an offline risk report for old-server strategies under hypothetical
private old-coverage/noise scenarios.

R146 generates task-wise submissions by combining:
  action from selected action branches,
  point from stable point anchors,
  server from R142/R143 server policies.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


OUTDIR = Path("r144_r146_risk_action_sweep")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
ALIGN_PATHS = [
    Path("r27_old_server_alignment_report.csv"),
    Path("artifacts/summaries/r27_old_server_alignment_report.csv"),
]

BASE_R67 = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"

SERVER_CANDIDATES = {
    "oldhard": UPLOAD_DIR / "submission_r142_r67_anchor_oldhard.csv",
    "oldsharpen005095": UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv",
    "oldrankpreserve005095": UPLOAD_DIR / "submission_r142_r67_anchor_oldrankpreserve005095.csv",
    "oldsoft09": UPLOAD_DIR / "submission_r142_r67_anchor_oldsoft_w0p9.csv",
    "oldsharpen005095_newscore": UPLOAD_DIR
    / "submission_r143_r67_anchor_oldsharpen005095_newscore_gapcal.csv",
    "oldsoft09_newscore": UPLOAD_DIR / "submission_r143_r67_anchor_oldsoft_w0p9_newscore_gapcal.csv",
}

ACTION_CANDIDATES = {
    "r67": BASE_R67,
    "r86_r67_w0p25": UPLOAD_DIR / "submission_r86_r67_w0p25_v3point_current_server.csv",
    "r95_r93_r88": UPLOAD_DIR
    / "submission_r95_r93_r88_g0p05_fl0p5_cap1p25_s100_v3point_current_server.csv",
    "r96_r92_r93": UPLOAD_DIR / "submission_r96_r92w0p25_r93w0p2_v3point_current_server.csv",
    "r101_destiny_gru": UPLOAD_DIR / "submission_r101_r103_destiny_gru.csv",
    "r105_r101_distill": UPLOAD_DIR / "submission_r105_r101_distill_aw0p03_pw0p03.csv",
    "r111_remaining_moe": UPLOAD_DIR / "submission_r111_remaining_moe_gru.csv",
    "r115_r111_server": UPLOAD_DIR / "submission_r115_r111_server_w0p2.csv",
}

POINT_CANDIDATES = {
    "r67point": BASE_R67,
    "r119point": UPLOAD_DIR / "submission_r119_point_w0p05.csv",
    "r108point": UPLOAD_DIR / "submission_r108_tlp_selective_base_r67_anchor.csv",
}


def load_first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {label}: {paths}")


def load_alignment() -> pd.DataFrame:
    path = load_first_existing(ALIGN_PATHS, "R27 old server alignment")
    df = pd.read_csv(path).dropna(subset=["rally_uid", "old_serverGetPoint"]).copy()
    df["rally_uid"] = df["rally_uid"].astype(int)
    df["old_serverGetPoint"] = df["old_serverGetPoint"].astype(int)
    return df[["rally_uid", "old_serverGetPoint"]].drop_duplicates("rally_uid")


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    out = pd.DataFrame({"rally_uid": rally_uids}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align")
    return out


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def simulate_risk(
    name: str,
    server: np.ndarray,
    base_server: np.ndarray,
    covered: np.ndarray,
    old_label: np.ndarray,
    coverage_rates: list[float],
    noise_rates: list[float],
    seeds: list[int],
) -> list[dict]:
    rows: list[dict] = []
    covered_idx = np.flatnonzero(covered)
    y_full = old_label[covered].astype(int)
    for cov_rate in coverage_rates:
        n_keep = int(round(len(covered_idx) * cov_rate))
        for noise in noise_rates:
            metrics = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                keep_idx = rng.choice(covered_idx, size=n_keep, replace=False) if n_keep > 0 else np.array([], dtype=int)
                private_server = base_server.copy()
                private_server[keep_idx] = server[keep_idx]
                y_noisy = old_label.copy().astype(int)
                if noise > 0 and n_keep > 0:
                    flip_idx = rng.choice(keep_idx, size=int(round(n_keep * noise)), replace=False)
                    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
                eval_mask = np.zeros(len(server), dtype=bool)
                eval_mask[keep_idx] = True
                auc_keep = safe_auc(y_noisy[eval_mask], private_server[eval_mask]) if eval_mask.any() else None
                auc_all_covered = safe_auc(y_noisy[covered], private_server[covered])
                metrics.append(
                    {
                        "mad_vs_base": float(np.mean(np.abs(private_server - base_server))),
                        "covered_mad_vs_base": float(np.mean(np.abs(private_server[covered] - base_server[covered]))),
                        "kept_proxy_auc": auc_keep,
                        "all_covered_proxy_auc": auc_all_covered,
                        "kept_pos_mean": float(private_server[eval_mask & (y_noisy == 1)].mean())
                        if (eval_mask & (y_noisy == 1)).any()
                        else np.nan,
                        "kept_neg_mean": float(private_server[eval_mask & (y_noisy == 0)].mean())
                        if (eval_mask & (y_noisy == 0)).any()
                        else np.nan,
                    }
                )
            mdf = pd.DataFrame(metrics)
            row = {
                "candidate": name,
                "coverage_rate": cov_rate,
                "noise_rate": noise,
                "kept_rows_mean": n_keep,
                "mad_vs_base_mean": float(mdf["mad_vs_base"].mean()),
                "covered_mad_vs_base_mean": float(mdf["covered_mad_vs_base"].mean()),
                "kept_proxy_auc_mean": float(mdf["kept_proxy_auc"].mean()) if mdf["kept_proxy_auc"].notna().any() else None,
                "all_covered_proxy_auc_mean": float(mdf["all_covered_proxy_auc"].mean())
                if mdf["all_covered_proxy_auc"].notna().any()
                else None,
                "kept_pos_mean": float(mdf["kept_pos_mean"].mean()),
                "kept_neg_mean": float(mdf["kept_neg_mean"].mean()),
                "separation_mean": float((mdf["kept_pos_mean"] - mdf["kept_neg_mean"]).mean()),
            }
            rows.append(row)
    return rows


def write_submission(template: pd.DataFrame, action: pd.Series, point: pd.Series, server: pd.Series, path: Path) -> None:
    out = template[["rally_uid"]].copy()
    out["actionId"] = action.astype(int).to_numpy()
    out["pointId"] = point.astype(int).to_numpy()
    out["serverGetPoint"] = np.round(server.astype(float).to_numpy(), 8)
    out.to_csv(path, index=False, float_format="%.8f")


def copy_candidate(path: Path) -> tuple[Path, Path]:
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    upload = UPLOAD_DIR / path.name
    selected = SELECTED_DIR / path.name
    upload.write_bytes(path.read_bytes())
    selected.write_bytes(path.read_bytes())
    return upload, selected


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    base = load_sub(BASE_R67)
    rally_uids = base["rally_uid"].astype(int).to_numpy()
    align = load_alignment()
    old_map = align.set_index("rally_uid")["old_serverGetPoint"].to_dict()
    old_label = base["rally_uid"].map(old_map).to_numpy(dtype=float)
    covered = ~np.isnan(old_label)
    old_label = np.nan_to_num(old_label, nan=0.5).astype(int)
    base_server = base["serverGetPoint"].to_numpy(dtype=float)

    # R144
    risk_rows: list[dict] = []
    risk_candidates = {
        k: v for k, v in SERVER_CANDIDATES.items() if v.exists() and k in {"oldhard", "oldsharpen005095", "oldrankpreserve005095", "oldsoft09", "oldsharpen005095_newscore"}
    }
    for name, path in risk_candidates.items():
        sub = load_sub(path, rally_uids)
        server = sub["serverGetPoint"].to_numpy(dtype=float)
        risk_rows.extend(
            simulate_risk(
                name,
                server,
                base_server,
                covered,
                old_label,
                coverage_rates=[0.0, 0.3, 0.5, 0.67, 1.0],
                noise_rates=[0.0, 0.01, 0.03],
                seeds=list(range(10)),
            )
        )
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(OUTDIR / "r144_private_risk_simulator.csv", index=False)

    # R146
    action_loaded = {k: load_sub(v, rally_uids) for k, v in ACTION_CANDIDATES.items() if v.exists()}
    point_loaded = {k: load_sub(v, rally_uids) for k, v in POINT_CANDIDATES.items() if v.exists()}
    server_loaded = {k: load_sub(v, rally_uids) for k, v in SERVER_CANDIDATES.items() if v.exists()}

    # Keep the sweep bounded: combine selected action branches with two point
    # anchors and four server policies. This yields upload-ready candidates
    # without flooding the queue.
    point_keys = [k for k in ["r67point", "r119point"] if k in point_loaded]
    server_keys = [
        k
        for k in ["oldhard", "oldsharpen005095", "oldrankpreserve005095", "oldsharpen005095_newscore"]
        if k in server_loaded
    ]
    action_keys = [
        k
        for k in [
            "r67",
            "r86_r67_w0p25",
            "r95_r93_r88",
            "r96_r92_r93",
            "r101_destiny_gru",
            "r105_r101_distill",
            "r111_remaining_moe",
        ]
        if k in action_loaded
    ]
    rows: list[dict] = []
    for akey in action_keys:
        for pkey in point_keys:
            for skey in server_keys:
                name = f"submission_r146_a{akey}__p{pkey}__s{skey}.csv"
                path = OUTDIR / name
                write_submission(
                    base,
                    action_loaded[akey]["actionId"],
                    point_loaded[pkey]["pointId"],
                    server_loaded[skey]["serverGetPoint"],
                    path,
                )
                upload, selected = copy_candidate(path)
                action_churn = float((action_loaded[akey]["actionId"].to_numpy() != base["actionId"].to_numpy()).mean())
                point_churn = float((point_loaded[pkey]["pointId"].to_numpy() != base["pointId"].to_numpy()).mean())
                server_mad = float(
                    np.mean(np.abs(server_loaded[skey]["serverGetPoint"].to_numpy(dtype=float) - base_server))
                )
                rows.append(
                    {
                        "candidate": name,
                        "action_source": akey,
                        "point_source": pkey,
                        "server_source": skey,
                        "action_churn_vs_r67": action_churn,
                        "point_churn_vs_r67": point_churn,
                        "server_mad_vs_r67": server_mad,
                        "path": str(path),
                        "upload_path": str(upload),
                        "selected_path": str(selected),
                    }
                )
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(OUTDIR / "r146_action_under_r142_sweep.csv", index=False)

    # Recommended probes: preserve R67 action first, then one alternative action
    # with stable R67 point and safer server.
    def first_match(action: str, point: str, server: str) -> dict | None:
        part = sweep_df[
            sweep_df["action_source"].eq(action)
            & sweep_df["point_source"].eq(point)
            & sweep_df["server_source"].eq(server)
        ]
        return part.iloc[0].to_dict() if len(part) else None

    recommendations = {
        "public_max_baseline": first_match("r67", "r67point", "oldhard"),
        "safer_sharpen": first_match("r67", "r67point", "oldsharpen005095"),
        "private_safer_rankpreserve": first_match("r67", "r67point", "oldrankpreserve005095"),
        "point_component_r119": first_match("r67", "r119point", "oldsharpen005095"),
        "action_alt_r96": first_match("r96_r92_r93", "r67point", "oldsharpen005095"),
        "action_alt_r105": first_match("r105_r101_distill", "r67point", "oldsharpen005095"),
        "newscore_hybrid": first_match("r67", "r67point", "oldsharpen005095_newscore"),
    }
    report = {
        "coverage": {
            "rows": int(len(base)),
            "old_covered_rows": int(covered.sum()),
            "old_coverage": float(covered.mean()),
        },
        "r144_candidates": list(risk_candidates.keys()),
        "r146_counts": {
            "action_sources": action_keys,
            "point_sources": point_keys,
            "server_sources": server_keys,
            "generated": int(len(sweep_df)),
        },
        "recommendations": recommendations,
        "external_data_note": "See final response for external datasets to expand pretraining.",
    }
    (OUTDIR / "r144_r146_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(__file__, Path("src/analysis") / Path(__file__).name)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
