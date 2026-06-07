"""R141-R143 server structure system.

R141 decomposes the R127 old-server jump by old-covered/new-only slices.
R142 generates old-server hard/soft/sharpen/rank-preserve safety variants.
R143 adds scoreboard posterior only for new-only rows, leaving old-covered rows
handled by the chosen R142 policy.

This script intentionally keeps actionId/pointId fixed from the base
submission.  Only serverGetPoint is modified.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


OUTDIR = Path("r141_r143_server_system")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
TEST_NEW_PATHS = [Path("test_new.csv"), Path("data/raw/test_new.csv")]
ALIGN_PATHS = [
    Path("r27_old_server_alignment_report.csv"),
    Path("artifacts/summaries/r27_old_server_alignment_report.csv"),
]

BASE_SUBMISSIONS = {
    "r67_anchor": UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv",
    "r124_r67_r119_r121": UPLOAD_DIR / "submission_r124_r67_public_anchor__r119_point_w0p05__r121_min_w0p2.csv",
    "r124_r67_r120_r121": UPLOAD_DIR / "submission_r124_r67_public_anchor__r120_motif_point__r121_min_w0p2.csv",
}


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def load_first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {label}: {paths}")


def first_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)
        .copy()
        .reset_index(drop=True)
    )


def load_alignment() -> pd.DataFrame:
    path = load_first_existing(ALIGN_PATHS, "R27 alignment report")
    align = pd.read_csv(path)
    align = align.dropna(subset=["rally_uid", "old_serverGetPoint"]).copy()
    align["rally_uid"] = align["rally_uid"].astype(int)
    align["old_serverGetPoint"] = align["old_serverGetPoint"].astype(float)
    keep = ["rally_uid", "old_serverGetPoint"]
    for col in ["prefix_relation", "old_prefix_len", "new_prefix_len"]:
        if col in align.columns:
            keep.append(col)
    return align[keep].drop_duplicates("rally_uid")


def scoreboard_interval(df: pd.DataFrame) -> pd.DataFrame:
    first = first_rows(df)
    first["pmin"] = first[["gamePlayerId", "gamePlayerOtherId"]].min(axis=1)
    first["pmax"] = first[["gamePlayerId", "gamePlayerOtherId"]].max(axis=1)
    rows: list[dict] = []
    group_cols = ["match", "numberGame", "pmin", "pmax"]
    for _, group in first.sort_values(group_cols + ["rally_id"]).groupby(group_cols, sort=False):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            cur = group.iloc[i]
            nxt = group.iloc[i + 1]
            gap = int(nxt["rally_id"] - cur["rally_id"])
            next_score = {
                int(nxt["gamePlayerId"]): int(nxt["scoreSelf"]),
                int(nxt["gamePlayerOtherId"]): int(nxt["scoreOther"]),
            }
            server_id = int(cur["gamePlayerId"])
            receiver_id = int(cur["gamePlayerOtherId"])
            if server_id not in next_score or receiver_id not in next_score:
                continue
            ds = int(next_score[server_id]) - int(cur["scoreSelf"])
            dr = int(next_score[receiver_id]) - int(cur["scoreOther"])
            valid = gap > 0 and ds >= 0 and dr >= 0 and ds + dr == gap
            if not valid:
                continue
            rows.append(
                {
                    "rally_uid": int(cur["rally_uid"]),
                    "future_gap": int(gap),
                    "future_server_score_count": int(ds),
                    "future_receiver_score_count": int(dr),
                    "future_server_score_rate": float(ds / gap),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "rally_uid",
                "future_gap",
                "future_server_score_count",
                "future_receiver_score_count",
                "future_server_score_rate",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("rally_uid")


def load_base_submission(path: Path, rally_uids: pd.Series) -> pd.DataFrame:
    sub = pd.read_csv(path)
    aligned = pd.DataFrame({"rally_uid": rally_uids.astype(int).to_numpy()}).merge(
        sub, on="rally_uid", how="left", validate="one_to_one"
    )
    if aligned[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align with all test_new rally_uids")
    return aligned


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full(len(values), 0.5)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(1, len(values) - 1)


def old_server_variant(
    base_server: np.ndarray,
    covered: np.ndarray,
    old_label: np.ndarray,
    mode: str,
    weight: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    out = base_server.copy()
    if mode == "base":
        return np.clip(out, 1e-6, 1 - 1e-6)
    if mode == "hard":
        out[covered] = old_label[covered]
    elif mode == "soft":
        if weight is None:
            raise ValueError("soft mode needs weight")
        out[covered] = weight * old_label[covered] + (1.0 - weight) * base_server[covered]
    elif mode == "sharpen":
        if lo is None or hi is None:
            raise ValueError("sharpen mode needs lo/hi")
        out[covered] = np.where(old_label[covered] >= 0.5, hi, lo)
    elif mode == "rank_preserve":
        if lo is None or hi is None:
            raise ValueError("rank_preserve mode needs lo/hi")
        pos = covered & (old_label >= 0.5)
        neg = covered & (old_label < 0.5)
        # Preserve base-server ordering inside each old-label group while still
        # forcing all old negatives below all old positives.
        out[neg] = lo + 0.20 * rank01(base_server[neg])
        out[pos] = (hi - 0.20) + 0.20 * rank01(base_server[pos])
    else:
        raise ValueError(mode)
    return np.clip(out, 1e-6, 1 - 1e-6)


def apply_scoreboard_new_only(
    server: np.ndarray,
    base_server: np.ndarray,
    new_only: np.ndarray,
    score_valid: np.ndarray,
    score_rate: np.ndarray,
    gap: np.ndarray,
    mode: str,
    weight: float | None = None,
) -> np.ndarray:
    out = server.copy()
    mask = new_only & score_valid
    if not mask.any():
        return np.clip(out, 1e-6, 1 - 1e-6)
    if mode == "none":
        return np.clip(out, 1e-6, 1 - 1e-6)
    if mode == "uniform":
        if weight is None:
            raise ValueError("uniform scoreboard mode needs weight")
        w = np.full(len(out), weight, dtype=float)
    elif mode == "gap_calibrated":
        # Conservative because score_rate is interval aggregate, not direct label.
        w = np.zeros(len(out), dtype=float)
        w[gap == 2] = 0.30
        w[gap == 3] = 0.25
        w[gap == 4] = 0.20
        w[gap >= 5] = 0.12
    elif mode == "gap_calibrated_strong":
        w = np.zeros(len(out), dtype=float)
        w[gap == 2] = 0.45
        w[gap == 3] = 0.35
        w[gap == 4] = 0.28
        w[gap >= 5] = 0.18
    else:
        raise ValueError(mode)
    out[mask] = (1.0 - w[mask]) * base_server[mask] + w[mask] * score_rate[mask]
    return np.clip(out, 1e-6, 1 - 1e-6)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def distribution_info(
    name: str,
    server: np.ndarray,
    base_server: np.ndarray,
    covered: np.ndarray,
    old_label: np.ndarray,
    new_only: np.ndarray,
    score_valid: np.ndarray,
) -> dict:
    row: dict = {
        "candidate": name,
        "mean": float(server.mean()),
        "std": float(server.std()),
        "mad_vs_base": float(np.mean(np.abs(server - base_server))),
        "corr_vs_base": float(np.corrcoef(base_server, server)[0, 1]) if np.std(server) > 0 else None,
        "extreme_0_1_rate": float(np.mean((server <= 1e-5) | (server >= 1 - 1e-5))),
        "extreme_02_98_rate": float(np.mean((server <= 0.02) | (server >= 0.98))),
        "covered_mean": float(server[covered].mean()),
        "new_only_mean": float(server[new_only].mean()),
        "score_valid_new_only_mean": float(server[new_only & score_valid].mean()) if (new_only & score_valid).any() else None,
        "covered_mad_vs_base": float(np.mean(np.abs(server[covered] - base_server[covered]))),
        "new_only_mad_vs_base": float(np.mean(np.abs(server[new_only] - base_server[new_only]))),
        "covered_old_auc": safe_auc(old_label[covered].astype(int), server[covered]),
        "covered_old_pos_mean": float(server[covered & (old_label >= 0.5)].mean()),
        "covered_old_neg_mean": float(server[covered & (old_label < 0.5)].mean()),
    }
    return row


def write_submission(template: pd.DataFrame, server: np.ndarray, path: Path) -> None:
    out = template.copy()
    out["serverGetPoint"] = np.round(np.clip(server, 1e-6, 1.0 - 1e-6), 8)
    out.to_csv(path, index=False, float_format="%.8f")


def copy_candidate(path: Path) -> tuple[Path, Path]:
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / path.name
    selected_path = SELECTED_DIR / path.name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    return upload_path, selected_path


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    test_new = pd.read_csv(load_first_existing(TEST_NEW_PATHS, "test_new.csv"))
    rally_uids = first_rows(test_new)["rally_uid"].astype(int).reset_index(drop=True)
    align = load_alignment()
    old_map = align.set_index("rally_uid")["old_serverGetPoint"].to_dict()
    old_label = rally_uids.map(old_map).to_numpy(dtype=float)
    covered = ~np.isnan(old_label)
    old_label = np.nan_to_num(old_label, nan=0.5)
    new_only = ~covered

    score = scoreboard_interval(test_new)
    score_rate_map = score.set_index("rally_uid")["future_server_score_rate"].to_dict() if len(score) else {}
    gap_map = score.set_index("rally_uid")["future_gap"].to_dict() if len(score) else {}
    score_rate = rally_uids.map(score_rate_map).to_numpy(dtype=float)
    score_valid = ~np.isnan(score_rate)
    score_rate = np.nan_to_num(score_rate, nan=0.5)
    gap = rally_uids.map(gap_map).fillna(-1).to_numpy(dtype=int)

    coverage_rows: list[dict] = []
    masks = {
        "all": np.ones(len(rally_uids), dtype=bool),
        "old_covered": covered,
        "new_only": new_only,
        "score_valid": score_valid,
        "score_valid_old_covered": score_valid & covered,
        "score_valid_new_only": score_valid & new_only,
    }
    for subset, mask in masks.items():
        row = {"subset": subset, "rows": int(mask.sum()), "ratio": float(mask.mean())}
        if mask.any():
            row["base_old_label_mean_if_covered"] = (
                float(old_label[mask & covered].mean()) if (mask & covered).any() else None
            )
            row["score_rate_mean_if_valid"] = (
                float(score_rate[mask & score_valid].mean()) if (mask & score_valid).any() else None
            )
        coverage_rows.append(row)
    for g in sorted(set(gap[score_valid].tolist())):
        mask = score_valid & (gap == g)
        coverage_rows.append(
            {
                "subset": f"score_gap_{int(g)}",
                "rows": int(mask.sum()),
                "ratio": float(mask.mean()),
                "old_covered_rows": int((mask & covered).sum()),
                "new_only_rows": int((mask & new_only).sum()),
                "score_rate_mean_if_valid": float(score_rate[mask].mean()),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(OUTDIR / "r141_server_coverage_decomposition.csv", index=False)

    all_generated: list[dict] = []
    all_decomp: list[dict] = []
    r142_rows: list[dict] = []
    r143_rows: list[dict] = []

    old_specs = [
        {"name": "base", "mode": "base"},
        {"name": "oldhard", "mode": "hard"},
        {"name": "oldsoft_w0p5", "mode": "soft", "weight": 0.5},
        {"name": "oldsoft_w0p7", "mode": "soft", "weight": 0.7},
        {"name": "oldsoft_w0p9", "mode": "soft", "weight": 0.9},
        {"name": "oldsharpen010090", "mode": "sharpen", "lo": 0.10, "hi": 0.90},
        {"name": "oldsharpen005095", "mode": "sharpen", "lo": 0.05, "hi": 0.95},
        {"name": "oldrankpreserve010090", "mode": "rank_preserve", "lo": 0.10, "hi": 0.90},
        {"name": "oldrankpreserve005095", "mode": "rank_preserve", "lo": 0.05, "hi": 0.95},
    ]
    score_specs = [
        {"name": "none", "mode": "none"},
        {"name": "newscore_w0p1", "mode": "uniform", "weight": 0.1},
        {"name": "newscore_w0p2", "mode": "uniform", "weight": 0.2},
        {"name": "newscore_w0p35", "mode": "uniform", "weight": 0.35},
        {"name": "newscore_gapcal", "mode": "gap_calibrated"},
        {"name": "newscore_gapcalstrong", "mode": "gap_calibrated_strong"},
    ]

    for base_name, base_path in BASE_SUBMISSIONS.items():
        if not base_path.exists():
            continue
        sub = load_base_submission(base_path, rally_uids)
        base_server = sub["serverGetPoint"].to_numpy(dtype=float)

        # R141/R142: old-server variants without scoreboard.
        for spec in old_specs:
            server = old_server_variant(
                base_server,
                covered,
                old_label,
                spec["mode"],
                weight=spec.get("weight"),
                lo=spec.get("lo"),
                hi=spec.get("hi"),
            )
            candidate = f"r142_{base_name}_{spec['name']}"
            info = distribution_info(candidate, server, base_server, covered, old_label, new_only, score_valid)
            info.update({"base_name": base_name, "old_policy": spec["name"], "score_policy": "none"})
            all_decomp.append(info)
            if spec["name"] != "base":
                out_name = f"submission_{candidate}.csv"
                path = OUTDIR / out_name
                write_submission(sub, server, path)
                upload_path, selected_path = copy_candidate(path)
                info.update({"path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)})
                r142_rows.append(info.copy())
                all_generated.append(info.copy())

        # R143: combine selected old policies with new-only scoreboard.
        for old_spec in old_specs:
            if old_spec["name"] == "base":
                continue
            old_server = old_server_variant(
                base_server,
                covered,
                old_label,
                old_spec["mode"],
                weight=old_spec.get("weight"),
                lo=old_spec.get("lo"),
                hi=old_spec.get("hi"),
            )
            for score_spec in score_specs[1:]:
                server = apply_scoreboard_new_only(
                    old_server,
                    base_server,
                    new_only,
                    score_valid,
                    score_rate,
                    gap,
                    score_spec["mode"],
                    weight=score_spec.get("weight"),
                )
                candidate = f"r143_{base_name}_{old_spec['name']}_{score_spec['name']}"
                info = distribution_info(candidate, server, base_server, covered, old_label, new_only, score_valid)
                info.update(
                    {
                        "base_name": base_name,
                        "old_policy": old_spec["name"],
                        "score_policy": score_spec["name"],
                        "score_applied_rows": int((new_only & score_valid).sum()),
                    }
                )
                out_name = f"submission_{candidate}.csv"
                path = OUTDIR / out_name
                write_submission(sub, server, path)
                upload_path, selected_path = copy_candidate(path)
                info.update({"path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)})
                r143_rows.append(info.copy())
                all_generated.append(info.copy())
                all_decomp.append(info.copy())

    decomp_df = pd.DataFrame(all_decomp)
    r142_df = pd.DataFrame(r142_rows)
    r143_df = pd.DataFrame(r143_rows)
    decomp_df.to_csv(OUTDIR / "r141_server_decomposition_report.csv", index=False)
    r142_df.to_csv(OUTDIR / "r142_old_server_safety_search.csv", index=False)
    r143_df.to_csv(OUTDIR / "r143_new_only_scoreboard_search.csv", index=False)

    # Short recommendations prioritize r67 anchor and include hard, sharpen,
    # rank-preserve, and one new-only scoreboard hybrid.
    def pick(df: pd.DataFrame, query: str) -> dict | None:
        part = df.query(query) if len(df) else pd.DataFrame()
        if part.empty:
            return None
        # For public probing, sort by small base deviation among the intended policy.
        return part.sort_values(["mad_vs_base", "candidate"]).iloc[0].to_dict()

    recommendations = {
        "r142_r67_hard": pick(r142_df, "base_name == 'r67_anchor' and old_policy == 'oldhard'"),
        "r142_r67_sharpen005095": pick(
            r142_df, "base_name == 'r67_anchor' and old_policy == 'oldsharpen005095'"
        ),
        "r142_r67_rankpreserve005095": pick(
            r142_df, "base_name == 'r67_anchor' and old_policy == 'oldrankpreserve005095'"
        ),
        "r142_r67_soft09": pick(r142_df, "base_name == 'r67_anchor' and old_policy == 'oldsoft_w0p9'"),
        "r143_r67_sharpen_newscore_gapcal": pick(
            r143_df,
            "base_name == 'r67_anchor' and old_policy == 'oldsharpen005095' and score_policy == 'newscore_gapcal'",
        ),
        "r143_r67_soft09_newscore_gapcal": pick(
            r143_df,
            "base_name == 'r67_anchor' and old_policy == 'oldsoft_w0p9' and score_policy == 'newscore_gapcal'",
        ),
    }

    report = {
        "coverage": {
            "total_rows": int(len(rally_uids)),
            "old_covered_rows": int(covered.sum()),
            "old_coverage": float(covered.mean()),
            "new_only_rows": int(new_only.sum()),
            "score_valid_rows": int(score_valid.sum()),
            "score_valid_coverage": float(score_valid.mean()),
            "score_valid_old_covered_rows": int((score_valid & covered).sum()),
            "score_valid_new_only_rows": int((score_valid & new_only).sum()),
            "score_gap_counts": {
                str(int(g)): int((score_valid & (gap == g)).sum()) for g in sorted(set(gap[score_valid].tolist()))
            },
        },
        "recommendations": recommendations,
        "generated_count": int(len(all_generated)),
        "notes": [
            "R142 changes only old-covered rows.",
            "R143 changes old-covered rows via R142 policy and applies scoreboard posterior only to new-only score-valid rows.",
            "Scoreboard posterior is an interval aggregate and should be treated as public/test-structure sensitive.",
        ],
    }
    (OUTDIR / "r141_r143_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(__file__, Path("src/analysis") / Path(__file__).name)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
