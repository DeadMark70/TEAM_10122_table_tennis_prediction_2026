"""R28C/R29 server posterior and terminal-point constraint.

R28C creates old-covered soft server pseudo-label submissions that are less
aggressive than direct replacement.  R29A then uses a server posterior to
adjust the V3 pointId=0 probability through the final-parity compatibility
rule, leaving actionId unchanged.

Compliance note:
  Files based on direct old-server labels or soft pseudo-labels are more
  sensitive than teacher-only files.  Treat them as candidates pending organizer
  clarification.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    validate_raw_data,
)
from baseline_v3 import apply_segmented_multipliers
from generate_r1_submission import compose_v3_full


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R28C/R29 terminal constraint experiment.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test-new", default="test_new.csv")
    parser.add_argument("--test-old", default="test_old.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--base-submission", default="submission_r1.csv")
    parser.add_argument("--teacher-submission", default="submission_r28_teacher_ow0p25_bw0p5.csv")
    parser.add_argument("--soft-weights", nargs="+", type=float, default=[0.1, 0.2, 0.35, 0.5])
    parser.add_argument("--soft-eps", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--beta-grid", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--r28c-report", default="r28c_soft_server_report.csv")
    parser.add_argument("--r29-report", default="r29_point0_constraint_report.csv")
    parser.add_argument("--selected", default="r29_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r29.json")
    parser.add_argument("--recommendation", default="r29_recommendation.md")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def old_server_map(test_old: pd.DataFrame) -> dict[int, int]:
    first = (
        test_old.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)[["rally_uid", "serverGetPoint"]]
    )
    return {int(r.rally_uid): int(r.serverGetPoint) for r in first.itertuples(index=False)}


def align_submission(path: str, rally_uids: pd.Series) -> pd.DataFrame:
    sub = pd.read_csv(path)
    aligned = pd.DataFrame({"rally_uid": rally_uids.astype(int).to_numpy()}).merge(
        sub, on="rally_uid", how="left", validate="one_to_one"
    )
    if aligned[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"Submission {path} does not align to test rows.")
    return aligned


def write_submission(template: pd.DataFrame, point_pred: np.ndarray, server_prob: np.ndarray, path: Path) -> None:
    out = template.copy()
    out["pointId"] = point_pred.astype(int)
    out["serverGetPoint"] = np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8)
    out.to_csv(path, index=False, float_format="%.8f")


def point0_adjust(point_prob: np.ndarray, server_prob: np.ndarray, prefix_len: np.ndarray, beta: float) -> np.ndarray:
    adjusted = point_prob.copy()
    next_strike = prefix_len.astype(int) + 1
    terminal_server_win = (next_strike % 2 == 0)
    compat = np.where(terminal_server_win, server_prob, 1.0 - server_prob)
    adjusted[:, 0] *= np.exp(float(beta) * (compat - 0.5))
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def write_recommendation(path: Path, selected: dict) -> None:
    lines = [
        "# R29 server-terminal constraint recommendation",
        "",
        "R28C and R29 are sensitivity experiments.  They should not be submitted until the organizer clarifies old-server usage.",
        "",
        "## R28C",
        "",
        f"- Lowest-churn soft server candidate: `{selected['r28c_lowest_churn']['submission']}`",
        f"- Candidate MAD vs base server: {selected['r28c_lowest_churn']['mad_vs_base']:.6f}",
        "",
        "## R29A",
        "",
        f"- Conservative point0 candidate: `{selected['r29_conservative']['submission']}`",
        f"- Point churn vs R1: {selected['r29_conservative']['point_churn_vs_base']:.6f}",
        f"- Server source: {selected['r29_conservative']['server_source']}, beta={selected['r29_conservative']['beta']}",
        "",
        "## Interpretation",
        "",
        "- R29 only changes `pointId` through the terminal class and leaves `actionId` unchanged.",
        "- If direct old-server is allowed, the direct-source R29 files estimate the strongest point0/server route.",
        "- If direct old-server is not allowed, prefer teacher or very-low-churn soft candidates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_raw = pd.read_csv(args.train)
    test_new_raw = pd.read_csv(args.test_new)
    test_old = pd.read_csv(args.test_old)
    old_map = old_server_map(test_old)

    train = add_role_and_score_features(train_raw)
    test_new = add_role_and_score_features(test_new_raw)
    validate_raw_data(train_raw, test_new_raw)

    v3_oof = load_pickle(args.v3_oof)
    test_prefix, _, v3_point, _ = compose_v3_full(train, test_new, v3_oof["tuning"])
    rally_uids = test_prefix["rally_uid"].astype(int)
    prefix_len = test_prefix["prefix_len"].to_numpy(dtype=int)

    base = align_submission(args.base_submission, rally_uids)
    teacher = align_submission(args.teacher_submission, rally_uids)
    base_server = base["serverGetPoint"].to_numpy(dtype=float)
    teacher_server = teacher["serverGetPoint"].to_numpy(dtype=float)
    base_point = base["pointId"].to_numpy(dtype=int)

    matched_mask = rally_uids.isin(old_map).to_numpy()
    old_label = rally_uids.map(old_map).fillna(np.nan).to_numpy(dtype=float)

    # R28C: soft old-covered pseudo-labels.
    r28c_rows: list[dict] = []
    soft_candidates: dict[str, np.ndarray] = {}
    for eps in args.soft_eps:
        pseudo = np.where(old_label >= 0.5, 1.0 - eps, eps)
        for w in args.soft_weights:
            server = teacher_server.copy()
            server[matched_mask] = (1.0 - w) * teacher_server[matched_mask] + w * pseudo[matched_mask]
            server = np.clip(server, 1e-6, 1.0 - 1e-6)
            name = f"soft_eps{str(eps).replace('.', 'p')}_w{str(w).replace('.', 'p')}"
            out_path = Path(f"submission_r28c_{name}.csv")
            tmp = base.copy()
            tmp["serverGetPoint"] = np.round(server, 8)
            tmp.to_csv(out_path, index=False, float_format="%.8f")
            row = {
                "variant": name,
                "eps": float(eps),
                "soft_weight": float(w),
                "corr_with_base": float(np.corrcoef(base_server, server)[0, 1]),
                "mad_vs_base": float(np.mean(np.abs(base_server - server))),
                "mad_vs_teacher": float(np.mean(np.abs(teacher_server - server))),
                "matched_mean": float(server[matched_mask].mean()),
                "new_only_mean": float(server[~matched_mask].mean()),
                "submission": str(out_path),
            }
            r28c_rows.append(row)
            soft_candidates[name] = server

    pd.DataFrame(r28c_rows).to_csv(args.r28c_report, index=False)

    direct_server = base_server.copy()
    direct_server[matched_mask] = old_label[matched_mask] * 0.998 + 0.001
    server_sources = {
        "base": base_server,
        "teacher": teacher_server,
        "direct_diagnostic": np.clip(direct_server, 1e-6, 1.0 - 1e-6),
    }
    # Add two soft representative sources.
    for name in ["soft_eps0p05_w0p1", "soft_eps0p05_w0p2", "soft_eps0p1_w0p1"]:
        if name in soft_candidates:
            server_sources[name] = soft_candidates[name]

    r29_rows: list[dict] = []
    for source_name, server_prob in server_sources.items():
        for beta in args.beta_grid:
            adj_point = point0_adjust(v3_point, server_prob, prefix_len, beta)
            point_pred = apply_segmented_multipliers(
                test_prefix, adj_point, v3_oof["tuning"].point_multipliers, POINT_CLASSES, v3_oof["tuning"].bins_mode
            )
            churn = float(np.mean(point_pred != base_point))
            point0_rate = float(np.mean(point_pred == 0))
            out_path = Path(f"submission_r29_{source_name}_beta{str(beta).replace('.', 'p')}.csv")
            write_submission(base, point_pred, server_prob, out_path)
            r29_rows.append(
                {
                    "server_source": source_name,
                    "beta": float(beta),
                    "point_churn_vs_base": churn,
                    "point0_rate": point0_rate,
                    "server_corr_with_base": float(np.corrcoef(base_server, server_prob)[0, 1]),
                    "server_mad_vs_base": float(np.mean(np.abs(base_server - server_prob))),
                    "submission": str(out_path),
                }
            )

    r29 = pd.DataFrame(r29_rows)
    r29.to_csv(args.r29_report, index=False)
    r28c = pd.DataFrame(r28c_rows)
    lowest_churn = r28c.sort_values(["mad_vs_base", "soft_weight", "eps"]).iloc[0].to_dict()
    conservative = (
        r29[r29["server_source"].isin(["teacher", "soft_eps0p05_w0p1"])]
        .sort_values(["point_churn_vs_base", "server_mad_vs_base", "beta"])
        .iloc[0]
        .to_dict()
    )
    selected = {
        "r28c_lowest_churn": lowest_churn,
        "r29_conservative": conservative,
        "notes": [
            "R29 changes pointId only through point0 terminal compatibility.",
            "Direct diagnostic variants depend on old-server direct labels and are high sensitivity.",
        ],
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    feature_report = {
        "args": vars(args),
        "rows": int(len(base)),
        "old_coverage": int(matched_mask.sum()),
        "old_coverage_ratio": float(matched_mask.mean()),
        "base_point0_rate": float(np.mean(base_point == 0)),
        "selected": selected,
    }
    Path(args.feature_report).write_text(json.dumps(feature_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_recommendation(Path(args.recommendation), selected)

    print("R28C candidates")
    print(r28c.sort_values("mad_vs_base").head(5).to_string(index=False))
    print("R29 candidates")
    print(r29.sort_values(["point_churn_vs_base", "server_mad_vs_base"]).head(10).to_string(index=False))
    print(f"wrote {args.r28c_report}, {args.r29_report}, {args.selected}, {args.recommendation}")


if __name__ == "__main__":
    main()
