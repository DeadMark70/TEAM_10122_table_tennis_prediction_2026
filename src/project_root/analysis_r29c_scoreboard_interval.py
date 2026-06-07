"""R29C scoreboard interval posterior for serverGetPoint.

This is a sensitivity experiment.  It uses only public test_new fields:
match, numberGame, rally_id, player ids, and starting score.  For consecutive
rallies in the same match/game/player-pair, score progression gives an
aggregate win rate for the interval.  That aggregate is not the current rally
label when the gap is greater than 1, but it is a strong server posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R29C scoreboard interval posterior.")
    parser.add_argument("--test-new", default="test_new.csv")
    parser.add_argument("--test-old", default="test_old.csv")
    parser.add_argument("--base-submission", default="submission_r1.csv")
    parser.add_argument("--teacher-submission", default="submission_r28_teacher_ow0p25_bw0p5.csv")
    parser.add_argument("--soft-submission", default="submission_r28c_soft_eps0p1_w0p1.csv")
    parser.add_argument("--direct-submission", default="submission_r28_old_server_direct_diagnostic.csv")
    parser.add_argument("--score-weights", nargs="+", type=float, default=[0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--report", default="r29c_scoreboard_interval_report.csv")
    parser.add_argument("--coverage-report", default="r29c_scoreboard_coverage_report.csv")
    parser.add_argument("--selected", default="r29c_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r29c.json")
    parser.add_argument("--recommendation", default="r29c_recommendation.md")
    return parser.parse_args()


def first_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)
        .copy()
        .reset_index(drop=True)
    )


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
    return pd.DataFrame(rows)


def old_server_map(test_old: pd.DataFrame) -> dict[int, int]:
    first = first_rows(test_old)
    return {int(r.rally_uid): int(r.serverGetPoint) for r in first.itertuples(index=False)}


def align_submission(path: str, rally_uids: pd.Series) -> pd.DataFrame:
    sub = pd.read_csv(path)
    aligned = pd.DataFrame({"rally_uid": rally_uids.astype(int).to_numpy()}).merge(
        sub, on="rally_uid", how="left", validate="one_to_one"
    )
    if aligned[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align with all test_new rally_uids")
    return aligned


def write_submission(template: pd.DataFrame, server: np.ndarray, path: Path) -> None:
    out = template.copy()
    out["serverGetPoint"] = np.round(np.clip(server, 1e-6, 1.0 - 1e-6), 8)
    out.to_csv(path, index=False, float_format="%.8f")


def write_recommendation(path: Path, selected: dict) -> None:
    lines = [
        "# R29C scoreboard interval posterior",
        "",
        "R29C uses public test_new score progression.  It is less direct than old-server replacement, but still rule-sensitive because it uses batch-level future score context.",
        "",
        "## Recommended candidates",
        "",
        f"- Conservative all-valid candidate: `{selected['conservative_all_valid']['submission']}`",
        f"- Conservative new-only candidate: `{selected['conservative_new_only']['submission']}`",
        f"- Diagnostic strongest candidate: `{selected['strongest']['submission']}`",
        "",
        "## Notes",
        "",
        "- `mode=all_valid` blends scoreboard posterior on every rally with a valid next-score interval.",
        "- `mode=new_only_valid` only applies scoreboard posterior to the 609 rows not covered by old server labels.",
        "- Keep direct-source variants as diagnostics until organizer clarification.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    test_new = pd.read_csv(args.test_new)
    test_old = pd.read_csv(args.test_old)
    rally_uids = first_rows(test_new)["rally_uid"].astype(int).reset_index(drop=True)
    base = align_submission(args.base_submission, rally_uids)
    teacher = align_submission(args.teacher_submission, rally_uids)
    soft = align_submission(args.soft_submission, rally_uids)
    direct = align_submission(args.direct_submission, rally_uids)

    score = scoreboard_interval(test_new)
    score_map = score.set_index("rally_uid")["future_server_score_rate"].to_dict()
    gap_map = score.set_index("rally_uid")["future_gap"].to_dict()
    score_rate = rally_uids.map(score_map).to_numpy(dtype=float)
    score_valid = ~np.isnan(score_rate)
    old_map = old_server_map(test_old)
    old_covered = rally_uids.isin(old_map).to_numpy()
    new_only = ~old_covered

    coverage_rows = []
    for name, mask in {
        "all": np.ones(len(rally_uids), dtype=bool),
        "old_covered": old_covered,
        "new_only": new_only,
        "score_valid": score_valid,
        "score_valid_old_covered": score_valid & old_covered,
        "score_valid_new_only": score_valid & new_only,
    }.items():
        coverage_rows.append({"subset": name, "rows": int(mask.sum()), "ratio": float(mask.mean())})
    gap_counts = pd.Series([gap_map.get(int(uid), np.nan) for uid in rally_uids]).dropna().astype(int).value_counts().sort_index()
    for gap, count in gap_counts.items():
        coverage_rows.append({"subset": f"gap_{int(gap)}", "rows": int(count), "ratio": float(count / len(rally_uids))})
    pd.DataFrame(coverage_rows).to_csv(args.coverage_report, index=False)

    sources = {
        "base": base["serverGetPoint"].to_numpy(dtype=float),
        "teacher": teacher["serverGetPoint"].to_numpy(dtype=float),
        "soft": soft["serverGetPoint"].to_numpy(dtype=float),
        "direct_diagnostic": direct["serverGetPoint"].to_numpy(dtype=float),
    }
    rows: list[dict] = []
    for source_name, source_server in sources.items():
        for mode in ["all_valid", "new_only_valid"]:
            if mode == "all_valid":
                apply_mask = score_valid
            else:
                apply_mask = score_valid & new_only
            for sw in args.score_weights:
                server = source_server.copy()
                server[apply_mask] = (1.0 - sw) * source_server[apply_mask] + sw * score_rate[apply_mask]
                server = np.clip(server, 1e-6, 1.0 - 1e-6)
                path = Path(
                    f"submission_r29c_{source_name}_{mode}_sw{str(sw).replace('.', 'p')}.csv"
                )
                write_submission(base, server, path)
                rows.append(
                    {
                        "source": source_name,
                        "mode": mode,
                        "score_weight": float(sw),
                        "applied_rows": int(apply_mask.sum()),
                        "applied_old_covered": int((apply_mask & old_covered).sum()),
                        "applied_new_only": int((apply_mask & new_only).sum()),
                        "corr_with_base": float(np.corrcoef(sources["base"], server)[0, 1]),
                        "mad_vs_base": float(np.mean(np.abs(sources["base"] - server))),
                        "matched_mean": float(server[old_covered].mean()),
                        "new_only_mean": float(server[new_only].mean()),
                        "submission": str(path),
                    }
                )
    report = pd.DataFrame(rows)
    report.to_csv(args.report, index=False)

    conservative_all = (
        report[(report["source"].eq("teacher")) & (report["mode"].eq("all_valid"))]
        .sort_values(["mad_vs_base", "score_weight"])
        .iloc[0]
        .to_dict()
    )
    conservative_new = (
        report[(report["source"].eq("teacher")) & (report["mode"].eq("new_only_valid"))]
        .sort_values(["mad_vs_base", "score_weight"])
        .iloc[0]
        .to_dict()
    )
    strongest = (
        report[(report["source"].eq("direct_diagnostic")) & (report["mode"].eq("all_valid"))]
        .sort_values(["score_weight"], ascending=False)
        .iloc[0]
        .to_dict()
    )
    selected = {
        "coverage": {
            "score_valid_rows": int(score_valid.sum()),
            "score_valid_ratio": float(score_valid.mean()),
            "score_valid_new_only_rows": int((score_valid & new_only).sum()),
            "score_valid_old_covered_rows": int((score_valid & old_covered).sum()),
        },
        "conservative_all_valid": conservative_all,
        "conservative_new_only": conservative_new,
        "strongest": strongest,
        "notes": [
            "Scoreboard interval posterior is aggregate when future_gap > 1.",
            "Direct-source variants are high sensitivity.",
        ],
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    feature_report = {
        "args": vars(args),
        "rallies": int(len(rally_uids)),
        "old_covered_rows": int(old_covered.sum()),
        "new_only_rows": int(new_only.sum()),
        "score_valid_rows": int(score_valid.sum()),
        "score_valid_new_only_rows": int((score_valid & new_only).sum()),
        "selected": selected,
    }
    Path(args.feature_report).write_text(json.dumps(feature_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_recommendation(Path(args.recommendation), selected)

    print("coverage")
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print("top conservative")
    print(report.sort_values("mad_vs_base").head(10).to_string(index=False))
    print(f"wrote {args.report}, {args.coverage_report}, {args.selected}, {args.recommendation}")


if __name__ == "__main__":
    main()
