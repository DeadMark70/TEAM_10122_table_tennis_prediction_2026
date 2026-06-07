"""R34 column-level candidate synthesis.

Combines the safest improved action branch from R33 with already-generated
server/point candidates from R28/R29/R29C.  This does not train anything and
does not change diagnostic high-sensitivity files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize final candidate CSVs from existing submissions.")
    parser.add_argument("--r33-safe", default="submission_r33_safe_point.csv")
    parser.add_argument("--r33-selected", default="submission_r33_oof_selected.csv")
    parser.add_argument("--r29-teacher", default="submission_r29_teacher_beta0p25.csv")
    parser.add_argument("--r28-teacher", default="submission_r28_teacher_ow0p25_bw0p5.csv")
    parser.add_argument("--r29c-newonly", default="submission_r29c_teacher_new_only_valid_sw0p1.csv")
    parser.add_argument("--r29c-allvalid", default="submission_r29c_teacher_all_valid_sw0p1.csv")
    parser.add_argument("--out-dir", default="r34_candidate_synthesis")
    return parser.parse_args()


def read_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"rally_uid", "actionId", "pointId", "serverGetPoint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("rally_uid").reset_index(drop=True)


def assert_aligned(base: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if not base["rally_uid"].equals(other["rally_uid"]):
        raise ValueError(f"{name} rally_uid order mismatch.")


def write_mix(
    base_action: pd.DataFrame,
    point_source: pd.DataFrame,
    server_source: pd.DataFrame,
    output: Path,
    description: str,
) -> dict:
    assert_aligned(base_action, point_source, "point_source")
    assert_aligned(base_action, server_source, "server_source")
    out = pd.DataFrame(
        {
            "rally_uid": base_action["rally_uid"].astype(int),
            "actionId": base_action["actionId"].astype(int),
            "pointId": point_source["pointId"].astype(int),
            "serverGetPoint": server_source["serverGetPoint"].astype(float).clip(1e-6, 1.0 - 1e-6).round(8),
        }
    )
    out.to_csv(output, index=False, float_format="%.8f")
    return {
        "file": str(output),
        "description": description,
        "rows": int(len(out)),
        "action_diff_vs_r33_safe": float((out["actionId"] != base_action["actionId"]).mean()),
        "point_diff_vs_r33_safe": float((out["pointId"] != base_action["pointId"]).mean()),
        "server_mean": float(out["serverGetPoint"].mean()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r33_safe = read_submission(args.r33_safe)
    r33_selected = read_submission(args.r33_selected)
    r29_teacher = read_submission(args.r29_teacher)
    r28_teacher = read_submission(args.r28_teacher)
    r29c_newonly = read_submission(args.r29c_newonly)
    r29c_allvalid = read_submission(args.r29c_allvalid)

    for name, df in [
        ("r33_selected", r33_selected),
        ("r29_teacher", r29_teacher),
        ("r28_teacher", r28_teacher),
        ("r29c_newonly", r29c_newonly),
        ("r29c_allvalid", r29c_allvalid),
    ]:
        assert_aligned(r33_safe, df, name)

    rows = []
    rows.append(
        write_mix(
            r33_safe,
            r33_safe,
            r28_teacher,
            out_dir / "submission_r34_r33action_v3point_r28server.csv",
            "R33 action, V3 point, R28 teacher server.",
        )
    )
    rows.append(
        write_mix(
            r33_safe,
            r29_teacher,
            r29_teacher,
            out_dir / "submission_r34_r33action_r29point_r29server.csv",
            "R33 action, R29 teacher point0, R29 teacher server.",
        )
    )
    rows.append(
        write_mix(
            r33_safe,
            r33_safe,
            r29c_newonly,
            out_dir / "submission_r34_r33action_v3point_r29c_newonly_server.csv",
            "R33 action, V3 point, R29C new-only scoreboard teacher server.",
        )
    )
    rows.append(
        write_mix(
            r33_safe,
            r33_safe,
            r29c_allvalid,
            out_dir / "submission_r34_r33action_v3point_r29c_allvalid_server.csv",
            "R33 action, V3 point, R29C all-valid scoreboard teacher server.",
        )
    )
    rows.append(
        write_mix(
            r33_selected,
            r33_selected,
            r28_teacher,
            out_dir / "submission_r34_r33selected_r28server.csv",
            "R33 selected action/point, R28 teacher server.",
        )
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "r34_summary.csv", index=False)
    (out_dir / "r34_recommendation.md").write_text(
        "\n".join(
            [
                "# R34 Candidate Synthesis",
                "",
                "These are column-level candidates built from existing outputs.",
                "Use only after the organizer clarifies old-test/server/scoreboard boundaries.",
                "",
                summary.to_csv(index=False),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"wrote {out_dir / 'r34_summary.csv'}")


if __name__ == "__main__":
    main()
