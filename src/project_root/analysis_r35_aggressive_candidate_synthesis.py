"""R35 aggressive candidate synthesis.

Creates high-risk / rule-dependent candidates by combining the improved R33
action branch with stronger existing server and point0 post-processing outputs.

No model is trained here.  Files with `direct` in their source remain
diagnostic-only unless the organizer explicitly permits that usage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aggressive candidate submissions.")
    parser.add_argument("--r33-safe", default="submission_r33_safe_point.csv")
    parser.add_argument("--out-dir", default="r35_aggressive_candidates")
    return parser.parse_args()


def read_sub(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("rally_uid").reset_index(drop=True)
    required = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return df


def assert_aligned(base: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if not base["rally_uid"].equals(other["rally_uid"]):
        raise ValueError(f"{name} rally_uid mismatch")


def synthesize(base_action: pd.DataFrame, point_src: pd.DataFrame, server_src: pd.DataFrame, out_path: Path) -> dict:
    assert_aligned(base_action, point_src, "point_src")
    assert_aligned(base_action, server_src, "server_src")
    out = pd.DataFrame(
        {
            "rally_uid": base_action["rally_uid"].astype(int),
            "actionId": base_action["actionId"].astype(int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float).clip(1e-6, 1.0 - 1e-6).round(8),
        }
    )
    out.to_csv(out_path, index=False, float_format="%.8f")
    return {
        "file": str(out_path),
        "rows": int(len(out)),
        "point_churn_vs_base": float((out["pointId"] != base_action["pointId"]).mean()),
        "server_mean": float(out["serverGetPoint"].mean()),
    }


def maybe_read(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return read_sub(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = out_dir / "diagnostic_high_sensitivity"
    diagnostic_dir.mkdir(exist_ok=True)

    base = read_sub(args.r33_safe)

    point_sources = {
        "v3point": base,
        "r29teacher_b0p25": maybe_read("submission_r29_teacher_beta0p25.csv"),
        "r29teacher_b1p5": maybe_read("submission_r29_teacher_beta1p5.csv"),
        "r29teacher_b2p0": maybe_read("submission_r29_teacher_beta2p0.csv"),
    }
    server_sources = {
        "r29c_teacher_all_sw0p35": maybe_read("submission_r29c_teacher_all_valid_sw0p35.csv"),
        "r29c_teacher_all_sw0p5": maybe_read("submission_r29c_teacher_all_valid_sw0p5.csv"),
        "r29c_teacher_all_sw0p75": maybe_read("submission_r29c_teacher_all_valid_sw0p75.csv"),
        "r29c_teacher_all_sw1p0": maybe_read("submission_r29c_teacher_all_valid_sw1p0.csv"),
        "r29c_teacher_newonly_sw0p35": maybe_read("submission_r29c_teacher_new_only_valid_sw0p35.csv"),
        "r29c_teacher_newonly_sw0p5": maybe_read("submission_r29c_teacher_new_only_valid_sw0p5.csv"),
        "r28_direct_oldserver": maybe_read("submission_r28_old_server_direct_diagnostic.csv"),
        "r29c_direct_all_sw1p0": maybe_read("submission_r29c_direct_diagnostic_all_valid_sw1p0.csv"),
    }

    rows: list[dict] = []
    regular_pairs = [
        ("v3point", "r29c_teacher_all_sw0p35"),
        ("v3point", "r29c_teacher_all_sw0p5"),
        ("v3point", "r29c_teacher_newonly_sw0p35"),
        ("r29teacher_b0p25", "r29c_teacher_all_sw0p35"),
        ("r29teacher_b1p5", "r29c_teacher_all_sw0p5"),
    ]
    diagnostic_pairs = [
        ("v3point", "r28_direct_oldserver"),
        ("v3point", "r29c_direct_all_sw1p0"),
        ("r29teacher_b1p5", "r29c_direct_all_sw1p0"),
    ]

    for point_name, server_name in regular_pairs:
        point_src = point_sources.get(point_name)
        server_src = server_sources.get(server_name)
        if point_src is None or server_src is None:
            continue
        path = out_dir / f"submission_r35_r33action_{point_name}_{server_name}.csv"
        row = synthesize(base, point_src, server_src, path)
        row.update({"point_source": point_name, "server_source": server_name, "risk": "rule_dependent_scoreboard"})
        rows.append(row)

    for point_name, server_name in diagnostic_pairs:
        point_src = point_sources.get(point_name)
        server_src = server_sources.get(server_name)
        if point_src is None or server_src is None:
            continue
        path = diagnostic_dir / f"submission_r35_DIAGNOSTIC_r33action_{point_name}_{server_name}.csv"
        row = synthesize(base, point_src, server_src, path)
        row.update({"point_source": point_name, "server_source": server_name, "risk": "diagnostic_high_sensitivity"})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "r35_summary.csv", index=False)
    (out_dir / "r35_recommendation.md").write_text(
        "\n".join(
            [
                "# R35 Aggressive Candidates",
                "",
                "These files combine R33 action with stronger scoreboard/server/point0 variants.",
                "Use only after organizer clarification. Diagnostic files are not recommended for normal submission.",
                "",
                summary.to_csv(index=False),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"wrote {out_dir / 'r35_summary.csv'}")


if __name__ == "__main__":
    main()
