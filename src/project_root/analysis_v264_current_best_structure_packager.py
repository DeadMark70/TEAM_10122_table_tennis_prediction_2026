from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v264_current_best_structure_packager"
UPLOAD_DIR = ROOT / "upload_candidates_20260519"

CLEAN_SOURCE = ROOT / "v261_action_conditioned_point_residual" / "submission_v261_cap0p01__v173action_r121server.csv"
V263_SERVER_SOURCE = ROOT / "v263_questionnaire_baseline" / "submission_v263c_server_w0p02__v173_v261cap1.csv"
OLDSHARPEN_SOURCE = ROOT / "v249_current_anchor_server_diagnostic" / "submission_v249_v173_v188cap5_oldsharpen005095.csv"

OPTIONAL_SERVER_SOURCES = {
    "oldrankpreserve005095": [
        ROOT / "r141_r143_server_system" / "submission_r142_r67_anchor_oldrankpreserve005095.csv",
        ROOT / "r177_generalization_finalizer" / "submission_r177_private_rank_r67_r119_oldrankpreserve005095.csv",
        ROOT / "upload_candidates_20260519" / "submission_r142_r67_anchor_oldrankpreserve005095.csv",
    ],
    "oldhard": [
        ROOT / "r141_r143_server_system" / "submission_r142_r67_anchor_oldhard.csv",
        ROOT / "r177_generalization_finalizer" / "submission_r177_public_max_r67_r119_oldhard.csv",
        ROOT / "upload_candidates_20260519" / "submission_r142_r67_anchor_oldhard.csv",
    ],
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    tier: str
    action_point_source: str
    server_source: str
    warning: str


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns are {list(df.columns)}, expected {EXPECTED_COLUMNS}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{path} has {len(df)} rows, expected {EXPECTED_ROWS}")
    if df["rally_uid"].duplicated().any():
        raise ValueError(f"{path} contains duplicate rally_uid values")
    return df.copy()


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame, path: Path) -> None:
    if not reference["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError(f"{path} rally_uid order does not align with clean anchor")


def compose_with_server(clean: pd.DataFrame, server_df: pd.DataFrame) -> pd.DataFrame:
    out = clean.copy()
    out["serverGetPoint"] = server_df["serverGetPoint"].to_numpy()
    return out[EXPECTED_COLUMNS]


def metric_row(
    name: str,
    tier: str,
    source: str,
    clean: pd.DataFrame,
    candidate: pd.DataFrame,
    warning: str,
) -> dict[str, object]:
    action_churn = float(np.mean(candidate["actionId"].to_numpy() != clean["actionId"].to_numpy()))
    point_churn = float(np.mean(candidate["pointId"].to_numpy() != clean["pointId"].to_numpy()))
    clean_server = clean["serverGetPoint"].astype(float).to_numpy()
    cand_server = candidate["serverGetPoint"].astype(float).to_numpy()
    server_mad = float(np.mean(np.abs(cand_server - clean_server)))
    if np.std(clean_server) == 0 or np.std(cand_server) == 0:
        server_corr = float("nan")
    else:
        server_corr = float(np.corrcoef(clean_server, cand_server)[0, 1])
    return {
        "candidate": name,
        "tier": tier,
        "source": source,
        "rows": len(candidate),
        "action_churn_vs_clean": action_churn,
        "point_churn_vs_clean": point_churn,
        "server_mad_vs_clean": server_mad,
        "server_corr_vs_clean": server_corr,
        "warning": warning,
    }


def write_submission(path: Path, df: pd.DataFrame) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path.name} would have invalid columns {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{path.name} would have invalid row count {len(df)}")
    df.to_csv(path, index=False)
    if UPLOAD_DIR.exists():
        shutil.copy2(path, UPLOAD_DIR / path.name)


def first_aligned_optional_source(clean: pd.DataFrame, label: str) -> tuple[Path, pd.DataFrame] | None:
    for path in OPTIONAL_SERVER_SOURCES[label]:
        if not path.exists():
            continue
        try:
            df = read_submission(path)
            assert_aligned(clean, df, path)
        except Exception as exc:
            print(f"skip optional {label} source {path}: {exc}")
            continue
        return path, df
    return None


def write_report_md(report_df: pd.DataFrame, output_paths: list[Path]) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        rows = []
        for _, row in df.iterrows():
            rows.append([str(row[col]) for col in headers])
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    lines = [
        "# V264 Current Best Structure Packager",
        "",
        "V264 freezes the current clean action/point anchor and only swaps server components.",
        "",
        "Clean candidates remain no-old/private-first. Old-server variants are diagnostic/high-risk and should not be treated as clean private-safe improvements.",
        "",
        "## Outputs",
        "",
    ]
    for path in output_paths:
        lines.append(f"- `{path.name}`")
    lines.extend(["", "## Metrics", ""])
    cols = [
        "candidate",
        "tier",
        "rows",
        "action_churn_vs_clean",
        "point_churn_vs_clean",
        "server_mad_vs_clean",
        "server_corr_vs_clean",
    ]
    lines.append(markdown_table(report_df[cols]))
    lines.extend(["", "## Warnings", ""])
    for _, row in report_df.iterrows():
        if row["warning"]:
            lines.append(f"- `{row['candidate']}`: {row['warning']}")
    (OUT_DIR / "v264_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = read_submission(CLEAN_SOURCE)
    v263 = read_submission(V263_SERVER_SOURCE)
    oldsharpen = read_submission(OLDSHARPEN_SOURCE)
    assert_aligned(clean, v263, V263_SERVER_SOURCE)
    assert_aligned(clean, oldsharpen, OLDSHARPEN_SOURCE)

    outputs: list[tuple[Path, pd.DataFrame, CandidateSpec]] = []

    outputs.append(
        (
            OUT_DIR / "submission_v264_clean_v261cap1_r121.csv",
            clean.copy(),
            CandidateSpec(
                "submission_v264_clean_v261cap1_r121.csv",
                "clean_no_old_anchor",
                str(CLEAN_SOURCE.relative_to(ROOT)),
                str(CLEAN_SOURCE.relative_to(ROOT)),
                "",
            ),
        )
    )

    outputs.append(
        (
            OUT_DIR / "submission_v264_clean_v261cap1_v263server_w0p02.csv",
            compose_with_server(clean, v263),
            CandidateSpec(
                "submission_v264_clean_v261cap1_v263server_w0p02.csv",
                "clean_no_old_server_microblend",
                str(CLEAN_SOURCE.relative_to(ROOT)),
                str(V263_SERVER_SOURCE.relative_to(ROOT)),
                "",
            ),
        )
    )

    outputs.append(
        (
            OUT_DIR / "submission_v264_structure_v261cap1_oldsharpen005095.csv",
            compose_with_server(clean, oldsharpen),
            CandidateSpec(
                "submission_v264_structure_v261cap1_oldsharpen005095.csv",
                "diagnostic_old_server_high_risk",
                str(CLEAN_SOURCE.relative_to(ROOT)),
                str(OLDSHARPEN_SOURCE.relative_to(ROOT)),
                "Old-server structure diagnostic; not a clean private-safe candidate.",
            ),
        )
    )

    for label, output_name in [
        ("oldrankpreserve005095", "submission_v264_structure_v261cap1_oldrankpreserve005095.csv"),
        ("oldhard", "submission_v264_structure_v261cap1_oldhard.csv"),
    ]:
        optional = first_aligned_optional_source(clean, label)
        if optional is None:
            print(f"no aligned optional source found for {label}; skipping")
            continue
        source_path, source_df = optional
        outputs.append(
            (
                OUT_DIR / output_name,
                compose_with_server(clean, source_df),
                CandidateSpec(
                    output_name,
                    "diagnostic_old_server_high_risk",
                    str(CLEAN_SOURCE.relative_to(ROOT)),
                    str(source_path.relative_to(ROOT)),
                    "Old-server structure diagnostic; not a clean private-safe candidate.",
                ),
            )
        )

    report_rows = []
    output_paths = []
    for path, df, spec in outputs:
        write_submission(path, df)
        output_paths.append(path)
        report_rows.append(
            {
                **metric_row(
                    name=spec.name,
                    tier=spec.tier,
                    source=spec.server_source,
                    clean=clean,
                    candidate=df,
                    warning=spec.warning,
                ),
                "action_point_source": spec.action_point_source,
                "server_source": spec.server_source,
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUT_DIR / "v264_packaging_report.csv", index=False)
    write_report_md(report_df, output_paths)

    print(f"wrote {len(output_paths)} submissions to {OUT_DIR}")
    if UPLOAD_DIR.exists():
        print(f"copied submissions to {UPLOAD_DIR}")


if __name__ == "__main__":
    main()
