"""V275 public-like validation lab for the clean V271-V275 line.

This script does not train models and does not create submissions.  It reads
the current clean anchor plus available clean candidate search tables, computes
test-like sanity metrics and slice concentration diagnostics, and writes a
small historical-public reference report.

No TTMATCH or old-server inputs are read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTDIR = Path("v275_public_like_validation_lab")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
TRAIN_PATH = Path("train.csv")
TEST_PATH = Path("test_new.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]

SEARCH_TABLES = [
    Path("v267_macro_f1_action_teacher/v267_action_search.csv"),
    Path("v268_macro_f1_point_residual/v268_point_search.csv"),
    Path("v269_clean_server_value_ranker/v269_server_search.csv"),
    Path("v270_clean_candidate_packager/v270_package_search.csv"),
    Path("v271_server_microblend_probe/v271_server_probe_search.csv"),
    Path("v272_action_conditioned_point_residual/v272_point_search.csv"),
    Path("v273_player_conditional_action_response/v273_action_search.csv"),
]

PUBLIC_HISTORY = [
    {"name": "clean_anchor_v261_cap1", "pl": 0.3576720, "label": "positive_anchor"},
    {"name": "v261_cap2", "pl": 0.3573505, "label": "point_cap_too_high"},
    {"name": "v220_weak_action", "pl": 0.3542440, "label": "action_micro_public_negative"},
    {"name": "v191_v166_action", "pl": 0.3509562, "label": "action_full_replace_negative"},
    {"name": "v249_oldserver", "pl": 0.4257757, "label": "structure_diagnostic_not_clean"},
]


def validate_submission_frame(df: pd.DataFrame, expected_rows: int = 1845) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"bad columns: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"bad rows: {len(df)}")


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 3:
        return 1
    if 4 <= point <= 6:
        return 2
    if 7 <= point <= 9:
        return 3
    return 0


def action_family(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return 0
    if 1 <= action <= 7:
        return 1
    if 8 <= action <= 11:
        return 2
    if 12 <= action <= 14:
        return 3
    if 15 <= action <= 18:
        return 4
    return 0


def prefix_bin(prefix_len: int) -> int:
    val = int(prefix_len)
    if val <= 1:
        return 1
    if val == 2:
        return 2
    if val == 3:
        return 3
    if 4 <= val <= 6:
        return 4
    return 5


def safe_mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def no_ttmatch_path_guard(paths: list[str]) -> None:
    bad = [p for p in paths if "TTMATCH" in str(p).upper()]
    if bad:
        raise ValueError(f"TTMATCH is banned from clean branch: {bad}")


def prefix_bin_label(value: int | float) -> str:
    bin_id = prefix_bin(int(value))
    return {1: "1", 2: "2", 3: "3", 4: "4-6", 5: "7+"}[bin_id]


def phase_label(prefix_len: int | float) -> str:
    val = int(prefix_len)
    if val == 1:
        return "receive"
    if val == 2:
        return "third_ball"
    if val == 3:
        return "fourth_ball"
    return "rally"


def distribution(values: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=int))
    freq = series.value_counts(normalize=True, dropna=False)
    return np.array([float(freq.get(int(cls), 0.0)) for cls in classes], dtype=float)


def tv_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)).sum())


def js_distance(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(np.clip(a[mask] / np.clip(b[mask], 1e-12, None), 1e-12, None))))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def load_submission(path: Path) -> pd.DataFrame:
    no_ttmatch_path_guard([str(path)])
    df = pd.read_csv(path)
    validate_submission_frame(df)
    return df


def load_raw_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    no_ttmatch_path_guard([str(TRAIN_PATH), str(TEST_PATH)])
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    overlap = set(train["rally_uid"].unique()) & set(test["rally_uid"].unique())
    if overlap:
        raise ValueError(f"train/test rally_uid overlap detected: {len(overlap)}")
    return train, test


def build_test_context(test: pd.DataFrame) -> pd.DataFrame:
    required = ["rally_uid", "strikeNumber", "actionId", "pointId"]
    missing = [c for c in required if c not in test.columns]
    if missing:
        raise ValueError(f"test_new missing columns: {missing}")
    last = test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).tail(1).copy()
    last = last.sort_values("rally_uid").reset_index(drop=True)
    last["prefix_len"] = pd.to_numeric(last["strikeNumber"], errors="coerce").fillna(0).astype(int)
    last["prefix_bin"] = last["prefix_len"].map(prefix_bin_label)
    last["phase"] = last["prefix_len"].map(phase_label)
    last["lag0_action_family"] = last["actionId"].map(action_family).astype(int)
    last["lag0_point_depth"] = last["pointId"].map(point_depth).astype(int)
    last["lag0_attack_flag"] = last["lag0_action_family"].eq(1).astype(int)
    last["lag0_long_flag"] = last["lag0_point_depth"].eq(3).astype(int)
    cols = [
        "rally_uid",
        "prefix_len",
        "prefix_bin",
        "phase",
        "lag0_action_family",
        "lag0_point_depth",
        "lag0_attack_flag",
        "lag0_long_flag",
    ]
    return last[cols]


def search_table_records() -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    missing_tables: list[str] = []
    no_ttmatch_path_guard([str(p) for p in SEARCH_TABLES])
    for table_path in SEARCH_TABLES:
        if not table_path.exists():
            missing_tables.append(str(table_path))
            continue
        table = pd.read_csv(table_path)
        if "candidate" not in table.columns:
            continue
        for _, row in table.iterrows():
            candidate = str(row.get("candidate", "")).strip()
            raw_path = str(row.get("path", "")).strip()
            if not raw_path:
                continue
            no_ttmatch_path_guard([raw_path])
            records.append(
                {
                    "candidate": candidate or Path(raw_path).name,
                    "path": raw_path,
                    "source_table": str(table_path),
                    "source_verdict": str(row.get("verdict", "")),
                    "search_metrics": json.dumps(
                        {k: v for k, v in row.to_dict().items() if k not in {"candidate", "path"}},
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
    return records, missing_tables


def candidate_inputs(anchor_path: Path) -> tuple[list[dict[str, object]], list[str], list[str]]:
    records, missing_tables = search_table_records()
    records.insert(
        0,
        {
            "candidate": "clean_anchor_v261_cap1",
            "path": str(anchor_path),
            "source_table": "",
            "source_verdict": "ANCHOR",
            "search_metrics": "{}",
        },
    )
    seen: set[str] = set()
    existing: list[dict[str, object]] = []
    missing_paths: list[str] = []
    for rec in records:
        path = Path(str(rec["path"]))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            existing.append(rec)
        else:
            missing_paths.append(key)
    return existing, missing_tables, missing_paths


def decision_label(rec: dict[str, object]) -> str:
    source_verdict = str(rec.get("source_verdict", "")).upper()
    candidate = str(rec.get("candidate", "")).lower()
    path = str(rec.get("path", "")).lower()
    action_churn = float(rec.get("action_churn", 0.0))
    point_churn = float(rec.get("point_churn", 0.0))
    server_mad = float(rec.get("server_mad", 0.0))
    point0_rate = float(rec.get("point0_rate", 0.0))
    point0_added_rows = int(rec.get("point0_added_rows", 0))
    serve_added_rows = int(rec.get("serve_added_rows", 0))

    if point0_rate < 0.24 or point0_rate > 0.31 or point0_added_rows > 8:
        return "REJECT_POINT0"
    if serve_added_rows > 2:
        return "REJECT_SERVE_EXPLOSION"
    if action_churn > 0.12 or point_churn > 0.05 or server_mad > 0.01:
        return "REJECT_CHURN"
    if "LOCAL_NEGATIVE" in source_verdict or "DO_NOT_SUBMIT" in source_verdict:
        return "REJECT_HISTORY_PATTERN"
    if "DIAGNOSTIC" in source_verdict or "diagnostic" in candidate or "diagnostic" in path:
        return "DIAGNOSTIC_ONLY"
    return "KEEP_CLEAN_REVIEW"


def evaluate_candidate(
    rec: dict[str, object],
    anchor: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = Path(str(rec["path"]))
    cand = load_submission(path)
    aligned = context[["rally_uid"]].merge(cand, on="rally_uid", how="left", validate="one_to_one")
    if aligned[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError(f"submission did not align with test context: {path}")
    anchor_aligned = context[["rally_uid"]].merge(anchor, on="rally_uid", how="left", validate="one_to_one")

    action_anchor = anchor_aligned["actionId"].astype(int).to_numpy()
    action_cand = aligned["actionId"].astype(int).to_numpy()
    point_anchor = anchor_aligned["pointId"].astype(int).to_numpy()
    point_cand = aligned["pointId"].astype(int).to_numpy()
    server_anchor = anchor_aligned["serverGetPoint"].astype(float).to_numpy()
    server_cand = aligned["serverGetPoint"].astype(float).to_numpy()

    action_diff = action_anchor != action_cand
    point_diff = point_anchor != point_cand
    server_delta = server_cand - server_anchor
    server_diff = np.abs(server_delta) > 1e-12
    changed = action_diff | point_diff | server_diff

    anchor_family_dist = distribution(np.array([action_family(v) for v in action_anchor]), range(5))
    cand_family_dist = distribution(np.array([action_family(v) for v in action_cand]), range(5))
    point0_added = (point_anchor != 0) & (point_cand == 0)
    anchor_serve = np.isin(action_anchor, [15, 16, 17, 18])
    cand_serve = np.isin(action_cand, [15, 16, 17, 18])
    changed_rows = int(changed.sum())

    out = {
        **rec,
        "rows": int(len(aligned)),
        "action_churn": float(action_diff.mean()),
        "point_churn": float(point_diff.mean()),
        "server_mad": safe_mad(server_cand, server_anchor),
        "server_max_abs_delta": float(np.max(np.abs(server_delta))) if len(server_delta) else 0.0,
        "family_tv_vs_anchor": tv_distance(anchor_family_dist, cand_family_dist),
        "family_js_vs_anchor": js_distance(anchor_family_dist, cand_family_dist),
        "point0_rate": float(np.mean(point_cand == 0)),
        "point0_added_rows": int(point0_added.sum()),
        "serve_15_18_count": int(cand_serve.sum()),
        "serve_added_rows": int((~anchor_serve & cand_serve).sum()),
        "changed_rows": changed_rows,
        "changed_rate": float(changed.mean()),
        "candidate_action_distribution": json.dumps(pd.Series(action_cand).value_counts().sort_index().to_dict(), sort_keys=True),
        "candidate_point_distribution": json.dumps(pd.Series(point_cand).value_counts().sort_index().to_dict(), sort_keys=True),
    }

    slice_rows = []
    slice_specs = [
        ("prefix_bin", "prefix_bin"),
        ("phase", "phase"),
        ("lag0_action_family", "lag0_action_family"),
        ("lag0_point_depth", "lag0_point_depth"),
        ("lag0_attack_flag", "lag0_attack_flag"),
        ("lag0_long_flag", "lag0_long_flag"),
    ]
    joined = context.reset_index(drop=True)
    max_share = 0.0
    max_slice = ""
    for feature, label in slice_specs:
        for value, idx in joined.groupby(feature, sort=True).groups.items():
            mask = np.zeros(len(joined), dtype=bool)
            mask[list(idx)] = True
            slice_changed = int(changed[mask].sum())
            share = float(slice_changed / changed_rows) if changed_rows else 0.0
            name = f"{label}={value}"
            if share > max_share:
                max_share = share
                max_slice = name
            slice_rows.append(
                {
                    "candidate": str(rec["candidate"]),
                    "slice": name,
                    "slice_feature": label,
                    "slice_value": str(value),
                    "rows": int(mask.sum()),
                    "row_share": float(mask.mean()),
                    "action_churn": float(action_diff[mask].mean()),
                    "point_churn": float(point_diff[mask].mean()),
                    "server_mad": safe_mad(server_cand[mask], server_anchor[mask]),
                    "changed_rows": slice_changed,
                    "changed_share_of_candidate_changes": share,
                    "point0_rate": float(np.mean(point_cand[mask] == 0)),
                    "serve_15_18_count": int(cand_serve[mask].sum()),
                }
            )
    out["max_changed_slice_share"] = max_share
    out["max_changed_slice"] = max_slice
    out["decision"] = decision_label(out)
    return out, slice_rows


def write_public_history(path: Path) -> None:
    rows = sorted(PUBLIC_HISTORY, key=lambda item: float(item["pl"]), reverse=True)
    lines = [
        "# V275 Historical Public Reference",
        "",
        "| name | public_lb | label | clean_use |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        clean_use = "diagnostic_only" if "oldserver" in row["name"] else "clean_reference"
        lines.append(f"| {row['name']} | {float(row['pl']):.7f} | {row['label']} | {clean_use} |")
    lines.extend(
        [
            "",
            "The old-server result is kept only as a structure diagnostic reference and is not a clean-branch input.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    candidate_sanity: pd.DataFrame,
    slice_sanity: pd.DataFrame,
    missing_tables: list[str],
    missing_paths: list[str],
    train_rows: int,
    test_rows: int,
) -> None:
    decision_counts = candidate_sanity["decision"].value_counts().to_dict() if not candidate_sanity.empty else {}
    keep = candidate_sanity[candidate_sanity["decision"].eq("KEEP_CLEAN_REVIEW")].copy() if not candidate_sanity.empty else pd.DataFrame()
    if not keep.empty:
        keep = keep.sort_values(["server_mad", "action_churn", "point_churn"], ascending=[True, True, True])
    lines = [
        "# V275 Public-Like Validation Lab",
        "",
        "No submission is generated by this script.",
        "",
        "## Summary",
        "",
        f"- Train rows read: `{train_rows}`",
        f"- Test rows read: `{test_rows}`",
        f"- Candidates evaluated: `{len(candidate_sanity)}`",
        f"- Slice rows written: `{len(slice_sanity)}`",
        f"- Decision counts: `{json.dumps(decision_counts, sort_keys=True)}`",
        f"- Missing optional search tables: `{len(missing_tables)}`",
        f"- Missing candidate paths from present tables: `{len(missing_paths)}`",
        "",
        "## KEEP_CLEAN_REVIEW candidates",
        "",
    ]
    if keep.empty:
        lines.append("- None")
    else:
        for row in keep.head(12).to_dict("records"):
            lines.append(
                f"- `{row['candidate']}`: action_churn={float(row['action_churn']):.6f}, "
                f"point_churn={float(row['point_churn']):.6f}, server_mad={float(row['server_mad']):.6f}, "
                f"max_slice={row['max_changed_slice']} ({float(row['max_changed_slice_share']):.3f})"
            )
    if missing_tables:
        lines.extend(["", "## Missing optional search tables", ""])
        lines.extend(f"- `{item}`" for item in missing_tables)
    if missing_paths:
        lines.extend(["", "## Missing candidate paths", ""])
        lines.extend(f"- `{item}`" for item in missing_paths[:20])
        if len(missing_paths) > 20:
            lines.append(f"- ... `{len(missing_paths) - 20}` more")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- TTMATCH paths are rejected before reading.",
            "- Old-server and old-test labels are not read.",
            "- Historical public rows are a manual reference table only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train, test = load_raw_inputs()
    context = build_test_context(test)
    anchor = load_submission(ANCHOR_PATH)
    if set(context["rally_uid"].astype(int)) != set(anchor["rally_uid"].astype(int)):
        raise ValueError("anchor rally_uid set does not match test_new rally_uid set")

    records, missing_tables, missing_paths = candidate_inputs(ANCHOR_PATH)
    sanity_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    for rec in records:
        out, slices = evaluate_candidate(rec, anchor, context)
        sanity_rows.append(out)
        slice_rows.extend(slices)

    candidate_sanity = pd.DataFrame(sanity_rows)
    slice_sanity = pd.DataFrame(slice_rows)
    if not candidate_sanity.empty:
        ordered = [
            "candidate",
            "decision",
            "path",
            "source_table",
            "source_verdict",
            "rows",
            "action_churn",
            "point_churn",
            "server_mad",
            "server_max_abs_delta",
            "family_tv_vs_anchor",
            "family_js_vs_anchor",
            "point0_rate",
            "point0_added_rows",
            "serve_15_18_count",
            "serve_added_rows",
            "changed_rows",
            "changed_rate",
            "max_changed_slice",
            "max_changed_slice_share",
        ]
        candidate_sanity = candidate_sanity[ordered + [c for c in candidate_sanity.columns if c not in ordered]]
    candidate_sanity.to_csv(OUTDIR / "v275_candidate_sanity.csv", index=False)
    slice_sanity.to_csv(OUTDIR / "v275_slice_sanity.csv", index=False)
    write_public_history(OUTDIR / "v275_public_history.md")
    write_report(
        OUTDIR / "v275_report.md",
        candidate_sanity,
        slice_sanity,
        missing_tables,
        missing_paths,
        train_rows=len(train),
        test_rows=len(test),
    )
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "candidates": int(len(candidate_sanity)),
                "slice_rows": int(len(slice_sanity)),
                "decision_counts": candidate_sanity["decision"].value_counts().to_dict() if not candidate_sanity.empty else {},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
