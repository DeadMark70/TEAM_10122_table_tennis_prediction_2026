"""R152 TTMATCH usage audit.

This is a read-only planning audit. It does not train models and does not
generate submissions.

Questions answered:
  1. Is TTMATCH schema compatible with AI CUP?
  2. Are ID overlaps likely true sample identity or ID reuse?
  3. How different are action/point/server distributions?
  4. If used only as transition prior, how much AI CUP/test context can it cover?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTERNAL = ROOT / "external_data" / "TTMATCH"
DATA_RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "r152_ttmatch_usage_audit"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"strickNumber": "strikeNumber", "strickId": "strikeId"})


def add_next_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["rally_uid", "strikeNumber"]).copy()
    keys = ["rally_uid"]
    for col in ["actionId", "pointId", "spinId", "handId", "strengthId", "positionId", "strikeId"]:
        df[f"next_{col}"] = df.groupby(keys)[col].shift(-1)
    df["is_last_observed"] = df.groupby(keys)["strikeNumber"].transform("max").eq(df["strikeNumber"])
    return df


def class_dist(df: pd.DataFrame, dataset: str, col: str, classes: list[int]) -> pd.DataFrame:
    vc = df[col].value_counts().reindex(classes, fill_value=0)
    total = float(vc.sum())
    return pd.DataFrame(
        {
            "dataset": dataset,
            "column": col,
            "value": classes,
            "count": vc.to_numpy(dtype=int),
            "rate": vc.to_numpy(dtype=float) / max(total, 1.0),
        }
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def distribution_compare(tt: pd.DataFrame, ai: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "actionId": list(range(19)),
        "pointId": list(range(10)),
        "spinId": sorted(set(tt["spinId"].dropna().astype(int)) | set(ai["spinId"].dropna().astype(int))),
        "strengthId": sorted(set(tt["strengthId"].dropna().astype(int)) | set(ai["strengthId"].dropna().astype(int))),
        "handId": sorted(set(tt["handId"].dropna().astype(int)) | set(ai["handId"].dropna().astype(int))),
        "positionId": sorted(set(tt["positionId"].dropna().astype(int)) | set(ai["positionId"].dropna().astype(int))),
        "serverGetPoint": [0, 1],
    }
    all_rows = []
    summary = []
    for col, classes in specs.items():
        left = class_dist(tt, "ttmatch_train", col, classes)
        right = class_dist(ai, "aicup_train", col, classes)
        merged = left.merge(right, on=["column", "value"], suffixes=("_ttmatch", "_aicup"))
        merged["rate_diff_tt_minus_aicup"] = merged["rate_ttmatch"] - merged["rate_aicup"]
        all_rows.append(merged)
        summary.append(
            {
                "column": col,
                "js_divergence": js_divergence(merged["rate_ttmatch"].to_numpy(), merged["rate_aicup"].to_numpy()),
                "max_abs_rate_diff": float(merged["rate_diff_tt_minus_aicup"].abs().max()),
                "ttmatch_unique_values": int((merged["count_ttmatch"] > 0).sum()),
                "aicup_unique_values": int((merged["count_aicup"] > 0).sum()),
            }
        )
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summary)


def id_collision_examples(tt: pd.DataFrame, ai: pd.DataFrame, right_name: str, n: int = 25) -> pd.DataFrame:
    key = ["rally_uid", "strikeNumber"]
    common = sorted(set(map(tuple, tt[key].itertuples(index=False, name=None))) & set(map(tuple, ai[key].itertuples(index=False, name=None))))
    rows = []
    compare_cols = [
        "sex",
        "match",
        "numberGame",
        "rally_id",
        "scoreSelf",
        "scoreOther",
        "gamePlayerId",
        "gamePlayerOtherId",
        "strikeId",
        "handId",
        "strengthId",
        "spinId",
        "pointId",
        "actionId",
        "positionId",
    ]
    tt_idx = tt.set_index(key)
    ai_idx = ai.set_index(key)
    for uid, sn in common[:n]:
        tr = tt_idx.loc[(uid, sn)]
        ar = ai_idx.loc[(uid, sn)]
        rec: dict[str, Any] = {"right_dataset": right_name, "rally_uid": uid, "strikeNumber": sn}
        same_count = 0
        comparable = 0
        for col in compare_cols:
            if col in tr.index and col in ar.index:
                tv = tr[col]
                av = ar[col]
                rec[f"tt_{col}"] = tv
                rec[f"{right_name}_{col}"] = av
                same = tv == av
                rec[f"same_{col}"] = bool(same)
                same_count += int(same)
                comparable += 1
        rec["same_public_fields"] = same_count
        rec["comparable_public_fields"] = comparable
        rec["same_rate"] = same_count / max(comparable, 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def transition_rows(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    x = add_next_labels(df)
    tr = x[~x["is_last_observed"]].copy()
    tr["dataset"] = dataset
    tr["phase_simple"] = np.select(
        [
            tr["strikeNumber"].eq(1),
            tr["strikeNumber"].eq(2),
            tr["strikeNumber"].eq(3),
        ],
        ["serve", "receive", "third_ball"],
        default="rally",
    )
    return tr


def context_coverage(tt_train: pd.DataFrame, ai_train: pd.DataFrame, ai_test: pd.DataFrame) -> pd.DataFrame:
    tt_tr = transition_rows(tt_train, "ttmatch_train")
    ai_tr = transition_rows(ai_train, "aicup_train")
    # For test_new, every observed row can be a last-prefix context for the unknown next stroke.
    ai_test_last = ai_test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid").tail(1).copy()
    ai_test_last["phase_simple"] = np.select(
        [
            ai_test_last["strikeNumber"].eq(1),
            ai_test_last["strikeNumber"].eq(2),
            ai_test_last["strikeNumber"].eq(3),
        ],
        ["serve", "receive", "third_ball"],
        default="rally",
    )
    contexts = {
        "k1_action_spin_point": ["phase_simple", "actionId", "spinId", "pointId"],
        "k2_action_spin_point_strength": ["phase_simple", "actionId", "spinId", "pointId", "strengthId"],
        "k3_action_point_hand_pos": ["phase_simple", "actionId", "pointId", "handId", "positionId"],
        "k4_action_point_only": ["actionId", "pointId"],
    }
    rows = []
    for name, cols in contexts.items():
        tt_keys = set(map(tuple, tt_tr[cols].astype("string").itertuples(index=False, name=None)))
        ai_keys = set(map(tuple, ai_tr[cols].astype("string").itertuples(index=False, name=None)))
        test_keys = set(map(tuple, ai_test_last[cols].astype("string").itertuples(index=False, name=None)))
        rows.append(
            {
                "context": name,
                "columns": ",".join(cols),
                "ttmatch_unique_contexts": len(tt_keys),
                "aicup_train_unique_contexts": len(ai_keys),
                "test_new_unique_contexts": len(test_keys),
                "aicup_train_contexts_covered_by_ttmatch": len(ai_keys & tt_keys),
                "aicup_train_context_coverage_rate": len(ai_keys & tt_keys) / max(len(ai_keys), 1),
                "test_new_contexts_covered_by_ttmatch": len(test_keys & tt_keys),
                "test_new_context_coverage_rate": len(test_keys & tt_keys) / max(len(test_keys), 1),
            }
        )
    return pd.DataFrame(rows)


def prefix_target_capacity(tt_train: pd.DataFrame, ai_train: pd.DataFrame) -> pd.DataFrame:
    tt_tr = transition_rows(tt_train, "ttmatch_train")
    ai_tr = transition_rows(ai_train, "aicup_train")
    rows = []
    for dataset, tr in [("ttmatch_train", tt_tr), ("aicup_train", ai_tr)]:
        rows.append(
            {
                "dataset": dataset,
                "transition_rows": int(len(tr)),
                "rallies": int(tr["rally_uid"].nunique()),
                "next_action_unique": int(tr["next_actionId"].nunique()),
                "next_point_unique": int(tr["next_pointId"].nunique()),
                "next_action_8_count": int(tr["next_actionId"].eq(8).sum()),
                "next_action_9_count": int(tr["next_actionId"].eq(9).sum()),
                "next_action_12_count": int(tr["next_actionId"].eq(12).sum()),
                "next_action_14_count": int(tr["next_actionId"].eq(14).sum()),
                "next_point_0_count": int(tr["next_pointId"].eq(0).sum()),
                "next_point_8_count": int(tr["next_pointId"].eq(8).sum()),
                "next_point_9_count": int(tr["next_pointId"].eq(9).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_report(meta: dict[str, Any], dist_summary: pd.DataFrame, coverage: pd.DataFrame) -> str:
    lines = [
        "# R152 TTMATCH Usage Audit",
        "",
        "R152 is a planning/audit step only. It does not train models or generate submissions.",
        "",
        "## Safety Position",
        "",
        "- TTMATCH is AI CUP-like tabular data and remains high risk until source/license/provenance are documented.",
        "- Exact public-full fingerprint overlap with current AI CUP train/test is 0 in R150, so the current evidence points more toward ID reuse / same schema than exact duplicate rows.",
        "- Do not use direct lookup or replacement into `test_new.csv`.",
        "- First acceptable experiment should be an isolated, low-weight transition prior or teacher-style augmentation.",
        "",
        "## Distribution Difference Summary",
        "",
        "| column | JS divergence | max abs rate diff |",
        "|---|---:|---:|",
    ]
    for _, row in dist_summary.iterrows():
        lines.append(f"| {row['column']} | {float(row['js_divergence']):.6f} | {float(row['max_abs_rate_diff']):.6f} |")
    lines.extend(
        [
            "",
            "## Context Coverage",
            "",
            "| context | AI CUP train coverage | test_new coverage |",
            "|---|---:|---:|",
        ]
    )
    for _, row in coverage.iterrows():
        lines.append(
            f"| {row['context']} | {float(row['aicup_train_context_coverage_rate']):.3f} | {float(row['test_new_context_coverage_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Use",
            "",
            "1. R153: TTMATCH transition-prior only, no player IDs and no `rally_uid`.",
            "2. Use smoothed priors such as `P(next_action | phase, lag0_action, lag0_spin, lag0_point)` and `P(next_point | phase, lag0_action, lag0_spin, lag0_point)`.",
            "3. Apply as low-weight logit residual to existing predictions, with churn caps.",
            "4. Avoid direct supervised merge until a strict isolation ablation shows public benefit and provenance is report-ready.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    tt_train = normalize_cols(read_csv(EXTERNAL / "train.csv"))
    tt_test = normalize_cols(read_csv(EXTERNAL / "test.csv"))
    ai_train = normalize_cols(read_csv(DATA_RAW / "train.csv"))
    ai_old = normalize_cols(read_csv(DATA_RAW / "test_old.csv"))
    ai_new = normalize_cols(read_csv(DATA_RAW / "test_new.csv"))

    dist, dist_summary = distribution_compare(tt_train, ai_train)
    dist.to_csv(OUTDIR / "r152_distribution_compare.csv", index=False)
    dist_summary.to_csv(OUTDIR / "r152_distribution_summary.csv", index=False)

    examples = pd.concat(
        [
            id_collision_examples(tt_train, ai_train, "aicup_train"),
            id_collision_examples(tt_train, ai_old, "aicup_test_old"),
            id_collision_examples(tt_train, ai_new, "aicup_test_new"),
            id_collision_examples(tt_test, ai_new, "aicup_test_new"),
        ],
        ignore_index=True,
    )
    examples.to_csv(OUTDIR / "r152_id_collision_examples.csv", index=False)

    coverage = context_coverage(tt_train, ai_train, ai_new)
    coverage.to_csv(OUTDIR / "r152_transition_context_coverage.csv", index=False)

    capacity = prefix_target_capacity(tt_train, ai_train)
    capacity.to_csv(OUTDIR / "r152_prefix_target_capacity.csv", index=False)

    meta = {
        "ttmatch_train_rows": int(len(tt_train)),
        "ttmatch_test_rows": int(len(tt_test)),
        "aicup_train_rows": int(len(ai_train)),
        "aicup_test_new_rows": int(len(ai_new)),
        "outputs": {
            "distribution_compare": str(OUTDIR / "r152_distribution_compare.csv"),
            "distribution_summary": str(OUTDIR / "r152_distribution_summary.csv"),
            "id_collision_examples": str(OUTDIR / "r152_id_collision_examples.csv"),
            "transition_context_coverage": str(OUTDIR / "r152_transition_context_coverage.csv"),
            "prefix_target_capacity": str(OUTDIR / "r152_prefix_target_capacity.csv"),
        },
        "decision": "Use TTMATCH only as isolated low-weight transition prior until provenance is verified.",
    }
    (OUTDIR / "r152_ttmatch_usage_audit_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r152_ttmatch_usage_audit_report.md").write_text(make_report(meta, dist_summary, coverage), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
