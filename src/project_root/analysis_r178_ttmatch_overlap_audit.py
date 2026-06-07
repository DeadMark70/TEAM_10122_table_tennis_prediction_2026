from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "r178_ttmatch_overlap_audit"


DATASETS = {
    "aicup_train": ROOT / "train.csv",
    "aicup_test_new": ROOT / "test_new.csv",
    "aicup_test_old": ROOT / "test_old.csv",
    "ttmatch_train": ROOT / "external_data" / "TTMATCH" / "train.csv",
    "ttmatch_test": ROOT / "external_data" / "TTMATCH" / "test.csv",
}

CANONICAL_RENAME = {
    "strickNumber": "strikeNumber",
    "strickId": "strikeId",
}

ID_COLS = {
    "rally_uid",
    "match",
    "rally_id",
    "gamePlayerId",
    "gamePlayerOtherId",
}

ORDER_COLS = ["strikeNumber"]
GROUP_KEYS = [
    ["rally_uid"],
    ["match", "numberGame", "rally_id"],
]

STRICT_STROKE_COLS = [
    "strikeNumber",
    "sex",
    "numberGame",
    "scoreSelf",
    "scoreOther",
    "serverGetPoint",
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

STROKE_CORE_COLS = [
    "strikeNumber",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]

STROKE_NO_STRIKE_COLS = [
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]

SCORE_CONTEXT_COLS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
]

DIST_COLS = [
    "sex",
    "numberGame",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "serverGetPoint",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=CANONICAL_RENAME)
    df["__row_index__"] = np.arange(len(df), dtype=np.int64)
    return df


def columns_present(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def unique_in_order(cols: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def normalize_value(v: object) -> str:
    if pd.isna(v):
        return "<NA>"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if np.isfinite(fv) and abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        return f"{fv:.8g}"
    s = str(v).strip()
    return s


def signature_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    norm = pd.DataFrame(index=df.index)
    for col in cols:
        norm[col] = df[col].map(normalize_value)
    return norm.agg("\x1f".join, axis=1).map(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
    )


def overlap_counts(left: pd.Series, right: pd.Series) -> dict[str, int | float]:
    lc = left.value_counts(dropna=False)
    rc = right.value_counts(dropna=False)
    common = lc.index.intersection(rc.index)
    min_pairs = int(np.minimum(lc.loc[common].to_numpy(), rc.loc[common].to_numpy()).sum())
    left_rows = int(lc.loc[common].sum()) if len(common) else 0
    right_rows = int(rc.loc[common].sum()) if len(common) else 0
    return {
        "overlap_unique_signatures": int(len(common)),
        "overlap_min_pair_count": min_pairs,
        "left_rows_with_overlap_signature": left_rows,
        "right_rows_with_overlap_signature": right_rows,
        "left_overlap_ratio": left_rows / max(len(left), 1),
        "right_overlap_ratio": right_rows / max(len(right), 1),
    }


def make_sequence_signatures(
    df: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.Series:
    cols = columns_present(df, unique_in_order(group_cols + value_cols + ORDER_COLS))
    if not all(c in df.columns for c in group_cols) or not value_cols:
        return pd.Series(dtype="object")
    work = df[cols + ["__row_index__"]].copy()
    sort_cols = group_cols + (ORDER_COLS if all(c in work.columns for c in ORDER_COLS) else [])
    sort_cols.append("__row_index__")
    work = work.sort_values(sort_cols)
    value_cols = columns_present(work, value_cols)

    sigs = []
    keys = []
    for key, group in work.groupby(group_cols, sort=False):
        row_tokens = []
        for _, row in group[value_cols].iterrows():
            row_tokens.append(",".join(normalize_value(row[c]) for c in value_cols))
        raw = "|".join(row_tokens)
        sigs.append(hashlib.sha1(raw.encode("utf-8")).hexdigest())
        if isinstance(key, tuple):
            keys.append("|".join(normalize_value(v) for v in key))
        else:
            keys.append(normalize_value(key))
    return pd.Series(sigs, index=pd.Index(keys, name="group_key"))


def js_divergence(left: pd.Series, right: pd.Series) -> float:
    lv = left.map(normalize_value).value_counts(normalize=True, dropna=False)
    rv = right.map(normalize_value).value_counts(normalize=True, dropna=False)
    cats = lv.index.union(rv.index)
    p = lv.reindex(cats, fill_value=0.0).to_numpy(dtype=float)
    q = rv.reindex(cats, fill_value=0.0).to_numpy(dtype=float)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(left: pd.Series, right: pd.Series) -> float:
    lv = left.map(normalize_value).value_counts(normalize=True, dropna=False)
    rv = right.map(normalize_value).value_counts(normalize=True, dropna=False)
    cats = lv.index.union(rv.index)
    return float(0.5 * np.abs(lv.reindex(cats, fill_value=0.0) - rv.reindex(cats, fill_value=0.0)).sum())


def top_common_values(left: pd.Series, right: pd.Series, n: int = 8) -> str:
    lv = left.map(normalize_value).value_counts(normalize=True, dropna=False)
    rv = right.map(normalize_value).value_counts(normalize=True, dropna=False)
    cats = lv.index.union(rv.index)
    rows = []
    for cat in cats:
        rows.append((cat, float(lv.get(cat, 0.0)), float(rv.get(cat, 0.0))))
    rows.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    return "; ".join(f"{k}:L={a:.3f},R={b:.3f}" for k, a, b in rows[:n])


def csv_block(df: pd.DataFrame, max_rows: int = 20) -> str:
    return "```csv\n" + df.head(max_rows).to_csv(index=False).strip() + "\n```"


def verdict(exact_df: pd.DataFrame, seq_df: pd.DataFrame) -> tuple[str, list[str]]:
    warnings = []
    high = False
    medium = False

    for _, row in exact_df.iterrows():
        if (
            row["ttmatch_dataset"] in {"ttmatch_train", "ttmatch_test"}
            and str(row["aicup_dataset"]).startswith("aicup_test")
            and row["signature_scope"] in {"canonical_non_id_common", "stroke_core"}
            and row["left_overlap_ratio"] > 0.01
        ):
            high = True
            warnings.append(
                f"{row['ttmatch_dataset']} overlaps {row['aicup_dataset']} on {row['signature_scope']} at {row['left_overlap_ratio']:.2%}."
            )
        elif row["signature_scope"] == "canonical_non_id_common" and row["left_overlap_ratio"] > 0.05:
            medium = True
            warnings.append(
                f"{row['ttmatch_dataset']} shares many non-id row signatures with {row['aicup_dataset']} ({row['left_overlap_ratio']:.2%})."
            )

    for _, row in seq_df.iterrows():
        if (
            str(row["aicup_dataset"]).startswith("aicup_test")
            and row["sequence_scope"] in {"stroke_core_sequence", "stroke_no_strike_sequence"}
            and row["left_overlap_ratio"] > 0.01
        ):
            high = True
            warnings.append(
                f"{row['ttmatch_dataset']} shares sequence signatures with {row['aicup_dataset']} ({row['left_overlap_ratio']:.2%})."
            )
        elif row["left_overlap_ratio"] > 0.05:
            medium = True
            warnings.append(
                f"{row['ttmatch_dataset']} sequence overlap with {row['aicup_dataset']} is {row['left_overlap_ratio']:.2%}."
            )

    if high:
        return "HIGH_RISK_DIRECT_OVERLAP_FOUND", warnings
    if medium:
        return "MODERATE_RISK_SCHEMA_OR_DISTRIBUTION_SIMILARITY", warnings
    return "NO_DIRECT_OVERLAP_FOUND_BUT_SCHEMA_IS_AICUP_LIKE", warnings


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    datasets = {name: read_dataset(path) for name, path in DATASETS.items() if path.exists()}

    schema_rows = []
    for name, df in datasets.items():
        schema_rows.append(
            {
                "dataset": name,
                "path": str(DATASETS[name]),
                "rows": len(df),
                "columns": len([c for c in df.columns if c != "__row_index__"]),
                "column_list": "|".join(c for c in df.columns if c != "__row_index__"),
                "rallies_by_rally_uid": df["rally_uid"].nunique() if "rally_uid" in df.columns else np.nan,
                "matches": df["match"].nunique() if "match" in df.columns else np.nan,
            }
        )
    schema_df = pd.DataFrame(schema_rows)

    tt_names = [n for n in datasets if n.startswith("ttmatch")]
    ai_names = [n for n in datasets if n.startswith("aicup")]

    overlap_rows = []
    for tt_name in tt_names:
        tt_df = datasets[tt_name]
        for ai_name in ai_names:
            ai_df = datasets[ai_name]
            common = sorted((set(tt_df.columns) & set(ai_df.columns)) - {"__row_index__"})
            scopes = {
                "canonical_all_common": common,
                "canonical_non_id_common": [c for c in common if c not in ID_COLS],
                "stroke_core": columns_present(tt_df, STROKE_CORE_COLS),
                "stroke_no_strike": columns_present(tt_df, STROKE_NO_STRIKE_COLS),
                "score_context": columns_present(tt_df, SCORE_CONTEXT_COLS),
            }
            for scope, cols in scopes.items():
                cols = [c for c in cols if c in ai_df.columns]
                if not cols:
                    continue
                left_sig = signature_series(tt_df, cols)
                right_sig = signature_series(ai_df, cols)
                counts = overlap_counts(left_sig, right_sig)
                overlap_rows.append(
                    {
                        "ttmatch_dataset": tt_name,
                        "aicup_dataset": ai_name,
                        "signature_scope": scope,
                        "columns": "|".join(cols),
                        **counts,
                    }
                )
    overlap_df = pd.DataFrame(overlap_rows)

    sequence_rows = []
    sequence_scopes = {
        "stroke_core_sequence": STROKE_CORE_COLS,
        "stroke_no_strike_sequence": STROKE_NO_STRIKE_COLS,
    }
    for tt_name in tt_names:
        tt_df = datasets[tt_name]
        for ai_name in ai_names:
            ai_df = datasets[ai_name]
            for group_cols in GROUP_KEYS:
                if not all(c in tt_df.columns for c in group_cols) or not all(c in ai_df.columns for c in group_cols):
                    continue
                for scope, value_cols in sequence_scopes.items():
                    value_cols_common = [c for c in value_cols if c in tt_df.columns and c in ai_df.columns]
                    if not value_cols_common:
                        continue
                    left_sig = make_sequence_signatures(tt_df, group_cols, value_cols_common)
                    right_sig = make_sequence_signatures(ai_df, group_cols, value_cols_common)
                    counts = overlap_counts(left_sig, right_sig)
                    sequence_rows.append(
                        {
                            "ttmatch_dataset": tt_name,
                            "aicup_dataset": ai_name,
                            "group_cols": "|".join(group_cols),
                            "sequence_scope": scope,
                            "columns": "|".join(value_cols_common),
                            "left_groups": int(len(left_sig)),
                            "right_groups": int(len(right_sig)),
                            **counts,
                        }
                    )
    sequence_df = pd.DataFrame(sequence_rows)

    id_rows = []
    id_scopes = {
        "rally_uid": ["rally_uid"],
        "match_numberGame_rally_id": ["match", "numberGame", "rally_id"],
        "match_numberGame_rally_id_players": [
            "match",
            "numberGame",
            "rally_id",
            "gamePlayerId",
            "gamePlayerOtherId",
        ],
    }
    for tt_name in tt_names:
        tt_df = datasets[tt_name]
        for ai_name in ai_names:
            ai_df = datasets[ai_name]
            for scope, cols in id_scopes.items():
                if not all(c in tt_df.columns for c in cols) or not all(c in ai_df.columns for c in cols):
                    continue
                tt_keys = tt_df[cols].drop_duplicates().map(normalize_value).agg("|".join, axis=1)
                ai_keys = ai_df[cols].drop_duplicates().map(normalize_value).agg("|".join, axis=1)
                counts = overlap_counts(tt_keys, ai_keys)
                id_rows.append(
                    {
                        "ttmatch_dataset": tt_name,
                        "aicup_dataset": ai_name,
                        "id_scope": scope,
                        "columns": "|".join(cols),
                        "left_unique_ids": int(len(tt_keys)),
                        "right_unique_ids": int(len(ai_keys)),
                        **counts,
                    }
                )
    id_df = pd.DataFrame(id_rows)

    dist_rows = []
    for tt_name in tt_names:
        tt_df = datasets[tt_name]
        for ai_name in ai_names:
            ai_df = datasets[ai_name]
            for col in DIST_COLS:
                if col not in tt_df.columns or col not in ai_df.columns:
                    continue
                dist_rows.append(
                    {
                        "ttmatch_dataset": tt_name,
                        "aicup_dataset": ai_name,
                        "column": col,
                        "ttmatch_unique": int(tt_df[col].nunique(dropna=False)),
                        "aicup_unique": int(ai_df[col].nunique(dropna=False)),
                        "js_divergence": js_divergence(tt_df[col], ai_df[col]),
                        "total_variation": total_variation(tt_df[col], ai_df[col]),
                        "top_values": top_common_values(tt_df[col], ai_df[col]),
                    }
                )
    dist_df = pd.DataFrame(dist_rows).sort_values(
        ["ttmatch_dataset", "aicup_dataset", "js_divergence"], ascending=[True, True, False]
    )

    risk_verdict, risk_warnings = verdict(overlap_df, sequence_df)
    summary = {
        "verdict": risk_verdict,
        "warnings": risk_warnings,
        "datasets": {
            name: {
                "path": str(DATASETS[name]),
                "rows": int(len(df)),
                "columns": [c for c in df.columns if c != "__row_index__"],
            }
            for name, df in datasets.items()
        },
        "max_non_id_overlap": (
            overlap_df[overlap_df["signature_scope"] == "canonical_non_id_common"]
            .sort_values("left_overlap_ratio", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ),
        "max_sequence_overlap": (
            sequence_df.sort_values("left_overlap_ratio", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ),
        "max_id_overlap": (
            id_df.sort_values("left_overlap_ratio", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ),
    }

    schema_df.to_csv(OUTDIR / "r178_schema_summary.csv", index=False)
    id_df.to_csv(OUTDIR / "r178_id_overlap_summary.csv", index=False)
    overlap_df.to_csv(OUTDIR / "r178_exact_row_overlap_summary.csv", index=False)
    sequence_df.to_csv(OUTDIR / "r178_sequence_overlap_summary.csv", index=False)
    dist_df.to_csv(OUTDIR / "r178_distribution_similarity.csv", index=False)
    (OUTDIR / "r178_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top_overlap = overlap_df.sort_values("left_overlap_ratio", ascending=False).head(20)
    top_seq = sequence_df.sort_values("left_overlap_ratio", ascending=False).head(20)
    top_ids = id_df.sort_values("left_overlap_ratio", ascending=False).head(20)
    top_dist = dist_df.sort_values("js_divergence", ascending=True).head(30)

    md = []
    md.append("# R178 TTMATCH Overlap Audit")
    md.append("")
    md.append(f"Verdict: `{risk_verdict}`")
    md.append("")
    if risk_warnings:
        md.append("## Warnings")
        for item in risk_warnings[:20]:
            md.append(f"- {item}")
        md.append("")
    md.append("## Schema Summary")
    md.append(csv_block(schema_df, 20))
    md.append("")
    md.append("## Top Exact Row Signature Overlaps")
    md.append(csv_block(top_overlap, 20))
    md.append("")
    md.append("## Top ID Overlaps")
    md.append(csv_block(top_ids, 20))
    md.append("")
    md.append("## Top Sequence Signature Overlaps")
    md.append(csv_block(top_seq, 20))
    md.append("")
    md.append("## Most Similar Marginal Distributions")
    md.append(csv_block(top_dist, 30))
    md.append("")
    md.append("## Interpretation")
    md.append(
        "- `canonical_all_common` includes every shared canonicalized column, including IDs when present."
    )
    md.append(
        "- `canonical_non_id_common` removes obvious identifiers such as `rally_uid`, `match`, `rally_id`, and player IDs."
    )
    md.append(
        "- `stroke_core` checks same-stroke feature tuples: strike/action/point/spin/hand/strength/position."
    )
    md.append(
        "- Sequence checks group rows by `rally_uid` and by `(match, numberGame, rally_id)` when available."
    )
    md.append(
        "- Distribution similarity alone is not direct leakage, but a schema-identical external dataset should still be limited to priors/distillation unless provenance is verified."
    )
    (OUTDIR / "r178_report.md").write_text("\n".join(md), encoding="utf-8")

    log_entry = f"""

## R178 TTMATCH overlap audit

- Script: `analysis_r178_ttmatch_overlap_audit.py`
- Output: `r178_ttmatch_overlap_audit/`
- Verdict: `{risk_verdict}`
- Files compared: `{', '.join(datasets.keys())}`
- Main artifacts:
  - `r178_schema_summary.csv`
  - `r178_id_overlap_summary.csv`
  - `r178_exact_row_overlap_summary.csv`
  - `r178_sequence_overlap_summary.csv`
  - `r178_distribution_similarity.csv`
  - `r178_report.md`
"""
    log_path = ROOT / "experiments_log.md"
    if log_path.exists():
        old = log_path.read_text(encoding="utf-8")
        if "## R178 TTMATCH overlap audit" not in old:
            log_path.write_text(old.rstrip() + log_entry + "\n", encoding="utf-8")

    print(f"R178 completed. Verdict={risk_verdict}. Output={OUTDIR}")


if __name__ == "__main__":
    main()
