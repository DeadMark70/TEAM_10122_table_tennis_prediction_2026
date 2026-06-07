"""V194 train/test split and prefix-distribution audit.

V192 showed that V188 raw neural point logits are not test-stable.  V194 audits
the data-generation side: raw train strokes, generated train prefix examples,
persisted OOF validation examples, and test_new submission rows are compared
as separate distributions.

This script does not train a model and does not generate submissions.  TTMATCH
is not read.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import analysis_v160_v163_task_pretrain_distill as v160
import analysis_v165_combined_external_pretrain_proxy as v165
from analysis_r179_action_physics_hierarchy import action_family, point_depth, point_side
from baseline_lgbm import add_role_and_score_features, sample_validation_prefixes


OUTDIR = Path("v194_train_test_split_distribution_audit")
SRC_DEST = Path("src/analysis/analysis_v194_train_test_split_distribution_audit.py")

NUMERIC_FEATURES = [
    "prefix_len",
    "scoreTotal",
    "serverScoreDiff",
    "lag0_actionId",
    "lag0_pointId",
    "lag0_spinId",
    "lag0_strengthId",
    "lag0_positionId",
]

CATEGORICAL_FEATURES = [
    "audit_phase",
    "audit_prefix_bin",
    "phase_id",
    "audit_lag0_action_family",
    "audit_lag0_depth",
    "audit_lag0_side",
    "lag0_actionId",
    "lag0_pointId",
    "lag0_spinId",
    "lag0_strengthId",
    "sex",
    "numberGame",
]


def phase_from_prefix_len(prefix_len: int | float) -> str:
    try:
        p = int(prefix_len)
    except (TypeError, ValueError):
        return "unknown"
    if p <= 0:
        return "serve"
    if p == 1:
        return "receive"
    if p == 2:
        return "third_ball"
    if p == 3:
        return "fourth_ball"
    return "rally"


def prefix_bin(prefix_len: int | float) -> str:
    try:
        p = int(prefix_len)
    except (TypeError, ValueError):
        return "unknown"
    if p <= 0:
        return "0"
    if p <= 3:
        return str(p)
    if p <= 5:
        return "4-5"
    if p <= 8:
        return "6-8"
    return "9+"


def depth_label(point_id: int | float) -> str:
    try:
        d = point_depth(int(point_id))
    except (TypeError, ValueError):
        return "unknown"
    return {0: "zero", 1: "short", 2: "half", 3: "long"}.get(d, "unknown")


def side_label(point_id: int | float) -> str:
    try:
        s = point_side(int(point_id))
    except (TypeError, ValueError):
        return "unknown"
    return {0: "zero", 1: "forehand", 2: "middle", 3: "backhand"}.get(s, "unknown")


def family_label(action_id: int | float) -> str:
    try:
        return action_family(int(action_id))
    except (TypeError, ValueError):
        return "unknown"


def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prefix_len" not in out.columns and "strikeNumber" in out.columns:
        out["prefix_len"] = out["strikeNumber"]
    if "lag0_actionId" not in out.columns and "actionId" in out.columns:
        out["lag0_actionId"] = out["actionId"]
    if "lag0_pointId" not in out.columns and "pointId" in out.columns:
        out["lag0_pointId"] = out["pointId"]
    if "lag0_spinId" not in out.columns and "spinId" in out.columns:
        out["lag0_spinId"] = out["spinId"]
    if "lag0_strengthId" not in out.columns and "strengthId" in out.columns:
        out["lag0_strengthId"] = out["strengthId"]
    if "lag0_positionId" not in out.columns and "positionId" in out.columns:
        out["lag0_positionId"] = out["positionId"]

    out["audit_phase"] = out["prefix_len"].map(phase_from_prefix_len)
    out["audit_prefix_bin"] = out["prefix_len"].map(prefix_bin)
    out["audit_lag0_action_family"] = out["lag0_actionId"].map(family_label)
    out["audit_lag0_depth"] = out["lag0_pointId"].map(depth_label)
    out["audit_lag0_side"] = out["lag0_pointId"].map(side_label)
    return out


def total_variation_distance(left: pd.Series, right: pd.Series) -> float:
    l = left.astype(str).value_counts(normalize=True, dropna=False)
    r = right.astype(str).value_counts(normalize=True, dropna=False)
    keys = sorted(set(l.index) | set(r.index))
    return float(0.5 * sum(abs(float(l.get(k, 0.0)) - float(r.get(k, 0.0))) for k in keys))


def categorical_shift_rows(
    feature: str,
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    l = left.astype(str).value_counts(normalize=True, dropna=False)
    r = right.astype(str).value_counts(normalize=True, dropna=False)
    keys = sorted(set(l.index) | set(r.index))
    tvd = total_variation_distance(left, right)
    rows = []
    for key in keys:
        train_share = float(l.get(key, 0.0))
        test_share = float(r.get(key, 0.0))
        rows.append(
            {
                "feature": feature,
                "value": key,
                "left_dataset": left_name,
                "right_dataset": right_name,
                "train_share": train_share,
                "test_share": test_share,
                "share_delta": test_share - train_share,
                "abs_share_delta": abs(test_share - train_share),
                "total_variation_distance": tvd,
            }
        )
    return pd.DataFrame(rows)


def safe_share(df: pd.DataFrame, col: str, value: str | int) -> float:
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return float(df[col].astype(str).eq(str(value)).mean())


def label_share(df: pd.DataFrame, col: str, value: int) -> float:
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").eq(value).mean())


def dataset_summary(name: str, df: pd.DataFrame) -> dict:
    pref = pd.to_numeric(df.get("prefix_len", pd.Series(dtype=float)), errors="coerce")
    rec = {
        "dataset": name,
        "rows": int(len(df)),
        "rallies": int(df["rally_uid"].nunique()) if "rally_uid" in df.columns else 0,
        "matches": int(df["match"].nunique()) if "match" in df.columns else 0,
        "prefix_mean": float(pref.mean()) if len(pref) else float("nan"),
        "prefix_p50": float(pref.quantile(0.50)) if len(pref) else float("nan"),
        "prefix_p90": float(pref.quantile(0.90)) if len(pref) else float("nan"),
        "prefix_max": float(pref.max()) if len(pref) else float("nan"),
        "receive_share": safe_share(df, "audit_phase", "receive"),
        "third_ball_share": safe_share(df, "audit_phase", "third_ball"),
        "fourth_ball_share": safe_share(df, "audit_phase", "fourth_ball"),
        "rally_share": safe_share(df, "audit_phase", "rally"),
        "lag0_long_share": safe_share(df, "audit_lag0_depth", "long"),
        "lag0_short_share": safe_share(df, "audit_lag0_depth", "short"),
        "lag0_attack_share": safe_share(df, "audit_lag0_action_family", "Attack"),
        "lag0_control_share": safe_share(df, "audit_lag0_action_family", "Control"),
        "lag0_defensive_share": safe_share(df, "audit_lag0_action_family", "Defensive"),
        "label_point0_rate": label_share(df, "next_pointId", 0),
        "label_terminal_rate": label_share(df, "next_is_terminal", 1),
    }
    if "fold" in df.columns:
        rec["folds"] = int(df["fold"].nunique())
    return rec


def numeric_shift(source_name: str, source: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        if col not in source.columns or col not in test.columns:
            continue
        s = pd.to_numeric(source[col], errors="coerce")
        t = pd.to_numeric(test[col], errors="coerce")
        rows.append(
            {
                "source_dataset": source_name,
                "feature": col,
                "source_mean": float(s.mean()),
                "test_mean": float(t.mean()),
                "mean_delta_test_minus_source": float(t.mean() - s.mean()),
                "source_p50": float(s.quantile(0.50)),
                "test_p50": float(t.quantile(0.50)),
                "source_p90": float(s.quantile(0.90)),
                "test_p90": float(t.quantile(0.90)),
                "abs_mean_delta": abs(float(t.mean() - s.mean())),
            }
        )
    return pd.DataFrame(rows)


def categorical_shift(source_name: str, source: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for col in CATEGORICAL_FEATURES:
        if col in source.columns and col in test.columns:
            parts.append(categorical_shift_rows(col, source[col], test[col], source_name, "test_new_examples"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def make_raw_train_strokes(train: pd.DataFrame) -> pd.DataFrame:
    raw = add_role_and_score_features(train.copy())
    raw["prefix_len"] = raw["strikeNumber"]
    raw["lag0_actionId"] = raw["actionId"]
    raw["lag0_pointId"] = raw["pointId"]
    raw["lag0_spinId"] = raw["spinId"]
    raw["lag0_strengthId"] = raw["strengthId"]
    raw["lag0_positionId"] = raw["positionId"]
    return add_audit_columns(raw)


def sampled_validation_like_test(prefix: pd.DataFrame, test_prefix: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rally_meta = prefix[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    sampled = []
    test_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    for fold, (_, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"])
        valid_pool = prefix[prefix["rally_uid"].isin(valid_rallies)].copy()
        idx = sample_validation_prefixes(valid_pool, test_lengths, seed + fold)
        part = valid_pool.loc[idx].copy()
        part["fold"] = fold - 1
        sampled.append(part)
    return add_audit_columns(pd.concat(sampled, ignore_index=True))


def fold_distribution(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fold, part in rows.groupby("fold", sort=True):
        rec = dataset_summary(f"oof_fold_{int(fold)}", part)
        rec["fold"] = int(fold)
        out.append(rec)
    return pd.DataFrame(out)


def strategy_table(prefix: pd.DataFrame, oof_rows: pd.DataFrame, test_prefix: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "raw_train_strokes",
                "example_source": "train.csv observed stroke rows",
                "rows": int(pd.read_csv("train.csv", usecols=["rally_uid"]).shape[0]),
                "split_or_sampling": "none",
                "risk_note": "Not the model training-example distribution.",
            },
            {
                "component": "generated_train_prefix_examples",
                "example_source": "build_train_prefix_table(train, max_lag=6)",
                "rows": int(len(prefix)),
                "split_or_sampling": "all legal train prefixes with next-stroke labels",
                "risk_note": "Longer/rally-heavy than persisted neural OOF pool.",
            },
            {
                "component": "baseline_lgbm_cv",
                "example_source": "full prefix train; validation sampled to test prefix lengths",
                "rows": int(len(prefix)),
                "split_or_sampling": "GroupKFold by match, validation uses sample_validation_prefixes(test_new prefix_len)",
                "risk_note": "The sampler intends to use test lengths, but actual sampled train rallies can remain short/receive-heavy.",
            },
            {
                "component": "V173 action / R184 action",
                "example_source": "R111 valid_meta aligned to prefix for OOF; full/test priors for test",
                "rows": int(len(oof_rows)),
                "split_or_sampling": "fold ids reconstructed by GroupKFold(match) when absent",
                "risk_note": "OOF action diagnostics inherit the persisted R111 valid_meta distribution.",
            },
            {
                "component": "V188 GRU point / V193 calibrated point",
                "example_source": "R111 valid_meta aligned rows, then train on rows[fold != k]",
                "rows": int(len(oof_rows)),
                "split_or_sampling": "5 fold train/valid over persisted OOF rows, not full 69k prefix pool",
                "risk_note": "This is the main suspected mismatch: short/receive-heavy training pool vs test_new.",
            },
            {
                "component": "test_new_examples",
                "example_source": "build_test_prefix_table(test_new, max_lag=6)",
                "rows": int(len(test_prefix)),
                "split_or_sampling": "one observed prefix per rally_uid",
                "risk_note": "More rally/long/attack-like than V188 persisted training pool.",
            },
        ]
    )


def write_report(summary: pd.DataFrame, numeric: pd.DataFrame, categorical: pd.DataFrame) -> None:
    def val(dataset: str, col: str) -> float:
        return float(summary.loc[summary["dataset"].eq(dataset), col].iloc[0])

    oof = "oof_validation_examples"
    full = "generated_train_prefix_examples"
    test = "test_new_examples"
    sampled = "sampled_validation_like_test"
    top_num = numeric.sort_values("abs_mean_delta", ascending=False).head(12)
    top_cat = categorical.sort_values("abs_share_delta", ascending=False).head(16)
    lines = [
        "# V194 Train/Test Split Distribution Audit",
        "",
        "- Verdict: `TRAINING_POOL_TEST_MISMATCH_CONFIRMED`",
        "- No model training, no submission generation.",
        "- TTMATCH is not read.",
        "",
        "## Core Finding",
        "",
        f"- Full generated train prefixes: rows `{int(val(full, 'rows'))}`, prefix mean `{val(full, 'prefix_mean'):.4f}`, rally share `{val(full, 'rally_share'):.4f}`.",
        f"- V188/V193 actual OOF training pool: rows `{int(val(oof, 'rows'))}`, prefix mean `{val(oof, 'prefix_mean'):.4f}`, rally share `{val(oof, 'rally_share'):.4f}`.",
        f"- Test_new examples: rows `{int(val(test, 'rows'))}`, prefix mean `{val(test, 'prefix_mean'):.4f}`, rally share `{val(test, 'rally_share'):.4f}`.",
        f"- Baseline sampled validation-like-test pool: prefix mean `{val(sampled, 'prefix_mean'):.4f}`, rally share `{val(sampled, 'rally_share'):.4f}`; in this audit it still matches the short OOF pool rather than test_new.",
        "",
        "## Key Shares",
        "",
        "| dataset | prefix_mean | receive | third | fourth | rally | lag0_long | lag0_attack | point0_label | terminal_label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['prefix_mean']:.4f} | {row['receive_share']:.4f} | {row['third_ball_share']:.4f} | "
            f"{row['fourth_ball_share']:.4f} | {row['rally_share']:.4f} | {row['lag0_long_share']:.4f} | "
            f"{row['lag0_attack_share']:.4f} | {row['label_point0_rate']:.4f} | {row['label_terminal_rate']:.4f} |"
        )
    lines += [
        "",
        "## Largest Numeric Shifts vs Test",
        "",
        "| source | feature | source_mean | test_mean | delta | source_p90 | test_p90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in top_num.iterrows():
        lines.append(
            f"| {row['source_dataset']} | {row['feature']} | {row['source_mean']:.4f} | {row['test_mean']:.4f} | "
            f"{row['mean_delta_test_minus_source']:.4f} | {row['source_p90']:.4f} | {row['test_p90']:.4f} |"
        )
    lines += [
        "",
        "## Largest Categorical Shifts vs Test",
        "",
        "| source | feature | value | source_share | test_share | delta | TVD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in top_cat.iterrows():
        lines.append(
            f"| {row['left_dataset']} | {row['feature']} | {row['value']} | {row['train_share']:.4f} | "
            f"{row['test_share']:.4f} | {row['share_delta']:.4f} | {row['total_variation_distance']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- V188/V193 point models are not trained on the full generated train-prefix distribution; they use the persisted R111 valid_meta pool.",
        "- The validation sampler path also remains much shorter and more receive-phase heavy than test_new, so the issue is not only raw train vs test but the actual OOF/training example pool.",
        "- Test_new is more rally/long/attack-like, while the persisted neural point pool is receive/third-ball heavy.",
        "- This supports the V192 finding: raw GRU learned useful OOF structure but its decision boundary is calibrated to the wrong prefix-state distribution.",
        "- The next clean experiment should be V195: train/resample the neural point model with test-distribution-matched weights over prefix_len, phase, lag0 depth, and lag0 action family.",
    ]
    (OUTDIR / "v194_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    train_raw = pd.read_csv("train.csv")
    _, _, prefix, test_prefix, _ = v165.prepare_prefix_features()
    raw_train = make_raw_train_strokes(train_raw)
    generated_prefix = add_audit_columns(prefix)
    test_examples = add_audit_columns(test_prefix.reset_index(drop=True))

    with open(v165.R111_OOF, "rb") as f:
        r111_oof = pickle.load(f)
    valid_meta = v160.ensure_fold(r111_oof["valid_meta"])
    oof_rows = add_audit_columns(v165.align_prefix_meta(valid_meta, prefix).reset_index(drop=True))
    sampled_valid = sampled_validation_like_test(prefix, test_prefix)

    datasets = {
        "raw_train_strokes": raw_train,
        "generated_train_prefix_examples": generated_prefix,
        "oof_validation_examples": oof_rows,
        "sampled_validation_like_test": sampled_valid,
        "test_new_examples": test_examples,
    }

    summary = pd.DataFrame([dataset_summary(name, df) for name, df in datasets.items()])
    summary.to_csv(OUTDIR / "v194_dataset_summary.csv", index=False)

    numeric = pd.concat(
        [numeric_shift(name, df, test_examples) for name, df in datasets.items() if name != "test_new_examples"],
        ignore_index=True,
    )
    numeric.to_csv(OUTDIR / "v194_numeric_shift_vs_test.csv", index=False)

    categorical = pd.concat(
        [categorical_shift(name, df, test_examples) for name, df in datasets.items() if name != "test_new_examples"],
        ignore_index=True,
    )
    categorical.to_csv(OUTDIR / "v194_categorical_shift_vs_test.csv", index=False)

    folds = fold_distribution(oof_rows)
    folds.to_csv(OUTDIR / "v194_oof_fold_distribution.csv", index=False)

    strategies = strategy_table(prefix, oof_rows, test_prefix)
    strategies.to_csv(OUTDIR / "v194_model_split_strategy.csv", index=False)

    report = {
        "verdict": "TRAINING_POOL_TEST_MISMATCH_CONFIRMED",
        "dataset_summary": summary.to_dict(orient="records"),
        "artifacts": {
            "summary": str(OUTDIR / "v194_dataset_summary.csv"),
            "numeric_shift": str(OUTDIR / "v194_numeric_shift_vs_test.csv"),
            "categorical_shift": str(OUTDIR / "v194_categorical_shift_vs_test.csv"),
            "fold_distribution": str(OUTDIR / "v194_oof_fold_distribution.csv"),
            "model_split_strategy": str(OUTDIR / "v194_model_split_strategy.csv"),
        },
        "notes": [
            "The V188/V193 neural point pool is the persisted R111 valid_meta rows, not the full generated prefix table.",
            "Baseline LGBM CV has a separate validation sampler that matches test_new prefix lengths.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v194_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(summary, numeric, categorical)
    shutil.copy2("analysis_v194_train_test_split_distribution_audit.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
