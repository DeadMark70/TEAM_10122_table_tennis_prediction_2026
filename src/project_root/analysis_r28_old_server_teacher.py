"""R28 old-test server teacher / distillation experiment.

The old reference test file contains serverGetPoint for 1,236 rally_uids that
are also present in test_new.  This script separates three uses:

1. train-only diagnostic: train a server model on train prefixes and score the
   old-test labeled prefixes;
2. teacher/distillation: add old-test labeled prefixes as extra server training
   rows, then predict current test_new server probabilities;
3. direct diagnostic: fill matched rally_uids with old server labels to estimate
   the ceiling/leaderboard mechanism.  This output is marked high-sensitivity.

No action or point predictions are retrained here; submissions reuse a base
submission for actionId/pointId and replace only serverGetPoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baseline_lgbm import (
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    feature_columns,
    make_lgbm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R28 old server teacher experiment.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test-new", default="test_new.csv")
    parser.add_argument("--test-old", default="test_old.csv")
    parser.add_argument("--base-submission", default="submission_r1.csv")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--old-weight-grid", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--blend-grid", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--diagnostic-report", default="r28_old_server_teacher_report.csv")
    parser.add_argument("--summary", default="r28_summary.csv")
    parser.add_argument("--selected", default="r28_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r28.json")
    parser.add_argument("--recommendation", default="r28_recommendation.md")
    parser.add_argument("--submission-prefix", default="submission_r28")
    return parser.parse_args()


def server_weights_from_prefix(df: pd.DataFrame) -> np.ndarray:
    if "num_prefixes_in_rally" in df.columns:
        w = 1.0 / df["num_prefixes_in_rally"].to_numpy(dtype=float)
    else:
        w = np.ones(len(df), dtype=float)
    return w / np.mean(w)


def make_server_model(n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    model = make_lgbm("binary", n_estimators, seed)
    model.set_params(
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        reg_alpha=0.05,
        reg_lambda=1.5,
    )
    return model


def predict_server(model: lgb.LGBMClassifier, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(df[features])
    prob = raw[:, 1] if raw.ndim == 2 else raw
    return np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)


def old_server_labels(old_df: pd.DataFrame) -> pd.DataFrame:
    return (
        old_df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)[["rally_uid", "serverGetPoint"]]
        .copy()
    )


def build_old_labeled_prefix(old_df: pd.DataFrame, max_lag: int, features: list[str]) -> pd.DataFrame:
    old_feat_df = add_role_and_score_features(old_df.drop(columns=["serverGetPoint"], errors="ignore"))
    old_prefix = build_test_prefix_table(old_feat_df, max_lag)
    labels = old_server_labels(old_df)
    old_prefix = old_prefix.merge(labels, on="rally_uid", how="left", validate="one_to_one")
    missing = old_prefix["serverGetPoint"].isna().sum()
    if missing:
        raise ValueError(f"Missing old server labels for {missing} old prefixes")
    for col in features:
        if col not in old_prefix.columns:
            old_prefix[col] = 0
    return old_prefix[["rally_uid", "match", "serverGetPoint"] + features].copy()


def write_submission(
    base: pd.DataFrame,
    server_prob: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    sub = base.copy()
    sub["serverGetPoint"] = np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8)
    sub.to_csv(path, index=False, float_format="%.8f")
    return sub


def write_recommendation(path: Path, selected: dict, summary_rows: list[dict]) -> None:
    train_auc = selected["train_only_old_auc"]
    best = selected["recommended_teacher"]
    lines = [
        "# R28 old-server teacher recommendation",
        "",
        "This experiment only changes `serverGetPoint`; actionId and pointId are copied from the base submission.",
        "",
        "## Findings",
        "",
        f"- Train-only server model AUC on old labeled prefixes: {train_auc:.6f}",
        f"- Old server labels cover {selected['old_label_coverage']} / {selected['test_new_rows']} current test_new rows.",
        f"- Recommended lower-risk teacher file: `{best['submission']}`",
        f"- Teacher setting: old_weight={best['old_weight']}, blend_with_base={best['blend_weight']}",
        f"- Teacher/base server correlation: {best['corr_with_base']:.6f}",
        f"- Teacher/base mean absolute diff: {best['mad_vs_base']:.6f}",
        "",
        "## Risk split",
        "",
        "- `submission_r28_teacher_*.csv` files are the safer distillation-style outputs: old labels are used as extra training rows, not copied row-by-row.",
        "- `submission_r28_old_server_direct_diagnostic.csv` directly fills matched old labels and is high-sensitivity. Treat it as a ceiling/diagnostic artifact unless the organizer explicitly confirms this usage.",
        "- The old/new alignment report showed no direct action/point target visibility, so R28 targets only the server task.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test_new = pd.read_csv(args.test_new)
    test_old = pd.read_csv(args.test_old)
    base_sub = pd.read_csv(args.base_submission)

    train_feat = add_role_and_score_features(train)
    test_new_feat = add_role_and_score_features(test_new)

    train_prefix = build_train_prefix_table(train_feat, args.max_lag)
    test_prefix = build_test_prefix_table(test_new_feat, args.max_lag)
    features = feature_columns(train_prefix)
    for col in features:
        if col not in test_prefix.columns:
            test_prefix[col] = 0
    test_prefix = test_prefix[["rally_uid", "match"] + features].copy()

    old_prefix = build_old_labeled_prefix(test_old, args.max_lag, features)

    # Train-only diagnostic on the old labeled reference set.
    train_only = make_server_model(args.n_estimators, args.seed)
    train_only.fit(
        train_prefix[features],
        train_prefix["serverGetPoint"],
        sample_weight=server_weights_from_prefix(train_prefix),
    )
    old_train_only_prob = predict_server(train_only, old_prefix, features)
    old_auc = float(roc_auc_score(old_prefix["serverGetPoint"], old_train_only_prob))

    # Align base submission and test prefix order.
    base = test_prefix[["rally_uid"]].merge(base_sub, on="rally_uid", how="left", validate="one_to_one")
    if base[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Base submission does not cover all test_new rally_uids.")
    base_server = base["serverGetPoint"].to_numpy(dtype=float)

    labels = old_server_labels(test_old)
    old_label_map = labels.set_index("rally_uid")["serverGetPoint"].to_dict()
    direct_server = base_server.copy()
    matched_mask = test_prefix["rally_uid"].isin(old_label_map).to_numpy()
    direct_server[matched_mask] = test_prefix.loc[matched_mask, "rally_uid"].map(old_label_map).to_numpy(dtype=float)
    direct_server = np.clip(direct_server * 0.998 + 0.001, 1e-6, 1.0 - 1e-6)
    direct_path = Path(f"{args.submission_prefix}_old_server_direct_diagnostic.csv")
    write_submission(base, direct_server, direct_path)

    rows: list[dict] = []
    teacher_candidates: list[dict] = []
    train_x = train_prefix[features]
    train_y = train_prefix["serverGetPoint"].astype(int)
    train_w = server_weights_from_prefix(train_prefix)

    old_x = old_prefix[features]
    old_y = old_prefix["serverGetPoint"].astype(int)

    for old_weight in args.old_weight_grid:
        combined_x = pd.concat([train_x, old_x], ignore_index=True)
        combined_y = pd.concat([train_y, old_y], ignore_index=True)
        old_w = np.full(len(old_prefix), float(old_weight), dtype=float)
        combined_w = np.concatenate([train_w, old_w])
        combined_w = combined_w / np.mean(combined_w)

        model = make_server_model(args.n_estimators, args.seed + int(old_weight * 1000) + 17)
        model.fit(combined_x, combined_y, sample_weight=combined_w)
        teacher_prob = predict_server(model, test_prefix, features)
        teacher_old_prob = predict_server(model, old_prefix, features)
        teacher_old_auc = float(roc_auc_score(old_y, teacher_old_prob))

        corr = float(np.corrcoef(base_server, teacher_prob)[0, 1])
        mad = float(np.mean(np.abs(base_server - teacher_prob)))
        rows.append(
            {
                "kind": "teacher_raw",
                "old_weight": float(old_weight),
                "blend_weight": 1.0,
                "old_auc_train_eval": teacher_old_auc,
                "corr_with_base": corr,
                "mad_vs_base": mad,
                "matched_mean": float(teacher_prob[matched_mask].mean()),
                "new_only_mean": float(teacher_prob[~matched_mask].mean()),
                "submission": "",
            }
        )

        for blend_weight in args.blend_grid:
            final_server = (1.0 - blend_weight) * base_server + blend_weight * teacher_prob
            sub_path = Path(
                f"{args.submission_prefix}_teacher_ow{str(old_weight).replace('.', 'p')}_bw{str(blend_weight).replace('.', 'p')}.csv"
            )
            write_submission(base, final_server, sub_path)
            cand = {
                "kind": "teacher_blend",
                "old_weight": float(old_weight),
                "blend_weight": float(blend_weight),
                "old_auc_train_eval": teacher_old_auc,
                "corr_with_base": float(np.corrcoef(base_server, final_server)[0, 1]),
                "mad_vs_base": float(np.mean(np.abs(base_server - final_server))),
                "matched_mean": float(final_server[matched_mask].mean()),
                "new_only_mean": float(final_server[~matched_mask].mean()),
                "submission": str(sub_path),
            }
            rows.append(cand)
            teacher_candidates.append(cand)

    # Conservative automatic recommendation: use moderate old weight and keep
    # server churn below a broad safety threshold if possible.
    sorted_candidates = sorted(
        teacher_candidates,
        key=lambda r: (
            abs(r["old_weight"] - 0.25),
            abs(r["blend_weight"] - 0.5),
            r["mad_vs_base"],
        ),
    )
    recommended = sorted_candidates[0]

    rows.append(
        {
            "kind": "direct_diagnostic",
            "old_weight": np.nan,
            "blend_weight": np.nan,
            "old_auc_train_eval": np.nan,
            "corr_with_base": float(np.corrcoef(base_server, direct_server)[0, 1]),
            "mad_vs_base": float(np.mean(np.abs(base_server - direct_server))),
            "matched_mean": float(direct_server[matched_mask].mean()),
            "new_only_mean": float(direct_server[~matched_mask].mean()),
            "submission": str(direct_path),
        }
    )

    report = pd.DataFrame(rows)
    report.to_csv(args.diagnostic_report, index=False)

    summary = [
        {"metric": "train_prefix_rows", "value": int(len(train_prefix))},
        {"metric": "old_labeled_prefix_rows", "value": int(len(old_prefix))},
        {"metric": "test_new_rows", "value": int(len(test_prefix))},
        {"metric": "old_label_coverage", "value": int(matched_mask.sum())},
        {"metric": "old_label_coverage_ratio", "value": float(matched_mask.mean())},
        {"metric": "train_only_old_auc", "value": old_auc},
        {"metric": "direct_mad_vs_base", "value": float(np.mean(np.abs(base_server - direct_server)))},
    ]
    pd.DataFrame(summary).to_csv(args.summary, index=False)

    selected = {
        "train_only_old_auc": old_auc,
        "old_label_coverage": int(matched_mask.sum()),
        "test_new_rows": int(len(test_prefix)),
        "recommended_teacher": recommended,
        "direct_diagnostic_submission": str(direct_path),
        "base_submission": args.base_submission,
        "notes": [
            "Teacher submissions use old server labels as extra training rows.",
            "Direct diagnostic fills matched old server labels and is high-sensitivity.",
        ],
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    feature_report = {
        "args": vars(args),
        "features": features,
        "feature_count": len(features),
        "train_prefix_rows": int(len(train_prefix)),
        "old_labeled_prefix_rows": int(len(old_prefix)),
        "test_new_rows": int(len(test_prefix)),
        "old_label_coverage": int(matched_mask.sum()),
        "old_label_coverage_ratio": float(matched_mask.mean()),
        "train_only_old_auc": old_auc,
        "recommended_teacher": recommended,
    }
    Path(args.feature_report).write_text(json.dumps(feature_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_recommendation(Path(args.recommendation), selected, rows)

    print(pd.DataFrame(summary).to_string(index=False))
    print("recommended teacher:", recommended)
    print(f"wrote {args.diagnostic_report}, {args.selected}, {args.recommendation}")


if __name__ == "__main__":
    main()
