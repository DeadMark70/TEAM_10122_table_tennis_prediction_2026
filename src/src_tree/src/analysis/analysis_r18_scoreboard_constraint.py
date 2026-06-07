"""R18 scoreboard-constrained server and point0 diagnostic.

This experiment uses score progression across future rallies in the same
match/game/player-pair group to infer an aggregate server win rate for the
current rally block.

Compliance note:
  This is a high-sensitivity feature because it uses future rows within the
  test batch. The script is diagnostic-only and does not generate submission.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_v10b_r1_ensemble import assert_aligned
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R18 scoreboard diagnostic.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test-new", default="test_new.csv")
    parser.add_argument("--test-old", default="test_old.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--v5-oof", default="oof_proba_v5.pkl")
    parser.add_argument("--v7-oof", default="oof_proba_v7.pkl")
    parser.add_argument("--v10b-oof", default="oof_proba_v10b.pkl")
    parser.add_argument("--v10b-selected", default="v10b_r1_selected.json")
    parser.add_argument("--r1-feature-report", default="feature_report_r1.json")
    parser.add_argument("--score-coverage-report", default="r18_score_coverage_report.csv")
    parser.add_argument("--server-blend-report", default="r18_server_blend_report.csv")
    parser.add_argument("--point0-report", default="r18_point0_adjust_report.csv")
    parser.add_argument("--old-test-report", default="r18_old_test_server_report.csv")
    parser.add_argument("--selected", default="r18_selected.json")
    parser.add_argument("--feature-report", default="feature_report_r18.json")
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def future_score_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    first = (
        df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .head(1)
        .copy()
    )
    first["pmin"] = first[["gamePlayerId", "gamePlayerOtherId"]].min(axis=1)
    first["pmax"] = first[["gamePlayerId", "gamePlayerOtherId"]].max(axis=1)
    rows = []
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
                valid = False
                ds = dr = np.nan
            else:
                ds = int(next_score[server_id]) - int(cur["scoreSelf"])
                dr = int(next_score[receiver_id]) - int(cur["scoreOther"])
                valid = gap > 0 and ds >= 0 and dr >= 0 and (ds + dr == gap)
            rows.append(
                {
                    "rally_uid": int(cur["rally_uid"]),
                    "future_gap": gap,
                    "future_server_points": float(ds) if valid else np.nan,
                    "future_receiver_points": float(dr) if valid else np.nan,
                    "future_server_score_rate": float(ds / gap) if valid else np.nan,
                    "future_score_valid": int(valid),
                }
            )
    out = pd.DataFrame(rows)
    # Last rally in each group has no next score aggregate.
    all_uids = pd.DataFrame({"rally_uid": first["rally_uid"].astype(int).unique()})
    out = all_uids.merge(out, on="rally_uid", how="left")
    out["future_score_valid"] = out["future_score_valid"].fillna(0).astype(int)
    return out


def build_safe_server(v3, v5, v7, v10, selected_v10) -> np.ndarray:
    _, _, v3_server = compose_v3(v3)
    r1_server = 0.8 * v3_server + 0.1 * v5["gru_server"] + 0.1 * v7["tr_server"]
    return (1.0 - float(selected_v10["server_v10_weight"])) * r1_server + float(
        selected_v10["server_v10_weight"]
    ) * v10["v10_server"]


def coverage_rows(name: str, df: pd.DataFrame, score_df: pd.DataFrame) -> list[dict]:
    rally_count = df["rally_uid"].nunique()
    valid = score_df[score_df["future_score_valid"].eq(1)].copy()
    rows = [
        {
            "dataset": name,
            "future_gap": "all",
            "rallies": int(rally_count),
            "valid_rallies": int(len(valid)),
            "coverage": float(len(valid) / rally_count) if rally_count else 0.0,
        }
    ]
    for gap, part in valid.groupby("future_gap"):
        rows.append(
            {
                "dataset": name,
                "future_gap": int(gap),
                "rallies": int(rally_count),
                "valid_rallies": int(len(part)),
                "coverage": float(len(part) / rally_count) if rally_count else 0.0,
            }
        )
    return rows


def score_rate_for_meta(meta: pd.DataFrame, score_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = meta[["rally_uid"]].merge(
        score_df[["rally_uid", "future_score_valid", "future_gap", "future_server_score_rate"]],
        on="rally_uid",
        how="left",
    )
    valid = merged["future_score_valid"].fillna(0).astype(int).to_numpy() == 1
    rate = merged["future_server_score_rate"].fillna(0.5).to_numpy(dtype=float)
    gap = merged["future_gap"].fillna(-1).to_numpy(dtype=int)
    return valid, rate, gap


def search_server_blend(meta: pd.DataFrame, base_server: np.ndarray, valid: np.ndarray, rate: np.ndarray) -> pd.DataFrame:
    y = meta["serverGetPoint"].to_numpy(dtype=int)
    rows = []
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        prob = base_server.copy()
        prob[valid] = (1.0 - lam) * prob[valid] + lam * rate[valid]
        rows.append(
            {
                "lambda": lam,
                "server_auc": float(roc_auc_score(y, prob)),
                "valid_coverage": float(valid.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("server_auc", ascending=False)


def search_point0_adjust(meta: pd.DataFrame, point_prob: np.ndarray, tuning: V3Tuning, valid: np.ndarray, rate: np.ndarray):
    rows = []
    next_strike = meta["prefix_len"].to_numpy(dtype=int) + 1
    terminal_server_win = next_strike % 2 == 0
    compat = np.where(terminal_server_win, rate, 1.0 - rate)
    base_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    base_f1 = f1_score(meta["next_pointId"], base_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        adj = point_prob.copy()
        factor = np.exp(beta * (compat[valid] - 0.5))
        adj[valid, 0] *= factor
        adj = normalize_rows(adj)
        pred = apply_segmented_multipliers(meta, adj, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
        point_f1 = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
        rows.append(
            {
                "beta": beta,
                "point_macro_f1": float(point_f1),
                "gain_vs_base": float(point_f1 - base_f1),
                "point_churn_vs_base": float((pred != base_pred).mean()),
                "point0_pred_count": int((pred == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("point_macro_f1", ascending=False)


def old_test_server_report(old_df: pd.DataFrame, new_score: pd.DataFrame) -> pd.DataFrame:
    first = old_df.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).head(1)
    labels = first[["rally_uid", "serverGetPoint"]].copy()
    merged = labels.merge(new_score, on="rally_uid", how="left")
    valid = merged["future_score_valid"].fillna(0).astype(int).eq(1)
    rows = []
    if valid.any():
        rows.append(
            {
                "subset": "valid_all",
                "rows": int(valid.sum()),
                "coverage": float(valid.mean()),
                "auc_future_rate": float(
                    roc_auc_score(merged.loc[valid, "serverGetPoint"], merged.loc[valid, "future_server_score_rate"])
                ),
            }
        )
        for gap, part in merged[valid].groupby("future_gap"):
            if part["serverGetPoint"].nunique() < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(part["serverGetPoint"], part["future_server_score_rate"]))
            rows.append({"subset": f"gap_{int(gap)}", "rows": int(len(part)), "coverage": float(len(part) / len(merged)), "auc_future_rate": auc})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test_new = pd.read_csv(args.test_new)
    test_old = pd.read_csv(args.test_old) if Path(args.test_old).exists() else None
    train_score = future_score_aggregate(train)
    new_score = future_score_aggregate(test_new)
    rows = []
    rows.extend(coverage_rows("train", train, train_score))
    rows.extend(coverage_rows("test_new", test_new, new_score))
    if test_old is not None:
        rows.extend(coverage_rows("test_old_self", test_old, future_score_aggregate(test_old)))
    pd.DataFrame(rows).to_csv(args.score_coverage_report, index=False)

    v3 = load_pickle(args.v3_oof)
    v5 = load_pickle(args.v5_oof)
    v7 = load_pickle(args.v7_oof)
    v10 = load_pickle(args.v10b_oof)
    selected_v10 = json.loads(Path(args.v10b_selected).read_text(encoding="utf-8"))
    meta = normalize_meta(v3["valid_meta"])
    for name, oof in [("V5", v5), ("V7", v7), ("V10B", v10)]:
        assert_aligned(meta, oof["valid_meta"], name)
    base_server = build_safe_server(v3, v5, v7, v10, selected_v10)
    valid, rate, gap = score_rate_for_meta(meta, train_score)
    server_report = search_server_blend(meta, base_server, valid, rate)
    server_report.to_csv(args.server_blend_report, index=False)

    _, v3_point, _ = compose_v3(v3)
    point0_report = search_point0_adjust(meta, v3_point, v3["tuning"], valid, rate)
    point0_report.to_csv(args.point0_report, index=False)

    old_report = old_test_server_report(test_old, new_score) if test_old is not None else pd.DataFrame()
    old_report.to_csv(args.old_test_report, index=False)

    selected = {
        "server_best": server_report.iloc[0].to_dict() if len(server_report) else {},
        "point0_best": point0_report.iloc[0].to_dict() if len(point0_report) else {},
        "old_test_server_auc": old_report.iloc[0].to_dict() if len(old_report) else {},
        "submit_recommendation": False,
        "compliance_note": "diagnostic_only_high_sensitivity_future_score_context",
    }
    Path(args.selected).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    metadata = {
        "coverage_report": args.score_coverage_report,
        "server_blend_report": args.server_blend_report,
        "point0_report": args.point0_report,
        "old_test_report": args.old_test_report,
        "selected": selected,
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("R18 selected:")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
