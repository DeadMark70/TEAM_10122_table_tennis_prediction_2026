"""R9 phase-aware local prior for pointId.

This experiment keeps action/server from the R8-safe branch and tests whether
small, fold-safe phase conditional point priors can improve V3 point for only
the local prefix lengths where R7 showed signal: prefix_len=1 and prefix_len=3.
All other prefix lengths fall back to V3 point probabilities.
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
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


CONDITION_LEVELS = {
    "len1_serve": [
        ["prefix_len", "sex", "serve_actionId", "serve_spinId", "serve_pointId"],
        ["prefix_len", "sex", "serve_actionId", "serve_spinId"],
        ["prefix_len", "sex", "serve_actionId", "serve_point_depth"],
        ["prefix_len", "sex", "serve_actionId"],
        ["prefix_len", "sex"],
        ["prefix_len"],
    ],
    "len3_transition": [
        ["prefix_len", "sex", "serve_actionId", "receive_actionId", "lag0_actionId", "lag0_pointId"],
        ["prefix_len", "sex", "receive_actionId", "lag0_actionId", "lag0_pointId"],
        ["prefix_len", "sex", "lag0_actionId", "lag0_pointId", "lag0_spinId"],
        ["prefix_len", "sex", "last2_action_transition", "last2_point_transition"],
        ["prefix_len", "sex", "lag0_actionId", "lag0_point_depth"],
        ["prefix_len", "sex", "lag0_actionId"],
        ["prefix_len", "sex"],
        ["prefix_len"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R9 phase-local prior audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--r8-selected", default="r8_action_only_selected.json")
    parser.add_argument("--summary", default="r9_phase_prior_summary.csv")
    parser.add_argument("--prefix-report", default="prefix_len_report_r9.csv")
    parser.add_argument("--selected", default="r9_selected.json")
    parser.add_argument("--oof-proba", default="oof_proba_r9.pkl")
    parser.add_argument("--submission", default="submission_r9.csv")
    parser.add_argument("--feature-report", default="feature_report_r9.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[5.0, 20.0, 50.0, 100.0])
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.08, 0.10, 0.15])
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["rally_uid", "match", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    return meta[cols].reset_index(drop=True).astype(
        {
            "rally_uid": int,
            "match": int,
            "prefix_len": int,
            "next_actionId": int,
            "next_pointId": int,
            "serverGetPoint": int,
        }
    )


def compose_v3(oof: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from baseline_v2 import blend_probs

    tuning = oof["tuning"]
    action = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)
    point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )
    return action, point, np.clip(server, 1e-6, 1.0 - 1e-6)


class BackoffPrior:
    def __init__(self, levels: list[list[str]], alpha: float) -> None:
        self.levels = levels
        self.alpha = float(alpha)
        self.global_prior: np.ndarray | None = None
        self.tables: list[dict[tuple[int, ...], np.ndarray]] = []

    def fit(self, df: pd.DataFrame, target: str) -> "BackoffPrior":
        counts = df[target].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=np.float64)
        if counts.sum() == 0:
            counts += 1.0
        self.global_prior = counts / counts.sum()
        self.tables = []
        for level in self.levels:
            table = {}
            for key, sub in df.groupby(level, sort=False):
                if not isinstance(key, tuple):
                    key = (key,)
                local = sub[target].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=np.float64)
                prob = (local + self.alpha * self.global_prior) / (local.sum() + self.alpha)
                table[tuple(int(v) for v in key)] = prob.astype(np.float32)
            self.tables.append(table)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.global_prior is None:
            raise RuntimeError("BackoffPrior is not fitted.")
        out = np.tile(self.global_prior.astype(np.float32), (len(df), 1))
        for row_idx, row in enumerate(df.itertuples(index=False)):
            values = row._asdict()
            for level, table in zip(self.levels, self.tables):
                key = tuple(int(values[col]) for col in level)
                prob = table.get(key)
                if prob is not None:
                    out[row_idx] = prob
                    break
        return out / out.sum(axis=1, keepdims=True)


def build_phase_tables(train: pd.DataFrame, test: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_df = add_remaining_bucket(build_train_prefix_table(train, max_lag))
    test_prefix = build_test_prefix_table(test, max_lag)
    prefix_df = add_phase_features(prefix_df, train)
    test_prefix = add_phase_features(test_prefix, test)
    return prefix_df, test_prefix


def align_phase_to_meta(phase_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    key = ["rally_uid", "prefix_len"]
    aligned = meta[key].reset_index().merge(phase_df, on=key, how="left", suffixes=("", "_phase"))
    if aligned["match_phase"].isna().any() if "match_phase" in aligned else False:
        raise ValueError("Could not align phase rows.")
    return aligned.drop(columns=["index"]).reset_index(drop=True)


def build_oof_phase_priors(
    prefix_phase: pd.DataFrame,
    valid_meta: pd.DataFrame,
    test_lengths: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    rally_meta = prefix_phase[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    prior_store = {
        f"{name}_alpha{alpha:g}": np.zeros((len(valid_meta), len(POINT_CLASSES)), dtype=np.float32)
        for name in CONDITION_LEVELS
        for alpha in args.alpha_grid
    }
    meta_keys = valid_meta[["rally_uid", "prefix_len"]].copy()
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        tr_ids = set(rally_meta.iloc[tr_idx]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_idx]["rally_uid"])
        fold_train = prefix_phase[prefix_phase["rally_uid"].isin(tr_ids)].copy()
        va_pool = prefix_phase[prefix_phase["rally_uid"].isin(va_ids)].copy()
        sampled_idx = sample_validation_prefixes(va_pool, test_lengths, args.seed + fold)
        fold_valid = va_pool.loc[sampled_idx].copy().reset_index(drop=True)
        aligned = meta_keys.reset_index().merge(
            fold_valid[["rally_uid", "prefix_len"]].reset_index(),
            on=["rally_uid", "prefix_len"],
            how="inner",
            suffixes=("_global", "_local"),
        )
        if len(aligned) != len(fold_valid):
            raise ValueError(f"Fold {fold} phase/meta alignment failed.")
        global_idx = aligned["index_global"].to_numpy(dtype=int)
        for name, levels in CONDITION_LEVELS.items():
            if name.startswith("len1"):
                train_part = fold_train[fold_train["prefix_len"].eq(1)].copy()
            else:
                train_part = fold_train[fold_train["prefix_len"].eq(3)].copy()
            for alpha in args.alpha_grid:
                model = BackoffPrior(levels, alpha).fit(train_part, "next_pointId")
                prior_store[f"{name}_alpha{alpha:g}"][global_idx] = model.predict(fold_valid)
    return prior_store


def score_point(meta: pd.DataFrame, prob: np.ndarray, tuning: V3Tuning) -> tuple[float, np.ndarray]:
    pred = apply_segmented_multipliers(meta, prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    score = f1_score(meta["next_pointId"], pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    return float(score), pred


def prefix_report(meta: pd.DataFrame, point_pred: np.ndarray, action_score: float, server_auc: float) -> pd.DataFrame:
    rows = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
        ("le2", meta["prefix_len"].le(2).to_numpy()),
        ("ge3", meta["prefix_len"].ge(3).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        point = f1_score(
            meta.iloc[idx]["next_pointId"],
            point_pred[idx],
            average="macro",
            labels=POINT_CLASSES,
            zero_division=0,
        )
        rows.append(
            {
                "prefix_len_bin": label,
                "count": int(len(idx)),
                "point_macro_f1": float(point),
                "overall_with_global_action_server": float(0.4 * action_score + 0.4 * point + 0.2 * server_auc),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    v3 = load_pickle(args.v3_oof)
    selected_r8 = json.loads(Path(args.r8_selected).read_text(encoding="utf-8"))
    meta = normalize_meta(v3["valid_meta"])
    _, v3_point, _ = compose_v3(v3)
    base_score, base_pred = score_point(meta, v3_point, v3["tuning"])
    prefix_phase, test_phase = build_phase_tables(train, test, args.max_lag)
    priors = build_oof_phase_priors(prefix_phase, meta, test_phase["prefix_len"].to_numpy(dtype=int), args)

    mask1 = meta["prefix_len"].eq(1).to_numpy()
    mask3 = meta["prefix_len"].eq(3).to_numpy()
    rows = []
    best = None
    for key1, prior1 in [(None, None)] + [(k, v) for k, v in priors.items() if k.startswith("len1")]:
        for key3, prior3 in [(None, None)] + [(k, v) for k, v in priors.items() if k.startswith("len3")]:
            for lam1 in args.lambda_grid:
                if key1 is None and lam1 > 0:
                    continue
                for lam3 in args.lambda_grid:
                    if key3 is None and lam3 > 0:
                        continue
                    mixed = v3_point.copy()
                    if key1 is not None and lam1 > 0:
                        mixed[mask1] = (1.0 - lam1) * v3_point[mask1] + lam1 * prior1[mask1]
                    if key3 is not None and lam3 > 0:
                        mixed[mask3] = (1.0 - lam3) * v3_point[mask3] + lam3 * prior3[mask3]
                    mixed = mixed / mixed.sum(axis=1, keepdims=True)
                    score, pred = score_point(meta, mixed, v3["tuning"])
                    f1_len1 = f1_score(
                        meta.loc[mask1, "next_pointId"], pred[mask1], average="macro", labels=POINT_CLASSES, zero_division=0
                    )
                    f1_len3 = f1_score(
                        meta.loc[mask3, "next_pointId"], pred[mask3], average="macro", labels=POINT_CLASSES, zero_division=0
                    )
                    churn = float((pred != base_pred).mean())
                    churn_local = float((pred[mask1 | mask3] != base_pred[mask1 | mask3]).mean())
                    row = {
                        "len1_prior": key1 or "none",
                        "len3_prior": key3 or "none",
                        "lambda_len1": float(lam1),
                        "lambda_len3": float(lam3),
                        "point_macro_f1": score,
                        "gain_vs_v3": score - base_score,
                        "point_f1_len1": float(f1_len1),
                        "point_f1_len3": float(f1_len3),
                        "churn_vs_v3": churn,
                        "local_churn_vs_v3": churn_local,
                    }
                    rows.append(row)
                    eligible = (
                        row["gain_vs_v3"] >= 0.0015
                        and row["churn_vs_v3"] <= 0.03
                        and (lam1 > 0 or lam3 > 0)
                    )
                    objective = row["point_macro_f1"] - 0.02 * row["churn_vs_v3"]
                    if eligible and (best is None or objective > best["objective"]):
                        best = {"objective": objective, "row": row, "prob": mixed, "pred": pred}

    summary = pd.DataFrame(rows).sort_values(["gain_vs_v3", "point_macro_f1"], ascending=False).reset_index(drop=True)
    summary.to_csv(args.summary, index=False)
    if best is None:
        selected = {
            "config": "base_v3",
            "point_macro_f1": base_score,
            "gain_vs_v3": 0.0,
            "reason": "No eligible R9 phase-local prior met stopping criteria.",
        }
        selected_prob = v3_point
        selected_pred = base_pred
    else:
        selected = best["row"]
        selected_prob = best["prob"]
        selected_pred = best["pred"]

    action_score = float(selected_r8["metrics"]["action_macro_f1"])
    server_auc = float(selected_r8["metrics"]["server_auc"])
    selected_point = f1_score(meta["next_pointId"], selected_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    overall = 0.4 * action_score + 0.4 * selected_point + 0.2 * server_auc
    selected_out = {
        **selected,
        "r8_action_macro_f1": action_score,
        "r8_server_auc": server_auc,
        "overall_with_r8_action_server": float(overall),
        "point_policy": "R9 phase prior if selected, otherwise V3",
    }
    Path(args.selected).write_text(json.dumps(selected_out, indent=2), encoding="utf-8")
    prefix_report(meta, selected_pred, action_score, server_auc).to_csv(args.prefix_report, index=False)
    with open(args.oof_proba, "wb") as f:
        pickle.dump({"valid_meta": meta, "r9_point": selected_prob, "selected": selected_out}, f)

    metadata = {"selected": selected_out, "top_rows": summary.head(10).to_dict(orient="records")}
    wrote_submission = False
    if selected_out.get("config") != "base_v3":
        # Full-test generation is intentionally not implemented until OOF passes
        # the conservative threshold. Current experiments have consistently
        # selected base V3 for point branches.
        metadata["submission_note"] = "R9 selected non-base in OOF, but full-test generation is not implemented in this audit script."
    else:
        metadata["submission_note"] = "No R9 config selected; submission not generated because point remains V3."
    metadata["wrote_submission"] = wrote_submission
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.head(15).to_string(index=False))
    print("selected", json.dumps(selected_out, indent=2))
    print(f"wrote {args.summary}, {args.prefix_report}, {args.selected}, {args.oof_proba}, {args.feature_report}")


if __name__ == "__main__":
    main()
