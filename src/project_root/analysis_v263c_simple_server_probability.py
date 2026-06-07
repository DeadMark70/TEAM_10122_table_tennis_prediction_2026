"""V263C simple server probability blend.

This local-only experiment trains simple class-balanced serverGetPoint models
on safe prefix features, then blends their test probabilities into the clean
V261 cap1 / R121 anchor at tiny weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v263_questionnaire_baseline_helpers import (
    OUTDIR,
    add_questionnaire_columns,
    load_v261_cap1_anchor,
    numeric_features,
    write_local_submission,
)
from baseline_lgbm import (
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


SEED = 2632
MAX_LAG = 6
FOLDS = 5
WEIGHTS = [0.005, 0.010, 0.020]
SEARCH_PATH = OUTDIR / "v263c_server_search.csv"
BLOCKED_FEATURES = {
    "rally_uid",
    "match",
    "server_id",
    "receiver_id",
    "gamePlayerId",
    "gamePlayerOtherId",
    "scoreSelf",
    "scoreOther",
    "next_actionId",
    "next_pointId",
    "next_is_terminal",
    "serverGetPoint",
    "remaining_len",
    "final_parity_even",
    "num_prefixes_in_rally",
}


def clean_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0).astype(np.float32)


def positive_proba(model, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    classes = [int(cls) for cls in model.classes_]
    if 1 in classes:
        prob = raw[:, classes.index(1)]
    else:
        prob = np.zeros(len(frame), dtype=float)
    return np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)


def make_models(seed: int) -> list:
    return [
        ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=8,
            class_weight="balanced",
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=seed + 1,
            n_jobs=1,
        ),
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed + 2),
        ),
    ]


def averaged_positive_proba(models: list, frame: pd.DataFrame) -> np.ndarray:
    probs = [positive_proba(model, frame) for model in models]
    return np.mean(np.column_stack(probs), axis=1)


def fit_models(train_x: pd.DataFrame, y: pd.Series, seed: int) -> list:
    models = make_models(seed)
    for model in models:
        model.fit(train_x, y)
    return models


def fold_safe_proxy(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    global_mean: float,
) -> np.ndarray:
    keys = ["sex", "prefix_bin"]
    stats = train_part.groupby(keys)["serverGetPoint"].mean()
    proxy = []
    for _, row in valid_part[keys].iterrows():
        key = tuple(int(row[k]) for k in keys)
        proxy.append(float(stats.get(key, global_mean)))
    return np.clip(np.asarray(proxy, dtype=float), 1e-6, 1.0 - 1e-6)


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)

    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)

    train_prefix = add_questionnaire_columns(build_train_prefix_table(train, MAX_LAG))
    test_prefix = add_questionnaire_columns(build_test_prefix_table(test, MAX_LAG))
    features = numeric_features(train_prefix, test_prefix, BLOCKED_FEATURES)
    if not features:
        raise ValueError("No numeric features available for V263C.")
    leaked = [c for c in features if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player feature leakage detected: {leaked}")
    return train_prefix, test_prefix, features


def run_oof(train_prefix: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray | float]:
    y = train_prefix["serverGetPoint"].astype(int).to_numpy()
    groups = train_prefix["match"].to_numpy()
    splitter = GroupKFold(n_splits=FOLDS)
    oof_model = np.zeros(len(train_prefix), dtype=float)
    oof_proxy = np.zeros(len(train_prefix), dtype=float)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_prefix, y, groups), start=1):
        fold_train = train_prefix.iloc[tr_idx].copy()
        fold_valid = train_prefix.iloc[va_idx].copy()
        train_matches = set(fold_train["match"].unique())
        valid_matches = set(fold_valid["match"].unique())
        if train_matches & valid_matches:
            raise RuntimeError("GroupKFold leakage: train/valid match overlap.")

        x_train = clean_numeric_frame(fold_train, features)
        x_valid = clean_numeric_frame(fold_valid, features)
        models = fit_models(x_train, fold_train["serverGetPoint"].astype(int), SEED + fold * 10)
        oof_model[va_idx] = averaged_positive_proba(models, x_valid)
        oof_proxy[va_idx] = fold_safe_proxy(fold_train, fold_valid, float(fold_train["serverGetPoint"].mean()))
        fold_rows.append(
            {
                "fold": fold,
                "valid_rows": int(len(va_idx)),
                "valid_matches": int(len(valid_matches)),
                "server_auc": safe_auc(y[va_idx], oof_model[va_idx]),
                "proxy_auc": safe_auc(y[va_idx], oof_proxy[va_idx]),
            }
        )
        print(json.dumps(fold_rows[-1], sort_keys=True))

    return {
        "y": y,
        "oof_model": oof_model,
        "oof_proxy": oof_proxy,
        "model_auc": safe_auc(y, oof_model),
        "proxy_auc": safe_auc(y, oof_proxy),
    }


def train_full_predict(train_prefix: pd.DataFrame, test_prefix: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x_train = clean_numeric_frame(train_prefix, features)
    x_test = clean_numeric_frame(test_prefix, features)
    models = fit_models(x_train, train_prefix["serverGetPoint"].astype(int), SEED)
    test_prob = averaged_positive_proba(models, x_test)
    return pd.DataFrame(
        {
            "rally_uid": test_prefix["rally_uid"].astype(int).to_numpy(),
            "model_serverGetPoint": test_prob,
        }
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_prefix, test_prefix, features = build_feature_tables()
    anchor = load_v261_cap1_anchor()
    if len(anchor) != 1845:
        raise ValueError(f"Anchor has {len(anchor)} rows, expected 1845.")

    oof = run_oof(train_prefix, features)
    test_model = train_full_predict(train_prefix, test_prefix, features)
    out_base = anchor.merge(test_model, on="rally_uid", how="left", validate="one_to_one")
    if out_base["model_serverGetPoint"].isna().any():
        missing = int(out_base["model_serverGetPoint"].isna().sum())
        raise ValueError(f"Missing model probabilities for {missing} anchor rows.")

    anchor_server = out_base["serverGetPoint"].to_numpy(dtype=float)
    model_server = out_base["model_serverGetPoint"].to_numpy(dtype=float)
    rows = []
    y = np.asarray(oof["y"], dtype=int)
    oof_proxy = np.asarray(oof["oof_proxy"], dtype=float)
    oof_model = np.asarray(oof["oof_model"], dtype=float)
    proxy_auc = float(oof["proxy_auc"])

    for weight in WEIGHTS:
        candidate = f"v263c_server_w{str(weight).replace('.', 'p')}__v173_v261cap1"
        blended_test = np.clip((1.0 - weight) * anchor_server + weight * model_server, 1e-6, 1.0 - 1e-6)
        blended_oof = np.clip((1.0 - weight) * oof_proxy + weight * oof_model, 1e-6, 1.0 - 1e-6)
        server_auc = safe_auc(y, blended_oof)
        mad = float(np.mean(np.abs(blended_test - anchor_server)))
        server_corr = corr(blended_test, anchor_server)
        verdict = "CANDIDATE_FOR_REVIEW" if (server_auc > proxy_auc and mad <= 0.02 and (np.isnan(server_corr) or server_corr >= 0.98)) else "LOCAL_NEGATIVE_DO_NOT_SUBMIT"

        sub = anchor.copy()
        sub["serverGetPoint"] = blended_test
        path = OUTDIR / f"submission_{candidate}.csv"
        write_local_submission(path, sub)
        check = pd.read_csv(path)
        if len(check) != 1845 or list(check.columns) != ["rally_uid", "actionId", "pointId", "serverGetPoint"]:
            raise ValueError(f"Bad submission shape for {path}")

        rows.append(
            {
                "candidate": candidate,
                "server_auc": server_auc,
                "delta_vs_proxy_base": float(server_auc - proxy_auc),
                "server_mad_vs_anchor": mad,
                "server_corr_vs_anchor": server_corr,
                "verdict": verdict,
                "path": str(path),
            }
        )

    search = pd.DataFrame(rows)
    search.to_csv(SEARCH_PATH, index=False)
    summary = {
        "train_prefix_rows": int(len(train_prefix)),
        "test_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "model_auc": float(oof["model_auc"]),
        "proxy_auc": proxy_auc,
        "best_candidate": search.sort_values("delta_vs_proxy_base", ascending=False).iloc[0].to_dict(),
        "search_path": str(SEARCH_PATH),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
