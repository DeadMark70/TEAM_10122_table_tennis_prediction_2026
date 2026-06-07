"""V269 clean server value/ranking head.

Train clean, prefix-only server probability rankers and blend their positive
probabilities into the current V261 cap1 / V173 action / R121 server anchor.
This script does not read TTMATCH, old-server artifacts, or upload directly.
"""

from __future__ import annotations

import json
import shutil
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
    add_questionnaire_columns,
    load_v261_cap1_anchor,
    numeric_features,
)
from baseline_lgbm import (
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


SEED = 269
MAX_LAG = 6
FOLDS = 5
OUTDIR = Path("v269_clean_server_value_ranker")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
WEIGHTS = [0.005, 0.010, 0.020, 0.030, 0.050]
SEARCH_PATH = OUTDIR / "v269_server_search.csv"
REPORT_PATH = OUTDIR / "v269_report.md"

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


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = 1845) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
    if not df["serverGetPoint"].between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")


def clean_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0).astype(np.float32)


def cap_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def probability_separation(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(pos.mean() - neg.mean())


def positive_proba(model, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    classes = [int(cls) for cls in model.classes_]
    if 1 not in classes:
        return np.full(len(frame), 1e-6, dtype=float)
    return cap_prob(raw[:, classes.index(1)].astype(float))


def make_models(seed: int) -> list:
    models: list = [
        ExtraTreesClassifier(
            n_estimators=140,
            min_samples_leaf=7,
            class_weight="balanced",
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=140,
            min_samples_leaf=7,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=seed + 1,
            n_jobs=1,
        ),
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed + 2),
        ),
    ]
    return models


def fit_models(train_x: pd.DataFrame, y: pd.Series, seed: int) -> list:
    models = make_models(seed)
    for model in models:
        model.fit(train_x, y)
    return models


def averaged_positive_proba(models: list, frame: pd.DataFrame) -> np.ndarray:
    probs = [positive_proba(model, frame) for model in models]
    return cap_prob(np.mean(np.column_stack(probs), axis=1))


def fold_safe_proxy(train_part: pd.DataFrame, valid_part: pd.DataFrame, global_mean: float) -> np.ndarray:
    keys = ["sex", "prefix_bin"]
    missing = [key for key in keys if key not in train_part.columns or key not in valid_part.columns]
    if missing:
        return np.full(len(valid_part), global_mean, dtype=float)
    stats = train_part.groupby(keys)["serverGetPoint"].mean()
    proxy = []
    for _, row in valid_part[keys].iterrows():
        key = tuple(int(row[k]) for k in keys)
        proxy.append(float(stats.get(key, global_mean)))
    return cap_prob(np.asarray(proxy, dtype=float))


def build_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_new.csv")
    validate_raw_data(train, test)

    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    train_prefix = add_questionnaire_columns(build_train_prefix_table(train, MAX_LAG))
    test_prefix = add_questionnaire_columns(build_test_prefix_table(test, MAX_LAG))

    features = numeric_features(train_prefix, test_prefix, BLOCKED_FEATURES)
    leaked = [c for c in features if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player feature leakage detected: {leaked}")
    if not features:
        raise ValueError("No safe numeric features available for V269.")
    return train_prefix, test_prefix, features


def run_oof(train_prefix: pd.DataFrame, features: list[str]) -> dict[str, object]:
    y = train_prefix["serverGetPoint"].astype(int).to_numpy()
    groups = train_prefix["match"].to_numpy()
    oof_model = np.zeros(len(train_prefix), dtype=float)
    oof_proxy = np.zeros(len(train_prefix), dtype=float)
    fold_rows = []

    splitter = GroupKFold(n_splits=FOLDS)
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
        oof_proxy[va_idx] = fold_safe_proxy(
            fold_train,
            fold_valid,
            float(fold_train["serverGetPoint"].mean()),
        )
        fold_rows.append(
            {
                "fold": fold,
                "valid_rows": int(len(va_idx)),
                "valid_matches": int(len(valid_matches)),
                "server_auc": safe_auc(y[va_idx], oof_model[va_idx]),
                "proxy_auc": safe_auc(y[va_idx], oof_proxy[va_idx]),
                "server_separation": probability_separation(y[va_idx], oof_model[va_idx]),
                "proxy_separation": probability_separation(y[va_idx], oof_proxy[va_idx]),
            }
        )
        print(json.dumps(fold_rows[-1], sort_keys=True))

    return {
        "y": y,
        "oof_model": cap_prob(oof_model),
        "oof_proxy": cap_prob(oof_proxy),
        "model_auc": safe_auc(y, oof_model),
        "proxy_auc": safe_auc(y, oof_proxy),
        "model_separation": probability_separation(y, oof_model),
        "proxy_separation": probability_separation(y, oof_proxy),
        "fold_rows": fold_rows,
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


def weight_name(weight: float) -> str:
    text = f"{weight:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def classify_risk(mad: float) -> str:
    if mad <= 0.002:
        return "safe"
    if mad <= 0.005:
        return "exploratory"
    return "high_churn"


def verdict_for(delta: float, mad: float) -> str:
    if delta > 0.0 and mad <= 0.002:
        return "CANDIDATE_FOR_REVIEW"
    if delta > 0.0 and mad <= 0.005:
        return "EXPLORATORY_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def copy_to_upload(path: Path) -> None:
    if UPLOAD_DIR.exists():
        shutil.copy2(path, UPLOAD_DIR / path.name)


def write_submission(path: Path, anchor: pd.DataFrame, server_prob: np.ndarray) -> None:
    out = anchor.copy()
    out["serverGetPoint"] = cap_prob(server_prob)
    validate_submission_frame(out)
    out[EXPECTED_COLUMNS].to_csv(path, index=False, float_format="%.8f")
    copy_to_upload(path)


def write_report(search: pd.DataFrame, summary: dict[str, object], copied_count: int) -> None:
    best = search.sort_values(["delta_vs_proxy_base", "server_mad_vs_anchor"], ascending=[False, True]).iloc[0]
    lines = [
        "# V269 Clean Server Value Ranker",
        "",
        "Clean server-only probability blends into the V261 cap1 anchor.",
        "",
        "## Policy",
        "",
        "- No TTMATCH input.",
        "- No old-server input.",
        "- No direct upload.",
        "- `actionId` and `pointId` are unchanged from the anchor for every candidate.",
        "",
        "## OOF Diagnostics",
        "",
        f"- Train prefix rows: `{summary['train_prefix_rows']}`",
        f"- Test rows: `{summary['test_rows']}`",
        f"- Safe numeric features: `{summary['feature_count']}`",
        f"- Direct model server AUC: `{summary['model_auc']:.6f}`",
        f"- Proxy/base server AUC: `{summary['proxy_auc']:.6f}`",
        f"- Direct model separation: `{summary['model_separation']:.6f}`",
        f"- Proxy/base separation: `{summary['proxy_separation']:.6f}`",
        "",
        "## Candidates",
        "",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"- `{row['candidate']}`: AUC={row['server_auc']:.6f}, "
            f"delta={row['delta_vs_proxy_base']:.6f}, "
            f"MAD={row['server_mad_vs_anchor']:.6f}, "
            f"corr={row['server_corr_vs_anchor']:.6f}, "
            f"risk={row['risk_tier']}, verdict={row['verdict']}"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Best local row: `{best['candidate']}` with verdict `{best['verdict']}`.",
            f"Copied submissions to `{UPLOAD_DIR}`: `{copied_count}`.",
            "",
            f"Search CSV: `{SEARCH_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_prefix, test_prefix, features = build_feature_tables()
    anchor = load_v261_cap1_anchor()
    validate_submission_frame(anchor)

    oof = run_oof(train_prefix, features)
    test_model = train_full_predict(train_prefix, test_prefix, features)
    out_base = anchor.merge(test_model, on="rally_uid", how="left", validate="one_to_one")
    if out_base["model_serverGetPoint"].isna().any():
        missing = int(out_base["model_serverGetPoint"].isna().sum())
        raise ValueError(f"Missing model probabilities for {missing} anchor rows.")

    anchor_server = cap_prob(out_base["serverGetPoint"].to_numpy(dtype=float))
    model_server = cap_prob(out_base["model_serverGetPoint"].to_numpy(dtype=float))
    y = np.asarray(oof["y"], dtype=int)
    oof_proxy = np.asarray(oof["oof_proxy"], dtype=float)
    oof_model = np.asarray(oof["oof_model"], dtype=float)
    proxy_auc = float(oof["proxy_auc"])

    rows = []
    copied_count = 0
    for weight in WEIGHTS:
        candidate = f"submission_v269_server_w{weight_name(weight)}__v173_v261cap1.csv"
        blended_test = cap_prob((1.0 - weight) * anchor_server + weight * model_server)
        blended_oof = cap_prob((1.0 - weight) * oof_proxy + weight * oof_model)
        server_auc = safe_auc(y, blended_oof)
        delta = float(server_auc - proxy_auc)
        mad = float(np.mean(np.abs(blended_test - anchor_server)))
        server_corr = corr(blended_test, anchor_server)
        path = OUTDIR / candidate

        write_submission(path, anchor, blended_test)
        copied_count += int(UPLOAD_DIR.exists())

        check = pd.read_csv(path)
        validate_submission_frame(check)
        if not check["actionId"].equals(anchor["actionId"]) or not check["pointId"].equals(anchor["pointId"]):
            raise ValueError(f"Action/point changed in {path}")

        rows.append(
            {
                "candidate": candidate,
                "path": str(path),
                "server_auc": server_auc,
                "delta_vs_proxy_base": delta,
                "server_mad_vs_anchor": mad,
                "server_corr_vs_anchor": server_corr,
                "risk_tier": classify_risk(mad),
                "verdict": verdict_for(delta, mad),
            }
        )

    search = pd.DataFrame(rows)
    search.to_csv(SEARCH_PATH, index=False)

    summary = {
        "outdir": str(OUTDIR),
        "train_prefix_rows": int(len(train_prefix)),
        "test_rows": int(len(test_prefix)),
        "feature_count": int(len(features)),
        "model_auc": float(oof["model_auc"]),
        "proxy_auc": proxy_auc,
        "model_separation": float(oof["model_separation"]),
        "proxy_separation": float(oof["proxy_separation"]),
        "candidates": [row["candidate"] for row in rows],
        "search_path": str(SEARCH_PATH),
        "report_path": str(REPORT_PATH),
        "copied_to_upload_candidates": copied_count,
    }
    write_report(search, summary, copied_count)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
