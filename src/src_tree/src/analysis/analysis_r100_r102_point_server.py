"""R100/R102 point/server final-push diagnostics.

R100:
  V3 point mirror-TTA. Mirror the public test prefix, run the same V3 point
  predictor, flip point probabilities back, and blend with original V3 point.
  This is diagnostic because we do not have a fold-mirrored OOF for exact local
  validation; generated candidates keep action/server from stable submissions.

R102:
  Legal server momentum/dominance expert. Uses only train labels and current
  public prefix fields. Player/pair rates are fold-safe for OOF; test uses full
  train statistics. No old-test server labels and no future scoreboard context.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r100_r102_point_server")
SELECTED_DIR = Path("submissions/selected")
R101_SUB_PATH = Path("r101_r103_destiny_gru/submission_r101_r103_destiny_gru.csv")


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


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


POINT_MIRROR = {0: 0, 1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}
POS_MIRROR = {0: 0, 1: 3, 2: 2, 3: 1}


def mirror_public_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pointId" in out.columns:
        out["pointId"] = out["pointId"].map(POINT_MIRROR).fillna(out["pointId"]).astype(int)
    if "positionId" in out.columns:
        out["positionId"] = out["positionId"].map(POS_MIRROR).fillna(out["positionId"]).astype(int)
    return out


def flip_point_prob_back(prob: np.ndarray) -> np.ndarray:
    out = np.zeros_like(prob)
    for original, mirrored in POINT_MIRROR.items():
        out[:, original] = prob[:, mirrored]
    return normalize_rows(out)


def write_submission(
    test_meta: pd.DataFrame,
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    server_prob: np.ndarray,
    name: str,
    extra: dict | None = None,
) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path = SELECTED_DIR / name
    selected_path.write_bytes(path.read_bytes())
    info = {
        "candidate": name,
        "path": str(path),
        "upload_path": str(upload_path),
        "selected_path": str(selected_path),
    }
    if extra:
        info.update(extra)
    return info


def add_server_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["serverScore"].astype(float)
    r = out["receiverScore"].astype(float)
    total = out["scoreTotal"].astype(float)
    diff = out["serverScoreDiff"].astype(float)
    out["r102_server_points_to_11"] = np.maximum(0.0, 11.0 - s)
    out["r102_receiver_points_to_11"] = np.maximum(0.0, 11.0 - r)
    out["r102_abs_score_diff"] = np.abs(diff)
    out["r102_score_pressure"] = ((s >= 9) | (r >= 9)).astype(int)
    out["r102_deuce_exact"] = ((s >= 10) & (r >= 10) & (np.abs(diff) <= 1)).astype(int)
    out["r102_server_can_close_next"] = ((s >= 10) & (diff >= 1)).astype(int)
    out["r102_receiver_can_close_next"] = ((r >= 10) & (diff <= -1)).astype(int)
    out["r102_late_game_x_lead"] = out["r102_score_pressure"] * diff
    out["r102_rally_id_log1p"] = np.log1p(out["rally_id"].astype(float)) if "rally_id" in out.columns else 0.0
    out["r102_score_total_sqrt"] = np.sqrt(np.maximum(total, 0.0))
    return out


def _stats_map(pool: pd.DataFrame, key_cols: list[str], target: str, alpha: float, global_mean: float):
    grp = pool.groupby(key_cols, dropna=False)[target].agg(["sum", "count"])
    rate = (grp["sum"] + alpha * global_mean) / (grp["count"] + alpha)
    count = grp["count"].astype(float)
    return rate, count


def _lookup_stats(df: pd.DataFrame, key_cols: list[str], rate: pd.Series, count: pd.Series, global_mean: float) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.MultiIndex.from_frame(df[key_cols]) if len(key_cols) > 1 else pd.Index(df[key_cols[0]])
    values = rate.reindex(idx).fillna(global_mean).to_numpy(dtype=float)
    supports = count.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    return values, supports


def attach_server_rate_features(df: pd.DataFrame, pool: pd.DataFrame, alpha: float = 30.0) -> pd.DataFrame:
    out = df.copy()
    global_mean = float(pool["serverGetPoint"].mean())
    specs = [
        ("server_id", ["server_id"], "r102_server_id_rate"),
        ("receiver_id", ["receiver_id"], "r102_receiver_id_rate"),
        ("next_hitter_id", ["next_hitter_id"], "r102_hitter_id_rate"),
        ("server_receiver_pair", ["server_id", "receiver_id"], "r102_pair_rate"),
    ]
    for _, key_cols, name in specs:
        rate, count = _stats_map(pool, key_cols, "serverGetPoint", alpha, global_mean)
        values, supports = _lookup_stats(out, key_cols, rate, count, global_mean)
        out[name] = values
        out[f"{name}_support_log1p"] = np.log1p(supports)
    out["r102_server_receiver_rate_diff"] = out["r102_server_id_rate"] - out["r102_receiver_id_rate"]
    out["r102_pair_minus_global"] = out["r102_pair_rate"] - global_mean
    return out


def train_r102_server_oof(
    prefix_aligned: pd.DataFrame,
    prefix: pd.DataFrame,
    test_prefix: pd.DataFrame,
    base_features: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    valid_base = add_server_pressure_features(prefix_aligned)
    train_base = add_server_pressure_features(prefix)
    test_base = add_server_pressure_features(test_prefix)

    extra_cols = [
        "r102_server_points_to_11",
        "r102_receiver_points_to_11",
        "r102_abs_score_diff",
        "r102_score_pressure",
        "r102_deuce_exact",
        "r102_server_can_close_next",
        "r102_receiver_can_close_next",
        "r102_late_game_x_lead",
        "r102_rally_id_log1p",
        "r102_score_total_sqrt",
        "r102_server_id_rate",
        "r102_server_id_rate_support_log1p",
        "r102_receiver_id_rate",
        "r102_receiver_id_rate_support_log1p",
        "r102_hitter_id_rate",
        "r102_hitter_id_rate_support_log1p",
        "r102_pair_rate",
        "r102_pair_rate_support_log1p",
        "r102_server_receiver_rate_diff",
        "r102_pair_minus_global",
    ]
    features = base_features + [c for c in extra_cols if c not in base_features]
    oof = np.zeros(len(valid_base), dtype=float)
    fold_rows = []

    for fold in sorted(valid_base["fold"].unique()):
        valid_idx = valid_base.index[valid_base["fold"].eq(fold)].to_numpy()
        valid_matches = set(valid_base.loc[valid_idx, "match"])
        train_pool = train_base[~train_base["match"].isin(valid_matches)].copy()
        valid_fold = attach_server_rate_features(valid_base.loc[valid_idx].copy(), train_pool)
        train_fold = attach_server_rate_features(train_pool.copy(), train_pool)

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=35,
            reg_lambda=4.0,
            random_state=3400 + int(fold),
            verbose=-1,
        )
        X = train_fold[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = train_fold["serverGetPoint"].astype(int)
        w = class_weight_sample(y, 2)
        model.fit(X, y, sample_weight=w)
        pred = model.predict_proba(valid_fold[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
        oof[valid_idx] = pred
        fold_rows.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_fold)),
                "valid_rows": int(len(valid_fold)),
                "r102_server_auc": float(roc_auc_score(valid_fold["serverGetPoint"].astype(int), pred)),
            }
        )

    full_train = attach_server_rate_features(train_base.copy(), train_base)
    full_test = attach_server_rate_features(test_base.copy(), train_base)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=550,
        learning_rate=0.025,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_samples=35,
        reg_lambda=4.0,
        random_state=43102,
        verbose=-1,
    )
    X_full = full_train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_full = full_train["serverGetPoint"].astype(int)
    model.fit(X_full, y_full, sample_weight=class_weight_sample(y_full, 2))
    test_pred = model.predict_proba(full_test[features].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]

    report = pd.DataFrame(fold_rows)
    mean_row = {"fold": 0, "train_rows": 0, "valid_rows": 0, "r102_server_auc": float(report["r102_server_auc"].mean())}
    report = pd.concat([report, pd.DataFrame([mean_row])], ignore_index=True)
    return oof, np.clip(test_pred, 1e-6, 1 - 1e-6), report, features


def align_submission(path: Path, test_meta: pd.DataFrame) -> pd.DataFrame:
    sub = pd.read_csv(path)
    merged = test_meta[["rally_uid", "prefix_len"]].merge(sub, on="rally_uid", how="left")
    if merged[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"Could not align submission {path}.")
    return merged


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, prefix, test_prefix, features = prepare_prefix_features()
    v3_oof = load_pickle("oof_proba_v3.pkl")
    art = load_pickle("v47_v50_action_experts/v47_v50_action_experts.pkl")
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    prefix_aligned = align_prefix_meta(meta, prefix)

    v3_meta = normalize_meta(v3_oof["valid_meta"])
    if not v3_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(
        meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]
    ):
        raise ValueError("V3 OOF does not align.")
    _, v3_point_oof, v3_server_oof = compose_v3(v3_oof)

    # R100 point mirror TTA.
    test_prefix_v3, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    _, mirrored_point_test = compose_v3_full_point(train_raw, mirror_public_table(test_raw), v3_oof["tuning"])
    mirrored_back = flip_point_prob_back(mirrored_point_test)
    if not test_prefix_v3["rally_uid"].reset_index(drop=True).equals(test_meta["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 point test rows do not align to test_meta.")

    current_sub = align_submission(CURRENT_SUB_PATH, test_meta)
    r101_sub = align_submission(R101_SUB_PATH, test_meta) if R101_SUB_PATH.exists() else current_sub
    outputs: list[dict] = []

    for w in [0.25, 0.50, 0.75]:
        point_prob = normalize_rows((1.0 - w) * v3_point_test + w * mirrored_back)
        point_pred = apply_segmented_multipliers(
            test_prefix_v3, point_prob, v3_oof["tuning"].point_multipliers, POINT_CLASSES, v3_oof["tuning"].bins_mode
        )
        outputs.append(
            write_submission(
                test_meta,
                current_sub["actionId"].to_numpy(dtype=int),
                point_pred,
                current_sub["serverGetPoint"].to_numpy(dtype=float),
                f"submission_r100_v3point_mirror_w{clean_float(w)}_current_action_server.csv",
                {"branch": "r100", "mirror_weight": w, "point_diff_vs_current": float(np.mean(point_pred != current_sub["pointId"].to_numpy(dtype=int)))},
            )
        )
        outputs.append(
            write_submission(
                test_meta,
                r101_sub["actionId"].to_numpy(dtype=int),
                point_pred,
                r101_sub["serverGetPoint"].to_numpy(dtype=float),
                f"submission_r100_v3point_mirror_w{clean_float(w)}_r101_action_server.csv",
                {"branch": "r100", "mirror_weight": w, "point_diff_vs_current": float(np.mean(point_pred != current_sub["pointId"].to_numpy(dtype=int)))},
            )
        )

    # R102 server expert.
    r102_oof, r102_test, r102_report, r102_features = train_r102_server_oof(prefix_aligned, prefix, test_prefix, features)
    r102_report.to_csv(OUTDIR / "r102_server_cv_report.csv", index=False)
    base_auc = roc_auc_score(prefix_aligned["serverGetPoint"].astype(int), v3_server_oof)
    r102_auc = roc_auc_score(prefix_aligned["serverGetPoint"].astype(int), r102_oof)
    blend_rows = []
    for w in [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00]:
        blend = (1.0 - w) * v3_server_oof + w * r102_oof
        blend_rows.append({"server_weight": w, "oof_server_auc": float(roc_auc_score(prefix_aligned["serverGetPoint"].astype(int), blend))})
    blend_report = pd.DataFrame(blend_rows)
    blend_report.to_csv(OUTDIR / "r102_server_blend_report.csv", index=False)
    best_w = float(blend_report.sort_values("oof_server_auc", ascending=False).iloc[0]["server_weight"])

    for w in sorted(set([best_w, 0.10, 0.20, 0.35])):
        server_current = (1.0 - w) * current_sub["serverGetPoint"].to_numpy(dtype=float) + w * r102_test
        outputs.append(
            write_submission(
                test_meta,
                current_sub["actionId"].to_numpy(dtype=int),
                current_sub["pointId"].to_numpy(dtype=int),
                server_current,
                f"submission_r102_server_w{clean_float(w)}_current_action_point.csv",
                {"branch": "r102", "server_weight": w},
            )
        )
        server_r101 = (1.0 - w) * r101_sub["serverGetPoint"].to_numpy(dtype=float) + w * r102_test
        outputs.append(
            write_submission(
                test_meta,
                r101_sub["actionId"].to_numpy(dtype=int),
                r101_sub["pointId"].to_numpy(dtype=int),
                server_r101,
                f"submission_r102_server_w{clean_float(w)}_r101_action_point.csv",
                {"branch": "r102", "server_weight": w},
            )
        )

    summary = {
        "r100": {
            "note": "Mirror-TTA point branch is test-only diagnostic; no OOF estimate.",
            "point_flip_mapping": POINT_MIRROR,
        },
        "r102": {
            "base_v3_server_auc": float(base_auc),
            "r102_server_auc": float(r102_auc),
            "best_blend_weight_vs_v3_oof": best_w,
            "best_blend_auc_vs_v3_oof": float(blend_report["oof_server_auc"].max()),
            "feature_count": len(r102_features),
        },
        "generated": outputs,
    }
    (OUTDIR / "r100_r102_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    shutil.copy2("analysis_r100_r102_point_server.py", "src/analysis/analysis_r100_r102_point_server.py")
    shutil.copy2("baseline_r101_r103_destiny_gru.py", "src/train/baseline_r101_r103_destiny_gru.py")
    for dst in [UPLOAD_DIR / R101_SUB_PATH.name, SELECTED_DIR / R101_SUB_PATH.name]:
        if R101_SUB_PATH.exists():
            dst.write_bytes(R101_SUB_PATH.read_bytes())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
