"""R104/R107 heterogeneous stack and one-step future rollout.

R104:
  Candidate-level LightGBM stacker over heterogeneous action/point sources:
  tabular/golden/style experts from V47-V50 and causal R101/R103 GRU probs.

R107-lite:
  One-step future-rollout prior. Use train transition statistics
  P(next action | current action, current point, phase) and
  P(next point | current action, current point, phase), projected through
  current action/point probabilities. This is blended into R104 as a soft
  future-tendency feature, not as a hard simulator.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r1_oof_ensemble import compose_v3, normalize_meta
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r67_r70_meta_priors import align_prefix_meta, compose_v3_full_point, prepare_prefix_features
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, class_weight_sample
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r104_r107_deep_stack")
SELECTED_DIR = Path("submissions/selected")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")
R101_OOF_PATH = Path("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
R101_TEST_PATH = Path("r101_r103_destiny_gru/test_proba_r101_r103.pkl")


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


def ranks(prob: np.ndarray) -> np.ndarray:
    order = np.argsort(-prob, axis=1)
    out = np.empty_like(order)
    out[np.arange(len(prob))[:, None], order] = np.arange(prob.shape[1])[None, :]
    return out


def prob_features(prefix: str, prob: np.ndarray, cand: np.ndarray, row_id: np.ndarray) -> dict[str, np.ndarray]:
    r = ranks(prob)
    order = np.argsort(-prob, axis=1)
    top = prob[np.arange(len(prob)), order[:, 0]]
    second = prob[np.arange(len(prob)), order[:, 1]]
    p = prob[row_id, cand]
    return {
        f"{prefix}_prob": p,
        f"{prefix}_logprob": np.log(np.clip(p, 1e-12, 1.0)),
        f"{prefix}_rank": r[row_id, cand].astype(float),
        f"{prefix}_is_top": (order[row_id, 0] == cand).astype(float),
        f"{prefix}_top_prob": top[row_id],
        f"{prefix}_margin": top[row_id] - second[row_id],
    }


def make_candidate_frame(
    meta: pd.DataFrame,
    y: np.ndarray | None,
    classes: list[int],
    sources: dict[str, np.ndarray],
    row_features: pd.DataFrame,
) -> pd.DataFrame:
    n = len(meta)
    k = len(classes)
    row_id = np.repeat(np.arange(n), k)
    cand = np.tile(np.array(classes, dtype=int), n)
    data: dict[str, np.ndarray] = {
        "row_id": row_id,
        "candidate": cand,
        "prefix_len": np.repeat(meta["prefix_len"].to_numpy(dtype=float), k),
        "prefix_le2": np.repeat(meta["prefix_len"].le(2).astype(float).to_numpy(), k),
    }
    if "phase_id" in row_features.columns:
        data["phase_id"] = np.repeat(row_features["phase_id"].to_numpy(dtype=float), k)
    if "serverScoreDiff" in row_features.columns:
        data["serverScoreDiff"] = np.repeat(row_features["serverScoreDiff"].to_numpy(dtype=float), k)
    if y is not None:
        data["target"] = (cand == np.repeat(y.astype(int), k)).astype(int)
    for name, prob in sources.items():
        data.update(prob_features(name, prob, cand, row_id))
    if k == 19:
        data["is_rare_control"] = np.isin(cand, [8, 9, 12, 14]).astype(float)
        data["is_control"] = np.isin(cand, [6, 8, 9, 10, 11, 13]).astype(float)
        data["is_attack"] = np.isin(cand, [1, 2, 3, 4, 5, 7]).astype(float)
        data["is_defense"] = np.isin(cand, [12, 13, 14]).astype(float)
        data["is_zero"] = (cand == 0).astype(float)
    else:
        data["is_point0"] = (cand == 0).astype(float)
        data["point_depth"] = np.where(cand > 0, ((cand - 1) // 3) + 1, 0).astype(float)
        data["point_side"] = np.where(cand > 0, ((cand - 1) % 3) + 1, 0).astype(float)
    return pd.DataFrame(data)


def fit_predict_candidate_oof(
    cand: pd.DataFrame,
    meta: pd.DataFrame,
    classes: list[int],
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    feature_cols = [c for c in cand.columns if c not in {"target"}]
    oof_score = np.zeros(len(cand), dtype=float)
    rows = []
    for fold in sorted(meta["fold"].unique()):
        valid_rows = set(np.where(meta["fold"].eq(fold))[0])
        is_valid = cand["row_id"].isin(valid_rows).to_numpy()
        train_c = cand.loc[~is_valid]
        valid_c = cand.loc[is_valid]
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_samples=30,
            reg_lambda=5.0,
            random_state=seed + int(fold),
            verbose=-1,
        )
        y = train_c["target"].astype(int)
        sample_weight = np.where(y.to_numpy() == 1, len(classes) - 1, 1.0)
        model.fit(train_c[feature_cols], y, sample_weight=sample_weight)
        oof_score[is_valid] = model.predict_proba(valid_c[feature_cols])[:, 1]
        pred = choose_by_score(valid_c, oof_score[is_valid], classes)
        true = meta.loc[list(valid_rows)].sort_index()
        target_col = "next_actionId" if len(classes) == 19 else "next_pointId"
        rows.append(
            {
                "fold": int(fold),
                "macro_f1": float(f1_score(true[target_col].to_numpy(dtype=int), pred, average="macro", labels=classes, zero_division=0)),
                "rows": int(len(true)),
            }
        )
    return oof_score, pd.DataFrame(rows)


def choose_by_score(cand: pd.DataFrame, score: np.ndarray, classes: list[int]) -> np.ndarray:
    tmp = cand[["row_id", "candidate"]].copy()
    tmp["score"] = score
    best = tmp.sort_values(["row_id", "score"], ascending=[True, False]).drop_duplicates("row_id")
    return best.sort_values("row_id")["candidate"].to_numpy(dtype=int)


def fit_full_candidate(cand_train: pd.DataFrame, cand_test: pd.DataFrame, classes: list[int], seed: int) -> np.ndarray:
    feature_cols = [c for c in cand_train.columns if c not in {"target"}]
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=650,
        learning_rate=0.025,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=30,
        reg_lambda=5.0,
        random_state=seed,
        verbose=-1,
    )
    y = cand_train["target"].astype(int)
    sample_weight = np.where(y.to_numpy() == 1, len(classes) - 1, 1.0)
    model.fit(cand_train[feature_cols], y, sample_weight=sample_weight)
    score = model.predict_proba(cand_test[feature_cols])[:, 1]
    return choose_by_score(cand_test, score, classes)


def transition_rollout_prior(
    rows: pd.DataFrame,
    pool: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
    n_action: int = 19,
    n_point: int = 10,
    alpha: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    global_action = np.bincount(pool["next_actionId"].astype(int), minlength=n_action).astype(float)
    global_action = (global_action + alpha / n_action) / (global_action.sum() + alpha)
    global_point = np.bincount(pool["next_pointId"].astype(int), minlength=n_point).astype(float)
    global_point = (global_point + alpha / n_point) / (global_point.sum() + alpha)
    grouped = {}
    key_cols = ["lag0_actionId", "lag0_pointId", "phase_id"]
    for key, g in pool.groupby(key_cols, dropna=False):
        a = np.bincount(g["next_actionId"].astype(int), minlength=n_action).astype(float)
        p = np.bincount(g["next_pointId"].astype(int), minlength=n_point).astype(float)
        grouped[key] = (
            (a + alpha * global_action) / (a.sum() + alpha),
            (p + alpha * global_point) / (p.sum() + alpha),
        )
    out_a = np.zeros((len(rows), n_action), dtype=float)
    out_p = np.zeros((len(rows), n_point), dtype=float)
    # Low-cost expectation over top action/point states instead of all 190 pairs.
    top_a = np.argsort(-action_prob, axis=1)[:, :3]
    top_p = np.argsort(-point_prob, axis=1)[:, :3]
    phase = rows["phase_id"].to_numpy(dtype=int) if "phase_id" in rows.columns else np.zeros(len(rows), dtype=int)
    for i in range(len(rows)):
        total = 0.0
        for a in top_a[i]:
            for p in top_p[i]:
                w = float(action_prob[i, a] * point_prob[i, p])
                pa, pp = grouped.get((int(a), int(p), int(phase[i])), (global_action, global_point))
                out_a[i] += w * pa
                out_p[i] += w * pp
                total += w
        if total > 0:
            out_a[i] /= total
            out_p[i] /= total
        else:
            out_a[i] = global_action
            out_p[i] = global_point
    return normalize_rows(out_a), normalize_rows(out_p)


def rollout_oof(
    rows: pd.DataFrame,
    prefix: pd.DataFrame,
    action_prob: np.ndarray,
    point_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_a = np.zeros((len(rows), 19), dtype=float)
    out_p = np.zeros((len(rows), 10), dtype=float)
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].copy()
        pa, pp = transition_rollout_prior(rows.loc[idx], pool, action_prob[idx], point_prob[idx])
        out_a[idx] = pa
        out_p[idx] = pp
    return out_a, out_p


def write_submission(test_meta, action_pred, point_pred, server_prob, name, extra=None):
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    info = {"candidate": name, "path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}
    if extra:
        info.update(extra)
    return info


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    art = load_pickle(ARTIFACT_PATH)
    r101_oof = load_pickle(R101_OOF_PATH)
    r101_test = load_pickle(R101_TEST_PATH)
    train_raw, test_raw, prefix, test_prefix, features = prepare_prefix_features()
    meta = art["valid_meta"].copy().reset_index(drop=True)
    test_meta = art["test_meta"].copy().reset_index(drop=True)
    rows = align_prefix_meta(meta, prefix)

    r101_meta = normalize_meta(r101_oof["valid_meta"])
    if not r101_meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]].equals(
        meta[["rally_uid", "prefix_len", "next_actionId", "next_pointId"]]
    ):
        raise ValueError("R101 OOF does not align to artifact meta.")
    if not r101_test["test_meta"]["rally_uid"].reset_index(drop=True).equals(test_meta["rally_uid"].reset_index(drop=True)):
        raise ValueError("R101 test probabilities do not align.")

    v3_oof = load_pickle("oof_proba_v3.pkl")
    _, v3_point_oof, v3_server_oof = compose_v3(v3_oof)
    v3_test_prefix, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    if not v3_test_prefix["rally_uid"].reset_index(drop=True).equals(test_meta["rally_uid"].reset_index(drop=True)):
        raise ValueError("V3 test point does not align.")

    current_action_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof = normalize_rows(0.80 * current_action_oof + 0.20 * golden_oof)
    action_sources_oof = {
        "current": current_action_oof,
        "golden": golden_oof,
        "r42": r42_oof,
        "r101": r101_oof["gru_action"],
    }
    for name, prob in art["experts_oof"].items():
        action_sources_oof[name] = prob

    current_test_action = art["current_test_action"]
    golden_test = art["experts_test"]["v47_golden_test_soft"]
    r42_test = normalize_rows(0.80 * current_test_action + 0.20 * golden_test)
    action_sources_test = {
        "current": current_test_action,
        "golden": golden_test,
        "r42": r42_test,
        "r101": r101_test["gru_action"],
    }
    for name, prob in art["experts_test"].items():
        # The matching OOF artifact for the historical golden teacher is named
        # v47_v64_oof_soft; keep the same feature prefix for train/test.
        test_name = "v47_v64_oof_soft" if name == "v47_golden_test_soft" else name
        action_sources_test[test_name] = prob

    r107_action_oof, r107_point_oof = rollout_oof(rows, prefix, r101_oof["gru_action"], r101_oof["gru_point"])
    r107_action_test, r107_point_test = transition_rollout_prior(test_prefix, prefix, r101_test["gru_action"], r101_test["gru_point"])
    action_sources_oof["r107_rollout"] = r107_action_oof
    action_sources_test["r107_rollout"] = r107_action_test

    point_sources_oof = {"v3": v3_point_oof, "r101": r101_oof["gru_point"], "r107_rollout": r107_point_oof}
    point_sources_test = {"v3": v3_point_test, "r101": r101_test["gru_point"], "r107_rollout": r107_point_test}

    action_cand = make_candidate_frame(meta, meta["next_actionId"].to_numpy(dtype=int), ACTION_CLASSES, action_sources_oof, rows)
    action_score, action_fold = fit_predict_candidate_oof(action_cand, meta, ACTION_CLASSES, seed=10400)
    action_oof_pred = choose_by_score(action_cand, action_score, ACTION_CLASSES)
    action_oof_f1 = float(f1_score(meta["next_actionId"].to_numpy(dtype=int), action_oof_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    action_test_cand = make_candidate_frame(test_meta, None, ACTION_CLASSES, action_sources_test, test_prefix)
    action_test_pred = fit_full_candidate(action_cand, action_test_cand, ACTION_CLASSES, seed=11400)

    point_cand = make_candidate_frame(meta, meta["next_pointId"].to_numpy(dtype=int), POINT_CLASSES, point_sources_oof, rows)
    point_score, point_fold = fit_predict_candidate_oof(point_cand, meta, POINT_CLASSES, seed=10450)
    point_oof_pred = choose_by_score(point_cand, point_score, POINT_CLASSES)
    point_oof_f1 = float(f1_score(meta["next_pointId"].to_numpy(dtype=int), point_oof_pred, average="macro", labels=POINT_CLASSES, zero_division=0))

    point_test_cand = make_candidate_frame(test_meta, None, POINT_CLASSES, point_sources_test, test_prefix)
    point_test_pred = fit_full_candidate(point_cand, point_test_cand, POINT_CLASSES, seed=11450)

    server_rows = []
    best_server = r101_test["gru_server"]
    best_server_w = 1.0
    best_server_auc = 0.0
    for w in [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]:
        blend = (1.0 - w) * v3_server_oof + w * r101_oof["gru_server"]
        auc = float(roc_auc_score(meta["serverGetPoint"].astype(int), blend))
        server_rows.append({"r101_weight": w, "oof_server_auc": auc})
        if auc > best_server_auc:
            best_server_auc = auc
            best_server_w = w
    best_server = (1.0 - best_server_w) * pd.read_csv(CURRENT_SUB_PATH)["serverGetPoint"].to_numpy(dtype=float) + best_server_w * r101_test["gru_server"]
    server_report = pd.DataFrame(server_rows)

    current_sub = test_meta[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    generated = []
    generated.append(
        write_submission(
            test_meta,
            action_test_pred,
            current_sub["pointId"].to_numpy(dtype=int),
            current_sub["serverGetPoint"].to_numpy(dtype=float),
            "submission_r104_deepstack_action_v3point_current_server.csv",
            {"action_oof_f1": action_oof_f1},
        )
    )
    generated.append(
        write_submission(
            test_meta,
            action_test_pred,
            point_test_pred,
            current_sub["serverGetPoint"].to_numpy(dtype=float),
            "submission_r104_deepstack_action_point_current_server.csv",
            {"action_oof_f1": action_oof_f1, "point_oof_f1": point_oof_f1},
        )
    )
    generated.append(
        write_submission(
            test_meta,
            action_test_pred,
            point_test_pred,
            best_server,
            f"submission_r104_deepstack_action_point_serverw{clean_float(best_server_w)}.csv",
            {"action_oof_f1": action_oof_f1, "point_oof_f1": point_oof_f1, "server_oof_auc": best_server_auc},
        )
    )

    report = {
        "r104": {
            "action_oof_f1": action_oof_f1,
            "point_oof_f1": point_oof_f1,
            "best_server_r101_weight": best_server_w,
            "best_server_auc": best_server_auc,
        },
        "r107_lite": {
            "note": "One-step rollout priors are used as stacker probability sources.",
            "rollout_action_single_f1": float(
                f1_score(meta["next_actionId"].to_numpy(dtype=int), r107_action_oof.argmax(axis=1), average="macro", labels=ACTION_CLASSES, zero_division=0)
            ),
            "rollout_point_single_f1": float(
                f1_score(meta["next_pointId"].to_numpy(dtype=int), r107_point_oof.argmax(axis=1), average="macro", labels=POINT_CLASSES, zero_division=0)
            ),
        },
        "generated": generated,
    }
    action_fold.to_csv(OUTDIR / "r104_action_fold_report.csv", index=False)
    point_fold.to_csv(OUTDIR / "r104_point_fold_report.csv", index=False)
    server_report.to_csv(OUTDIR / "r104_server_blend_report.csv", index=False)
    (OUTDIR / "r104_r107_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r104_r107_deep_stack.py", "src/analysis/analysis_r104_r107_deep_stack.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
