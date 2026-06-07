"""V235 player-conditional response action teacher.

This experiment tests the hypothesis that action is primarily a response policy:
incoming ball state + target hitter style + phase.  It builds fold-safe
smoothed player response priors and blends them with the public-positive V173
anchor.  Point is fixed at V188 cap5 and server is fixed at R121.

No TTMATCH and no old-server labels are read.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v194_train_test_split_distribution_audit import add_audit_columns
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import V3Tuning, GrUTuning, TransformerTuning, distill_v173_soft_anchor
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from analysis_v230_action_soft_teacher_factory import geometric_log_blend, normalize_rows_safe
from analysis_v233_public_like_validation_lab import density_ratio_weights, weighted_macro_f1
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v235_player_conditional_response_teacher")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v235_player_conditional_response_teacher.py")


def smoothed_action_prior(y: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=19).astype(float)
    prior = counts + float(smoothing)
    return prior / prior.sum()


def response_prior_from_counts(keys: np.ndarray, counts: np.ndarray, global_prior: np.ndarray, smoothing: float = 10.0) -> np.ndarray:
    keys = np.asarray(keys, dtype=int)
    counts = np.asarray(counts, dtype=float)
    global_prior = np.asarray(global_prior, dtype=float)
    out = np.zeros((len(keys), 19), dtype=float)
    for i, key in enumerate(keys):
        if 0 <= key < len(counts):
            row = counts[key] + float(smoothing) * global_prior
        else:
            row = float(smoothing) * global_prior
        out[i] = row
    return normalize_rows_safe(out)


def _phase_name(rows: pd.DataFrame) -> pd.Series:
    if "audit_phase" in rows.columns:
        return rows["audit_phase"].astype(str)
    if "is_receive_prediction" in rows.columns:
        out = pd.Series(["rally"] * len(rows), index=rows.index)
        out[pd.to_numeric(rows["is_receive_prediction"], errors="coerce").fillna(0).astype(bool)] = "receive"
        out[pd.to_numeric(rows.get("is_third_ball_prediction", 0), errors="coerce").fillna(0).astype(bool)] = "third_ball"
        return out
    return pd.Series(["rally"] * len(rows), index=rows.index)


def _lag_depth(rows: pd.DataFrame) -> pd.Series:
    if "audit_lag0_depth" in rows.columns:
        return rows["audit_lag0_depth"].astype(str)
    p = pd.to_numeric(rows.get("lag0_pointId", 0), errors="coerce").fillna(0).astype(int)
    return pd.Series(np.select([p.between(1, 3), p.between(4, 6), p.between(7, 9)], ["short", "half", "long"], default="zero"), index=rows.index)


def _merge_target_players(rows: pd.DataFrame, raw_prefix: pd.DataFrame) -> pd.DataFrame:
    if "prefix_len" in rows.columns:
        left = rows[["rally_uid", "prefix_len"]].copy()
        left["prefix_len"] = pd.to_numeric(left["prefix_len"], errors="coerce").fillna(0).astype(int)
        right = raw_prefix[["rally_uid", "strikeNumber", "gamePlayerId", "gamePlayerOtherId"]].copy()
        right["strikeNumber"] = pd.to_numeric(right["strikeNumber"], errors="coerce").fillna(0).astype(int)
        merged = left.merge(right, left_on=["rally_uid", "prefix_len"], right_on=["rally_uid", "strikeNumber"], how="left")
    else:
        merged = rows[["rally_uid"]].merge(raw_prefix[["rally_uid", "gamePlayerId", "gamePlayerOtherId"]], on="rally_uid", how="left")
    out = rows.copy()
    out["target_hitter_id"] = merged["gamePlayerOtherId"].fillna(-1).astype(int).to_numpy()
    out["target_receiver_id"] = merged["gamePlayerId"].fillna(-1).astype(int).to_numpy()
    return out


def response_context(rows: pd.DataFrame) -> pd.DataFrame:
    phase = _phase_name(rows)
    depth = _lag_depth(rows)
    lag_action = pd.to_numeric(rows.get("lag0_actionId", 0), errors="coerce").fillna(0).astype(int)
    lag_spin = pd.to_numeric(rows.get("lag0_spinId", 0), errors="coerce").fillna(0).astype(int)
    hitter = pd.to_numeric(rows.get("target_hitter_id", -1), errors="coerce").fillna(-1).astype(int)
    return pd.DataFrame(
        {
            "target_hitter_id": hitter,
            "phase": phase.astype(str),
            "lag0_actionId": lag_action,
            "lag0_spinId": lag_spin,
            "lag0_depth": depth.astype(str),
        }
    )


def _factorize_combined(train_ctx: pd.DataFrame, apply_ctx: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    train_key = train_ctx[cols].astype(str).agg("||".join, axis=1)
    apply_key = apply_ctx[cols].astype(str).agg("||".join, axis=1)
    all_keys = pd.Index(pd.concat([train_key, apply_key], ignore_index=True).unique())
    mapping = {k: i for i, k in enumerate(all_keys)}
    return train_key.map(mapping).to_numpy(dtype=int), apply_key.map(mapping).to_numpy(dtype=int), len(mapping)


def _counts_by_key(keys: np.ndarray, y: np.ndarray, n_keys: int) -> np.ndarray:
    counts = np.zeros((int(n_keys), 19), dtype=float)
    np.add.at(counts, (np.asarray(keys, dtype=int), np.asarray(y, dtype=int)), 1.0)
    return counts


def build_response_prior(train_ctx: pd.DataFrame, train_y: np.ndarray, apply_ctx: pd.DataFrame, smoothing: float = 20.0) -> np.ndarray:
    global_prior = smoothed_action_prior(train_y, smoothing=1.0)
    specs = [
        (["target_hitter_id", "phase", "lag0_actionId", "lag0_depth"], 0.45),
        (["target_hitter_id", "phase", "lag0_spinId"], 0.25),
        (["phase", "lag0_actionId", "lag0_depth"], 0.20),
        (["phase"], 0.10),
    ]
    prior = np.zeros((len(apply_ctx), 19), dtype=float)
    for cols, weight in specs:
        train_keys, apply_keys, n_keys = _factorize_combined(train_ctx, apply_ctx, cols)
        counts = _counts_by_key(train_keys, train_y, n_keys)
        prior += float(weight) * response_prior_from_counts(apply_keys, counts, global_prior, smoothing=smoothing)
    return normalize_rows_safe(prior)


def _evaluate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    iw = weighted_macro_f1(y, pred, weights)
    base_iw = weighted_macro_f1(y, anchor, weights)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "iw_action_macro_f1": float(iw),
        "iw_delta_vs_v173": float(iw - base_iw),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
    }


def _write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    raw_train = pd.read_csv("train.csv")
    raw_test = pd.read_csv("test_new.csv")
    rows = add_audit_columns(_merge_target_players(data["rows"].copy(), raw_train))
    test_rows = add_audit_columns(_merge_target_players(state["test_rows"].copy(), raw_test))
    y = rows["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    point = pd.read_csv(POINT_ANCHOR)
    server = load_sub(SERVER_ANCHOR, point["rally_uid"].astype(int).to_numpy())
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, _ = distill_v173_soft_anchor(data, v173_oof, v173_test)

    train_context_all = response_context(rows)
    test_context = response_context(test_rows)
    oof_prior = np.zeros((len(rows), 19), dtype=float)
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        oof_prior[valid] = build_response_prior(train_context_all.loc[train].reset_index(drop=True), y[train], train_context_all.loc[valid].reset_index(drop=True))
    test_prior = build_response_prior(train_context_all.reset_index(drop=True), y, test_context.reset_index(drop=True))

    weight_ctx_train = pd.DataFrame(
        {
            "prefix_bin": rows["prefix_len"].map(lambda v: "1" if int(v) <= 1 else "2" if int(v) == 2 else "3" if int(v) == 3 else "4_6" if int(v) <= 6 else "7_plus"),
            "phase": _phase_name(rows),
            "lag0_family": rows["audit_lag0_action_family"].astype(str),
            "lag0_depth": _lag_depth(rows),
        }
    )
    weight_ctx_test = pd.DataFrame(
        {
            "prefix_bin": test_rows["prefix_len"].map(lambda v: "1" if int(v) <= 1 else "2" if int(v) == 2 else "3" if int(v) == 3 else "4_6" if int(v) <= 6 else "7_plus"),
            "phase": _phase_name(test_rows),
            "lag0_family": test_rows["audit_lag0_action_family"].astype(str),
            "lag0_depth": _lag_depth(test_rows),
        }
    )
    weights = density_ratio_weights(weight_ctx_train, weight_ctx_test, ["prefix_bin", "phase", "lag0_family", "lag0_depth"])
    variants = {
        "v235_response_w0p05": 0.05,
        "v235_response_w0p10": 0.10,
        "v235_response_w0p20": 0.20,
        "v235_response_aggressive_w0p35": 0.35,
    }
    records = [_evaluate("v173_anchor", y, v173_oof, v173_oof, weights)]
    generated = []
    for name, w in variants.items():
        prob_oof = geometric_log_blend(v173_prob_oof, oof_prior, w)
        prob_test = geometric_log_blend(v173_prob_test, test_prior, w)
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = _evaluate(name, y, pred, v173_oof, weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
        rec["test_changed_rows"] = int(np.sum(test_pred != v173_test))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", prob_test)
        generated.append(_write_submission(f"submission_{name}__pv188cap5__sr121.csv", test_pred, point, server))
    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173"], ascending=[False, False])
    search.to_csv(OUTDIR / "v235_action_search.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {"verdict": verdict, "best_delta_vs_v173_anchor": best_delta, "best": search.head(10).to_dict(orient="records"), "generated": generated}
    (OUTDIR / "v235_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v235_report.md").write_text(f"# V235 Player Response Teacher\n\n- Verdict: `{verdict}`\n- Best delta vs V173: `{best_delta:.6f}`\n", encoding="utf-8")
    shutil.copy2("analysis_v235_player_conditional_response_teacher.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
