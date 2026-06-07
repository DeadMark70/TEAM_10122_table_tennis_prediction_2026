"""V195 distribution-matched V188-style point GRU.

V194 found that V188/V193 trained on a short, receive-heavy R111 valid_meta
pool while test_new is more rally/long/attack-like.  V195 keeps the V188 GRU
point-intent architecture and changes only training examples/sampling:

  A. R111 pool + test-likeness importance sampling
  B. R111 pool + stratified resampling
  C. full generated prefix pool + test-likeness importance sampling

Submissions, if generated, keep action=V173 and server=R121.  TTMATCH is not
read.
"""

from __future__ import annotations

import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import analysis_v160_v163_task_pretrain_distill as v160
from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot, point_pred
from analysis_r187_point_intent_student import add_r186_priors
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, prepare_prefix_features
from analysis_v188_point_intent_gru import (
    BATCH_SIZE,
    DEVICE,
    LOSS_SETTINGS,
    MAX_SEQ_LEN,
    R186_TEST,
    R186_TRAIN,
    STROKE_COLS,
    PointIntentGRU,
    StrokeDataset,
    batch_loss,
    build_padded_stroke_tensor,
    capped_residual_labels,
    collate,
    load_pickle,
    predict_proba,
    raw_groups,
    row_log_blend,
    sequences_for_rows,
    set_seed,
    static_matrix,
    teacher_matrix,
)
from analysis_v194_train_test_split_distribution_audit import add_audit_columns, prefix_bin
from baseline_lgbm import POINT_CLASSES


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


OUTDIR = Path("v195_distribution_matched_point_gru")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v195_distribution_matched_point_gru.py")

MATCH_COLS = [
    "v195_prefix_bin",
    "r184_phase",
    "r184_lag0_depth",
    "r184_lag0_family",
    "lag0_spinId",
    "lag0_strengthId",
]
ALPHAS = [0.05, 0.075]
CHURN_CAPS = [0.02, 0.03, 0.05]
EPOCHS = 10
PATIENCE = 2


def key_frame(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def distribution_match_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    *,
    clip: tuple[float, float] = (0.20, 5.0),
    smooth: float = 3.0,
) -> np.ndarray:
    train_key = key_frame(train, cols)
    test_key = key_frame(test, cols)
    train_counts = train_key.value_counts()
    test_counts = test_key.value_counts()
    keys = sorted(set(train_counts.index) | set(test_counts.index))
    k = max(len(keys), 1)
    train_total = float(len(train_key) + smooth * k)
    test_total = float(len(test_key) + smooth * k)
    ratio = {}
    for key in keys:
        train_share = (float(train_counts.get(key, 0)) + smooth) / train_total
        test_share = (float(test_counts.get(key, 0)) + smooth) / test_total
        ratio[key] = test_share / max(train_share, 1e-12)
    w = train_key.map(ratio).to_numpy(dtype=float)
    w = np.clip(w, clip[0], clip[1])
    return w / max(float(w.mean()), 1e-12)


def stratified_resample_indices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    w = distribution_match_weights(train, test, cols, clip=(0.05, 20.0), smooth=1.0)
    p = w / w.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(len(train)), size=int(n), replace=True, p=p)


def add_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_audit_columns(df)
    out["v195_prefix_bin"] = out["prefix_len"].map(prefix_bin)
    return out


def assign_match_folds(prefix: pd.DataFrame, oof_rows: pd.DataFrame) -> pd.DataFrame:
    match_fold = oof_rows[["match", "fold"]].drop_duplicates()
    if match_fold["match"].duplicated().any():
        raise ValueError("A match appears in multiple folds")
    out = prefix.merge(match_fold, on="match", how="left", validate="many_to_one")
    if out["fold"].isna().any():
        missing = out.loc[out["fold"].isna(), "match"].unique()[:10]
        raise ValueError(f"Could not assign folds for matches: {missing}")
    out["fold"] = out["fold"].astype(int)
    return out


def train_model_sampled(
    train_ds: StrokeDataset,
    valid_ds: StrokeDataset,
    vocab_sizes: list[int],
    static_dim: int,
    loss_weights: dict[str, float],
    seed: int,
    sample_weights: np.ndarray | None = None,
) -> tuple[PointIntentGRU, float]:
    set_seed(seed)
    model = PointIntentGRU(vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch.strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
                loss = batch_loss(out, batch.point.to(DEVICE), batch.teacher.to(DEVICE), loss_weights)
                val_loss += float(loss.item()) * len(batch.point)
                n += len(batch.point)
        val_loss /= max(n, 1)
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    score = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_score = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "point_macro_f1": score,
        "delta_vs_base": score - base_score,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "pred_point0_rate": float(np.mean(pred == 0)),
    }
    for k in [0, 1, 3, 4, 7, 8, 9]:
        rec[f"point{k}_f1"] = float(rep[str(k)]["f1-score"])
    rec.update(meta)
    return rec


def distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=10)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def prepare_data() -> dict:
    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    rows = add_r186_priors(rows, pd.read_csv(R186_TRAIN))
    test_rows = add_r186_priors(test_rows, pd.read_csv(R186_TEST))
    rows = add_match_columns(rows)
    test_rows = add_match_columns(test_rows)

    r111_oof = load_pickle(R111_OOF)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    v3_oof = load_pickle("oof_proba_v3.pkl")
    tuning = r111_oof["tuning"]
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)
    prefix_train = add_r185_columns(prefix, None, pool=True)
    v173_action_oof_prob = one_hot(state["v173_pred_oof"], 19)
    v173_action_test_prob = one_hot(state["v173_pred_test"], 19)
    r119_oof = r119_oof_prior(rows, prefix_train, v173_action_oof_prob)
    r119_test = action_conditioned_point_prior(test_rows, prefix_train, v173_action_test_prob)
    local_base_prob_oof = normalize_rows_safe(0.95 * base_point_oof + 0.05 * r119_oof)
    local_base_prob_test = normalize_rows_safe(0.95 * base_point_test + 0.05 * r119_test)
    local_base_pred_oof = point_pred(rows, local_base_prob_oof, tuning)

    full_pool = assign_match_folds(add_r185_columns(prefix, None, pool=True), rows)
    full_pool = add_r186_priors(full_pool, pd.read_csv(R186_TRAIN))
    full_pool = add_match_columns(full_pool)
    pool_action_prior, pool_point_prior = v160.foldsafe_internal_priors(prefix, full_pool)

    train_groups = raw_groups("train.csv")
    test_groups = raw_groups("test_new.csv")
    oof_seq, oof_len = build_padded_stroke_tensor(sequences_for_rows(rows, train_groups), MAX_SEQ_LEN, 0)
    full_seq, full_len = build_padded_stroke_tensor(sequences_for_rows(full_pool, train_groups), MAX_SEQ_LEN, 0)
    test_seq, test_len = build_padded_stroke_tensor(sequences_for_rows(test_rows, test_groups), MAX_SEQ_LEN, 0)
    vocab_sizes = [
        int(max(oof_seq[:, :, i].max(), full_seq[:, :, i].max(), test_seq[:, :, i].max()) + 1)
        for i in range(len(STROKE_COLS))
    ]

    x_oof, oof_stats = static_matrix(rows, v173_action_oof_prob, local_base_prob_oof)
    x_test_oofstats, _ = static_matrix(test_rows, v173_action_test_prob, local_base_prob_test, oof_stats)
    x_full, full_stats = static_matrix(full_pool, pool_action_prior, pool_point_prior)
    x_test_fullstats, _ = static_matrix(test_rows, v173_action_test_prob, local_base_prob_test, full_stats)

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    return {
        "rows": rows,
        "test_rows": test_rows,
        "full_pool": full_pool,
        "oof_seq": oof_seq,
        "oof_len": oof_len,
        "full_seq": full_seq,
        "full_len": full_len,
        "test_seq": test_seq,
        "test_len": test_len,
        "x_oof": x_oof,
        "x_test_oofstats": x_test_oofstats,
        "x_full": x_full,
        "x_test_fullstats": x_test_fullstats,
        "teacher_oof": teacher_matrix(rows),
        "teacher_test": teacher_matrix(test_rows),
        "teacher_full": teacher_matrix(full_pool),
        "y_oof": rows["next_pointId"].astype(int).to_numpy(),
        "y_full": full_pool["next_pointId"].astype(int).to_numpy(),
        "base_prob_oof": local_base_prob_oof,
        "base_prob_test": local_base_prob_test,
        "base_pred_oof": local_base_pred_oof,
        "test_base_point": base_sub["pointId"].astype(int).to_numpy(),
        "base_sub": base_sub,
        "vocab_sizes": vocab_sizes,
    }


def run_scheme(name: str, data: dict, mode: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = data["rows"]
    full_pool = data["full_pool"]
    test_rows = data["test_rows"]
    y = data["y_oof"]
    weights = dict(LOSS_SETTINGS)["r186_w005"]
    oof_prob = np.zeros((len(rows), 10), dtype=float)
    fold_test_probs = []
    fold_records = []
    if mode in {"importance", "stratified"}:
        train_rows_all = rows
        train_seq_all = data["oof_seq"]
        train_len_all = data["oof_len"]
        train_static_all = data["x_oof"]
        train_teacher_all = data["teacher_oof"]
        train_y_all = y
        train_weight_all = distribution_match_weights(train_rows_all, test_rows, MATCH_COLS)
        test_static = data["x_test_oofstats"]
    elif mode == "full_importance":
        train_rows_all = full_pool
        train_seq_all = data["full_seq"]
        train_len_all = data["full_len"]
        train_static_all = data["x_full"]
        train_teacher_all = data["teacher_full"]
        train_y_all = data["y_full"]
        train_weight_all = distribution_match_weights(train_rows_all, test_rows, MATCH_COLS)
        test_static = data["x_test_fullstats"]
    else:
        raise ValueError(mode)

    test_ds = StrokeDataset(
        data["test_seq"],
        data["test_len"],
        test_static,
        np.zeros(len(data["test_seq"]), dtype=np.int64),
        data["teacher_test"],
    )
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_mask = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        valid_ds = StrokeDataset(
            data["oof_seq"][valid_mask],
            data["oof_len"][valid_mask],
            data["x_oof"][valid_mask],
            y[valid_mask],
            data["teacher_oof"][valid_mask],
        )
        train_mask = ~train_rows_all["fold"].astype(int).eq(int(fold)).to_numpy()
        train_idx = np.where(train_mask)[0]
        sample_weights = None
        if mode == "stratified":
            local = train_rows_all.iloc[train_idx].reset_index(drop=True)
            sampled_local = stratified_resample_indices(local, test_rows, MATCH_COLS, n=len(train_idx), seed=1950 + int(fold))
            train_idx = train_idx[sampled_local]
        elif mode in {"importance", "full_importance"}:
            sample_weights = train_weight_all[train_idx]
        train_ds = StrokeDataset(
            train_seq_all[train_idx],
            train_len_all[train_idx],
            train_static_all[train_idx],
            train_y_all[train_idx],
            train_teacher_all[train_idx],
        )
        model, val_loss = train_model_sampled(
            train_ds,
            valid_ds,
            data["vocab_sizes"],
            train_static_all.shape[1],
            weights,
            1950 + int(fold),
            sample_weights=sample_weights,
        )
        oof_prob[valid_mask] = predict_proba(model, valid_ds)
        fold_test_probs.append(predict_proba(model, test_ds))
        pred = oof_prob[valid_mask].argmax(axis=1)
        fold_records.append(
            {
                "scheme": name,
                "fold": int(fold),
                "train_rows": int(len(train_idx)),
                "val_loss": float(val_loss),
                "raw_point_macro_f1": float(f1_score(y[valid_mask], pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
                "raw_point0_rate": float(np.mean(pred == 0)),
            }
        )
    return normalize_rows_safe(oof_prob), normalize_rows_safe(np.mean(fold_test_probs, axis=0)), fold_records


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    random.seed(195)
    np.random.seed(195)
    set_seed(195)
    data = prepare_data()
    y = data["y_oof"]
    base = data["base_pred_oof"]
    test_base = data["test_base_point"]
    search_records = [eval_candidate("local_v173_r119_base", y, base, base, {"scheme": "base"})]
    fold_records = []
    pred_store: dict[str, np.ndarray] = {}

    schemes = [
        ("v195a_r111_importance", "importance"),
        ("v195b_r111_stratified", "stratified"),
        ("v195c_fullpool_importance", "full_importance"),
    ]
    for scheme, mode in schemes:
        oof_prob, test_prob, folds = run_scheme(scheme, data, mode)
        fold_records.extend(folds)
        raw_oof = oof_prob.argmax(axis=1).astype(int)
        raw_test = test_prob.argmax(axis=1).astype(int)
        search_records.append(
            eval_candidate(
                f"{scheme}_raw_argmax",
                y,
                raw_oof,
                base,
                {
                    "scheme": scheme,
                    "mode": mode,
                    "alpha": 1.0,
                    "cap": 1.0,
                    "test_raw_point0_rate": float(np.mean(raw_test == 0)),
                    "test_raw_distribution": json.dumps(distribution(raw_test), sort_keys=True),
                    "test_raw_p0_mean": float(test_prob[:, 0].mean()),
                },
            )
        )
        for alpha in ALPHAS:
            blended = row_log_blend(data["base_prob_oof"], oof_prob, alpha)
            blended_test = row_log_blend(data["base_prob_test"], test_prob, alpha)
            for cap in CHURN_CAPS:
                pred, _ = capped_residual_labels(base, blended, cap)
                test_pred, test_changed = capped_residual_labels(test_base, blended_test, cap)
                name = f"{scheme}_a{str(alpha).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}"
                rec = eval_candidate(name, y, pred, base, {"scheme": scheme, "mode": mode, "alpha": alpha, "cap": cap})
                rec["test_churn_vs_v173_r119"] = float(np.mean(test_pred != test_base))
                rec["test_changed_rows"] = int(np.sum(test_changed))
                rec["test_distribution"] = json.dumps(distribution(test_pred), sort_keys=True)
                search_records.append(rec)
                pred_store[name] = test_pred

    search = pd.DataFrame(search_records)
    search["tier"] = np.select(
        [search["point_churn_vs_base"].le(0.02), search["point_churn_vs_base"].le(0.05)],
        ["clean", "probe"],
        default="high_churn",
    )
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v195_search.csv", index=False)
    pd.DataFrame(fold_records).to_csv(OUTDIR / "v195_fold_metrics.csv", index=False)

    generated = []
    emitted = set()
    positive = search[(search["candidate"].str.startswith("v195")) & search["delta_vs_base"].gt(0) & search["tier"].isin(["clean", "probe"])]
    for tier in ["clean", "probe"]:
        part = positive[positive["tier"].eq(tier)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        if name in emitted or name not in pred_store:
            continue
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        info.update(rec)
        info["submission"] = sub_name
        generated.append(info)
        emitted.add(name)

    raw_rows = search[search["candidate"].str.endswith("_raw_argmax")].to_dict(orient="records")
    report = {
        "verdict": "CANDIDATES_GENERATED" if generated else "NO_POSITIVE_CANDIDATE",
        "device": DEVICE,
        "base": search[search["candidate"].eq("local_v173_r119_base")].iloc[0].to_dict(),
        "raw_rows": raw_rows,
        "best_clean": search[search["tier"].eq("clean")].head(12).to_dict(orient="records"),
        "best_probe": search[search["tier"].eq("probe")].head(12).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "V195 changes train sampling/pool only; the V188 GRU architecture and heads are unchanged.",
            "A/B use the R111 aligned pool with distribution-matched sampling; C uses the full generated prefix pool.",
            "Full-pool training uses fold-safe internal action/point priors as train-time static priors because V173 OOF probabilities are only available for the R111 pool.",
            "Submissions keep action=V173 and server=R121.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v195_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v195_report.md").write_text(
        "# V195 Distribution-Matched Point GRU\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Device: `{DEVICE}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Raw Test Stability\n\n"
        + "\n".join(
            f"- `{r['candidate']}` OOF `{r['point_macro_f1']:.6f}`, test raw p0 `{r.get('test_raw_point0_rate', float('nan')):.6f}`, dist `{r.get('test_raw_distribution', '{}')}`"
            for r in raw_rows
        )
        + "\n\n## Generated\n\n"
        + ("\n".join(f"- `{g['upload_path']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_v173_r119']:.6f}`" for g in generated) or "- none")
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v195_distribution_matched_point_gru.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated_count": len(generated), "search": str(OUTDIR / "v195_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
