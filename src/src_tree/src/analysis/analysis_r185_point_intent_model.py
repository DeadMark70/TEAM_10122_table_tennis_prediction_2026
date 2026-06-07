"""R185 point intent model.

Point is reframed as terminal / depth / side / action-conditioned intent
instead of a flat 10-class replacement.  V173 action is used as the action
condition because it has positive public signal in the no-old setting.

Generated submissions use:
  action = V173
  point  = R185 candidate
  server = R121
No old-server labels or TTMATCH data are used.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from analysis_r1_oof_ensemble import compose_v3
from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
from analysis_r179_action_physics_hierarchy import normalize_rows_safe, phase_name, point_depth, point_side
from analysis_r184_receiver_affordance_refiner import add_affordance_columns, rebuild_v173_best_actions
from analysis_r67_r70_meta_priors import compose_v3_full_point
from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, R111_TEST, prepare_prefix_features
from baseline_lgbm import POINT_CLASSES
from baseline_v3 import apply_segmented_multipliers


OUTDIR = Path("r185_point_intent_model")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_r185_point_intent_model.py")

BASE_V173 = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
R121 = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"

CONF_GRID = [0.26, 0.30, 0.34, 0.38, 0.42]
MARGIN_GRID = [0.02, 0.04, 0.06, 0.08]
MAX_POINT_CHURN = 0.08


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


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"{path} does not align")
    return out


def one_hot(labels: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(labels), n), dtype=float)
    out[np.arange(len(labels)), np.asarray(labels, dtype=int)] = 1.0
    return out


def point_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode).astype(int)


def target_depth_id(point_id: int) -> int:
    d = point_depth(point_id)
    return max(0, d - 1)


def target_side_id(point_id: int) -> int:
    s = point_side(point_id)
    return max(0, s - 1)


def depth_label(values: np.ndarray) -> np.ndarray:
    return np.array([point_depth(v) for v in values], dtype=int)


def side_label(values: np.ndarray) -> np.ndarray:
    return np.array([point_side(v) for v in values], dtype=int)


def add_r185_columns(df: pd.DataFrame, action_labels: np.ndarray | None, *, pool: bool) -> pd.DataFrame:
    out = add_affordance_columns(df)
    if pool:
        out["r185_action"] = out["next_actionId"].astype(int)
    else:
        if action_labels is None:
            raise ValueError("action_labels required for non-pool rows")
        out["r185_action"] = np.asarray(action_labels, dtype=int)
    out["r185_action_family"] = out["r185_action"].map(lambda x: "zero" if int(x) == 0 else ("attack" if 1 <= int(x) <= 7 else ("control" if 8 <= int(x) <= 11 else ("defense" if 12 <= int(x) <= 14 else "serve"))))
    out["r185_incoming_state"] = (
        out["r184_phase"].astype(str)
        + "|a="
        + out["lag0_actionId"].astype(str)
        + "|p="
        + out["lag0_pointId"].astype(str)
        + "|s="
        + out["lag0_spinId"].astype(str)
        + "|str="
        + out["lag0_strengthId"].astype(str)
    )
    out["r185_affordance_state"] = (
        out["r184_phase"].astype(str)
        + "|fam="
        + out["r184_lag0_family"].astype(str)
        + "|depth="
        + out["r184_lag0_depth"].astype(str)
        + "|side="
        + out["r184_lag0_side"].astype(str)
    )
    return out


def global_dist(labels: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n).astype(float) + 1.0
    return counts / counts.sum()


def make_lookup(pool: pd.DataFrame, key_cols: list[str], labels: np.ndarray, n: int, alpha: float, prior: np.ndarray) -> dict[tuple, np.ndarray]:
    tmp = pool.reset_index(drop=True)
    labels = np.asarray(labels, dtype=int)
    out: dict[tuple, np.ndarray] = {}
    for key, idx in tmp.groupby(key_cols, dropna=False).groups.items():
        vals = labels[list(idx)]
        counts = np.bincount(vals, minlength=n).astype(float)
        out[key if isinstance(key, tuple) else (key,)] = normalize_rows_safe((counts + alpha * prior)[None, :])[0]
    return out


def apply_lookup(rows: pd.DataFrame, specs: list[tuple[list[str], dict[tuple, np.ndarray]]], prior: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), len(prior)), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        dist = None
        for cols, lookup in specs:
            key = tuple(getattr(row, c) for c in cols)
            dist = lookup.get(key)
            if dist is not None:
                break
        out[i] = prior if dist is None else dist
    return normalize_rows_safe(out)


def direct_point_prior(pool: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    y = pool["next_pointId"].astype(int).to_numpy()
    prior = global_dist(y, 10)
    specs = [
        (["r184_phase", "r185_action", "r185_incoming_state"], make_lookup(pool, ["r184_phase", "r185_action", "r185_incoming_state"], y, 10, 35.0, prior)),
        (["r185_action", "r185_affordance_state"], make_lookup(pool, ["r185_action", "r185_affordance_state"], y, 10, 45.0, prior)),
        (["r184_phase", "r185_action"], make_lookup(pool, ["r184_phase", "r185_action"], y, 10, 60.0, prior)),
        (["r185_action"], make_lookup(pool, ["r185_action"], y, 10, 80.0, prior)),
    ]
    return apply_lookup(rows, specs, prior)


def structured_point_prior(pool: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    y = pool["next_pointId"].astype(int).to_numpy()
    term_y = (y == 0).astype(int)
    non = y != 0
    term_prior = global_dist(term_y, 2)
    term_specs = [
        (["r185_action", "r185_incoming_state"], make_lookup(pool, ["r185_action", "r185_incoming_state"], term_y, 2, 30.0, term_prior)),
        (["r184_phase", "r185_action"], make_lookup(pool, ["r184_phase", "r185_action"], term_y, 2, 50.0, term_prior)),
        (["r185_action"], make_lookup(pool, ["r185_action"], term_y, 2, 80.0, term_prior)),
    ]
    terminal = apply_lookup(rows, term_specs, term_prior)[:, 1]

    d_y = np.array([target_depth_id(v) for v in y[non]], dtype=int)
    s_y = np.array([target_side_id(v) for v in y[non]], dtype=int)
    pool_non = pool.loc[non].reset_index(drop=True)
    d_prior = global_dist(d_y, 3)
    s_prior = global_dist(s_y, 3)
    d_specs = [
        (["r185_action", "r185_affordance_state"], make_lookup(pool_non, ["r185_action", "r185_affordance_state"], d_y, 3, 35.0, d_prior)),
        (["r184_phase", "r185_action"], make_lookup(pool_non, ["r184_phase", "r185_action"], d_y, 3, 60.0, d_prior)),
        (["r185_action"], make_lookup(pool_non, ["r185_action"], d_y, 3, 80.0, d_prior)),
    ]
    s_specs = [
        (["r185_action", "r185_affordance_state"], make_lookup(pool_non, ["r185_action", "r185_affordance_state"], s_y, 3, 35.0, s_prior)),
        (["r184_phase", "r185_action"], make_lookup(pool_non, ["r184_phase", "r185_action"], s_y, 3, 60.0, s_prior)),
        (["r185_action"], make_lookup(pool_non, ["r185_action"], s_y, 3, 80.0, s_prior)),
    ]
    depth = apply_lookup(rows, d_specs, d_prior)
    side = apply_lookup(rows, s_specs, s_prior)

    out = np.zeros((len(rows), 10), dtype=float)
    out[:, 0] = terminal
    for d in range(3):
        for s in range(3):
            pid = 1 + d * 3 + s
            out[:, pid] = (1.0 - terminal) * depth[:, d] * side[:, s]
    return normalize_rows_safe(out)


def side_affordance_prior(pool: pd.DataFrame, rows: pd.DataFrame, base_point: np.ndarray) -> np.ndarray:
    y = pool["next_pointId"].astype(int).to_numpy()
    non = y != 0
    s_y = np.array([target_side_id(v) for v in y[non]], dtype=int)
    pool_non = pool.loc[non].reset_index(drop=True)
    s_prior = global_dist(s_y, 3)
    specs = [
        (["r185_action", "r185_affordance_state"], make_lookup(pool_non, ["r185_action", "r185_affordance_state"], s_y, 3, 30.0, s_prior)),
        (["r184_phase", "r185_action", "r184_lag0_depth"], make_lookup(pool_non, ["r184_phase", "r185_action", "r184_lag0_depth"], s_y, 3, 55.0, s_prior)),
        (["r185_action"], make_lookup(pool_non, ["r185_action"], s_y, 3, 80.0, s_prior)),
    ]
    side = apply_lookup(rows, specs, s_prior)
    out = np.zeros((len(rows), 10), dtype=float)
    for i, bp in enumerate(np.asarray(base_point, dtype=int)):
        if bp == 0:
            out[i, 0] = 1.0
            continue
        d = point_depth(bp)
        for s in range(3):
            pid = 1 + (d - 1) * 3 + s
            out[i, pid] = side[i, s]
    return normalize_rows_safe(out)


def foldsafe_priors(rows: pd.DataFrame, prefix: pd.DataFrame, base_point: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: np.zeros((len(rows), 10), dtype=float) for k in ["r185a_structured", "r185b_action_direct", "r185c_side_affordance"]}
    for fold in sorted(rows["fold"].unique()):
        idx = rows.index[rows["fold"].eq(fold)].to_numpy()
        valid_matches = set(rows.loc[idx, "match"])
        pool = prefix[~prefix["match"].isin(valid_matches)].reset_index(drop=True)
        part = rows.loc[idx].reset_index(drop=True)
        out["r185a_structured"][idx] = structured_point_prior(pool, part)
        out["r185b_action_direct"][idx] = direct_point_prior(pool, part)
        out["r185c_side_affordance"][idx] = side_affordance_prior(pool, part, base_point[idx])
    return out


def full_priors(prefix: pd.DataFrame, rows: pd.DataFrame, base_point: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r185a_structured": structured_point_prior(prefix, rows),
        "r185b_action_direct": direct_point_prior(prefix, rows),
        "r185c_side_affordance": side_affordance_prior(prefix, rows, base_point),
    }


def override_from_prior(base: np.ndarray, prior: np.ndarray, conf: float, margin: float) -> np.ndarray:
    sorted_prior = np.sort(prior, axis=1)
    top = prior.argmax(axis=1).astype(int)
    top_conf = sorted_prior[:, -1]
    top_margin = sorted_prior[:, -1] - sorted_prior[:, -2]
    use = (top != base) & (top_conf >= conf) & (top_margin >= margin)
    out = base.copy()
    out[use] = top[use]
    return out


def depth_macro(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(depth_label(y), depth_label(pred), labels=[0, 1, 2, 3], average="macro", zero_division=0))


def side_macro_nonterminal(y: np.ndarray, pred: np.ndarray) -> float:
    mask = y != 0
    if not np.any(mask):
        return 0.0
    return float(f1_score(side_label(y[mask]), side_label(pred[mask]), labels=[1, 2, 3], average="macro", zero_division=0))


def per_class_subset(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rep = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    return {f"point{k}_f1": float(rep[str(k)]["f1-score"]) for k in [0, 1, 3, 4, 7, 8, 9]}


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, source: str, conf: float, margin: float) -> dict:
    rec = {
        "candidate": name,
        "source": source,
        "conf": conf,
        "margin": margin,
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "depth_macro_f1": depth_macro(y, pred),
        "side_macro_f1_nonterminal": side_macro_nonterminal(y, pred),
    }
    rec.update(per_class_subset(y, pred))
    return rec


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


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, test_prefix, _ = prepare_prefix_features()
    prefix = add_r185_columns(prefix, None, pool=True)
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)

    r111_oof = load_pickle(R111_OOF)
    r111_test = load_pickle(R111_TEST)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    tuning = r111_oof["tuning"]
    v3_oof = load_pickle("oof_proba_v3.pkl")
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)

    v173_action_oof_prob = one_hot(state["v173_pred_oof"], 19)
    v173_action_test_prob = one_hot(state["v173_pred_test"], 19)
    r119_oof = r119_oof_prior(rows, prefix, v173_action_oof_prob)
    r119_test = action_conditioned_point_prior(test_rows, prefix, v173_action_test_prob)
    local_base_prob_oof = normalize_rows_safe(0.95 * base_point_oof + 0.05 * r119_oof)
    local_base_prob_test = normalize_rows_safe(0.95 * base_point_test + 0.05 * r119_test)
    local_base_pred_oof = point_pred(rows, local_base_prob_oof, tuning)

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    test_base_point = base_sub["pointId"].astype(int).to_numpy()

    oof_priors = foldsafe_priors(rows, prefix, local_base_pred_oof)
    test_priors = full_priors(prefix, test_rows, test_base_point)
    y = rows["next_pointId"].astype(int).to_numpy()

    search_rows = [
        eval_candidate("local_r119_v173_base", y, local_base_pred_oof, local_base_pred_oof, "base", 0.0, 0.0)
    ]
    pred_store: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for source, prior in oof_priors.items():
        for conf in CONF_GRID:
            for margin in MARGIN_GRID:
                pred = override_from_prior(local_base_pred_oof, prior, conf, margin)
                rec = eval_candidate(f"{source}_c{str(conf).replace('.', 'p')}_m{str(margin).replace('.', 'p')}", y, pred, local_base_pred_oof, source, conf, margin)
                search_rows.append(rec)
                test_pred = override_from_prior(test_base_point, test_priors[source], conf, margin)
                pred_store[rec["candidate"]] = (pred, test_pred, rec)

    search = pd.DataFrame(search_rows)
    search["tier"] = np.where(search["point_churn_vs_base"].le(MAX_POINT_CHURN), "clean", "probe")
    search = search.sort_values(["tier", "point_macro_f1", "point_churn_vs_base"], ascending=[True, False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r185_search.csv", index=False)

    generated = []
    clean = search[(search["tier"].eq("clean")) & (search["source"].ne("base"))].copy()
    emitted_models: set[str] = set()
    for source in ["r185a_structured", "r185b_action_direct", "r185c_side_affordance"]:
        part = clean[clean["source"].eq(source)]
        if part.empty:
            continue
        rec = part.iloc[0].to_dict()
        name = str(rec["candidate"])
        _, test_pred, _ = pred_store[name]
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, base_sub, test_pred)
        info.update(rec)
        info["submission"] = sub_name
        generated.append(info)
        emitted_models.add(name)

    # Also emit the best clean candidate overall if it is not already included.
    if not clean.empty:
        rec = clean.iloc[0].to_dict()
        name = str(rec["candidate"])
        if name not in emitted_models:
            _, test_pred, _ = pred_store[name]
            sub_name = f"submission_{name}__v173action_r121server.csv"
            info = write_submission(sub_name, base_sub, test_pred)
            info.update(rec)
            info["submission"] = sub_name
            generated.append(info)
            emitted_models.add(name)

    report = {
        "base": search[search["source"].eq("base")].iloc[0].to_dict(),
        "best_clean": clean.head(10).to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "R185 uses V173 action as the action condition.",
            "Generated submissions keep V173 action and R121 server fixed.",
            "Point candidates are thresholded overrides from terminal/depth/side/action-conditioned priors, not full hard decoder replacements.",
            "R119 local base is reconstructed with V173 one-hot action for OOF support; test churn is measured against the submitted V173/R119/R121 point anchor.",
        ],
    }
    (OUTDIR / "r185_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r185_report.md").write_text(
        "# R185 Point Intent Model\n\n"
        "## Local Base\n\n"
        f"- Point Macro-F1: `{report['base']['point_macro_f1']:.6f}`\n"
        f"- Depth Macro-F1: `{report['base']['depth_macro_f1']:.6f}`\n"
        f"- Side Macro-F1 nonterminal: `{report['base']['side_macro_f1_nonterminal']:.6f}`\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(
            f"- `{g['upload_path']}` source `{g['source']}`, OOF point `{g['point_macro_f1']:.6f}`, churn `{g['point_churn_vs_base']:.6f}`"
            for g in generated
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r185_point_intent_model.py", SRC_DEST)
    print(json.dumps({"generated_count": len(generated), "metrics": str(OUTDIR / "r185_search.csv")}, indent=2))


if __name__ == "__main__":
    main()
