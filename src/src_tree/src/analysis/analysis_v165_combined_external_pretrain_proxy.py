"""V165 combined external pretraining proxy.

This script tests whether CoachAI badminton sequence data can be combined with
previous external priors:

- OpenTTGames coarse action-family prior.
- DeepMind / TT-MatchDynamics physics landing priors.
- AICUP internal task-adaptive transition priors.
- CoachAI coarse shot-family and landing-grid transition priors.

The implementation is intentionally conservative:
- It does not append external rows to AICUP training.
- It does not map CoachAI badminton shot labels directly to AICUP actionId.
- It uses only low-dimensional family/grid priors and low-weight logit blends.
- It does not use old-test server labels.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parent
SRC_ANALYSIS = ROOT_DIR / "src" / "analysis"
for p in [ROOT_DIR, SRC_ANALYSIS]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from analysis_r116_r119_point_server import apply_predictions  # noqa: E402
from analysis_r151b_r154_physics_prior_integration import blend_nonterminal_point, point_pred  # noqa: E402
from analysis_r155_r156_external_pretrain_priors import (  # noqa: E402
    FAMILY_TO_ACTIONS,
    FAMILIES,
    OPEN_EVENTS,
    blend_action,
    estimate_opentt_aux_priors,
    opentt_action_prior,
)
from analysis_r157_r158_physics_pretraining_smoke import (  # noqa: E402
    build_external_prototypes,
    build_mpm_tables,
    load_external_states,
    mpm_prior_for_rows,
    prototype_prior_for_rows,
)
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features  # noqa: E402
from analysis_v160_v163_task_pretrain_distill import (  # noqa: E402
    action_pred,
    build_test_internal_prefixes,
    ensure_fold,
    foldsafe_internal_priors,
    full_internal_priors,
    log_blend,
)
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, validate_raw_data  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


OUTDIR = Path("v165_combined_external_pretrain_proxy")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
COACHAI = Path("external_data") / "CoachAI-Projects-main"

R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R121_MIN = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R154_BEST = UPLOAD_DIR / "submission_r154_md_mirror_a0p01_r67_anchor.csv"
R119_POINT = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R101_OOF = Path("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
R101_TEST = Path("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
R111_OOF = Path("r111_remaining_moe_gru/oof_proba_r111.pkl")
R111_TEST = Path("r111_remaining_moe_gru/test_proba_r111.pkl")

GRID_ORDER = [
    "left_near",
    "middle_near",
    "right_near",
    "left_mid",
    "middle_mid",
    "right_mid",
    "left_far",
    "middle_far",
    "right_far",
]

GRID_TO_POINT_DIRECT = {
    "left_near": 1,
    "middle_near": 2,
    "right_near": 3,
    "left_mid": 4,
    "middle_mid": 5,
    "right_mid": 6,
    "left_far": 7,
    "middle_far": 8,
    "right_far": 9,
}
GRID_TO_POINT_MIRROR = {
    "left_near": 3,
    "middle_near": 2,
    "right_near": 1,
    "left_mid": 6,
    "middle_mid": 5,
    "right_mid": 4,
    "left_far": 9,
    "middle_far": 8,
    "right_far": 7,
}


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def phase_from_round(n: int | float) -> str:
    try:
        n = int(n)
    except Exception:
        return "rally"
    if n <= 1:
        return "receive"
    if n == 2:
        return "third_ball"
    if n == 3:
        return "fourth_ball"
    return "rally"


def phase_from_prefix_len(prefix_len: int | float) -> str:
    return phase_from_round(prefix_len)


def aicup_action_family(action_id: int) -> str:
    action_id = int(action_id)
    if action_id in [15, 16, 17, 18]:
        return "serve"
    if action_id in [1, 2, 3, 4, 5, 6, 7]:
        return "attack"
    if action_id in [8, 9, 10, 11]:
        return "control"
    if action_id in [12, 13, 14]:
        return "defensive"
    return "unknown"


COACHAI_TYPE_TO_FAMILY = {
    # Track 2 English labels.
    "short service": "serve",
    "long service": "serve",
    "smash": "attack",
    "drive": "attack",
    "push/rush": "attack",
    "net shot": "control",
    "drop": "control",
    "clear": "defensive",
    "lob": "defensive",
    "defensive shot": "defensive",
    # Raw ShuttleSet Chinese labels.
    "發短球": "serve",
    "發長球": "serve",
    "殺球": "attack",
    "點扣": "attack",
    "撲球": "attack",
    "推球": "attack",
    "平球": "attack",
    "後場抽平球": "attack",
    "小平球": "attack",
    "放小球": "control",
    "擋小球": "control",
    "勾球": "control",
    "切球": "control",
    "過度切球": "control",
    "挑球": "defensive",
    "長球": "defensive",
    "防守回挑": "defensive",
    "防守回抽": "defensive",
    "未知球種": "unknown",
}


def side_from_quantile(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    q1, q2 = vals.quantile([1 / 3, 2 / 3])
    return pd.cut(vals, [-np.inf, q1, q2, np.inf], labels=["left", "middle", "right"]).astype(str)


def depth_from_quantile(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    q1, q2 = vals.quantile([1 / 3, 2 / 3])
    return pd.cut(vals, [-np.inf, q1, q2, np.inf], labels=["near", "mid", "far"]).astype(str)


def point_to_grid(point: int) -> str:
    point = int(point)
    for grid, p in GRID_TO_POINT_DIRECT.items():
        if p == point:
            return grid
    return "unknown"


def grid_distribution_to_point(counter: Counter, mirror: bool = False, alpha: float = 1.0) -> np.ndarray:
    mapping = GRID_TO_POINT_MIRROR if mirror else GRID_TO_POINT_DIRECT
    vec = np.full(9, alpha, dtype=float)
    for grid, cnt in counter.items():
        p = mapping.get(str(grid))
        if p is not None:
            vec[p - 1] += float(cnt)
    return vec / vec.sum()


def canonicalize_coachai_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    out["family"] = out["type"].astype(str).map(COACHAI_TYPE_TO_FAMILY).fillna("unknown")
    out["phase"] = out["ball_round"].map(phase_from_round)
    out["landing_side"] = side_from_quantile(out["landing_x"])
    out["landing_depth"] = depth_from_quantile(out["landing_y"])
    out["grid"] = out["landing_side"] + "_" + out["landing_depth"]
    return out


def load_coachai_sequences(max_raw_files: int | None = None) -> pd.DataFrame:
    frames = []
    track2 = COACHAI / "CoachAI-Challenge-IJCAI2023" / "Track 2_ Stroke Forecasting" / "data" / "train.csv"
    if track2.exists():
        frames.append(canonicalize_coachai_df(pd.read_csv(track2, low_memory=False), "coachai_track2_train"))

    raw_root = COACHAI / "CoachAI-Challenge-IJCAI2023" / "ShuttleSet22" / "set"
    raw_files = sorted(
        p for p in raw_root.rglob("*.csv") if p.name.lower() not in {"match.csv", "homography.csv"}
    )
    if max_raw_files is not None:
        raw_files = raw_files[:max_raw_files]
    raw_frames = []
    for path in raw_files:
        df = pd.read_csv(path, low_memory=False)
        df["_file_key"] = path.parent.name + "/" + path.name
        raw_frames.append(df)
    if raw_frames:
        frames.append(canonicalize_coachai_df(pd.concat(raw_frames, ignore_index=True), "coachai_shuttleset22_raw"))

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["_seq_key"] = (
        data["source"].astype(str)
        + "|"
        + data.get("match_id", data.get("_file_key", "unknown")).astype(str)
        + "|"
        + data.get("set", 0).astype(str)
        + "|"
        + data["rally"].astype(str)
    )
    data["ball_round"] = pd.to_numeric(data["ball_round"], errors="coerce")
    data = data.dropna(subset=["ball_round"]).sort_values(["_seq_key", "ball_round"]).reset_index(drop=True)
    return data


def build_coachai_transition_priors(data: pd.DataFrame, alpha: float = 25.0) -> tuple[dict, pd.DataFrame]:
    rows = []
    for _, g in data.groupby("_seq_key", sort=False):
        g = g.sort_values("ball_round")
        if len(g) < 2:
            continue
        cur = g.iloc[:-1].copy()
        nxt = g.iloc[1:].copy()
        cur["next_family"] = nxt["family"].to_numpy()
        cur["next_grid"] = nxt["grid"].to_numpy()
        rows.append(cur[["source", "phase", "family", "grid", "next_family", "next_grid"]])
    trans = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    fam_global_counts = Counter(trans["next_family"].astype(str))
    fam_global = smooth_counter_to_vec(fam_global_counts, FAMILIES, alpha=alpha)
    grid_global_counts = Counter(trans["next_grid"].astype(str))
    grid_global = smooth_counter_to_vec(grid_global_counts, GRID_ORDER, alpha=alpha)

    fam_phase = {}
    fam_phase_cur = {}
    grid_phase = {}
    grid_phase_cur = {}
    for key, group in trans.groupby(["phase"], dropna=False):
        key = key if isinstance(key, str) else key[0]
        fam_phase[key] = smooth_counter_to_vec(Counter(group["next_family"].astype(str)), FAMILIES, fam_global, alpha)
        grid_phase[key] = smooth_counter_to_vec(Counter(group["next_grid"].astype(str)), GRID_ORDER, grid_global, alpha)
    for key, group in trans.groupby(["phase", "family"], dropna=False):
        fam_phase_cur[key] = smooth_counter_to_vec(Counter(group["next_family"].astype(str)), FAMILIES, fam_global, alpha)
    for key, group in trans.groupby(["phase", "grid"], dropna=False):
        grid_phase_cur[key] = smooth_counter_to_vec(Counter(group["next_grid"].astype(str)), GRID_ORDER, grid_global, alpha)

    stats = (
        trans.groupby(["source", "phase", "next_family"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source", "phase", "rows"], ascending=[True, True, False])
    )
    priors = {
        "rows": int(len(trans)),
        "sequences": int(data["_seq_key"].nunique()),
        "families": FAMILIES,
        "grids": GRID_ORDER,
        "fam_global": fam_global,
        "grid_global": grid_global,
        "fam_phase": fam_phase,
        "fam_phase_cur": fam_phase_cur,
        "grid_phase": grid_phase,
        "grid_phase_cur": grid_phase_cur,
    }
    return priors, stats


def smooth_counter_to_vec(counter: Counter, classes: list[str], base: np.ndarray | None = None, alpha: float = 25.0) -> np.ndarray:
    if base is None:
        base = np.ones(len(classes), dtype=float) / len(classes)
    counts = np.array([float(counter.get(c, 0.0)) for c in classes], dtype=float)
    return (counts + alpha * base) / (counts.sum() + alpha)


def action_prior_from_family_prior(rows: pd.DataFrame, fam_prior: np.ndarray, train_labels: pd.Series) -> np.ndarray:
    within = {}
    y = train_labels.astype(int).to_numpy()
    for fam in FAMILIES:
        actions = FAMILY_TO_ACTIONS.get(fam, [0])
        vals = np.array([np.sum(y == a) + 1.0 for a in actions], dtype=float)
        within[fam] = vals / vals.sum()
    out = np.zeros((len(rows), len(ACTION_CLASSES)), dtype=float)
    for i in range(len(rows)):
        for j, fam in enumerate(FAMILIES):
            actions = FAMILY_TO_ACTIONS.get(fam, [0])
            for a, p in zip(actions, within[fam]):
                out[i, ACTION_CLASSES.index(a)] += fam_prior[i, j] * p
    return normalize_rows(out)


def coachai_family_prior_for_rows(rows: pd.DataFrame, priors: dict, train_labels: pd.Series) -> np.ndarray:
    fam_vecs = np.zeros((len(rows), len(FAMILIES)), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        phase = phase_from_prefix_len(getattr(row, "prefix_len"))
        cur_family = aicup_action_family(getattr(row, "lag0_actionId"))
        fam_vecs[i] = priors["fam_phase_cur"].get((phase, cur_family), priors["fam_phase"].get(phase, priors["fam_global"]))
    return action_prior_from_family_prior(rows, fam_vecs, train_labels)


def coachai_grid_prior_for_rows(rows: pd.DataFrame, priors: dict, mirror_mode: str = "avg") -> np.ndarray:
    grid_vecs = np.zeros((len(rows), len(GRID_ORDER)), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        phase = phase_from_prefix_len(getattr(row, "prefix_len"))
        grid = point_to_grid(getattr(row, "lag0_pointId"))
        grid_vecs[i] = priors["grid_phase_cur"].get((phase, grid), priors["grid_phase"].get(phase, priors["grid_global"]))

    out_direct = np.zeros((len(rows), 9), dtype=float)
    out_mirror = np.zeros((len(rows), 9), dtype=float)
    for i, vec in enumerate(grid_vecs):
        counter = Counter({g: float(vec[j]) for j, g in enumerate(GRID_ORDER)})
        out_direct[i] = grid_distribution_to_point(counter, mirror=False, alpha=0.0)
        out_mirror[i] = grid_distribution_to_point(counter, mirror=True, alpha=0.0)
    if mirror_mode == "direct":
        return normalize_rows(out_direct)
    if mirror_mode == "mirror":
        return normalize_rows(out_mirror)
    return normalize_rows(0.5 * out_direct + 0.5 * out_mirror)


def eval_action(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict) -> dict:
    y = meta["next_actionId"].astype(int).to_numpy()
    pred = action_pred(meta, prob, tuning)
    base = action_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)),
        "action_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=ACTION_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 5, 8, 9, 12, 14]:
        rec[f"f1_action_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_action_{k}"] = int(np.sum(pred == k))
    rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    base = point_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 1, 3, 5, 7, 8, 9]:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_point_{k}"] = int(np.sum(pred == k))
    rec.update(extra)
    return rec


def load_submission(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, rally_uids: np.ndarray, action: np.ndarray, point: np.ndarray, server: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(
        {
            "rally_uid": rally_uids.astype(int),
            "actionId": action.astype(int),
            "pointId": point.astype(int),
            "serverGetPoint": np.round(np.clip(server.astype(float), 1e-6, 1.0 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"candidate": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    _, _, prefix, test_prefix, _ = prepare_prefix_features()

    r111_oof = load_pickle(R111_OOF)
    r111_test = load_pickle(R111_TEST)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)

    tuning: GrUTuning = r111_oof["tuning"]
    valid_meta = ensure_fold(r111_oof["valid_meta"])
    rows = align_prefix_meta(valid_meta, prefix).reset_index(drop=True)
    test_rows = test_prefix.reset_index(drop=True)
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])
    base_point_oof = normalize_rows(0.50 * r111_oof["gru_point"] + 0.50 * r101_oof["gru_point"])
    base_point_test = normalize_rows(0.50 * r111_test["gru_point"] + 0.50 * r101_test["gru_point"])
    base_server_oof = 0.5 * r111_oof["gru_server"] + 0.5 * r101_oof["gru_server"]

    base_action_metric = float(
        f1_score(rows["next_actionId"].astype(int), action_pred(rows, base_action_oof, tuning), labels=ACTION_CLASSES, average="macro", zero_division=0)
    )
    base_point_metric = float(
        f1_score(rows["next_pointId"].astype(int), point_pred(rows, base_point_oof, tuning), labels=POINT_CLASSES, average="macro", zero_division=0)
    )
    base_server_auc = float(roc_auc_score(rows["serverGetPoint"].astype(int), base_server_oof))

    # AICUP internal priors.
    internal_action_oof, internal_point_oof = foldsafe_internal_priors(prefix, rows)
    test_internal = build_test_internal_prefixes(test_raw)
    internal_action_test, internal_point_test = full_internal_priors(prefix, test_rows, test_internal)

    # Previous external priors.
    if OPEN_EVENTS.exists():
        opentt_priors, opentt_table = estimate_opentt_aux_priors(pd.read_csv(OPEN_EVENTS))
        opentt_table.to_csv(OUTDIR / "v165_opentt_prior_table.csv", index=False)
        opentt_action_oof = opentt_action_prior(rows, base_action_oof, opentt_priors)
        opentt_action_test = opentt_action_prior(test_rows, base_action_test, opentt_priors)
    else:
        opentt_priors = {"segments_count": 0, "transition_rows": 0}
        opentt_action_oof = base_action_oof.copy()
        opentt_action_test = base_action_test.copy()

    dm, md = load_external_states()
    mpm_tables, mpm_global, _ = build_mpm_tables(dm, mirror=True)
    mpm_oof = mpm_prior_for_rows(rows, mpm_tables, mpm_global)
    mpm_test = mpm_prior_for_rows(test_rows, mpm_tables, mpm_global)
    proto = build_external_prototypes(dm, md, mirror=False)
    proto_oof = prototype_prior_for_rows(rows, proto, tau=0.35, k=250)
    proto_test = prototype_prior_for_rows(test_rows, proto, tau=0.35, k=250)

    # New CoachAI priors.
    coachai_data = load_coachai_sequences()
    coachai_priors, coachai_stats = build_coachai_transition_priors(coachai_data)
    coachai_stats.to_csv(OUTDIR / "v165_coachai_transition_family_stats.csv", index=False)
    coachai_family_oof = coachai_family_prior_for_rows(rows, coachai_priors, prefix["next_actionId"])
    coachai_family_test = coachai_family_prior_for_rows(test_rows, coachai_priors, prefix["next_actionId"])
    coachai_point_oof = {
        mode: coachai_grid_prior_for_rows(rows, coachai_priors, mode) for mode in ["direct", "mirror", "avg"]
    }
    coachai_point_test = {
        mode: coachai_grid_prior_for_rows(test_rows, coachai_priors, mode) for mode in ["direct", "mirror", "avg"]
    }

    # Action: internal + OpenTT + CoachAI family.
    action_rows = []
    action_probs: dict[str, dict[str, np.ndarray]] = {}
    for ai in [0.0, 0.0025, 0.005, 0.01, 0.02]:
        a_oof = base_action_oof if ai == 0 else log_blend(base_action_oof, internal_action_oof, ai)
        a_test = base_action_test if ai == 0 else log_blend(base_action_test, internal_action_test, ai)
        for ao in [0.0, 0.005, 0.01, 0.02, 0.03]:
            ao_oof = a_oof if ao == 0 else blend_action(a_oof, opentt_action_oof, ao)
            ao_test = a_test if ao == 0 else blend_action(a_test, opentt_action_test, ao)
            for ac in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03]:
                if ai == 0 and ao == 0 and ac == 0:
                    continue
                prob = ao_oof if ac == 0 else blend_action(ao_oof, coachai_family_oof, ac)
                test_prob = ao_test if ac == 0 else blend_action(ao_test, coachai_family_test, ac)
                name = f"v165_action_i{clean_float(ai)}_op{clean_float(ao)}_ca{clean_float(ac)}"
                rec = eval_action(rows, prob, base_action_oof, tuning, name, {"alpha_internal": ai, "alpha_opentt": ao, "alpha_coachai": ac})
                action_rows.append(rec)
                action_probs[name] = {"oof": prob, "test": test_prob}
    action_search = pd.DataFrame(action_rows).sort_values("action_macro_f1", ascending=False).reset_index(drop=True)
    action_search.to_csv(OUTDIR / "v165_combined_action_search.csv", index=False)

    # Point: internal + physics + CoachAI grid. Preserve point0 and only refine 1..9.
    point_rows = []
    point_probs: dict[str, dict[str, np.ndarray]] = {}
    for ai in [0.0, 0.0025, 0.005, 0.01]:
        p_oof = base_point_oof if ai == 0 else blend_nonterminal_point(base_point_oof, internal_point_oof[:, 1:10], ai)
        p_test = base_point_test if ai == 0 else blend_nonterminal_point(base_point_test, internal_point_test[:, 1:10], ai)
        for am in [0.0, 0.0025, 0.005, 0.01]:
            pm_oof = p_oof if am == 0 else blend_nonterminal_point(p_oof, mpm_oof, am)
            pm_test = p_test if am == 0 else blend_nonterminal_point(p_test, mpm_test, am)
            for ap in [0.0, 0.001, 0.0025, 0.005]:
                pp_oof = pm_oof if ap == 0 else blend_nonterminal_point(pm_oof, proto_oof, ap)
                pp_test = pm_test if ap == 0 else blend_nonterminal_point(pm_test, proto_test, ap)
                for mode in ["direct", "mirror", "avg"]:
                    for ac in [0.0, 0.001, 0.0025, 0.005, 0.01]:
                        if ai == 0 and am == 0 and ap == 0 and ac == 0:
                            continue
                        prob = pp_oof if ac == 0 else blend_nonterminal_point(pp_oof, coachai_point_oof[mode], ac)
                        test_prob = pp_test if ac == 0 else blend_nonterminal_point(pp_test, coachai_point_test[mode], ac)
                        name = (
                            f"v165_point_i{clean_float(ai)}_m{clean_float(am)}_p{clean_float(ap)}"
                            f"_ca{mode}{clean_float(ac)}"
                        )
                        rec = eval_point(
                            rows,
                            prob,
                            base_point_oof,
                            tuning,
                            name,
                            {"alpha_internal": ai, "alpha_mpm": am, "alpha_proto": ap, "coachai_mode": mode, "alpha_coachai": ac},
                        )
                        point_rows.append(rec)
                        point_probs[name] = {"oof": prob, "test": test_prob}
    point_search = pd.DataFrame(point_rows).sort_values("point_macro_f1", ascending=False).reset_index(drop=True)
    point_search.to_csv(OUTDIR / "v165_combined_point_search.csv", index=False)

    # Generate conservative no-old candidates.
    r67_sub = load_submission(R67_ANCHOR, rally_uids)
    r121_sub = load_submission(R121_MIN, rally_uids) if R121_MIN.exists() else r67_sub
    r154_sub = load_submission(R154_BEST, rally_uids) if R154_BEST.exists() else r67_sub
    r119_sub = load_submission(R119_POINT, rally_uids) if R119_POINT.exists() else r67_sub

    best_action = action_search.iloc[0]["candidate"]
    best_point = point_search.iloc[0]["candidate"]

    action_sources = {
        "r67_public": r67_sub["actionId"].astype(int).to_numpy(),
        "v165_combined": action_pred(test_rows, action_probs[best_action]["test"], tuning),
    }
    point_sources = {
        "r67_v3": r67_sub["pointId"].astype(int).to_numpy(),
        "r119_public_point": r119_sub["pointId"].astype(int).to_numpy(),
        "r154_safe_physics": r154_sub["pointId"].astype(int).to_numpy(),
        "v165_combined": point_pred(test_rows, point_probs[best_point]["test"], tuning),
    }
    server_sources = {
        "r67_current_server": r67_sub["serverGetPoint"].astype(float).to_numpy(),
        "r121_min_w0p2": r121_sub["serverGetPoint"].astype(float).to_numpy(),
    }

    generated = []
    combos = [
        ("r67_public", "v165_combined", "r121_min_w0p2"),
        ("r67_public", "v165_combined", "r67_current_server"),
        ("r67_public", "r154_safe_physics", "r121_min_w0p2"),
        ("r67_public", "r119_public_point", "r121_min_w0p2"),
        ("v165_combined", "v165_combined", "r121_min_w0p2"),
    ]
    for a_key, p_key, s_key in combos:
        name = f"submission_v165_no_old__a{a_key}__p{p_key}__s{s_key}.csv"
        info = write_submission(name, rally_uids, action_sources[a_key], point_sources[p_key], server_sources[s_key])
        info.update({"action_source": a_key, "point_source": p_key, "server_source": s_key})
        generated.append(info)

    summary = {
        "safety": {
            "uses_old_test_server_labels": False,
            "uses_test_new_hidden_targets": False,
            "external_rows_appended_to_train": False,
            "coachai_direct_actionid_mapping": False,
        },
        "base_metrics": {
            "base_action_macro_f1": base_action_metric,
            "base_point_macro_f1": base_point_metric,
            "base_server_auc": base_server_auc,
        },
        "coachai": {
            "canonical_rows": int(len(coachai_data)),
            "transition_rows": int(coachai_priors["rows"]),
            "sequences": int(coachai_priors["sequences"]),
        },
        "opentt": {
            "segments": int(opentt_priors.get("segments_count", 0)),
            "transition_rows": int(opentt_priors.get("transition_rows", 0)),
        },
        "best_action": action_search.head(10).to_dict(orient="records"),
        "best_point": point_search.head(10).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "v165_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "v165_report.md").write_text(
        "# V165 Combined External Pretraining Proxy\n\n"
        "## Safety\n\n"
        "- No old-test server replacement.\n"
        "- No external rows appended to AICUP train.\n"
        "- CoachAI badminton labels are used only as coarse family/grid priors.\n\n"
        "## External Sources Combined\n\n"
        "- CoachAI: coarse shot-family and landing-grid transition prior.\n"
        "- OpenTTGames: coarse table-tennis action-family prior.\n"
        "- DeepMind + TT-MatchDynamics: physics landing-grid prior.\n"
        "- AICUP train/test prefix internals: task-adaptive transition prior.\n\n"
        "## Best Local Results\n\n"
        f"- Base action Macro-F1: `{base_action_metric:.6f}`\n"
        f"- Best V165 action: `{action_search.iloc[0]['candidate']}` = `{action_search.iloc[0]['action_macro_f1']:.6f}`\n"
        f"- Base point Macro-F1: `{base_point_metric:.6f}`\n"
        f"- Best V165 point: `{point_search.iloc[0]['candidate']}` = `{point_search.iloc[0]['point_macro_f1']:.6f}`\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), SRC_ANALYSIS / Path(__file__).name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
