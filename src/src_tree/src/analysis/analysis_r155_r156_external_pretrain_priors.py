"""R155/R156 external pretraining-prior probes.

Direction 3:
  Use OpenTTGames as an action-family / terminal auxiliary teacher.  This is
  implemented as low-weight action-family and point0 terminal logit residuals,
  not as direct AICUP label training.

Direction 4:
  Use R151 safe physics priors as a prior-guided point logit bias.  This reuses
  only DeepMind and TT-MatchDynamics priors; TTMATCH is excluded entirely.

The script produces OOF diagnostics and upload-ready candidate files.  It does
not use external rally_uid lookup and does not append external rows to AICUP
training.
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

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ANALYSIS_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from analysis_r151b_r154_physics_prior_integration import (  # noqa: E402
    blend_nonterminal_point,
    md_conditioned_prior,
    normalize_nonzero_prior,
    point_pred,
    static_grid_prior_from_r151,
)
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features  # noqa: E402
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES  # noqa: E402
from baseline_v3 import apply_segmented_multipliers  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


OUTDIR = Path("r155_r156_external_pretrain_priors")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
OPEN_EVENTS = Path("external_data/openttgames/processed/openttgames_events.csv")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R142_SERVER = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"
R143_SERVER = UPLOAD_DIR / "submission_r143_r67_anchor_oldsharpen005095_newscore_gapcal.csv"
R111_OOF = Path("r111_remaining_moe_gru/oof_proba_r111.pkl")
R111_TEST = Path("r111_remaining_moe_gru/test_proba_r111.pkl")
R101_OOF = Path("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
R101_TEST = Path("r101_r103_destiny_gru/test_proba_r101_r103.pkl")


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


ACTION_TO_FAMILY = {
    0: "unknown",
    1: "attack",
    2: "attack",
    3: "attack",
    4: "attack",
    5: "attack",
    6: "attack",
    7: "attack",
    8: "control",
    9: "control",
    10: "control",
    11: "control",
    12: "defensive",
    13: "defensive",
    14: "defensive",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

TECHNIQUE_TO_FAMILY = {
    "serve": "serve",
    "loop": "attack",
    "flick": "attack",
    "smash": "attack",
    "push": "control",
    "block": "defensive",
    "chop": "defensive",
    "lob": "defensive",
}

FAMILY_TO_ACTIONS = {
    "serve": [15, 16, 17, 18],
    "attack": [1, 2, 3, 4, 5, 6, 7],
    "control": [8, 9, 10, 11],
    "defensive": [12, 13, 14],
    "unknown": [0],
}

FAMILIES = ["serve", "attack", "control", "defensive", "unknown"]


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def phase_from_prefix_len(prefix_len: int) -> str:
    if prefix_len <= 1:
        return "receive"
    if prefix_len == 2:
        return "third_ball"
    if prefix_len == 3:
        return "fourth_ball"
    return "rally"


def action_family(action_id: int) -> str:
    return ACTION_TO_FAMILY.get(int(action_id), "unknown")


def family_distribution_to_actions(fam_dist: dict[str, float], base_action: np.ndarray) -> np.ndarray:
    out = np.zeros_like(base_action, dtype=float)
    for fam, mass in fam_dist.items():
        idx = [ACTION_CLASSES.index(a) for a in FAMILY_TO_ACTIONS.get(fam, [0])]
        base_mass = float(base_action[idx].sum())
        if base_mass > 1e-9:
            out[idx] += float(mass) * base_action[idx] / base_mass
        else:
            out[idx] += float(mass) / len(idx)
    return normalize_rows(out.reshape(1, -1))[0]


def parse_opentt_segments(events: pd.DataFrame) -> list[dict]:
    """Return stroke segments and terminal marker from OpenTTGames events."""
    segments: list[dict] = []
    for video_id, video in events.sort_values(["video_id", "frame"]).groupby("video_id", sort=False):
        cur: list[dict] = []
        terminal_type = "none"
        for row in video.itertuples(index=False):
            event_type = str(row.event_type)
            if event_type == "stroke":
                technique = str(row.technique) if str(row.technique) else "unknown"
                if technique == "serve" and cur:
                    segments.append({"video_id": video_id, "strokes": cur, "terminal_type": terminal_type})
                    cur = []
                    terminal_type = "none"
                cur.append(
                    {
                        "technique": technique,
                        "family": TECHNIQUE_TO_FAMILY.get(technique, "unknown"),
                    }
                )
            elif event_type == "rally_ending":
                terminal_type = str(row.rally_ending_type) if str(row.rally_ending_type) else "terminal"
                if cur:
                    segments.append({"video_id": video_id, "strokes": cur, "terminal_type": terminal_type})
                    cur = []
                    terminal_type = "none"
            elif event_type == "empty_event":
                if cur:
                    segments.append({"video_id": video_id, "strokes": cur, "terminal_type": terminal_type})
                    cur = []
                    terminal_type = "none"
        if cur:
            segments.append({"video_id": video_id, "strokes": cur, "terminal_type": terminal_type})
    return [seg for seg in segments if len(seg["strokes"]) >= 1]


def estimate_opentt_aux_priors(events: pd.DataFrame, alpha: float = 20.0) -> tuple[dict, pd.DataFrame]:
    segments = parse_opentt_segments(events)
    fam_global = Counter()
    fam_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    terminal_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    rows: list[dict] = []

    for seg in segments:
        strokes = seg["strokes"]
        for i, cur in enumerate(strokes):
            phase = phase_from_prefix_len(i + 1)
            key = (phase, cur["family"])
            is_last = i == len(strokes) - 1
            terminal_counts[key][int(is_last)] += 1
            if i + 1 < len(strokes):
                nxt = strokes[i + 1]
                fam_counts[key][nxt["family"]] += 1
                fam_global[nxt["family"]] += 1
                rows.append(
                    {
                        "phase": phase,
                        "current_family": cur["family"],
                        "next_family": nxt["family"],
                        "is_terminal_next": int(is_last),
                    }
                )
            else:
                fam_global["unknown"] += 1
                rows.append(
                    {
                        "phase": phase,
                        "current_family": cur["family"],
                        "next_family": "unknown",
                        "is_terminal_next": int(is_last),
                    }
                )

    total_global = sum(fam_global.values())
    fam_global_prob = {
        fam: (fam_global[fam] + 1.0) / (total_global + len(FAMILIES)) for fam in FAMILIES
    }
    global_terminal = sum(v[1] for v in terminal_counts.values()) / max(1, sum(sum(v.values()) for v in terminal_counts.values()))

    fam_prior: dict[tuple[str, str], dict[str, float]] = {}
    term_prior: dict[tuple[str, str], float] = {}
    for key in set(fam_counts) | set(terminal_counts):
        counter = fam_counts.get(key, Counter())
        n = float(sum(counter.values()))
        denom = n + alpha
        fam_prior[key] = {
            fam: (counter[fam] + alpha * fam_global_prob[fam]) / denom for fam in FAMILIES
        }
        tcounter = terminal_counts.get(key, Counter())
        tn = float(sum(tcounter.values()))
        term_prior[key] = (tcounter[1] + alpha * global_terminal) / (tn + alpha) if tn + alpha > 0 else global_terminal

    report_rows = []
    for key in sorted(fam_prior):
        phase, fam = key
        rec = {"phase": phase, "current_family": fam, "terminal_prior": term_prior.get(key, global_terminal)}
        rec.update({f"next_family_{f}": fam_prior[key][f] for f in FAMILIES})
        rec["support"] = int(sum(fam_counts.get(key, Counter()).values()))
        rec["terminal_support"] = int(sum(terminal_counts.get(key, Counter()).values()))
        report_rows.append(rec)

    priors = {
        "family_prior": fam_prior,
        "terminal_prior": term_prior,
        "family_global": fam_global_prob,
        "terminal_global": global_terminal,
        "segments_count": len(segments),
        "transition_rows": len(rows),
    }
    return priors, pd.DataFrame(report_rows)


def opentt_action_prior(rows: pd.DataFrame, base_action: np.ndarray, priors: dict) -> np.ndarray:
    out = np.zeros_like(base_action, dtype=float)
    fam_prior: dict = priors["family_prior"]
    global_dist: dict = priors["family_global"]
    for i, row in enumerate(rows.itertuples(index=False)):
        prefix_len = int(getattr(row, "prefix_len"))
        lag0_action = int(getattr(row, "lag0_actionId", 0))
        key = (phase_from_prefix_len(prefix_len), action_family(lag0_action))
        dist = fam_prior.get(key, global_dist)
        out[i] = family_distribution_to_actions(dist, base_action[i])
    return normalize_rows(out)


def opentt_terminal_prior(rows: pd.DataFrame, priors: dict) -> np.ndarray:
    term_prior: dict = priors["terminal_prior"]
    global_terminal = float(priors["terminal_global"])
    out = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows.itertuples(index=False)):
        prefix_len = int(getattr(row, "prefix_len"))
        lag0_action = int(getattr(row, "lag0_actionId", 0))
        key = (phase_from_prefix_len(prefix_len), action_family(lag0_action))
        out[i] = term_prior.get(key, global_terminal)
    return np.clip(out, 1e-5, 1 - 1e-5)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def action_pred(meta: pd.DataFrame, prob: np.ndarray, tuning: GrUTuning) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode).astype(int)


def eval_action(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict | None = None) -> dict:
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
    for k in [0, 8, 9, 12, 14]:
        rec[f"f1_action_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_action_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict | None = None) -> dict:
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
    for k in [0, 5, 7, 8, 9]:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_point_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def eval_server(meta: pd.DataFrame, server: np.ndarray, base_server: np.ndarray, name: str, extra: dict | None = None) -> dict:
    y = meta["serverGetPoint"].astype(int).to_numpy()
    rec = {
        "candidate": name,
        "server_auc": float(roc_auc_score(y, server)) if len(np.unique(y)) == 2 else np.nan,
        "server_mad_vs_base": float(np.mean(np.abs(server - base_server))),
    }
    if extra:
        rec.update(extra)
    return rec


def blend_action(base: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    return normalize_rows((1.0 - alpha) * normalize_rows(base) + alpha * normalize_rows(prior))


def terminal_point_bias(base_point: np.ndarray, term_prior: np.ndarray, beta: float) -> np.ndarray:
    out = normalize_rows(base_point).copy()
    p0 = out[:, 0]
    shifted = sigmoid(logit(p0) + beta * (logit(term_prior) - logit(np.full_like(term_prior, np.mean(term_prior)))))
    # Keep this conservative: preserve non-terminal distribution exactly.
    non = np.clip(1.0 - p0, 1e-9, 1.0)
    q = out[:, 1:10] / non[:, None]
    out[:, 0] = shifted
    out[:, 1:10] = (1.0 - shifted)[:, None] * q
    return normalize_rows(out)


def load_submission(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    return pd.DataFrame({"rally_uid": rally_uids}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, template: pd.DataFrame, action: np.ndarray, point: np.ndarray, server: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = template[["rally_uid"]].copy()
    sub["actionId"] = action.astype(int)
    sub["pointId"] = point.astype(int)
    sub["serverGetPoint"] = np.round(server.astype(float), 8)
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    # Load AI CUP OOF/test representations.
    r111_oof = load_pickle(R111_OOF)
    r111_test = load_pickle(R111_TEST)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    tuning: GrUTuning = r111_oof["tuning"]
    meta = r111_oof["valid_meta"].copy().reset_index(drop=True)
    test_meta = r111_test["test_meta"].copy().reset_index(drop=True)

    _train_raw, _test_raw, prefix, test_prefix, _features = prepare_prefix_features()
    meta_for_align = meta.copy()
    if "fold" not in meta_for_align.columns:
        meta_for_align["fold"] = 0
    rows = align_prefix_meta(meta_for_align, prefix).reset_index(drop=True)
    # Test metadata has no labels; test_prefix is already one row per test sample
    # in submission order after prepare_prefix_features().
    test_rows = test_prefix.reset_index(drop=True)
    if not test_rows["rally_uid"].reset_index(drop=True).equals(test_meta["rally_uid"].reset_index(drop=True)):
        test_rows = pd.DataFrame({"rally_uid": test_meta["rally_uid"].astype(int)}).merge(
            test_prefix,
            on="rally_uid",
            how="left",
            validate="one_to_one",
        )

    base_action_oof = normalize_rows(r111_oof["gru_action"])
    base_action_test = normalize_rows(r111_test["gru_action"])
    base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * r111_oof["gru_point"])
    base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * r111_test["gru_point"])
    base_server_oof = np.asarray(r111_oof["gru_server"], dtype=float)
    base_server_test = np.asarray(r111_test["gru_server"], dtype=float)

    # Direction 3: OpenTT family / terminal priors.
    events = pd.read_csv(OPEN_EVENTS)
    opentt_priors, opentt_report = estimate_opentt_aux_priors(events)
    opentt_report.to_csv(OUTDIR / "r155_opentt_family_terminal_prior_table.csv", index=False)
    action_prior_oof = opentt_action_prior(rows, base_action_oof, opentt_priors)
    action_prior_test = opentt_action_prior(test_rows, base_action_test, opentt_priors)
    term_prior_oof = opentt_terminal_prior(rows, opentt_priors)
    term_prior_test = opentt_terminal_prior(test_rows, opentt_priors)

    action_rows: list[dict] = []
    action_rows.append(eval_action(meta, base_action_oof, base_action_oof, tuning, "r111_action_base", {"alpha": 0.0}))
    action_candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]:
        prob_oof = blend_action(base_action_oof, action_prior_oof, alpha)
        prob_test = blend_action(base_action_test, action_prior_test, alpha)
        name = f"r155_opentt_family_a{clean_float(alpha)}"
        action_candidates[name] = (prob_oof, prob_test)
        action_rows.append(eval_action(meta, prob_oof, base_action_oof, tuning, name, {"alpha": alpha}))
    action_search = pd.DataFrame(action_rows).sort_values(["action_macro_f1", "action_churn_vs_base"], ascending=[False, True])
    action_search.to_csv(OUTDIR / "r155_action_family_search.csv", index=False)

    # Direction 4 plus Direction 3 terminal: point bias search.
    static_md = static_grid_prior_from_r151("matchdynamics", "mirror", depth_only=False)
    static_dm = static_grid_prior_from_r151("deepmind", "mirror", depth_only=False)
    prior_md_oof = md_conditioned_prior(rows, "mirror", static_md)
    prior_md_test = md_conditioned_prior(test_rows, "mirror", static_md)
    prior_dm_oof = np.tile(static_dm.reshape(1, -1), (len(rows), 1))
    prior_dm_test = np.tile(static_dm.reshape(1, -1), (len(test_rows), 1))
    prior_mix_oof = normalize_nonzero_prior(0.7 * prior_md_oof + 0.3 * prior_dm_oof)
    prior_mix_test = normalize_nonzero_prior(0.7 * prior_md_test + 0.3 * prior_dm_test)

    point_rows: list[dict] = []
    point_rows.append(eval_point(meta, base_point_oof, base_point_oof, tuning, "r101_r111_point_base", {"alpha": 0.0, "beta": 0.0}))
    point_candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in [0.0025, 0.005, 0.01, 0.02]:
        for beta in [0.0, 0.02, 0.05, 0.10]:
            phys_oof = blend_nonterminal_point(base_point_oof, prior_mix_oof, alpha)
            phys_test = blend_nonterminal_point(base_point_test, prior_mix_test, alpha)
            if beta > 0:
                phys_oof = terminal_point_bias(phys_oof, term_prior_oof, beta)
                phys_test = terminal_point_bias(phys_test, term_prior_test, beta)
            name = f"r156_phys_terminal_a{clean_float(alpha)}_b{clean_float(beta)}"
            point_candidates[name] = (phys_oof, phys_test)
            point_rows.append(
                eval_point(meta, phys_oof, base_point_oof, tuning, name, {"alpha": alpha, "beta": beta})
            )
    point_search = pd.DataFrame(point_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True])
    point_search.to_csv(OUTDIR / "r156_point_physics_terminal_search.csv", index=False)

    # Server auxiliary probe: OpenTT terminal prior as tiny server/terminal teacher.
    server_rows: list[dict] = []
    server_rows.append(eval_server(meta, base_server_oof, base_server_oof, "r111_server_base", {"beta": 0.0}))
    server_candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for beta in [0.01, 0.02, 0.05, 0.10]:
        # This is intentionally weak: terminal prior is not a direct server label.
        centered_oof = term_prior_oof - np.mean(term_prior_oof)
        centered_test = term_prior_test - np.mean(term_prior_oof)
        prob_oof = sigmoid(logit(base_server_oof) + beta * centered_oof)
        prob_test = sigmoid(logit(base_server_test) + beta * centered_test)
        name = f"r155_terminal_server_b{clean_float(beta)}"
        server_candidates[name] = (prob_oof, prob_test)
        server_rows.append(eval_server(meta, prob_oof, base_server_oof, name, {"beta": beta}))
    server_search = pd.DataFrame(server_rows).sort_values(["server_auc", "server_mad_vs_base"], ascending=[False, True])
    server_search.to_csv(OUTDIR / "r155_terminal_server_search.csv", index=False)

    # Upload-ready low-risk candidates: keep server from strong R142/R143 policy
    # when available; otherwise use base GRU server.  Use test predictions from
    # the best local small-churn rows.
    template = pd.read_csv(R67_ANCHOR)
    rally_uids = template["rally_uid"].astype(int).to_numpy()
    server_template = None
    for path in [R143_SERVER, R142_SERVER, R67_ANCHOR]:
        if path.exists():
            server_template = load_submission(path, rally_uids)["serverGetPoint"].to_numpy(dtype=float)
            break
    if server_template is None:
        server_template = base_server_test

    generated: list[dict] = []
    best_action_names = [n for n in action_search["candidate"].tolist() if n != "r111_action_base"][:3]
    best_point_names = [n for n in point_search["candidate"].tolist() if n != "r101_r111_point_base"][:3]

    for aname in best_action_names[:2]:
        _, atest = action_candidates[aname]
        pred_action = action_pred(test_meta, atest, tuning)
        sub_name = f"submission_{aname}_r67point_r142server.csv"
        info = write_submission(sub_name, template, pred_action, template["pointId"].to_numpy(), server_template)
        row = action_search[action_search["candidate"].eq(aname)].iloc[0].to_dict()
        generated.append({**row, **info, "kind": "action_family"})

    for pname in best_point_names[:3]:
        _, ptest = point_candidates[pname]
        pred_point = point_pred(test_meta, ptest, tuning)
        sub_name = f"submission_r67action_{pname}_r142server.csv"
        info = write_submission(sub_name, template, template["actionId"].to_numpy(), pred_point, server_template)
        row = point_search[point_search["candidate"].eq(pname)].iloc[0].to_dict()
        generated.append({**row, **info, "kind": "point_physics_terminal"})

    # Combined best action + best point, conservative.
    if best_action_names and best_point_names:
        aname = best_action_names[0]
        pname = best_point_names[0]
        pred_action = action_pred(test_meta, action_candidates[aname][1], tuning)
        pred_point = point_pred(test_meta, point_candidates[pname][1], tuning)
        sub_name = f"submission_{aname}__{pname}_r142server.csv"
        info = write_submission(sub_name, template, pred_action, pred_point, server_template)
        generated.append({"candidate": sub_name, "kind": "combined", "action_source": aname, "point_source": pname, **info})

    pd.DataFrame(generated).to_csv(OUTDIR / "r155_r156_generated_candidates.csv", index=False)

    report = {
        "safety": {
            "uses_ttmatch": False,
            "uses_test_labels": False,
            "uses_rally_uid_lookup": False,
            "external_rows_appended_to_train": False,
        },
        "opentt": {
            "segments": int(opentt_priors["segments_count"]),
            "transition_rows": int(opentt_priors["transition_rows"]),
            "terminal_global": float(opentt_priors["terminal_global"]),
        },
        "best_action": action_search.head(8).to_dict(orient="records"),
        "best_point": point_search.head(8).to_dict(orient="records"),
        "best_server": server_search.head(5).to_dict(orient="records"),
        "generated_count": len(generated),
        "outputs": {
            "action_search": str(OUTDIR / "r155_action_family_search.csv"),
            "point_search": str(OUTDIR / "r156_point_physics_terminal_search.csv"),
            "server_search": str(OUTDIR / "r155_terminal_server_search.csv"),
            "generated": str(OUTDIR / "r155_r156_generated_candidates.csv"),
        },
    }
    (OUTDIR / "r155_r156_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r155_r156_report.md").write_text(
        "# R155/R156 External Pretraining-Prior Probes\n\n"
        "This run implements Direction 3 and Direction 4 as low-risk external teacher/prior probes.\n\n"
        "## Safety\n\n"
        "- No TTMATCH is used.\n"
        "- No AI CUP test labels are used.\n"
        "- No rally_uid lookup or external row append is used.\n\n"
        "## OpenTTGames Auxiliary Prior\n\n"
        f"- Parsed segments: `{report['opentt']['segments']}`\n"
        f"- Transition rows: `{report['opentt']['transition_rows']}`\n"
        f"- Global terminal prior: `{report['opentt']['terminal_global']:.6f}`\n\n"
        "## Top Action-Family Rows\n\n"
        + action_search.head(8).to_csv(index=False)
        + "\n## Top Point Physics/Terminal Rows\n\n"
        + point_search.head(8).to_csv(index=False)
        + "\n## Generated Candidates\n\n"
        + pd.DataFrame(generated).to_csv(index=False)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
