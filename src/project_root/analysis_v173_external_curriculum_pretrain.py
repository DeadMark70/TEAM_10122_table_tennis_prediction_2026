"""V173A external curriculum pretraining proxy.

This tests the curriculum idea without directly appending external rows to
AICUP train:

Stage A external coarse priors:
  - OpenTTGames coarse action-family prior.
  - CoachAI coarse family and landing-grid transition prior.
  - DeepMind / TT-MatchDynamics physics landing priors.

Stage B AICUP task-adaptive priors:
  - train prefixes plus test_new observed internal transitions for test priors.

Stage C supervised student anchor:
  - R111/R101 causal GRU blend.

Stage D task teacher:
  - R166 combined teacher targets.

The output is a low-weight teacher/proxy sweep, not a full GPU retrain.  It is
private-safe: no old-test server labels and no hidden test target.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
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

import analysis_v165_combined_external_pretrain_proxy as v165  # noqa: E402
import analysis_v160_v163_task_pretrain_distill as v160  # noqa: E402
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, validate_raw_data  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


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


OUTDIR = Path("v173_external_curriculum_pretrain")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v173_external_curriculum_pretrain.py")

R166_TARGETS = Path("r166_teacher_distillation/r166_teacher_targets.npz")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R119_POINT = UPLOAD_DIR / "submission_r119_point_w0p05.csv"
R121_MIN = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
R154_POINT = UPLOAD_DIR / "submission_r154_md_mirror_a0p01_r67_anchor.csv"
R142_OLDSHARPEN = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def row_log_blend(base: np.ndarray, prior: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    base = normalize_rows(np.clip(base, 1e-9, None))
    prior = normalize_rows(np.clip(prior, 1e-9, None))
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        a = np.full(len(base), float(a), dtype=float)
    a = np.clip(a, 0.0, 1.0)[:, None]
    out = np.exp((1.0 - a) * np.log(base) + a * np.log(prior))
    return normalize_rows(out)


def row_nonterminal_blend(base: np.ndarray, prior: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    base = normalize_rows(np.clip(base, 1e-9, None))
    prior = normalize_rows(np.clip(prior, 1e-9, None))
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        a = np.full(len(base), float(a), dtype=float)
    a = np.clip(a, 0.0, 1.0)[:, None]
    out = base.copy()
    mass = np.clip(1.0 - out[:, 0], 1e-9, 1.0)
    base9 = normalize_rows(out[:, 1:10] / mass[:, None])
    prior9 = normalize_rows(prior[:, 1:10])
    q = np.exp((1.0 - a) * np.log(np.clip(base9, 1e-9, None)) + a * np.log(np.clip(prior9, 1e-9, None)))
    out[:, 1:10] = mass[:, None] * normalize_rows(q)
    return normalize_rows(out)


def action_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return v165.action_pred(meta, prob, tuning)


def point_pred(meta: pd.DataFrame, prob: np.ndarray, tuning) -> np.ndarray:
    return v165.point_pred(meta, prob, tuning)


def eval_action(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict | None = None) -> dict:
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
    for k in [0, 5, 8, 9, 10, 11, 12, 13, 14]:
        rec[f"f1_action_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_action_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning, name: str, extra: dict | None = None) -> dict:
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
    if extra:
        rec.update(extra)
    return rec


def eval_combo(meta: pd.DataFrame, action_prob: np.ndarray, point_prob: np.ndarray, server: np.ndarray, tuning, name: str) -> dict:
    action_f1 = float(
        f1_score(
            meta["next_actionId"].astype(int),
            action_pred(meta, action_prob, tuning),
            labels=ACTION_CLASSES,
            average="macro",
            zero_division=0,
        )
    )
    point_f1 = float(
        f1_score(
            meta["next_pointId"].astype(int),
            point_pred(meta, point_prob, tuning),
            labels=POINT_CLASSES,
            average="macro",
            zero_division=0,
        )
    )
    server_auc = float(roc_auc_score(meta["serverGetPoint"].astype(int), server))
    return {
        "candidate": name,
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "server_auc": server_auc,
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def weighted_mix(parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    total = np.zeros_like(parts[0][1], dtype=float)
    weight_sum = 0.0
    for weight, prob in parts:
        if weight <= 0:
            continue
        total += float(weight) * normalize_rows(np.clip(prob, 1e-9, None))
        weight_sum += float(weight)
    if weight_sum <= 0:
        return normalize_rows(parts[0][1])
    return normalize_rows(total / weight_sum)


def point9_to_point10(prob9: np.ndarray) -> np.ndarray:
    """Convert external landing-grid priors over 1..9 into pointId 0..9.

    The zero/terminal class is intentionally left at a tiny mass because these
    external physical priors should only guide non-terminal landing geometry.
    """
    prob9 = normalize_rows(np.clip(prob9, 1e-9, None))
    out = np.full((len(prob9), 10), 1e-9, dtype=float)
    out[:, 1:10] = prob9
    return normalize_rows(out)


def load_submission(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    out = pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError(f"Submission did not align: {path}")
    return out


def write_submission(name: str, rally_uids: np.ndarray, action: np.ndarray, point: np.ndarray, server: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(
        {
            "rally_uid": rally_uids.astype(int),
            "actionId": action.astype(int),
            "pointId": point.astype(int),
            "serverGetPoint": np.round(np.clip(server.astype(float), 1e-6, 1 - 1e-6), 8),
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

    targets = np.load(R166_TARGETS)
    r111_oof = load_pickle(v165.R111_OOF)
    r111_test = load_pickle(v165.R111_TEST)
    r101_oof = load_pickle(v165.R101_OOF)
    r101_test = load_pickle(v165.R101_TEST)
    tuning = r111_oof["tuning"]

    _, _, prefix, test_prefix, _ = v165.prepare_prefix_features()
    rows = v165.align_prefix_meta(v160.ensure_fold(r111_oof["valid_meta"]), prefix).reset_index(drop=True)
    test_rows = test_prefix.reset_index(drop=True)
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])
    base_point_oof = normalize_rows(0.50 * r111_oof["gru_point"] + 0.50 * r101_oof["gru_point"])
    base_point_test = normalize_rows(0.50 * r111_test["gru_point"] + 0.50 * r101_test["gru_point"])
    base_server_oof = 0.5 * r111_oof["gru_server"] + 0.5 * r101_oof["gru_server"]

    teacher_action_oof = targets["teacher_action_oof"]
    teacher_action_test = targets["teacher_action_test"]
    teacher_point_oof = targets["teacher_point_oof"]
    teacher_point_test = targets["teacher_point_test"]

    base_metrics = eval_combo(rows, base_action_oof, base_point_oof, base_server_oof, tuning, "r111_r101_base")

    # Stage A external priors.
    if v165.OPEN_EVENTS.exists():
        opentt_priors, opentt_table = v165.estimate_opentt_aux_priors(pd.read_csv(v165.OPEN_EVENTS))
        opentt_table.to_csv(OUTDIR / "v173_opentt_prior_table.csv", index=False)
        opentt_action_oof = v165.opentt_action_prior(rows, base_action_oof, opentt_priors)
        opentt_action_test = v165.opentt_action_prior(test_rows, base_action_test, opentt_priors)
    else:
        opentt_priors = {"segments_count": 0, "transition_rows": 0}
        opentt_action_oof = base_action_oof.copy()
        opentt_action_test = base_action_test.copy()

    coachai_data = v165.load_coachai_sequences()
    coachai_priors, coachai_stats = v165.build_coachai_transition_priors(coachai_data)
    coachai_stats.to_csv(OUTDIR / "v173_coachai_transition_stats.csv", index=False)
    coachai_action_oof = v165.coachai_family_prior_for_rows(rows, coachai_priors, prefix["next_actionId"])
    coachai_action_test = v165.coachai_family_prior_for_rows(test_rows, coachai_priors, prefix["next_actionId"])
    coachai_point_oof = v165.coachai_grid_prior_for_rows(rows, coachai_priors, "avg")
    coachai_point_test = v165.coachai_grid_prior_for_rows(test_rows, coachai_priors, "avg")

    dm, md = v165.load_external_states()
    mpm_tables, mpm_global, _ = v165.build_mpm_tables(dm, mirror=True)
    mpm_oof = v165.mpm_prior_for_rows(rows, mpm_tables, mpm_global)
    mpm_test = v165.mpm_prior_for_rows(test_rows, mpm_tables, mpm_global)
    proto = v165.build_external_prototypes(dm, md, mirror=False)
    proto_oof = v165.prototype_prior_for_rows(rows, proto, tau=0.35, k=250)
    proto_test = v165.prototype_prior_for_rows(test_rows, proto, tau=0.35, k=250)

    external_action_oof = weighted_mix([(0.55, opentt_action_oof), (0.45, coachai_action_oof)])
    external_action_test = weighted_mix([(0.55, opentt_action_test), (0.45, coachai_action_test)])
    external_point_oof = weighted_mix(
        [
            (0.35, point9_to_point10(coachai_point_oof)),
            (0.35, point9_to_point10(mpm_oof)),
            (0.30, point9_to_point10(proto_oof)),
        ]
    )
    external_point_test = weighted_mix(
        [
            (0.35, point9_to_point10(coachai_point_test)),
            (0.35, point9_to_point10(mpm_test)),
            (0.30, point9_to_point10(proto_test)),
        ]
    )

    # Stage B internal task-adaptive priors.
    internal_action_oof, internal_point_oof = v160.foldsafe_internal_priors(prefix, rows)
    test_internal = v160.build_test_internal_prefixes(test_raw)
    internal_action_test, internal_point_test = v160.full_internal_priors(prefix, test_rows, test_internal)

    # Stage D teacher schedules.  We deliberately keep external weights small.
    action_schedules = [
        ("ext05_int10_teacher85", 0.05, 0.10, 0.85),
        ("ext10_int10_teacher80", 0.10, 0.10, 0.80),
        ("ext10_int20_teacher70", 0.10, 0.20, 0.70),
        ("ext15_int15_teacher70", 0.15, 0.15, 0.70),
        ("ext20_int10_teacher70", 0.20, 0.10, 0.70),
        ("ext20_int20_teacher60", 0.20, 0.20, 0.60),
        ("ext30_int20_teacher50", 0.30, 0.20, 0.50),
    ]
    point_schedules = [
        ("ext05_int10_teacher85", 0.05, 0.10, 0.85),
        ("ext10_int10_teacher80", 0.10, 0.10, 0.80),
        ("ext10_int20_teacher70", 0.10, 0.20, 0.70),
        ("ext15_int15_teacher70", 0.15, 0.15, 0.70),
        ("ext20_int10_teacher70", 0.20, 0.10, 0.70),
    ]

    action_teacher_map: dict[str, dict[str, np.ndarray]] = {}
    for name, we, wi, wt in action_schedules:
        action_teacher_map[name] = {
            "oof": weighted_mix([(we, external_action_oof), (wi, internal_action_oof), (wt, teacher_action_oof)]),
            "test": weighted_mix([(we, external_action_test), (wi, internal_action_test), (wt, teacher_action_test)]),
        }
    point_teacher_map: dict[str, dict[str, np.ndarray]] = {}
    for name, we, wi, wt in point_schedules:
        point_teacher_map[name] = {
            "oof": weighted_mix([(we, external_point_oof), (wi, internal_point_oof), (wt, teacher_point_oof)]),
            "test": weighted_mix([(we, external_point_test), (wi, internal_point_test), (wt, teacher_point_test)]),
        }

    action_rows, action_probs = [], {}
    for schedule_name, teacher in action_teacher_map.items():
        for alpha in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
            prob = row_log_blend(base_action_oof, teacher["oof"], alpha)
            test_prob = row_log_blend(base_action_test, teacher["test"], alpha)
            name = f"v173_action_{schedule_name}_a{clean_float(alpha)}"
            action_rows.append(eval_action(rows, prob, base_action_oof, tuning, name, {"schedule": schedule_name, "alpha": alpha}))
            action_probs[name] = {"oof": prob, "test": test_prob}
    action_search = pd.DataFrame(action_rows).sort_values(["action_macro_f1", "action_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    action_search.to_csv(OUTDIR / "v173_action_curriculum_search.csv", index=False)

    point_rows, point_probs = [], {}
    for schedule_name, teacher in point_teacher_map.items():
        for alpha in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]:
            prob = row_nonterminal_blend(base_point_oof, teacher["oof"], alpha)
            test_prob = row_nonterminal_blend(base_point_test, teacher["test"], alpha)
            name = f"v173_point_{schedule_name}_a{clean_float(alpha)}"
            point_rows.append(eval_point(rows, prob, base_point_oof, tuning, name, {"schedule": schedule_name, "alpha": alpha}))
            point_probs[name] = {"oof": prob, "test": test_prob}
    point_search = pd.DataFrame(point_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    point_search.to_csv(OUTDIR / "v173_point_curriculum_search.csv", index=False)

    combo_rows = []
    for an in list(action_search.head(8)["candidate"]):
        for pn in list(point_search.head(8)["candidate"]):
            combo_rows.append(eval_combo(rows, action_probs[an]["oof"], point_probs[pn]["oof"], base_server_oof, tuning, f"{an}__{pn}__base_server"))
    combo_search = pd.DataFrame(combo_rows).sort_values("overall", ascending=False).reset_index(drop=True)
    combo_search.to_csv(OUTDIR / "v173_curriculum_combo_search.csv", index=False)

    r67_sub = load_submission(R67_ANCHOR, rally_uids)
    r119_sub = load_submission(R119_POINT, rally_uids)
    r154_sub = load_submission(R154_POINT, rally_uids)
    r121_sub = load_submission(R121_MIN, rally_uids)
    oldsharp_sub = load_submission(R142_OLDSHARPEN, rally_uids) if R142_OLDSHARPEN.exists() else None

    best_action = str(action_search.iloc[0]["candidate"])
    best_point = str(point_search.iloc[0]["candidate"])
    best_combo = combo_search.iloc[0]
    combo_action = str(best_combo["candidate"]).split("__")[0]
    combo_point = str(best_combo["candidate"]).split("__")[1]

    def action_from(name: str) -> np.ndarray:
        return action_pred(test_rows, action_probs[name]["test"], tuning)

    def point_from(name: str) -> np.ndarray:
        return point_pred(test_rows, point_probs[name]["test"], tuning)

    generated = []
    candidates = [
        ("v173_best_action", action_from(best_action), "r119_public_point", r119_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("v173_point_probe", r67_sub["actionId"].astype(int).to_numpy(), "v173_best_point", point_from(best_point), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("v173_best_action", action_from(best_action), "r154_safe_physics", r154_sub["pointId"].astype(int).to_numpy(), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
        ("v173_combo", action_from(combo_action), "v173_combo_point", point_from(combo_point), "r121_min_w0p2", r121_sub["serverGetPoint"].astype(float).to_numpy()),
    ]
    if oldsharp_sub is not None:
        candidates.append(("v173_combo", action_from(combo_action), "v173_combo_point", point_from(combo_point), "oldsharpen005095", oldsharp_sub["serverGetPoint"].astype(float).to_numpy()))

    for a_key, action, p_key, point, s_key, server in candidates:
        info = write_submission(f"submission_v173__a{a_key}__p{p_key}__s{s_key}.csv", rally_uids, action, point, server)
        info.update({"action_source": a_key, "point_source": p_key, "server_source": s_key})
        generated.append(info)

    summary = {
        "base_metrics": base_metrics,
        "external_sources": {
            "opentt_segments": int(opentt_priors.get("segments_count", 0)),
            "opentt_transition_rows": int(opentt_priors.get("transition_rows", 0)),
            "coachai_rows": int(len(coachai_data)),
            "coachai_sequences": int(coachai_data["_seq_key"].nunique()),
            "test_internal_transition_rows": int(len(test_internal)),
        },
        "best_action": action_search.head(10).to_dict(orient="records"),
        "best_point": point_search.head(10).to_dict(orient="records"),
        "best_combo": combo_search.head(10).to_dict(orient="records"),
        "generated": generated,
    }
    (OUTDIR / "v173_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "v173_report.md").write_text(
        "# V173A External Curriculum Pretraining Proxy\n\n"
        "## Best Results\n\n"
        f"- Base R111/R101 overall `{base_metrics['overall']:.6f}` "
        f"(action `{base_metrics['action_macro_f1']:.6f}`, point `{base_metrics['point_macro_f1']:.6f}`, server `{base_metrics['server_auc']:.6f}`).\n"
        f"- Best action: `{best_action}` action `{action_search.iloc[0]['action_macro_f1']:.6f}`, churn `{action_search.iloc[0]['action_churn_vs_base']:.4%}`.\n"
        f"- Best point: `{best_point}` point `{point_search.iloc[0]['point_macro_f1']:.6f}`, churn `{point_search.iloc[0]['point_churn_vs_base']:.4%}`.\n"
        f"- Best combo: `{best_combo['candidate']}` overall `{best_combo['overall']:.6f}`.\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}`" for g in generated)
        + "\n",
        encoding="utf-8",
    )
    with open("experiments_log.md", "a", encoding="utf-8") as f:
        f.write(
            "\n\n## V173A external curriculum pretraining proxy\n\n"
            f"- Base R111/R101 blend: overall {base_metrics['overall']:.6f}, action {base_metrics['action_macro_f1']:.6f}, "
            f"point {base_metrics['point_macro_f1']:.6f}, server {base_metrics['server_auc']:.6f}.\n"
            f"- Best action: `{best_action}` action {action_search.iloc[0]['action_macro_f1']:.6f}, "
            f"churn {action_search.iloc[0]['action_churn_vs_base']:.4%}.\n"
            f"- Best point: `{best_point}` point {point_search.iloc[0]['point_macro_f1']:.6f}, "
            f"churn {point_search.iloc[0]['point_churn_vs_base']:.4%}.\n"
            f"- Best combo: `{best_combo['candidate']}` overall {best_combo['overall']:.6f}.\n"
            "- External curriculum used OpenTT/CoachAI/physics priors, internal observed transitions, and R166 task teachers; no old-server labels.\n"
        )
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
