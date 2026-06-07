"""R184 receiver-affordance action refiner.

This extends the public-positive V173 action branch with receiver-affordance
gates.  The goal is to ask whether V173 is winning because it better models
what the receiver can plausibly do from the incoming ball state.

Generated submissions keep point/server fixed to the no-old private-first
anchor: R119 point + R121 server.  No old-server labels are used.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT_DIR = Path(__file__).resolve().parent
SRC_ANALYSIS = ROOT_DIR / "src" / "analysis"
for p in [ROOT_DIR, SRC_ANALYSIS]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import analysis_v160_v163_task_pretrain_distill as v160  # noqa: E402
import analysis_v165_combined_external_pretrain_proxy as v165  # noqa: E402
import analysis_v173_external_curriculum_pretrain as v173  # noqa: E402
from analysis_r179_action_physics_hierarchy import action_family, phase_name, point_depth, point_side  # noqa: E402
from baseline_lgbm import ACTION_CLASSES, validate_raw_data  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


OUTDIR = Path("r184_receiver_affordance_refiner")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_r184_receiver_affordance_refiner.py")

BASE_NO_OLD = UPLOAD_DIR / "submission_r177_no_old_safe_r67_r119_r121.csv"
V173_ACTION = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
R166_TARGETS = Path("r166_teacher_distillation/r166_teacher_targets.npz")


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


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()


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


def depth_name(point_id: int) -> str:
    return {0: "zero", 1: "short", 2: "half", 3: "long"}[point_depth(point_id)]


def side_name(point_id: int) -> str:
    return {0: "zero", 1: "fh", 2: "mid", 3: "bh"}[point_side(point_id)]


def add_affordance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r184_phase"] = [phase_name(p, l) for p, l in zip(out["phase_id"], out["prefix_len"])]
    out["r184_lag0_family"] = [action_family(v) for v in out["lag0_actionId"]]
    out["r184_lag0_depth"] = [depth_name(v) for v in out["lag0_pointId"]]
    out["r184_lag0_side"] = [side_name(v) for v in out["lag0_pointId"]]
    out["r184_state"] = (
        out["r184_phase"].astype(str)
        + "|fam="
        + out["r184_lag0_family"].astype(str)
        + "|depth="
        + out["r184_lag0_depth"].astype(str)
        + "|spin="
        + out["lag0_spinId"].astype(str)
        + "|str="
        + out["lag0_strengthId"].astype(str)
    )
    out["r184_state_simple"] = (
        out["r184_phase"].astype(str)
        + "|fam="
        + out["r184_lag0_family"].astype(str)
        + "|depth="
        + out["r184_lag0_depth"].astype(str)
    )
    return out


def rebuild_v173_best_actions() -> dict:
    """Rebuild V173 best action OOF/test predictions and aligned metadata."""
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
    rows = add_affordance_columns(rows)
    test_rows = add_affordance_columns(test_prefix.reset_index(drop=True))
    rally_uids = r111_test["test_meta"]["rally_uid"].astype(int).to_numpy()

    base_action_oof = normalize_rows(0.65 * r111_oof["gru_action"] + 0.35 * r101_oof["gru_action"])
    base_action_test = normalize_rows(0.65 * r111_test["gru_action"] + 0.35 * r101_test["gru_action"])

    teacher_action_oof = targets["teacher_action_oof"]
    teacher_action_test = targets["teacher_action_test"]

    if v165.OPEN_EVENTS.exists():
        opentt_priors, _ = v165.estimate_opentt_aux_priors(pd.read_csv(v165.OPEN_EVENTS))
        opentt_action_oof = v165.opentt_action_prior(rows, base_action_oof, opentt_priors)
        opentt_action_test = v165.opentt_action_prior(test_rows, base_action_test, opentt_priors)
    else:
        opentt_action_oof = base_action_oof.copy()
        opentt_action_test = base_action_test.copy()

    coachai_data = v165.load_coachai_sequences()
    coachai_priors, _ = v165.build_coachai_transition_priors(coachai_data)
    coachai_action_oof = v165.coachai_family_prior_for_rows(rows, coachai_priors, prefix["next_actionId"])
    coachai_action_test = v165.coachai_family_prior_for_rows(test_rows, coachai_priors, prefix["next_actionId"])

    external_action_oof = v173.weighted_mix([(0.55, opentt_action_oof), (0.45, coachai_action_oof)])
    external_action_test = v173.weighted_mix([(0.55, opentt_action_test), (0.45, coachai_action_test)])

    internal_action_oof, _ = v160.foldsafe_internal_priors(prefix, rows)
    test_internal = v160.build_test_internal_prefixes(test_raw)
    internal_action_test, _ = v160.full_internal_priors(prefix, test_rows, test_internal)

    best = pd.read_csv(OUTDIR.parent / "v173_external_curriculum_pretrain" / "v173_action_curriculum_search.csv").iloc[0]
    schedule = str(best["schedule"])
    alpha = float(best["alpha"])
    weights = {
        "ext05_int10_teacher85": (0.05, 0.10, 0.85),
        "ext10_int10_teacher80": (0.10, 0.10, 0.80),
        "ext10_int20_teacher70": (0.10, 0.20, 0.70),
        "ext15_int15_teacher70": (0.15, 0.15, 0.70),
        "ext20_int10_teacher70": (0.20, 0.10, 0.70),
        "ext20_int20_teacher60": (0.20, 0.20, 0.60),
        "ext30_int20_teacher50": (0.30, 0.20, 0.50),
    }
    we, wi, wt = weights[schedule]
    action_teacher_oof = v173.weighted_mix([(we, external_action_oof), (wi, internal_action_oof), (wt, teacher_action_oof)])
    action_teacher_test = v173.weighted_mix([(we, external_action_test), (wi, internal_action_test), (wt, teacher_action_test)])
    v173_prob_oof = v173.row_log_blend(base_action_oof, action_teacher_oof, alpha)
    v173_prob_test = v173.row_log_blend(base_action_test, action_teacher_test, alpha)

    base_pred_oof = v173.action_pred(rows, base_action_oof, tuning)
    v173_pred_oof = v173.action_pred(rows, v173_prob_oof, tuning)
    base_pred_test = v173.action_pred(test_rows, base_action_test, tuning)
    v173_pred_test = v173.action_pred(test_rows, v173_prob_test, tuning)

    return {
        "rows": rows,
        "test_rows": test_rows,
        "rally_uids": rally_uids,
        "base_pred_oof": base_pred_oof,
        "v173_pred_oof": v173_pred_oof,
        "base_pred_test": base_pred_test,
        "v173_pred_test": v173_pred_test,
        "best_candidate": str(best["candidate"]),
        "schedule": schedule,
        "alpha": alpha,
    }


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def build_frame(meta: pd.DataFrame, base: np.ndarray, teacher: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
    out = meta.copy()
    out["base_action"] = np.asarray(base, dtype=int)
    out["teacher_action"] = np.asarray(teacher, dtype=int)
    out["changed"] = out["base_action"].ne(out["teacher_action"])
    out["base_family"] = [action_family(v) for v in out["base_action"]]
    out["teacher_family"] = [action_family(v) for v in out["teacher_action"]]
    if y is not None:
        out["y"] = np.asarray(y, dtype=int)
        out["base_correct"] = out["base_action"].eq(out["y"])
        out["teacher_correct"] = out["teacher_action"].eq(out["y"])
    return out


def support_keys_oof(oof: pd.DataFrame, key_cols: list[str], *, min_rows: int, min_changed: int, min_delta: float) -> set[tuple]:
    rows = []
    for key, part in oof.groupby(key_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                "key": key_tuple,
                "rows": len(part),
                "changed_rows": int(part["changed"].sum()),
                "base_acc": float(part["base_correct"].mean()),
                "teacher_acc": float(part["teacher_correct"].mean()),
                "delta_acc": float(part["teacher_correct"].mean() - part["base_correct"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return set()
    keep = df["rows"].ge(min_rows) & df["changed_rows"].ge(min_changed) & df["delta_acc"].ge(min_delta)
    return set(df.loc[keep, "key"])


def mask_by_keys(df: pd.DataFrame, key_cols: list[str], keys: set[tuple]) -> np.ndarray:
    if not keys:
        return np.zeros(len(df), dtype=bool)
    vals = [tuple(row) for row in df[key_cols].itertuples(index=False, name=None)]
    return np.array([v in keys for v in vals], dtype=bool)


def transition_mask(df: pd.DataFrame, pairs: set[tuple[int, int]]) -> np.ndarray:
    return np.array([(int(b), int(t)) in pairs for b, t in zip(df["base_action"], df["teacher_action"])], dtype=bool)


def compose(base: np.ndarray, teacher: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    mask = np.asarray(mask, dtype=bool)
    out[mask] = np.asarray(teacher, dtype=int)[mask]
    return out


def write_candidate(name: str, template: pd.DataFrame, action: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = template[["rally_uid", "pointId", "serverGetPoint"]].copy()
    sub.insert(1, "actionId", np.asarray(action, dtype=int))
    sub = sub[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"candidate": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def candidate_record(
    name: str,
    note: str,
    oof_action: np.ndarray,
    test_action: np.ndarray,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    teacher_oof: np.ndarray,
    teacher_test: np.ndarray,
) -> dict:
    y = oof["y"].to_numpy(dtype=int)
    changed_oof = oof_action != base_oof
    changed_test = test_action != base_test
    return {
        "candidate": name,
        "note": note,
        "oof_action_macro_f1": f1(y, oof_action),
        "oof_delta_vs_base": f1(y, oof_action) - f1(y, base_oof),
        "oof_delta_vs_full_v173": f1(y, oof_action) - f1(y, teacher_oof),
        "oof_churn_vs_base": float(np.mean(changed_oof)),
        "oof_changed_rows": int(np.sum(changed_oof)),
        "test_churn_vs_no_old": float(np.mean(changed_test)),
        "test_changed_rows": int(np.sum(changed_test)),
        "test_accept_full_v173_share": float(np.sum((test_action == teacher_test) & (teacher_test != base_test)) / max(1, np.sum(teacher_test != base_test))),
        "receive_test_changed": int(np.sum(changed_test & test["r184_phase"].eq("receive").to_numpy())),
        "third_ball_test_changed": int(np.sum(changed_test & test["r184_phase"].eq("third_ball").to_numpy())),
        "fourth_ball_test_changed": int(np.sum(changed_test & test["r184_phase"].eq("fourth_ball").to_numpy())),
        "rally_test_changed": int(np.sum(changed_test & test["r184_phase"].eq("rally").to_numpy())),
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    state = rebuild_v173_best_actions()
    no_old = load_sub(BASE_NO_OLD, state["rally_uids"])
    submitted_v173 = load_sub(V173_ACTION, state["rally_uids"])
    no_old_action = no_old["actionId"].astype(int).to_numpy()
    submitted_v173_action = submitted_v173["actionId"].astype(int).to_numpy()

    oof = build_frame(state["rows"], state["base_pred_oof"], state["v173_pred_oof"], state["rows"]["next_actionId"].astype(int).to_numpy())
    test = build_frame(state["test_rows"], no_old_action, submitted_v173_action)
    test_rebuild_churn = float(np.mean(submitted_v173_action != state["v173_pred_test"]))

    state_keys = support_keys_oof(oof, ["r184_state_simple"], min_rows=60, min_changed=8, min_delta=0.01)
    pair_keys = support_keys_oof(oof, ["r184_phase", "base_action", "teacher_action"], min_rows=8, min_changed=8, min_delta=0.02)
    rich_state_keys = support_keys_oof(oof, ["r184_state"], min_rows=30, min_changed=5, min_delta=0.02)

    core_pairs = {
        (4, 10),
        (4, 11),
        (7, 10),
        (7, 11),
        (1, 3),
        (1, 10),
        (6, 13),
        (13, 6),
        (13, 5),
        (2, 13),
    }
    changed_oof = oof["changed"].to_numpy()
    changed_test = test["changed"].to_numpy()

    specs = [
        (
            "state_supported_simple",
            "OOF-supported coarse incoming state gate",
            changed_oof & mask_by_keys(oof, ["r184_state_simple"], state_keys),
            changed_test & mask_by_keys(test, ["r184_state_simple"], state_keys),
        ),
        (
            "state_supported_rich",
            "OOF-supported richer incoming state gate",
            changed_oof & mask_by_keys(oof, ["r184_state"], rich_state_keys),
            changed_test & mask_by_keys(test, ["r184_state"], rich_state_keys),
        ),
        (
            "state_pair_supported",
            "OOF-supported phase/base/target transition gate",
            changed_oof & mask_by_keys(oof, ["r184_phase", "base_action", "teacher_action"], pair_keys),
            changed_test & mask_by_keys(test, ["r184_phase", "base_action", "teacher_action"], pair_keys),
        ),
        (
            "receive_affordance_control",
            "receive-phase short/half incoming-ball control affordance",
            changed_oof
            & oof["r184_phase"].eq("receive").to_numpy()
            & oof["r184_lag0_depth"].isin(["short", "half"]).to_numpy()
            & oof["teacher_action"].isin([4, 6, 7, 10, 11, 12]).to_numpy(),
            changed_test
            & test["r184_phase"].eq("receive").to_numpy()
            & test["r184_lag0_depth"].isin(["short", "half"]).to_numpy()
            & test["teacher_action"].isin([4, 6, 7, 10, 11, 12]).to_numpy(),
        ),
        (
            "third_ball_intent",
            "third-ball terminal/control/defense intent affordance",
            changed_oof & oof["r184_phase"].eq("third_ball").to_numpy() & oof["teacher_action"].isin([2, 3, 10, 11, 13]).to_numpy(),
            changed_test & test["r184_phase"].eq("third_ball").to_numpy() & test["teacher_action"].isin([2, 3, 10, 11, 13]).to_numpy(),
        ),
        (
            "attack_to_control",
            "accept V173 attack-to-control corrections",
            changed_oof & oof["base_family"].eq("Attack").to_numpy() & oof["teacher_family"].eq("Control").to_numpy(),
            changed_test & test["base_family"].eq("Attack").to_numpy() & test["teacher_family"].eq("Control").to_numpy(),
        ),
        (
            "attack_to_terminal",
            "accept V173 attack-to-terminal-ish corrections",
            changed_oof & oof["base_family"].eq("Attack").to_numpy() & oof["teacher_action"].isin([0, 3]).to_numpy(),
            changed_test & test["base_family"].eq("Attack").to_numpy() & test["teacher_action"].isin([0, 3]).to_numpy(),
        ),
        (
            "defense_rebalance",
            "accept V173 counter/block/fast-push defensive rebalance pairs",
            changed_oof & transition_mask(oof, {(6, 13), (13, 6), (13, 5), (2, 13), (5, 13)}),
            changed_test & transition_mask(test, {(6, 13), (13, 6), (13, 5), (2, 13), (5, 13)}),
        ),
        (
            "core_physical_transitions",
            "accept hand-selected high-frequency physical transition set",
            changed_oof & transition_mask(oof, core_pairs),
            changed_test & transition_mask(test, core_pairs),
        ),
    ]

    # A pragmatic union of OOF-supported and interpretable affordance gates.
    combo_oof = (
        specs[0][2]
        | specs[2][2]
        | specs[3][2]
        | specs[4][2]
        | specs[7][2]
    )
    combo_test = (
        specs[0][3]
        | specs[2][3]
        | specs[3][3]
        | specs[4][3]
        | specs[7][3]
    )
    specs.append(("affordance_union", "union of OOF-supported states plus receive/third/defense affordance gates", combo_oof, combo_test))

    generated = []
    records = []
    for short, note, mask_oof, mask_test in specs:
        oof_action = compose(state["base_pred_oof"], state["v173_pred_oof"], mask_oof)
        test_action = compose(no_old_action, submitted_v173_action, mask_test)
        name = f"submission_r184_{slug(short)}__pr119_sr121.csv"
        info = write_candidate(name, no_old, test_action)
        rec = candidate_record(
            name,
            note,
            oof_action,
            test_action,
            oof,
            test,
            state["base_pred_oof"],
            no_old_action,
            state["v173_pred_oof"],
            submitted_v173_action,
        )
        info.update(rec)
        generated.append(info)
        records.append(rec)

    base_f1 = f1(oof["y"].to_numpy(dtype=int), state["base_pred_oof"])
    full_v173_f1 = f1(oof["y"].to_numpy(dtype=int), state["v173_pred_oof"])
    full_v173_churn = float(np.mean(state["v173_pred_oof"] != state["base_pred_oof"]))

    pd.DataFrame(records).sort_values(["oof_action_macro_f1", "test_churn_vs_no_old"], ascending=[False, True]).to_csv(
        OUTDIR / "r184_candidate_metrics.csv", index=False
    )

    state_stats = []
    for key, part in oof.groupby("r184_state_simple", dropna=False):
        state_stats.append(
            {
                "state": key,
                "rows": int(len(part)),
                "changed_rows": int(part["changed"].sum()),
                "base_acc": float(part["base_correct"].mean()),
                "v173_acc": float(part["teacher_correct"].mean()),
                "delta_acc": float(part["teacher_correct"].mean() - part["base_correct"].mean()),
            }
        )
    pd.DataFrame(state_stats).sort_values(["delta_acc", "changed_rows"], ascending=[False, False]).to_csv(
        OUTDIR / "r184_oof_state_support.csv", index=False
    )

    report = {
        "base_oof_action_macro_f1": base_f1,
        "full_v173_oof_action_macro_f1": full_v173_f1,
        "full_v173_oof_delta_vs_base": full_v173_f1 - base_f1,
        "full_v173_oof_churn_vs_base": full_v173_churn,
        "v173_test_rebuild_churn_vs_submitted": test_rebuild_churn,
        "v173_best_candidate": state["best_candidate"],
        "v173_schedule": state["schedule"],
        "v173_alpha": state["alpha"],
        "supported_state_simple_count": len(state_keys),
        "supported_rich_state_count": len(rich_state_keys),
        "supported_pair_count": len(pair_keys),
        "generated_count": len(generated),
        "generated": generated,
        "notes": [
            "R184 uses V173 as a public-positive action teacher.",
            "All submissions keep no-old R119 point and R121 server fixed.",
            "OOF metrics are computed by applying the same affordance masks to rebuilt V173 OOF predictions.",
            "Public result of full V173 remains stronger evidence than R184 OOF until these ablations are public-probed.",
        ],
    }
    (OUTDIR / "r184_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r184_report.md").write_text(
        "# R184 Receiver Affordance Refiner\n\n"
        "## OOF Anchors\n\n"
        f"- Base action OOF Macro-F1: `{base_f1:.6f}`\n"
        f"- Full V173 action OOF Macro-F1: `{full_v173_f1:.6f}`\n"
        f"- Full V173 OOF delta: `{full_v173_f1 - base_f1:.6f}`\n"
        f"- Full V173 OOF churn: `{full_v173_churn:.6f}`\n"
        f"- V173 rebuild churn vs submitted test action: `{test_rebuild_churn:.6f}`\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(
            f"- `{g['upload_path']}` OOF `{g['oof_action_macro_f1']:.6f}`, "
            f"OOF delta `{g['oof_delta_vs_base']:.6f}`, test churn `{g['test_churn_vs_no_old']:.6f}`"
            for g in generated
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r184_receiver_affordance_refiner.py", SRC_DEST)
    print(json.dumps({"generated_count": len(generated), "metrics": str(OUTDIR / "r184_candidate_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
