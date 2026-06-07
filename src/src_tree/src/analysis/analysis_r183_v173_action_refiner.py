"""R183 V173 action refiner and clean no-old action probes.

V173 has a positive public signal as an action-only replacement under the
R119 point + R121 server setting.  This script keeps that setting fixed and
creates lower-scope action ablations plus clean R146 action-sweep variants.

No old-server labels are used in generated submissions.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r179_action_physics_hierarchy import phase_name
from analysis_r67_r70_meta_priors import prepare_prefix_features


OUTDIR = Path("r183_v173_action_refiner")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_r183_v173_action_refiner.py")

BASE_NO_OLD = UPLOAD_DIR / "submission_r177_no_old_safe_r67_r119_r121.csv"
V173_ACTION = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
V166_ACTION = UPLOAD_DIR / "submission_r166__ar166_best_action__pr119_public_point__sr121_min_w0p2.csv"

ACTION_LABELS = {
    0: "zero",
    1: "drive",
    2: "counter_drive",
    3: "smash",
    4: "twist",
    5: "fast_drive",
    6: "fast_push",
    7: "flip",
    8: "pimple_long_push",
    9: "pimple_fast_push",
    10: "long_push",
    11: "drop_shot",
    12: "chop",
    13: "block",
    14: "lob",
    15: "serve_traditional",
    16: "serve_hook",
    17: "serve_reverse",
    18: "serve_squat",
}

R146_SOURCE_FILES = {
    "r86_r67_w0p25": UPLOAD_DIR / "submission_r146_ar86_r67_w0p25__pr119point__soldsharpen005095.csv",
    "r95_r93_r88": UPLOAD_DIR / "submission_r146_ar95_r93_r88__pr119point__soldsharpen005095.csv",
    "r96_r92_r93": UPLOAD_DIR / "submission_r146_ar96_r92_r93__pr119point__soldsharpen005095.csv",
    "r101_destiny_gru": UPLOAD_DIR / "submission_r146_ar101_destiny_gru__pr119point__soldsharpen005095.csv",
    "r105_r101_distill": UPLOAD_DIR / "submission_r146_ar105_r101_distill__pr119point__soldsharpen005095.csv",
    "r111_remaining_moe": UPLOAD_DIR / "submission_r146_ar111_remaining_moe__pr119point__soldsharpen005095.csv",
}


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


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()


def write_candidate(name: str, template: pd.DataFrame, action: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    out = template[["rally_uid", "pointId", "serverGetPoint"]].copy()
    out.insert(1, "actionId", np.asarray(action, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"candidate": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def compose_with_mask(base_action: np.ndarray, teacher_action: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(base_action, dtype=int).copy()
    out[np.asarray(mask, dtype=bool)] = np.asarray(teacher_action, dtype=int)[np.asarray(mask, dtype=bool)]
    return out


def family(action: int) -> str:
    a = int(action)
    if a == 0:
        return "Zero"
    if 1 <= a <= 7:
        return "Attack"
    if 8 <= a <= 11:
        return "Control"
    if 12 <= a <= 14:
        return "Defensive"
    if 15 <= a <= 18:
        return "Serve"
    return "Other"


def candidate_metrics(
    name: str,
    action: np.ndarray,
    base_action: np.ndarray,
    v173_action: np.ndarray,
    meta: pd.DataFrame,
    source: str,
    note: str,
) -> dict:
    changed = action != base_action
    v173_changed = v173_action != base_action
    accepted_v173 = changed & v173_changed & (action == v173_action)
    by_phase = meta.assign(changed=changed).groupby("r183_phase")["changed"].agg(["count", "sum", "mean"]).reset_index()
    target_counts = pd.Series(action[changed]).value_counts().sort_index()
    return {
        "candidate": name,
        "source": source,
        "note": note,
        "action_churn_vs_no_old": float(np.mean(changed)),
        "changed_rows": int(np.sum(changed)),
        "accepted_v173_change_rows": int(np.sum(accepted_v173)),
        "accepted_v173_change_share": float(np.sum(accepted_v173) / max(1, np.sum(v173_changed))),
        "receive_changed": int(by_phase.loc[by_phase["r183_phase"].eq("receive"), "sum"].sum()),
        "third_ball_changed": int(by_phase.loc[by_phase["r183_phase"].eq("third_ball"), "sum"].sum()),
        "fourth_ball_changed": int(by_phase.loc[by_phase["r183_phase"].eq("fourth_ball"), "sum"].sum()),
        "rally_changed": int(by_phase.loc[by_phase["r183_phase"].eq("rally"), "sum"].sum()),
        "target_action_counts": ";".join(f"{int(k)}:{int(v)}" for k, v in target_counts.items()),
    }


def transition_mask(base: np.ndarray, teacher: np.ndarray, pairs: set[tuple[int, int]]) -> np.ndarray:
    return np.array([(int(b), int(t)) in pairs for b, t in zip(base, teacher)], dtype=bool)


def build_action_frame(base: pd.DataFrame, v173: pd.DataFrame, v166: pd.DataFrame) -> pd.DataFrame:
    _, _, _, test_prefix, _ = prepare_prefix_features()
    cols = [
        "rally_uid",
        "prefix_len",
        "phase_id",
        "lag0_actionId",
        "lag0_pointId",
        "lag0_spinId",
        "lag0_strengthId",
        "next_hitter_is_server",
        "serverScoreDiff",
    ]
    frame = base[["rally_uid", "actionId"]].rename(columns={"actionId": "base_action"}).merge(
        v173[["rally_uid", "actionId"]].rename(columns={"actionId": "v173_action"}),
        on="rally_uid",
        validate="one_to_one",
    )
    frame = frame.merge(
        v166[["rally_uid", "actionId"]].rename(columns={"actionId": "v166_action"}),
        on="rally_uid",
        validate="one_to_one",
    )
    frame = frame.merge(test_prefix[cols], on="rally_uid", validate="one_to_one")
    frame["r183_phase"] = [phase_name(p, l) for p, l in zip(frame["phase_id"], frame["prefix_len"])]
    frame["base_family"] = [family(v) for v in frame["base_action"]]
    frame["v173_family"] = [family(v) for v in frame["v173_action"]]
    return frame


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    base = load_sub(BASE_NO_OLD)
    rally_uids = base["rally_uid"].astype(int).to_numpy()
    v173 = load_sub(V173_ACTION, rally_uids)
    v166 = load_sub(V166_ACTION, rally_uids)
    frame = build_action_frame(base, v173, v166)

    base_action = frame["base_action"].astype(int).to_numpy()
    v173_action = frame["v173_action"].astype(int).to_numpy()
    v166_action = frame["v166_action"].astype(int).to_numpy()
    phase = frame["r183_phase"].to_numpy()
    changed_v173 = v173_action != base_action

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
    receive_control_targets = {4, 6, 7, 10, 11, 12}
    third_tactical_targets = {2, 3, 10, 11, 13}
    early_phases = {"receive", "third_ball", "fourth_ball"}

    specs: list[tuple[str, str, str, np.ndarray, np.ndarray]] = [
        ("v173_receive_only", "v173_ablation", "accept V173 changes only in receive phase", v173_action, changed_v173 & (phase == "receive")),
        ("v173_third_ball_only", "v173_ablation", "accept V173 changes only in third-ball phase", v173_action, changed_v173 & (phase == "third_ball")),
        (
            "v173_receive_third_ball",
            "v173_ablation",
            "accept V173 changes in receive and third-ball phases",
            v173_action,
            changed_v173 & np.isin(phase, ["receive", "third_ball"]),
        ),
        ("v173_early_phase", "v173_ablation", "accept V173 changes in receive/third/fourth phases", v173_action, changed_v173 & np.isin(phase, list(early_phases))),
        (
            "v173_receive_control_only",
            "v173_ablation",
            "receive-phase V173 changes only when target is control/legal short-ball action",
            v173_action,
            changed_v173 & (phase == "receive") & np.isin(v173_action, list(receive_control_targets)),
        ),
        (
            "v173_third_tactical_only",
            "v173_ablation",
            "third-ball V173 changes only for tactical terminal/control/defense targets",
            v173_action,
            changed_v173 & (phase == "third_ball") & np.isin(v173_action, list(third_tactical_targets)),
        ),
        (
            "v173_core_transitions",
            "v173_ablation",
            "accept only the high-frequency physically plausible V173 transition set",
            v173_action,
            changed_v173 & transition_mask(base_action, v173_action, core_pairs),
        ),
        (
            "v173_v166_agree",
            "agreement",
            "accept only rows where V173 and V166 agree away from no-old baseline",
            v173_action,
            changed_v173 & (v173_action == v166_action),
        ),
        (
            "v173_v166_agree_early",
            "agreement",
            "accept V173/V166 agreement only in receive/third/fourth phases",
            v173_action,
            changed_v173 & (v173_action == v166_action) & np.isin(phase, list(early_phases)),
        ),
    ]

    generated: list[dict] = []
    metrics: list[dict] = []
    for short_name, source, note, teacher, mask in specs:
        action = compose_with_mask(base_action, teacher, mask)
        name = f"submission_r183_{short_name}__pr119_sr121.csv"
        info = write_candidate(name, base, action)
        rec = candidate_metrics(name, action, base_action, v173_action, frame, source, note)
        info.update(rec)
        generated.append(info)
        metrics.append(rec)

    r146_loaded: dict[str, pd.DataFrame] = {}
    for tag, path in R146_SOURCE_FILES.items():
        if not path.exists():
            continue
        r146_loaded[tag] = load_sub(path, rally_uids)

    for tag, sub in r146_loaded.items():
        action = sub["actionId"].astype(int).to_numpy()
        name = f"submission_r183_no_old_{slug(tag)}__pr119_sr121.csv"
        info = write_candidate(name, base, action)
        rec = candidate_metrics(name, action, base_action, v173_action, frame, "r146_no_old", f"R146 action source {tag} repackaged with R119 point and R121 server")
        info.update(rec)
        generated.append(info)
        metrics.append(rec)

        agree = changed_v173 & (v173_action == action)
        agree_action = compose_with_mask(base_action, v173_action, agree)
        agree_name = f"submission_r183_v173_agree_{slug(tag)}__pr119_sr121.csv"
        agree_info = write_candidate(agree_name, base, agree_action)
        agree_rec = candidate_metrics(
            agree_name,
            agree_action,
            base_action,
            v173_action,
            frame,
            "v173_r146_agreement",
            f"accept V173 changes only where R146 action source {tag} agrees",
        )
        agree_info.update(agree_rec)
        generated.append(agree_info)
        metrics.append(agree_rec)

    metrics_df = pd.DataFrame(metrics).sort_values(["source", "action_churn_vs_no_old", "changed_rows"]).reset_index(drop=True)
    metrics_df.to_csv(OUTDIR / "r183_candidate_metrics.csv", index=False)

    transition_rows = []
    changed = frame[frame["base_action"].ne(frame["v173_action"])].copy()
    for (phase_name_, base_id, v173_id), part in changed.groupby(["r183_phase", "base_action", "v173_action"]):
        transition_rows.append(
            {
                "phase": phase_name_,
                "base_action": int(base_id),
                "base_label": ACTION_LABELS.get(int(base_id), str(base_id)),
                "v173_action": int(v173_id),
                "v173_label": ACTION_LABELS.get(int(v173_id), str(v173_id)),
                "rows": int(len(part)),
            }
        )
    pd.DataFrame(transition_rows).sort_values("rows", ascending=False).to_csv(OUTDIR / "r183_v173_transition_audit.csv", index=False)

    phase_rows = []
    for phase_name_, part in frame.groupby("r183_phase"):
        phase_rows.append(
            {
                "phase": phase_name_,
                "rows": int(len(part)),
                "v173_changed_rows": int(part["base_action"].ne(part["v173_action"]).sum()),
                "v173_churn": float(part["base_action"].ne(part["v173_action"]).mean()),
                "v166_changed_rows": int(part["base_action"].ne(part["v166_action"]).sum()),
                "v166_churn": float(part["base_action"].ne(part["v166_action"]).mean()),
            }
        )
    pd.DataFrame(phase_rows).to_csv(OUTDIR / "r183_phase_churn_summary.csv", index=False)

    report = {
        "generated_count": len(generated),
        "generated": generated,
        "notes": [
            "All R183 submissions keep point/server from submission_r177_no_old_safe_r67_r119_r121.csv.",
            "No old-server labels are used in generated R183 submissions.",
            "V173 public-positive full action replacement remains the current no-old action anchor until V166 public result is known.",
            "R183 ablations are intended for later low-churn public/private-safe probing, not to replace the already-positive full V173 result without evidence.",
        ],
    }
    (OUTDIR / "r183_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r183_report.md").write_text(
        "# R183 V173 Action Refiner\n\n"
        "## Generated Candidates\n\n"
        + "\n".join(f"- `{g['upload_path']}` ({g['source']}, churn `{g['action_churn_vs_no_old']:.6f}`)" for g in generated)
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r183_v173_action_refiner.py", SRC_DEST)
    print(json.dumps({"generated_count": len(generated), "metrics": str(OUTDIR / "r183_candidate_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
