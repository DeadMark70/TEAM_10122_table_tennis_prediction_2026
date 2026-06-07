"""V197 action teacher surgery.

Extract low/medium-churn subsets from public-positive action teachers while
keeping the current no-old point/server anchor fixed:

  action = candidate surgery output
  point  = V188 r186_w005 cap5
  server = R121

No point/server changes are made.  TTMATCH is not read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r179_action_physics_hierarchy import point_depth
from analysis_v165_combined_external_pretrain_proxy import prepare_prefix_features
from analysis_v194_train_test_split_distribution_audit import add_audit_columns


OUTDIR = Path("v197_action_teacher_surgery")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v197_action_teacher_surgery.py")

ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
V166 = UPLOAD_DIR / "submission_r166__ar166_best_action__pr119_public_point__sr121_min_w0p2.csv"
R183_RECEIVE = UPLOAD_DIR / "submission_r183_v173_receive_control_only__pr119_sr121.csv"
R183_THIRD = UPLOAD_DIR / "submission_r183_v173_third_tactical_only__pr119_sr121.csv"
R184_ATTACK_CONTROL = UPLOAD_DIR / "submission_r184_attack_to_control__pr119_sr121.csv"
R184_STATE_PAIR = UPLOAD_DIR / "submission_r184_state_pair_supported__pr119_sr121.csv"


def apply_action_gate(base: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    out[np.asarray(mask, dtype=bool)] = np.asarray(source, dtype=int)[np.asarray(mask, dtype=bool)]
    return out


def transition_gate(
    rows: pd.DataFrame,
    base: np.ndarray,
    source: np.ndarray,
    *,
    phases: set[str],
    pairs: set[tuple[int, int]],
    short_only: bool = False,
) -> np.ndarray:
    phase_mask = rows["audit_phase"].astype(str).isin(phases).to_numpy()
    pair_mask = np.array([(int(b), int(s)) in pairs for b, s in zip(base, source)], dtype=bool)
    mask = phase_mask & pair_mask
    if short_only:
        depth = rows["lag0_pointId"].map(lambda p: point_depth(int(p))).to_numpy()
        mask &= depth == 1
    return mask


def load_sub(path: Path, rally_uids: np.ndarray) -> pd.DataFrame:
    sub = pd.read_csv(path)
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, anchor: pd.DataFrame, action: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = anchor[["rally_uid", "pointId", "serverGetPoint"]].copy()
    out.insert(1, "actionId", np.asarray(action, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def dist(values: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=19)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    _, _, _, test_prefix, _ = prepare_prefix_features()
    rows = add_audit_columns(test_prefix.reset_index(drop=True))
    rally_uids = rows["rally_uid"].astype(int).to_numpy()
    anchor = load_sub(ANCHOR, rally_uids)
    base = anchor["actionId"].astype(int).to_numpy()
    sources = {
        "v166": load_sub(V166, rally_uids)["actionId"].astype(int).to_numpy(),
        "r183_receive": load_sub(R183_RECEIVE, rally_uids)["actionId"].astype(int).to_numpy(),
        "r183_third": load_sub(R183_THIRD, rally_uids)["actionId"].astype(int).to_numpy(),
        "r184_attack_control": load_sub(R184_ATTACK_CONTROL, rally_uids)["actionId"].astype(int).to_numpy(),
        "r184_state_pair": load_sub(R184_STATE_PAIR, rally_uids)["actionId"].astype(int).to_numpy(),
    }

    candidates: dict[str, np.ndarray] = {}
    candidates["v197_v166_full"] = sources["v166"]

    receive_pairs = {(4, 10), (4, 11), (7, 10), (7, 11)}
    receive_mask = transition_gate(rows, base, sources["r183_receive"], phases={"receive"}, pairs=receive_pairs, short_only=True)
    candidates["v197_receive_control_short"] = apply_action_gate(base, sources["r183_receive"], receive_mask)

    third_pairs = {(1, 3), (1, 10), (5, 3), (6, 13), (6, 10)}
    third_mask = transition_gate(rows, base, sources["r183_third"], phases={"third_ball"}, pairs=third_pairs)
    candidates["v197_third_tactical"] = apply_action_gate(base, sources["r183_third"], third_mask)

    agree_v166_r184 = (sources["v166"] == sources["r184_attack_control"]) & (sources["v166"] != base)
    candidates["v197_v166_r184_agree_attack_control"] = apply_action_gate(base, sources["v166"], agree_v166_r184)

    agree_v166_state = (sources["v166"] == sources["r184_state_pair"]) & (sources["v166"] != base)
    candidates["v197_v166_r184_agree_state_pair"] = apply_action_gate(base, sources["v166"], agree_v166_state)

    core_mask = receive_mask | third_mask | agree_v166_r184
    candidates["v197_core_surgery_union"] = apply_action_gate(base, sources["v166"], agree_v166_r184)
    candidates["v197_core_surgery_union"] = apply_action_gate(candidates["v197_core_surgery_union"], sources["r183_receive"], receive_mask)
    candidates["v197_core_surgery_union"] = apply_action_gate(candidates["v197_core_surgery_union"], sources["r183_third"], third_mask)

    generated = []
    summary = []
    for name, action in candidates.items():
        changed = action != base
        sub_name = f"submission_{name}__pv188_r186_w005_cap5__sr121.csv"
        info = write_submission(sub_name, anchor, action)
        rec = {
            "candidate": name,
            "submission": sub_name,
            "action_churn_vs_v173_anchor": float(changed.mean()),
            "changed_rows": int(changed.sum()),
            "receive_changed": int(np.sum(changed & rows["audit_phase"].eq("receive").to_numpy())),
            "third_changed": int(np.sum(changed & rows["audit_phase"].eq("third_ball").to_numpy())),
            "rally_changed": int(np.sum(changed & rows["audit_phase"].eq("rally").to_numpy())),
            "action_distribution": json.dumps(dist(action), sort_keys=True),
        }
        rec.update(info)
        generated.append(info)
        summary.append(rec)

    pd.DataFrame(summary).sort_values("action_churn_vs_v173_anchor").to_csv(OUTDIR / "v197_candidate_summary.csv", index=False)
    report = {
        "verdict": "GENERATED",
        "generated": summary,
        "notes": [
            "All candidates use V188 r186_w005 cap5 point and R121 server.",
            "V197 changes action only.",
            "Low-churn surgery candidates should be evaluated by R200 before upload.",
            "TTMATCH is not read.",
        ],
    }
    (OUTDIR / "v197_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v197_report.md").write_text(
        "# V197 Action Teacher Surgery\n\n"
        f"- Generated candidates: `{len(summary)}`\n"
        "- Fixed point/server: V188 cap5 + R121.\n"
        "- No TTMATCH.\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v197_action_teacher_surgery.py", SRC_DEST)
    print(json.dumps({"generated": len(summary), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
