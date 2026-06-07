"""V336 lightweight action MoE around the strict V173/V306/V300 anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    action_distribution_report,
    drop_leaky_columns,
    macro_f1,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v336_action_moe"
TEST_META_PATH = ROOT / "test_new.csv"
TRAIN_PATH = ROOT / "train.csv"
V173_ACTION_PATH = (
    ROOT
    / "v173_external_curriculum_pretrain"
    / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
)
V306_ANCHOR_PATH = (
    ROOT
    / "v306_point0_addition_probe"
    / "submission_v306_p0_cap0p01__v173action_v300server.csv"
)
V300_SERVER_PATH = (
    ROOT
    / "v300_clean_server_blend_recycler"
    / "submission_v300_best_safe_repack__v173action_v261point_server.csv"
)

ACTION_CLASSES = list(range(19))
SERVE_CLASSES = {15, 16, 17, 18}
MIN_OOF_DELTA = 0.003


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame


def _rally_level_meta(raw: pd.DataFrame, rally_uid: pd.Series) -> pd.DataFrame:
    sort_cols = [col for col in ["rally_uid", "strikeNumber"] if col in raw.columns]
    meta = raw.sort_values(sort_cols).groupby("rally_uid", sort=False).tail(1).reset_index(drop=True)
    order = pd.DataFrame({"rally_uid": rally_uid})
    aligned = order.merge(meta, on="rally_uid", how="left", validate="one_to_one")
    if len(aligned) != len(order) or aligned.isna().all(axis=1).any():
        raise ValueError("could not align rally-level metadata to anchor rally_uid")
    return aligned


def _blocked_report(missing: list[str]) -> dict[str, Any]:
    return {
        "version": "V336",
        "decision": "BLOCKED_MISSING_ANCHOR",
        "missing_anchors": missing,
        "generated_submission_count": 0,
        "generated_submissions": [],
        "policy": {
            "no_old_server": True,
            "no_ttmatch": True,
            "no_upload_directory_writes": True,
            "strict_leakage_filter": True,
        },
    }


def load_anchor_frames() -> dict[str, Any]:
    missing = []
    for label, path in {
        "v173_action_submission": V173_ACTION_PATH,
        "v306_point_action_anchor": V306_ANCHOR_PATH,
        "v300_clean_server": V300_SERVER_PATH,
        "test_meta": TEST_META_PATH,
    }.items():
        if not path.exists():
            missing.append(f"{label}:{relative_path(path)}")
    if missing:
        raise FileNotFoundError("; ".join(missing))

    action_anchor = _load_submission(V173_ACTION_PATH)
    package_anchor = _load_submission(V306_ANCHOR_PATH)
    server_anchor = _load_submission(V300_SERVER_PATH)
    if not action_anchor["rally_uid"].equals(package_anchor["rally_uid"]):
        raise ValueError("V173 and V306 rally_uid mismatch")
    if not server_anchor["rally_uid"].equals(package_anchor["rally_uid"]):
        raise ValueError("V300 and V306 rally_uid mismatch")

    test_meta = _rally_level_meta(pd.read_csv(TEST_META_PATH), package_anchor["rally_uid"])
    package = package_anchor.copy()
    package["serverGetPoint"] = server_anchor["serverGetPoint"].to_numpy(dtype=float)
    validate_submission_schema(package)

    frames: dict[str, Any] = {
        "action_anchor": action_anchor,
        "package_anchor": package,
        "test_meta": test_meta,
        "anchor_paths": {
            "v173_action": relative_path(V173_ACTION_PATH),
            "v306_point": relative_path(V306_ANCHOR_PATH),
            "v300_server": relative_path(V300_SERVER_PATH),
        },
    }
    # Do not fabricate an OOF anchor from train labels. A true action OOF anchor
    # must be fold-safe and aligned to validation rows; otherwise local gains are
    # meaningless. This script can still inspect test churn and export nothing.
    frames["oof_status"] = "UNAVAILABLE_STRICT_V173_OOF"
    return frames


def _series(frame: pd.DataFrame, name: str, default: Any = 0) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def build_rule_action_experts(test_meta: pd.DataFrame, base_action: np.ndarray) -> dict[str, np.ndarray]:
    """Build conservative rule experts using observed-prefix fields only."""
    frame = drop_leaky_columns(test_meta).reset_index(drop=True)
    base = np.asarray(base_action, dtype=int)
    if len(frame) != len(base):
        raise ValueError("test_meta and base_action length mismatch")

    phase = _series(frame, "phase_id", "").astype(str).str.lower()
    strike_number = pd.to_numeric(_series(frame, "strikeNumber", 0), errors="coerce").fillna(0).astype(int)
    lag0 = pd.to_numeric(_series(frame, "lag0_actionId", base), errors="coerce").fillna(pd.Series(base)).astype(int)
    spin = pd.to_numeric(_series(frame, "spinId", 0), errors="coerce").fillna(0).astype(int)
    strength = pd.to_numeric(_series(frame, "strengthId", 0), errors="coerce").fillna(0).astype(int)
    position = pd.to_numeric(_series(frame, "positionId", 0), errors="coerce").fillna(0).astype(int)

    experts: dict[str, np.ndarray] = {}

    pred = base.copy()
    mask = phase.str.contains("receive", na=False) | (strike_number <= 2)
    pred[mask.to_numpy() & np.isin(base, [10, 11, 12, 13, 14])] = 4
    experts["receive_control"] = pred

    pred = base.copy()
    mask = phase.str.contains("third", na=False) | (strike_number == 3)
    pred[mask.to_numpy() & (strength >= 2).to_numpy()] = 3
    experts["third_attack"] = pred

    pred = base.copy()
    mask = (strike_number >= 5).to_numpy() & (spin >= 4).to_numpy()
    pred[mask & np.isin(base, [0, 3, 5, 7, 8, 9, 12, 14])] = 6
    experts["rally_defense"] = pred

    pred = base.copy()
    mask = np.isin(base, [0, 3, 5, 7, 8, 9, 12, 14])
    pred[mask & (position >= 2).to_numpy()] = 6
    experts["weak_0_3_5_7_8_9_12_14"] = pred

    pred = base.copy()
    mask = (lag0 >= 10).to_numpy() & (strike_number >= 4).to_numpy()
    pred[mask] = 6
    experts["transition_backoff"] = pred

    for name, values in experts.items():
        values[np.isin(values, list(SERVE_CLASSES))] = base[np.isin(values, list(SERVE_CLASSES))]
        experts[name] = values.astype(int)
    return experts


def score_action_candidates(
    oof_meta: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    expert_oof_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    _ = drop_leaky_columns(oof_meta)
    base_score = macro_f1(y_true, base_oof, labels=ACTION_CLASSES)
    rows = []
    for name, pred in expert_oof_dict.items():
        pred_arr = np.asarray(pred, dtype=int)
        if len(pred_arr) != len(base_oof):
            raise ValueError(f"{name} length mismatch")
        score = macro_f1(y_true, pred_arr, labels=ACTION_CLASSES)
        dist = action_distribution_report(base_oof, pred_arr)
        rows.append(
            {
                "candidate": name,
                "base_action_macro_f1": base_score,
                "candidate_action_macro_f1": score,
                "action_oof_delta": score - base_score,
                "changed_action_rows": dist["changed_rows"],
                "serve_15_18_delta": dist["serve_15_18_delta"],
                "serve_15_18_explosion": dist["serve_15_18_explosion"],
            }
        )
    return pd.DataFrame(rows)


def select_by_budget(base: np.ndarray, cand: np.ndarray, utility: np.ndarray, budget: int) -> np.ndarray:
    base_arr = np.asarray(base).copy()
    cand_arr = np.asarray(cand)
    util = np.asarray(utility, dtype=float)
    if not (len(base_arr) == len(cand_arr) == len(util)):
        raise ValueError("base, cand, and utility length mismatch")
    selected = base_arr.copy()
    if budget <= 0:
        return selected
    eligible = np.where((base_arr != cand_arr) & np.isfinite(util))[0]
    if len(eligible) == 0:
        return selected
    order = eligible[np.argsort(-util[eligible], kind="mergesort")]
    take = order[: min(int(budget), len(order))]
    selected[take] = cand_arr[take]
    return selected


def _budgets(n_rows: int) -> list[tuple[str, int]]:
    fixed = [(f"b{b}", b) for b in (10, 20, 40, 80)]
    churn = [(f"churn{int(rate * 100):02d}", int(np.floor(n_rows * rate))) for rate in (0.05, 0.10)]
    return fixed + churn


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report_path = safe_output_path(OUTDIR, "search_report.json")
    missing = [
        label
        for label, path in {
            "v173_action_submission": V173_ACTION_PATH,
            "v306_point_action_anchor": V306_ANCHOR_PATH,
            "v300_clean_server": V300_SERVER_PATH,
        }.items()
        if not path.exists()
    ]
    if missing:
        report = _blocked_report(missing)
        write_json(report_path, report)
        return report

    frames = load_anchor_frames()
    base_test = frames["package_anchor"]["actionId"].astype(int).to_numpy()
    test_experts = build_rule_action_experts(frames["test_meta"], base_test)
    rows = []
    generated = []
    best_row: dict[str, Any] | None = None

    if {"oof_meta", "y_true", "base_oof"} <= set(frames):
        base_oof = frames["base_oof"]
        oof_experts = build_rule_action_experts(frames["oof_meta"], base_oof)
        candidate_scores = score_action_candidates(frames["oof_meta"], frames["y_true"], base_oof, oof_experts)
        candidate_scores.to_csv(safe_output_path(OUTDIR, "candidate_summary.csv"), index=False)
    else:
        candidate_scores = pd.DataFrame()

    for name, cand_test in test_experts.items():
        if candidate_scores.empty:
            oof_delta = 0.0
        else:
            match = candidate_scores[candidate_scores["candidate"] == name]
            oof_delta = float(match["action_oof_delta"].iloc[0]) if len(match) else 0.0
        utility = np.where(cand_test != base_test, 1.0 + max(oof_delta, 0.0), 0.0)
        for budget_name, budget in _budgets(len(base_test)):
            selected = select_by_budget(base_test, cand_test, utility, budget)
            dist = action_distribution_report(base_test, selected)
            evidence_pass = (
                oof_delta >= MIN_OOF_DELTA
                and not dist["serve_15_18_explosion"]
                and dist["changed_rows"] > 0
            )
            row = {
                "candidate": f"{name}_{budget_name}",
                "expert": name,
                "budget": int(budget),
                "action_oof_delta": oof_delta,
                "changed_action_rows": dist["changed_rows"],
                "changed_rate": dist["changed_rate"],
                "serve_15_18_delta": dist["serve_15_18_delta"],
                "serve_15_18_explosion": dist["serve_15_18_explosion"],
                "evidence_pass": bool(evidence_pass),
                "decision": "REVIEW_ACTION" if evidence_pass else "DO_NOT_UPLOAD",
            }
            rows.append(row)
            if evidence_pass and (best_row is None or row["action_oof_delta"] > best_row["action_oof_delta"]):
                best_row = row | {"selected_action": selected}

    churn_df = pd.DataFrame(rows)
    churn_df.to_csv(safe_output_path(OUTDIR, "action_churn_report.csv"), index=False)
    if not Path(safe_output_path(OUTDIR, "candidate_summary.csv")).exists():
        pd.DataFrame(rows).to_csv(safe_output_path(OUTDIR, "candidate_summary.csv"), index=False)

    if best_row is not None:
        out = frames["package_anchor"].copy()
        out["actionId"] = np.asarray(best_row["selected_action"], dtype=int)
        out = out.loc[:, SUBMISSION_COLUMNS]
        validate_submission_schema(out)
        filename = f"submission_v336_{best_row['candidate']}__v306point_v300server.csv"
        out_path = safe_output_path(OUTDIR, filename)
        out.to_csv(out_path, index=False, float_format="%.8f")
        generated.append({"candidate": best_row["candidate"], "path": relative_path(out_path)})

    report = {
        "version": "V336",
        "decision": "REVIEW_ACTION" if generated else "DO_NOT_UPLOAD",
        "anchor_paths": frames["anchor_paths"],
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "best_candidate": None
        if best_row is None
        else {k: v for k, v in best_row.items() if k != "selected_action"},
        "candidate_count": int(len(rows)),
        "feature_policy": {
            "dropped_leaky_columns": True,
            "leaky_prefixes": ["next_", "y_", "true_", "label_"],
            "leaky_exact": ["actionId", "pointId", "serverGetPoint", "rally_uid", "rally_id"],
        },
        "policy": {
            "no_old_server": True,
            "no_ttmatch": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        {
            "outdir": relative_path(OUTDIR),
            "decision": report["decision"],
            "generated_submission_count": report["generated_submission_count"],
        }
    )


if __name__ == "__main__":
    main()
