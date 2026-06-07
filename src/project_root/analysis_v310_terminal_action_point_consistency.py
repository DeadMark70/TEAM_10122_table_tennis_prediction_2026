"""V310 terminal action/point consistency research for V306 point0 rows.

This is local-only diagnostics.  It identifies the V306 point0 additions versus
V300, then tests whether action labels on the same style of point0 rows should
be changed under a conservative OOF analog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import ACTION_CLASSES
from analysis_v261_action_conditioned_point_residual import EXPECTED_COLUMNS, normalize_rows_safe
from analysis_v305_rebuild_v261_from_literal_v188 import point_column
from analysis_v306_point0_addition_probe import (
    V300_SUBMISSION,
    apply_point0_additions,
    build_v261_literal_probabilities,
    load_artifacts,
)


OUTDIR = Path("v310_terminal_action_point_consistency")
V306_SUBMISSION = Path("v306_point0_addition_probe/submission_v306_p0_cap0p01__v173action_v300server.csv")
ACTION_OOF_PROB = Path("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_oof_action_prob.npy")
ACTION_TEST_PROB = Path("v238_v173_reconstruction_ablation/v238_v173_phase_external_r166_test_action_prob.npy")
V173_ACTION_SUBMISSION = Path(
    "v173_external_curriculum_pretrain/submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"
)
V306_CAP = 0.01
TERMINAL_ACTION_PROB_MIN = 0.35
TERMINAL_ACTION_MARGIN_MIN = 0.10
BLOCK_ACTION_PROB_MIN = 0.30
BLOCK_ACTION_MARGIN_MIN = 0.08
DEFENSIVE_FAMILY = {12, 13, 14}


def terminal_compatibility_mask(point: np.ndarray, action: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=int)
    action = np.asarray(action, dtype=int)
    if point.shape != action.shape:
        raise ValueError("point and action must have matching shapes")
    return (point == 0) == (action == 0)


def validate_submission_schema(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"submission columns {list(frame.columns)} != {EXPECTED_COLUMNS}")
    return frame.loc[:, EXPECTED_COLUMNS].copy()


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing submission: {path}")
    return validate_submission_schema(pd.read_csv(path))


def _load_action_prob(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing action probability artifact: {path}")
    prob = normalize_rows_safe(np.load(path))
    if prob.shape != (expected_rows, len(ACTION_CLASSES)):
        raise ValueError(f"{path} shape {prob.shape} != {(expected_rows, len(ACTION_CLASSES))}")
    return prob


def identify_v306_changed_rows(
    v300: pd.DataFrame,
    v306: pd.DataFrame,
    selector_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    if len(v300) != len(v306):
        raise ValueError("V300 and V306 submissions must have matching row counts")
    changed_vs_v300 = v300["pointId"].astype(int).ne(v306["pointId"].astype(int))
    if selector_mask is None:
        row_mask = changed_vs_v300.to_numpy()
    else:
        row_mask = np.asarray(selector_mask, dtype=bool)
        if row_mask.shape != (len(v300),):
            raise ValueError("selector_mask must match submission row count")
    rows = pd.DataFrame(
        {
            "row_id": np.arange(len(v300), dtype=int),
            "rally_uid": v300["rally_uid"].astype(int),
            "v300_actionId": v300["actionId"].astype(int),
            "v300_pointId": v300["pointId"].astype(int),
            "v300_serverGetPoint": v300["serverGetPoint"].astype(int),
            "v306_actionId": v306["actionId"].astype(int),
            "v306_pointId": v306["pointId"].astype(int),
            "v306_serverGetPoint": v306["serverGetPoint"].astype(int),
            "v306_selector_row": row_mask,
            "point_changed_vs_v300": changed_vs_v300,
            "v300_nonzero_to_v306_point0": changed_vs_v300
            & v300["pointId"].astype(int).ne(0)
            & v306["pointId"].astype(int).eq(0),
            "action_changed": v300["actionId"].astype(int).ne(v306["actionId"].astype(int)),
            "v306_terminal_compatible": terminal_compatibility_mask(
                v306["pointId"].astype(int).to_numpy(), v306["actionId"].astype(int).to_numpy()
            ),
        }
    )
    return rows[row_mask].reset_index(drop=True)


def _candidate_actions(
    base_action: np.ndarray,
    action_prob: np.ndarray,
    point0_mask: np.ndarray,
    defensive_anchor: np.ndarray | None = None,
) -> pd.DataFrame:
    base_action = np.asarray(base_action, dtype=int)
    point0_mask = np.asarray(point0_mask, dtype=bool)
    prob = normalize_rows_safe(action_prob)
    if len(base_action) != len(prob) or len(base_action) != len(point0_mask):
        raise ValueError("base action, probability, and point0 mask lengths must match")
    clipped_base = np.clip(base_action, 0, prob.shape[1] - 1)
    base_prob = prob[np.arange(len(prob)), clipped_base]
    terminal_margin = prob[:, 0] - base_prob
    block_margin = prob[:, 13] - base_prob
    defensive_ok = np.isin(base_action, list(DEFENSIVE_FAMILY))
    if defensive_anchor is not None:
        defensive_ok = defensive_ok & np.isin(np.asarray(defensive_anchor, dtype=int), list(DEFENSIVE_FAMILY))

    terminal_ok = (
        point0_mask
        & (base_action != 0)
        & (prob[:, 0] >= TERMINAL_ACTION_PROB_MIN)
        & (terminal_margin >= TERMINAL_ACTION_MARGIN_MIN)
    )
    block_ok = (
        point0_mask
        & (base_action != 13)
        & defensive_ok
        & (prob[:, 13] >= BLOCK_ACTION_PROB_MIN)
        & (block_margin >= BLOCK_ACTION_MARGIN_MIN)
    )

    rows: list[dict[str, Any]] = []
    for row_id in np.where(terminal_ok | block_ok)[0]:
        if terminal_ok[row_id]:
            rows.append(
                {
                    "row_id": int(row_id),
                    "candidate_action": 0,
                    "candidate_type": "action0_terminal",
                    "base_action": int(base_action[row_id]),
                    "candidate_prob": float(prob[row_id, 0]),
                    "base_prob": float(base_prob[row_id]),
                    "action_margin": float(terminal_margin[row_id]),
                }
            )
        if block_ok[row_id]:
            rows.append(
                {
                    "row_id": int(row_id),
                    "candidate_action": 13,
                    "candidate_type": "action13_defensive",
                    "base_action": int(base_action[row_id]),
                    "candidate_prob": float(prob[row_id, 13]),
                    "base_prob": float(base_prob[row_id]),
                    "action_margin": float(block_margin[row_id]),
                }
            )
    columns = ["row_id", "candidate_action", "candidate_type", "base_action", "candidate_prob", "base_prob", "action_margin"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _select_candidates(candidates: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    selected = candidates.sort_values(
        ["action_margin", "candidate_prob", "row_id"], ascending=[False, False, True]
    ).drop_duplicates("row_id")
    return selected.head(max_rows).reset_index(drop=True)


def _apply_action_changes(base_action: np.ndarray, selected: pd.DataFrame) -> np.ndarray:
    out = np.asarray(base_action, dtype=int).copy()
    for row in selected.itertuples(index=False):
        out[int(row.row_id)] = int(row.candidate_action)
    return out


def _action_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def decision_label(action_oof_delta: float, changed_action_rows: int) -> str:
    if float(action_oof_delta) >= 0.0015 and int(changed_action_rows) <= 10:
        return "REVIEW_ACTION"
    return "DO_NOT_UPLOAD"


def _write_submission(anchor: pd.DataFrame, point: np.ndarray, action: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out["actionId"] = np.asarray(action, dtype=int)
    out = validate_submission_schema(out)
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def _write_report_md(report: dict[str, Any]) -> None:
    best = report["best_candidate"]
    lines = [
        "# V310 Terminal Action-Point Consistency",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Decision: `{best.get('decision', 'DO_NOT_UPLOAD')}`",
        f"- V306 changed rows found: `{report['v306_changed_rows']}`",
        f"- V306 nonzero->point0 rows: `{report['v306_nonzero_to_point0_rows']}`",
        f"- Best action OOF delta: `{float(best.get('action_oof_delta', 0.0)):.6f}`",
        f"- Best changed action rows: `{int(best.get('changed_action_rows', 0))}`",
        "",
        "Conclusion: action labels should remain unchanged unless the REVIEW_ACTION gate is met.",
    ]
    (OUTDIR / "v310_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    v300 = _load_submission(V300_SUBMISSION)
    v306 = _load_submission(V306_SUBMISSION)

    artifacts = load_artifacts()
    y_point, model_oof_prob, model_test_prob, folds = build_v261_literal_probabilities(artifacts)
    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_base_point = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base_point = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    oof_budget = int(np.floor(len(oof_base_point) * V306_CAP))
    test_budget = int(np.floor(len(test_base_point) * V306_CAP))
    oof_point, oof_p0_rows, _ = apply_point0_additions(oof_base_point, model_oof_prob, oof_budget)
    test_point, test_p0_rows, _ = apply_point0_additions(test_base_point, model_test_prob, test_budget)
    changed_rows = identify_v306_changed_rows(v300, v306, test_p0_rows)
    changed_rows.to_csv(OUTDIR / "v310_changed_rows.csv", index=False)

    action_oof_prob = _load_action_prob(ACTION_OOF_PROB, len(oof_base_point))
    action_test_prob = _load_action_prob(ACTION_TEST_PROB, len(test_base_point))
    y_action = artifacts["meta"]["next_actionId"].astype(int).to_numpy() if "next_actionId" in artifacts["meta"].columns else None
    if y_action is None or len(y_action) != len(action_oof_prob):
        from analysis_v261_action_conditioned_point_residual import build_frames

        train_df, _, _ = build_frames()
        y_action = train_df["next_actionId"].astype(int).to_numpy()
    oof_base_action = action_oof_prob.argmax(axis=1).astype(int)
    test_base_action = v300["actionId"].astype(int).to_numpy()
    v173_action = _load_submission(V173_ACTION_SUBMISSION)["actionId"].astype(int).to_numpy() if V173_ACTION_SUBMISSION.exists() else None

    oof_candidates = _candidate_actions(oof_base_action, action_oof_prob, oof_p0_rows)
    test_candidates = _candidate_actions(test_base_action, action_test_prob, test_p0_rows, v173_action)
    selected_oof = _select_candidates(oof_candidates, max_rows=10)
    selected_test = _select_candidates(test_candidates, max_rows=10)

    baseline_action_score = _action_score(y_action, oof_base_action)
    candidate_oof_action = _apply_action_changes(oof_base_action, selected_oof)
    candidate_action_score = _action_score(y_action, candidate_oof_action)
    action_oof_delta = candidate_action_score - baseline_action_score
    decision = decision_label(action_oof_delta, len(selected_test))

    records = [
        {
            "candidate": "keep_action_unchanged",
            "action_macro_f1": baseline_action_score,
            "action_oof_delta": 0.0,
            "oof_changed_action_rows": 0,
            "changed_action_rows": 0,
            "test_point0_rows": int(test_p0_rows.sum()),
            "decision": "BASELINE",
            "submission": "submission_v310_v306point_keep_v300action_v300server.csv",
        },
        {
            "candidate": "terminal_gated_action_edits",
            "action_macro_f1": candidate_action_score,
            "action_oof_delta": action_oof_delta,
            "oof_changed_action_rows": int(len(selected_oof)),
            "changed_action_rows": int(len(selected_test)),
            "test_point0_rows": int(test_p0_rows.sum()),
            "decision": decision,
            "submission": "submission_v310_v306point_terminal_action_edits_v300server.csv" if decision == "REVIEW_ACTION" else "",
        },
    ]
    search = pd.DataFrame(records)
    search.to_csv(OUTDIR / "v310_action_point_consistency_search.csv", index=False)

    baseline_path = _write_submission(
        v300,
        v306["pointId"].astype(int).to_numpy(),
        test_base_action,
        "submission_v310_v306point_keep_v300action_v300server.csv",
    )
    submissions = [{"candidate": "keep_action_unchanged", "path": baseline_path}]
    if decision == "REVIEW_ACTION":
        test_action = _apply_action_changes(test_base_action, selected_test)
        path = _write_submission(v300, v306["pointId"].astype(int).to_numpy(), test_action, records[1]["submission"])
        submissions.append({"candidate": "terminal_gated_action_edits", "path": path})

    if not selected_test.empty:
        selected_test = selected_test.copy()
        selected_test["rally_uid"] = v300.loc[selected_test["row_id"], "rally_uid"].astype(int).to_numpy()
        selected_test["v306_pointId"] = v306.loc[selected_test["row_id"], "pointId"].astype(int).to_numpy()
    selected_test.to_csv(OUTDIR / "v310_test_action_candidates.csv", index=False)
    selected_oof.to_csv(OUTDIR / "v310_oof_action_candidates.csv", index=False)

    best = records[1]
    report = {
        "version": "V310",
        "verdict": "HAS_REVIEW_ACTION_CANDIDATE" if decision == "REVIEW_ACTION" else "NO_ACTION_UPLOAD_WORTHY_CANDIDATE",
        "best_candidate": best,
        "v306_changed_rows": int(len(changed_rows)),
        "v306_nonzero_to_point0_rows": int(test_p0_rows.sum()),
        "v300_nonzero_to_v306_point0_rows": int(changed_rows["v300_nonzero_to_v306_point0"].sum()) if not changed_rows.empty else 0,
        "action_probability_source": {"oof": str(ACTION_OOF_PROB), "test": str(ACTION_TEST_PROB)},
        "v173_action_anchor": str(V173_ACTION_SUBMISSION) if V173_ACTION_SUBMISSION.exists() else None,
        "submissions": submissions,
        "folds": folds,
        "notes": [
            "Action edits are local research only.",
            "Action0 candidates require point0 rows, high action0 probability, and positive margin.",
            "Action13 candidates require the current/V173 action to already be in the defensive family.",
            "Decision gate is REVIEW_ACTION only when action OOF delta >= 0.0015 and changed action rows <= 10.",
        ],
    }
    (OUTDIR / "v310_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report_md(report)
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "verdict": report["verdict"],
                "decision": decision,
                "action_oof_delta": action_oof_delta,
                "changed_action_rows": len(selected_test),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
