"""V362 point hierarchical specialist factory.

This experiment builds point-only candidate submissions against the V338 public
positive anchor. It uses existing clean point submissions as row-level evidence
and a small train-derived depth backoff as a conservative support signal.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v362_point_hierarchical_specialists"
ANCHOR_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
TRAIN_PATHS = (ROOT / "train.csv", ROOT / "data" / "raw" / "train.csv")
TEST_PATHS = (ROOT / "test_new.csv", ROOT / "data" / "raw" / "test_new.csv")
SOURCE_DIRS = (
    "v306_point0_addition_probe",
    "v307_point0_dose_extension",
    "v341_no_p0_point_pack",
    "v345_nonpoint0_utility_optimizer",
)
V343_BANK_PATH = ROOT / "v343_row_candidate_bank" / "candidate_bank.csv"
V345_SCORED_PATH = ROOT / "v345_nonpoint0_utility_optimizer" / "scored_candidates.csv"
RARE_POINTS = {1, 3, 4, 6, 7, 9}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def point_to_depth(point: int) -> str:
    value = int(point)
    if value == 0:
        return "terminal"
    if 1 <= value <= 3:
        return "short"
    if 4 <= value <= 6:
        return "half"
    if 7 <= value <= 9:
        return "long"
    raise ValueError(f"pointId outside 0..9: {point}")


def point_to_side(point: int) -> str:
    value = int(point)
    if value == 0:
        return "terminal"
    if not 1 <= value <= 9:
        raise ValueError(f"pointId outside 0..9: {point}")
    return {1: "left", 2: "middle", 0: "right"}[value % 3]


def prefix_len_bin(prefix_len: int) -> str:
    value = int(prefix_len)
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value <= 6:
        return "4_6"
    return "7p"


def apply_no_p0_policy(
    base: pd.Series,
    proposed: pd.Series,
    allow_existing_zero: bool = True,
) -> pd.Series:
    base_i = pd.Series(base).astype(int).reset_index(drop=True)
    prop_i = pd.Series(proposed).astype(int).reset_index(drop=True)
    if len(base_i) != len(prop_i):
        raise ValueError("base and proposed point series length mismatch")

    out = prop_i.copy()
    proposed_zero = out.eq(0)
    if allow_existing_zero:
        blocked = proposed_zero & base_i.ne(0)
    else:
        blocked = proposed_zero
    out.loc[blocked] = base_i.loc[blocked]
    out.index = pd.Series(proposed).index
    out.name = getattr(proposed, "name", None)
    return out


def package_point_candidate(anchor: pd.DataFrame, point_pred: pd.Series) -> pd.DataFrame:
    if len(anchor) != len(point_pred):
        raise ValueError("anchor and point prediction length mismatch")
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["pointId"] = pd.Series(point_pred).to_numpy(dtype=int)
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("actionId changed while packaging point candidate")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("serverGetPoint changed while packaging point candidate")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def load_anchor_submission(expected_rows: int | None = 1845) -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"missing V338 anchor submission: {ANCHOR_PATH}")
    return load_submission(ANCHOR_PATH, expected_rows=expected_rows)


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def build_depth_backoff(anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = first_existing(TRAIN_PATHS)
    test_path = first_existing(TEST_PATHS)
    if train_path is None or test_path is None:
        empty_pred = pd.DataFrame(
            {
                "row_id": np.arange(len(anchor)),
                "depth_pred": [point_to_depth(v) for v in anchor["pointId"].astype(int)],
                "side_pred": [point_to_side(v) for v in anchor["pointId"].astype(int)],
                "point_pred": anchor["pointId"].astype(int).to_numpy(),
                "support": 0,
                "source": "anchor_fallback",
            }
        )
        return pd.DataFrame(), empty_pred, {"available": False, "reason": "missing_train_or_test"}

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        empty_pred = pd.DataFrame(
            {
                "row_id": np.arange(len(anchor)),
                "depth_pred": [point_to_depth(v) for v in anchor["pointId"].astype(int)],
                "side_pred": [point_to_side(v) for v in anchor["pointId"].astype(int)],
                "point_pred": anchor["pointId"].astype(int).to_numpy(),
                "support": 0,
                "source": "anchor_fallback",
            }
        )
        return pd.DataFrame(), empty_pred, {"available": False, "reason": "missing_required_columns"}

    train = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    train["next_rally_uid"] = train["rally_uid"].shift(-1)
    train["target_point"] = train["pointId"].shift(-1)
    pairs = train[train["rally_uid"].eq(train["next_rally_uid"])].copy()
    pairs["target_point"] = pairs["target_point"].astype(int)
    pairs["context_depth"] = pairs["pointId"].astype(int).map(point_to_depth)
    pairs["context_side"] = pairs["pointId"].astype(int).map(point_to_side)
    pairs["phase"] = pairs["strikeNumber"].astype(int).map(prefix_len_bin)
    pairs["target_depth"] = pairs["target_point"].map(point_to_depth)
    pairs["target_side"] = pairs["target_point"].map(point_to_side)

    global_point = int(pairs["target_point"].mode().iloc[0]) if not pairs.empty else 8
    rows: list[dict[str, Any]] = []
    for keys, group in pairs.groupby(["actionId", "context_depth", "phase"], sort=False):
        action_id, context_depth, phase = keys
        point_mode = int(group["target_point"].mode().iloc[0])
        rows.append(
            {
                "actionId": int(action_id),
                "context_depth": str(context_depth),
                "phase": str(phase),
                "point_pred": point_mode,
                "depth_pred": point_to_depth(point_mode),
                "side_pred": point_to_side(point_mode),
                "support": int(len(group)),
            }
        )
    table = pd.DataFrame(rows)
    lookup = {
        (int(row.actionId), str(row.context_depth), str(row.phase)): row
        for row in table.itertuples(index=False)
    }

    pred_rows: list[dict[str, Any]] = []
    test_aligned = test.reset_index(drop=True)
    if len(test_aligned) != len(anchor) or not test_aligned["rally_uid"].reset_index(drop=True).equals(
        anchor["rally_uid"].reset_index(drop=True)
    ):
        test_aligned = pd.DataFrame(
            {
                "actionId": anchor["actionId"].astype(int),
                "pointId": anchor["pointId"].astype(int),
                "strikeNumber": np.ones(len(anchor), dtype=int),
            }
        )

    for row_id, row in test_aligned.iterrows():
        context_depth = point_to_depth(int(row["pointId"]))
        phase = prefix_len_bin(int(row.get("strikeNumber", 1)))
        key = (int(row["actionId"]), context_depth, phase)
        found = lookup.get(key)
        if found is None or int(found.support) < 5:
            point_pred = global_point
            support = 0
            source = "global"
        else:
            point_pred = int(found.point_pred)
            support = int(found.support)
            source = "action_depth_phase"
        pred_rows.append(
            {
                "row_id": int(row_id),
                "depth_pred": point_to_depth(point_pred),
                "side_pred": point_to_side(point_pred),
                "point_pred": point_pred,
                "support": support,
                "source": source,
            }
        )
    report = {
        "available": True,
        "train_path": relative_path(train_path),
        "test_path": relative_path(test_path),
        "table_rows": int(len(table)),
        "policy": "train_next-row depth/side backoff by actionId, lag0 point depth, phase bin",
    }
    return table, pd.DataFrame(pred_rows), report


def load_bank_support() -> tuple[set[tuple[int, int]], dict[str, Any]]:
    support: set[tuple[int, int]] = set()
    reports: list[dict[str, Any]] = []
    if V343_BANK_PATH.exists():
        bank = pd.read_csv(V343_BANK_PATH)
        if {"row_id", "candidate_value"}.issubset(bank.columns):
            for row in bank[["row_id", "candidate_value"]].dropna().itertuples(index=False):
                support.add((int(row.row_id), int(row.candidate_value)))
            reports.append({"path": relative_path(V343_BANK_PATH), "rows": int(len(bank))})
    if V345_SCORED_PATH.exists():
        scored = pd.read_csv(V345_SCORED_PATH)
        if {"row_id", "candidate_value"}.issubset(scored.columns):
            for row in scored[["row_id", "candidate_value"]].dropna().itertuples(index=False):
                support.add((int(row.row_id), int(row.candidate_value)))
            reports.append({"path": relative_path(V345_SCORED_PATH), "rows": int(len(scored))})
    return support, {"sources": reports, "supported_row_candidates": len(support)}


def source_risk(source_dir: str, source_name: str) -> str:
    text = f"{source_dir}/{source_name}".lower()
    if "v341" in text:
        return "high"
    if "v306" in text or "v307" in text:
        return "normal"
    if "v345" in text:
        return "normal"
    return "unknown"


def load_source_votes(anchor: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    base_point = anchor["pointId"].astype(int).to_numpy()

    for source_dir_name in SOURCE_DIRS:
        source_dir = ROOT / source_dir_name
        if not source_dir.exists():
            skipped.append({"source_dir": source_dir_name, "reason": "missing_dir"})
            continue
        for path in sorted(source_dir.glob("submission*.csv")):
            try:
                frame = load_submission(path, expected_rows=len(anchor))
            except Exception as exc:
                skipped.append({"path": relative_path(path), "reason": f"schema:{exc}"})
                continue
            if not frame["rally_uid"].reset_index(drop=True).equals(anchor["rally_uid"].reset_index(drop=True)):
                skipped.append({"path": relative_path(path), "reason": "rally_uid_mismatch"})
                continue
            if not frame["actionId"].reset_index(drop=True).equals(anchor["actionId"].reset_index(drop=True)):
                skipped.append({"path": relative_path(path), "reason": "action_changed"})
                continue
            if not frame["serverGetPoint"].reset_index(drop=True).equals(
                anchor["serverGetPoint"].reset_index(drop=True)
            ):
                skipped.append({"path": relative_path(path), "reason": "server_changed"})
                continue
            cand_point = frame["pointId"].astype(int).to_numpy()
            changed = np.flatnonzero(cand_point != base_point)
            scanned.append({"path": relative_path(path), "changed_vs_v338": int(len(changed))})
            for row_id in changed:
                old = int(base_point[row_id])
                new = int(cand_point[row_id])
                rows.append(
                    {
                        "row_id": int(row_id),
                        "rally_uid": anchor.at[int(row_id), "rally_uid"],
                        "base_point": old,
                        "candidate_point": new,
                        "transition": f"{old}->{new}",
                        "source_dir": source_dir_name,
                        "source": path.stem,
                        "source_risk": source_risk(source_dir_name, path.stem),
                        "base_depth": point_to_depth(old),
                        "candidate_depth": point_to_depth(new),
                        "base_side": point_to_side(old),
                        "candidate_side": point_to_side(new),
                        "same_depth": point_to_depth(old) == point_to_depth(new),
                        "same_side": point_to_side(old) == point_to_side(new),
                        "is_point0_addition": bool(old != 0 and new == 0),
                        "is_point0_removal": bool(old == 0 and new != 0),
                        "is_nonterminal_swap": bool(old != 0 and new != 0),
                        "is_rare_point_repair": bool(new in RARE_POINTS or old in RARE_POINTS),
                    }
                )
    report = {"scanned": scanned, "skipped": skipped, "vote_rows": len(rows)}
    return pd.DataFrame(rows), report


def score_votes(votes: pd.DataFrame, depth_pred: pd.DataFrame, bank_support: set[tuple[int, int]]) -> pd.DataFrame:
    if votes.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "rally_uid",
                "base_point",
                "candidate_point",
                "transition",
                "score",
                "agreement_count",
            ]
        )
    pred = depth_pred.set_index("row_id").to_dict(orient="index")
    grouped_rows: list[dict[str, Any]] = []
    keys = ["row_id", "rally_uid", "base_point", "candidate_point"]
    for values, group in votes.groupby(keys, sort=False):
        row_id, rally_uid, base_point, candidate_point = values
        source_dirs = sorted(group["source_dir"].astype(str).unique().tolist())
        sources = sorted(group["source"].astype(str).unique().tolist())
        risks = sorted(group["source_risk"].astype(str).unique().tolist())
        agreement = len(sources)
        row_pred = pred.get(int(row_id), {})
        depth_agree = row_pred.get("depth_pred") == point_to_depth(int(candidate_point))
        side_agree = row_pred.get("side_pred") == point_to_side(int(candidate_point))
        bank_agree = (int(row_id), int(candidate_point)) in bank_support
        v341_only = risks == ["high"]
        score = 0.0
        score += agreement * 1.0
        score += len(source_dirs) * 0.35
        score += 1.25 if bank_agree else 0.0
        score += 0.8 if depth_agree else 0.0
        score += 0.25 if side_agree else 0.0
        score += 0.4 if bool(group["same_depth"].any()) else 0.0
        score += 0.35 if bool(group["is_rare_point_repair"].any()) else 0.0
        score -= 2.5 if bool(group["is_point0_addition"].any()) else 0.0
        score -= 0.8 if v341_only else 0.0
        grouped_rows.append(
            {
                "row_id": int(row_id),
                "rally_uid": rally_uid,
                "base_point": int(base_point),
                "candidate_point": int(candidate_point),
                "transition": f"{int(base_point)}->{int(candidate_point)}",
                "base_depth": point_to_depth(int(base_point)),
                "candidate_depth": point_to_depth(int(candidate_point)),
                "base_side": point_to_side(int(base_point)),
                "candidate_side": point_to_side(int(candidate_point)),
                "same_depth": bool(group["same_depth"].any()),
                "same_side": bool(group["same_side"].any()),
                "is_point0_addition": bool(group["is_point0_addition"].any()),
                "is_point0_removal": bool(group["is_point0_removal"].any()),
                "is_nonterminal_swap": bool(group["is_nonterminal_swap"].any()),
                "is_rare_point_repair": bool(group["is_rare_point_repair"].any()),
                "agreement_count": int(agreement),
                "source_dir_count": int(len(source_dirs)),
                "source_dirs": "|".join(source_dirs),
                "sources": "|".join(sources),
                "source_risks": "|".join(risks),
                "bank_agree": bool(bank_agree),
                "depth_agree": bool(depth_agree),
                "side_agree": bool(side_agree),
                "depth_support": int(row_pred.get("support", 0) or 0),
                "score": float(score),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(
        ["score", "agreement_count", "source_dir_count", "row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )


def select_unique_rows(scored: pd.DataFrame, budget: int | None = None) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    ordered = scored.sort_values(
        ["score", "agreement_count", "source_dir_count", "row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = ordered.drop_duplicates("row_id", keep="first")
    if budget is not None:
        selected = selected.head(int(budget))
    return selected.reset_index(drop=True)


def build_point_prediction(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    proposed = anchor["pointId"].astype(int).copy()
    for row in selected.itertuples(index=False):
        proposed.iat[int(row.row_id)] = int(row.candidate_point)
    return apply_no_p0_policy(anchor["pointId"], proposed, allow_existing_zero=True)


def transition_counts(base: pd.Series, candidate: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for old, new in zip(base.astype(int), candidate.astype(int)):
        if old != new:
            counts[f"{int(old)}->{int(new)}"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def audit_candidate(anchor: pd.DataFrame, candidate: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    point_report = point_distribution_report(anchor["pointId"], candidate["pointId"])
    return {
        "selected_rows": int(len(selected)),
        "point_churn_vs_v338": int(point_report["changed_rows"]),
        "point0_additions": int(point_report["point0_additions"]),
        "point0_removals": int(point_report["point0_removals"]),
        "transition_counts": json.dumps(transition_counts(anchor["pointId"], candidate["pointId"]), sort_keys=True),
        "action_preserved": bool(candidate["actionId"].equals(anchor["actionId"])),
        "server_preserved": bool(candidate["serverGetPoint"].equals(anchor["serverGetPoint"])),
        "agreement_count_min": int(selected["agreement_count"].min()) if not selected.empty else 0,
        "agreement_count_mean": float(selected["agreement_count"].mean()) if not selected.empty else 0.0,
        "score_sum": float(selected["score"].sum()) if not selected.empty else 0.0,
    }


def classify_candidate_risk(point_churn: int, point0_additions: int) -> str:
    if point0_additions == 0 and point_churn <= 5:
        return "safe"
    if point0_additions == 0 and point_churn <= 15:
        return "normal"
    return "research"


def candidate_specs(scored: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    no_p0 = scored[~scored["is_point0_addition"].astype(bool)].copy() if not scored.empty else scored.copy()
    nonterminal = no_p0[no_p0["is_nonterminal_swap"].astype(bool)].copy() if not no_p0.empty else no_p0.copy()
    specs: list[tuple[str, pd.DataFrame]] = []

    depth_agree = no_p0[
        no_p0["depth_agree"].astype(bool) & no_p0["bank_agree"].astype(bool)
    ].copy() if not no_p0.empty else no_p0.copy()
    specs.append(("v362_depth_agree_only", select_unique_rows(depth_agree, budget=12)))

    highconf = nonterminal[
        (nonterminal["agreement_count"].astype(int) >= 2)
        | nonterminal["bank_agree"].astype(bool)
        | nonterminal["depth_agree"].astype(bool)
    ].copy() if not nonterminal.empty else nonterminal.copy()
    specs.append(("v362_nonterminal_highconf_b12", select_unique_rows(highconf, budget=12)))
    specs.append(("v362_nonterminal_highconf_b24", select_unique_rows(highconf, budget=24)))

    rare = no_p0[no_p0["is_rare_point_repair"].astype(bool)].copy() if not no_p0.empty else no_p0.copy()
    specs.append(("v362_rare_point_repair", select_unique_rows(rare, budget=18)))

    research = no_p0[
        no_p0["depth_agree"].astype(bool) | no_p0["side_agree"].astype(bool) | no_p0["same_depth"].astype(bool)
    ].copy() if not no_p0.empty else no_p0.copy()
    specs.append(("v362_research_depth_side", select_unique_rows(research, budget=48)))
    return specs


def run_pipeline(expected_rows: int | None = 1845) -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor_submission(expected_rows=expected_rows)
    depth_table, depth_pred, depth_report = build_depth_backoff(anchor)
    depth_table.to_csv(safe_output_path(OUTDIR, "depth_backoff_table.csv"), index=False)
    depth_pred.to_csv(safe_output_path(OUTDIR, "depth_backoff_predictions.csv"), index=False)

    bank_support, bank_report = load_bank_support()
    votes, source_report = load_source_votes(anchor)
    votes.to_csv(safe_output_path(OUTDIR, "source_votes.csv"), index=False)
    scored = score_votes(votes, depth_pred, bank_support)
    scored.to_csv(safe_output_path(OUTDIR, "scored_candidates.csv"), index=False)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate, selected in candidate_specs(scored):
        point_pred = build_point_prediction(anchor, selected)
        submission = package_point_candidate(anchor, point_pred)
        filename = f"submission_{candidate}__v173action_v300server.csv"
        out_path = safe_output_path(OUTDIR, filename)
        submission.to_csv(out_path, index=False)
        selected_path = safe_output_path(OUTDIR, f"selected_{candidate}.csv")
        selected.to_csv(selected_path, index=False)
        audit = audit_candidate(anchor, submission, selected)
        risk_level = classify_candidate_risk(audit["point_churn_vs_v338"], audit["point0_additions"])
        row = {
            "candidate": candidate,
            "risk_level": risk_level,
            "path": relative_path(out_path),
            "selected_path": relative_path(selected_path),
            **audit,
        }
        summary_rows.append(row)
        generated.append({"candidate": candidate, "path": row["path"], "point_churn_vs_v338": row["point_churn_vs_v338"]})

    summary = pd.DataFrame(summary_rows)
    summary_path = safe_output_path(OUTDIR, "candidate_summary.csv")
    summary.to_csv(summary_path, index=False)
    recommended = None
    if not summary.empty:
        eligible = summary[
            summary["risk_level"].isin(["safe", "normal"])
            & summary["action_preserved"].astype(bool)
            & summary["server_preserved"].astype(bool)
            & summary["point0_additions"].eq(0)
        ].copy()
        if not eligible.empty:
            recommended = eligible.sort_values(
                ["risk_level", "score_sum", "point_churn_vs_v338"],
                ascending=[True, False, True],
                kind="mergesort",
            ).iloc[0].to_dict()

    report = {
        "version": "V362",
        "anchor": relative_path(ANCHOR_PATH),
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "generated_submission_count": len(generated),
        "generated_submissions": generated,
        "recommended_candidate": recommended,
        "candidate_summary": relative_path(summary_path),
        "scored_candidates": relative_path(safe_output_path(OUTDIR, "scored_candidates.csv")),
        "depth_backoff_report": depth_report,
        "bank_report": bank_report,
        "source_report": source_report,
        "policy": {
            "no_ttmatch_inputs": True,
            "no_old_server_labels": True,
            "manual_row_edits": False,
            "preserve_v338_action": True,
            "preserve_v300_server": True,
            "no_nonterminal_to_point0_exports": True,
            "output_dir": relative_path(OUTDIR),
        },
    }
    write_json(safe_output_path(OUTDIR, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            json_safe(
                {
                    "outdir": OUTDIR,
                    "decision": report["decision"],
                    "generated_submission_count": report["generated_submission_count"],
                    "generated_submissions": report["generated_submissions"],
                    "recommended_candidate": report["recommended_candidate"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
