"""V370 point breakthrough pool.

Builds a row-level point candidate bank against the current V362 clean anchor
and exports safe, medium, and aggressive point-only submissions.
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
OUTDIR = ROOT / "v370_point_breakthrough_pool"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
FALLBACK_ANCHOR_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"

SUBMISSION_SOURCE_DIRS = (
    "v362_point_hierarchical_specialists",
    "v338_joint_moe_pack",
    "v341_no_p0_point_pack",
    "v306_point0_addition_probe",
    "v307_point0_dose_extension",
    "v300_clean_server_blend_recycler",
    "v261_action_conditioned_point_residual",
    "v272_action_conditioned_point_residual",
    "v277_v272b_point_refinement",
    "v345_nonpoint0_utility_optimizer",
)
EVIDENCE_SOURCES = {
    "v362:scored_candidates": ROOT / "v362_point_hierarchical_specialists" / "scored_candidates.csv",
    "v343:candidate_bank": ROOT / "v343_row_candidate_bank" / "candidate_bank.csv",
    "v345:scored_candidates": ROOT / "v345_nonpoint0_utility_optimizer" / "scored_candidates.csv",
    "v354:row_evidence": ROOT / "v354_independent_row_evidence" / "row_evidence.csv",
}
CORE_FAMILIES = {"v338", "v362", "v341", "v300", "v261"}
POINT0_POSITIVE_FAMILIES = {"v306", "v307"}


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


def source_family(source_name: str) -> str:
    text = str(source_name).lower()
    for family in (
        "v362",
        "v338",
        "v341",
        "v340",
        "v339",
        "v306",
        "v307",
        "v300",
        "v261",
        "v272",
        "v277",
        "v343",
        "v345",
        "v354",
    ):
        if family in text:
            if family in {"v339", "v340"}:
                return "v341"
            return family
    return text.split(":", 1)[0].split("_", 1)[0] or "unknown"


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame.loc[:, SUBMISSION_COLUMNS], expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def load_anchor() -> pd.DataFrame:
    path = ANCHOR_PATH if ANCHOR_PATH.exists() else FALLBACK_ANCHOR_PATH
    if not path.exists():
        raise FileNotFoundError(f"missing V370 anchor: {ANCHOR_PATH}")
    return load_submission(path)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _source_allowed(path: Path) -> bool:
    text = path.as_posix().lower()
    return "ttmatch" not in text and "oldserver" not in text and "old_server" not in text


def load_candidate_sources() -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for dirname in SUBMISSION_SOURCE_DIRS:
        source_dir = ROOT / dirname
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("submission*.csv")):
            if not _source_allowed(path):
                continue
            frame = _safe_read_csv(path)
            if frame is None or not {"rally_uid", "pointId"}.issubset(frame.columns):
                continue
            key = f"{dirname}:{path.stem}"
            sources[key] = frame

    for key, path in EVIDENCE_SOURCES.items():
        if not path.exists() or not _source_allowed(path):
            continue
        frame = _safe_read_csv(path)
        if frame is not None:
            sources[key] = frame
    return sources


def _candidate_column(frame: pd.DataFrame) -> str | None:
    for col in ("candidate_point", "candidate_value", "new_point", "pointId"):
        if col in frame.columns:
            return col
    return None


def _base_column(frame: pd.DataFrame) -> str | None:
    for col in ("base_point", "anchor_value", "old_point"):
        if col in frame.columns:
            return col
    return None


def _iter_source_votes(
    anchor: pd.DataFrame,
    source_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidate_col = _candidate_column(frame)
    if candidate_col is None:
        return []

    anchor_reset = anchor.reset_index(drop=True)
    family = source_family(source_name)
    rows: list[dict[str, Any]] = []
    if candidate_col == "pointId" and "rally_uid" in frame.columns:
        src = frame.reset_index(drop=True)
        if len(src) == len(anchor_reset) and src["rally_uid"].equals(anchor_reset["rally_uid"]):
            aligned = src
            row_ids = range(len(aligned))
        else:
            aligned = anchor_reset[["rally_uid"]].merge(
                src[["rally_uid", "pointId"]],
                on="rally_uid",
                how="left",
                sort=False,
            )
            row_ids = range(len(aligned))
        for row_id in row_ids:
            value = aligned.at[int(row_id), "pointId"]
            if pd.isna(value):
                continue
            candidate = int(value)
            base = int(anchor_reset.at[int(row_id), "pointId"])
            if candidate == base:
                continue
            rows.append(
                {
                    "row_id": int(row_id),
                    "rally_uid": anchor_reset.at[int(row_id), "rally_uid"],
                    "base_point": base,
                    "candidate_point": candidate,
                    "source": source_name,
                    "source_family": family,
                }
            )
        return rows

    if "row_id" not in frame.columns:
        return []
    base_col = _base_column(frame)
    for raw in frame.dropna(subset=["row_id", candidate_col]).itertuples(index=False):
        row = raw._asdict()
        row_id = int(row["row_id"])
        if row_id < 0 or row_id >= len(anchor_reset):
            continue
        candidate = int(row[candidate_col])
        base = int(anchor_reset.at[row_id, "pointId"])
        if base_col is not None and not pd.isna(row.get(base_col)):
            observed_base = int(row[base_col])
            if observed_base != base and family not in {"v362", "v354"}:
                continue
        if candidate == base:
            continue
        rows.append(
            {
                "row_id": row_id,
                "rally_uid": anchor_reset.at[row_id, "rally_uid"],
                "base_point": base,
                "candidate_point": candidate,
                "source": source_name,
                "source_family": family,
            }
        )
    return rows


def collect_row_candidates(
    anchor: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    vote_rows: list[dict[str, Any]] = []
    for source_name, frame in sources.items():
        source_votes = _iter_source_votes(anchor, source_name, frame)
        seen: set[tuple[int, int, str]] = set()
        for row in source_votes:
            key = (int(row["row_id"]), int(row["candidate_point"]), str(row["source"]))
            if key in seen:
                continue
            seen.add(key)
            vote_rows.append(row)
    if not vote_rows:
        return pd.DataFrame(
            columns=[
                "row_id",
                "rally_uid",
                "base_point",
                "candidate_point",
                "support_count",
            ]
        )

    votes = pd.DataFrame(vote_rows)
    grouped: list[dict[str, Any]] = []
    keys = ["row_id", "rally_uid", "base_point", "candidate_point"]
    for values, group in votes.groupby(keys, sort=False):
        row_id, rally_uid, base_point, candidate_point = values
        sources_for_row = sorted(group["source"].astype(str).unique().tolist())
        families = sorted(group["source_family"].astype(str).unique().tolist())
        point0_support = sorted(set(families) & POINT0_POSITIVE_FAMILIES)
        base = int(base_point)
        candidate = int(candidate_point)
        grouped.append(
            {
                "row_id": int(row_id),
                "rally_uid": rally_uid,
                "base_point": base,
                "candidate_point": candidate,
                "transition": f"{base}->{candidate}",
                "base_depth": point_to_depth(base),
                "candidate_depth": point_to_depth(candidate),
                "base_side": point_to_side(base),
                "candidate_side": point_to_side(candidate),
                "same_depth": point_to_depth(base) == point_to_depth(candidate),
                "same_side": point_to_side(base) == point_to_side(candidate),
                "is_point0_addition": bool(base != 0 and candidate == 0),
                "is_point0_removal": bool(base == 0 and candidate != 0),
                "is_nonterminal_swap": bool(base != 0 and candidate != 0),
                "support_count": int(len(sources_for_row)),
                "source_family_count": int(len(families)),
                "source_families": "|".join(families),
                "sources": "|".join(sources_for_row),
                "point0_support_count": int(len(point0_support)),
                "v338_family_support": bool("v338" in families),
                "v362_family_support": bool("v362" in families),
                "v341_family_support": bool("v341" in families),
                "v300_family_support": bool("v300" in families),
                "v261_family_support": bool("v261" in families),
                "v354_evidence": bool("v354" in families),
                "v343_evidence": bool("v343" in families),
                "v345_evidence": bool("v345" in families),
                "v341_expansion_risk": bool(families == ["v341"]),
            }
        )
    return pd.DataFrame(grouped)


def apply_point0_support_policy(
    rows: pd.DataFrame,
    min_point0_support: int = 2,
) -> pd.DataFrame:
    out = rows.copy()
    base = out["base_point"].astype(int)
    candidate = out["candidate_point"].astype(int)
    support = pd.to_numeric(out.get("point0_support_count", 0), errors="coerce").fillna(0)
    p0_addition = base.ne(0) & candidate.eq(0)
    out["allowed"] = (~p0_addition) | support.ge(int(min_point0_support))
    return out


def _train_transition_support(anchor: pd.DataFrame) -> pd.DataFrame:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        return pd.DataFrame(columns=["row_id", "candidate_point", "train_backoff_support"])
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        return pd.DataFrame(columns=["row_id", "candidate_point", "train_backoff_support"])

    train = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    train["next_rally_uid"] = train["rally_uid"].shift(-1)
    train["target_point"] = train["pointId"].shift(-1)
    pairs = train[train["rally_uid"].eq(train["next_rally_uid"])].copy()
    if pairs.empty:
        return pd.DataFrame(columns=["row_id", "candidate_point", "train_backoff_support"])
    pairs["phase"] = pairs["strikeNumber"].astype(int).map(prefix_len_bin)
    counts = (
        pairs.groupby(["actionId", "pointId", "phase", "target_point"], dropna=False)
        .size()
        .rename("train_backoff_support")
        .reset_index()
    )
    counts["actionId"] = counts["actionId"].astype(int)
    counts["pointId"] = counts["pointId"].astype(int)
    counts["target_point"] = counts["target_point"].astype(int)

    test_aligned = test.reset_index(drop=True)
    anchor_reset = anchor.reset_index(drop=True)
    if len(test_aligned) != len(anchor_reset) or not test_aligned["rally_uid"].equals(
        anchor_reset["rally_uid"]
    ):
        test_aligned = anchor_reset[["rally_uid", "actionId", "pointId"]].copy()
        test_aligned["strikeNumber"] = 1
    contexts = pd.DataFrame(
        {
            "row_id": np.arange(len(anchor_reset), dtype=int),
            "actionId": test_aligned["actionId"].astype(int).to_numpy(),
            "pointId": anchor_reset["pointId"].astype(int).to_numpy(),
            "phase": test_aligned["strikeNumber"].astype(int).map(prefix_len_bin).to_numpy(),
        }
    )
    expanded = contexts.merge(counts, on=["actionId", "pointId", "phase"], how="left")
    expanded = expanded.rename(columns={"target_point": "candidate_point"})
    expanded = expanded.dropna(subset=["candidate_point"])
    expanded["candidate_point"] = expanded["candidate_point"].astype(int)
    expanded["train_backoff_support"] = expanded["train_backoff_support"].astype(int)
    return expanded[["row_id", "candidate_point", "train_backoff_support"]]


def _attach_train_support(bank: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    out = bank.copy()
    support = _train_transition_support(anchor)
    if support.empty:
        out["train_backoff_support"] = 0
        return out
    out = out.merge(support, on=["row_id", "candidate_point"], how="left")
    out["train_backoff_support"] = out["train_backoff_support"].fillna(0).astype(int)
    return out


def rank_point_rows(bank: pd.DataFrame) -> pd.DataFrame:
    if bank.empty:
        out = bank.copy()
        out["score"] = pd.Series(dtype=float)
        return out

    out = bank.copy()
    support = pd.to_numeric(out["support_count"], errors="coerce").fillna(0)
    family_count = pd.to_numeric(out.get("source_family_count", 0), errors="coerce").fillna(0)
    train_support = pd.to_numeric(out.get("train_backoff_support", 0), errors="coerce").fillna(0)
    core_count = sum(out.get(f"{family}_family_support", False).astype(bool) for family in CORE_FAMILIES)

    score = support * 1.0
    score += family_count * 0.40
    score += core_count * 0.70
    score += out.get("same_depth", False).astype(bool) * 0.55
    score += out.get("same_side", False).astype(bool) * 0.20
    score += out.get("v354_evidence", False).astype(bool) * 1.50
    score += out.get("v343_evidence", False).astype(bool) * 0.80
    score += out.get("v345_evidence", False).astype(bool) * 0.90
    score += np.minimum(train_support, 25) * 0.04
    score -= out.get("is_point0_addition", False).astype(bool) * 2.75
    score -= out.get("v341_expansion_risk", False).astype(bool) * 1.20
    score -= train_support.eq(0) * 0.35
    out["score"] = score.astype(float)
    return out.sort_values(
        ["score", "support_count", "source_family_count", "row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def package_point_submission(anchor: pd.DataFrame, point_pred: pd.Series) -> pd.DataFrame:
    if len(anchor) != len(point_pred):
        raise ValueError("anchor and point prediction length mismatch")
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["pointId"] = pd.Series(point_pred).to_numpy(dtype=int)
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("actionId changed while packaging point submission")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("serverGetPoint changed while packaging point submission")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def select_unique_rows(scored: pd.DataFrame, budget: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    allowed = scored.sort_values(
        ["score", "support_count", "source_family_count", "row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return allowed.drop_duplicates("row_id", keep="first").head(int(budget)).reset_index(drop=True)


def build_point_prediction(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    point_pred = anchor["pointId"].astype(int).copy().reset_index(drop=True)
    for row in selected.itertuples(index=False):
        point_pred.iat[int(row.row_id)] = int(row.candidate_point)
    return point_pred


def transition_counts(base: pd.Series, candidate: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for old, new in zip(base.astype(int), candidate.astype(int)):
        if int(old) != int(new):
            counts[f"{int(old)}->{int(new)}"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def audit_candidate(anchor: pd.DataFrame, candidate: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    report = point_distribution_report(anchor["pointId"], candidate["pointId"])
    return {
        "selected_rows": int(len(selected)),
        "point_churn_vs_v362": int(report["changed_rows"]),
        "point0_additions": int(report["point0_additions"]),
        "point0_removals": int(report["point0_removals"]),
        "transition_counts": json.dumps(
            transition_counts(anchor["pointId"], candidate["pointId"]),
            sort_keys=True,
        ),
        "action_preserved": bool(candidate["actionId"].equals(anchor["actionId"])),
        "server_preserved": bool(candidate["serverGetPoint"].equals(anchor["serverGetPoint"])),
        "support_count_min": int(selected["support_count"].min()) if not selected.empty else 0,
        "support_count_mean": float(selected["support_count"].mean()) if not selected.empty else 0.0,
        "score_sum": float(selected["score"].sum()) if not selected.empty else 0.0,
    }


def _write_submission(filename: str, frame: pd.DataFrame) -> Path:
    path = safe_output_path(OUTDIR, filename)
    frame.to_csv(path, index=False, float_format="%.8f")
    return path


def candidate_specs(scored: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if scored.empty:
        return [
            ("safe", scored.copy()),
            ("medium", scored.copy()),
            ("aggressive", scored.copy()),
        ]
    policy = apply_point0_support_policy(scored, min_point0_support=2)
    no_p0 = policy[~policy["is_point0_addition"].astype(bool)].copy()
    aggressive_pool = policy[policy["allowed"].astype(bool)].copy()
    return [
        ("safe", select_unique_rows(no_p0, budget=12)),
        ("medium", select_unique_rows(no_p0, budget=36)),
        ("aggressive", select_unique_rows(aggressive_pool, budget=72)),
    ]


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor()
    sources = load_candidate_sources()
    bank = collect_row_candidates(anchor, sources)
    bank = _attach_train_support(bank, anchor)
    scored = rank_point_rows(bank)
    scored_path = safe_output_path(OUTDIR, "row_candidate_bank.csv")
    scored.to_csv(scored_path, index=False)

    generated: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for tier, selected in candidate_specs(scored):
        point_pred = build_point_prediction(anchor, selected)
        candidate = package_point_submission(anchor, point_pred)
        filename = f"submission_v370_point_pool_{tier}__v173action_v300server.csv"
        path = _write_submission(filename, candidate)
        selected_path = safe_output_path(OUTDIR, f"selected_v370_point_pool_{tier}.csv")
        selected.to_csv(selected_path, index=False)
        audit = audit_candidate(anchor, candidate, selected)
        summary = {
            "candidate": f"v370_point_pool_{tier}",
            "risk_tier": tier,
            "path": relative_path(path),
            "selected_path": relative_path(selected_path),
            **audit,
        }
        summary_rows.append(summary)
        generated.append(
            {
                "candidate": summary["candidate"],
                "path": relative_path(path),
                "risk_tier": tier,
                "point_churn_vs_v362": audit["point_churn_vs_v362"],
                "point0_additions": audit["point0_additions"],
                "score_sum": audit["score_sum"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = safe_output_path(OUTDIR, "candidate_summary.csv")
    summary.to_csv(summary_path, index=False)
    if summary.empty:
        decision = "DO_NOT_UPLOAD"
        best_candidate: dict[str, Any] = {}
    else:
        uploadable = summary[
            summary["server_preserved"].astype(bool)
            & summary["action_preserved"].astype(bool)
            & summary["point_churn_vs_v362"].between(5, 50)
        ].copy()
        if uploadable.empty:
            decision = "HOLD"
            best_candidate = summary.sort_values("score_sum", ascending=False).iloc[0].to_dict()
        else:
            decision = "HAS_UPLOAD_CANDIDATE"
            best_candidate = uploadable.sort_values(
                ["point0_additions", "risk_tier", "score_sum"],
                ascending=[True, True, False],
                kind="mergesort",
            ).iloc[0].to_dict()

    report = {
        "experiment": "v370_point_breakthrough_pool",
        "anchor_path": relative_path(ANCHOR_PATH if ANCHOR_PATH.exists() else FALLBACK_ANCHOR_PATH),
        "policy": {
            "safe": "no point0 additions, budget 12 within the 5-20 target",
            "medium": "no point0 additions, budget 36 within the 20-50 target",
            "aggressive": "point0 additions allowed only with two point0-positive families",
        },
        "source_count": len(sources),
        "bank_rows": int(len(scored)),
        "generated_submissions": generated,
        "candidate_summary": relative_path(summary_path),
        "row_candidate_bank": relative_path(scored_path),
        "best_candidate": best_candidate,
        "decision": decision,
    }
    report_path = safe_output_path(OUTDIR, "search_report.json")
    write_json(report_path, json_safe(report))
    return report


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
