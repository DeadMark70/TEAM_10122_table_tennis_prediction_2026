"""V371 joint causal consistency lab.

This experiment scores row-level action/point consistency under simple
table-tennis causal structure, then exports low-churn clean submissions.  It
uses source submissions as evidence only and preserves the V300 server column.
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
    action_distribution_report,
    point_distribution_report,
    safe_output_path as contract_safe_output_path,
    validate_submission_schema,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v371_joint_causal_consistency_lab"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
FALLBACK_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
V370_BANK_PATH = ROOT / "v370_point_breakthrough_pool" / "row_candidate_bank.csv"
V361_EVIDENCE_PATH = ROOT / "v361_action_hierarchical_specialists" / "row_evidence.csv"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"


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
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def safe_output_path(filename: str) -> Path:
    path = contract_safe_output_path(OUTDIR, filename)
    blocked = ("ttmatch", "oldserver", "old_server", "upload_candidates_20260519")
    lowered = path.as_posix().lower()
    if any(token in lowered for token in blocked):
        raise ValueError(f"refusing blocked V371 output path: {path}")
    if path.parent.resolve() != OUTDIR.resolve():
        raise ValueError(f"V371 outputs must stay directly under {OUTDIR}: {path}")
    return path


def action_to_family(action: int) -> str:
    action_id = int(action)
    if action_id == 0:
        return "zero"
    if 1 <= action_id <= 7:
        return "attack"
    if 8 <= action_id <= 11:
        return "control"
    if 12 <= action_id <= 14:
        return "defensive"
    if 15 <= action_id <= 18:
        return "serve"
    return "unknown"


def point_to_depth(point: int) -> str:
    point_id = int(point)
    if point_id == 0:
        return "terminal"
    if 1 <= point_id <= 3:
        return "short"
    if 4 <= point_id <= 6:
        return "half"
    if 7 <= point_id <= 9:
        return "long"
    return "unknown"


def phase_from_prefix(prefix_len: int) -> str:
    value = int(prefix_len)
    if value <= 1:
        return "receive"
    if value == 2:
        return "third_ball"
    if value == 3:
        return "fourth_ball"
    return "rally"


COMPATIBILITY = {
    "zero": {"terminal": 1.00, "short": 0.02, "half": 0.02, "long": 0.02, "unknown": 0.10},
    "serve": {"terminal": 0.08, "short": 0.78, "half": 0.48, "long": 0.28, "unknown": 0.30},
    "control": {"terminal": 0.18, "short": 0.86, "half": 0.66, "long": 0.30, "unknown": 0.40},
    "attack": {"terminal": 0.20, "short": 0.36, "half": 0.66, "long": 0.88, "unknown": 0.45},
    "defensive": {"terminal": 0.20, "short": 0.50, "half": 0.76, "long": 0.66, "unknown": 0.45},
    "unknown": {"terminal": 0.20, "short": 0.40, "half": 0.45, "long": 0.45, "unknown": 0.35},
}


def compatibility_score(action_family: str, point_depth: str) -> float:
    """Static action-family to point-depth plausibility score."""
    family = str(action_family)
    depth = str(point_depth)
    return float(COMPATIBILITY.get(family, COMPATIBILITY["unknown"]).get(depth, 0.25))


def terminal_inconsistency_flags(rows: pd.DataFrame) -> pd.Series:
    action_zero = pd.to_numeric(rows["actionId"], errors="coerce").fillna(-1).astype(int).eq(0)
    point_zero = pd.to_numeric(rows["pointId"], errors="coerce").fillna(-1).astype(int).eq(0)
    return action_zero.ne(point_zero)


def package_joint_candidate(
    anchor: pd.DataFrame,
    action_pred: pd.Series | np.ndarray | list[int],
    point_pred: pd.Series | np.ndarray | list[int],
) -> pd.DataFrame:
    if len(anchor) != len(action_pred) or len(anchor) != len(point_pred):
        raise ValueError("anchor, action predictions, and point predictions must have matching lengths")
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["actionId"] = pd.Series(action_pred).to_numpy(dtype=int)
    out["pointId"] = pd.Series(point_pred).to_numpy(dtype=int)
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("serverGetPoint changed while packaging V371 candidate")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def load_anchor(expected_rows: int | None = 1845) -> pd.DataFrame:
    if ANCHOR_PATH.exists():
        return load_submission(ANCHOR_PATH, expected_rows=expected_rows)
    if FALLBACK_PATH.exists():
        return load_submission(FALLBACK_PATH, expected_rows=expected_rows)
    raise FileNotFoundError(f"missing V371 anchor and fallback: {ANCHOR_PATH}, {FALLBACK_PATH}")


def load_fallback(anchor: pd.DataFrame) -> pd.DataFrame:
    if not FALLBACK_PATH.exists():
        return anchor.copy()
    return load_submission(FALLBACK_PATH, expected_rows=len(anchor))


def _source_name(path: Path) -> str:
    return path.name.replace(".csv", "")


def load_v370_point_candidates(anchor: pd.DataFrame) -> pd.DataFrame:
    if not V370_BANK_PATH.exists():
        return pd.DataFrame(columns=["row_index", "candidate_point", "v370_score", "v370_support", "v370_sources"])
    bank = pd.read_csv(V370_BANK_PATH)
    if bank.empty:
        return pd.DataFrame(columns=["row_index", "candidate_point", "v370_score", "v370_support", "v370_sources"])

    candidate_col = next((c for c in ["candidate_point", "proposed_point", "point_pred", "new_point"] if c in bank.columns), None)
    if candidate_col is None:
        return pd.DataFrame(columns=["row_index", "candidate_point", "v370_score", "v370_support", "v370_sources"])

    work = bank.copy()
    if "row_index" not in work.columns:
        if "rally_uid" not in work.columns:
            return pd.DataFrame(columns=["row_index", "candidate_point", "v370_score", "v370_support", "v370_sources"])
        index_map = anchor.reset_index().loc[:, ["index", "rally_uid"]].rename(columns={"index": "row_index"})
        work = work.merge(index_map, on="rally_uid", how="inner")

    score_col = next((c for c in ["score", "candidate_score", "rank_score"] if c in work.columns), None)
    support_col = next((c for c in ["support_count", "source_count", "v370_support"] if c in work.columns), None)
    source_col = next((c for c in ["sources", "source_names"] if c in work.columns), None)
    work["candidate_point"] = pd.to_numeric(work[candidate_col], errors="coerce").fillna(-1).astype(int)
    work["v370_score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0) if score_col else 0.0
    work["v370_support"] = pd.to_numeric(work[support_col], errors="coerce").fillna(1).astype(int) if support_col else 1
    work["v370_sources"] = work[source_col].astype(str) if source_col else _source_name(V370_BANK_PATH)
    work = work[work["candidate_point"].between(0, 9)].copy()
    work = work.sort_values(["row_index", "v370_score", "v370_support"], ascending=[True, False, False])
    return work.drop_duplicates("row_index").loc[:, ["row_index", "candidate_point", "v370_score", "v370_support", "v370_sources"]]


def load_v361_action_candidates(anchor: pd.DataFrame) -> pd.DataFrame:
    if not V361_EVIDENCE_PATH.exists():
        return pd.DataFrame(columns=["row_index", "candidate_action", "v361_score", "v361_support", "v361_sources"])
    evidence = pd.read_csv(V361_EVIDENCE_PATH)
    if evidence.empty or "proposed_action" not in evidence.columns:
        return pd.DataFrame(columns=["row_index", "candidate_action", "v361_score", "v361_support", "v361_sources"])
    work = evidence.copy()
    if "row_index" not in work.columns:
        if "rally_uid" not in work.columns:
            return pd.DataFrame(columns=["row_index", "candidate_action", "v361_score", "v361_support", "v361_sources"])
        index_map = anchor.reset_index().loc[:, ["index", "rally_uid"]].rename(columns={"index": "row_index"})
        work = work.merge(index_map, on="rally_uid", how="inner")
    work["candidate_action"] = pd.to_numeric(work["proposed_action"], errors="coerce").fillna(-1).astype(int)
    work["v361_score"] = pd.to_numeric(work.get("score", 0.0), errors="coerce").fillna(0.0)
    work["v361_support"] = pd.to_numeric(work.get("source_count", 1), errors="coerce").fillna(1).astype(int)
    work["v361_sources"] = work.get("sources", _source_name(V361_EVIDENCE_PATH)).astype(str)
    work = work[work["candidate_action"].between(0, 18)].copy()
    work = work.sort_values(["row_index", "v361_score", "v361_support"], ascending=[True, False, False])
    return work.drop_duplicates("row_index").loc[:, ["row_index", "candidate_action", "v361_score", "v361_support", "v361_sources"]]


def load_test_context(anchor: pd.DataFrame) -> pd.DataFrame:
    context = anchor[["rally_uid"]].copy()
    context["prefix_len"] = 0
    context["phase"] = "unknown"
    context["lag_action_family"] = "unknown"
    context["lag_point_depth"] = "unknown"
    context["spin_key"] = -1
    context["strength_key"] = -1
    if not TEST_PATH.exists():
        return context

    test = pd.read_csv(TEST_PATH)
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    if not required.issubset(test.columns):
        return context
    last = test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", as_index=False).tail(1)
    merged = context[["rally_uid"]].merge(last, on="rally_uid", how="left", validate="one_to_one")
    context["prefix_len"] = pd.to_numeric(merged.get("strikeNumber"), errors="coerce").fillna(0).astype(int)
    context["phase"] = context["prefix_len"].map(phase_from_prefix)
    context["lag_action_family"] = pd.to_numeric(merged.get("actionId"), errors="coerce").fillna(-1).astype(int).map(action_to_family)
    context["lag_point_depth"] = pd.to_numeric(merged.get("pointId"), errors="coerce").fillna(-1).astype(int).map(point_to_depth)
    if "spinId" in merged.columns:
        context["spin_key"] = pd.to_numeric(merged["spinId"], errors="coerce").fillna(-1).astype(int)
    if "strengthId" in merged.columns:
        context["strength_key"] = pd.to_numeric(merged["strengthId"], errors="coerce").fillna(-1).astype(int)
    return context


def _mode_and_support(group: pd.DataFrame, column: str) -> tuple[int, int]:
    counts = group[column].astype(int).value_counts()
    if counts.empty:
        return 0, 0
    return int(counts.index[0]), int(counts.iloc[0])


def build_support_model() -> dict[str, Any]:
    model: dict[str, Any] = {
        "available": False,
        "family_depth": {},
        "depth_family": {},
        "context_tables": [],
        "global_action": 10,
        "global_point": 8,
    }
    if not TRAIN_PATH.exists():
        return model
    train = pd.read_csv(TRAIN_PATH)
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    if not required.issubset(train.columns):
        return model

    train = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    train["action_family"] = train["actionId"].astype(int).map(action_to_family)
    train["point_depth"] = train["pointId"].astype(int).map(point_to_depth)
    model["global_action"] = int(train["actionId"].mode().iloc[0])
    model["global_point"] = int(train["pointId"].mode().iloc[0])

    for family, group in train.groupby("action_family", sort=False):
        total = max(len(group), 1)
        for depth, count in group["point_depth"].value_counts().items():
            model["family_depth"][(str(family), str(depth))] = float(count / total)
    for depth, group in train.groupby("point_depth", sort=False):
        total = max(len(group), 1)
        for family, count in group["action_family"].value_counts().items():
            model["depth_family"][(str(depth), str(family))] = float(count / total)

    prev = train.groupby("rally_uid")[["actionId", "pointId"]].shift(1)
    train["prev_action"] = prev["actionId"]
    train["prev_point"] = prev["pointId"]
    if "spinId" in train.columns:
        train["prev_spin"] = train.groupby("rally_uid")["spinId"].shift(1)
    else:
        train["prev_spin"] = -1
    if "strengthId" in train.columns:
        train["prev_strength"] = train.groupby("rally_uid")["strengthId"].shift(1)
    else:
        train["prev_strength"] = -1
    rows = train[train["prev_action"].notna() & train["prev_point"].notna()].copy()
    if rows.empty:
        model["available"] = True
        return model
    rows["phase"] = (rows["strikeNumber"].astype(int) - 1).map(phase_from_prefix)
    rows["lag_action_family"] = rows["prev_action"].astype(int).map(action_to_family)
    rows["lag_point_depth"] = rows["prev_point"].astype(int).map(point_to_depth)
    rows["spin_key"] = pd.to_numeric(rows["prev_spin"], errors="coerce").fillna(-1).astype(int)
    rows["strength_key"] = pd.to_numeric(rows["prev_strength"], errors="coerce").fillna(-1).astype(int)

    key_sets = [
        ["phase", "lag_action_family", "lag_point_depth", "spin_key", "strength_key"],
        ["phase", "lag_action_family", "lag_point_depth"],
        ["lag_action_family", "lag_point_depth"],
        ["phase"],
    ]
    tables: list[dict[tuple[Any, ...], dict[str, int]]] = []
    for keys in key_sets:
        table: dict[tuple[Any, ...], dict[str, int]] = {}
        for key, group in rows.groupby(keys, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            action_mode, action_support = _mode_and_support(group, "actionId")
            point_mode, point_support = _mode_and_support(group, "pointId")
            table[key_tuple] = {
                "action_pred": action_mode,
                "action_support": action_support,
                "point_pred": point_mode,
                "point_support": point_support,
                "rows": int(len(group)),
            }
        tables.append({"keys": keys, "table": table})
    model["context_tables"] = tables
    model["available"] = True
    return model


def context_lookup(model: dict[str, Any], context_row: pd.Series) -> dict[str, Any]:
    for spec in model.get("context_tables", []):
        keys = spec["keys"]
        key = tuple(context_row.get(k) for k in keys)
        hit = spec["table"].get(key)
        if hit is not None:
            out = dict(hit)
            out["context_level"] = ",".join(keys)
            return out
    return {
        "action_pred": int(model.get("global_action", 10)),
        "action_support": 0,
        "point_pred": int(model.get("global_point", 8)),
        "point_support": 0,
        "rows": 0,
        "context_level": "global",
    }


def pair_consistency_score(
    action_id: int,
    point_id: int,
    context_pred: dict[str, Any],
    model: dict[str, Any],
) -> float:
    family = action_to_family(action_id)
    depth = point_to_depth(point_id)
    score = compatibility_score(family, depth)
    score += 0.35 * float(model.get("family_depth", {}).get((family, depth), 0.0))
    score += 0.25 * float(model.get("depth_family", {}).get((depth, family), 0.0))
    if int(context_pred.get("action_pred", -1)) == int(action_id):
        score += min(float(context_pred.get("action_support", 0)) / 80.0, 0.25)
    if int(context_pred.get("point_pred", -1)) == int(point_id):
        score += min(float(context_pred.get("point_support", 0)) / 80.0, 0.25)
    if (int(action_id) == 0) != (int(point_id) == 0):
        score -= 0.85
    if int(point_id) == 0 and int(action_id) != 0:
        score -= 0.15
    return float(score)


def _append_candidate(
    candidates: dict[int, dict[int, set[str]]],
    row_index: int,
    value: int,
    source: str,
) -> None:
    candidates.setdefault(int(row_index), {}).setdefault(int(value), set()).add(source)


def build_consistency_evidence(
    anchor: pd.DataFrame,
    fallback: pd.DataFrame,
    v370_points: pd.DataFrame,
    v361_actions: pd.DataFrame,
    context: pd.DataFrame,
    model: dict[str, Any],
) -> pd.DataFrame:
    point_candidates: dict[int, dict[int, set[str]]] = {}
    action_candidates: dict[int, dict[int, set[str]]] = {}
    for i, row in anchor.reset_index(drop=True).iterrows():
        _append_candidate(point_candidates, i, int(row["pointId"]), "anchor")
        _append_candidate(action_candidates, i, int(row["actionId"]), "anchor")
    for i, point in enumerate(fallback["pointId"].astype(int).to_numpy()):
        if point != int(anchor.iloc[i]["pointId"]):
            _append_candidate(point_candidates, i, point, "v338_fallback")
    for row in v370_points.itertuples(index=False):
        _append_candidate(point_candidates, int(row.row_index), int(row.candidate_point), "v370_bank")
    for row in v361_actions.itertuples(index=False):
        _append_candidate(action_candidates, int(row.row_index), int(row.candidate_action), "v361_evidence")

    v361_score = dict(zip(v361_actions["row_index"].astype(int), v361_actions["v361_score"].astype(float))) if not v361_actions.empty else {}
    v361_support = dict(zip(v361_actions["row_index"].astype(int), v361_actions["v361_support"].astype(int))) if not v361_actions.empty else {}
    v370_score = dict(zip(v370_points["row_index"].astype(int), v370_points["v370_score"].astype(float))) if not v370_points.empty else {}
    v370_support = dict(zip(v370_points["row_index"].astype(int), v370_points["v370_support"].astype(int))) if not v370_points.empty else {}

    rows: list[dict[str, Any]] = []
    for i, base in anchor.reset_index(drop=True).iterrows():
        context_pred = context_lookup(model, context.iloc[i])
        base_action = int(base["actionId"])
        base_point = int(base["pointId"])
        base_score = pair_consistency_score(base_action, base_point, context_pred, model)
        terminal_flag = bool((base_action == 0) != (base_point == 0))
        for cand_action, action_sources in action_candidates.get(i, {base_action: {"anchor"}}).items():
            for cand_point, point_sources in point_candidates.get(i, {base_point: {"anchor"}}).items():
                if cand_action == base_action and cand_point == base_point:
                    continue
                change_type = (
                    "joint"
                    if cand_action != base_action and cand_point != base_point
                    else "action_only"
                    if cand_action != base_action
                    else "point_only"
                )
                cand_score = pair_consistency_score(cand_action, cand_point, context_pred, model)
                source_bonus = 0.0
                if "v361_evidence" in action_sources:
                    source_bonus += min(v361_score.get(i, 0.0) / 100.0, 0.25)
                if "v370_bank" in point_sources:
                    source_bonus += min(v370_score.get(i, 0.0) / 100.0, 0.20)
                if "v338_fallback" in point_sources:
                    source_bonus += 0.06
                candidate_score = cand_score + source_bonus
                improvement = candidate_score - base_score
                rows.append(
                    {
                        "row_index": int(i),
                        "rally_uid": int(base["rally_uid"]),
                        "base_action": base_action,
                        "base_point": base_point,
                        "proposed_action": int(cand_action),
                        "proposed_point": int(cand_point),
                        "change_type": change_type,
                        "base_family": action_to_family(base_action),
                        "base_depth": point_to_depth(base_point),
                        "proposed_family": action_to_family(cand_action),
                        "proposed_depth": point_to_depth(cand_point),
                        "base_score": base_score,
                        "candidate_score": candidate_score,
                        "improvement": improvement,
                        "terminal_inconsistent_base": terminal_flag,
                        "point0_addition": bool(base_point != 0 and cand_point == 0),
                        "point0_removal": bool(base_point == 0 and cand_point != 0),
                        "context_action_pred": int(context_pred.get("action_pred", -1)),
                        "context_point_pred": int(context_pred.get("point_pred", -1)),
                        "context_action_support": int(context_pred.get("action_support", 0)),
                        "context_point_support": int(context_pred.get("point_support", 0)),
                        "context_level": str(context_pred.get("context_level", "")),
                        "source_count": int(len(action_sources | point_sources))
                        + int(v361_support.get(i, 0))
                        + int(v370_support.get(i, 0)),
                        "sources": ";".join(sorted(action_sources | point_sources)),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "row_index",
                "rally_uid",
                "base_action",
                "base_point",
                "proposed_action",
                "proposed_point",
                "change_type",
                "base_score",
                "candidate_score",
                "improvement",
                "sources",
            ]
        )
    evidence = pd.DataFrame(rows)
    evidence = evidence.sort_values(
        ["improvement", "candidate_score", "source_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return evidence


def select_best_per_row(evidence: pd.DataFrame, allowed_types: set[str], max_rows: int, min_improvement: float) -> pd.DataFrame:
    if evidence.empty:
        return evidence.copy()
    work = evidence[evidence["change_type"].isin(allowed_types) & evidence["improvement"].ge(min_improvement)].copy()
    if work.empty:
        return work
    work = work.sort_values(["row_index", "improvement", "candidate_score"], ascending=[True, False, False])
    work = work.drop_duplicates("row_index")
    work = work.sort_values(["improvement", "candidate_score", "source_count"], ascending=[False, False, False])
    return work.head(max_rows).copy()


def apply_rows(anchor: pd.DataFrame, selected: pd.DataFrame, allow_action: bool) -> pd.DataFrame:
    action_pred = anchor["actionId"].astype(int).copy()
    point_pred = anchor["pointId"].astype(int).copy()
    for row in selected.itertuples(index=False):
        idx = int(row.row_index)
        if allow_action:
            action_pred.iloc[idx] = int(row.proposed_action)
        point_pred.iloc[idx] = int(row.proposed_point)
    return package_joint_candidate(anchor, action_pred=action_pred, point_pred=point_pred)


def transition_counts(base: pd.Series, candidate: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for left, right in zip(base.astype(int), candidate.astype(int)):
        if int(left) != int(right):
            counts[f"{int(left)}->{int(right)}"] += 1
    return dict(sorted(counts.items()))


def summarize_candidate(
    name: str,
    path: Path,
    anchor: pd.DataFrame,
    candidate: pd.DataFrame,
    selected: pd.DataFrame,
    risk: str,
) -> dict[str, Any]:
    point_report = point_distribution_report(anchor["pointId"], candidate["pointId"])
    action_report = action_distribution_report(anchor["actionId"], candidate["actionId"])
    return {
        "candidate": name,
        "risk": risk,
        "path": relative_path(path),
        "selected_rows": int(len(selected)),
        "changed_action_rows": int(action_report["changed_rows"]),
        "changed_point_rows": int(point_report["changed_rows"]),
        "point0_additions": int(point_report["point0_additions"]),
        "point0_removals": int(point_report["point0_removals"]),
        "server_preserved": bool(candidate["serverGetPoint"].equals(anchor["serverGetPoint"])),
        "action_transitions": json.dumps(transition_counts(anchor["actionId"], candidate["actionId"]), sort_keys=True),
        "point_transitions": json.dumps(transition_counts(anchor["pointId"], candidate["pointId"]), sort_keys=True),
        "mean_improvement": float(selected["improvement"].mean()) if len(selected) else 0.0,
        "score_sum": float(selected["candidate_score"].sum()) if len(selected) else 0.0,
        "decision": "RESEARCH_ONLY" if risk == "research" else ("HAS_UPLOAD_CANDIDATE" if len(selected) else "DO_NOT_UPLOAD"),
    }


def run() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor()
    fallback = load_fallback(anchor)
    v370_points = load_v370_point_candidates(anchor)
    v361_actions = load_v361_action_candidates(anchor)
    context = load_test_context(anchor)
    model = build_support_model()
    evidence = build_consistency_evidence(anchor, fallback, v370_points, v361_actions, context, model)

    safe_selected = select_best_per_row(evidence, {"point_only"}, max_rows=12, min_improvement=0.08)
    safe_selected = safe_selected[~safe_selected["point0_addition"]].copy() if not safe_selected.empty else safe_selected
    joint_low_selected = select_best_per_row(evidence, {"joint", "action_only", "point_only"}, max_rows=12, min_improvement=0.12)
    research_selected = select_best_per_row(evidence, {"joint", "action_only", "point_only"}, max_rows=48, min_improvement=0.03)

    safe_candidate = apply_rows(anchor, safe_selected, allow_action=False)
    joint_low_candidate = apply_rows(anchor, joint_low_selected, allow_action=True)
    research_candidate = apply_rows(anchor, research_selected, allow_action=True)

    evidence_path = safe_output_path("consistency_evidence.csv")
    safe_path = safe_output_path("submission_v371_point_consistency_safe__v173action_v300server.csv")
    joint_low_path = safe_output_path("submission_v371_joint_consistency_low__v300server.csv")
    research_path = safe_output_path("submission_v371_joint_consistency_research__v300server.csv")
    summary_path = safe_output_path("candidate_summary.csv")
    report_path = safe_output_path("search_report.json")

    evidence.to_csv(evidence_path, index=False)
    safe_candidate.to_csv(safe_path, index=False)
    joint_low_candidate.to_csv(joint_low_path, index=False)
    research_candidate.to_csv(research_path, index=False)

    summaries = [
        summarize_candidate("v371_point_consistency_safe", safe_path, anchor, safe_candidate, safe_selected, "safe"),
        summarize_candidate("v371_joint_consistency_low", joint_low_path, anchor, joint_low_candidate, joint_low_selected, "low"),
        summarize_candidate("v371_joint_consistency_research", research_path, anchor, research_candidate, research_selected, "research"),
    ]
    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_path, index=False)

    uploadable = summary[summary["decision"].eq("HAS_UPLOAD_CANDIDATE")].copy()
    if uploadable.empty:
        top_candidate = None
        recommendation = "HOLD"
    else:
        uploadable = uploadable.sort_values(
            ["risk", "score_sum", "changed_point_rows", "changed_action_rows"],
            ascending=[True, False, True, True],
        )
        top_candidate = uploadable.iloc[0].to_dict()
        recommendation = "HAS_UPLOAD_CANDIDATE"

    report = {
        "experiment": "v371_joint_causal_consistency_lab",
        "anchor": relative_path(ANCHOR_PATH if ANCHOR_PATH.exists() else FALLBACK_PATH),
        "fallback": relative_path(FALLBACK_PATH),
        "inputs": {
            "v370_bank_available": bool(V370_BANK_PATH.exists()),
            "v370_rows": int(len(v370_points)),
            "v361_evidence_available": bool(V361_EVIDENCE_PATH.exists()),
            "v361_rows": int(len(v361_actions)),
            "train_support_available": bool(model.get("available", False)),
        },
        "evidence_rows": int(len(evidence)),
        "terminal_inconsistent_anchor_rows": int(terminal_inconsistency_flags(anchor).sum()),
        "candidate_summary": summaries,
        "top_candidate": top_candidate,
        "recommendation": recommendation,
        "notes": [
            "Safe candidate changes pointId only and blocks point0 additions.",
            "Joint candidates preserve serverGetPoint from the clean V300 server column.",
            "No TTMATCH, old-server labels, or manual row edits are used.",
        ],
    }
    write_json(report_path, report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
