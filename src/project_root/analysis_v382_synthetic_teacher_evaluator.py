from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_DIR = Path("v382_synthetic_teacher_evaluator")
V381_SYNTHETIC = Path("v381_rare_synthetic_grammar_generator") / "synthetic_rare_grammar.csv"
POINT_CANDIDATES = Path("v370_point_breakthrough_pool") / "row_candidate_bank.csv"
CONSISTENCY_EVIDENCE = Path("v371_joint_causal_consistency_lab") / "consistency_evidence.csv"
ACTION_CANDIDATES = Path("v372_action_weakness_redux") / "action_candidate_bank.csv"
ANCHOR_SUBMISSIONS = [
    Path("v362_point_hierarchical_specialists")
    / "submission_v362_depth_agree_only__v173action_v300server.csv",
    Path("v374_physical_rule_audit")
    / "submission_v374_v370_safe_keep_only__v173action_v300server.csv",
]

POINT_DEPTH = {
    0: "terminal",
    1: "short",
    2: "short",
    3: "short",
    4: "half",
    5: "half",
    6: "half",
    7: "long",
    8: "long",
    9: "long",
}

FAMILY_DEFAULT_DEPTH = {
    "attack": "long",
    "defensive": "long",
    "receive": "short",
    "control": "half",
    "serve": "short",
    "zero": "terminal",
    "unknown": "half",
}


def output_filenames() -> list[str]:
    return [
        "point_candidate_synthetic_scores.csv",
        "action_candidate_synthetic_scores.csv",
        "teacher_summary.csv",
        "search_report.json",
    ]


def fallback_teacher_rows() -> pd.DataFrame:
    rows = [
        ("fallback_attack_long", "attack", "long", "left", False, 1.15),
        ("fallback_attack_long_right", "attack", "long", "right", False, 1.15),
        ("fallback_attack_terminal", "attack", "terminal", "terminal", True, 0.95),
        ("fallback_receive_short", "receive", "short", "left", False, 0.9),
        ("fallback_receive_half", "receive", "half", "middle", False, 0.75),
        ("fallback_control_short", "control", "short", "middle", False, 0.85),
        ("fallback_control_half", "control", "half", "right", False, 0.85),
        ("fallback_defensive_long", "defensive", "long", "right", False, 0.95),
        ("fallback_zero_terminal", "zero", "terminal", "terminal", True, 1.1),
    ]
    return pd.DataFrame(
        [
            {
                "synthetic_id": f"v382_{rule_id}",
                "rule_id": rule_id,
                "provenance": "self_made_table_tennis_grammar_fallback",
                "source_type": "deterministic_fallback_teacher",
                "target_action_family": family,
                "target_point_depth": depth,
                "target_point_side": side,
                "terminal": terminal,
                "compatibility_label": True,
                "weight": weight,
            }
            for rule_id, family, depth, side, terminal, weight in rows
        ]
    )


def _norm_text(value: object, default: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().lower()
    return text if text else default


def _norm_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _point_depth(point_id: object, fallback: object = None) -> str:
    fallback_text = _norm_text(fallback, default="")
    if fallback_text:
        return fallback_text
    try:
        return POINT_DEPTH.get(int(float(point_id)), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score_against_teacher(
    action_family: str,
    point_depth: str,
    terminal: bool,
    teacher_rows: pd.DataFrame,
) -> float:
    family = _norm_text(action_family)
    depth = _norm_text(point_depth)
    is_terminal = bool(terminal)
    total_weight = 0.0
    weighted_score = 0.0

    for _, row in teacher_rows.iterrows():
        weight = float(row.get("weight", 1.0) or 1.0)
        teacher_family = _norm_text(row.get("target_action_family"))
        teacher_depth = _norm_text(row.get("target_point_depth"))
        teacher_terminal = _norm_bool(row.get("terminal"))
        compatible = _norm_bool(row.get("compatibility_label", True))

        score = 0.08
        if family == teacher_family:
            score += 0.34
        if depth == teacher_depth:
            score += 0.34
        if is_terminal == teacher_terminal:
            score += 0.18
        if is_terminal and teacher_terminal and family in {teacher_family, "zero", "attack"}:
            score += 0.08
        if not compatible:
            score -= 0.25

        weighted_score += _bounded(score) * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0
    return round(_bounded(weighted_score / total_weight), 6)


def synthetic_teacher_score(
    action_family: str,
    point_depth: str,
    terminal: bool,
    teacher_rows: pd.DataFrame | None = None,
) -> float:
    rows = fallback_teacher_rows() if teacher_rows is None else teacher_rows
    return _score_against_teacher(action_family, point_depth, terminal, rows)


def load_teacher(root: Path = Path(".")) -> tuple[pd.DataFrame, str, bool]:
    synthetic_path = root / V381_SYNTHETIC
    if synthetic_path.exists():
        rows = pd.read_csv(synthetic_path)
        source = "v381_synthetic_rare_grammar"
        missing_v381 = False
    else:
        rows = fallback_teacher_rows()
        source = "deterministic_fallback"
        missing_v381 = True

    if "weight" not in rows.columns:
        rows["weight"] = 1.0
    if "compatibility_label" not in rows.columns:
        rows["compatibility_label"] = True
    return rows, source, missing_v381


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _join_consistency(candidates: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or consistency.empty:
        return candidates.copy()
    out = candidates.copy()
    join_cols: list[str] = []
    if "row_id" in out.columns and "row_index" in consistency.columns:
        consistency = consistency.rename(columns={"row_index": "row_id"})
        join_cols = ["row_id"]
    elif "row_index" in out.columns and "row_index" in consistency.columns:
        join_cols = ["row_index"]
    elif "rally_uid" in out.columns and "rally_uid" in consistency.columns:
        join_cols = ["rally_uid"]

    if not join_cols:
        return out

    keep = join_cols + [
        col
        for col in ["proposed_family", "proposed_depth", "context_action_support", "context_point_support"]
        if col in consistency.columns
    ]
    return out.merge(consistency[keep].drop_duplicates(join_cols), on=join_cols, how="left")


def score_point_candidates(
    point_candidates: pd.DataFrame,
    consistency: pd.DataFrame,
    teacher_rows: pd.DataFrame,
    synthetic_source: str,
) -> pd.DataFrame:
    scored = _join_consistency(point_candidates, consistency)
    if scored.empty:
        return scored

    teacher_scores: list[float] = []
    inferred_families: list[str] = []
    inferred_depths: list[str] = []
    for _, row in scored.iterrows():
        depth = _point_depth(row.get("candidate_point"), row.get("candidate_depth"))
        point_id = row.get("candidate_point")
        terminal = depth == "terminal" or _norm_bool(row.get("is_point0_addition"))
        family = _norm_text(row.get("proposed_family"), default="")
        if not family:
            family = "zero" if terminal else ("attack" if depth == "long" else "control")
        teacher_scores.append(synthetic_teacher_score(family, depth, terminal, teacher_rows))
        inferred_families.append(family)
        inferred_depths.append(depth)
        _ = point_id

    out = scored.copy()
    out["synthetic_source"] = synthetic_source
    out["synthetic_action_family"] = inferred_families
    out["synthetic_point_depth"] = inferred_depths
    out["synthetic_teacher_score"] = teacher_scores
    base_score = pd.to_numeric(out.get("score", 0.0), errors="coerce").fillna(0.0)
    out["synthetic_adjusted_score"] = (base_score + out["synthetic_teacher_score"] * 10.0).round(6)
    return out.sort_values(["synthetic_adjusted_score", "synthetic_teacher_score"], ascending=False)


def score_action_candidates(
    action_candidates: pd.DataFrame,
    consistency: pd.DataFrame,
    teacher_rows: pd.DataFrame,
    synthetic_source: str,
) -> pd.DataFrame:
    scored = _join_consistency(action_candidates, consistency)
    if scored.empty:
        return scored

    teacher_scores: list[float] = []
    inferred_depths: list[str] = []
    for _, row in scored.iterrows():
        family = _norm_text(row.get("candidate_family"), default="")
        if not family:
            family = _norm_text(row.get("proposed_family"))
        depth = _norm_text(row.get("proposed_depth"), default="")
        if not depth:
            depth = FAMILY_DEFAULT_DEPTH.get(family, "half")
        terminal = depth == "terminal" or family == "zero"
        teacher_scores.append(synthetic_teacher_score(family, depth, terminal, teacher_rows))
        inferred_depths.append(depth)

    out = scored.copy()
    out["synthetic_source"] = synthetic_source
    out["synthetic_point_depth"] = inferred_depths
    out["synthetic_teacher_score"] = teacher_scores
    base_score = pd.to_numeric(out.get("score", 0.0), errors="coerce").fillna(0.0)
    out["synthetic_adjusted_score"] = (base_score + out["synthetic_teacher_score"] * 10.0).round(6)
    return out.sort_values(["synthetic_adjusted_score", "synthetic_teacher_score"], ascending=False)


def _summary_rows(
    teacher_rows: pd.DataFrame,
    synthetic_source: str,
    missing_v381: bool,
    point_scores: pd.DataFrame,
    action_scores: pd.DataFrame,
    missing_inputs: Iterable[str],
) -> pd.DataFrame:
    rows = [
        {"metric": "synthetic_source", "value": synthetic_source},
        {"metric": "missing_v381", "value": str(missing_v381)},
        {"metric": "synthetic_teacher_rows", "value": str(len(teacher_rows))},
        {"metric": "point_candidates_scored", "value": str(len(point_scores))},
        {"metric": "action_candidates_scored", "value": str(len(action_scores))},
        {"metric": "missing_inputs", "value": "|".join(missing_inputs)},
    ]
    if not point_scores.empty:
        rows.append(
            {
                "metric": "top_point_synthetic_adjusted_score",
                "value": str(point_scores.iloc[0]["synthetic_adjusted_score"]),
            }
        )
    if not action_scores.empty:
        rows.append(
            {
                "metric": "top_action_synthetic_adjusted_score",
                "value": str(action_scores.iloc[0]["synthetic_adjusted_score"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_synthetic_teacher(
    root: str | Path = Path("."),
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    root = Path(root)
    out_dir = Path(output_dir) if output_dir is not None else root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_rows, synthetic_source, missing_v381 = load_teacher(root)
    point_path = root / POINT_CANDIDATES
    consistency_path = root / CONSISTENCY_EVIDENCE
    action_path = root / ACTION_CANDIDATES

    missing_inputs = [
        str(path)
        for path in [point_path, consistency_path, action_path, *(root / p for p in ANCHOR_SUBMISSIONS)]
        if not path.exists()
    ]

    consistency = _read_optional_csv(consistency_path)
    point_candidates = _read_optional_csv(point_path)
    action_candidates = _read_optional_csv(action_path)

    point_scores = score_point_candidates(point_candidates, consistency, teacher_rows, synthetic_source)
    action_scores = score_action_candidates(action_candidates, consistency, teacher_rows, synthetic_source)
    summary = _summary_rows(
        teacher_rows,
        synthetic_source,
        missing_v381,
        point_scores,
        action_scores,
        missing_inputs,
    )

    point_scores.to_csv(out_dir / "point_candidate_synthetic_scores.csv", index=False)
    action_scores.to_csv(out_dir / "action_candidate_synthetic_scores.csv", index=False)
    summary.to_csv(out_dir / "teacher_summary.csv", index=False)

    emitted_submission_csvs = sorted(path.name for path in out_dir.glob("submission_*.csv"))
    report = {
        "version": "v382_synthetic_teacher_evaluator",
        "purpose": "Auxiliary synthetic rare grammar teacher scores for existing candidates only.",
        "synthetic_source": synthetic_source,
        "missing_v381": missing_v381,
        "synthetic_rows": int(len(teacher_rows)),
        "point_candidates_scored": int(len(point_scores)),
        "action_candidates_scored": int(len(action_scores)),
        "missing_inputs": missing_inputs,
        "outputs": output_filenames(),
        "emitted_submission_csvs": emitted_submission_csvs,
        "policy": [
            "No submission CSVs emitted by V382.",
            "No hidden test labels, TTMATCH, old-server labels, or exact-label external mapping used.",
            "Synthetic data is used only as compatibility, rare-class, and depth evidence.",
        ],
    }
    (out_dir / "search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    report = evaluate_synthetic_teacher()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
