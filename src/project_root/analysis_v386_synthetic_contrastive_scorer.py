from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTDIR = Path("v386_synthetic_contrastive_scorer")
V385_GRAMMAR = Path("v385_expanded_synthetic_grammar") / "expanded_synthetic_grammar.csv"
V382_DIR = Path("v382_synthetic_teacher_evaluator")
POINT_INPUT = V382_DIR / "point_candidate_synthetic_scores.csv"
ACTION_INPUT = V382_DIR / "action_candidate_synthetic_scores.csv"

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

ACTION_FAMILY_BY_ID = {
    1: "attack",
    2: "attack",
    3: "attack",
    4: "receive",
    5: "control",
    6: "control",
    7: "receive",
    8: "defensive",
    9: "defensive",
    10: "defensive",
    11: "control",
    12: "attack",
    13: "setup",
    14: "setup",
    15: "serve",
    16: "serve",
    17: "serve",
    18: "serve",
}

DEPTH_FAMILY_COMPATIBILITY = {
    "attack": {"long", "terminal"},
    "defensive": {"long", "terminal"},
    "control": {"short", "half"},
    "receive": {"short", "half"},
    "setup": {"short", "half"},
    "serve": {"short"},
}


def output_filenames() -> list[str]:
    return [
        "point_candidate_contrastive_scores.csv",
        "action_candidate_contrastive_scores.csv",
        "search_report.json",
    ]


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


def _norm_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_int(value: object, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _point_depth(point_id: object, fallback: object = None) -> str:
    fallback_text = _norm_text(fallback, default="")
    if fallback_text:
        return fallback_text
    return POINT_DEPTH.get(_norm_int(point_id, default=-1), "unknown")


def _action_family(action_id: object, fallback: object = None) -> str:
    fallback_text = _norm_text(fallback, default="")
    if fallback_text:
        return fallback_text
    return ACTION_FAMILY_BY_ID.get(_norm_int(action_id, default=-1), "unknown")


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _terminal_support(row: pd.Series) -> bool:
    phase = _norm_text(row.get("phase"), default="")
    if phase in {"terminal", "late_attack", "kill", "finish"}:
        return True
    if _norm_bool(row.get("terminal")) or _norm_bool(row.get("terminal_support")):
        return True
    return _norm_float(row.get("point0_support_count")) > 0


def _serve_context(row: pd.Series) -> bool:
    phase = _norm_text(row.get("phase"), default="")
    return phase in {"serve", "service"}


def score_action_point_compatibility(action_id: int, point_id: int, phase: str) -> float:
    family = _action_family(action_id)
    depth = _point_depth(point_id)
    score = 0.5

    if depth in DEPTH_FAMILY_COMPATIBILITY.get(family, set()):
        score += 0.28
    else:
        score -= 0.28

    if point_id == 0:
        score += 0.08 if _norm_text(phase) in {"terminal", "late_attack", "kill", "finish"} else -0.24
    if family == "serve" and _norm_text(phase) not in {"serve", "service"}:
        score -= 0.35
    if family == "attack" and depth == "long":
        score += 0.08
    if family == "control" and depth == "long":
        score -= 0.12

    return round(_bounded(score), 6)


def _grammar_score(
    grammar_rows: pd.DataFrame,
    family: str,
    depth: str,
    phase: str,
) -> float:
    if grammar_rows.empty:
        return 0.0

    rows = grammar_rows.copy()
    if "target_action_family" in rows.columns:
        rows = rows[rows["target_action_family"].map(_norm_text) == family]
    if "target_point_depth" in rows.columns:
        rows = rows[rows["target_point_depth"].map(_norm_text) == depth]
    if "phase" in rows.columns:
        exact_phase = rows[rows["phase"].map(_norm_text) == phase]
        if not exact_phase.empty:
            rows = exact_phase

    if rows.empty:
        return 0.0

    total_weight = 0.0
    total = 0.0
    for _, row in rows.iterrows():
        weight = _norm_float(row.get("weight"), default=1.0)
        label = _norm_text(row.get("compatibility_label"), default="compatible")
        compatible = label in {"compatible", "true", "1", "yes"}
        total += (0.22 if compatible else -0.28) * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return _bounded(total / total_weight, -0.3, 0.25)


def _score_row(row: pd.Series, grammar_rows: pd.DataFrame | None = None) -> tuple[float, bool]:
    candidate_action = row.get("candidate_action", row.get("base_action"))
    candidate_point = row.get("candidate_point", row.get("base_point"))
    phase = _norm_text(row.get("phase"), default="rally")
    family = _action_family(
        candidate_action,
        row.get("candidate_family", row.get("proposed_family", row.get("synthetic_action_family"))),
    )
    depth = _point_depth(candidate_point, row.get("candidate_depth", row.get("proposed_depth")))

    score = score_action_point_compatibility(
        action_id=_norm_int(candidate_action, default=-1),
        point_id=_norm_int(candidate_point, default=-1),
        phase=phase,
    )
    if grammar_rows is not None:
        score += _grammar_score(grammar_rows, family, depth, phase)

    base_depth = _point_depth(row.get("base_point"), row.get("base_depth"))
    base_family = _action_family(row.get("base_action"), row.get("base_family"))
    if base_depth == depth and depth != "unknown":
        score += 0.08
    if base_family == family and family != "unknown":
        score += 0.06

    score += min(_norm_float(row.get("support_count")) / 100.0, 0.12)
    score += min(_norm_float(row.get("source_family_count")) / 20.0, 0.10)
    score += min(_norm_float(row.get("context_action_support")) / 5000.0, 0.05)
    score += min(_norm_float(row.get("context_point_support")) / 5000.0, 0.05)

    is_point0_addition = _norm_int(row.get("candidate_point"), default=-1) == 0 and _norm_int(
        row.get("base_point"), default=-1
    ) != 0
    if is_point0_addition and not _terminal_support(row):
        score -= 0.65

    candidate_action_id = _norm_int(candidate_action, default=-1)
    serve_blocked = candidate_action_id in {15, 16, 17, 18} and not _serve_context(row)
    if serve_blocked:
        score -= 0.7

    allowed = score > 0.45 and not (is_point0_addition and not _terminal_support(row)) and not serve_blocked
    return round(_bounded(score), 6), bool(allowed)


def score_candidate_frame(
    frame: pd.DataFrame,
    grammar_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["synthetic_compatibility_score"] = []
        out["contrastive_score"] = []
        out["synthetic_allowed"] = []
        return out

    scores: list[float] = []
    allowed: list[bool] = []
    for _, row in out.iterrows():
        score, is_allowed = _score_row(row, grammar_rows)
        scores.append(score)
        allowed.append(is_allowed)

    if "score" in out.columns:
        base_score = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    else:
        base_score = pd.Series(0.0, index=out.index)
    out["synthetic_compatibility_score"] = scores
    out["contrastive_score"] = (base_score + out["synthetic_compatibility_score"] * 10.0).round(6)
    out["synthetic_allowed"] = pd.Series(allowed, dtype=object)
    if "is_point0_addition" not in out.columns:
        out["is_point0_addition"] = [
            _norm_int(row.get("candidate_point"), default=-1) == 0
            and _norm_int(row.get("base_point"), default=-1) != 0
            for _, row in out.iterrows()
        ]
    return out.sort_values(["synthetic_allowed", "contrastive_score"], ascending=[False, False])


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_v385(root: Path) -> tuple[pd.DataFrame, bool]:
    grammar_path = root / V385_GRAMMAR
    if not grammar_path.exists():
        return pd.DataFrame(), True
    grammar = pd.read_csv(grammar_path)
    return grammar, False


def _missing_inputs(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def run_pipeline(
    root: str | Path = Path("."),
    outdir: str | Path = OUTDIR,
) -> dict[str, object]:
    root = Path(root)
    output_dir = Path(outdir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grammar_rows, missing_v385 = _load_v385(root)
    point_path = root / POINT_INPUT
    action_path = root / ACTION_INPUT
    point_candidates = _read_optional_csv(point_path)
    action_candidates = _read_optional_csv(action_path)

    grammar_for_scoring = None if missing_v385 else grammar_rows
    point_scores = score_candidate_frame(point_candidates, grammar_for_scoring)
    action_scores = score_candidate_frame(action_candidates, grammar_for_scoring)

    point_scores.to_csv(output_dir / "point_candidate_contrastive_scores.csv", index=False)
    action_scores.to_csv(output_dir / "action_candidate_contrastive_scores.csv", index=False)

    emitted_submission_csvs = sorted(path.name for path in output_dir.glob("submission_*.csv"))
    report = {
        "version": "v386_synthetic_contrastive_scorer",
        "purpose": "Contrastive synthetic compatibility scores for existing point/action candidates only.",
        "missing_v385": missing_v385,
        "synthetic_source": "deterministic_fallback" if missing_v385 else "v385_expanded_synthetic_grammar",
        "v385_rows": int(len(grammar_rows)),
        "point_candidates_scored": int(len(point_scores)),
        "action_candidates_scored": int(len(action_scores)),
        "point_allowed_count": int(point_scores.get("synthetic_allowed", pd.Series(dtype=bool)).sum()),
        "action_allowed_count": int(action_scores.get("synthetic_allowed", pd.Series(dtype=bool)).sum()),
        "missing_inputs": _missing_inputs([root / V385_GRAMMAR, point_path, action_path]),
        "outputs": output_filenames(),
        "emitted_submission_csvs": emitted_submission_csvs,
        "policy": [
            "No submission CSVs emitted by V386.",
            "No hidden test labels, TTMATCH, old-server labels, or manual row edits used.",
            "Synthetic data is used only as candidate compatibility evidence.",
        ],
    }
    (output_dir / "search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
