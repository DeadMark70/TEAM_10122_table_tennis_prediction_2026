"""V385 expanded synthetic grammar generator.

Builds a deterministic positive/negative table-tennis grammar corpus for
auxiliary teacher/scorer use. Rows are self-made synthetic evidence only and do
not reference real rally identifiers.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


OUTDIR = Path("v385_expanded_synthetic_grammar")
PROVENANCE = "self_made_expanded_table_tennis_grammar"

PHASES = ("receive", "third_ball", "rally", "late_attack")
PREFIX_LEN_BINS = ("short_prefix", "mid_prefix", "long_prefix")
LAST_ACTION_FAMILIES = ("attack", "control", "defensive", "setup")
SPINS = ("topspin", "underspin", "sidespin", "flat")
STRENGTHS = ("soft", "medium", "strong")

SHORT_POINTS = {1, 2, 3}
HALF_POINTS = {4, 5, 6}
LONG_POINTS = {7, 8, 9}
TERMINAL_POINTS = {0}

ACTION_FAMILY_BY_ID = {
    3: "attack",
    4: "setup",
    5: "attack",
    7: "setup",
    8: "control",
    9: "control",
    11: "control",
    12: "defensive",
    14: "defensive",
}

ACTION_IDS_BY_FAMILY = {
    "attack": (3, 5),
    "control": (8, 9, 11),
    "defensive": (12, 14),
    "setup": (4, 7),
}

POINT_SIDE_BY_ID = {
    0: "terminal",
    1: "left",
    2: "middle",
    3: "right",
    4: "left",
    5: "middle",
    6: "right",
    7: "left",
    8: "middle",
    9: "right",
}

FIELDNAMES = [
    "synthetic_id",
    "rally_uid",
    "rule_id",
    "provenance",
    "source_type",
    "phase",
    "prefix_len_bin",
    "last_action_family",
    "last_spin",
    "last_strength",
    "terminal_context",
    "target_action_family",
    "target_action_id_optional",
    "target_point_depth",
    "target_point_side",
    "target_point_id_optional",
    "compatibility_label",
    "weight",
]


def point_depth(point_id: int) -> str:
    if point_id in SHORT_POINTS:
        return "short"
    if point_id in HALF_POINTS:
        return "half"
    if point_id in LONG_POINTS:
        return "long"
    if point_id in TERMINAL_POINTS:
        return "terminal"
    return "unknown"


def compatible_action_point(action_id: int, point_id: int) -> bool:
    """Return whether an action/point pair is physically plausible."""
    if point_id == 0:
        return action_id in {3, 12, 14}
    if action_id in {8, 9, 11}:
        return point_id in SHORT_POINTS | HALF_POINTS
    if action_id in {4, 7}:
        return point_id in SHORT_POINTS | HALF_POINTS | LONG_POINTS
    if action_id in {3, 5, 12, 14}:
        return point_id in HALF_POINTS | LONG_POINTS | TERMINAL_POINTS
    return False


def _terminal_context(phase: str, prefix_len_bin: str, last_action_family: str, strength: str) -> bool:
    return (
        phase == "late_attack"
        or prefix_len_bin == "long_prefix"
        or (last_action_family in {"attack", "defensive"} and strength == "strong")
    )


def _positive_target(
    phase: str,
    last_action_family: str,
    spin: str,
    strength: str,
    rng: random.Random,
) -> tuple[str, int, int, str]:
    if last_action_family == "control":
        family = "control" if phase != "third_ball" else "setup"
        point_pool = (1, 2, 3, 4, 5, 6)
        rule_id = "positive_control_short_half"
    elif last_action_family == "setup" or phase == "receive":
        family = "setup"
        point_pool = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        rule_id = "positive_setup_depth_flexible"
    elif last_action_family == "defensive":
        family = "defensive"
        point_pool = (7, 8, 9, 0) if strength != "soft" else (4, 5, 6, 7, 8, 9)
        rule_id = "positive_defensive_long_terminal"
    else:
        family = "attack"
        point_pool = (7, 8, 9, 0) if spin != "underspin" else (4, 5, 6, 7, 8, 9)
        rule_id = "positive_attack_long_terminal"

    action_id = rng.choice(ACTION_IDS_BY_FAMILY[family])
    point_id = rng.choice(point_pool)
    if not compatible_action_point(action_id, point_id):
        point_id = next(pid for pid in point_pool if compatible_action_point(action_id, pid))
    return family, action_id, point_id, rule_id


def _negative_target(
    phase: str,
    prefix_len_bin: str,
    last_action_family: str,
    strength: str,
    terminal_context: bool,
    rng: random.Random,
) -> tuple[str, int, int, str]:
    if last_action_family == "control" and strength == "strong":
        return "control", 11, rng.choice((7, 8, 9)), "negative_control_long_corner_pressure"
    if last_action_family == "defensive":
        return "defensive", 12, rng.choice((1, 2, 3)), "negative_defensive_short_side_attack"
    if last_action_family == "attack" and not (phase == "receive"):
        return "attack", 3, rng.choice((1, 2, 3)), "negative_attack_short_control"
    if not terminal_context and prefix_len_bin != "long_prefix":
        return "attack", 3, 0, "negative_nonterminal_point0"
    return "control", 11, rng.choice((7, 8, 9)), "negative_control_long_corner_pressure"


def _build_row(
    row_index: int,
    sample_index: int,
    phase: str,
    prefix_len_bin: str,
    last_action_family: str,
    spin: str,
    strength: str,
    label: str,
    target_family: str,
    action_id: int,
    point_id: int,
    rule_id: str,
    terminal_context: bool,
) -> dict[str, Any]:
    synthetic_id = f"synthetic_v385_{row_index:06d}_{sample_index:02d}"
    depth = point_depth(point_id)
    return {
        "synthetic_id": synthetic_id,
        "rally_uid": synthetic_id,
        "rule_id": rule_id,
        "provenance": PROVENANCE,
        "source_type": "synthetic_rule",
        "phase": phase,
        "prefix_len_bin": prefix_len_bin,
        "last_action_family": last_action_family,
        "last_spin": spin,
        "last_strength": strength,
        "terminal_context": bool(terminal_context),
        "target_action_family": target_family,
        "target_action_id_optional": int(action_id),
        "target_point_depth": depth,
        "target_point_side": POINT_SIDE_BY_ID.get(point_id, "unknown"),
        "target_point_id_optional": int(point_id),
        "compatibility_label": label,
        "weight": 1.0 if label == "compatible" else 0.35,
    }


def generate_expanded_synthetic_examples(n_per_combo: int = 2, seed: int = 385) -> list[dict[str, Any]]:
    if n_per_combo <= 0:
        raise ValueError("n_per_combo must be positive")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    row_index = 0
    for phase in PHASES:
        for prefix_len_bin in PREFIX_LEN_BINS:
            for last_action_family in LAST_ACTION_FAMILIES:
                for spin in SPINS:
                    for strength in STRENGTHS:
                        terminal_context = _terminal_context(phase, prefix_len_bin, last_action_family, strength)
                        for sample_index in range(1, n_per_combo + 1):
                            row_index += 1
                            if (row_index + sample_index) % 3 == 0:
                                target = _negative_target(
                                    phase,
                                    prefix_len_bin,
                                    last_action_family,
                                    strength,
                                    terminal_context,
                                    rng,
                                )
                                label = "incompatible"
                            else:
                                target = _positive_target(phase, last_action_family, spin, strength, rng)
                                label = "compatible"
                            target_family, action_id, point_id, rule_id = target
                            physical = compatible_action_point(action_id, point_id)
                            if label == "compatible" and not physical:
                                raise AssertionError(f"compatible grammar made invalid pair {action_id}/{point_id}")
                            if label == "incompatible" and physical and not (
                                rule_id == "negative_nonterminal_point0" and not terminal_context
                            ):
                                raise AssertionError(f"incompatible grammar made valid pair {action_id}/{point_id}")
                            rows.append(
                                _build_row(
                                    row_index,
                                    sample_index,
                                    phase,
                                    prefix_len_bin,
                                    last_action_family,
                                    spin,
                                    strength,
                                    label,
                                    target_family,
                                    action_id,
                                    point_id,
                                    rule_id,
                                    terminal_context,
                                )
                            )
    return rows


def coverage_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    counters = {
        "phase": Counter(row["phase"] for row in rows),
        "prefix_len_bin": Counter(row["prefix_len_bin"] for row in rows),
        "last_action_family": Counter(row["last_action_family"] for row in rows),
        "target_action_family": Counter(row["target_action_family"] for row in rows),
        "target_point_depth": Counter(row["target_point_depth"] for row in rows),
        "compatibility_label": Counter(row["compatibility_label"] for row in rows),
        "rule_id": Counter(row["rule_id"] for row in rows),
    }
    summary_rows: list[dict[str, Any]] = []
    for dimension, counter in counters.items():
        for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
            summary_rows.append({"dimension": dimension, "value": value, "count": int(count)})
    return pd.DataFrame(summary_rows)


def run_pipeline(outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = generate_expanded_synthetic_examples()
    frame = pd.DataFrame(rows, columns=FIELDNAMES)
    summary = coverage_summary(rows)

    grammar_csv = outdir / "expanded_synthetic_grammar.csv"
    summary_csv = outdir / "expanded_coverage_summary.csv"
    report_json = outdir / "search_report.json"

    frame.to_csv(grammar_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    labels = frame["compatibility_label"].value_counts().to_dict()
    report = {
        "version": "v385",
        "purpose": "Expanded self-made positive/negative synthetic grammar corpus for auxiliary compatibility scoring.",
        "row_count": int(len(frame)),
        "compatible_count": int(labels.get("compatible", 0)),
        "incompatible_count": int(labels.get("incompatible", 0)),
        "unique_rules": int(frame["rule_id"].nunique()),
        "all_synthetic_uid": bool(frame["rally_uid"].astype(str).str.startswith("synthetic_").all()),
        "provenance": PROVENANCE,
        "outputs": {
            "expanded_synthetic_grammar": grammar_csv.as_posix(),
            "expanded_coverage_summary": summary_csv.as_posix(),
        },
        "constraints": [
            "No real test rally_uid values are used.",
            "Every synthetic rally_uid starts with synthetic_.",
            "Synthetic rows are auxiliary teacher/scorer evidence only.",
        ],
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "row_count": report["row_count"],
        "compatible_count": report["compatible_count"],
        "incompatible_count": report["incompatible_count"],
        "unique_rules": report["unique_rules"],
        "grammar_csv": grammar_csv.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "report_json": report_json.as_posix(),
    }


def main() -> None:
    result = run_pipeline()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
