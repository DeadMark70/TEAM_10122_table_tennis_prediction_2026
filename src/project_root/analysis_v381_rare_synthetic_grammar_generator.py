"""V381 rare synthetic grammar generator.

Creates generic, self-made table-tennis grammar rows for rare action and point
contexts. The rows are synthetic teacher material only; they do not encode or
reference any real test rally identifiers.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OUTPUT_DIR = Path("v381_rare_synthetic_grammar_generator")
V380_REGISTRY = Path("v380_synthetic_authenticity_registry/synthetic_rule_registry.json")
PROVENANCE = "self_made_table_tennis_grammar"

RARE_ACTION_IDS = (3, 4, 5, 7, 8, 9, 12, 14)
RARE_POINT_IDS = (0, 1, 3, 4, 6, 7, 9)
SHORT_POINTS = {1, 3}
HALF_POINTS = {4, 6}
LONG_POINTS = {7, 9}
TERMINAL_POINTS = {0}

FIELDNAMES = [
    "synthetic_id",
    "rally_uid",
    "rule_id",
    "provenance",
    "source_type",
    "prefix_len_bin",
    "phase",
    "last_action_family",
    "last_point_depth",
    "last_spin",
    "last_strength",
    "target_action_family",
    "target_action_id_optional",
    "target_point_depth",
    "target_point_side",
    "target_point_id_optional",
    "terminal",
    "compatibility_label",
    "weight",
]


def _fallback_rules() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "rare_action_3_terminal_long_attack",
            "target_family": "attack",
            "target_action_ids": [3],
            "depths": ["long", "terminal"],
            "point_ids": [7, 9, 0],
            "primary_point_id": 7,
            "sides": ["left", "right", "center"],
            "terminal": [True, False],
            "phase": "late_attack",
            "last_action_family": "setup",
            "last_depth": "half",
            "weight": 1.25,
        },
        {
            "rule_id": "rare_action_4_receive_short_half",
            "target_family": "receive",
            "target_action_ids": [4],
            "depths": ["short", "half"],
            "point_ids": [1, 3, 4, 6],
            "primary_point_id": 1,
            "sides": ["left", "right"],
            "terminal": [False],
            "phase": "receive",
            "last_action_family": "serve",
            "last_depth": "short",
            "weight": 1.15,
        },
        {
            "rule_id": "rare_action_5_fast_attack_long",
            "target_family": "attack",
            "target_action_ids": [5],
            "depths": ["long", "half"],
            "point_ids": [4, 6, 7, 9],
            "primary_point_id": 4,
            "sides": ["left", "right"],
            "terminal": [False],
            "phase": "early_attack",
            "last_action_family": "receive",
            "last_depth": "half",
            "weight": 1.2,
        },
        {
            "rule_id": "rare_action_7_early_attack_by_depth",
            "target_family": "early_attack",
            "target_action_ids": [7],
            "depths": ["short", "half", "long"],
            "point_ids": [1, 3, 4, 6, 7, 9],
            "primary_point_id": 6,
            "sides": ["left", "right", "center"],
            "terminal": [False],
            "phase": "early_attack",
            "last_action_family": "receive",
            "last_depth": "short",
            "weight": 1.2,
        },
        {
            "rule_id": "rare_action_8_short_control",
            "target_family": "control",
            "target_action_ids": [8],
            "depths": ["short"],
            "point_ids": [1, 3],
            "primary_point_id": 3,
            "sides": ["left", "right"],
            "terminal": [False],
            "phase": "control",
            "last_action_family": "control",
            "last_depth": "short",
            "weight": 1.1,
        },
        {
            "rule_id": "rare_action_9_half_control",
            "target_family": "control",
            "target_action_ids": [9],
            "depths": ["half", "short"],
            "point_ids": [1, 3, 4, 6],
            "primary_point_id": 4,
            "sides": ["left", "right"],
            "terminal": [False],
            "phase": "control",
            "last_action_family": "receive",
            "last_depth": "short",
            "weight": 1.1,
        },
        {
            "rule_id": "rare_action_12_defensive_long",
            "target_family": "defense",
            "target_action_ids": [12],
            "depths": ["long"],
            "point_ids": [7, 9],
            "primary_point_id": 9,
            "sides": ["left", "right", "center"],
            "terminal": [False, True],
            "phase": "defense",
            "last_action_family": "attack",
            "last_depth": "long",
            "weight": 1.2,
        },
        {
            "rule_id": "rare_action_14_defensive_terminal",
            "target_family": "defense",
            "target_action_ids": [14],
            "depths": ["long", "terminal"],
            "point_ids": [7, 9, 0],
            "primary_point_id": 0,
            "sides": ["left", "right", "center"],
            "terminal": [True, False],
            "phase": "defense",
            "last_action_family": "attack",
            "last_depth": "long",
            "weight": 1.25,
        },
        {
            "rule_id": "rare_point_1_3_short_side",
            "target_family": "control",
            "target_action_ids": [8, 9, 11],
            "depths": ["short"],
            "point_ids": [1, 3],
            "primary_point_id": 1,
            "sides": ["left", "right"],
            "terminal": [False],
            "phase": "short_game",
            "last_action_family": "serve",
            "last_depth": "short",
            "weight": 1.0,
        },
        {
            "rule_id": "rare_point_4_6_half_side",
            "target_family": "receive",
            "target_action_ids": [4, 5, 7],
            "depths": ["half"],
            "point_ids": [4, 6],
            "primary_point_id": 6,
            "sides": ["left", "right"],
            "terminal": [False],
            "phase": "transition",
            "last_action_family": "control",
            "last_depth": "short",
            "weight": 1.0,
        },
        {
            "rule_id": "rare_point_7_9_long_side",
            "target_family": "attack",
            "target_action_ids": [3, 5, 12, 14],
            "depths": ["long"],
            "point_ids": [7, 9],
            "primary_point_id": 7,
            "sides": ["left", "right"],
            "terminal": [False, True],
            "phase": "long_rally",
            "last_action_family": "attack",
            "last_depth": "half",
            "weight": 1.0,
        },
        {
            "rule_id": "rare_point_0_terminal",
            "target_family": "terminal",
            "target_action_ids": [3, 12, 14],
            "depths": ["terminal"],
            "point_ids": [0],
            "primary_point_id": 0,
            "sides": ["none"],
            "terminal": [True],
            "phase": "terminal",
            "last_action_family": "attack",
            "last_depth": "long",
            "weight": 1.3,
        },
    ]


def _coerce_v380_rules(registry: Any) -> list[dict[str, Any]]:
    if isinstance(registry, dict):
        raw_rules = registry.get("rules") or registry.get("synthetic_rules") or []
    else:
        raw_rules = registry

    rules: list[dict[str, Any]] = []
    for raw in raw_rules if isinstance(raw_rules, list) else []:
        if not isinstance(raw, dict):
            continue
        rule_id = str(raw.get("rule_id", "")).strip()
        if not rule_id:
            continue
        allowed = set(raw.get("allowed_targets") or [])
        target_family = raw.get("target_family") or raw.get("action_family") or "grammar"
        rules.append(
            {
                "rule_id": rule_id,
                "target_family": target_family,
                "target_action_ids": list(
                    raw.get("target_action_ids") or raw.get("rare_action_ids") or raw.get("rare_actions") or []
                ),
                "depths": list(raw.get("depths") or raw.get("target_depths") or ["short", "half", "long"]),
                "point_ids": list(raw.get("point_ids") or raw.get("rare_point_ids") or raw.get("rare_points") or []),
                "primary_point_id": raw.get("primary_point_id"),
                "sides": list(raw.get("sides") or raw.get("target_sides") or ["left", "right"]),
                "terminal": list(raw.get("terminal") or ([True] if "terminal" in allowed else [False])),
                "phase": raw.get("phase") or "registry_rule",
                "last_action_family": raw.get("last_action_family") or "context",
                "last_depth": raw.get("last_point_depth") or raw.get("last_depth") or "half",
                "weight": float(raw.get("weight", 1.0)),
            }
        )
    return [rule for rule in rules if rule["target_action_ids"] or rule["point_ids"]]


def load_rule_registry(registry_path: Path | str = V380_REGISTRY) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = Path(registry_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
        if isinstance(registry, dict):
            rare_actions = set(registry.get("rare_action_ids") or [])
            rare_points = set(registry.get("rare_point_ids") or [])
            if set(RARE_ACTION_IDS).issubset(rare_actions) and set(RARE_POINT_IDS).issubset(rare_points):
                return _fallback_rules(), {
                    "v380_registry_source": "v380_registry",
                    "v380_registry_note": (
                        f"Loaded V380 registry from {path.as_posix()}; V381 expanded its permitted rare "
                        "action/point sets into detailed deterministic grammar templates."
                    ),
                }
        rules = _coerce_v380_rules(registry)
        if rules:
            return rules, {
                "v380_registry_source": "v380_registry",
                "v380_registry_note": f"Loaded synthetic rules from {path.as_posix()}.",
            }
        return _fallback_rules(), {
            "v380_registry_source": "fallback_rules",
            "v380_registry_note": f"V380 registry at {path.as_posix()} had no usable rare grammar rules; used built-in fallback rules.",
        }

    return _fallback_rules(), {
        "v380_registry_source": "fallback_rules",
        "v380_registry_note": f"V380 registry absent at {path.as_posix()}; used built-in robust fallback rules.",
    }


def point_depth(point_id: int | None) -> str:
    if point_id in SHORT_POINTS:
        return "short"
    if point_id in HALF_POINTS:
        return "half"
    if point_id in LONG_POINTS:
        return "long"
    if point_id in TERMINAL_POINTS:
        return "terminal"
    return "unknown"


def compatible_action_point(action_id: int | None, point_id: int | None) -> bool:
    if action_id is None or point_id is None:
        return False
    if point_id == 0:
        return action_id in {3, 12, 14}
    if action_id == 11:
        return point_id in SHORT_POINTS | HALF_POINTS
    if action_id in {8, 9}:
        return point_id in SHORT_POINTS | HALF_POINTS
    if action_id in {4, 7}:
        return point_id in SHORT_POINTS | HALF_POINTS | LONG_POINTS
    if action_id in {3, 5, 12, 14}:
        return point_id in HALF_POINTS | LONG_POINTS | TERMINAL_POINTS
    return point_id in RARE_POINT_IDS


def _choice(rng: random.Random, values: list[Any]) -> Any:
    if not values:
        return None
    return values[rng.randrange(len(values))]


def _row_for_rule(rule: dict[str, Any], rule_index: int, sample_index: int, rng: random.Random) -> dict[str, Any]:
    action_id = _choice(rng, list(rule["target_action_ids"]))
    if sample_index == 1 and rule.get("primary_point_id") in set(rule["point_ids"]):
        point_id = rule["primary_point_id"]
    else:
        point_id = _choice(rng, list(rule["point_ids"]))
    depth = point_depth(point_id)
    if depth == "unknown":
        depth = _choice(rng, list(rule["depths"]))
    terminal = bool(_choice(rng, list(rule["terminal"])))
    if point_id == 0:
        terminal = True
        depth = "terminal"
    side = "none" if depth == "terminal" else _choice(rng, list(rule["sides"]))
    compatible = compatible_action_point(action_id, point_id)
    prefix_len_bin = _choice(rng, ["short_prefix", "mid_prefix", "long_prefix"])
    last_spin = _choice(rng, ["underspin", "topspin", "sidespin", "flat"])
    last_strength = _choice(rng, ["soft", "medium", "strong"])

    synthetic_id = f"synthetic_v381_{rule_index:03d}_{sample_index:03d}"
    return {
        "synthetic_id": synthetic_id,
        "rally_uid": f"synthetic_rally_{rule_index:03d}_{sample_index:03d}",
        "rule_id": rule["rule_id"],
        "provenance": PROVENANCE,
        "source_type": "synthetic_rule",
        "prefix_len_bin": prefix_len_bin,
        "phase": rule["phase"],
        "last_action_family": rule["last_action_family"],
        "last_point_depth": rule["last_depth"],
        "last_spin": last_spin,
        "last_strength": last_strength,
        "target_action_family": rule["target_family"],
        "target_action_id_optional": action_id,
        "target_point_depth": depth,
        "target_point_side": side,
        "target_point_id_optional": point_id,
        "terminal": terminal,
        "compatibility_label": "compatible" if compatible else "incompatible",
        "weight": float(rule["weight"] if compatible else min(rule["weight"], 0.5)),
    }


def generate_synthetic_examples(n_per_rule: int = 8, seed: int = 381) -> list[dict[str, Any]]:
    if n_per_rule <= 0:
        raise ValueError("n_per_rule must be positive")

    rules, _meta = load_rule_registry()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(rules, start=1):
        for sample_index in range(1, n_per_rule + 1):
            rows.append(_row_for_rule(rule, rule_index, sample_index, rng))
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def coverage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters = {
        "rule_id": Counter(row["rule_id"] for row in rows),
        "target_action_id_optional": Counter(row["target_action_id_optional"] for row in rows),
        "target_point_id_optional": Counter(row["target_point_id_optional"] for row in rows),
        "target_point_depth": Counter(row["target_point_depth"] for row in rows),
        "target_point_side": Counter(row["target_point_side"] for row in rows),
        "compatibility_label": Counter(row["compatibility_label"] for row in rows),
    }
    summary: list[dict[str, Any]] = []
    for dimension, counter in counters.items():
        for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
            summary.append({"dimension": dimension, "value": value, "count": count})
    return summary


def write_outputs(
    out_dir: Path | str = OUTPUT_DIR,
    n_per_rule: int = 12,
    seed: int = 381,
    registry_path: Path | str = V380_REGISTRY,
) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules, registry_meta = load_rule_registry(registry_path)
    rng = random.Random(seed)
    rows = [
        _row_for_rule(rule, rule_index, sample_index, rng)
        for rule_index, rule in enumerate(rules, start=1)
        for sample_index in range(1, n_per_rule + 1)
    ]

    grammar_csv = output_dir / "synthetic_rare_grammar.csv"
    summary_csv = output_dir / "synthetic_coverage_summary.csv"
    report_json = output_dir / "search_report.json"
    summary = coverage_summary(rows)

    _write_csv(grammar_csv, rows, FIELDNAMES)
    _write_csv(summary_csv, summary, ["dimension", "value", "count"])

    report = {
        "version": "v381",
        "purpose": "Generate self-made rare-class synthetic sequence examples from generic table-tennis grammar.",
        "source_type": "synthetic_rule",
        "provenance": PROVENANCE,
        "n_per_rule": n_per_rule,
        "row_count": len(rows),
        "rule_count": len(rules),
        "rare_action_ids": list(RARE_ACTION_IDS),
        "rare_point_ids": list(RARE_POINT_IDS),
        "outputs": {
            "synthetic_rare_grammar": grammar_csv.as_posix(),
            "synthetic_coverage_summary": summary_csv.as_posix(),
        },
        "constraints": [
            "No TTMATCH or old-server labels used.",
            "No hidden test labels used.",
            "No real test rally_uid values used; every rally_uid starts with synthetic_.",
            "Synthetic data is for auxiliary rare grammar teacher use only.",
        ],
        **registry_meta,
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "row_count": len(rows),
        "rule_count": len(rules),
        "grammar_csv": grammar_csv.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "report_json": report_json.as_posix(),
        **registry_meta,
    }


def main() -> None:
    result = write_outputs()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
