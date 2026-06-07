"""V380 synthetic authenticity registry.

This script defines the allowed self-made synthetic grammar rules before any
synthetic data is generated. It writes only audit artifacts under the V380
output directory and does not read hidden labels, test-row answers, TTMATCH, or
old-server sources.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v380_synthetic_authenticity_registry"

PROVENANCE = "self_made_table_tennis_grammar"
INTENDED_USE = "rare_action_point_auxiliary_teacher_only"
RARE_ACTION_IDS = (3, 4, 5, 7, 8, 9, 12, 14)
RARE_POINT_IDS = (1, 3, 4, 6, 7, 9, 0)
COARSE_TARGETS = (
    "action_family",
    "point_depth",
    "point_side",
    "terminal",
    "action_point_compatibility",
)
REQUIRED_RULE_FIELDS = (
    "rule_id",
    "motivation",
    "allowed_targets",
    "provenance",
    "not_test_specific",
)
FORBIDDEN_TEXT = (
    "ttmatch",
    "old-server",
    "old_server",
    "oldserver",
    "hidden label",
    "hidden test",
    "test row",
    "manual_test",
    "manual test",
)


def point_depth(point_id: int) -> str:
    value = int(point_id)
    if value == 0:
        return "terminal"
    if value in {1, 2, 3}:
        return "short"
    if value in {4, 5, 6}:
        return "half"
    if value in {7, 8, 9}:
        return "long"
    raise ValueError(f"pointId outside 0..9: {point_id}")


def point_side(point_id: int) -> str:
    value = int(point_id)
    if value == 0:
        return "terminal"
    if not 1 <= value <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return {1: "left", 2: "middle", 0: "right"}[value % 3]


def action_family(action_id: int) -> str:
    value = int(action_id)
    if 1 <= value <= 7:
        return "attack"
    if 8 <= value <= 11:
        return "control"
    if 12 <= value <= 14:
        return "defensive"
    if 15 <= value <= 18:
        return "serve"
    if value == 0:
        return "zero"
    return "unknown"


def build_rule_registry() -> list[dict[str, Any]]:
    """Return permitted synthetic rules with explicit governance metadata."""

    return [
        {
            "rule_id": "rare_attack_long_terminal_grammar",
            "motivation": "Represent rare attack classes as generic long or terminal pressure contexts.",
            "allowed_targets": [
                "action_family",
                "point_depth",
                "point_side",
                "terminal",
                "action_point_compatibility",
            ],
            "rare_action_ids": [3, 4, 5, 7],
            "rare_point_ids": [7, 9, 0],
            "target_family": "attack",
            "target_depths": ["long", "terminal"],
            "target_sides": ["left", "right", "terminal"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "rare_control_short_half_grammar",
            "motivation": "Represent rare control classes as generic short and half-table placement contexts.",
            "allowed_targets": [
                "action_family",
                "point_depth",
                "point_side",
                "action_point_compatibility",
            ],
            "rare_action_ids": [8, 9],
            "rare_point_ids": [1, 3, 4, 6],
            "target_family": "control",
            "target_depths": ["short", "half"],
            "target_sides": ["left", "right"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "rare_defensive_long_terminal_grammar",
            "motivation": "Represent rare defensive classes as generic recovery, long return, or terminal contexts.",
            "allowed_targets": [
                "action_family",
                "point_depth",
                "point_side",
                "terminal",
                "action_point_compatibility",
            ],
            "rare_action_ids": [12, 14],
            "rare_point_ids": [7, 9, 0],
            "target_family": "defensive",
            "target_depths": ["long", "terminal"],
            "target_sides": ["left", "right", "terminal"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "rare_short_side_point_grammar",
            "motivation": "Cover rare short-side point placements without using any real test-row answer.",
            "allowed_targets": ["point_depth", "point_side", "action_point_compatibility"],
            "rare_action_ids": [8, 9],
            "rare_point_ids": [1, 3],
            "target_family": "control",
            "target_depths": ["short"],
            "target_sides": ["left", "right"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "rare_half_side_point_grammar",
            "motivation": "Cover rare half-table side placements using only coarse table-tennis grammar.",
            "allowed_targets": ["point_depth", "point_side", "action_point_compatibility"],
            "rare_action_ids": [4, 7, 8, 9],
            "rare_point_ids": [4, 6],
            "target_family": "mixed_attack_control",
            "target_depths": ["half"],
            "target_sides": ["left", "right"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "rare_long_side_point_grammar",
            "motivation": "Cover rare long-side placements for attack and defensive compatibility checks.",
            "allowed_targets": ["point_depth", "point_side", "action_point_compatibility"],
            "rare_action_ids": [3, 5, 12, 14],
            "rare_point_ids": [7, 9],
            "target_family": "attack_or_defensive",
            "target_depths": ["long"],
            "target_sides": ["left", "right"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "rare_terminal_point0_grammar",
            "motivation": "Treat point0 as a terminal grammar target only, never as a manual row correction.",
            "allowed_targets": ["terminal", "action_point_compatibility"],
            "rare_action_ids": [3, 12, 14],
            "rare_point_ids": [0],
            "target_family": "terminal",
            "target_depths": ["terminal"],
            "target_sides": ["terminal"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
        {
            "rule_id": "coarse_action_point_compatibility_guard",
            "motivation": "Provide a generic compatibility guard for future synthetic teacher scoring.",
            "allowed_targets": ["action_family", "point_depth", "point_side", "terminal", "action_point_compatibility"],
            "rare_action_ids": list(RARE_ACTION_IDS),
            "rare_point_ids": list(RARE_POINT_IDS),
            "target_family": "coarse_compatibility",
            "target_depths": ["short", "half", "long", "terminal"],
            "target_sides": ["left", "right", "terminal"],
            "provenance": PROVENANCE,
            "not_test_specific": True,
        },
    ]


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


def validate_rule_registry(rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate registry provenance, scope, target vocabulary, and coverage."""

    errors: list[str] = []
    warnings: list[str] = []
    seen_rule_ids: set[str] = set()

    if not isinstance(rules, list) or not rules:
        errors.append("registry must be a non-empty list of rules")
        return {"ok": False, "errors": errors, "warnings": warnings}

    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rule[{idx}] must be a mapping")
            continue

        rule_id = str(rule.get("rule_id", f"rule[{idx}]"))
        for field in REQUIRED_RULE_FIELDS:
            if field not in rule:
                errors.append(f"{rule_id}: missing required field {field}")

        if not rule.get("rule_id"):
            errors.append(f"rule[{idx}]: missing rule_id")
        elif rule_id in seen_rule_ids:
            errors.append(f"{rule_id}: duplicate rule_id")
        seen_rule_ids.add(rule_id)

        allowed_targets = rule.get("allowed_targets")
        if not isinstance(allowed_targets, list) or not allowed_targets:
            errors.append(f"{rule_id}: allowed_targets must be a non-empty list")
        else:
            invalid_targets = sorted(set(allowed_targets) - set(COARSE_TARGETS))
            if invalid_targets:
                errors.append(f"{rule_id}: invalid allowed_targets {invalid_targets}")

        if not rule.get("motivation"):
            errors.append(f"{rule_id}: missing motivation")

        if not rule.get("provenance"):
            errors.append(f"{rule_id}: missing provenance")
        elif rule.get("provenance") != PROVENANCE:
            errors.append(f"{rule_id}: provenance must be {PROVENANCE}")

        if rule.get("not_test_specific") is not True:
            errors.append(f"{rule_id}: rule is test-specific or lacks not_test_specific=True")

        text = _flatten_text(rule).lower()
        forbidden_hits = sorted({term for term in FORBIDDEN_TEXT if term in text})
        if forbidden_hits:
            errors.append(f"{rule_id}: forbidden test-specific or disallowed source text {forbidden_hits}")

    covered_actions = {int(a) for rule in rules for a in rule.get("rare_action_ids", [])}
    covered_points = {int(p) for rule in rules for p in rule.get("rare_point_ids", [])}
    covered_targets = {target for rule in rules for target in rule.get("allowed_targets", [])}
    missing_actions = sorted(set(RARE_ACTION_IDS) - covered_actions)
    missing_points = sorted(set(RARE_POINT_IDS) - covered_points)
    missing_targets = sorted(set(COARSE_TARGETS) - covered_targets)
    if missing_actions:
        warnings.append(f"missing rare action coverage: {missing_actions}")
    if missing_points:
        warnings.append(f"missing rare point coverage: {missing_points}")
    if missing_targets:
        warnings.append(f"missing coarse target coverage: {missing_targets}")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def build_rare_class_targets(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rule_by_action: dict[int, list[str]] = {action: [] for action in RARE_ACTION_IDS}
    rule_by_point: dict[int, list[str]] = {point: [] for point in RARE_POINT_IDS}
    for rule in rules:
        for action in rule.get("rare_action_ids", []):
            if int(action) in rule_by_action:
                rule_by_action[int(action)].append(rule["rule_id"])
        for point in rule.get("rare_point_ids", []):
            if int(point) in rule_by_point:
                rule_by_point[int(point)].append(rule["rule_id"])

    rows: list[dict[str, Any]] = []
    for action in RARE_ACTION_IDS:
        rows.append(
            {
                "target_type": "rare_action",
                "target_id": action,
                "target_family": action_family(action),
                "target_depth": "",
                "target_side": "",
                "coverage_rule_ids": ";".join(sorted(rule_by_action[action])),
                "minimum_synthetic_examples": 8,
                "intended_use": INTENDED_USE,
                "provenance": PROVENANCE,
                "not_test_specific": True,
            }
        )

    for point in RARE_POINT_IDS:
        rows.append(
            {
                "target_type": "rare_point",
                "target_id": point,
                "target_family": "terminal" if point == 0 else "placement",
                "target_depth": point_depth(point),
                "target_side": point_side(point),
                "coverage_rule_ids": ";".join(sorted(rule_by_point[point])),
                "minimum_synthetic_examples": 8,
                "intended_use": INTENDED_USE,
                "provenance": PROVENANCE,
                "not_test_specific": True,
            }
        )
    return rows


def authenticity_checklist(rules: list[dict[str, Any]], validation: dict[str, Any]) -> str:
    status = "PASS" if validation["ok"] and not validation["warnings"] else "REVIEW"
    lines = [
        "# V380 Synthetic Authenticity Checklist",
        "",
        f"- Registry validation status: {status}",
        f"- Rule count: {len(rules)}",
        f"- Provenance: {PROVENANCE}",
        "- Source type: self-made rule-based table-tennis grammar",
        "- Intended use: rare action/point auxiliary teacher and compatibility scoring only",
        "- Not used for: test-row answer generation, manual row correction, hidden labels, TTMATCH, old-server labels",
        "- Required rule fields: rule_id, motivation, allowed_targets, provenance, not_test_specific",
        "- Required synthetic record fields for downstream tasks: synthetic_id, rule_id, provenance, target_family, target_depth/side when applicable",
        "",
        "## Validation Messages",
    ]
    if validation["errors"]:
        lines.extend(f"- ERROR: {msg}" for msg in validation["errors"])
    if validation["warnings"]:
        lines.extend(f"- WARNING: {msg}" for msg in validation["warnings"])
    if not validation["errors"] and not validation["warnings"]:
        lines.append("- No validation errors or warnings.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path = OUTDIR) -> dict[str, str]:
    rules = build_rule_registry()
    validation = validate_rule_registry(rules)
    if not validation["ok"]:
        raise ValueError(f"invalid synthetic rule registry: {validation['errors']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "synthetic_rule_registry.json"
    targets_path = output_dir / "rare_class_targets.csv"
    checklist_path = output_dir / "authenticity_checklist.md"
    report_path = output_dir / "search_report.json"

    registry_payload = {
        "version": "v380",
        "source_type": "self_made_rule_based_synthetic_data_registry",
        "intended_use": INTENDED_USE,
        "disallowed_uses": [
            "test_row_answer_generation",
            "manual_test_row_correction",
            "hidden_test_label_mapping",
            "ttmatch",
            "old_server_labels",
        ],
        "coarse_targets": list(COARSE_TARGETS),
        "rare_action_ids": list(RARE_ACTION_IDS),
        "rare_point_ids": list(RARE_POINT_IDS),
        "validation": validation,
        "rules": rules,
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_rows = build_rare_class_targets(rules)
    with targets_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0].keys()))
        writer.writeheader()
        writer.writerows(target_rows)

    checklist_path.write_text(authenticity_checklist(rules, validation), encoding="utf-8")

    search_report = {
        "version": "v380",
        "task": "synthetic_authenticity_registry",
        "status": "complete",
        "inputs_read": [],
        "outputs": [
            registry_path.relative_to(ROOT).as_posix(),
            targets_path.relative_to(ROOT).as_posix(),
            checklist_path.relative_to(ROOT).as_posix(),
            report_path.relative_to(ROOT).as_posix(),
        ],
        "rule_count": len(rules),
        "rare_action_ids": list(RARE_ACTION_IDS),
        "rare_point_ids": list(RARE_POINT_IDS),
        "coarse_targets": list(COARSE_TARGETS),
        "provenance": PROVENANCE,
        "not_test_specific": True,
        "disallowed_sources_confirmed_absent": ["TTMATCH", "old-server", "hidden test labels"],
        "manual_row_edits": False,
        "notes": [
            "Registry is generic table-tennis grammar for downstream synthetic generation.",
            "No candidate submission or test-label source is read by V380.",
        ],
    }
    report_path.write_text(json.dumps(search_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "registry": str(registry_path),
        "targets": str(targets_path),
        "checklist": str(checklist_path),
        "search_report": str(report_path),
    }


def main() -> dict[str, str]:
    paths = write_outputs()
    print(json.dumps(paths, indent=2, sort_keys=True))
    return paths


if __name__ == "__main__":
    main()
