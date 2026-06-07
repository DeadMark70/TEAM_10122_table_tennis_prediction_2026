"""V340 no-p0 point agreement ensemble.

Builds point-only local candidates from existing point submissions by aligning
on rally_uid. Exports stay local to v340_no_p0_point_agreement_ensemble and
preserve V306/V338 actionId and serverGetPoint exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
OUTDIR = ROOT / "v340_no_p0_point_agreement_ensemble"
BASE_ANCHOR_PATH = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
PUBLIC_ANCHOR_PATH = (
    ROOT
    / "v338_joint_moe_pack"
    / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
)
MIN_EVIDENCE_DELTA = 0.003


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    delta: float | None = None


@dataclass(frozen=True)
class VariantSpec:
    name: str
    min_agreement: int
    budget: int | None
    filter_kind: str = "all"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "v338_point_moe_no_p0_add_b24",
        PUBLIC_ANCHOR_PATH,
        0.00320717625047956,
    ),
    SourceSpec(
        "v337_point_moe_no_p0_add_b24",
        ROOT / "v337_point_moe" / "submission_v337_point_moe_no_p0_add_b24__v173action_v300server.csv",
        0.00320717625047956,
    ),
    SourceSpec(
        "v334_point_only_v333_no_p0_add_cap24",
        ROOT
        / "v334_joint_hierarchical_action_point"
        / "submission_v334_point_only_v333_no_p0_add_cap24__v173action_v300server.csv",
        0.00320717625047956,
    ),
    SourceSpec(
        "v333_no_p0_add_cap24",
        ROOT / "v333_hierarchical_point_model" / "submission_v333_no_p0_add_cap24__v173action_v300server.csv",
        0.00320717625047956,
    ),
    SourceSpec(
        "v322_modelbank_agree12",
        ROOT / "v322_nonterminal_point_modelbank" / "submission_v322_modelbank_agree12__v173action_v300server.csv",
        0.0002883216849463577,
    ),
    SourceSpec(
        "v322_modelbank_agree24",
        ROOT / "v322_nonterminal_point_modelbank" / "submission_v322_modelbank_agree24__v173action_v300server.csv",
        0.0000513697773361077,
    ),
    SourceSpec(
        "v322_long_half_combo18",
        ROOT / "v322_nonterminal_point_modelbank" / "submission_v322_long_half_combo18__v173action_v300server.csv",
        0.0002517238455426174,
    ),
    SourceSpec(
        "v322_actioncond_highmargin18",
        ROOT / "v322_nonterminal_point_modelbank" / "submission_v322_actioncond_highmargin18__v173action_v300server.csv",
        0.0002517238455426174,
    ),
    SourceSpec(
        "v272_actioncond_cap0p005",
        ROOT
        / "v272_action_conditioned_point_residual"
        / "submission_v272_point_actioncond_cap0p005__v173action_r121server.csv",
        None,
    ),
    SourceSpec(
        "v272_actioncond_cap0p010",
        ROOT
        / "v272_action_conditioned_point_residual"
        / "submission_v272_point_actioncond_cap0p010__v173action_r121server.csv",
        None,
    ),
    SourceSpec(
        "v272_actioncond_cap0p015",
        ROOT
        / "v272_action_conditioned_point_residual"
        / "submission_v272_point_actioncond_cap0p015__v173action_r121server.csv",
        None,
    ),
    SourceSpec(
        "v272_actioncond_table_cap0p010",
        ROOT
        / "v272_action_conditioned_point_residual"
        / "submission_v272_point_actioncond_table_cap0p010__v173action_r121server.csv",
        None,
    ),
)


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("v340_agree2_b12", 2, 12),
    VariantSpec("v340_agree2_b24", 2, 24),
    VariantSpec("v340_agree2_b36", 2, 36),
    VariantSpec("v340_agree3_all", 3, None),
    VariantSpec("v340_agree2_longside_only_b24", 2, 24, "longside_only"),
    VariantSpec("v340_agree2_depth_preserve_b24", 2, 24, "depth_preserve"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    return (point - 1) // 3


def agreement_score(base: np.ndarray, sources: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row agreement count and most common nonzero target.

    Targets are counted only when a source proposes a nonzero value different
    from base. Nonzero-to-zero additions are blocked by returning base.
    """

    base_arr = np.asarray(base, dtype=int)
    if not sources:
        return np.zeros(len(base_arr), dtype=int), base_arr.copy()
    arrays = {name: np.asarray(values, dtype=int) for name, values in sources.items()}
    if any(len(values) != len(base_arr) for values in arrays.values()):
        raise ValueError("all sources must have the same length as base")

    score = np.zeros(len(base_arr), dtype=int)
    target = base_arr.copy()
    for i, old in enumerate(base_arr):
        counts: dict[int, int] = {}
        first_seen: dict[int, int] = {}
        for order, values in enumerate(arrays.values()):
            proposed = int(values[i])
            if proposed == int(old) or proposed == 0:
                continue
            counts[proposed] = counts.get(proposed, 0) + 1
            first_seen.setdefault(proposed, order)
        if counts:
            best_count = max(counts.values())
            best_targets = [value for value, count in counts.items() if count == best_count]
            chosen = min(best_targets, key=lambda value: first_seen[value])
            score[i] = int(best_count)
            target[i] = int(chosen)
    return score, target


def enforce_no_p0_add(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    base_arr = np.asarray(base, dtype=int)
    out = np.asarray(cand, dtype=int).copy()
    if len(base_arr) != len(out):
        raise ValueError("base and cand must have the same length")
    blocked = (base_arr != 0) & (out == 0)
    out[blocked] = base_arr[blocked]
    return out


def select_candidate(
    base: np.ndarray,
    target: np.ndarray,
    agreement: np.ndarray,
    *,
    min_agreement: int,
    budget: int | None,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    base_arr = np.asarray(base, dtype=int)
    target_arr = enforce_no_p0_add(base_arr, np.asarray(target, dtype=int))
    score = np.asarray(agreement, dtype=int)
    if not (len(base_arr) == len(target_arr) == len(score)):
        raise ValueError("base, target, and agreement must have the same length")
    eligible = (target_arr != base_arr) & (score >= int(min_agreement))
    if eligible_mask is not None:
        mask = np.asarray(eligible_mask, dtype=bool)
        if len(mask) != len(base_arr):
            raise ValueError("eligible_mask length mismatch")
        eligible &= mask
    out = base_arr.copy()
    if not eligible.any():
        return out
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    if budget is not None:
        order = order[: min(int(budget), len(order))]
    out[order] = target_arr[order]
    return enforce_no_p0_add(base_arr, out)


def build_export_frame(anchor: pd.DataFrame, point: np.ndarray) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["pointId"] = np.asarray(point, dtype=int)
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("V340 export changed actionId")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("V340 export changed serverGetPoint")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def _load_submission(path: Path, expected_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    if frame["rally_uid"].duplicated().any():
        raise ValueError(f"duplicate rally_uid in {path}")
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def _align_source_points(source: pd.DataFrame, anchor_uids: pd.Series) -> np.ndarray:
    aligned = source.set_index("rally_uid").reindex(anchor_uids)
    if aligned["pointId"].isna().any():
        raise ValueError("source is missing anchor rally_uid values")
    return aligned["pointId"].to_numpy(dtype=int)


def load_sources(anchor: pd.DataFrame, specs: tuple[SourceSpec, ...] = SOURCE_SPECS) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    sources: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.path.exists():
            rows.append({"source": spec.name, "path": relative_path(spec.path), "status": "missing", "delta": spec.delta})
            continue
        try:
            frame = _load_submission(spec.path, expected_rows=len(anchor))
            point = _align_source_points(frame, anchor["rally_uid"])
        except Exception as exc:
            rows.append(
                {
                    "source": spec.name,
                    "path": relative_path(spec.path),
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "delta": spec.delta,
                }
            )
            continue
        sources[spec.name] = point
        rows.append(
            {
                "source": spec.name,
                "path": relative_path(spec.path),
                "status": "loaded",
                "delta": spec.delta,
                "changed_rows": int(np.sum(point != anchor["pointId"].to_numpy(dtype=int))),
                "point0_additions": int(np.sum((anchor["pointId"].to_numpy(dtype=int) != 0) & (point == 0))),
            }
        )
    return sources, rows


def _source_delta_by_name() -> dict[str, float]:
    return {spec.name: float(spec.delta) for spec in SOURCE_SPECS if spec.delta is not None}


def _row_delta_proxy(base: np.ndarray, target: np.ndarray, sources: dict[str, np.ndarray]) -> np.ndarray:
    deltas = _source_delta_by_name()
    out = np.zeros(len(base), dtype=float)
    for i, chosen in enumerate(np.asarray(target, dtype=int)):
        votes = [
            deltas[name]
            for name, values in sources.items()
            if name in deltas and int(values[i]) == int(chosen) and int(chosen) != int(base[i])
        ]
        out[i] = max(votes) if votes else 0.0
    return out


def _eligible_mask(base: np.ndarray, target: np.ndarray, kind: str) -> np.ndarray:
    base_arr = np.asarray(base, dtype=int)
    target_arr = np.asarray(target, dtype=int)
    if kind == "all":
        return np.ones(len(base_arr), dtype=bool)
    if kind == "longside_only":
        return np.isin(base_arr, [7, 8, 9]) | np.isin(target_arr, [7, 8, 9])
    if kind == "depth_preserve":
        return np.array(
            [
                old != 0 and new != 0 and point_depth(int(old)) == point_depth(int(new))
                for old, new in zip(base_arr, target_arr)
            ],
            dtype=bool,
        )
    raise ValueError(f"unknown filter kind: {kind}")


def _transition_counts(base: np.ndarray, cand: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for old, new in zip(np.asarray(base, dtype=int), np.asarray(cand, dtype=int)):
        if int(old) == int(new):
            continue
        key = f"{int(old)}->{int(new)}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_empty_blocked(outdir: Path, reason: str) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(safe_output_path(outdir, "candidate_summary.csv"), index=False)
    pd.DataFrame().to_csv(safe_output_path(outdir, "agreement_report.csv"), index=False)
    report = {
        "version": "V340",
        "decision": "BLOCKED",
        "verdict": "BLOCKED",
        "reason": reason,
        "generated_submissions": [],
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "fixed_action": True,
            "fixed_server": True,
            "no_upload_candidates_writes": True,
            "manual_row_edits": False,
            "no_point0_additions": True,
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), report)
    return report


def run_pipeline(*, outdir: Path = OUTDIR, expected_rows: int | None = 1845) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not BASE_ANCHOR_PATH.exists():
        return _write_empty_blocked(outdir, f"missing base anchor: {relative_path(BASE_ANCHOR_PATH)}")

    anchor = _load_submission(BASE_ANCHOR_PATH, expected_rows=expected_rows)
    public_anchor = _load_submission(PUBLIC_ANCHOR_PATH, expected_rows=len(anchor)) if PUBLIC_ANCHOR_PATH.exists() else None
    base_point = anchor["pointId"].to_numpy(dtype=int)
    public_point = public_anchor["pointId"].to_numpy(dtype=int) if public_anchor is not None else base_point.copy()

    sources, source_rows = load_sources(anchor)
    agreement, target = agreement_score(base_point, sources)
    row_delta = _row_delta_proxy(base_point, target, sources)

    summary_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    best_export: dict[str, Any] | None = None

    for spec in VARIANTS:
        mask = _eligible_mask(base_point, target, spec.filter_kind)
        pred = select_candidate(
            base_point,
            target,
            agreement,
            min_agreement=spec.min_agreement,
            budget=spec.budget,
            eligible_mask=mask,
        )
        changed = pred != base_point
        churn = point_distribution_report(base_point, pred)
        public_churn = point_distribution_report(public_point, pred)
        max_agreement = int(agreement[changed].max()) if changed.any() else 0
        evidence_delta_proxy = float(row_delta[changed].max()) if changed.any() else 0.0
        duplicate_public_anchor = bool(public_anchor is not None and np.array_equal(pred, public_point))
        export_allowed = bool(
            churn["changed_rows"] > 0
            and churn["point0_additions"] == 0
            and (
                evidence_delta_proxy >= MIN_EVIDENCE_DELTA
                or (max_agreement >= 3 and churn["changed_rows"] <= 24)
            )
        )
        decision = "EXPORT_LOCAL" if export_allowed else "DO_NOT_EXPORT"
        record = {
            "candidate": spec.name,
            "min_agreement": spec.min_agreement,
            "budget": -1 if spec.budget is None else spec.budget,
            "filter_kind": spec.filter_kind,
            "sources_loaded": len(sources),
            "test_changed_rows": int(churn["changed_rows"]),
            "changed_vs_v338_rows": int(public_churn["changed_rows"]),
            "point0_additions": int(churn["point0_additions"]),
            "point0_removals": int(churn["point0_removals"]),
            "max_agreement_changed": max_agreement,
            "mean_agreement_changed": float(agreement[changed].mean()) if changed.any() else 0.0,
            "evidence_delta_proxy": evidence_delta_proxy,
            "duplicate_public_anchor": duplicate_public_anchor,
            "transition_counts": json.dumps(_transition_counts(base_point, pred), sort_keys=True),
            "decision": decision,
        }
        if export_allowed:
            filename = f"submission_{spec.name}__v173action_v300server.csv"
            frame = build_export_frame(anchor, pred)
            path = safe_output_path(outdir, filename)
            frame.to_csv(path, index=False, float_format="%.8f")
            record["submission"] = filename
            record["path"] = relative_path(path)
            generated_item = {
                "candidate": spec.name,
                "submission": filename,
                "path": relative_path(path),
                "evidence_delta_proxy": evidence_delta_proxy,
                "test_changed_rows": int(churn["changed_rows"]),
                "duplicate_public_anchor": duplicate_public_anchor,
            }
            generated.append(generated_item)
            if best_export is None or (
                evidence_delta_proxy,
                max_agreement,
                -int(churn["changed_rows"]),
            ) > (
                float(best_export.get("evidence_delta_proxy", 0.0)),
                int(best_export.get("max_agreement_changed", 0)),
                -int(best_export.get("test_changed_rows", 0)),
            ):
                best_export = {**generated_item, "max_agreement_changed": max_agreement}
        summary_rows.append(record)

        for idx in np.where(changed)[0]:
            voters = [
                name
                for name, values in sources.items()
                if int(values[idx]) == int(pred[idx]) and int(pred[idx]) != int(base_point[idx])
            ]
            agreement_rows.append(
                {
                    "candidate": spec.name,
                    "row_id": int(idx),
                    "rally_uid": anchor.iloc[idx]["rally_uid"],
                    "base_pointId": int(base_point[idx]),
                    "new_pointId": int(pred[idx]),
                    "agreement": int(agreement[idx]),
                    "evidence_delta_proxy": float(row_delta[idx]),
                    "source_votes": "|".join(voters),
                    "point0_addition": bool(base_point[idx] != 0 and pred[idx] == 0),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["decision", "evidence_delta_proxy", "max_agreement_changed", "test_changed_rows"],
        ascending=[True, False, False, True],
    )
    summary.to_csv(safe_output_path(outdir, "candidate_summary.csv"), index=False)
    pd.DataFrame(agreement_rows).to_csv(safe_output_path(outdir, "agreement_report.csv"), index=False)
    pd.DataFrame(source_rows).to_csv(safe_output_path(outdir, "source_report.csv"), index=False)

    report = {
        "version": "V340",
        "decision": "HAS_EXPORT" if generated else "DO_NOT_UPLOAD",
        "verdict": "HAS_NO_P0_AGREEMENT_EXPORT" if generated else "NO_EXPORT_NO_EVIDENCE",
        "best_candidate": best_export or {},
        "recommended_candidate": (
            best_export
            if best_export and not bool(best_export.get("duplicate_public_anchor"))
            else None
        ),
        "generated_submissions": generated,
        "sources_loaded": len(sources),
        "source_report": relative_path(outdir / "source_report.csv"),
        "candidate_summary": relative_path(outdir / "candidate_summary.csv"),
        "agreement_report": relative_path(outdir / "agreement_report.csv"),
        "base_anchor": relative_path(BASE_ANCHOR_PATH),
        "public_anchor": relative_path(PUBLIC_ANCHOR_PATH),
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "fixed_action": "V173 action from V306 package anchor",
            "fixed_server": "V300/R121 server from V306 package anchor",
            "no_upload_candidates_writes": True,
            "manual_row_edits": False,
            "no_point0_additions": True,
            "aligned_by": "rally_uid",
        },
        "notes": [
            "Agreement counts only nonzero source targets different from the base point.",
            "Nonzero-to-zero target proposals are blocked before selection and export.",
            "recommended_candidate is null when the best local export duplicates V338.",
        ],
    }
    write_json(safe_output_path(outdir, "search_report.json"), report)
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            _json_safe(
                {
                    "outdir": relative_path(OUTDIR),
                    "decision": report.get("decision"),
                    "verdict": report.get("verdict"),
                    "generated": len(report.get("generated_submissions", [])),
                    "best": (report.get("best_candidate") or {}).get("candidate"),
                    "recommended": (
                        None
                        if report.get("recommended_candidate") is None
                        else report.get("recommended_candidate", {}).get("candidate")
                    ),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
