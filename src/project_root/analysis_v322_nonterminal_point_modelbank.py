"""V322 nonterminal point model bank.

This branch starts from the V306 point anchor and writes only local V322
outputs. It changes no point0 rows. The search combines three nonterminal
point specialists: long-side 7/8/9, half-depth 4/5/6, and action-conditioned
nonterminal priors. Rows pass only by two-family agreement or a supported
high-margin single-family vote.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_v316_nonterminal_point_correction import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    FOCUS_POINTS,
    LONGSIDE_POINTS,
    NONTERMINAL_TARGETS,
    SHORTMID_POINTS,
    _confusion_pairs,
    build_best_nonterminal_candidates,
    build_bundle as build_v316_bundle,
    count_point0_changes,
    decision_label,
    distribution,
    lookup_support,
    validate_submission_frame,
)
from baseline_lgbm import POINT_CLASSES


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v322_nonterminal_point_modelbank"
SEARCH_PATH = OUTDIR / "v322_modelbank_search.csv"
CHANGED_ROWS_PATH = OUTDIR / "v322_changed_rows.csv"
REPORT_JSON_PATH = OUTDIR / "v322_report.json"
REPORT_MD_PATH = OUTDIR / "v322_report.md"
LOCAL_ONLY_BANNED_PARTS = {"upload_candidates_20260519", "selected"}
V316_REPORT_PATH = ROOT / "v316_nonterminal_point_correction" / "v316_report.json"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    submission: str
    budget: int
    family: str
    high_margin: float
    min_support: int
    agreement_bonus: float = 0.035


@dataclass(frozen=True)
class SpecialistBank:
    candidate: np.ndarray
    score: np.ndarray
    margin: np.ndarray
    agree_count: np.ndarray
    best_family: np.ndarray
    family_candidates: dict[str, np.ndarray]
    family_margins: dict[str, np.ndarray]


CANDIDATES = [
    CandidateSpec(
        "v322_modelbank_agree12",
        "submission_v322_modelbank_agree12__v173action_v300server.csv",
        12,
        "modelbank_agree",
        high_margin=0.18,
        min_support=7,
    ),
    CandidateSpec(
        "v322_modelbank_agree24",
        "submission_v322_modelbank_agree24__v173action_v300server.csv",
        24,
        "modelbank_agree",
        high_margin=0.18,
        min_support=9,
    ),
    CandidateSpec(
        "v322_long_half_combo18",
        "submission_v322_long_half_combo18__v173action_v300server.csv",
        18,
        "long_half_combo",
        high_margin=0.18,
        min_support=6,
    ),
    CandidateSpec(
        "v322_actioncond_highmargin18",
        "submission_v322_actioncond_highmargin18__v173action_v300server.csv",
        18,
        "actioncond_highmargin",
        high_margin=0.18,
        min_support=10,
        agreement_bonus=0.015,
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def ensure_local_output_path(path: str | Path) -> Path:
    path_obj = Path(path)
    parts = {part.lower() for part in path_obj.parts}
    if parts & LOCAL_ONLY_BANNED_PARTS:
        raise ValueError(f"V322 outputs are local-only; banned path: {path}")
    resolved = path_obj if path_obj.is_absolute() else ROOT / path_obj
    try:
        resolved.resolve().relative_to(OUTDIR.resolve())
    except ValueError as exc:
        raise ValueError(f"V322 outputs are local-only under {OUTDIR}: {path}") from exc
    return resolved


def _normalize_prob(prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(prob, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return arr / denom


def blend_probs(*weighted_probs: tuple[float, np.ndarray]) -> np.ndarray:
    if not weighted_probs:
        raise ValueError("at least one probability source is required")
    total = None
    weight_sum = 0.0
    for weight, prob in weighted_probs:
        if weight <= 0:
            continue
        arr = _normalize_prob(prob)
        total = arr * float(weight) if total is None else total + arr * float(weight)
        weight_sum += float(weight)
    if total is None or weight_sum <= 0:
        raise ValueError("positive probability weights are required")
    return _normalize_prob(total / weight_sum)


def specialist_vote_bank(
    base_point: np.ndarray,
    specialists: Mapping[str, tuple[np.ndarray, Iterable[int]]],
    *,
    min_vote_margin: float = 0.02,
) -> SpecialistBank:
    """Combine specialist point votes and count same-target agreement."""
    base = np.asarray(base_point, dtype=int)
    if not specialists:
        raise ValueError("at least one specialist is required")

    family_candidates: dict[str, np.ndarray] = {}
    family_margins: dict[str, np.ndarray] = {}
    vote_records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, (prob, allowed_targets) in specialists.items():
        allowed = set(allowed_targets) & NONTERMINAL_TARGETS
        cand, margin = build_best_nonterminal_candidates(base, prob, allowed_targets=allowed)
        active = (
            np.isin(base, sorted(allowed))
            & (cand != 0)
            & (cand != base)
            & np.isfinite(margin)
            & (margin >= float(min_vote_margin))
        )
        family_candidates[name] = cand
        family_margins[name] = np.where(active, margin, -np.inf)
        vote_records.append((name, cand, margin, active))

    out_candidate = base.copy()
    out_score = np.full(len(base), -np.inf, dtype=float)
    out_margin = np.zeros(len(base), dtype=float)
    out_agree = np.zeros(len(base), dtype=int)
    out_family = np.full(len(base), "", dtype=object)

    for i, old in enumerate(base):
        votes: dict[int, list[tuple[str, float]]] = {}
        for name, cand, margin, active in vote_records:
            if not active[i]:
                continue
            votes.setdefault(int(cand[i]), []).append((name, float(margin[i])))
        if not votes:
            continue
        best_target = None
        best_key = None
        for target, rows in votes.items():
            agree = len(rows)
            mean_margin = float(np.mean([row[1] for row in rows]))
            max_margin = float(np.max([row[1] for row in rows]))
            key = (agree, mean_margin, max_margin, -abs(int(target) - int(old)))
            if best_key is None or key > best_key:
                best_key = key
                best_target = target
        assert best_target is not None
        rows = votes[best_target]
        out_candidate[i] = int(best_target)
        out_agree[i] = len(rows)
        out_margin[i] = float(np.max([row[1] for row in rows]))
        out_score[i] = float(np.mean([row[1] for row in rows]) + 0.035 * (len(rows) - 1))
        row_families = {row[0] for row in rows}
        if int(old) in LONGSIDE_POINTS and "long_side" in row_families:
            out_family[i] = "long_side"
        elif int(old) in SHORTMID_POINTS and "half_depth" in row_families:
            out_family[i] = "half_depth"
        else:
            out_family[i] = max(rows, key=lambda row: row[1])[0]

    return SpecialistBank(
        candidate=out_candidate,
        score=out_score,
        margin=out_margin,
        agree_count=out_agree,
        best_family=out_family,
        family_candidates=family_candidates,
        family_margins=family_margins,
    )


def select_modelbank_replacements(
    base_point: np.ndarray,
    candidate_point: np.ndarray,
    score: np.ndarray,
    *,
    agree_count: np.ndarray,
    margin: np.ndarray,
    support: np.ndarray,
    budget: int,
    high_margin: float = 0.25,
    min_support: int = 5,
    allowed_pairs: set[tuple[int, int]] | None = None,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(base_point, dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    agree = np.asarray(agree_count, dtype=int)
    margin_arr = np.asarray(margin, dtype=float)
    support_arr = np.asarray(support, dtype=int)
    if not (len(base) == len(cand) == len(score_arr) == len(agree) == len(margin_arr) == len(support_arr)):
        raise ValueError("all row-level arrays must have the same length")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    gate_arr = np.ones(len(base), dtype=bool) if gate is None else np.asarray(gate, dtype=bool)
    if len(gate_arr) != len(base):
        raise ValueError("gate must have the same length as base_point")

    agreement_gate = agree >= 2
    high_margin_gate = (margin_arr >= float(high_margin)) & (support_arr >= int(min_support))
    eligible = (
        (base != 0)
        & (cand != 0)
        & (base != cand)
        & np.isin(base, sorted(FOCUS_POINTS))
        & np.isin(cand, sorted(FOCUS_POINTS))
        & gate_arr
        & np.isfinite(score_arr)
        & (score_arr > 0)
        & (agreement_gate | high_margin_gate)
    )
    if allowed_pairs is not None:
        pair_ok = np.array([(int(old), int(new)) in allowed_pairs for old, new in zip(base, cand)], dtype=bool)
        eligible &= pair_ok

    selected = np.zeros(len(base), dtype=bool)
    if budget == 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score_arr[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected


def _same_group_pairs() -> set[tuple[int, int]]:
    return _confusion_pairs(SHORTMID_POINTS) | _confusion_pairs(LONGSIDE_POINTS)


def _specialist_sources(bundle: dict[str, Any], split: str) -> Mapping[str, tuple[np.ndarray, Iterable[int]]]:
    model = bundle[f"model_{split}_prob"]
    prior = bundle[f"prior_{split}_prob"]
    v188 = bundle["artifacts"][f"v188_{split}_prob"]
    return {
        "long_side": (blend_probs((0.62, model), (0.23, prior), (0.15, v188)), LONGSIDE_POINTS),
        "half_depth": (blend_probs((0.50, model), (0.40, prior), (0.10, v188)), SHORTMID_POINTS),
        "action_conditioned": (blend_probs((0.72, prior), (0.20, model), (0.08, v188)), FOCUS_POINTS),
    }


def _bank_for_spec(bundle: dict[str, Any], split: str, spec: CandidateSpec) -> SpecialistBank:
    base = bundle[f"base_{split}_point"]
    sources = dict(_specialist_sources(bundle, split))
    if spec.family == "actioncond_highmargin":
        sources["action_conditioned"] = _specialist_sources(bundle, split)["action_conditioned"]
    return specialist_vote_bank(base, sources)


def _support_for_bank(bundle: dict[str, Any], split: str, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_key = f"{'test' if split == 'test' else 'train'}_df"
    return lookup_support(bundle[frame_key], bundle["prior_table"], bundle["prior_key_cols"], candidate)


def apply_selection(base_point: np.ndarray, candidate_point: np.ndarray, selected: np.ndarray) -> np.ndarray:
    pred = np.asarray(base_point, dtype=int).copy()
    mask = np.asarray(selected, dtype=bool)
    pred[mask] = np.asarray(candidate_point, dtype=int)[mask]
    return pred


def _candidate_score(bank: SpecialistBank, target_support: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    if spec.family == "modelbank_agree":
        return np.asarray(bank.score, dtype=float)
    support_term = np.log1p(np.asarray(target_support, dtype=float)) * 0.008
    agreement_term = np.maximum(np.asarray(bank.agree_count, dtype=float) - 1.0, 0.0) * spec.agreement_bonus
    return np.asarray(bank.score, dtype=float) + agreement_term + support_term


def _family_gate(bank: SpecialistBank, base_point: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    base = np.asarray(base_point, dtype=int)
    if spec.family == "modelbank_agree":
        return np.ones(len(base), dtype=bool)
    if spec.family == "long_half_combo":
        return (
            np.isin(base, sorted(FOCUS_POINTS))
            & np.isin(bank.candidate, sorted(FOCUS_POINTS))
            & np.array([fam in {"long_side", "half_depth", "action_conditioned"} for fam in bank.best_family], dtype=bool)
        )
    if spec.family == "actioncond_highmargin":
        action_cand = bank.family_candidates["action_conditioned"]
        action_margin = bank.family_margins["action_conditioned"]
        return (bank.candidate == action_cand) & np.isfinite(action_margin) & (action_margin > 0)
    raise ValueError(f"unknown candidate family: {spec.family}")


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = validate_submission_frame(out.loc[:, EXPECTED_COLUMNS])
    path = ensure_local_output_path(OUTDIR / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path.relative_to(ROOT))


def changed_rows_frame(
    spec: CandidateSpec,
    selected: np.ndarray,
    pred: np.ndarray,
    bank: SpecialistBank,
    score: np.ndarray,
    support: np.ndarray,
    target_support: np.ndarray,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    idx = np.where(selected)[0]
    if len(idx) == 0:
        return pd.DataFrame()
    anchor = bundle["anchor"]
    base = bundle["base_test_point"]
    rows = pd.DataFrame(
        {
            "candidate": spec.name,
            "row_id": idx,
            "rally_uid": anchor.iloc[idx]["rally_uid"].astype(int).to_numpy(),
            "actionId": anchor.iloc[idx]["actionId"].astype(int).to_numpy(),
            "old_pointId": base[idx],
            "new_pointId": pred[idx],
            "change": [f"{int(old)}->{int(new)}" for old, new in zip(base[idx], pred[idx])],
            "score": score[idx],
            "raw_bank_score": bank.score[idx],
            "margin": bank.margin[idx],
            "agree_count": bank.agree_count[idx],
            "best_family": bank.best_family[idx],
            "prior_slice_support": support[idx],
            "prior_target_support": target_support[idx],
            "long_side_margin": bank.family_margins["long_side"][idx],
            "half_depth_margin": bank.family_margins["half_depth"][idx],
            "action_conditioned_margin": bank.family_margins["action_conditioned"][idx],
            "lag0_actionId": bundle["test_df"].iloc[idx]["lag0_actionId"].to_numpy()
            if "lag0_actionId" in bundle["test_df"]
            else np.nan,
            "lag0_pointId": bundle["test_df"].iloc[idx]["lag0_pointId"].to_numpy()
            if "lag0_pointId" in bundle["test_df"]
            else np.nan,
            "prefix_len": bundle["test_df"].iloc[idx]["prefix_len"].to_numpy()
            if "prefix_len" in bundle["test_df"]
            else np.nan,
            "serverGetPoint": anchor.iloc[idx]["serverGetPoint"].to_numpy(dtype=float),
        }
    )
    return rows.sort_values(["score", "agree_count", "margin"], ascending=[False, False, False]).reset_index(drop=True)


def load_v316_best_delta() -> float:
    if not V316_REPORT_PATH.exists():
        return 0.0
    report = json.loads(V316_REPORT_PATH.read_text(encoding="utf-8"))
    best = report.get("best_candidate", {})
    return float(best.get("local_delta_vs_v306_point_anchor", 0.0) or 0.0)


def evaluate_candidate(spec: CandidateSpec, bundle: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    oof_bank = _bank_for_spec(bundle, "oof", spec)
    test_bank = _bank_for_spec(bundle, "test", spec)
    oof_support, oof_target_support = _support_for_bank(bundle, "oof", oof_bank.candidate)
    test_support, test_target_support = _support_for_bank(bundle, "test", test_bank.candidate)
    oof_score = _candidate_score(oof_bank, oof_target_support, spec)
    test_score = _candidate_score(test_bank, test_target_support, spec)
    cap = spec.budget / len(bundle["base_test_point"])
    oof_budget = int(np.floor(len(bundle["base_oof_point"]) * cap))
    allowed_pairs = _same_group_pairs()
    oof_selected = select_modelbank_replacements(
        bundle["base_oof_point"],
        oof_bank.candidate,
        oof_score,
        agree_count=oof_bank.agree_count,
        margin=oof_bank.margin,
        support=oof_target_support,
        budget=oof_budget,
        high_margin=spec.high_margin,
        min_support=spec.min_support,
        allowed_pairs=allowed_pairs,
        gate=_family_gate(oof_bank, bundle["base_oof_point"], spec),
    )
    test_selected = select_modelbank_replacements(
        bundle["base_test_point"],
        test_bank.candidate,
        test_score,
        agree_count=test_bank.agree_count,
        margin=test_bank.margin,
        support=test_target_support,
        budget=spec.budget,
        high_margin=spec.high_margin,
        min_support=spec.min_support,
        allowed_pairs=allowed_pairs,
        gate=_family_gate(test_bank, bundle["base_test_point"], spec),
    )
    oof_pred = apply_selection(bundle["base_oof_point"], oof_bank.candidate, oof_selected)
    test_pred = apply_selection(bundle["base_test_point"], test_bank.candidate, test_selected)
    counts = count_point0_changes(bundle["base_test_point"], test_pred)
    if counts["point0_additions"] or counts["point0_removals"]:
        raise ValueError(f"{spec.name} attempted point0 changes: {counts}")

    score = float(f1_score(bundle["y"], oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    delta = score - float(bundle["base_score"])
    changed = int(test_selected.sum())
    path = write_submission(bundle["anchor"], test_pred, spec.submission)
    changed_rows = changed_rows_frame(
        spec,
        test_selected,
        test_pred,
        test_bank,
        test_score,
        test_support,
        test_target_support,
        bundle,
    )
    agreement_rows = int((test_bank.agree_count[test_selected] >= 2).sum()) if changed else 0
    high_margin_rows = int(
        ((test_bank.margin[test_selected] >= spec.high_margin) & (test_target_support[test_selected] >= spec.min_support)).sum()
    ) if changed else 0
    record = {
        "candidate": spec.name,
        "submission": spec.submission,
        "path": path,
        "family": spec.family,
        "budget": spec.budget,
        "oof_budget": oof_budget,
        "point_macro_f1": score,
        "local_delta_vs_v306_point_anchor": delta,
        "local_delta_vs_v316_best": delta - load_v316_best_delta(),
        "base_point_macro_f1": bundle["base_score"],
        "test_changed_rows": changed,
        "oof_changed_rows": int(oof_selected.sum()),
        "test_churn": changed / len(bundle["base_test_point"]),
        "point0_additions": counts["point0_additions"],
        "point0_removals": counts["point0_removals"],
        "agreement_rows": agreement_rows,
        "high_margin_rows": high_margin_rows,
        "changed_456_rows": int(np.isin(bundle["base_test_point"][test_selected], sorted(SHORTMID_POINTS)).sum()) if changed else 0,
        "changed_789_rows": int(np.isin(bundle["base_test_point"][test_selected], sorted(LONGSIDE_POINTS)).sum()) if changed else 0,
        "score_mean_changed": float(test_score[test_selected].mean()) if changed else 0.0,
        "margin_mean_changed": float(test_bank.margin[test_selected].mean()) if changed else 0.0,
        "agree_count_mean_changed": float(test_bank.agree_count[test_selected].mean()) if changed else 0.0,
        "prior_slice_support_mean_changed": float(test_support[test_selected].mean()) if changed else 0.0,
        "prior_target_support_mean_changed": float(test_target_support[test_selected].mean()) if changed else 0.0,
        "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
        "decision": decision_label(delta, changed, counts),
        "risk_tier": "low" if changed <= 18 and delta >= 0 else "medium" if changed <= 36 else "high",
        "packaging": "V173 action + V300 server from V306 public anchor",
    }
    return record, changed_rows


def _v322_decision(search: pd.DataFrame) -> tuple[str, str]:
    review = search[
        search["decision"].eq("REVIEW")
        & (search["point0_additions"].eq(0))
        & (search["point0_removals"].eq(0))
        & (
            search["local_delta_vs_v316_best"].ge(0.0005)
            | (
                search["agreement_rows"].ge(search["test_changed_rows"] * 0.75)
                & search["prior_target_support_mean_changed"].ge(8.0)
                & search["test_changed_rows"].le(18)
            )
        )
    ]
    if not review.empty:
        return "HAS_REVIEW_CANDIDATE", "REVIEW"
    return "DIAGNOSTIC_ONLY", "DO_NOT_UPLOAD"


def write_reports(search: pd.DataFrame, changed_rows: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    search = search.sort_values(
        ["local_delta_vs_v306_point_anchor", "agreement_rows", "test_changed_rows"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    search.to_csv(SEARCH_PATH, index=False)
    if changed_rows.empty:
        changed_rows = pd.DataFrame(
            columns=["candidate", "row_id", "rally_uid", "actionId", "old_pointId", "new_pointId", "change", "score"]
        )
    changed_rows.to_csv(CHANGED_ROWS_PATH, index=False)
    verdict, recommendation = _v322_decision(search)
    best = search.iloc[0].to_dict() if not search.empty else {}
    report = {
        "version": "V322",
        "verdict": verdict,
        "upload_recommendation": recommendation,
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "policy": {
            "base_point_anchor": "v306_point0_addition_probe/submission_v306_p0_cap0p01__v173action_v300server.csv",
            "fixed_action_server": "V173 action + V300 server",
            "no_point0_additions": True,
            "no_point0_removals": True,
            "no_upload_copy": True,
            "no_ttmatch": True,
            "no_old_server": True,
        },
        "base_point_macro_f1": bundle["base_score"],
        "v316_best_delta": load_v316_best_delta(),
        "best_candidate": best,
        "top_candidates": search.head(4).to_dict(orient="records"),
        "review_candidates": search[search["decision"].eq("REVIEW")].to_dict(orient="records"),
        "features_count": bundle["features_count"],
        "prior_key_cols": bundle["prior_key_cols"],
        "notes": [
            "All candidates start from the V306 public point anchor.",
            "Rows are eligible only for nonzero-to-nonzero 4/5/6 or 7/8/9 same-group changes.",
            "Selection requires at least two specialist families agreeing or a high-margin action/model-bank vote with target support.",
            "No files are written to upload_candidates_20260519 or submissions/selected.",
        ],
    }
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    lines = [
        "# V322 Nonterminal Point Model Bank",
        "",
        f"- Verdict: `{verdict}`",
        f"- Upload recommendation: `{recommendation}`",
        f"- Base point Macro-F1: `{float(bundle['base_score']):.6f}`",
        f"- V316 best delta: `{float(report['v316_best_delta']):+.6f}`",
        f"- Best candidate: `{best.get('candidate', 'none')}`",
        f"- Best local delta: `{float(best.get('local_delta_vs_v306_point_anchor', 0.0)):+.6f}`",
        f"- Best delta vs V316: `{float(best.get('local_delta_vs_v316_best', 0.0)):+.6f}`",
        "",
        "## Candidates",
        "",
        "| candidate | delta | vs V316 | rows | agree rows | high margin | 4/5/6 | 7/8/9 | point0 +/- | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in search.to_dict(orient="records"):
        lines.append(
            f"| `{row['candidate']}` | {float(row['local_delta_vs_v306_point_anchor']):+.6f} | "
            f"{float(row['local_delta_vs_v316_best']):+.6f} | {int(row['test_changed_rows'])} | "
            f"{int(row['agreement_rows'])} | {int(row['high_margin_rows'])} | "
            f"{int(row['changed_456_rows'])} | {int(row['changed_789_rows'])} | "
            f"{int(row['point0_additions'])}/{int(row['point0_removals'])} | `{row['decision']}` |"
        )
    lines.extend(["", f"Search CSV: `{SEARCH_PATH.relative_to(ROOT).as_posix()}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bundle = build_v316_bundle()
    records: list[dict[str, Any]] = []
    changed: list[pd.DataFrame] = []
    for spec in CANDIDATES:
        record, rows = evaluate_candidate(spec, bundle)
        records.append(record)
        if not rows.empty:
            changed.append(rows)
    search = pd.DataFrame(records)
    changed_rows = pd.concat(changed, ignore_index=True) if changed else pd.DataFrame()
    return write_reports(search, changed_rows, bundle)


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": report["outdir"],
                "verdict": report["verdict"],
                "best": best.get("candidate"),
                "best_delta": best.get("local_delta_vs_v306_point_anchor"),
                "best_delta_vs_v316": best.get("local_delta_vs_v316_best"),
                "best_rows": best.get("test_changed_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
