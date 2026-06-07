"""V308 point0 row-group ablation and source-class analysis.

This script compares the V306 public-positive point0 candidate against V300
and the V305 literal base, annotates the 18 changed rows, and writes local
subgroup submissions under v308_point0_row_ablation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r179_action_physics_hierarchy import action_family, point_depth, point_side
from analysis_v194_train_test_split_distribution_audit import phase_from_prefix_len
from analysis_v261_action_conditioned_point_residual import EXPECTED_COLUMNS, distribution, normalize_rows_safe
from analysis_v305_rebuild_v261_from_literal_v188 import point_column
from analysis_v306_point0_addition_probe import (
    CURRENT_BEST_PL,
    V300_SUBMISSION,
    build_v261_literal_probabilities,
    load_artifacts,
    load_submission,
)
from baseline_lgbm import POINT_CLASSES


OUTDIR = Path("v308_point0_row_ablation")
V305_LITERAL_TEST = Path("v305_literal_v188_point_artifact/v305_v188_cap5_test_pred.csv")
V306_BEST = Path("v306_point0_addition_probe/submission_v306_p0_cap0p01__v173action_v300server.csv")
V306_SEARCH = Path("v306_point0_addition_probe/v306_point0_search.csv")
TEST_NEW = Path("test_new.csv")
DECISION_DELTA_GATE = 0.002


@dataclass(frozen=True)
class SubgroupSpec:
    name: str
    selector_type: str
    test_mask: np.ndarray
    oof_selector: Callable[[np.ndarray, np.ndarray], np.ndarray]


def detect_point0_changed_rows(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Return rows where pointId changes from a nonzero class to 0."""
    for col in EXPECTED_COLUMNS:
        if col not in base.columns or col not in candidate.columns:
            raise ValueError(f"missing required submission column: {col}")
    if len(base) != len(candidate):
        raise ValueError(f"row count mismatch: {len(base)} != {len(candidate)}")
    if not base["rally_uid"].equals(candidate["rally_uid"]):
        raise ValueError("rally_uid order mismatch")

    source = base["pointId"].astype(int).to_numpy()
    target = candidate["pointId"].astype(int).to_numpy()
    mask = (source != 0) & (target == 0) & (source != target)
    idx = np.where(mask)[0]
    return pd.DataFrame(
        {
            "row_id": idx.astype(int),
            "rally_uid": base.iloc[idx]["rally_uid"].astype(int).to_numpy(),
            "source_point": source[idx].astype(int),
            "candidate_point": target[idx].astype(int),
        }
    )


def decision_label(literal_oof_delta: float, changed_rows: int) -> str:
    if float(literal_oof_delta) >= DECISION_DELTA_GATE and int(changed_rows) <= 18:
        return "REVIEW"
    return "DIAGNOSTIC"


def _top_margin_mask(changed: pd.DataFrame, top_n: int) -> np.ndarray:
    mask = np.zeros(len(changed), dtype=bool)
    if len(changed) == 0 or top_n <= 0:
        return mask
    order = np.argsort(-changed["model_p0_margin"].astype(float).to_numpy(), kind="mergesort")
    mask[order[: min(top_n, len(order))]] = True
    return mask


def build_subgroup_masks(changed: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build requested subgroup masks over the changed-row table."""
    source = changed["source_point"].astype(int).to_numpy()
    masks: dict[str, np.ndarray] = {
        "former_7_8_9_to_0": np.isin(source, [7, 8, 9]),
        "former_8_9_to_0": np.isin(source, [8, 9]),
        "former_4_5_6_to_0": np.isin(source, [4, 5, 6]),
        "high_margin_top9": _top_margin_mask(changed, 9),
        "high_margin_top14": _top_margin_mask(changed, 14),
        "high_margin_top18": _top_margin_mask(changed, 18),
    }
    for cls in sorted(pd.unique(source)):
        masks[f"leave_source_{int(cls)}_out"] = source != int(cls)
    return masks


def _source_selector(classes: set[int], base_mask: np.ndarray | None = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def selector(base: np.ndarray, margin: np.ndarray) -> np.ndarray:
        mask = np.isin(base.astype(int), list(classes)) & (base != 0) & np.isfinite(margin) & (margin > 0)
        if base_mask is not None:
            mask &= np.asarray(base_mask, dtype=bool)
        return mask

    return selector


def _top_selector(budget: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def selector(base: np.ndarray, margin: np.ndarray) -> np.ndarray:
        eligible = (base != 0) & np.isfinite(margin) & (margin > 0)
        mask = np.zeros(len(base), dtype=bool)
        idx = np.where(eligible)[0]
        if len(idx) == 0 or budget <= 0:
            return mask
        order = idx[np.argsort(-margin[idx], kind="mergesort")]
        mask[order[: min(int(budget), len(order))]] = True
        return mask

    return selector


def _leave_source_out_selector(cls: int, base_mask: np.ndarray | None = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def selector(base: np.ndarray, margin: np.ndarray) -> np.ndarray:
        mask = (base != int(cls)) & (base != 0) & np.isfinite(margin) & (margin > 0)
        if base_mask is not None:
            mask &= np.asarray(base_mask, dtype=bool)
        return mask

    return selector


def build_subgroup_specs(
    changed: pd.DataFrame,
    oof_rows: int,
    test_rows: int,
    v306_oof_mask: np.ndarray | None = None,
) -> list[SubgroupSpec]:
    masks = build_subgroup_masks(changed)
    specs: list[SubgroupSpec] = []
    source_sets = {
        "former_7_8_9_to_0": {7, 8, 9},
        "former_8_9_to_0": {8, 9},
        "former_4_5_6_to_0": {4, 5, 6},
    }
    for name, mask in masks.items():
        if not bool(mask.any()) and name not in source_sets:
            continue
        if name in source_sets:
            selector = _source_selector(source_sets[name], v306_oof_mask)
            selector_type = "source_class_group"
        elif name.startswith("high_margin_top"):
            n = int(name.rsplit("top", 1)[1])
            selector = _top_selector(int(np.floor(oof_rows * min(n, test_rows) / test_rows)))
            selector_type = "high_margin_rank"
        elif name.startswith("leave_source_"):
            cls = int(name.removeprefix("leave_source_").removesuffix("_out"))
            selector = _leave_source_out_selector(cls, v306_oof_mask)
            selector_type = "leave_one_source_class_out"
        else:
            continue
        specs.append(SubgroupSpec(name=name, selector_type=selector_type, test_mask=mask, oof_selector=selector))
    return specs


def point_margin(base: np.ndarray, prob: np.ndarray) -> np.ndarray:
    p = normalize_rows_safe(prob)
    clipped = np.clip(np.asarray(base, dtype=int), 0, p.shape[1] - 1)
    return p[:, 0] - p[np.arange(len(clipped)), clipped]


def apply_masked_point0(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred = np.asarray(base, dtype=int).copy()
    pred[np.asarray(mask, dtype=bool)] = 0
    return pred


def label_depth(value: int | float) -> str:
    return {0: "zero", 1: "short", 2: "half", 3: "long"}.get(point_depth(int(value)), "unknown")


def label_side(value: int | float) -> str:
    return {0: "zero", 1: "forehand", 2: "middle", 3: "backhand"}.get(point_side(int(value)), "unknown")


def build_test_context(test_new: pd.DataFrame) -> pd.DataFrame:
    rows = test_new.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", as_index=False).tail(1).copy()
    rows["prefix_len"] = rows["strikeNumber"].astype(int)
    rows["phase"] = rows["prefix_len"].map(phase_from_prefix_len)
    rows["last_action_family"] = rows["actionId"].map(lambda x: action_family(int(x)))
    rows["lag0_point_depth"] = rows["pointId"].map(label_depth)
    rows["lag0_point_side"] = rows["pointId"].map(label_side)
    keep = [
        "rally_uid",
        "prefix_len",
        "phase",
        "actionId",
        "pointId",
        "last_action_family",
        "lag0_point_depth",
        "lag0_point_side",
    ]
    return rows[keep].rename(columns={"actionId": "lag0_actionId", "pointId": "lag0_pointId"})


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[EXPECTED_COLUMNS]
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def literal_base_submission(anchor: pd.DataFrame, cap5_test: pd.DataFrame) -> pd.DataFrame:
    out = anchor.copy()
    if not out["rally_uid"].equals(cap5_test["rally_uid"]):
        raise ValueError("V300 anchor and V305 literal cap5 rally_uid order mismatch")
    out["pointId"] = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    return out[EXPECTED_COLUMNS]


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    v300 = load_submission(V300_SUBMISSION)
    v306 = load_submission(V306_BEST)
    artifacts = load_artifacts()
    cap5_test = artifacts["cap5_test"]
    v305 = literal_base_submission(v300, cap5_test)
    changed = detect_point0_changed_rows(v305, v306)
    if len(changed) != 18:
        raise ValueError(f"expected 18 V306 point0 changed rows, found {len(changed)}")

    y, model_oof_prob, model_test_prob, folds = build_v261_literal_probabilities(artifacts)
    cap5_oof = artifacts["cap5_oof"]
    oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    model_oof_margin = point_margin(oof_base, model_oof_prob)
    model_test_margin = point_margin(test_base, model_test_prob)
    v306_oof_mask = _top_selector(int(np.floor(len(oof_base) * 0.01)))(oof_base, model_oof_margin)

    if not np.array_equal(v305["pointId"].astype(int).to_numpy(), test_base):
        raise ValueError("V305 local submission pointId does not match literal cap5 test base")
    if not np.array_equal(v300["rally_uid"].to_numpy(), v305["rally_uid"].to_numpy()):
        raise ValueError("V300/V305 rally_uid mismatch")

    v306_search = pd.read_csv(V306_SEARCH) if V306_SEARCH.exists() else pd.DataFrame()
    test_context = build_test_context(pd.read_csv(TEST_NEW))
    changed["model_p0_margin"] = model_test_margin[changed["row_id"].to_numpy()]
    changed["v300_point"] = v300.iloc[changed["row_id"].to_numpy()]["pointId"].astype(int).to_numpy()
    changed["v306_action"] = v306.iloc[changed["row_id"].to_numpy()]["actionId"].astype(int).to_numpy()
    changed["v300_serverGetPoint"] = v300.iloc[changed["row_id"].to_numpy()]["serverGetPoint"].astype(float).to_numpy()
    changed = changed.merge(test_context, on="rally_uid", how="left")
    changed = changed.sort_values("model_p0_margin", ascending=False).reset_index(drop=True)
    changed["model_margin_rank"] = np.arange(1, len(changed) + 1)
    changed["change"] = changed["source_point"].astype(str) + "->0"
    changed.to_csv(OUTDIR / "v308_changed_rows.csv", index=False)

    base_score = float(f1_score(y, oof_base, labels=POINT_CLASSES, average="macro", zero_division=0))
    current_best_point = v300["pointId"].astype(int).to_numpy()
    records: list[dict[str, object]] = []
    submissions: list[dict[str, object]] = []
    test_row_ids = changed["row_id"].astype(int).to_numpy()

    for spec in build_subgroup_specs(changed, len(oof_base), len(test_base), v306_oof_mask=v306_oof_mask):
        test_mask_all = np.zeros(len(test_base), dtype=bool)
        selected_rows = test_row_ids[spec.test_mask]
        test_mask_all[selected_rows] = True
        oof_mask = spec.oof_selector(oof_base, model_oof_margin)
        test_pred = apply_masked_point0(test_base, test_mask_all)
        oof_pred = apply_masked_point0(oof_base, oof_mask)
        score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
        delta = score - base_score
        changed_rows = int(test_mask_all.sum())
        name = f"submission_v308_{spec.name}__v173action_v300server.csv"
        path = write_submission(v300, test_pred, name)
        decision = decision_label(delta, changed_rows)
        row = {
            "candidate": spec.name,
            "selector_type": spec.selector_type,
            "point_macro_f1": score,
            "literal_oof_delta": delta,
            "test_changed_rows": changed_rows,
            "oof_changed_rows": int(oof_mask.sum()),
            "source_classes": json.dumps(sorted(changed.loc[spec.test_mask, "source_point"].astype(int).unique().tolist())),
            "model_p0_margin_min_changed": float(model_test_margin[test_mask_all].min()) if changed_rows else 0.0,
            "model_p0_margin_mean_changed": float(model_test_margin[test_mask_all].mean()) if changed_rows else 0.0,
            "test_churn_vs_v305_literal": float(np.mean(test_pred != test_base)),
            "test_churn_vs_current_best_v300": float(np.mean(test_pred != current_best_point)),
            "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
            "server_source": "v300",
            "action_source": "v173",
            "submission": name,
            "path": path,
            "decision": decision,
        }
        records.append(row)
        submissions.append({"candidate": spec.name, "submission": name, "path": path, "decision": decision})

    search = pd.DataFrame(records).sort_values(
        ["decision", "literal_oof_delta", "test_changed_rows"],
        ascending=[False, False, True],
    )
    search.to_csv(OUTDIR / "v308_ablation_search.csv", index=False)
    best = search.sort_values(["literal_oof_delta", "test_changed_rows"], ascending=[False, True]).head(1)
    best_dict = best.iloc[0].to_dict() if not best.empty else {}
    review = search[search["decision"].eq("REVIEW")]

    report = {
        "verdict": "HAS_REVIEW_CANDIDATE" if not review.empty else "DIAGNOSTIC_ONLY",
        "decision_rule": f"REVIEW if literal OOF delta >= {DECISION_DELTA_GATE} and changed rows <= 18; otherwise diagnostic",
        "current_clean_best": V300_SUBMISSION.name,
        "current_clean_best_pl": CURRENT_BEST_PL,
        "v306_public_positive_pl": 0.3577905,
        "v305_literal_base": V305_LITERAL_TEST.name,
        "v306_best": V306_BEST.name,
        "v306_search_best": v306_search.head(1).to_dict(orient="records"),
        "changed_rows": int(len(changed)),
        "source_class_counts": {str(k): int(v) for k, v in changed["source_point"].value_counts().sort_index().items()},
        "base_literal_point_macro_f1": base_score,
        "best_candidate": best_dict,
        "review_candidates": review.to_dict(orient="records"),
        "submissions": submissions,
        "folds": folds,
        "notes": [
            "Rows are detected by comparing V306 best to V305 literal cap0p01 base.",
            "All V308 submissions preserve V300 action/server columns and only alter pointId.",
            "Per-row model margins are recomputed with the V306/V305 residual probability path.",
            "Outputs are local-only under v308_point0_row_ablation.",
        ],
    }
    (OUTDIR / "v308_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v308_report.md").write_text(
        "# V308 Point0 Row Ablation\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Decision rule: {report['decision_rule']}\n"
        f"- Changed rows: `{len(changed)}`\n"
        f"- Source class counts: `{report['source_class_counts']}`\n"
        f"- Best subgroup: `{best_dict.get('candidate', 'none')}`\n"
        f"- Best literal OOF delta: `{float(best_dict.get('literal_oof_delta', 0.0)):.6f}`\n"
        f"- Best changed rows: `{int(best_dict.get('test_changed_rows', 0))}`\n"
        f"- Best decision: `{best_dict.get('decision', 'none')}`\n\n"
        "## Top Candidates\n\n"
        + "\n".join(
            f"- `{r['candidate']}` delta `{float(r['literal_oof_delta']):.6f}` rows `{int(r['test_changed_rows'])}` decision `{r['decision']}`"
            for r in search.head(8).to_dict(orient="records")
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "best": best_dict.get("candidate")}, indent=2))


if __name__ == "__main__":
    main()
