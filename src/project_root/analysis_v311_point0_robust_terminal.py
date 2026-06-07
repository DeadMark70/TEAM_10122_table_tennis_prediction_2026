"""V311 robust point0 terminal classifier search.

This research pass keeps the V305 literal point base, consumes the V306/V307
point0 references, and writes only local V311 artifacts. Every packaged
candidate changes point labels only: V173 action and V300 server stay fixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_lgbm import POINT_CLASSES
from analysis_v261_action_conditioned_point_residual import (
    EXPECTED_COLUMNS,
    add_foldsafe_proxy_columns,
    build_frames,
    distribution,
    normalize_rows_safe,
    numeric_feature_columns,
    point_depth,
    point_side,
    train_oof_prob,
)
from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column
from analysis_v306_point0_addition_probe import (
    ARTIFACT_DIR,
    V300_SUBMISSION,
    cap_token,
    load_artifacts,
    load_submission,
)


OUTDIR = Path("v311_point0_robust_terminal")
V306_SEARCH = Path("v306_point0_addition_probe/v306_point0_search.csv")
V306_REPORT = Path("v306_point0_addition_probe/v306_report.json")
V307_SEARCH = Path("v307_point0_dose_extension/v307_point0_dose_search.csv")
V307_REPORT = Path("v307_point0_dose_extension/v307_report.json")
V306_PUBLIC_BEST_PL = 0.3577905
ROW_BUDGETS = [18, 20, 22, 24, 27, 30, 36]


@dataclass(frozen=True)
class SchemeSpec:
    name: str
    description: str
    gate_kind: str


SCHEMES = [
    SchemeSpec("v188_margin", "V188 r186 p0 margin only", "all"),
    SchemeSpec("v261_terminal_margin", "V261-like tabular terminal/model p0 margin", "all"),
    SchemeSpec("longside_margin", "long-side-only p0 margin", "longside"),
    SchemeSpec("agreement_model_prior", "agreement score between model p0 and prior p0", "agreement"),
]


def select_point0_rows(
    base_point: np.ndarray,
    candidate_point: np.ndarray,
    score: np.ndarray,
    budget: int,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    """Select top-scored rows where a nonzero base point is changed to point0."""
    base = np.asarray(base_point, dtype=int)
    cand = np.asarray(candidate_point, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    if not (len(base) == len(cand) == len(score_arr)):
        raise ValueError("base_point, candidate_point, and score must have the same length")
    if gate is None:
        gate_arr = np.ones(len(base), dtype=bool)
    else:
        gate_arr = np.asarray(gate, dtype=bool)
        if len(gate_arr) != len(base):
            raise ValueError("gate must have the same length as base_point")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    eligible = (base != 0) & (cand == 0) & gate_arr & np.isfinite(score_arr) & (score_arr > 0)
    selected = np.zeros(len(base), dtype=bool)
    if budget == 0 or not eligible.any():
        return selected
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score_arr[idx], kind="mergesort")]
    selected[order[: min(int(budget), len(order))]] = True
    return selected


def decision_label(
    local_delta: float,
    changed_rows: int,
    v306_delta: float,
    v307_budget24_delta: float,
) -> str:
    """Apply the V311 strict review gates from the task brief."""
    delta = float(local_delta)
    rows = int(changed_rows)
    if delta > float(v306_delta) and rows <= 24:
        return "REVIEW_SAFE"
    if delta > float(v307_budget24_delta) and rows <= 36:
        return "REVIEW_EXPLORE"
    return "DIAGNOSTIC"


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None
    return value


def load_reference_outputs() -> dict[str, Any]:
    required = [V306_SEARCH, V306_REPORT, V307_SEARCH, V307_REPORT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V306/V307 reference outputs: " + ", ".join(missing))

    v306_search = pd.read_csv(V306_SEARCH)
    v307_search = pd.read_csv(V307_SEARCH)
    v306_report = json.loads(V306_REPORT.read_text(encoding="utf-8"))
    v307_report = json.loads(V307_REPORT.read_text(encoding="utf-8"))
    v306_best_delta = float(v306_report.get("best_candidate", {}).get("literal_oof_delta", np.nan))
    if not np.isfinite(v306_best_delta):
        row = v306_search[v306_search["candidate"].eq("v306_p0_cap0p01")]
        if row.empty:
            raise ValueError("Could not resolve V306 reference delta")
        v306_best_delta = float(row.iloc[0]["literal_oof_delta"])
    budget24 = v307_search[v307_search["candidate"].eq("v307_p0_budget24")]
    if budget24.empty:
        raise ValueError("Could not resolve V307 budget24 reference delta")
    cap0p02 = v307_search[v307_search["candidate"].eq("v307_p0_cap0p02")]
    return {
        "v306_search": v306_search,
        "v307_search": v307_search,
        "v306_report": v306_report,
        "v307_report": v307_report,
        "v306_delta": v306_best_delta,
        "v307_budget24_delta": float(budget24.iloc[0]["literal_oof_delta"]),
        "v307_cap0p02_delta": float(cap0p02.iloc[0]["literal_oof_delta"]) if not cap0p02.empty else None,
    }


def p0_margin(prob: np.ndarray, base_point: np.ndarray) -> np.ndarray:
    p = normalize_rows_safe(prob)
    base = np.asarray(base_point, dtype=int)
    if p.ndim != 2 or len(p) != len(base):
        raise ValueError("prob and base_point row counts must match")
    return p[:, 0] - p[np.arange(len(base)), np.clip(base, 0, p.shape[1] - 1)]


def terminal_margin(terminal_positive_prob: np.ndarray) -> np.ndarray:
    p = np.asarray(terminal_positive_prob, dtype=float)
    return p - (1.0 - p)


def add_base_point_features(frame: pd.DataFrame, base_point: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    base = np.asarray(base_point, dtype=int)
    out["v311_base_point"] = base
    out["v311_base_depth"] = [point_depth(x) for x in base]
    out["v311_base_side"] = [point_side(x) for x in base]
    out["v311_base_is_long"] = np.isin(base, [7, 8, 9]).astype(int)
    return out


def foldsafe_terminal_slice_prior(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    terminal_target: np.ndarray,
    key_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(terminal_target, dtype=int)
    global_prob = float((target.sum() + 1.0) / (len(target) + 2.0))
    oof = np.full(len(train_df), global_prob, dtype=float)

    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(int(fold)).to_numpy()
        fit = train_df.loc[~valid, key_cols].copy()
        fit["target"] = target[~valid]
        table = fit.groupby(key_cols)["target"].agg(["sum", "count"]).reset_index()
        table["v311_terminal_slice_prior"] = (table["sum"] + 1.0) / (table["count"] + 2.0)
        merged = train_df.loc[valid, key_cols].merge(
            table[key_cols + ["v311_terminal_slice_prior"]],
            on=key_cols,
            how="left",
        )
        vals = merged["v311_terminal_slice_prior"].to_numpy(dtype=float).copy()
        vals[~np.isfinite(vals)] = global_prob
        oof[valid] = vals

    fit_all = train_df.loc[:, key_cols].copy()
    fit_all["target"] = target
    table = fit_all.groupby(key_cols)["target"].agg(["sum", "count"]).reset_index()
    table["v311_terminal_slice_prior"] = (table["sum"] + 1.0) / (table["count"] + 2.0)
    merged_test = test_df.loc[:, key_cols].merge(
        table[key_cols + ["v311_terminal_slice_prior"]],
        on=key_cols,
        how="left",
    )
    test_vals = merged_test["v311_terminal_slice_prior"].to_numpy(dtype=float).copy()
    test_vals[~np.isfinite(test_vals)] = global_prob
    return oof, test_vals


def build_model_bundle(artifacts: dict[str, object]) -> dict[str, Any]:
    train_df, test_df, _ = build_frames()
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns(train_df, test_df)
    train_df = align_train_to_literal_meta(train_df, artifacts["meta"])
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0

    cap5_oof = artifacts["cap5_oof"]
    cap5_test = artifacts["cap5_test"]
    oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    test_base = cap5_test[point_column(cap5_test)].astype(int).to_numpy()
    y = train_df["next_pointId"].astype(int).to_numpy()
    if len(oof_base) != len(y):
        raise ValueError(f"OOF base length {len(oof_base)} != y length {len(y)}")

    train_df = add_base_point_features(train_df, oof_base)
    test_df = add_base_point_features(test_df, test_base)
    key_cols = [
        c
        for c in [
            "lag0_action_family",
            "lag0_point_depth",
            "lag0_point_side",
            "v261_action_family",
            "v311_base_depth",
            "v311_base_side",
            "v311_base_is_long",
        ]
        if c in train_df.columns and c in test_df.columns
    ]
    terminal_target = y == 0
    slice_oof, slice_test = foldsafe_terminal_slice_prior(train_df, test_df, terminal_target.astype(int), key_cols)
    train_df["v311_terminal_slice_prior"] = slice_oof
    test_df["v311_terminal_slice_prior"] = slice_test

    features = [c for c in numeric_feature_columns(train_df, include_proxy=True) if c in test_df]
    point_oof, point_test, point_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        features,
        seed=31110,
        n_estimators=300,
        min_samples_leaf=4,
    )
    terminal_oof, terminal_test, terminal_folds = train_oof_prob(
        train_df,
        test_df,
        terminal_target.astype(int),
        [0, 1],
        features,
        seed=31140,
        n_estimators=340,
        min_samples_leaf=5,
    )
    return {
        "train_df": train_df,
        "test_df": test_df,
        "y": y,
        "oof_base": oof_base,
        "test_base": test_base,
        "point_oof": point_oof,
        "point_test": point_test,
        "terminal_oof_p0": terminal_oof[:, 1],
        "terminal_test_p0": terminal_test[:, 1],
        "folds": proxy_folds
        + [{"stage": "v311_point_model", **r} for r in point_folds]
        + [{"stage": "v311_terminal_model", **r} for r in terminal_folds],
        "features": features,
        "slice_key_cols": key_cols,
    }


def score_arrays(
    base_point: np.ndarray,
    v188_prob: np.ndarray,
    point_prob: np.ndarray,
    terminal_p0: np.ndarray,
) -> dict[str, np.ndarray]:
    prior_margin = p0_margin(v188_prob, base_point)
    model_margin = p0_margin(point_prob, base_point)
    term_margin = terminal_margin(terminal_p0)
    return {
        "v188_margin": prior_margin,
        "v261_terminal_margin": 0.55 * model_margin + 0.45 * term_margin,
        "longside_margin": model_margin,
        "agreement_model_prior": np.minimum(model_margin, prior_margin) + 0.20 * term_margin,
        "model_p0_margin": model_margin,
        "terminal_p0_margin": term_margin,
        "v188_p0_margin": prior_margin,
    }


def scheme_gate(spec: SchemeSpec, base_point: np.ndarray, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = np.asarray(base_point, dtype=int)
    if spec.gate_kind == "longside":
        return np.isin(base, [7, 8, 9])
    if spec.gate_kind == "agreement":
        return (scores["model_p0_margin"] > 0) & (scores["v188_p0_margin"] > 0) & (scores["terminal_p0_margin"] > 0)
    return np.ones(len(base), dtype=bool)


def apply_selected(base_point: np.ndarray, selected: np.ndarray) -> np.ndarray:
    pred = np.asarray(base_point, dtype=int).copy()
    pred[np.asarray(selected, dtype=bool)] = 0
    return pred


def write_submission(anchor: pd.DataFrame, point: np.ndarray, name: str) -> str:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[EXPECTED_COLUMNS]
    if len(out) != 1845:
        raise ValueError(f"{name} has {len(out)} rows")
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def changed_rows_frame(
    selected: np.ndarray,
    test_df: pd.DataFrame,
    anchor: pd.DataFrame,
    base_point: np.ndarray,
    scores: dict[str, np.ndarray],
    candidate: str,
) -> pd.DataFrame:
    idx = np.where(selected)[0]
    rows = pd.DataFrame(
        {
            "row_id": idx,
            "rally_uid": test_df.iloc[idx]["rally_uid"].astype(int).to_numpy(),
            "candidate": candidate,
            "base_point": np.asarray(base_point, dtype=int)[idx],
            "candidate_point": np.zeros(len(idx), dtype=int),
            "model_p0_margin": scores["model_p0_margin"][idx],
            "terminal_p0_margin": scores["terminal_p0_margin"][idx],
            "v188_p0_margin": scores["v188_p0_margin"][idx],
            "agreement_score": scores["agreement_model_prior"][idx],
            "v300_actionId": anchor.iloc[idx]["actionId"].astype(int).to_numpy(),
            "v300_serverGetPoint": anchor.iloc[idx]["serverGetPoint"].to_numpy(dtype=float),
            "prefix_len": test_df.iloc[idx]["prefix_len"].to_numpy() if "prefix_len" in test_df.columns else np.nan,
            "lag0_actionId": test_df.iloc[idx]["lag0_actionId"].to_numpy() if "lag0_actionId" in test_df.columns else np.nan,
            "lag0_pointId": test_df.iloc[idx]["lag0_pointId"].to_numpy() if "lag0_pointId" in test_df.columns else np.nan,
            "change": [f"{int(x)}->0" for x in np.asarray(base_point, dtype=int)[idx]],
        }
    )
    return rows.sort_values(["model_p0_margin", "terminal_p0_margin"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    if not ARTIFACT_DIR.exists():
        raise FileNotFoundError("Run analysis_v305_export_literal_v188_point_artifact.py first.")
    refs = load_reference_outputs()
    artifacts = load_artifacts()
    bundle = build_model_bundle(artifacts)
    anchor = load_submission(V300_SUBMISSION)
    current_v300_point = anchor["pointId"].astype(int).to_numpy()

    oof_base = bundle["oof_base"]
    test_base = bundle["test_base"]
    y = bundle["y"]
    base_score = float(f1_score(y, oof_base, labels=POINT_CLASSES, average="macro", zero_division=0))

    oof_scores = score_arrays(oof_base, artifacts["v188_oof_prob"], bundle["point_oof"], bundle["terminal_oof_p0"])
    test_scores = score_arrays(test_base, artifacts["v188_test_prob"], bundle["point_test"], bundle["terminal_test_p0"])
    candidate_zero_oof = np.zeros(len(oof_base), dtype=int)
    candidate_zero_test = np.zeros(len(test_base), dtype=int)

    records: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    best_selected: np.ndarray | None = None
    best_candidate_name = ""

    records.append(
        {
            "candidate": "v188_literal_cap5_base",
            "scheme": "baseline",
            "budget": 0,
            "oof_budget": 0,
            "point_macro_f1": base_score,
            "literal_oof_delta": 0.0,
            "test_changed_rows": 0,
            "oof_changed_rows": 0,
            "decision": "BASELINE",
            "server_source": "v300",
            "packaging": "V173 action + V300 server",
            "test_churn_vs_v188_cap5": 0.0,
            "test_churn_vs_current_v300": float(np.mean(test_base != current_v300_point)),
        }
    )

    for spec in SCHEMES:
        oof_gate = scheme_gate(spec, oof_base, oof_scores)
        test_gate = scheme_gate(spec, test_base, test_scores)
        for budget in ROW_BUDGETS:
            cap = budget / len(test_base)
            oof_budget = int(np.floor(len(oof_base) * cap))
            oof_selected = select_point0_rows(oof_base, candidate_zero_oof, oof_scores[spec.name], oof_budget, gate=oof_gate)
            test_selected = select_point0_rows(test_base, candidate_zero_test, test_scores[spec.name], budget, gate=test_gate)
            oof_pred = apply_selected(oof_base, oof_selected)
            test_pred = apply_selected(test_base, test_selected)
            score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
            delta = score - base_score
            changed_rows = int(test_selected.sum())
            decision = decision_label(delta, changed_rows, refs["v306_delta"], refs["v307_budget24_delta"])
            candidate = f"v311_{spec.name}_budget{budget}"
            submission = f"submission_{candidate}__v173action_v300server.csv"
            path = write_submission(anchor, test_pred, submission)
            submissions.append({"candidate": candidate, "submission": submission, "path": path})
            rec = {
                "candidate": candidate,
                "scheme": spec.name,
                "scheme_description": spec.description,
                "budget": budget,
                "oof_budget": oof_budget,
                "cap": cap,
                "point_macro_f1": score,
                "literal_oof_delta": delta,
                "test_changed_rows": changed_rows,
                "oof_changed_rows": int(oof_selected.sum()),
                "point0_additions": changed_rows,
                "point0_removals": 0,
                "test_churn_vs_v188_cap5": float(np.mean(test_selected)),
                "test_churn_vs_current_v300": float(np.mean(test_pred != current_v300_point)),
                "model_p0_margin_min_changed": float(test_scores["model_p0_margin"][test_selected].min()) if changed_rows else 0.0,
                "model_p0_margin_mean_changed": float(test_scores["model_p0_margin"][test_selected].mean()) if changed_rows else 0.0,
                "terminal_p0_margin_mean_changed": float(test_scores["terminal_p0_margin"][test_selected].mean()) if changed_rows else 0.0,
                "v188_p0_margin_mean_changed": float(test_scores["v188_p0_margin"][test_selected].mean()) if changed_rows else 0.0,
                "agreement_score_mean_changed": float(test_scores["agreement_model_prior"][test_selected].mean()) if changed_rows else 0.0,
                "longside_changed_rows": int(np.isin(test_base[test_selected], [7, 8, 9]).sum()) if changed_rows else 0,
                "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                "decision": decision,
                "server_source": "v300",
                "packaging": "V173 action + V300 server",
                "submission": submission,
                "path": path,
            }
            records.append(rec)

    search = pd.DataFrame(records)
    candidate_rows = search[search["candidate"].astype(str).str.startswith("v311_")].copy()
    candidate_rows = candidate_rows.sort_values(["literal_oof_delta", "test_changed_rows"], ascending=[False, True])

    auto_record: dict[str, Any] | None = None
    if not candidate_rows.empty and float(candidate_rows.iloc[0]["literal_oof_delta"]) > 0:
        source = candidate_rows.iloc[0]
        source_spec = next(s for s in SCHEMES if s.name == source["scheme"])
        test_selected = select_point0_rows(
            test_base,
            candidate_zero_test,
            test_scores[source_spec.name],
            int(source["budget"]),
            gate=scheme_gate(source_spec, test_base, test_scores),
        )
        test_pred = apply_selected(test_base, test_selected)
        auto_name = "v311_auto_calibrated"
        auto_submission = "submission_v311_auto_calibrated__v173action_v300server.csv"
        auto_path = write_submission(anchor, test_pred, auto_submission)
        auto_record = source.to_dict()
        auto_record.update(
            {
                "candidate": auto_name,
                "scheme": "auto_calibrated",
                "scheme_description": f"Auto-selected from {source['candidate']}",
                "submission": auto_submission,
                "path": auto_path,
            }
        )
        submissions.append({"candidate": auto_name, "submission": auto_submission, "path": auto_path})
        search = pd.concat([search, pd.DataFrame([auto_record])], ignore_index=True)

    search = search.sort_values(["literal_oof_delta", "test_changed_rows"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v311_point0_search.csv", index=False)
    top_candidates = search[search["candidate"].astype(str).str.startswith("v311_")].head(5).to_dict(orient="records")
    if top_candidates:
        top = top_candidates[0]
        top_spec_name = top["scheme"] if top["scheme"] != "auto_calibrated" else auto_record["scheme_description"].split()[-1]
        selected_row = search[search["candidate"].eq(top["candidate"])].iloc[0]
        if selected_row["scheme"] == "auto_calibrated" and auto_record is not None:
            source_candidate = auto_record["scheme_description"].replace("Auto-selected from ", "")
            selected_row = search[search["candidate"].eq(source_candidate)].iloc[0]
        top_spec = next(s for s in SCHEMES if s.name == selected_row["scheme"])
        best_selected = select_point0_rows(
            test_base,
            candidate_zero_test,
            test_scores[top_spec.name],
            int(selected_row["budget"]),
            gate=scheme_gate(top_spec, test_base, test_scores),
        )
        best_candidate_name = str(top["candidate"])
        changed_rows_frame(best_selected, bundle["test_df"], anchor, test_base, test_scores, best_candidate_name).to_csv(
            OUTDIR / "v311_changed_rows.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(OUTDIR / "v311_changed_rows.csv", index=False)

    review = search[search["decision"].isin(["REVIEW_SAFE", "REVIEW_EXPLORE"])]
    verdict = "HAS_REVIEW_CANDIDATE" if not review.empty else "DIAGNOSTIC_ONLY"
    best = top_candidates[0] if top_candidates else {}
    report = {
        "verdict": verdict,
        "decision": best.get("decision", "DIAGNOSTIC"),
        "current_clean_public_best": "V306 p0 cap0p01",
        "current_clean_public_best_pl": V306_PUBLIC_BEST_PL,
        "packaging": "V173 action with V300 server only",
        "v188_literal_cap5_point_macro_f1": base_score,
        "v306_reference_delta": refs["v306_delta"],
        "v307_budget24_reference_delta": refs["v307_budget24_delta"],
        "v307_cap0p02_reference_delta": refs["v307_cap0p02_delta"],
        "best_candidate": best,
        "top5_candidates": top_candidates,
        "review_candidates": review.head(10).to_dict(orient="records"),
        "auto_calibrated_candidate": auto_record,
        "submissions": submissions,
        "folds": bundle["folds"],
        "slice_key_cols": bundle["slice_key_cols"],
        "features_count": len(bundle["features"]),
        "notes": [
            "Consumes V305 literal artifacts plus V306/V307 search/report outputs.",
            "Eligible rows always require base point != 0 and candidate point = 0.",
            "Scoring schemes include V188 p0 margin, V261-like terminal/model margin, long-side-only margin, and model/prior agreement.",
            "Budgets evaluated: 18, 20, 22, 24, 27, 30, 36.",
            "Decision gates are strict: REVIEW_SAFE if local delta > V306 delta and rows <= 24; REVIEW_EXPLORE if local delta > V307 budget24 and rows <= 36.",
            "No files are copied to upload_candidates_20260519 or submissions/selected.",
        ],
    }
    (OUTDIR / "v311_report.json").write_text(json.dumps(sanitize_json(report), indent=2), encoding="utf-8")

    lines = [
        "# V311 Point0 Robust Terminal Classifier",
        "",
        f"- Verdict: `{verdict}`",
        f"- Best candidate: `{best.get('candidate', 'none')}`",
        f"- Best decision: `{best.get('decision', 'none')}`",
        f"- Best local delta: `{float(best.get('literal_oof_delta', 0.0)):.6f}`",
        f"- Best changed rows: `{int(best.get('test_changed_rows', 0))}`",
        f"- Stronger than V307 budget24: `{float(best.get('literal_oof_delta', 0.0)) > refs['v307_budget24_delta']}`",
        f"- Stronger than V307 cap0p02: `{refs['v307_cap0p02_delta'] is not None and float(best.get('literal_oof_delta', 0.0)) > float(refs['v307_cap0p02_delta'])}`",
        "",
        "## Top 5 Candidates",
        "",
    ]
    if top_candidates:
        lines.extend(
            [
                f"- `{r['candidate']}` scheme `{r['scheme']}` delta `{float(r['literal_oof_delta']):.6f}` rows `{int(r['test_changed_rows'])}` decision `{r['decision']}`"
                for r in top_candidates
            ]
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {note}" for note in report["notes"]])
    (OUTDIR / "v311_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "verdict": verdict,
                "best": best.get("candidate"),
                "best_delta": best.get("literal_oof_delta"),
                "best_rows": best.get("test_changed_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
