"""V262 table-tennis-native external sequence teacher.

Builds coarse sequence priors from the V255 canonical external corpus using
only table-tennis-native sources, then transfers them into fold-safe AICUP
action candidates centered on V173. External exact labels are never mapped to
AICUP exact actionId.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from analysis_v243_v247_action_experiment_common import (
    context_weights,
    evaluate_action,
    feature_columns,
    load_action_context,
)


ROOT = Path(".")
OUTDIR = ROOT / "v262_tabletennis_external_sequence_teacher"
CORPUS_PATH = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"
RANDOM_STATE = 262
FAMILY_COLUMNS = ["Zero", "Attack", "Control", "Defensive", "Serve"]
FAMILY_TO_ACTIONS = {
    "Zero": [0],
    "Attack": list(range(1, 8)),
    "Control": list(range(8, 12)),
    "Defensive": list(range(12, 15)),
    "Serve": list(range(15, 19)),
}
NATIVE_SOURCES = {
    "openttgames",
    "sonytabletennis",
    "TT3D",
    "TT-MatchDynamics",
    "DeepMindrobottabletennis",
}
EXCLUDED_SOURCE_MARKERS = ("coachai", "shuttleset", "ttmatch")
WEAK_ACTIONS = np.array([0, 3, 4, 5, 7, 8, 9, 12, 14], dtype=int)


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=1, keepdims=True)
    zero = sums[:, 0] <= 0.0
    if np.any(~zero):
        arr[~zero] = arr[~zero] / sums[~zero]
    if np.any(zero):
        arr[zero] = 1.0 / arr.shape[1]
    return arr


def action_family_name(action_id: int) -> str:
    action = int(action_id)
    if action == 0:
        return "Zero"
    if 1 <= action <= 7:
        return "Attack"
    if 8 <= action <= 11:
        return "Control"
    if 12 <= action <= 14:
        return "Defensive"
    if 15 <= action <= 18:
        return "Serve"
    return "Zero"


def phase_from_aicup(rows: pd.DataFrame) -> pd.Series:
    if "phase" in rows.columns:
        raw = rows["phase"].astype(str).str.lower()
        return raw.map(
            {
                "receive": "receive_like",
                "third_ball": "third_ball_like",
                "fourth_ball": "fourth_ball_like",
                "rally": "rally_like",
                "serve": "serve_like",
                "terminal": "terminal_like",
            }
        ).fillna("rally_like")
    prefix = pd.to_numeric(rows.get("prefix_len", 0), errors="coerce").fillna(0).astype(int)
    values = np.select(
        [prefix <= 1, prefix == 3, prefix == 4],
        ["receive_like", "third_ball_like", "fourth_ball_like"],
        default="rally_like",
    )
    return pd.Series(values, index=rows.index)


def lag0_family_from_aicup(rows: pd.DataFrame) -> pd.Series:
    if "lag0_family" in rows.columns:
        raw = rows["lag0_family"].astype(str)
        return raw.where(raw.isin(FAMILY_COLUMNS), "Zero")
    lag0 = pd.to_numeric(rows.get("lag0_actionId", 0), errors="coerce").fillna(0).astype(int)
    return lag0.map(action_family_name)


def depth_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 10:
        return pd.Series("missing", index=values.index)
    q1, q2 = numeric.quantile([0.33, 0.66]).tolist()
    return pd.Series(np.select([numeric <= q1, numeric <= q2], ["short", "mid"], default="deep"), index=values.index)


def load_native_corpus() -> tuple[pd.DataFrame, dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Missing V255 corpus: {CORPUS_PATH}")
    corpus = pd.read_csv(CORPUS_PATH, low_memory=False)
    source = corpus["source_dataset"].fillna("").astype(str)
    lower = source.str.lower()
    blocked = lower.apply(lambda text: any(marker in text for marker in EXCLUDED_SOURCE_MARKERS))
    native = source.isin(NATIVE_SOURCES)
    yellow_ttmd = source.eq("TT-MatchDynamics") & corpus["risk_tier"].astype(str).str.upper().eq("YELLOW")
    allowed = native & ~blocked & (~source.eq("TT-MatchDynamics") | yellow_ttmd)
    filtered = corpus.loc[allowed].copy()
    filtered = filtered[filtered["coarse_family"].isin(FAMILY_COLUMNS)].copy()
    if filtered.empty:
        raise RuntimeError("No table-tennis-native V255 rows survived filtering.")
    source_counts = filtered["source_dataset"].value_counts().sort_index().astype(int).to_dict()
    audit = {
        "raw_rows": int(len(corpus)),
        "native_rows": int(len(filtered)),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "excluded_rows": int(len(corpus) - len(filtered)),
        "coachai_rows_used": int(lower.str.contains("coachai", na=False).loc[filtered.index].sum()),
        "shuttleset_rows_used": int(lower.str.contains("shuttleset", na=False).loc[filtered.index].sum()),
        "ttmatch_rows_used": int(source.loc[filtered.index].eq("TTMATCH").sum()),
    }
    if audit["coachai_rows_used"] or audit["shuttleset_rows_used"] or audit["ttmatch_rows_used"]:
        raise RuntimeError(f"Blocked external source leaked into V262: {audit}")
    return filtered, audit


def smoothed_family_counts(group: pd.DataFrame, global_prob: pd.Series, alpha: float = 12.0) -> pd.Series:
    counts = group["coarse_family"].value_counts().reindex(FAMILY_COLUMNS, fill_value=0).astype(float)
    prob = counts + alpha * global_prob
    return prob / max(float(prob.sum()), 1e-12)


def build_external_prior_tables(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = corpus.copy()
    data["phase"] = data["phase"].fillna("rally_like").astype(str)
    data["terminal_like"] = data["terminal_like"].astype(str).str.lower().isin(["true", "1", "yes"])
    data["speed"] = pd.to_numeric(data.get("speed", np.nan), errors="coerce")
    data["spin"] = pd.to_numeric(data.get("spin", np.nan), errors="coerce")
    data["landing_y"] = pd.to_numeric(data.get("landing_y", np.nan), errors="coerce")
    data["depth_bin"] = depth_bin(data["landing_y"])
    data = data.sort_values(["sequence_id", "event_index"]).copy()
    data["prev_family"] = data.groupby("sequence_id")["coarse_family"].shift(1).fillna("Zero")
    data.loc[~data["prev_family"].isin(FAMILY_COLUMNS), "prev_family"] = "Zero"

    base_counts = data["coarse_family"].value_counts().reindex(FAMILY_COLUMNS, fill_value=0).astype(float) + 1.0
    global_prob = base_counts / base_counts.sum()

    phase_rows = []
    for phase, group in data.groupby("phase", dropna=False):
        rec = {"phase": str(phase), "count": int(len(group))}
        rec.update({f"family_{k}": float(v) for k, v in smoothed_family_counts(group, global_prob).items()})
        rec["terminal_rate"] = float(group["terminal_like"].mean())
        for col in ["speed", "spin", "landing_y"]:
            rec[f"{col}_mean"] = float(group[col].mean()) if group[col].notna().any() else 0.0
            rec[f"{col}_std"] = float(group[col].std()) if group[col].notna().sum() > 1 else 0.0
        for bucket in ["short", "mid", "deep", "missing"]:
            rec[f"depth_{bucket}_rate"] = float(group["depth_bin"].eq(bucket).mean())
        phase_rows.append(rec)

    transition_rows = []
    for (phase, prev_family), group in data.groupby(["phase", "prev_family"], dropna=False):
        rec = {"phase": str(phase), "prev_family": str(prev_family), "count": int(len(group))}
        rec.update({f"family_{k}": float(v) for k, v in smoothed_family_counts(group, global_prob).items()})
        rec["terminal_rate"] = float(group["terminal_like"].mean())
        transition_rows.append(rec)
    return pd.DataFrame(phase_rows), pd.DataFrame(transition_rows)


def priors_for_aicup(rows: pd.DataFrame, phase_table: pd.DataFrame, transition_table: pd.DataFrame) -> pd.DataFrame:
    phase = phase_from_aicup(rows)
    prev_family = lag0_family_from_aicup(rows)

    phase_idx = phase_table.set_index("phase")
    trans_idx = transition_table.set_index(["phase", "prev_family"])
    phase_cols = [col for col in phase_table.columns if col not in {"phase"}]
    trans_cols = [col for col in transition_table.columns if col not in {"phase", "prev_family"}]
    phase_fallback = phase_idx[phase_cols].mean(numeric_only=True).fillna(0.0)
    trans_fallback = trans_idx[trans_cols].mean(numeric_only=True).fillna(0.0)

    phase_records = []
    trans_records = []
    for p, prev in zip(phase, prev_family):
        phase_records.append(phase_idx.loc[p, phase_cols].fillna(phase_fallback).to_dict() if p in phase_idx.index else phase_fallback.to_dict())
        key = (p, prev)
        trans_records.append(trans_idx.loc[key, trans_cols].fillna(trans_fallback).to_dict() if key in trans_idx.index else trans_fallback.to_dict())

    phase_frame = pd.DataFrame(phase_records, index=rows.index).add_prefix("v262_phase_")
    trans_frame = pd.DataFrame(trans_records, index=rows.index).add_prefix("v262_trans_")
    out = pd.concat([phase_frame, trans_frame], axis=1).fillna(0.0)
    family_cols = [f"v262_trans_family_{family}" for family in FAMILY_COLUMNS]
    phase_family_cols = [f"v262_phase_family_{family}" for family in FAMILY_COLUMNS]
    for col in family_cols + phase_family_cols:
        if col not in out:
            out[col] = 0.0
    return out


def align_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(train.columns) | set(test.columns))
    return train.reindex(columns=cols, fill_value=0.0), test.reindex(columns=cols, fill_value=0.0)


def train_external_feature_teacher(ctx: dict, x: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = ctx["rows"]
    y = ctx["y"]
    oof = np.zeros((len(rows), 19), dtype=float)
    test = np.zeros((len(x_test), 19), dtype=float)
    fold_metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        model = ExtraTreesClassifier(
            n_estimators=120,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE + int(fold),
            n_jobs=1,
        )
        model.fit(x.loc[train].fillna(0.0), y[train])
        valid_raw = model.predict_proba(x.loc[valid].fillna(0.0))
        valid_prob = np.zeros((int(valid.sum()), 19), dtype=float)
        for j, cls in enumerate(model.classes_):
            valid_prob[:, int(cls)] = valid_raw[:, j]
        oof[valid] = valid_prob
        test_raw = model.predict_proba(x_test.fillna(0.0))
        test_prob = np.zeros((len(x_test), 19), dtype=float)
        for j, cls in enumerate(model.classes_):
            test_prob[:, int(cls)] = test_raw[:, j]
        test += test_prob / max(rows["fold"].nunique(), 1)
        fold_metrics.append({"fold": int(fold), "train_rows": int(train.sum()), "valid_rows": int(valid.sum())})
    return normalize_rows_safe(oof), normalize_rows_safe(test), fold_metrics


def one_hot(labels: np.ndarray, n_classes: int = 19) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=float)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def blend_with_anchor(anchor_labels: np.ndarray, teacher_prob: np.ndarray, weight: float) -> np.ndarray:
    return normalize_rows_safe((1.0 - weight) * one_hot(anchor_labels, teacher_prob.shape[1]) + weight * teacher_prob)


def classgate(anchor: np.ndarray, prob: np.ndarray) -> np.ndarray:
    raw = prob.argmax(axis=1).astype(int)
    conf = prob.max(axis=1)
    margin = np.sort(prob, axis=1)[:, -1] - np.sort(prob, axis=1)[:, -2]
    anchor_family = np.array([action_family_name(v) for v in anchor])
    raw_family = np.array([action_family_name(v) for v in raw])
    out = anchor.copy()
    take = np.isin(raw, WEAK_ACTIONS) & (raw != anchor) & (raw_family == anchor_family) & (conf >= 0.24) & (margin >= 0.025)
    out[take] = raw[take]
    return out


def action_distribution(labels: np.ndarray) -> dict[str, int]:
    counts = pd.Series(labels.astype(int)).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def verdict_from_deltas(delta: float, iw_delta: float) -> str:
    if delta >= 0.003 and iw_delta >= 0.001:
        return "CANDIDATE_FOR_PUBLIC_PROBE"
    if delta > 0.0 and iw_delta >= 0.0:
        return "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    corpus, corpus_audit = load_native_corpus()
    phase_table, transition_table = build_external_prior_tables(corpus)

    ctx = load_action_context()
    base_cols = feature_columns(ctx["rows"], drop_keywords=("next_",))
    train_base = ctx["rows"].loc[:, base_cols].copy()
    test_base = ctx["test_rows"].reindex(columns=base_cols, fill_value=0.0).copy()
    train_prior = priors_for_aicup(ctx["rows"], phase_table, transition_table)
    test_prior = priors_for_aicup(ctx["test_rows"], phase_table, transition_table)
    x_train, x_test = align_frames(
        pd.concat([train_base.reset_index(drop=True), train_prior.reset_index(drop=True)], axis=1),
        pd.concat([test_base.reset_index(drop=True), test_prior.reset_index(drop=True)], axis=1),
    )

    teacher_oof, teacher_test, fold_metrics = train_external_feature_teacher(ctx, x_train, x_test)
    weights = context_weights(ctx["rows"], ctx["test_rows"])
    variants = {
        "v262_native_raw_action": (teacher_oof.argmax(axis=1).astype(int), teacher_test.argmax(axis=1).astype(int)),
        "v262_native_v173blend_w0p05": (
            blend_with_anchor(ctx["v173_oof"], teacher_oof, 0.05).argmax(axis=1).astype(int),
            blend_with_anchor(ctx["v173_test"], teacher_test, 0.05).argmax(axis=1).astype(int),
        ),
        "v262_native_v173blend_w0p10": (
            blend_with_anchor(ctx["v173_oof"], teacher_oof, 0.10).argmax(axis=1).astype(int),
            blend_with_anchor(ctx["v173_test"], teacher_test, 0.10).argmax(axis=1).astype(int),
        ),
        "v262_native_classgate": (classgate(ctx["v173_oof"], teacher_oof), classgate(ctx["v173_test"], teacher_test)),
    }

    records = [evaluate_action("v173_anchor", ctx["y"], ctx["v173_oof"], ctx["v173_oof"], weights)]
    for name, (pred, test_pred) in variants.items():
        rec = evaluate_action(name, ctx["y"], pred, ctx["v173_oof"], weights)
        rec["test_action_churn_vs_v173"] = float(np.mean(test_pred != ctx["v173_test"]))
        rec["test_changed_rows"] = int(np.sum(test_pred != ctx["v173_test"]))
        rec["test_action_distribution"] = json.dumps(action_distribution(test_pred), sort_keys=True)
        records.append(rec)

    search = pd.DataFrame(records).sort_values(
        ["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"],
        ascending=[False, False, False],
    )
    search.to_csv(OUTDIR / "v262_action_search.csv", index=False)
    non_anchor = search[search["candidate"].ne("v173_anchor")]
    best = non_anchor.iloc[0].to_dict() if len(non_anchor) else {}
    best_delta = float(best.get("delta_vs_v173_anchor", 0.0))
    best_iw = float(best.get("iw_delta_vs_v173", 0.0))
    verdict = verdict_from_deltas(best_delta, best_iw)
    upload_recommendation = "do_not_upload"
    if verdict == "CANDIDATE_FOR_PUBLIC_PROBE":
        upload_recommendation = "controller_review_required_before_upload"

    report = {
        "worker": "V262",
        "verdict": verdict,
        "upload_recommendation": upload_recommendation,
        "best_candidate": best,
        "best_delta_vs_v173_anchor": best_delta,
        "best_public_like_iw_delta": best_iw,
        "corpus_audit": corpus_audit,
        "source_counts": corpus_audit["source_counts"],
        "external_rows": int(len(corpus)),
        "feature_count": int(x_train.shape[1]),
        "fold_metrics": fold_metrics,
        "outputs": {
            "search": str(OUTDIR / "v262_action_search.csv"),
            "report_json": str(OUTDIR / "v262_report.json"),
            "report_md": str(OUTDIR / "v262_report.md"),
        },
        "top_candidates": search.head(8).to_dict(orient="records"),
    }
    (OUTDIR / "v262_report.json").write_text(json.dumps(json_clean(report), indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# V262 Table-Tennis-Native External Sequence Teacher",
        "",
        f"Verdict: `{verdict}`",
        f"Upload recommendation: `{upload_recommendation}`",
        f"External rows used: `{len(corpus)}`",
        f"Best delta vs V173: `{best_delta:.6f}`",
        f"Best public-like/IW delta: `{best_iw:.6f}`",
        "",
        "Source counts:",
        "",
    ]
    for source, count in sorted(corpus_audit["source_counts"].items()):
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(["", "Top action candidates:", ""])
    for _, row in search.head(6).iterrows():
        lines.append(
            f"- `{row['candidate']}`: macro-F1 `{row['action_macro_f1']:.6f}`, "
            f"delta `{row['delta_vs_v173_anchor']:.6f}`, "
            f"IW `{row['iw_delta_vs_v173']:.6f}`, "
            f"weak `{row['weak_delta_vs_v173']:.6f}`, "
            f"test changed `{row.get('test_changed_rows', 0)}`"
        )
    lines.extend(
        [
            "",
            "Guards:",
            "",
            "- CoachAI/ShuttleSet badminton rows used: `0`",
            "- TTMATCH rows used: `0`",
            "- Exact external labels mapped to AICUP actionId: `False`",
            "- Controller-owned upload/selected outputs written: `False`",
        ]
    )
    (OUTDIR / "v262_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "verdict": verdict,
                "best_candidate": best.get("candidate", ""),
                "best_delta": best_delta,
                "best_public_like_iw_delta": best_iw,
                "external_rows": int(len(corpus)),
                "source_counts": corpus_audit["source_counts"],
                "upload_recommendation": upload_recommendation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
