"""V305 safe exporter for literal V188 r186_w005 point artifacts.

This script rebuilds the row-level V188 r186_w005 OOF/test point probabilities
without calling V188 submission writers. Outputs are restricted to
v305_literal_v188_point_artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


OUTDIR = Path("v305_literal_v188_point_artifact")
V188_OUTDIR = Path("v188_point_intent_gru")
KNOWN_V188_CAP5 = V188_OUTDIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
ALPHA = 0.05
CAPS = (0.02, 0.03, 0.05)
POINT_CLASSES = list(range(10))


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def normalize_rows_safe(x: np.ndarray) -> np.ndarray:
    """Return finite non-negative rows that sum to one, repairing bad rows."""
    arr = np.asarray(x, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"expected 2D probability matrix, got shape {arr.shape}")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    sums = arr.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(sums[:, 0]) | (sums[:, 0] <= 0.0)
    if np.any(bad):
        arr[bad] = 1.0 / arr.shape[1]
        sums = arr.sum(axis=1, keepdims=True)
    return arr / np.clip(sums, 1e-12, None)


def cap_residual_pred(base_labels: np.ndarray, prob: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply argmax residual predictions to the top-margin changed rows only."""
    base = np.asarray(base_labels, dtype=np.int64)
    proba = normalize_rows_safe(np.asarray(prob, dtype=float))
    if len(base) != len(proba):
        raise ValueError(f"base/prob row mismatch: {len(base)} vs {len(proba)}")
    pred = proba.argmax(axis=1).astype(np.int64)
    changed = pred != base
    max_rows = int(np.floor(len(base) * float(cap)))
    if max_rows < 0:
        raise ValueError(f"cap must be non-negative, got {cap}")
    if int(changed.sum()) > max_rows:
        rows = np.arange(len(base))
        margin = proba[rows, pred] - proba[rows, base]
        cand = np.where(changed)[0]
        keep = cand[np.argsort(margin[cand])[::-1][:max_rows]]
        changed = np.zeros(len(base), dtype=bool)
        changed[keep] = True
    out = base.copy()
    out[changed] = pred[changed]
    return out, changed


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def _row_log_blend(base_prob: np.ndarray, residual_prob: np.ndarray, alpha: float) -> np.ndarray:
    base = np.clip(normalize_rows_safe(base_prob), 1e-12, 1.0)
    residual = np.clip(normalize_rows_safe(residual_prob), 1e-12, 1.0)
    logp = (1.0 - alpha) * np.log(base) + alpha * np.log(residual)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(logp))


def _paths() -> dict[str, Path]:
    return {
        "base_oof": OUTDIR / "v305_v188_local_base_oof_proba.npy",
        "base_test": OUTDIR / "v305_v188_local_base_test_proba.npy",
        "r186_oof": OUTDIR / "v305_v188_r186_w005_oof_proba.npy",
        "r186_test": OUTDIR / "v305_v188_r186_w005_test_proba.npy",
        "oof_pred": OUTDIR / "v305_v188_cap5_oof_pred.csv",
        "test_pred": OUTDIR / "v305_v188_cap5_test_pred.csv",
        "oof_meta": OUTDIR / "v305_v188_oof_meta.csv",
        "report_json": OUTDIR / "v305_report.json",
        "report_md": OUTDIR / "v305_report.md",
    }


def _load_cached_arrays(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    required = [paths["base_oof"], paths["base_test"], paths["r186_oof"], paths["r186_test"]]
    if not all(p.exists() for p in required):
        return None
    return tuple(normalize_rows_safe(np.load(p)) for p in required)  # type: ignore[return-value]


def _prepare_context() -> dict[str, Any]:
    from analysis_r1_oof_ensemble import compose_v3
    from analysis_r116_r119_point_server import action_conditioned_point_prior, r119_oof_prior
    from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
    from analysis_r185_point_intent_model import BASE_V173, R121, add_r185_columns, load_sub, one_hot, point_pred
    from analysis_r187_point_intent_student import add_r186_priors
    from analysis_r67_r70_meta_priors import compose_v3_full_point
    from analysis_v165_combined_external_pretrain_proxy import R101_OOF, R101_TEST, R111_OOF, prepare_prefix_features
    from analysis_v188_point_intent_gru import R186_TEST, R186_TRAIN, load_pickle

    state = rebuild_v173_best_actions()
    train_raw, test_raw, prefix, _, _ = prepare_prefix_features()
    rows = add_r185_columns(state["rows"], state["v173_pred_oof"], pool=False)
    test_rows = add_r185_columns(state["test_rows"], state["v173_pred_test"], pool=False)
    rows = add_r186_priors(rows, pd.read_csv(R186_TRAIN))
    test_rows = add_r186_priors(test_rows, pd.read_csv(R186_TEST))

    r111_oof = load_pickle(R111_OOF)
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    v3_oof = load_pickle("oof_proba_v3.pkl")
    tuning = r111_oof["tuning"]
    _, v3_point_oof, _ = compose_v3(v3_oof)
    _, v3_point_test = compose_v3_full_point(train_raw, test_raw, v3_oof["tuning"])
    base_point_oof = normalize_rows_safe(0.97 * r101_oof["gru_point"] + 0.03 * v3_point_oof)
    base_point_test = normalize_rows_safe(0.97 * r101_test["gru_point"] + 0.03 * v3_point_test)

    prefix_train = add_r185_columns(prefix, None, pool=True)
    v173_action_oof_prob = one_hot(state["v173_pred_oof"], 19)
    v173_action_test_prob = one_hot(state["v173_pred_test"], 19)
    r119_oof = r119_oof_prior(rows, prefix_train, v173_action_oof_prob)
    r119_test = action_conditioned_point_prior(test_rows, prefix_train, v173_action_test_prob)
    local_base_prob_oof = normalize_rows_safe(0.95 * base_point_oof + 0.05 * r119_oof)
    local_base_prob_test = normalize_rows_safe(0.95 * base_point_test + 0.05 * r119_test)
    local_base_pred_oof = point_pred(rows, local_base_prob_oof, tuning)

    base_sub = load_sub(BASE_V173, state["rally_uids"])
    r121_sub = load_sub(R121, state["rally_uids"])
    base_sub["serverGetPoint"] = r121_sub["serverGetPoint"].astype(float).to_numpy()
    test_base_point = base_sub["pointId"].astype(int).to_numpy()

    return {
        "rows": rows,
        "test_rows": test_rows,
        "state": state,
        "v173_action_oof_prob": v173_action_oof_prob,
        "v173_action_test_prob": v173_action_test_prob,
        "local_base_prob_oof": local_base_prob_oof,
        "local_base_prob_test": local_base_prob_test,
        "local_base_pred_oof": local_base_pred_oof,
        "test_base_point": test_base_point,
        "base_sub": base_sub,
    }


def _train_literal_r186_w005(context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from analysis_v188_point_intent_gru import (
        LOSS_SETTINGS,
        MAX_SEQ_LEN,
        StrokeDataset,
        build_padded_stroke_tensor,
        predict_proba,
        raw_groups,
        sequences_for_rows,
        set_seed,
        static_matrix,
        teacher_matrix,
        train_model,
    )

    set_seed(188)
    rows = context["rows"]
    test_rows = context["test_rows"]
    train_seq, train_len = build_padded_stroke_tensor(sequences_for_rows(rows, raw_groups("train.csv")), MAX_SEQ_LEN, 0)
    test_seq, test_len = build_padded_stroke_tensor(sequences_for_rows(test_rows, raw_groups("test_new.csv")), MAX_SEQ_LEN, 0)
    vocab_sizes = [int(max(train_seq[:, :, i].max(), test_seq[:, :, i].max()) + 1) for i in range(train_seq.shape[2])]
    x_static, stats = static_matrix(rows, context["v173_action_oof_prob"], context["local_base_prob_oof"])
    x_test_static, _ = static_matrix(test_rows, context["v173_action_test_prob"], context["local_base_prob_test"], stats)
    teacher = teacher_matrix(rows)
    teacher_test = teacher_matrix(test_rows)
    y = rows["next_pointId"].astype(int).to_numpy()
    weights = dict(LOSS_SETTINGS)["r186_w005"]

    oof_prob = np.zeros((len(rows), len(POINT_CLASSES)), dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted(rows["fold"].unique()):
        valid = rows["fold"].eq(fold).to_numpy()
        train = ~valid
        train_ds = StrokeDataset(train_seq[train], train_len[train], x_static[train], y[train], teacher[train])
        valid_ds = StrokeDataset(train_seq[valid], train_len[valid], x_static[valid], y[valid], teacher[valid])
        model, val_loss = train_model(train_ds, valid_ds, vocab_sizes, x_static.shape[1], weights, 1880 + int(fold))
        oof_prob[valid] = predict_proba(model, valid_ds)
        fold_pred = oof_prob[valid].argmax(axis=1)
        fold_metrics.append(
            {
                "fold": int(fold),
                "valid_rows": int(valid.sum()),
                "val_loss": float(val_loss),
                "raw_point_macro_f1": _macro_f1(y[valid], fold_pred),
            }
        )

    full_ds = StrokeDataset(train_seq, train_len, x_static, y, teacher)
    hold = max(1, len(train_seq) // 10)
    hold_ds = StrokeDataset(train_seq[:hold], train_len[:hold], x_static[:hold], y[:hold], teacher[:hold])
    test_ds = StrokeDataset(test_seq, test_len, x_test_static, np.zeros(len(test_seq), dtype=np.int64), teacher_test)
    full_model, _ = train_model(full_ds, hold_ds, vocab_sizes, x_static.shape[1], weights, 1988)
    test_prob = predict_proba(full_model, test_ds)
    return normalize_rows_safe(oof_prob), normalize_rows_safe(test_prob), fold_metrics


def _write_oof_meta(rows: pd.DataFrame, local_base_pred: np.ndarray, raw_pred: np.ndarray, paths: dict[str, Path]) -> None:
    cols = [c for c in ["rally_uid", "fold", "prefix_len", "next_actionId", "next_pointId", "next_serverGetPoint"] if c in rows.columns]
    meta = rows[cols].copy() if cols else pd.DataFrame(index=np.arange(len(rows)))
    meta.insert(0, "row_id", np.arange(len(rows), dtype=int))
    meta["local_base_point_pred"] = np.asarray(local_base_pred, dtype=int)
    meta["raw_r186_w005_point_pred"] = np.asarray(raw_pred, dtype=int)
    meta.to_csv(paths["oof_meta"], index=False)


def _write_prediction_csvs(
    context: dict[str, Any],
    oof_raw_pred: np.ndarray,
    test_raw_pred: np.ndarray,
    oof_cap5_pred: np.ndarray,
    oof_cap5_changed: np.ndarray,
    test_cap5_pred: np.ndarray,
    test_cap5_changed: np.ndarray,
    paths: dict[str, Path],
) -> None:
    rows = context["rows"]
    oof = pd.DataFrame(
        {
            "row_id": np.arange(len(rows), dtype=int),
            "local_base_point_pred": np.asarray(context["local_base_pred_oof"], dtype=int),
            "raw_r186_w005_point_pred": np.asarray(oof_raw_pred, dtype=int),
            "cap0p05_point_pred": np.asarray(oof_cap5_pred, dtype=int),
            "cap0p05_changed": np.asarray(oof_cap5_changed, dtype=bool),
            "next_pointId": rows["next_pointId"].astype(int).to_numpy(),
        }
    )
    if "rally_uid" in rows.columns:
        oof.insert(1, "rally_uid", rows["rally_uid"].astype(int).to_numpy())
    oof.to_csv(paths["oof_pred"], index=False)

    base_sub = context["base_sub"]
    test = pd.DataFrame(
        {
            "rally_uid": base_sub["rally_uid"].astype(int).to_numpy(),
            "local_base_point_pred": np.asarray(context["test_base_point"], dtype=int),
            "raw_r186_w005_point_pred": np.asarray(test_raw_pred, dtype=int),
            "cap0p05_point_pred": np.asarray(test_cap5_pred, dtype=int),
            "cap0p05_changed": np.asarray(test_cap5_changed, dtype=bool),
        }
    )
    test.to_csv(paths["test_pred"], index=False)


def _known_cap5_match(test_cap5_pred: np.ndarray, rally_uids: np.ndarray) -> dict[str, Any]:
    if not KNOWN_V188_CAP5.exists():
        return {"path": str(KNOWN_V188_CAP5), "exists": False, "point_equal": None, "mismatch_rows": None}
    known = pd.read_csv(KNOWN_V188_CAP5)
    aligned = pd.DataFrame({"rally_uid": np.asarray(rally_uids, dtype=int), "generated_pointId": test_cap5_pred}).merge(
        known[["rally_uid", "pointId"]].rename(columns={"pointId": "known_pointId"}),
        on="rally_uid",
        how="left",
        validate="one_to_one",
    )
    equal = aligned["known_pointId"].notna().all() and np.array_equal(
        aligned["generated_pointId"].astype(int).to_numpy(),
        aligned["known_pointId"].astype(int).to_numpy(),
    )
    return {
        "path": str(KNOWN_V188_CAP5),
        "exists": True,
        "point_equal": bool(equal),
        "mismatch_rows": int((aligned["generated_pointId"].astype(int) != aligned["known_pointId"].fillna(-1).astype(int)).sum()),
    }


def _write_report(report: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["report_json"].write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    cap_lines = [
        f"| {cap_name} | {rec['point_macro_f1']:.6f} | {rec['changed_rows']} | {rec['test_changed_rows']} |"
        for cap_name, rec in report["caps"].items()
    ]
    paths["report_md"].write_text(
        "# V305 Literal V188 Point Artifact\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Train rows: `{report['row_counts']['oof']}`\n"
        f"- Test rows: `{report['row_counts']['test']}`\n"
        f"- Base point Macro-F1: `{report['metrics']['base_point_macro_f1']:.6f}`\n"
        f"- Raw r186_w005 Macro-F1: `{report['metrics']['raw_r186_w005_macro_f1']:.6f}`\n"
        f"- Known cap5 point match: `{report['known_v188_cap5']['point_equal']}`\n\n"
        f"- Known cap5 mismatched rows: `{report['known_v188_cap5']['mismatch_rows']}`\n\n"
        "## Caps\n\n"
        "| Cap | OOF Macro-F1 | OOF Changed Rows | Test Changed Rows |\n"
        "| --- | ---: | ---: | ---: |\n"
        + "\n".join(cap_lines)
        + "\n\n## Outputs\n\n"
        + "\n".join(f"- `{p}`" for p in report["outputs"].values())
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )


def export_literal_v188_artifact() -> dict[str, Any]:
    OUTDIR.mkdir(exist_ok=True)
    paths = _paths()
    context = _prepare_context()
    cached = _load_cached_arrays(paths)
    fold_metrics: list[dict[str, Any]] = []
    if cached is None:
        base_oof = context["local_base_prob_oof"]
        base_test = context["local_base_prob_test"]
        r186_oof, r186_test, fold_metrics = _train_literal_r186_w005(context)
        np.save(paths["base_oof"], base_oof)
        np.save(paths["base_test"], base_test)
        np.save(paths["r186_oof"], r186_oof)
        np.save(paths["r186_test"], r186_test)
        source = "recomputed"
    else:
        base_oof, base_test, r186_oof, r186_test = cached
        source = "cached_arrays"

    y = context["rows"]["next_pointId"].astype(int).to_numpy()
    local_base_pred_oof = context["local_base_pred_oof"]
    oof_raw_pred = r186_oof.argmax(axis=1).astype(int)
    test_raw_pred = r186_test.argmax(axis=1).astype(int)
    blended_oof = _row_log_blend(base_oof, r186_oof, ALPHA)
    blended_test = _row_log_blend(base_test, r186_test, ALPHA)

    caps: dict[str, dict[str, Any]] = {}
    cap5_oof_pred: np.ndarray | None = None
    cap5_oof_changed: np.ndarray | None = None
    cap5_test_pred: np.ndarray | None = None
    cap5_test_changed: np.ndarray | None = None
    for cap in CAPS:
        pred, changed = cap_residual_pred(local_base_pred_oof, blended_oof, cap)
        test_pred, test_changed = cap_residual_pred(context["test_base_point"], blended_test, cap)
        cap_key = f"cap{int(round(cap * 100)):02d}"
        caps[cap_key] = {
            "cap": float(cap),
            "point_macro_f1": _macro_f1(y, pred),
            "changed_rows": int(changed.sum()),
            "point_churn_vs_base": float(np.mean(pred != local_base_pred_oof)),
            "test_changed_rows": int(test_changed.sum()),
            "test_churn_vs_base": float(np.mean(test_pred != context["test_base_point"])),
        }
        if np.isclose(cap, 0.05):
            cap5_oof_pred = pred
            cap5_oof_changed = changed
            cap5_test_pred = test_pred
            cap5_test_changed = test_changed

    assert cap5_oof_pred is not None
    assert cap5_oof_changed is not None
    assert cap5_test_pred is not None
    assert cap5_test_changed is not None

    _write_oof_meta(context["rows"], local_base_pred_oof, oof_raw_pred, paths)
    _write_prediction_csvs(
        context,
        oof_raw_pred,
        test_raw_pred,
        cap5_oof_pred,
        cap5_oof_changed,
        cap5_test_pred,
        cap5_test_changed,
        paths,
    )

    known_cap5 = _known_cap5_match(cap5_test_pred, context["base_sub"]["rally_uid"].to_numpy())
    notes = [
        "Action/server submission files are not generated by this exporter.",
        "Only row-level point probabilities, point predictions, metadata, and reports are written.",
    ]
    if known_cap5["exists"] and not known_cap5["point_equal"]:
        notes.append(
            "Generated cap5 test points do not exactly match the existing V188 cap5 CSV; original V188 probability arrays were not present, so this run is a safe recomputation artifact."
        )

    report = {
        "verdict": "EXPORTED",
        "source": source,
        "row_counts": {"oof": int(len(y)), "test": int(len(context["test_base_point"]))},
        "shapes": {
            "local_base_oof": list(base_oof.shape),
            "local_base_test": list(base_test.shape),
            "r186_w005_oof": list(r186_oof.shape),
            "r186_w005_test": list(r186_test.shape),
        },
        "metrics": {
            "base_point_macro_f1": _macro_f1(y, local_base_pred_oof),
            "raw_r186_w005_macro_f1": _macro_f1(y, oof_raw_pred),
        },
        "caps": caps,
        "known_v188_cap5": known_cap5,
        "fold_metrics": fold_metrics,
        "outputs": {name: str(path) for name, path in paths.items() if name not in {"report_json", "report_md"}},
        "notes": notes,
    }
    _write_report(report, paths)
    return report


def main() -> None:
    report = export_literal_v188_artifact()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "source": report["source"],
                "oof_shape": report["shapes"]["r186_w005_oof"],
                "test_shape": report["shapes"]["r186_w005_test"],
                "cap5_macro_f1": report["caps"]["cap05"]["point_macro_f1"],
                "known_cap5_point_equal": report["known_v188_cap5"]["point_equal"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
