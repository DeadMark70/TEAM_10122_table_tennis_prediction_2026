"""V295 true/rebuilt-OOF point weak-class specialists.

This script consumes the V294 row-level point artifact and tests constrained
point-only changes over the current clean V261 anchor.  It deliberately avoids
action/server changes, TTMATCH, old-server labels, and manual row edits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v295_true_oof_point_specialists"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v295_true_oof_point_specialists.py"
V294_DIR = ROOT / "v294_point_oof_artifact_builder"
ANCHOR_PATH = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"

POINT_CLASSES = list(range(10))
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
LONG_CLASSES = {7, 8, 9}
RARE_CLASSES = {1, 3, 4}


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
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def normalize_rows_safe(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr[~np.isfinite(arr)] = 0.0
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= eps
    if np.any(bad):
        arr[bad] = 1.0 / arr.shape[1]
        denom = arr.sum(axis=1, keepdims=True)
    return arr / denom


def clean_score_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, 0.0, None)


def class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        label: float(f1_score(y_true == label, y_pred == label, zero_division=0))
        for label in POINT_CLASSES
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def weighted_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    scores: list[float] = []
    for label in POINT_CLASSES:
        yt = y_true == label
        yp = y_pred == label
        tp = float(weights[yt & yp].sum())
        fp = float(weights[~yt & yp].sum())
        fn = float(weights[yt & ~yp].sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def public_like_weights(oof_base: pd.DataFrame) -> np.ndarray:
    prefix = pd.to_numeric(oof_base.get("prefix_len", 0), errors="coerce").fillna(0).to_numpy()
    phase = oof_base.get("phase", pd.Series([""] * len(oof_base))).astype(str).to_numpy()
    lag0_action = pd.to_numeric(oof_base.get("lag0_actionId", 0), errors="coerce").fillna(0).to_numpy()
    lag0_point = pd.to_numeric(oof_base.get("lag0_pointId", 0), errors="coerce").fillna(0).to_numpy()
    weights = np.ones(len(oof_base), dtype=float)
    weights += 0.15 * (prefix >= 3)
    weights += 0.15 * np.isin(phase, ["rally", "4", "fourth"])
    weights += 0.10 * np.isin(lag0_point, [7, 8, 9])
    weights += 0.10 * np.isin(lag0_action, [1, 2, 3, 4, 5, 6, 7])
    return weights


def _candidate_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = ["row_id", "candidate_point", "score", "specialist", "reason"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).loc[:, cols]


def long789_candidates(base: np.ndarray, proba: np.ndarray, mode: str = "proba") -> pd.DataFrame:
    base = np.asarray(base, dtype=int)
    prob = clean_score_matrix(proba)
    rows: list[dict[str, Any]] = []
    long_idx = np.array(sorted(LONG_CLASSES), dtype=int)
    for row_id, base_point in enumerate(base):
        if int(base_point) not in LONG_CLASSES:
            continue
        long_prob = prob[row_id, long_idx]
        target_pos = int(np.argmax(long_prob))
        target = int(long_idx[target_pos])
        if target == int(base_point):
            challenger_pos = [
                pos for pos, point in enumerate(long_idx.tolist()) if int(point) != int(base_point)
            ]
            target_pos = max(challenger_pos, key=lambda pos: float(long_prob[pos]))
            target = int(long_idx[target_pos])
        if mode == "margin":
            score = float(prob[row_id, target] - prob[row_id, int(base_point)])
        else:
            score = float(prob[row_id, target])
        rows.append(
            {
                "row_id": row_id,
                "candidate_point": target,
                "score": score,
                "specialist": "long789",
                "reason": f"{mode}_long_internal",
            }
        )
    return _candidate_frame(rows)


def rare134_ovr_candidates(base: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    base = np.asarray(base, dtype=int)
    prob = clean_score_matrix(proba)
    rare_idx = np.array(sorted(RARE_CLASSES), dtype=int)
    rows: list[dict[str, Any]] = []
    for row_id, base_point in enumerate(base):
        if int(base_point) in {8, 9}:
            continue
        rare_prob = prob[row_id, rare_idx]
        target_pos = int(np.argmax(rare_prob))
        target = int(rare_idx[target_pos])
        if target == int(base_point):
            challenger_pos = [
                pos for pos, point in enumerate(rare_idx.tolist()) if int(point) != int(base_point)
            ]
            target_pos = max(challenger_pos, key=lambda pos: float(rare_prob[pos]))
            target = int(rare_idx[target_pos])
        score = float(prob[row_id, target] - prob[row_id, int(base_point)])
        rows.append(
            {
                "row_id": row_id,
                "candidate_point": target,
                "score": score,
                "specialist": "rare134",
                "reason": "rare134_ovr",
            }
        )
    return _candidate_frame(rows)


def point0_conservative_candidates(base: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    base = np.asarray(base, dtype=int)
    prob = clean_score_matrix(proba)
    rows: list[dict[str, Any]] = []
    for row_id, base_point in enumerate(base):
        if int(base_point) == 0:
            continue
        p0 = float(prob[row_id, 0])
        score = p0 - float(prob[row_id, int(base_point)])
        if p0 < 0.25:
            continue
        rows.append(
            {
                "row_id": row_id,
                "candidate_point": 0,
                "score": score,
                "specialist": "point0",
                "reason": "p0_ge_0p25",
            }
        )
    return _candidate_frame(rows)


def agreement_candidates(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty or right.empty:
        return _candidate_frame([])
    merged = left.merge(
        right,
        on=["row_id", "candidate_point"],
        suffixes=("_proba", "_margin"),
    )
    if merged.empty:
        return _candidate_frame([])
    return _candidate_frame(
        [
            {
                "row_id": int(row.row_id),
                "candidate_point": int(row.candidate_point),
                "score": float((row.score_proba + row.score_margin) / 2.0),
                "specialist": "long789",
                "reason": "proba_margin_agree",
            }
            for row in merged.itertuples(index=False)
        ]
    )


def apply_candidates(base: np.ndarray, candidates: pd.DataFrame, cap: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pred = np.asarray(base, dtype=int).copy()
    selected = np.zeros(len(pred), dtype=bool)
    if candidates.empty or cap <= 0.0:
        return pred, selected, candidates.head(0).copy()
    max_rows = len(pred) if cap >= 1.0 else int(math.floor(len(pred) * float(cap)))
    if max_rows <= 0:
        return pred, selected, candidates.head(0).copy()
    candidates = candidates.copy()
    row_ids_all = candidates["row_id"].astype(int).to_numpy()
    if (row_ids_all < 0).any() or (row_ids_all >= len(pred)).any():
        raise ValueError("candidate row_id out of range")
    candidates = candidates[
        candidates["candidate_point"].astype(int).to_numpy() != pred[row_ids_all]
    ]
    if candidates.empty:
        return pred, selected, candidates.head(0).copy()
    ranked = (
        candidates.sort_values(["score", "row_id"], ascending=[False, True])
        .drop_duplicates("row_id", keep="first")
        .head(max_rows)
        .copy()
    )
    row_ids = ranked["row_id"].astype(int).to_numpy()
    selected[row_ids] = True
    pred[row_ids] = ranked["candidate_point"].astype(int).to_numpy()
    return pred, selected, ranked


def write_submission(out_dir: Path, filename: str, pred_point: np.ndarray, anchor: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> Path:
    if len(anchor) != expected_rows:
        raise ValueError(f"anchor row count mismatch: {len(anchor)}")
    out = anchor.copy()
    out["pointId"] = np.asarray(pred_point, dtype=int)
    if list(out.columns) != SUBMISSION_COLUMNS:
        out = out.loc[:, SUBMISSION_COLUMNS]
    if len(out) != expected_rows:
        raise ValueError(f"submission row count mismatch: {len(out)}")
    if not out["pointId"].between(0, 9).all():
        raise ValueError("pointId out of range")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    out.to_csv(path, index=False)
    return path


def evaluate_variant(
    name: str,
    specialist_group: str,
    cap: float,
    train_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    pred_oof, selected_oof, _selected_oof_rows = apply_candidates(base_oof, train_candidates, cap)
    pred_test, selected_test, selected_test_rows = apply_candidates(base_test, test_candidates, cap)

    base_score = macro_f1(y_true, base_oof)
    point_score = macro_f1(y_true, pred_oof)
    base_public_like = weighted_macro_f1(y_true, base_oof, weights)
    public_like = weighted_macro_f1(y_true, pred_oof, weights)
    base_f1 = class_f1(y_true, base_oof)
    cand_f1 = class_f1(y_true, pred_oof)
    long_delta = float(np.mean([cand_f1[c] - base_f1[c] for c in [7, 8, 9]]))
    rare_delta = float(np.mean([cand_f1[c] - base_f1[c] for c in [1, 3, 4]]))
    point0_delta = float(cand_f1[0] - base_f1[0])
    test_changed_rows = int(selected_test.sum())
    point_churn = float(test_changed_rows / len(base_test))
    test_point0_rate_delta = float(np.mean(pred_test == 0) - np.mean(base_test == 0))
    recommendation = "DO_NOT_UPLOAD"
    if (
        point_score - base_score >= 0.0015
        and public_like - base_public_like >= 0.0008
        and 3 <= test_changed_rows <= 20
        and test_point0_rate_delta <= 0.005
        and not (specialist_group == "long789" and long_delta <= 0.0)
    ):
        recommendation = "REVIEW_UPLOAD"
    rec = {
        "candidate": name,
        "specialist_group": specialist_group,
        "cap": cap,
        "point_macro_f1": point_score,
        "base_point_macro_f1": base_score,
        "delta_vs_v294_base": point_score - base_score,
        "public_like_point_macro_f1": public_like,
        "base_public_like_point_macro_f1": base_public_like,
        "public_like_delta": public_like - base_public_like,
        "point_churn": point_churn,
        "train_changed_rows": int(selected_oof.sum()),
        "test_changed_rows": test_changed_rows,
        "test_point0_rate_delta": test_point0_rate_delta,
        "long789_mean_delta": long_delta,
        "rare134_mean_delta": rare_delta,
        "point0_f1_delta": point0_delta,
        "upload_recommendation": recommendation,
    }
    return rec, pred_oof, pred_test, selected_test_rows


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    oof_base = pd.read_csv(V294_DIR / "v294_point_oof_base.csv")
    oof_proba = normalize_rows_safe(np.load(V294_DIR / "v294_point_oof_proba.npy"))
    test_base = pd.read_csv(V294_DIR / "v294_point_test_base.csv")
    test_proba = normalize_rows_safe(np.load(V294_DIR / "v294_point_test_proba.npy"))
    anchor = pd.read_csv(ANCHOR_PATH)
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"bad anchor columns: {list(anchor.columns)}")
    if len(anchor) != EXPECTED_ROWS or len(test_base) != EXPECTED_ROWS:
        raise ValueError("test row count mismatch")
    if len(oof_base) != oof_proba.shape[0] or len(test_base) != test_proba.shape[0]:
        raise ValueError("V294 proba/base length mismatch")
    if oof_proba.shape[1] != 10 or test_proba.shape[1] != 10:
        raise ValueError("point proba must have 10 columns")
    return oof_base, oof_proba, test_base, test_proba, anchor


def build_class_report(y_true: np.ndarray, base: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    base_f1 = class_f1(y_true, base)
    pred_f1 = class_f1(y_true, pred)
    return pd.DataFrame(
        [
            {
                "pointId": cls,
                "base_f1": base_f1[cls],
                "candidate_f1": pred_f1[cls],
                "delta": pred_f1[cls] - base_f1[cls],
                "support": int(np.sum(y_true == cls)),
                "base_pred_count": int(np.sum(base == cls)),
                "candidate_pred_count": int(np.sum(pred == cls)),
            }
            for cls in POINT_CLASSES
        ]
    )


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_base_df, oof_proba, test_base_df, test_proba, anchor = load_inputs()
    y_true = oof_base_df["y_true_point"].astype(int).to_numpy()
    base_oof = oof_base_df["base_point_oof"].astype(int).to_numpy()
    base_test = anchor["pointId"].astype(int).to_numpy()
    weights = public_like_weights(oof_base_df)

    long_train = long789_candidates(base_oof, oof_proba, mode="proba")
    long_test = long789_candidates(base_test, test_proba, mode="proba")
    long_margin_train = long789_candidates(base_oof, oof_proba, mode="margin")
    long_margin_test = long789_candidates(base_test, test_proba, mode="margin")
    long_agree_train = agreement_candidates(long_train, long_margin_train)
    long_agree_test = agreement_candidates(long_test, long_margin_test)
    rare_train = rare134_ovr_candidates(base_oof, oof_proba)
    rare_test = rare134_ovr_candidates(base_test, test_proba)
    p0_train = point0_conservative_candidates(base_oof, oof_proba)
    p0_test = point0_conservative_candidates(base_test, test_proba)

    variants = [
        ("v295_long789_proba_cap0p0025", "long789", 0.0025, long_train, long_test),
        ("v295_long789_proba_cap0p005", "long789", 0.005, long_train, long_test),
        ("v295_long789_margin_cap0p005", "long789", 0.005, long_margin_train, long_margin_test),
        ("v295_rare134_ovr_cap0p0025", "rare134", 0.0025, rare_train, rare_test),
        ("v295_point0_conservative_cap0p0025", "point0", 0.0025, p0_train, p0_test),
        ("v295_bank_no_point0_cap0p005", "bank_no_point0", 0.005, pd.concat([long_train, rare_train], ignore_index=True), pd.concat([long_test, rare_test], ignore_index=True)),
        ("v295_bank_long789_only_agree_cap0p005", "bank_long789_only_agree", 0.005, long_agree_train, long_agree_test),
    ]

    records: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    generated: list[str] = []
    for name, group, cap, train_candidates, test_candidates in variants:
        rec, pred_oof, pred_test, selected_test_rows = evaluate_variant(
            name, group, cap, train_candidates, test_candidates, y_true, base_oof, base_test, weights
        )
        filename = f"submission_{name}__v173action_r121server.csv"
        path = write_submission(OUT_DIR, filename, pred_test, anchor)
        rec["path"] = str(path.relative_to(ROOT))
        records.append(rec)
        predictions[name] = pred_oof
        generated.append(str(path.relative_to(ROOT)))
        if not selected_test_rows.empty:
            audit = selected_test_rows.copy()
            audit["candidate"] = name
            audit["rally_uid"] = anchor.iloc[audit["row_id"].astype(int).to_numpy()]["rally_uid"].to_numpy()
            audit["base_point"] = base_test[audit["row_id"].astype(int).to_numpy()]
            audit_rows.append(audit)

    search = pd.DataFrame(records).sort_values(
        ["upload_recommendation", "delta_vs_v294_base", "public_like_delta", "test_changed_rows"],
        ascending=[False, False, False, True],
    )
    search.to_csv(OUT_DIR / "v295_candidate_search.csv", index=False)
    if audit_rows:
        changed_audit = pd.concat(audit_rows, ignore_index=True)
    else:
        changed_audit = pd.DataFrame(
            columns=["row_id", "candidate_point", "score", "specialist", "reason", "candidate", "rally_uid", "base_point"]
        )
    changed_audit.to_csv(OUT_DIR / "v295_changed_row_audit.csv", index=False)

    best_row = search.sort_values(["delta_vs_v294_base", "public_like_delta", "test_changed_rows"], ascending=[False, False, True]).iloc[0]
    best_name = str(best_row["candidate"])
    class_report = build_class_report(y_true, base_oof, predictions[best_name])
    class_report.to_csv(OUT_DIR / "v295_class_report.csv", index=False)

    report = _json_safe(
        {
            "version": "V295",
            "anchor_submission": str(ANCHOR_PATH.relative_to(ROOT)),
            "v294_source": "rebuilt_v261_like",
            "fixed_output": {
                "actionId": "copied exactly from V261 anchor",
                "serverGetPoint": "copied exactly from V261 anchor",
                "pointId": "constrained V295 point specialist variants",
            },
            "no_ttmatch_no_old_server": True,
            "best_candidate": best_row.to_dict(),
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "upload_recommendation": "REVIEW_UPLOAD" if search["upload_recommendation"].eq("REVIEW_UPLOAD").any() else "DO_NOT_UPLOAD",
            "candidate_source_counts": {
                "long789_train": len(long_train),
                "long789_test": len(long_test),
                "rare134_train": len(rare_train),
                "rare134_test": len(rare_test),
                "point0_train": len(p0_train),
                "point0_test": len(p0_test),
                "long789_agree_train": len(long_agree_train),
                "long789_agree_test": len(long_agree_test),
            },
            "concerns": [
                "V294 is a rebuilt_v261_like OOF base, not a literal stored V261 OOF artifact.",
                "V296 historical risk labels should still be consulted before upload; long789 is only YELLOW unless true OOF gain is large.",
            ],
        }
    )
    (OUT_DIR / "v295_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    lines = [
        "# V295 true/rebuilt-OOF point specialists",
        "",
        f"Anchor: `{ANCHOR_PATH.relative_to(ROOT)}`",
        "Fixed fields: `actionId` and `serverGetPoint` copied from anchor.",
        "TTMATCH/old-server: not used.",
        f"Generated submissions: `{len(generated)}`",
        f"Upload recommendation: `{report['upload_recommendation']}`",
        "",
        "## Best candidate",
        "",
        f"Candidate: `{best_row['candidate']}`",
        f"Point Macro-F1: `{float(best_row['point_macro_f1']):.6f}`",
        f"Delta vs V294 base: `{float(best_row['delta_vs_v294_base']):.6f}`",
        f"Public-like delta: `{float(best_row['public_like_delta']):.6f}`",
        f"Test changed rows: `{int(best_row['test_changed_rows'])}`",
        "",
        "## Candidates",
        "",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"- `{row['candidate']}`: delta={float(row['delta_vs_v294_base']):.6f}, "
            f"public_like={float(row['public_like_delta']):.6f}, "
            f"churn={float(row['point_churn']):.6f}, test_changed={int(row['test_changed_rows'])}, "
            f"recommendation=`{row['upload_recommendation']}`"
        )
    lines.extend(["", "## Concerns", "", *[f"- {item}" for item in report["concerns"]]])
    (OUT_DIR / "v295_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": str(OUT_DIR.relative_to(ROOT)),
                "best_candidate": best.get("candidate", ""),
                "best_delta_vs_v294_base": best.get("delta_vs_v294_base", 0.0),
                "best_public_like_delta": best.get("public_like_delta", 0.0),
                "best_test_changed_rows": best.get("test_changed_rows", 0),
                "generated_submissions": report["generated_submission_count"],
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
