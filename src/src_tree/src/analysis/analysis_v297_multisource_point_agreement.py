"""V297 multi-source point agreement explorer.

Clean-line diagnostic: use several independently trained point probability
sources as voters and only produce low-churn point edits over the current V261
clean anchor.  Action/server are copied unchanged.
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v297_multisource_point_agreement"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v297_multisource_point_agreement.py"
ANCHOR_PATH = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
V294_DIR = ROOT / "v294_point_oof_artifact_builder"
EXPECTED_ROWS = 1845
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
POINT_CLASSES = list(range(10))
LONG = {7, 8, 9}


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


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def point0_rate(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=int) == 0))


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_sources() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, np.ndarray]]:
    r111_oof = _load_pickle(ROOT / "r111_remaining_moe_gru" / "oof_proba_r111.pkl")
    r111_test = _load_pickle(ROOT / "r111_remaining_moe_gru" / "test_proba_r111.pkl")
    r101_oof = _load_pickle(ROOT / "r101_r103_destiny_gru" / "oof_proba_r101_r103.pkl")
    r101_test = _load_pickle(ROOT / "r101_r103_destiny_gru" / "test_proba_r101_r103.pkl")
    r180_oof = normalize_rows_safe(np.load(ROOT / "r180_point_physics_calibration" / "r180_best_point_oof.npy"))
    r180_test = normalize_rows_safe(np.load(ROOT / "r180_point_physics_calibration" / "r180_best_point_test.npy"))
    v294_test = normalize_rows_safe(np.load(V294_DIR / "v294_point_test_proba.npy"))

    meta = r111_oof["valid_meta"].reset_index(drop=True).copy()
    test_meta = r111_test["test_meta"].reset_index(drop=True).copy()
    if len(meta) != r180_oof.shape[0] or len(test_meta) != r180_test.shape[0]:
        raise ValueError("source row counts are not aligned")
    oof_sources = {
        "r111": normalize_rows_safe(r111_oof["gru_point"]),
        "r101": normalize_rows_safe(r101_oof["gru_point"]),
        "r180": r180_oof,
    }
    test_sources = {
        "r111": normalize_rows_safe(r111_test["point"]),
        "r101": normalize_rows_safe(r101_test["point"]),
        "r180": r180_test,
        "v294": v294_test,
    }
    return meta, oof_sources, test_meta, test_sources


def source_vote_candidates(base: np.ndarray, sources: dict[str, np.ndarray], mode: str) -> pd.DataFrame:
    base = np.asarray(base, dtype=int)
    names = list(sources)
    top = {name: normalize_rows_safe(prob).argmax(axis=1).astype(int) for name, prob in sources.items()}
    conf = {name: normalize_rows_safe(prob).max(axis=1) for name, prob in sources.items()}
    rows: list[dict[str, Any]] = []
    for row_id, base_point in enumerate(base):
        votes = [int(top[name][row_id]) for name in names]
        counts = pd.Series(votes).value_counts()
        candidate = int(counts.index[0])
        agree = int(counts.iloc[0])
        if candidate == int(base_point):
            continue
        if mode == "long789" and not (int(base_point) in LONG and candidate in LONG):
            continue
        if mode == "no_point0" and candidate == 0:
            continue
        if mode == "no_rare134" and candidate in {1, 3, 4}:
            continue
        if agree < (3 if mode in {"all_strong", "long789"} else 2):
            continue
        voters = [name for name in names if int(top[name][row_id]) == candidate]
        score = float(agree + np.mean([float(conf[name][row_id]) for name in voters]))
        rows.append(
            {
                "row_id": row_id,
                "candidate_point": candidate,
                "score": score,
                "agree": agree,
                "mode": mode,
                "voters": "+".join(voters),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["row_id", "candidate_point", "score", "agree", "mode", "voters"])
    return pd.DataFrame(rows)


def apply_candidates(base: np.ndarray, candidates: pd.DataFrame, cap: float) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.asarray(base, dtype=int).copy()
    if candidates.empty or cap <= 0:
        return pred, candidates.head(0).copy()
    max_rows = int(math.floor(len(pred) * cap))
    if max_rows <= 0:
        return pred, candidates.head(0).copy()
    selected = (
        candidates.sort_values(["score", "agree", "row_id"], ascending=[False, False, True])
        .drop_duplicates("row_id", keep="first")
        .head(max_rows)
        .copy()
    )
    rows = selected["row_id"].astype(int).to_numpy()
    pred[rows] = selected["candidate_point"].astype(int).to_numpy()
    return pred, selected


def evaluate_variant(
    name: str,
    cap: float,
    mode: str,
    oof_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    y_true: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    pred_oof, selected_oof = apply_candidates(base_oof, oof_candidates, cap)
    pred_test, selected_test = apply_candidates(base_test, test_candidates, cap)
    base_score = macro_f1(y_true, base_oof)
    score = macro_f1(y_true, pred_oof)
    rec = {
        "candidate": name,
        "mode": mode,
        "cap": cap,
        "point_macro_f1": score,
        "base_point_macro_f1": base_score,
        "delta_vs_aligned_base": score - base_score,
        "oof_changed_rows": int(len(selected_oof)),
        "test_changed_rows": int(len(selected_test)),
        "point_churn": float(len(selected_test) / len(base_test)),
        "point0_rate_delta": point0_rate(pred_test) - point0_rate(base_test),
        "upload_recommendation": "DO_NOT_UPLOAD",
    }
    if rec["delta_vs_aligned_base"] >= 0.001 and 3 <= rec["test_changed_rows"] <= 20 and rec["point0_rate_delta"] <= 0.003:
        rec["upload_recommendation"] = "REVIEW_UPLOAD"
    return rec, pred_oof, pred_test, selected_test


def write_submission(path: Path, point_pred: np.ndarray, anchor: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> None:
    out = anchor.copy()
    out["pointId"] = np.asarray(point_pred, dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if len(out) != expected_rows:
        raise ValueError("bad submission row count")
    out.to_csv(path, index=False)


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, oof_sources, _test_meta, test_sources = load_sources()
    anchor = pd.read_csv(ANCHOR_PATH)
    if list(anchor.columns) != SUBMISSION_COLUMNS or len(anchor) != EXPECTED_ROWS:
        raise ValueError("bad anchor submission")
    y_true = meta["next_pointId"].astype(int).to_numpy()
    base_oof = normalize_rows_safe(oof_sources["r111"]).argmax(axis=1).astype(int)
    base_test = anchor["pointId"].astype(int).to_numpy()

    variants: list[tuple[str, float, str]] = []
    for mode in ["all_strong", "long789", "no_point0", "no_rare134"]:
        for cap in [0.0025, 0.005, 0.01]:
            variants.append((f"v297_{mode}_cap{str(cap).replace('.', 'p')}", cap, mode))

    records: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    generated: list[str] = []
    predictions: dict[str, np.ndarray] = {}
    for name, cap, mode in variants:
        oof_cands = source_vote_candidates(base_oof, oof_sources, mode)
        test_cands = source_vote_candidates(base_test, test_sources, mode)
        rec, pred_oof, pred_test, selected_test = evaluate_variant(
            name, cap, mode, oof_cands, test_cands, y_true, base_oof, base_test
        )
        filename = f"submission_{name}__v173action_r121server.csv"
        out_path = OUT_DIR / filename
        write_submission(out_path, pred_test, anchor)
        rec["path"] = str(out_path.relative_to(ROOT))
        records.append(rec)
        generated.append(str(out_path.relative_to(ROOT)))
        predictions[name] = pred_oof
        if not selected_test.empty:
            audit = selected_test.copy()
            audit["candidate"] = name
            audit["rally_uid"] = anchor.iloc[audit["row_id"].astype(int).to_numpy()]["rally_uid"].to_numpy()
            audit["base_point"] = base_test[audit["row_id"].astype(int).to_numpy()]
            audit_rows.append(audit)

    search = pd.DataFrame(records).sort_values(
        ["upload_recommendation", "delta_vs_aligned_base", "test_changed_rows"],
        ascending=[False, False, True],
    )
    search.to_csv(OUT_DIR / "v297_candidate_search.csv", index=False)
    if audit_rows:
        pd.concat(audit_rows, ignore_index=True).to_csv(OUT_DIR / "v297_changed_row_audit.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "v297_changed_row_audit.csv", index=False)
    best = search.iloc[0].to_dict()
    report = {
        "version": "V297",
        "anchor_submission": str(ANCHOR_PATH.relative_to(ROOT)),
        "generated_submissions": generated,
        "best_candidate": best,
        "upload_recommendation": "REVIEW_UPLOAD" if search["upload_recommendation"].eq("REVIEW_UPLOAD").any() else "DO_NOT_UPLOAD",
        "local_oof_note": "OOF base is aligned R111 argmax, not literal V261 OOF; use as source-agreement diagnostic only.",
        "no_ttmatch_no_old_server": True,
    }
    (OUT_DIR / "v297_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V297 multi-source point agreement",
        "",
        f"Anchor: `{ANCHOR_PATH.relative_to(ROOT)}`",
        "Action/server fixed. TTMATCH/old-server not used.",
        f"Upload recommendation: `{report['upload_recommendation']}`",
        "",
        "## Best candidate",
        "",
        f"- candidate: `{best['candidate']}`",
        f"- delta vs aligned base: `{float(best['delta_vs_aligned_base']):.6f}`",
        f"- test changed rows: `{int(best['test_changed_rows'])}`",
        "",
        "## Note",
        "",
        report["local_oof_note"],
    ]
    (OUT_DIR / "v297_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), SRC_DEST)
    return report


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": str(OUT_DIR.relative_to(ROOT)),
                "best_candidate": best["candidate"],
                "best_delta_vs_aligned_base": best["delta_vs_aligned_base"],
                "best_test_changed_rows": best["test_changed_rows"],
                "generated_submissions": len(report["generated_submissions"]),
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
