"""V268 macro-F1-aware clean point residual search.

This branch keeps the current clean anchor fixed for action/server:

  v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv

Only ``pointId`` is changed, using capped residual edits from balanced point
classifiers trained on leakage-safe train/test prefix features.  Outputs are
local review probes; no upload is performed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_v263_questionnaire_baseline_helpers import (
    add_questionnaire_columns,
    point_depth,
    point_side,
)
from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


OUTDIR = Path("v268_macro_f1_point_residual")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
MAX_LAG = 6
RARE_POINT_CLASSES = [1, 3, 4, 7, 8, 9]
MAX_POINT0_ADDITIONS = 18
BLOCKED_FEATURES = {
    "rally_uid",
    "match",
    "server_id",
    "receiver_id",
    "gamePlayerId",
    "gamePlayerOtherId",
    "scoreSelf",
    "scoreOther",
    "next_actionId",
    "next_pointId",
    "next_is_terminal",
    "serverGetPoint",
    "remaining_len",
    "final_parity_even",
    "num_prefixes_in_rally",
    "fold",
}


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")


def class_prior_logit_adjustment(class_counts: np.ndarray, tau: float) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=float)
    counts = np.maximum(counts, 1.0)
    prior = counts / counts.sum()
    return -float(tau) * np.log(prior)


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def cap_by_score(scores: np.ndarray, cap: float) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    budget = int(np.floor(len(arr) * float(cap)))
    mask = np.zeros(len(arr), dtype=bool)
    if budget <= 0:
        return mask
    clean = np.where(np.isfinite(arr), arr, -np.inf)
    order = np.argsort(-clean, kind="mergesort")[:budget]
    mask[order] = True
    return mask


def point0_rate_ok(point_pred: np.ndarray, *, lower: float = 0.24, upper: float = 0.31) -> bool:
    rate = float(np.mean(np.asarray(point_pred, dtype=int) == 0))
    return lower <= rate <= upper


def action_family(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return 0
    if 1 <= action <= 7:
        return 1
    if 8 <= action <= 11:
        return 2
    if 12 <= action <= 14:
        return 3
    if 15 <= action <= 18:
        return 4
    return 0


def safe_predict_proba(model: object, frame: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(classes)), dtype=float)
    class_pos = {int(cls): idx for idx, cls in enumerate(classes)}
    for j, cls in enumerate(model.classes_):
        pos = class_pos.get(int(cls))
        if pos is not None:
            out[:, pos] = raw[:, j]
    return normalize_rows_safe(out)


def load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing clean anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    validate_submission_frame(anchor)
    return anchor


def add_point_context(train_df: pd.DataFrame, test_df: pd.DataFrame, anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.merge(
        anchor[["rally_uid", "actionId", "pointId"]],
        on="rally_uid",
        how="left",
        validate="one_to_one",
    )
    if test_out[["actionId", "pointId"]].isna().any().any():
        raise ValueError("Clean V261 anchor did not align one-to-one with test prefix rows.")

    # Train uses only observed prefix context. Test uses the fixed clean anchor
    # action/point context because that is the residual base being edited.
    train_out["v268_action_context"] = train_out["lag0_actionId"].astype(int)
    train_out["v268_action_family_context"] = train_out["lag0_action_family"].astype(int)
    train_out["v268_anchor_point_context"] = train_out["lag0_pointId"].astype(int)
    train_out["v268_anchor_point_depth"] = train_out["lag0_point_depth"].astype(int)
    train_out["v268_anchor_point_side"] = train_out["lag0_point_side"].astype(int)
    train_out["v268_point0_context"] = train_out["lag0_pointId"].astype(int).eq(0).astype(int)

    test_out = test_out.rename(columns={"actionId": "v268_action_context", "pointId": "v268_anchor_point_context"})
    test_out["v268_action_context"] = test_out["v268_action_context"].astype(int)
    test_out["v268_action_family_context"] = test_out["v268_action_context"].map(action_family)
    test_out["v268_anchor_point_context"] = test_out["v268_anchor_point_context"].astype(int)
    test_out["v268_anchor_point_depth"] = test_out["v268_anchor_point_context"].map(point_depth)
    test_out["v268_anchor_point_side"] = test_out["v268_anchor_point_context"].map(point_side)
    test_out["v268_point0_context"] = test_out["v268_anchor_point_context"].eq(0).astype(int)
    return train_out, test_out


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = add_questionnaire_columns(build_train_prefix_table(train_raw, MAX_LAG))
    test_df = add_questionnaire_columns(build_test_prefix_table(test_raw, MAX_LAG))
    anchor = load_anchor_submission()
    if len(test_df) != EXPECTED_ROWS:
        raise ValueError(f"test prefix rows={len(test_df)}, expected {EXPECTED_ROWS}")
    train_df, test_df = add_point_context(train_df, test_df, anchor)
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), anchor


def numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for col in train_df.columns:
        if col in BLOCKED_FEATURES or col not in test_df:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            features.append(col)
    leaked = [c for c in features if "PlayerId" in c or c in {"server_id", "receiver_id"}]
    if leaked:
        raise ValueError(f"Raw player feature leakage detected: {leaked}")
    return features


def clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].replace([np.inf, -np.inf], 0).fillna(0)


def fit_models(fold: int) -> list[object]:
    return [
        ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=4,
            class_weight="balanced",
            max_features="sqrt",
            random_state=2680 + fold,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=2780 + fold,
            n_jobs=1,
        ),
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.75,
                class_weight="balanced",
                max_iter=600,
                solver="lbfgs",
                random_state=2880 + fold,
            ),
        ),
    ]


def train_point_oof_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    y = train_df["next_pointId"].astype(int).to_numpy()
    oof = np.zeros((len(train_df), len(POINT_CLASSES)), dtype=float)
    test_sum = np.zeros((len(test_df), len(POINT_CLASSES)), dtype=float)
    folds: list[dict[str, int]] = []
    splitter = GroupKFold(n_splits=5)
    x_test = clean_matrix(test_df, features)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(train_df, y, groups=train_df["match"].astype(int))):
        x_fit = clean_matrix(train_df.iloc[fit_idx], features)
        x_valid = clean_matrix(train_df.iloc[valid_idx], features)
        valid_prob = np.zeros((len(valid_idx), len(POINT_CLASSES)), dtype=float)
        test_prob = np.zeros((len(test_df), len(POINT_CLASSES)), dtype=float)
        models = fit_models(fold)
        for model in models:
            model.fit(x_fit, y[fit_idx])
            valid_prob += safe_predict_proba(model, x_valid, POINT_CLASSES)
            test_prob += safe_predict_proba(model, x_test, POINT_CLASSES)
        oof[valid_idx] = normalize_rows_safe(valid_prob / len(models))
        test_sum += normalize_rows_safe(test_prob / len(models))
        folds.append({"fold": int(fold), "train_rows": int(len(fit_idx)), "valid_rows": int(len(valid_idx))})
        print(f"fold {fold}: train={len(fit_idx)} valid={len(valid_idx)} models={len(models)}")
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / len(folds)), folds


def apply_logit_adjustment(prob: np.ndarray, class_counts: np.ndarray, tau: float) -> np.ndarray:
    p = normalize_rows_safe(prob)
    adjusted = np.log(np.clip(p, 1e-12, 1.0)) + class_prior_logit_adjustment(class_counts, tau)[None, :]
    adjusted -= adjusted.max(axis=1, keepdims=True)
    return normalize_rows_safe(np.exp(adjusted))


def residual_labels(base_labels: np.ndarray, prob: np.ndarray, cap: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    p = normalize_rows_safe(prob)
    classes = np.asarray(POINT_CLASSES, dtype=int)
    top_pos = p.argmax(axis=1)
    top = classes[top_pos]
    base_pos = np.array([POINT_CLASSES.index(int(label)) if int(label) in POINT_CLASSES else 0 for label in base], dtype=int)
    gain = p[np.arange(len(p)), top_pos] - p[np.arange(len(p)), base_pos]
    eligible_score = np.where((top != base) & np.isfinite(gain) & (gain > 0), gain, -np.inf)
    if cap is None:
        changed = np.isfinite(eligible_score)
    else:
        changed = cap_by_score(eligible_score, cap)
    out = base.copy()
    out[changed] = top[changed]
    return out, changed, gain


def distribution_json(labels: np.ndarray) -> str:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(POINT_CLASSES))
    return pd.Series({str(i): int(v) for i, v in enumerate(counts) if v > 0}).to_json()


def class_f1(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    return float(f1_score(y_true == int(cls), y_pred == int(cls), zero_division=0))


def rare_point_mean_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([class_f1(y_true, y_pred, cls) for cls in RARE_POINT_CLASSES]))


def candidate_verdict(
    *,
    diagnostic: bool,
    ordinary_delta_vs_base: float,
    rare_f1: float,
    base_rare_f1: float,
    point0_rate_test: float,
    point0_added_rows: int,
    point_churn: float,
    cap: float | None,
) -> str:
    if diagnostic:
        if point0_added_rows > MAX_POINT0_ADDITIONS or not (0.24 <= point0_rate_test <= 0.31):
            return "DIAGNOSTIC_POINT0_INFLATION"
        return "DIAGNOSTIC_ONLY"
    if point0_added_rows > MAX_POINT0_ADDITIONS:
        return "REJECT_POINT0_INFLATION"
    if not (0.24 <= point0_rate_test <= 0.31):
        return "REJECT_POINT0_RATE"
    if cap is not None and point_churn > cap + (1.0 / EXPECTED_ROWS):
        return "REJECT_CHURN_TOO_HIGH"
    if ordinary_delta_vs_base > 0.0 and rare_f1 >= base_rare_f1:
        return "CANDIDATE_FOR_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def write_submission(path: Path, anchor: pd.DataFrame, point_pred: np.ndarray) -> None:
    out = anchor.copy()
    out["pointId"] = np.asarray(point_pred, dtype=int)
    out = out[EXPECTED_COLUMNS]
    validate_submission_frame(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.8f")
    if UPLOAD_DIR.exists():
        shutil.copy2(path, UPLOAD_DIR / path.name)


def add_candidate_record(
    *,
    records: list[dict[str, object]],
    submissions: list[str],
    candidate: str,
    path: Path,
    anchor: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    base_rare_f1: float,
    oof_pred: np.ndarray,
    test_pred: np.ndarray,
    test_changed: np.ndarray,
    cap: float | None,
    diagnostic: bool,
) -> None:
    test_base = anchor["pointId"].astype(int).to_numpy()
    point_macro_f1 = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_macro_f1 = float(f1_score(y, base_oof, labels=POINT_CLASSES, average="macro", zero_division=0))
    rare_f1 = rare_point_mean_f1(y, oof_pred)
    point0 = class_f1(y, oof_pred, 0)
    point_churn = float(np.mean(test_changed))
    point0_rate_test = float(np.mean(np.asarray(test_pred, dtype=int) == 0))
    point0_added_rows = int(np.sum((test_base != 0) & (np.asarray(test_pred, dtype=int) == 0)))
    point0_removed_rows = int(np.sum((test_base == 0) & (np.asarray(test_pred, dtype=int) != 0)))
    verdict = candidate_verdict(
        diagnostic=diagnostic,
        ordinary_delta_vs_base=point_macro_f1 - base_macro_f1,
        rare_f1=rare_f1,
        base_rare_f1=base_rare_f1,
        point0_rate_test=point0_rate_test,
        point0_added_rows=point0_added_rows,
        point_churn=point_churn,
        cap=cap,
    )
    write_submission(path, anchor, test_pred)
    submissions.append(path.name)
    records.append(
        {
            "candidate": candidate,
            "path": str(path),
            "ordinary_point_macro_f1": point_macro_f1,
            "ordinary_delta_vs_base": point_macro_f1 - base_macro_f1,
            "rare_point_mean_f1": rare_f1,
            "point0_f1": point0,
            "point_churn": point_churn,
            "point0_rate_test": point0_rate_test,
            "point0_added_rows": point0_added_rows,
            "verdict": verdict,
            "cap": float(cap) if cap is not None else np.nan,
            "changed_rows": int(np.sum(test_changed)),
            "point0_removed_rows": point0_removed_rows,
            "point0_rate_ok": point0_rate_ok(test_pred),
            "test_point_distribution": distribution_json(test_pred),
        }
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, anchor = build_frames()
    features = numeric_features(train_df, test_df)
    y = train_df["next_pointId"].astype(int).to_numpy()
    class_counts = np.bincount(y, minlength=len(POINT_CLASSES))
    print(f"train rows={len(train_df)} test rows={len(test_df)} features={len(features)}")
    model_oof, model_test, folds = train_point_oof_test(train_df, test_df, features)

    base_oof = train_df["lag0_pointId"].astype(int).clip(0, len(POINT_CLASSES) - 1).to_numpy()
    base_macro_f1 = float(f1_score(y, base_oof, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_rare_f1 = rare_point_mean_f1(y, base_oof)
    test_base = anchor["pointId"].astype(int).to_numpy()

    records: list[dict[str, object]] = []
    submissions: list[str] = []
    configs = [
        ("v268_point_logadj_tau0p10_cap0p0025", "submission_v268_point_logadj_tau0p10_cap0p0025__v173action_r121server.csv", 0.10, 0.0025),
        ("v268_point_logadj_tau0p20_cap0p005", "submission_v268_point_logadj_tau0p20_cap0p005__v173action_r121server.csv", 0.20, 0.005),
        ("v268_point_logadj_tau0p35_cap0p010", "submission_v268_point_logadj_tau0p35_cap0p010__v173action_r121server.csv", 0.35, 0.010),
        ("v268_point_logadj_tau0p35_cap0p015", "submission_v268_point_logadj_tau0p35_cap0p015__v173action_r121server.csv", 0.35, 0.015),
    ]
    for candidate, filename, tau, cap in configs:
        oof_prob = apply_logit_adjustment(model_oof, class_counts, tau)
        test_prob = apply_logit_adjustment(model_test, class_counts, tau)
        oof_pred, _, _ = residual_labels(base_oof, oof_prob, cap)
        test_pred, test_changed, _ = residual_labels(test_base, test_prob, cap)
        add_candidate_record(
            records=records,
            submissions=submissions,
            candidate=candidate,
            path=OUTDIR / filename,
            anchor=anchor,
            y=y,
            base_oof=base_oof,
            base_rare_f1=base_rare_f1,
            oof_pred=oof_pred,
            test_pred=test_pred,
            test_changed=test_changed,
            cap=cap,
            diagnostic=False,
        )

    raw_oof_pred = np.asarray(POINT_CLASSES, dtype=int)[model_oof.argmax(axis=1)]
    raw_test_pred = np.asarray(POINT_CLASSES, dtype=int)[model_test.argmax(axis=1)]
    raw_changed = raw_test_pred != test_base
    add_candidate_record(
        records=records,
        submissions=submissions,
        candidate="v268_point_balanced_raw_diagnostic",
        path=OUTDIR / "submission_v268_point_balanced_raw_diagnostic__v173action_r121server.csv",
        anchor=anchor,
        y=y,
        base_oof=base_oof,
        base_rare_f1=base_rare_f1,
        oof_pred=raw_oof_pred,
        test_pred=raw_test_pred,
        test_changed=raw_changed,
        cap=None,
        diagnostic=True,
    )

    search = pd.DataFrame(records)
    ordered_cols = [
        "candidate",
        "path",
        "ordinary_point_macro_f1",
        "ordinary_delta_vs_base",
        "rare_point_mean_f1",
        "point0_f1",
        "point_churn",
        "point0_rate_test",
        "point0_added_rows",
        "verdict",
    ]
    search = search[ordered_cols + [c for c in search.columns if c not in ordered_cols]]
    search.to_csv(OUTDIR / "v268_point_search.csv", index=False)

    reviewable = search[search["verdict"].eq("CANDIDATE_FOR_REVIEW")]
    if reviewable.empty:
        best = search.sort_values(["ordinary_delta_vs_base", "point0_added_rows", "point_churn"], ascending=[False, True, True]).iloc[0]
    else:
        best = reviewable.sort_values(["ordinary_delta_vs_base", "point_churn"], ascending=[False, True]).iloc[0]

    report_lines = [
        "# V268 Macro-F1 Point Residual",
        "",
        "Clean point-only branch. Fixed action/server anchor:",
        "",
        "```text",
        str(ANCHOR_PATH),
        "action = V173",
        "server = R121",
        "changed field = pointId only",
        "```",
        "",
        "## Summary",
        "",
        f"- Train prefix rows: `{len(train_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Numeric feature count: `{len(features)}`",
        f"- Base OOF point Macro-F1 proxy: `{base_macro_f1:.6f}`",
        f"- Base rare point mean F1: `{base_rare_f1:.6f}`",
        f"- Anchor point0 rate: `{float(np.mean(test_base == 0)):.6f}`",
        f"- Best row: `{best['candidate']}` / verdict `{best['verdict']}`",
        "",
        "## Candidates",
        "",
    ]
    for row in search.to_dict("records"):
        report_lines.append(
            f"- `{row['candidate']}`: OOF={float(row['ordinary_point_macro_f1']):.6f}, "
            f"delta={float(row['ordinary_delta_vs_base']):.6f}, "
            f"churn={float(row['point_churn']):.6f}, "
            f"point0_rate={float(row['point0_rate_test']):.6f}, "
            f"point0_added={int(row['point0_added_rows'])}, verdict=`{row['verdict']}`"
        )
    report_lines.extend(
        [
            "",
            "## Policy Checks",
            "",
            "- No TTMATCH inputs are read.",
            "- No old-server artifacts are read.",
            "- No automatic upload is performed; submissions are copied only to the local upload candidate folder when it exists.",
            f"- Point0 additions above `{MAX_POINT0_ADDITIONS}` rows are rejected or marked diagnostic.",
        ]
    )
    (OUTDIR / "v268_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTDIR / "v268_run_summary.json").write_text(
        json.dumps(
            {
                "branch": "v268_macro_f1_point_residual",
                "outdir": str(OUTDIR),
                "anchor": str(ANCHOR_PATH),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "feature_count": int(len(features)),
                "base_point_macro_f1": base_macro_f1,
                "best_candidate": best.to_dict(),
                "generated_submissions": submissions,
                "copied_to_upload_candidates": bool(UPLOAD_DIR.exists()),
                "folds": folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "generated_submissions": submissions,
                "best_candidate": str(best["candidate"]),
                "best_verdict": str(best["verdict"]),
                "best_point0_rate_test": float(best["point0_rate_test"]),
                "best_point_churn": float(best["point_churn"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
