"""V261 action-conditioned point residual search.

This script keeps the current public anchor fixed for action/server and writes
only local point-residual candidates.  Test submissions are capped edits to:

  action = V173
  point  = V188 r186_w005 alpha=0.05 cap=0.05
  server = R121

The fold-safe point model uses current numeric prefix features plus fold-safe
action/terminal proxy features.  It does not raw-replace point labels; each
submission changes only the top-scored residual rows under the requested cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)


OUTDIR = Path("v261_action_conditioned_point_residual")
ANCHOR_SUBMISSION = Path("upload_candidates_20260519/submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv")
V188_SEARCH = Path("v188_point_intent_gru/v188_search.csv")
CAPS = [0.01, 0.02, 0.03, 0.05]
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


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


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 3:
        return 1
    if 4 <= point <= 6:
        return 2
    if 7 <= point <= 9:
        return 3
    return 0


def point_side(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 9:
        return ((point - 1) % 3) + 1
    return 0


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero, :] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def predict_full(model: ExtraTreesClassifier, frame: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(classes)), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, classes.index(int(cls))] = raw[:, j]
    return normalize_rows_safe(out)


def load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing fixed anchor submission: {ANCHOR_SUBMISSION}")
    sub = pd.read_csv(ANCHOR_SUBMISSION)
    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Anchor columns differ from expected submission schema: {list(sub.columns)}")
    return sub


def numeric_feature_columns(df: pd.DataFrame, *, include_proxy: bool) -> list[str]:
    blocked = {
        "rally_uid",
        "match",
        "next_actionId",
        "next_pointId",
        "next_is_terminal",
        "serverGetPoint",
        "fold",
    }
    if not include_proxy:
        blocked.update(
            {
                "v261_action_proxy",
                "v261_action_family",
                "v261_terminal_proxy",
                "v261_anchor_point",
                "v261_anchor_depth",
                "v261_anchor_side",
            }
        )
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def add_geometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag0_point_depth"] = out["lag0_pointId"].astype(int).map(point_depth)
    out["lag0_point_side"] = out["lag0_pointId"].astype(int).map(point_side)
    out["lag0_action_family"] = out["lag0_actionId"].astype(int).map(action_family)
    return out


def add_test_anchor_columns(test_df: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    out = test_df.merge(anchor[["rally_uid", "actionId", "pointId"]], on="rally_uid", how="left", validate="one_to_one")
    if out[["actionId", "pointId"]].isna().any().any():
        raise ValueError("Anchor submission did not align to test prefix rows.")
    out = out.rename(columns={"actionId": "v261_action_proxy", "pointId": "v261_anchor_point"})
    out["v261_action_proxy"] = out["v261_action_proxy"].astype(int)
    out["v261_anchor_point"] = out["v261_anchor_point"].astype(int)
    out["v261_action_family"] = out["v261_action_proxy"].map(action_family)
    out["v261_anchor_depth"] = out["v261_anchor_point"].map(point_depth)
    out["v261_anchor_side"] = out["v261_anchor_point"].map(point_side)
    return out


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = build_train_prefix_table(train_raw, 6)
    test_df = build_test_prefix_table(test_raw, 6)
    train_df = add_geometry_columns(train_df)
    test_df = add_geometry_columns(test_df)

    splitter = GroupKFold(n_splits=5)
    train_df["fold"] = -1
    rally_meta = train_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    for fold, (_, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"])):
        valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"].astype(int))
        train_df.loc[train_df["rally_uid"].isin(valid_rallies), "fold"] = fold
    if train_df["fold"].lt(0).any():
        raise RuntimeError("Fold assignment failed.")

    anchor = load_anchor_submission()
    test_df = add_test_anchor_columns(test_df, anchor)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), anchor


def fit_classifier(seed: int, n_estimators: int, min_samples_leaf: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def train_oof_prob(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: np.ndarray,
    classes: list[int],
    features: list[str],
    *,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof = np.zeros((len(train_df), len(classes)), dtype=float)
    test_sum = np.zeros((len(test_df), len(classes)), dtype=float)
    fold_rows: list[dict] = []
    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        model = fit_classifier(seed + int(fold), n_estimators, min_samples_leaf)
        model.fit(train_df.loc[train, features].fillna(0), target[train])
        oof[valid] = predict_full(model, train_df.loc[valid, features].fillna(0), classes)
        test_sum += predict_full(model, test_df.loc[:, features].fillna(0), classes)
        fold_rows.append({"fold": int(fold), "train_rows": int(train.sum()), "valid_rows": int(valid.sum())})
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / len(fold_rows)), fold_rows


def add_foldsafe_proxy_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    base_features = numeric_feature_columns(train_df, include_proxy=False)
    for col in base_features:
        if col not in test_df:
            test_df[col] = 0

    action_oof, action_test, action_folds = train_oof_prob(
        train_df,
        test_df,
        train_df["next_actionId"].astype(int).to_numpy(),
        list(range(19)),
        base_features,
        seed=2610,
        n_estimators=120,
        min_samples_leaf=5,
    )
    terminal_oof, terminal_test, terminal_folds = train_oof_prob(
        train_df,
        test_df,
        train_df["next_pointId"].eq(0).astype(int).to_numpy(),
        [0, 1],
        base_features,
        seed=2710,
        n_estimators=120,
        min_samples_leaf=8,
    )

    out_train = train_df.copy()
    out_test = test_df.copy()
    out_train["v261_action_proxy"] = action_oof.argmax(axis=1).astype(int)
    out_train["v261_action_family"] = out_train["v261_action_proxy"].map(action_family)
    out_train["v261_terminal_proxy"] = terminal_oof[:, 1]
    out_train["v261_anchor_point"] = -1
    out_train["v261_anchor_depth"] = -1
    out_train["v261_anchor_side"] = -1
    out_test["v261_terminal_proxy"] = terminal_test[:, 1]
    return out_train, out_test, [{"stage": "action_proxy", **r} for r in action_folds] + [{"stage": "terminal_proxy", **r} for r in terminal_folds]


def capped_residual_labels(base_labels: np.ndarray, prob: np.ndarray, max_churn: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    p = normalize_rows_safe(prob)
    top = p.argmax(axis=1).astype(int)
    candidate_score = p[np.arange(len(p)), top] - p[np.arange(len(p)), np.clip(base, 0, p.shape[1] - 1)]
    eligible = (top != base) & np.isfinite(candidate_score) & (candidate_score > 0)
    budget = int(np.floor(len(base) * float(max_churn)))
    changed = np.zeros(len(base), dtype=bool)
    if budget > 0 and eligible.any():
        eligible_idx = np.where(eligible)[0]
        order = eligible_idx[np.argsort(-candidate_score[eligible_idx])]
        changed[order[: min(budget, len(order))]] = True
    out = base.copy()
    out[changed] = top[changed]
    return out, changed, candidate_score


def distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=10)
    return {str(i): int(v) for i, v in enumerate(counts) if v > 0}


def v188_anchor_metric() -> dict:
    if not V188_SEARCH.exists():
        return {"candidate": "v188_r186_w005_a0p05_cap0p05", "point_macro_f1": float("nan")}
    search = pd.read_csv(V188_SEARCH)
    row = search[search["candidate"].eq("v188_r186_w005_a0p05_cap0p05")]
    if row.empty:
        return {"candidate": "v188_r186_w005_a0p05_cap0p05", "point_macro_f1": float("nan")}
    return row.iloc[0].to_dict()


def write_submission(anchor: pd.DataFrame, point: np.ndarray, cap: float) -> dict:
    name = f"submission_v261_cap{str(cap).replace('.', 'p')}__v173action_r121server.csv"
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[EXPECTED_COLUMNS]
    if len(out) != 1845:
        raise ValueError(f"{name} has {len(out)} rows, expected 1845")
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return {"submission": name, "path": str(path)}


def verdict_from_best_delta(best_delta: float, max_test_churn: float, direct_anchor_oof: bool) -> str:
    if direct_anchor_oof and best_delta >= 0.003 and max_test_churn <= 0.05:
        return "CANDIDATE_FOR_PUBLIC_PROBE"
    if best_delta > 0:
        return "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    train_df, test_df, anchor = build_frames()
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns(train_df, test_df)
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0

    point_features = numeric_feature_columns(train_df, include_proxy=True)
    point_features = [c for c in point_features if c in test_df]
    y = train_df["next_pointId"].astype(int).to_numpy()

    base_features = numeric_feature_columns(train_df, include_proxy=False)
    base_features = [c for c in base_features if c in test_df]
    base_oof_prob, _, base_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        base_features,
        seed=2810,
        n_estimators=160,
        min_samples_leaf=6,
    )
    model_oof_prob, model_test_prob, point_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        point_features,
        seed=2910,
        n_estimators=220,
        min_samples_leaf=4,
    )

    base_oof_pred = base_oof_prob.argmax(axis=1).astype(int)
    base_proxy_f1 = float(f1_score(y, base_oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    raw_model_pred = model_oof_prob.argmax(axis=1).astype(int)
    raw_model_f1 = float(f1_score(y, raw_model_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    anchor_metric = v188_anchor_metric()
    known_anchor_f1 = float(anchor_metric.get("point_macro_f1", float("nan")))

    records: list[dict] = [
        {
            "candidate": "v188_r186_w005_a0p05_cap0p05_known_anchor",
            "cap": 0.05,
            "point_macro_f1": known_anchor_f1,
            "delta_vs_base_proxy": float("nan"),
            "delta_vs_v188_known_anchor": 0.0,
            "oof_churn_vs_base_proxy": float("nan"),
            "oof_changed_rows": int(anchor_metric.get("changed_rows", 0)) if pd.notna(anchor_metric.get("changed_rows", np.nan)) else 0,
            "test_churn_vs_current_anchor": 0.0,
            "test_changed_rows": 0,
            "point0_rate": float("nan"),
            "test_point_distribution": json.dumps(distribution(anchor["pointId"].astype(int).to_numpy()), sort_keys=True),
        },
        {
            "candidate": "v261_action_conditioned_raw_model_diagnostic",
            "cap": 1.0,
            "point_macro_f1": raw_model_f1,
            "delta_vs_base_proxy": raw_model_f1 - base_proxy_f1,
            "delta_vs_v188_known_anchor": raw_model_f1 - known_anchor_f1 if np.isfinite(known_anchor_f1) else float("nan"),
            "oof_churn_vs_base_proxy": float(np.mean(raw_model_pred != base_oof_pred)),
            "oof_changed_rows": int(np.sum(raw_model_pred != base_oof_pred)),
            "test_churn_vs_current_anchor": float("nan"),
            "test_changed_rows": 0,
            "point0_rate": float(np.mean(raw_model_pred == 0)),
            "test_point_distribution": "{}",
        },
    ]

    submissions: list[dict] = []
    test_base = anchor["pointId"].astype(int).to_numpy()
    for cap in CAPS:
        oof_pred, oof_changed, _ = capped_residual_labels(base_oof_pred, model_oof_prob, cap)
        test_pred, test_changed, _ = capped_residual_labels(test_base, model_test_prob, cap)
        score = float(f1_score(y, oof_pred, labels=POINT_CLASSES, average="macro", zero_division=0))
        info = write_submission(anchor, test_pred, cap)
        rec = {
            "candidate": f"v261_action_conditioned_cap{str(cap).replace('.', 'p')}",
            "cap": cap,
            "point_macro_f1": score,
            "delta_vs_base_proxy": score - base_proxy_f1,
            "delta_vs_v188_known_anchor": score - known_anchor_f1 if np.isfinite(known_anchor_f1) else float("nan"),
            "oof_churn_vs_base_proxy": float(np.mean(oof_changed)),
            "oof_changed_rows": int(oof_changed.sum()),
            "test_churn_vs_current_anchor": float(np.mean(test_changed)),
            "test_changed_rows": int(test_changed.sum()),
            "point0_rate": float(np.mean(oof_pred == 0)),
            "test_point_distribution": json.dumps(distribution(test_pred), sort_keys=True),
        }
        rec.update(info)
        records.append(rec)
        submissions.append(rec)

    search = pd.DataFrame(records)
    search = search.sort_values(["delta_vs_v188_known_anchor", "delta_vs_base_proxy"], ascending=[False, False]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v261_point_search.csv", index=False)

    candidate_rows = search[search["candidate"].str.startswith("v261_action_conditioned_cap")]
    best = candidate_rows.iloc[0].to_dict()
    best_delta = float(best["delta_vs_v188_known_anchor"]) if np.isfinite(float(best["delta_vs_v188_known_anchor"])) else float(best["delta_vs_base_proxy"])
    verdict = verdict_from_best_delta(best_delta, float(candidate_rows["test_churn_vs_current_anchor"].max()), direct_anchor_oof=False)
    upload_recommendation = "do_not_upload_keep_current_anchor_until_direct_v188_oof_gate" if verdict != "CANDIDATE_FOR_PUBLIC_PROBE" else "review_public_probe_candidate_before_upload"

    report = {
        "verdict": verdict,
        "upload_recommendation": upload_recommendation,
        "fixed_anchor": {
            "action": "V173/current anchor",
            "point": "V188 r186_w005 alpha=0.05 cap=0.05/current anchor",
            "server": "R121/current anchor",
            "anchor_submission": str(ANCHOR_SUBMISSION),
        },
        "known_v188_cap5_metric": anchor_metric,
        "base_proxy_point_macro_f1": base_proxy_f1,
        "raw_action_conditioned_model_point_macro_f1": raw_model_f1,
        "best_candidate": best,
        "submissions": submissions,
        "folds": proxy_folds + [{"stage": "base_point_proxy", **r} for r in base_folds] + [{"stage": "action_conditioned_point", **r} for r in point_folds],
        "notes": [
            "Outputs are local-only under v261_action_conditioned_point_residual.",
            "No upload_candidates or submissions/selected files are written.",
            "Point labels are residual-capped edits to the fixed V188 cap5 test anchor.",
            "OOF residual metrics use a fold-safe tabular base proxy; the known V188 cap5 OOF score is included as the current-anchor reference.",
            "The heavy V173 context loader is avoided; test action features are the fixed anchor V173 actions, while train action features are fold-safe proxies.",
        ],
    }
    (OUTDIR / "v261_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v261_report.md").write_text(
        "# V261 Action-Conditioned Point Residual\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best candidate: `{best['candidate']}`\n"
        f"- Best delta vs V188 known anchor: `{float(best['delta_vs_v188_known_anchor']):.6f}`\n"
        f"- Best delta vs fold-safe base proxy: `{float(best['delta_vs_base_proxy']):.6f}`\n"
        f"- Test churn: `{float(best['test_churn_vs_current_anchor']):.6f}` ({int(best['test_changed_rows'])} rows)\n"
        f"- Upload recommendation: `{upload_recommendation}`\n\n"
        "## Caps\n\n"
        + "\n".join(
            f"- cap `{float(r['cap']):.2f}`: OOF delta vs anchor `{float(r['delta_vs_v188_known_anchor']):.6f}`, "
            f"OOF changed `{int(r['oof_changed_rows'])}`, test changed `{int(r['test_changed_rows'])}`"
            for r in submissions
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {note}" for note in report["notes"])
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "outdir": str(OUTDIR),
        "verdict": verdict,
        "best_candidate": best["candidate"],
        "best_delta_vs_v188_known_anchor": best["delta_vs_v188_known_anchor"],
        "best_delta_vs_base_proxy": best["delta_vs_base_proxy"],
        "generated_submissions": [s["submission"] for s in submissions],
        "upload_recommendation": upload_recommendation,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
