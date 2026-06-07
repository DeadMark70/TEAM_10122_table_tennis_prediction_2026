"""V218 weak-class action booster.

V217 found useful low-churn action changes by expected macro-F1 utility.  V218
keeps the same candidate pool and fixed point/server anchors, but selects
candidate rows directly with class-specific weights so weak action classes can
be probed without broad action replacement.

Point remains V188 cap5 and server remains R121.  No external rows and no
TTMATCH are read.
"""

from __future__ import annotations

import __main__
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import analysis_v217_macro_f1_utility_reranker as v217
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v209_action_selector_reranker import (
    add_probability_features,
    action_point_compatibility,
    build_action_candidate_frame,
    distill_v173_soft_anchor,
    load_point_anchor_labels,
    rebuild_r166_best_action,
    rebuild_r184_sources,
    source_probs_for_selector,
)
from analysis_v216_terminal_action_tuner import (
    POINT_ANCHOR,
    SERVER_ANCHOR,
    build_terminal_action_candidate,
)
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v218_action_weakclass_booster")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v218_action_weakclass_booster.py")

WEAK_CLASSES = {0, 3, 5, 7, 8, 9, 12, 14}
STYLE_RARE_CLASSES = {8, 9, 12, 14}
ATTACK_REPAIR_CLASSES = {3, 5, 7}
ZERO_SAFE_CLASSES = {0, 13}

SCHEMES = [
    {
        "name": "v218_weak_all_cap0p002",
        "cap": 0.002,
        "allowed": WEAK_CLASSES,
        "weights": {0: 1.45, 3: 1.35, 5: 1.25, 7: 1.30, 8: 1.75, 9: 1.55, 12: 1.20, 14: 1.80},
        "per_class_cap": {0: 5, 3: 6, 5: 5, 7: 5, 8: 4, 9: 5, 12: 5, 14: 4},
    },
    {
        "name": "v218_weak_all_cap0p005",
        "cap": 0.005,
        "allowed": WEAK_CLASSES,
        "weights": {0: 1.45, 3: 1.35, 5: 1.25, 7: 1.30, 8: 1.75, 9: 1.55, 12: 1.20, 14: 1.80},
        "per_class_cap": {0: 10, 3: 10, 5: 8, 7: 8, 8: 7, 9: 8, 12: 8, 14: 7},
    },
    {
        "name": "v218_style_rare_cap0p003",
        "cap": 0.003,
        "allowed": STYLE_RARE_CLASSES,
        "weights": {8: 2.10, 9: 1.75, 12: 1.25, 14: 2.20},
        "per_class_cap": {8: 8, 9: 8, 12: 5, 14: 6},
    },
    {
        "name": "v218_attack_repair_cap0p005",
        "cap": 0.005,
        "allowed": ATTACK_REPAIR_CLASSES,
        "weights": {3: 1.45, 5: 1.35, 7: 1.35},
        "per_class_cap": {3: 12, 5: 10, 7: 10},
    },
    {
        "name": "v218_zero_safe_cap0p003",
        "cap": 0.003,
        "allowed": ZERO_SAFE_CLASSES,
        "weights": {0: 1.80, 13: 1.10},
        "per_class_cap": {0: 8, 13: 5},
    },
]


def macro_f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def apply_class_utility_weights(
    frame: pd.DataFrame,
    class_weights: dict[int, float],
    default_weight: float = 0.35,
) -> pd.DataFrame:
    """Add weighted_utility without changing candidate ordering within a class."""
    out = frame.copy()
    weights = out["candidate_action"].astype(int).map(lambda x: float(class_weights.get(int(x), default_weight)))
    out["weighted_utility"] = out["utility"].astype(float) * weights.astype(float)
    return out


def select_weighted_candidate_changes(
    anchor_labels: np.ndarray,
    candidate_frame: pd.DataFrame,
    total_cap: float,
    per_class_cap: dict[int, int] | None = None,
    allowed_classes: set[int] | None = None,
    min_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select at most one weighted candidate per row with optional target caps."""
    anchor = np.asarray(anchor_labels, dtype=int)
    selected = np.zeros(len(anchor), dtype=bool)
    out = anchor.copy()
    max_rows = int(np.floor(len(anchor) * float(total_cap)))
    if max_rows <= 0 or candidate_frame.empty:
        return out, selected

    frame = candidate_frame.copy()
    frame = frame[frame["weighted_utility"].astype(float) > float(min_score)]
    if "anchor_action" in frame.columns:
        frame = frame[frame["candidate_action"].astype(int) != frame["anchor_action"].astype(int)]
    else:
        row_id = frame["row_id"].astype(int).to_numpy()
        frame = frame[frame["candidate_action"].astype(int).to_numpy() != anchor[row_id]]
    if allowed_classes is not None:
        allowed = {int(x) for x in allowed_classes}
        frame = frame[frame["candidate_action"].astype(int).isin(allowed)]
    if frame.empty:
        return out, selected

    if "utility" not in frame.columns:
        frame["utility"] = frame["weighted_utility"]
    frame = frame.sort_values(["weighted_utility", "utility"], ascending=[False, False])
    class_counts: dict[int, int] = {}
    per_class_cap = {int(k): int(v) for k, v in (per_class_cap or {}).items()}
    for row in frame.itertuples(index=False):
        rid = int(row.row_id)
        cand = int(row.candidate_action)
        if selected[rid]:
            continue
        if per_class_cap and class_counts.get(cand, 0) >= per_class_cap.get(cand, max_rows):
            continue
        selected[rid] = True
        out[rid] = cand
        class_counts[cand] = class_counts.get(cand, 0) + 1
        if int(selected.sum()) >= max_rows:
            break
    return out, selected


def class_f1_table(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for label in ACTION_CLASSES:
        f_anchor = f1_score(y, anchor, labels=[label], average="macro", zero_division=0)
        f_pred = f1_score(y, pred, labels=[label], average="macro", zero_division=0)
        rows.append(
            {
                "action": int(label),
                "support": int((np.asarray(y) == int(label)).sum()),
                "anchor_f1": float(f_anchor),
                "candidate_f1": float(f_pred),
                "delta": float(f_pred - f_anchor),
            }
        )
    return pd.DataFrame(rows)


def candidate_utility_frames(
    rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y: np.ndarray,
    sources_oof: dict[str, np.ndarray],
    sources_test: dict[str, np.ndarray],
    probs_oof: dict[str, np.ndarray],
    probs_test: dict[str, np.ndarray],
    point_oof: np.ndarray,
    point_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    base_frame = build_action_candidate_frame(rows, sources_oof, truth=y, anchor_name="v173")
    test_frame = build_action_candidate_frame(test_rows, sources_test, truth=None, anchor_name="v173")
    oof_parts = []
    metrics = []

    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_rows = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_rows_mask = ~valid_rows
        train_ids = set(np.where(train_rows_mask)[0])
        valid_ids = set(np.where(valid_rows)[0])
        compat = action_point_compatibility(y[train_rows_mask], point_oof[train_rows_mask], smoothing=1.0)
        train = v217.add_features(base_frame[base_frame["row_id"].isin(train_ids)].copy(), probs_oof, point_oof, compat)
        valid = v217.add_features(base_frame[base_frame["row_id"].isin(valid_ids)].copy(), probs_oof, point_oof, compat)
        x_train = v217.selector_features(train)
        clf = v217.train_correctness_model(x_train, train["is_correct"].astype(int).to_numpy())
        x_valid = v217.align_columns(v217.selector_features(valid), list(x_train.columns))
        p_valid = clf.predict_proba(x_valid)[:, 1]
        valid_actions = valid["candidate_action"].astype(int).to_numpy()
        valid_rows_idx = valid["row_id"].astype(int).to_numpy()
        gain, loss = v217.row_delta_tables(y, sources_oof["v173"], valid_rows_idx, valid_actions)
        util = v217.expected_macro_f1_delta(p_valid, gain, loss)
        part = valid[["row_id", "source", "candidate_action", "anchor_action", "differs_anchor"]].copy()
        part["p_correct"] = p_valid
        part["utility"] = util
        oof_parts.append(part)
        y_valid = valid["is_correct"].astype(int).to_numpy()
        metrics.append(
            {
                "fold": int(fold),
                "auc": float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) > 1 else np.nan,
                "valid_candidate_rows": int(len(valid)),
            }
        )

    compat_full = action_point_compatibility(y, point_oof, smoothing=1.0)
    train = v217.add_features(base_frame.copy(), probs_oof, point_oof, compat_full)
    test = v217.add_features(test_frame.copy(), probs_test, point_test, compat_full)
    x_train = v217.selector_features(train)
    clf = v217.train_correctness_model(x_train, train["is_correct"].astype(int).to_numpy())
    p_test = clf.predict_proba(v217.align_columns(v217.selector_features(test), list(x_train.columns)))[:, 1]
    test_rows_idx = test["row_id"].astype(int).to_numpy()
    test_actions = test["candidate_action"].astype(int).to_numpy()
    class_gain = np.full(len(point_test), 1.0 / len(ACTION_CLASSES), dtype=float)
    class_loss = np.full(len(point_test), 0.5 / len(ACTION_CLASSES), dtype=float)
    util_test = v217.expected_macro_f1_delta(p_test, class_gain[test_rows_idx], class_loss[test_rows_idx])
    test_out = test[["row_id", "source", "candidate_action", "anchor_action", "differs_anchor"]].copy()
    test_out["p_correct"] = p_test
    test_out["utility"] = util_test
    return pd.concat(oof_parts, ignore_index=True), test_out, metrics


def write_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    # Older pickle files were produced by scripts run as __main__.
    __main__.V3Tuning = v217.V3Tuning
    __main__.GrUTuning = v217.GrUTuning
    __main__.TransformerTuning = v217.TransformerTuning

    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    point_oof, point_test = load_point_anchor_labels(data, point)
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = point["actionId"].astype(int).to_numpy()
    v173_prob_oof, v173_prob_test, distill_metrics = distill_v173_soft_anchor(data, v173_oof, v173_test)
    r166_oof, r166_test, r166_prob_oof, r166_prob_test = rebuild_r166_best_action(state["rows"], state["test_rows"])
    r184_oof, r184_test = rebuild_r184_sources(state, point)
    v216_oof, _ = build_terminal_action_candidate(v173_oof, point_oof, v173_prob_oof)
    v216_test, _ = build_terminal_action_candidate(v173_test, point["pointId"].astype(int).to_numpy(), v173_prob_test)

    sources_oof = {"v173": v173_oof, "r166": r166_oof, **r184_oof, "v216_terminal": v216_oof}
    sources_test = {"v173": v173_test, "r166": r166_test, **r184_test, "v216_terminal": v216_test}
    probs_oof = source_probs_for_selector(v173_prob_oof, r166_prob_oof, v173_prob_oof)
    probs_oof["utility_model"] = probs_oof.pop("v208")
    probs_test = source_probs_for_selector(v173_prob_test, r166_prob_test, v173_prob_test)
    probs_test["utility_model"] = probs_test.pop("v208")

    oof_frame, test_frame, fold_metrics = candidate_utility_frames(
        data["rows"],
        state["test_rows"],
        y,
        sources_oof,
        sources_test,
        probs_oof,
        probs_test,
        point_oof,
        point_test,
    )
    base_score = macro_f1_score(y, v173_oof)
    records = [
        {
            "candidate": "v173_anchor",
            "action_macro_f1": base_score,
            "delta_vs_v173_anchor": 0.0,
            "action_churn_vs_v173_anchor": 0.0,
            "changed_rows": 0,
            "test_changed_rows": 0,
        }
    ]
    generated = []
    class_tables = []
    for scheme in SCHEMES:
        weighted_oof = apply_class_utility_weights(oof_frame, scheme["weights"], default_weight=0.20)
        weighted_test = apply_class_utility_weights(test_frame, scheme["weights"], default_weight=0.20)
        pred, changed = select_weighted_candidate_changes(
            v173_oof,
            weighted_oof,
            total_cap=float(scheme["cap"]),
            per_class_cap=scheme["per_class_cap"],
            allowed_classes=set(scheme["allowed"]),
            min_score=0.0,
        )
        test_pred, test_changed = select_weighted_candidate_changes(
            v173_test,
            weighted_test,
            total_cap=float(scheme["cap"]),
            per_class_cap=scheme["per_class_cap"],
            allowed_classes=set(scheme["allowed"]),
            min_score=0.0,
        )
        score = macro_f1_score(y, pred)
        rec = {
            "candidate": scheme["name"],
            "action_macro_f1": score,
            "delta_vs_v173_anchor": score - base_score,
            "action_churn_vs_v173_anchor": float(np.mean(pred != v173_oof)),
            "changed_rows": int(changed.sum()),
            "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
            "test_changed_rows": int(test_changed.sum()),
            "changed_target_classes": json.dumps(pd.Series(pred[changed]).value_counts().sort_index().to_dict()),
        }
        records.append(rec)
        class_table = class_f1_table(y, v173_oof, pred)
        class_table.insert(0, "candidate", scheme["name"])
        class_tables.append(class_table)
        info = write_submission(f"submission_{scheme['name']}__pv188cap5__sr121.csv", test_pred, point, server)
        info.update(rec)
        generated.append(info)

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v218_action_search.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v218_selector_fold_metrics.csv", index=False)
    pd.concat(class_tables, ignore_index=True).to_csv(OUTDIR / "v218_class_f1_delta.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(8).to_dict(orient="records"),
        "notes": [
            "V218 selects candidate rows directly with class-specific utility weights.",
            "Weak classes are targeted without replacing the full V173 action anchor.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v218_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v218_report.md").write_text(
        "# V218 Weak-Class Action Booster\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v218_action_weakclass_booster.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
