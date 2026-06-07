"""V204/V205 TT-ShuttleNet residual selector and point0 calibrator.

V203 showed that ShuttleNet-style type/area separation has local point signal,
but raw test argmax collapses to point0 and the direct cap5 residual missed
Public.  This script does not train a larger encoder.  It learns a row-level
selector over V203 residual candidates:

  input: base point probabilities, TT-ShuttleNet probabilities, and context
  target: whether the candidate residual change is correct in OOF
  output: trust-ranked capped residual submissions

Submissions keep action=V173 and server=R121.  Raw neural argmax is diagnostic
only.  TTMATCH and ShuttleSet external rows are not read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_v188_point_intent_gru import row_log_blend
from baseline_lgbm import POINT_CLASSES
from train_v203_tt_shuttlenet import (
    OUTDIR as V203_OUTDIR,
    apply_point_residual,
    distribution,
    long_attack_gate,
    prepare_data,
    run_v203,
)


OUTDIR = Path("v204_ttshuttle_residual_selector")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v204_ttshuttle_residual_selector.py")
ALPHA = 0.075
CAPS = [0.01, 0.02, 0.03, 0.05]


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


def point0_transition_kind(base_label: int, neural_label: int) -> str:
    base = int(base_label)
    neural = int(neural_label)
    if base == neural:
        return "unchanged"
    if neural == 0:
        return "to_point0"
    if base == 0:
        return "from_point0"
    return "nonterminal"


def _label_score(prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return prob[np.arange(len(prob)), np.asarray(labels, dtype=int)]


def _margin(prob: np.ndarray) -> np.ndarray:
    part = np.partition(np.asarray(prob, dtype=float), -2, axis=1)
    return part[:, -1] - part[:, -2]


def build_candidate_change_frame(
    rows: pd.DataFrame,
    base_labels: np.ndarray,
    neural_labels: np.ndarray,
    truth_labels: np.ndarray | None,
    base_prob: np.ndarray,
    neural_prob: np.ndarray,
) -> pd.DataFrame:
    base = np.asarray(base_labels, dtype=int)
    neural = np.asarray(neural_labels, dtype=int)
    changed = neural != base
    out = pd.DataFrame(index=np.arange(len(base)))
    out["row_idx"] = np.arange(len(base))
    out["changed"] = changed.astype(int)
    out["base_label"] = base
    out["neural_label"] = neural
    if truth_labels is None:
        out["is_correct_change"] = np.nan
    else:
        truth = np.asarray(truth_labels, dtype=int)
        out["truth_label"] = truth
        out["is_correct_change"] = ((neural == truth) & (base != truth) & changed).astype(int)
    out["transition_kind"] = [point0_transition_kind(b, n) for b, n in zip(base, neural)]
    out["base_p0"] = base_prob[:, 0]
    out["neural_p0"] = neural_prob[:, 0]
    out["p0_delta"] = neural_prob[:, 0] - base_prob[:, 0]
    out["base_label_prob"] = _label_score(base_prob, base)
    out["neural_label_prob"] = _label_score(neural_prob, neural)
    out["prob_gain"] = out["neural_label_prob"] - out["base_label_prob"]
    out["base_margin"] = _margin(base_prob)
    out["neural_margin"] = _margin(neural_prob)
    out["margin_delta"] = out["neural_margin"] - out["base_margin"]
    prefix = pd.to_numeric(rows.get("prefix_len", pd.Series([0] * len(rows))), errors="coerce").fillna(0)
    out["prefix_len"] = prefix.to_numpy(dtype=float)
    for col in ["r184_phase", "r184_lag0_depth", "r184_lag0_family", "lag0_spinId", "lag0_strengthId"]:
        if col in rows.columns:
            out[col] = rows[col].astype(str).to_numpy()
    return out


def selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "base_p0",
        "neural_p0",
        "p0_delta",
        "base_label_prob",
        "neural_label_prob",
        "prob_gain",
        "base_margin",
        "neural_margin",
        "margin_delta",
        "prefix_len",
    ]
    x = frame[numeric].copy()
    for col in ["transition_kind", "r184_phase", "r184_lag0_depth", "r184_lag0_family", "lag0_spinId", "lag0_strengthId"]:
        if col in frame.columns:
            x = pd.concat([x, pd.get_dummies(frame[col].astype(str), prefix=col, dtype=float)], axis=1)
    return x.astype(float)


def train_selector(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        C=0.35,
        max_iter=1000,
        random_state=204,
    )
    clf.fit(x, y)
    return clf


def align_columns(x: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
    return out[columns].astype(float)


def fold_safe_selector_trust(frame: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, list[dict], list[str]]:
    trust = np.zeros(len(frame), dtype=float)
    metrics = []
    columns_ref: list[str] | None = None
    changed = frame["changed"].eq(1).to_numpy()
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid_mask = rows["fold"].astype(int).eq(int(fold)).to_numpy() & changed
        train_mask = ~rows["fold"].astype(int).eq(int(fold)).to_numpy() & changed
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        y_train = frame.loc[train_mask, "is_correct_change"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            trust[valid_mask] = y_train.mean() if len(y_train) else 0.0
            continue
        x_train = selector_features(frame.loc[train_mask])
        columns = list(x_train.columns)
        columns_ref = columns if columns_ref is None else sorted(set(columns_ref) | set(columns))
        x_train = align_columns(x_train, columns)
        clf = train_selector(x_train, y_train)
        x_valid = align_columns(selector_features(frame.loc[valid_mask]), columns)
        pred = clf.predict_proba(x_valid)[:, 1]
        trust[valid_mask] = pred
        y_valid = frame.loc[valid_mask, "is_correct_change"].astype(int).to_numpy()
        metrics.append(
            {
                "fold": int(fold),
                "valid_changed_rows": int(valid_mask.sum()),
                "positive_rate": float(y_valid.mean()),
                "trust_mean": float(pred.mean()),
                "auc": float(roc_auc_score(y_valid, pred)) if len(np.unique(y_valid)) > 1 else np.nan,
            }
        )
    return trust, metrics, columns_ref or list(selector_features(frame.loc[changed]).columns)


def full_selector_trust(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    train_mask = train_frame["changed"].eq(1).to_numpy()
    y = train_frame.loc[train_mask, "is_correct_change"].astype(int).to_numpy()
    if train_mask.sum() == 0 or len(np.unique(y)) < 2:
        return np.zeros(len(test_frame), dtype=float)
    x_train = selector_features(train_frame.loc[train_mask])
    columns = list(x_train.columns)
    clf = train_selector(x_train, y)
    x_test = align_columns(selector_features(test_frame), columns)
    out = np.zeros(len(test_frame), dtype=float)
    changed = test_frame["changed"].eq(1).to_numpy()
    if changed.any():
        out[changed] = clf.predict_proba(x_test.loc[changed])[:, 1]
    return out


def select_changes_by_trust(
    base_labels: np.ndarray,
    neural_labels: np.ndarray,
    trust: np.ndarray,
    max_churn: float,
    allow_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    neural = np.asarray(neural_labels, dtype=int)
    changed = neural != base
    if allow_mask is not None:
        changed &= np.asarray(allow_mask, dtype=bool)
    max_rows = int(np.floor(len(base) * float(max_churn)))
    cand = np.where(changed)[0]
    keep = cand[np.argsort(np.asarray(trust, dtype=float)[cand])[::-1][:max_rows]]
    final = np.zeros(len(base), dtype=bool)
    final[keep] = True
    out = base.copy()
    out[final] = neural[final]
    return out, final


def point0_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame["transition_kind"].isin(["to_point0", "from_point0"]).to_numpy()


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, base: np.ndarray, meta: dict) -> dict:
    score = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    base_score = float(f1_score(y, base, labels=POINT_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "point_macro_f1": score,
        "delta_vs_base": score - base_score,
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
        "pred_point0_rate": float(np.mean(pred == 0)),
    }
    rec.update(meta)
    return rec


def write_submission(name: str, base_sub: pd.DataFrame, point: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = base_sub[["rally_uid", "actionId", "serverGetPoint"]].copy()
    out.insert(2, "pointId", np.asarray(point, dtype=int))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    oof_prob, test_prob, folds = run_v203(data)
    base = data["base_pred_oof"]
    y = data["y_oof"]
    test_base = data["test_base_point"]
    blend = row_log_blend(data["base_prob_oof"], normalize_rows_safe(oof_prob), ALPHA)
    blend_test = row_log_blend(data["base_prob_test"], normalize_rows_safe(test_prob), ALPHA)
    neural = blend.argmax(axis=1).astype(int)
    neural_test = blend_test.argmax(axis=1).astype(int)

    frame = build_candidate_change_frame(data["rows"], base, neural, y, data["base_prob_oof"], blend)
    test_frame = build_candidate_change_frame(data["test_rows"], test_base, neural_test, None, data["base_prob_test"], blend_test)
    trust_oof, selector_metrics, _ = fold_safe_selector_trust(frame, data["rows"])
    trust_test = full_selector_trust(frame, test_frame)
    frame.assign(selector_trust=trust_oof).to_csv(OUTDIR / "v204_oof_change_frame.csv", index=False)
    test_frame.assign(selector_trust=trust_test).to_csv(OUTDIR / "v204_test_change_frame.csv", index=False)
    pd.DataFrame(selector_metrics).to_csv(OUTDIR / "v204_selector_fold_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTDIR / "v204_ttshuttle_fold_metrics.csv", index=False)

    records = [
        eval_candidate(
            "v204_ttselector_raw_candidate_pool",
            y,
            neural,
            base,
            {
                "alpha": ALPHA,
                "cap": 1.0,
                "gate": "none",
                "test_raw_point0_rate": float(np.mean(test_prob.argmax(axis=1) == 0)),
            },
        )
    ]
    pred_store: dict[str, np.ndarray] = {}
    for cap in CAPS:
        name = f"v204_ttselector_a0p075_cap{str(cap).replace('.', 'p')}"
        pred, changed = select_changes_by_trust(base, neural, trust_oof, cap)
        test_pred, test_changed = select_changes_by_trust(test_base, neural_test, trust_test, cap)
        records.append(
            eval_candidate(
                name,
                y,
                pred,
                base,
                {
                    "alpha": ALPHA,
                    "cap": cap,
                    "gate": "selector_all",
                    "test_churn_vs_v173_r119": float(np.mean(test_pred != test_base)),
                    "test_changed_rows": int(test_changed.sum()),
                    "test_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                    "trust_mean_changed_test": float(trust_test[test_changed].mean()) if test_changed.any() else 0.0,
                },
            )
        )
        pred_store[name] = test_pred

    allow_oof = point0_mask(frame)
    allow_test = point0_mask(test_frame)
    for cap in [0.01, 0.02, 0.03]:
        name = f"v205_point0_ttselector_a0p075_cap{str(cap).replace('.', 'p')}"
        pred, changed = select_changes_by_trust(base, neural, trust_oof, cap, allow_mask=allow_oof)
        test_pred, test_changed = select_changes_by_trust(test_base, neural_test, trust_test, cap, allow_mask=allow_test)
        records.append(
            eval_candidate(
                name,
                y,
                pred,
                base,
                {
                    "alpha": ALPHA,
                    "cap": cap,
                    "gate": "point0_transitions_only",
                    "test_churn_vs_v173_r119": float(np.mean(test_pred != test_base)),
                    "test_changed_rows": int(test_changed.sum()),
                    "test_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                    "trust_mean_changed_test": float(trust_test[test_changed].mean()) if test_changed.any() else 0.0,
                },
            )
        )
        pred_store[name] = test_pred

    gate_oof = long_attack_gate(data["rows"])
    gate_test = long_attack_gate(data["test_rows"])
    name = "v204_ttselector_long_attack_a0p075_cap0p03"
    pred, changed = select_changes_by_trust(base, neural, trust_oof, 0.03, allow_mask=gate_oof)
    test_pred, test_changed = select_changes_by_trust(test_base, neural_test, trust_test, 0.03, allow_mask=gate_test)
    records.append(
        eval_candidate(
            name,
            y,
            pred,
            base,
            {
                "alpha": ALPHA,
                "cap": 0.03,
                "gate": "long_attack_selector",
                "test_churn_vs_v173_r119": float(np.mean(test_pred != test_base)),
                "test_changed_rows": int(test_changed.sum()),
                "test_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                "trust_mean_changed_test": float(trust_test[test_changed].mean()) if test_changed.any() else 0.0,
            },
        )
    )
    pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v204_search.csv", index=False)

    generated = []
    for name in [
        "v204_ttselector_a0p075_cap0p02",
        "v204_ttselector_a0p075_cap0p03",
        "v204_ttselector_a0p075_cap0p05",
        "v205_point0_ttselector_a0p075_cap0p02",
        "v204_ttselector_long_attack_a0p075_cap0p03",
    ]:
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        info.update(search[search["candidate"].eq(name)].iloc[0].to_dict())
        generated.append(info)

    positive = frame.loc[frame["changed"].eq(1), "is_correct_change"].astype(int)
    report = {
        "verdict": "GENERATED",
        "device": "selector",
        "candidate_pool_rows": int(frame["changed"].sum()),
        "candidate_pool_positive_rate": float(positive.mean()) if len(positive) else 0.0,
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": [
            "V204 learns a row-level trust selector over V203 residual candidates.",
            "V205 gates only point0 transitions to test point0 calibration safely.",
            "Raw TT-ShuttleNet argmax remains diagnostic only.",
            "Generated submissions keep action=V173 and server=R121.",
            "TTMATCH and ShuttleSet external rows are not read.",
        ],
    }
    (OUTDIR / "v204_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v204_report.md").write_text(
        "# V204 TT-ShuttleNet Residual Selector\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Candidate pool rows: `{report['candidate_pool_rows']}`\n"
        f"- Candidate pool positive rate: `{report['candidate_pool_positive_rate']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + "\n".join(
            f"- `{g['submission']}` OOF `{g['point_macro_f1']:.6f}`, delta `{g['delta_vs_base']:.6f}`, test churn `{g.get('test_churn_vs_v173_r119', 0):.6f}`"
            for g in generated
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v204_ttshuttle_residual_selector.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
