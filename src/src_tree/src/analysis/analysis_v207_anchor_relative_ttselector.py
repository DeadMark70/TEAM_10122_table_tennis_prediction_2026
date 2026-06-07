"""V207 anchor-relative TTSelector.

V204 proved that a TT-ShuttleNet residual selector is better than direct
TT-ShuttleNet residuals, but it learned against the older V173/R119 local base.
V207 changes the target: proposed TT-ShuttleNet changes must improve the current
public-positive V188 r186_w005 cap5 point anchor.

The script rebuilds V188 cap5 OOF labels fold-safely, trains a row-level
selector/ranker for TT-ShuttleNet changes relative to that anchor, and exports
low-churn capped submissions.  Raw neural argmax remains diagnostic only.
TTMATCH and ShuttleSet external rows are not read.
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
from analysis_v188_point_intent_gru import (
    LOSS_SETTINGS,
    StrokeDataset,
    capped_residual_labels,
    predict_proba,
    row_log_blend,
    set_seed,
    train_model,
)
from analysis_v195_distribution_matched_point_gru import distribution, prepare_data
from analysis_v204_ttshuttle_residual_selector import (
    build_candidate_change_frame,
    point0_transition_kind,
    selector_features,
)
from baseline_lgbm import POINT_CLASSES
from train_v203_tt_shuttlenet import apply_point_residual, long_attack_gate, run_v203


OUTDIR = Path("v207_anchor_relative_ttselector")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v207_anchor_relative_ttselector.py")

V188_ALPHA = 0.05
V188_CAP = 0.05
TT_ALPHA = 0.075
CAPS = [0.01, 0.02, 0.03]
DEVICE_NOTE = "uses V188/V203 imported device settings"


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


def build_anchor_relative_frame(
    rows: pd.DataFrame,
    anchor_labels: np.ndarray,
    candidate_labels: np.ndarray,
    truth_labels: np.ndarray | None,
    anchor_prob: np.ndarray,
    candidate_prob: np.ndarray,
) -> pd.DataFrame:
    frame = build_candidate_change_frame(rows, anchor_labels, candidate_labels, truth_labels, anchor_prob, candidate_prob)
    frame = frame.rename(
        columns={
            "base_label": "anchor_label",
            "neural_label": "candidate_label",
            "transition_kind": "anchor_transition_kind",
            "base_p0": "anchor_p0",
            "neural_p0": "candidate_p0",
            "base_label_prob": "anchor_label_prob",
            "neural_label_prob": "candidate_label_prob",
            "base_margin": "anchor_margin",
            "neural_margin": "candidate_margin",
        }
    )
    if truth_labels is not None:
        truth = np.asarray(truth_labels, dtype=int)
        anchor = np.asarray(anchor_labels, dtype=int)
        candidate = np.asarray(candidate_labels, dtype=int)
        frame["is_anchor_improvement"] = ((candidate == truth) & (anchor != truth) & (candidate != anchor)).astype(int)
        frame["anchor_was_correct"] = (anchor == truth).astype(int)
    else:
        frame["is_anchor_improvement"] = np.nan
        frame["anchor_was_correct"] = np.nan
    return frame


def transition_policy_mask(frame: pd.DataFrame, point0_mode: str = "strict") -> np.ndarray:
    kind = frame["anchor_transition_kind"].astype(str)
    changed = frame["changed"].eq(1).to_numpy() if "changed" in frame.columns else np.ones(len(frame), dtype=bool)
    if point0_mode == "loose":
        return changed
    if point0_mode != "strict":
        raise ValueError(point0_mode)
    allow = changed.copy()
    to_p0 = kind.eq("to_point0").to_numpy()
    strict_to_p0 = (
        (pd.to_numeric(frame["candidate_p0"], errors="coerce").fillna(0).to_numpy() >= 0.35)
        & (pd.to_numeric(frame["prob_gain"], errors="coerce").fillna(0).to_numpy() >= 0.03)
    )
    allow &= (~to_p0) | strict_to_p0
    return allow


def testlike_slice(rows: pd.DataFrame) -> pd.Series:
    phase = rows.get("r184_phase", rows.get("audit_phase", pd.Series(["unknown"] * len(rows)))).astype(str)
    depth = rows.get("r184_lag0_depth", rows.get("audit_lag0_depth", pd.Series(["unknown"] * len(rows)))).astype(str)
    family = rows.get("r184_lag0_family", rows.get("audit_lag0_action_family", pd.Series(["unknown"] * len(rows)))).astype(str)
    prefix = pd.to_numeric(rows.get("prefix_len", pd.Series([0] * len(rows))), errors="coerce").fillna(0)
    out = np.full(len(rows), "other", dtype=object)
    out[(phase.eq("rally") | depth.eq("long") | family.eq("Attack") | prefix.ge(3)).to_numpy()] = "testlike"
    out[phase.eq("receive").to_numpy()] = "receive"
    return pd.Series(out)


def select_by_slice_caps(
    anchor_labels: np.ndarray,
    candidate_labels: np.ndarray,
    trust: np.ndarray,
    slices: pd.Series,
    caps: dict[str, float],
    allow_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor_labels, dtype=int)
    candidate = np.asarray(candidate_labels, dtype=int)
    changed = candidate != anchor
    if allow_mask is not None:
        changed &= np.asarray(allow_mask, dtype=bool)
    selected = np.zeros(len(anchor), dtype=bool)
    slice_values = pd.Series(slices).astype(str).to_numpy()
    for name, cap in caps.items():
        mask = changed & (slice_values == str(name))
        max_rows = int(np.floor(mask.sum() * float(cap)))
        cand = np.where(mask)[0]
        if len(cand) == 0 or max_rows <= 0:
            continue
        keep = cand[np.argsort(np.asarray(trust, dtype=float)[cand])[::-1][:max_rows]]
        selected[keep] = True
    out = anchor.copy()
    out[selected] = candidate[selected]
    return out, selected


def train_selector(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", C=0.30, max_iter=1000, random_state=207)
    clf.fit(x, y)
    return clf


def align_columns(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[cols].astype(float)


def anchor_selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "anchor_p0": "base_p0",
            "candidate_p0": "neural_p0",
            "anchor_label_prob": "base_label_prob",
            "candidate_label_prob": "neural_label_prob",
            "anchor_margin": "base_margin",
            "candidate_margin": "neural_margin",
            "anchor_transition_kind": "transition_kind",
        }
    )
    x = selector_features(renamed)
    if "anchor_was_correct" in frame.columns:
        x["anchor_was_correct_prior"] = pd.to_numeric(frame["anchor_was_correct"], errors="coerce").fillna(0.0)
    return x


def fold_safe_trust(frame: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, list[dict], list[str]]:
    trust = np.zeros(len(frame), dtype=float)
    metrics = []
    changed = frame["changed"].eq(1).to_numpy()
    final_cols: list[str] = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy() & changed
        train = ~rows["fold"].astype(int).eq(int(fold)).to_numpy() & changed
        if valid.sum() == 0 or train.sum() == 0:
            continue
        y_train = frame.loc[train, "is_anchor_improvement"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            trust[valid] = float(y_train.mean()) if len(y_train) else 0.0
            continue
        x_train = anchor_selector_features(frame.loc[train])
        cols = list(x_train.columns)
        final_cols = sorted(set(final_cols) | set(cols))
        clf = train_selector(x_train, y_train)
        x_valid = align_columns(anchor_selector_features(frame.loc[valid]), cols)
        pred = clf.predict_proba(x_valid)[:, 1]
        trust[valid] = pred
        y_valid = frame.loc[valid, "is_anchor_improvement"].astype(int).to_numpy()
        metrics.append(
            {
                "fold": int(fold),
                "valid_changed_rows": int(valid.sum()),
                "positive_rate": float(y_valid.mean()),
                "trust_mean": float(pred.mean()),
                "auc": float(roc_auc_score(y_valid, pred)) if len(np.unique(y_valid)) > 1 else np.nan,
            }
        )
    return trust, metrics, final_cols


def full_trust(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    mask = train_frame["changed"].eq(1).to_numpy()
    y = train_frame.loc[mask, "is_anchor_improvement"].astype(int).to_numpy()
    out = np.zeros(len(test_frame), dtype=float)
    changed = test_frame["changed"].eq(1).to_numpy()
    if mask.sum() == 0 or len(np.unique(y)) < 2 or changed.sum() == 0:
        return out
    x_train = anchor_selector_features(train_frame.loc[mask])
    cols = list(x_train.columns)
    clf = train_selector(x_train, y)
    x_test = align_columns(anchor_selector_features(test_frame.loc[changed]), cols)
    out[changed] = clf.predict_proba(x_test)[:, 1]
    return out


def rebuild_v188_cap5_anchor(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = data["rows"]
    y = data["y_oof"]
    weights = dict(LOSS_SETTINGS)["r186_w005"]
    oof_prob = np.zeros((len(rows), 10), dtype=float)
    fold_records = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        train_ds = StrokeDataset(data["oof_seq"][train], data["oof_len"][train], data["x_oof"][train], y[train], data["teacher_oof"][train])
        valid_ds = StrokeDataset(data["oof_seq"][valid], data["oof_len"][valid], data["x_oof"][valid], y[valid], data["teacher_oof"][valid])
        model, val_loss = train_model(train_ds, valid_ds, data["vocab_sizes"], data["x_oof"].shape[1], weights, 1880 + int(fold))
        oof_prob[valid] = predict_proba(model, valid_ds)
        raw = oof_prob[valid].argmax(axis=1)
        fold_records.append(
            {
                "component": "v188_anchor_rebuild",
                "fold": int(fold),
                "val_loss": float(val_loss),
                "raw_point_macro_f1": float(f1_score(y[valid], raw, labels=POINT_CLASSES, average="macro", zero_division=0)),
            }
        )

    full_ds = StrokeDataset(data["oof_seq"], data["oof_len"], data["x_oof"], y, data["teacher_oof"])
    hold = max(1, len(y) // 10)
    hold_ds = StrokeDataset(data["oof_seq"][:hold], data["oof_len"][:hold], data["x_oof"][:hold], y[:hold], data["teacher_oof"][:hold])
    test_ds = StrokeDataset(data["test_seq"], data["test_len"], data["x_test_oofstats"], np.zeros(len(data["test_seq"]), dtype=np.int64), data["teacher_test"])
    full_model, _ = train_model(full_ds, hold_ds, data["vocab_sizes"], data["x_oof"].shape[1], weights, 1988)
    test_prob = predict_proba(full_model, test_ds)

    blend_oof = row_log_blend(data["base_prob_oof"], normalize_rows_safe(oof_prob), V188_ALPHA)
    blend_test = row_log_blend(data["base_prob_test"], normalize_rows_safe(test_prob), V188_ALPHA)
    anchor_oof, _ = capped_residual_labels(data["base_pred_oof"], blend_oof, V188_CAP)
    anchor_test, _ = capped_residual_labels(data["test_base_point"], blend_test, V188_CAP)
    return anchor_oof, anchor_test, blend_oof, blend_test, fold_records


def eval_candidate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, meta: dict) -> dict:
    score = float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0))
    anchor_score = float(f1_score(y, anchor, labels=POINT_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "point_macro_f1": score,
        "delta_vs_v188_anchor": score - anchor_score,
        "point_churn_vs_v188_anchor": float(np.mean(pred != anchor)),
        "changed_rows_vs_v188_anchor": int(np.sum(pred != anchor)),
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
    set_seed(207)
    data = prepare_data()
    y = data["y_oof"]

    anchor_oof, anchor_test, anchor_prob_oof, anchor_prob_test, v188_folds = rebuild_v188_cap5_anchor(data)
    tt_oof, tt_test, tt_folds = run_v203(data)
    tt_blend_oof = row_log_blend(data["base_prob_oof"], normalize_rows_safe(tt_oof), TT_ALPHA)
    tt_blend_test = row_log_blend(data["base_prob_test"], normalize_rows_safe(tt_test), TT_ALPHA)
    candidate_oof = tt_blend_oof.argmax(axis=1).astype(int)
    candidate_test = tt_blend_test.argmax(axis=1).astype(int)

    frame = build_anchor_relative_frame(data["rows"], anchor_oof, candidate_oof, y, anchor_prob_oof, tt_blend_oof)
    test_frame = build_anchor_relative_frame(data["test_rows"], anchor_test, candidate_test, None, anchor_prob_test, tt_blend_test)
    trust_oof, selector_metrics, feature_cols = fold_safe_trust(frame, data["rows"])
    trust_test = full_trust(frame, test_frame)
    frame.assign(selector_trust=trust_oof, slice=testlike_slice(data["rows"]).to_numpy()).to_csv(OUTDIR / "v207_oof_anchor_frame.csv", index=False)
    test_frame.assign(selector_trust=trust_test, slice=testlike_slice(data["test_rows"]).to_numpy()).to_csv(OUTDIR / "v207_test_anchor_frame.csv", index=False)
    pd.DataFrame(selector_metrics).to_csv(OUTDIR / "v207_selector_fold_metrics.csv", index=False)
    pd.DataFrame(v188_folds + tt_folds).to_csv(OUTDIR / "v207_model_fold_metrics.csv", index=False)

    records = [
        eval_candidate(
            "v188_cap5_rebuilt_anchor",
            y,
            anchor_oof,
            anchor_oof,
            {
                "gate": "anchor",
                "test_churn_vs_v188_rebuilt": 0.0,
                "test_changed_rows": 0,
                "test_distribution": json.dumps(distribution(anchor_test), sort_keys=True),
            },
        )
    ]
    pred_store: dict[str, np.ndarray] = {}
    slices_oof = testlike_slice(data["rows"])
    slices_test = testlike_slice(data["test_rows"])
    policies = [
        ("strict_cap1", {"receive": 0.0, "testlike": 0.01, "other": 0.005}, "strict"),
        ("strict_cap2", {"receive": 0.0, "testlike": 0.02, "other": 0.01}, "strict"),
        ("strict_cap3", {"receive": 0.0, "testlike": 0.03, "other": 0.015}, "strict"),
        ("loose_cap2", {"receive": 0.005, "testlike": 0.02, "other": 0.01}, "loose"),
    ]
    for tag, caps, mode in policies:
        allow_oof = transition_policy_mask(frame, point0_mode=mode)
        allow_test = transition_policy_mask(test_frame, point0_mode=mode)
        pred, changed = select_by_slice_caps(anchor_oof, candidate_oof, trust_oof, slices_oof, caps, allow_oof)
        test_pred, test_changed = select_by_slice_caps(anchor_test, candidate_test, trust_test, slices_test, caps, allow_test)
        name = f"v207_anchor_ttselector_{tag}"
        records.append(
            eval_candidate(
                name,
                y,
                pred,
                anchor_oof,
                {
                    "gate": tag,
                    "test_churn_vs_v188_rebuilt": float(np.mean(test_pred != anchor_test)),
                    "test_changed_rows": int(test_changed.sum()),
                    "test_distribution": json.dumps(distribution(test_pred), sort_keys=True),
                    "trust_mean_changed_test": float(trust_test[test_changed].mean()) if test_changed.any() else 0.0,
                },
            )
        )
        pred_store[name] = test_pred

    search = pd.DataFrame(records).sort_values(["delta_vs_v188_anchor", "point_churn_vs_v188_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v207_search.csv", index=False)
    generated = []
    for name in ["v207_anchor_ttselector_strict_cap1", "v207_anchor_ttselector_strict_cap2", "v207_anchor_ttselector_strict_cap3"]:
        sub_name = f"submission_{name}__v173action_r121server.csv"
        info = write_submission(sub_name, data["base_sub"], pred_store[name])
        info.update(search[search["candidate"].eq(name)].iloc[0].to_dict())
        generated.append(info)

    changed_pool = frame["changed"].eq(1)
    positive = frame.loc[changed_pool, "is_anchor_improvement"].astype(int)
    report = {
        "verdict": "GENERATED",
        "candidate_pool_rows": int(changed_pool.sum()),
        "candidate_pool_positive_rate": float(positive.mean()) if len(positive) else 0.0,
        "generated": generated,
        "best": search.head(8).to_dict(orient="records"),
        "notes": [
            "V207 selector target is improvement over rebuilt V188 r186_w005 cap5 OOF anchor.",
            "Slice caps avoid receive-heavy changes and concentrate on test-like rows.",
            "Strict point0 mode blocks weak nonzero-to-point0 transitions.",
            "Raw TT-ShuttleNet argmax is not exported.",
            "TTMATCH and ShuttleSet external rows are not read.",
        ],
    }
    (OUTDIR / "v207_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v207_report.md").write_text(
        "# V207 Anchor-Relative TTSelector\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Candidate pool rows: `{report['candidate_pool_rows']}`\n"
        f"- Candidate pool positive rate: `{report['candidate_pool_positive_rate']:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + "\n".join(
            f"- `{g['submission']}` OOF delta vs V188 `{g['delta_vs_v188_anchor']:.6f}`, churn `{g['point_churn_vs_v188_anchor']:.6f}`, test churn `{g['test_churn_vs_v188_rebuilt']:.6f}`"
            for g in generated
        )
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v207_anchor_relative_ttselector.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
