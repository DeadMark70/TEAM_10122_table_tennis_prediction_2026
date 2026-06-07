"""V288 specialist feature discovery diagnostics.

This module keeps V288 as a local diagnostic layer. Generated submissions keep
pointId and serverGetPoint fixed to the V261/R121 anchor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline_lgbm import ACTION_CLASSES
from analysis_v286_weak_action_specialist_pretraining import (
    action_family,
    class_f1,
    phase_bin,
    point_depth,
    score_pressure_bin,
)

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v288_specialist_feature_discovery"
V286_OUTDIR = ROOT / "v286_weak_action_specialist_pretraining"
ANCHOR_SUBMISSION = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
V286_OOF = V286_OUTDIR / "v286_specialist_oof.csv"
V286_STAGE1 = V286_OUTDIR / "v286_stage1_pretrain_table.csv"
V286_SUB_BY_CHURN = {
    0.005: V286_OUTDIR / "submission_v286_weak_spec_churn0p005__pv261cap1__sr121.csv",
    0.010: V286_OUTDIR / "submission_v286_weak_spec_churn0p010__pv261cap1__sr121.csv",
    0.020: V286_OUTDIR / "submission_v286_weak_spec_churn0p020__pv261cap1__sr121.csv",
}


SPECIALIST_GROUPS = {
    "fast_attack_57": [5, 7],
    "terminal_03": [0, 3],
    "style_control_89": [8, 9],
    "short_control_411": [4, 11],
    "defensive_1214": [12, 14],
}

PROTECTED_ACTIONS = [1, 10, 12, 13]
SERVE_ACTIONS = [15, 16, 17, 18]
TEACHER_ACTIONS = [0, 3, 5, 7, 8, 9, 14]
EXPORT_VARIANTS = [
    ("v288_fast57_c0p005", "fast_attack_57", {5, 7}, 0.005, "submission_v288_fast57_c0p005__pv261cap1__sr121.csv"),
    ("v288_fast57_c0p010", "fast_attack_57", {5, 7}, 0.010, "submission_v288_fast57_c0p010__pv261cap1__sr121.csv"),
    ("v288_terminal03_c0p005", "terminal_03", {0, 3}, 0.005, "submission_v288_terminal03_c0p005__pv261cap1__sr121.csv"),
    ("v288_terminal03_c0p010", "terminal_03", {0, 3}, 0.010, "submission_v288_terminal03_c0p010__pv261cap1__sr121.csv"),
    ("v288_bank_safe_c0p005", "bank_safe", {0, 3, 5, 7}, 0.005, "submission_v288_bank_safe_c0p005__pv261cap1__sr121.csv"),
    ("v288_bank_safe_c0p010", "bank_safe", {0, 3, 5, 7}, 0.010, "submission_v288_bank_safe_c0p010__pv261cap1__sr121.csv"),
    (
        "v288_bank_diagnostic_c0p020",
        "bank_diagnostic",
        {0, 3, 5, 7, 8, 9, 14},
        0.020,
        "submission_v288_bank_diagnostic_c0p020__pv261cap1__sr121.csv",
    ),
]


def group_for_action(action: int) -> str:
    action = int(action)
    for group, actions in SPECIALIST_GROUPS.items():
        if action in actions:
            return group
    return ""


def feature_family_columns() -> dict[str, list[str]]:
    return {
        "phase_prefix": [
            "prefix_len",
            "phase_bin",
            "prefix_len_bin",
            "is_receive",
            "is_third",
            "is_fourth",
            "is_rally",
        ],
        "incoming_ball": [
            "lag0_actionId",
            "lag0_action_family",
            "lag0_pointId",
            "lag0_point_depth",
            "lag0_spin",
            "lag0_strength",
            "lag0_positionId",
            "lag0_action_point_pair",
            "lag0_spin_strength_pair",
        ],
        "score_pressure": [
            "scoreSelf",
            "scoreOther",
            "scoreTotal",
            "serverScoreDiff",
            "score_pressure_bin",
            "is_deuce_like",
            "is_game_point_like",
        ],
        "external_clean_prior": [
            "v286_ext_family_Attack",
            "v286_ext_family_Control",
            "v286_ext_family_Defensive",
            "v286_ext_family_Zero",
            "v286_ext_terminal_rate",
        ],
        "teacher_specialist": [
            "anchor_action",
            "specialist_p_0",
            "specialist_p_3",
            "specialist_p_5",
            "specialist_p_7",
            "specialist_p_8",
            "specialist_p_9",
            "specialist_p_14",
            "support_0",
            "support_3",
            "support_5",
            "support_7",
            "support_8",
            "support_9",
            "support_14",
        ],
        "support_backoff": [
            "support_exact",
            "support_family_depth",
            "support_phase",
            "support_global",
        ],
        "style_response": [
            "actor_cond_family_rate",
            "actor_cond_action_rate",
            "receiver_pressure_family_rate",
            "style_trust",
            "pair_familiarity_bin",
        ],
    }


def _prefix_len_bin(value: int) -> str:
    value = int(value)
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value <= 6:
        return "4_6"
    return "7_plus"


def build_basic_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["prefix_len"] = pd.to_numeric(out.get("prefix_len", 0), errors="coerce").fillna(0).astype(int)
    out["phase_bin"] = out["prefix_len"].map(phase_bin)
    out["prefix_len_bin"] = out["prefix_len"].map(_prefix_len_bin)
    out["is_receive"] = out["phase_bin"].eq("receive").astype(int)
    out["is_third"] = out["phase_bin"].eq("third").astype(int)
    out["is_fourth"] = out["phase_bin"].eq("fourth").astype(int)
    out["is_rally"] = out["phase_bin"].eq("rally").astype(int)

    out["lag0_actionId"] = pd.to_numeric(out.get("lag0_actionId", 0), errors="coerce").fillna(0).astype(int)
    out["lag0_pointId"] = pd.to_numeric(out.get("lag0_pointId", 0), errors="coerce").fillna(0).astype(int)
    out["lag0_spin"] = pd.to_numeric(out.get("lag0_spinId", out.get("lag0_spin", 0)), errors="coerce").fillna(0).astype(int)
    out["lag0_strength"] = (
        pd.to_numeric(out.get("lag0_strengthId", out.get("lag0_strength", 0)), errors="coerce").fillna(0).astype(int)
    )
    out["lag0_positionId"] = pd.to_numeric(out.get("lag0_positionId", 0), errors="coerce").fillna(0).astype(int)
    out["lag0_action_family"] = out["lag0_actionId"].map(action_family)
    out["lag0_point_depth"] = out["lag0_pointId"].map(point_depth)
    out["lag0_action_point_pair"] = out["lag0_actionId"].astype(str) + "_" + out["lag0_pointId"].astype(str)
    out["lag0_spin_strength_pair"] = out["lag0_spin"].astype(str) + "_" + out["lag0_strength"].astype(str)

    out["scoreSelf"] = pd.to_numeric(out.get("scoreSelf", out.get("serverScore", 0)), errors="coerce").fillna(0)
    out["scoreOther"] = pd.to_numeric(out.get("scoreOther", out.get("receiverScore", 0)), errors="coerce").fillna(0)
    out["scoreTotal"] = pd.to_numeric(out.get("scoreTotal", out["scoreSelf"] + out["scoreOther"]), errors="coerce").fillna(0)
    out["serverScoreDiff"] = pd.to_numeric(
        out.get("serverScoreDiff", out["scoreSelf"] - out["scoreOther"]), errors="coerce"
    ).fillna(0)
    out["score_pressure_bin"] = [
        score_pressure_bin(total, diff) for total, diff in zip(out["scoreTotal"], out["serverScoreDiff"])
    ]
    out["is_deuce_like"] = ((out["scoreTotal"] >= 18) & (out["serverScoreDiff"].abs() <= 1)).astype(int)
    out["is_game_point_like"] = ((out["scoreSelf"] >= 10) | (out["scoreOther"] >= 10)).astype(int)
    return out


def family_average_precision(y_binary: np.ndarray, score: np.ndarray) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_binary.sum() <= 0:
        return 0.0
    if y_binary.sum() == len(y_binary):
        return 1.0
    return float(average_precision_score(y_binary, score))


def safe_roc_auc(y_binary: np.ndarray, score: np.ndarray) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    if len(np.unique(y_binary)) < 2:
        return 0.5
    return float(roc_auc_score(y_binary, np.asarray(score, dtype=float)))


def feature_family_audit(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    records = []
    families = feature_family_columns()
    for group_name, actions in SPECIALIST_GROUPS.items():
        y_group = np.isin(np.asarray(y, dtype=int), actions).astype(int)
        for family, cols in families.items():
            available = [col for col in cols if col in frame.columns]
            if not available:
                records.append(
                    {
                        "group": group_name,
                        "family": family,
                        "available_cols": 0,
                        "best_col": "",
                        "best_ap": 0.0,
                        "best_auc": 0.5,
                    }
                )
                continue

            best_col = ""
            best_ap = -1.0
            best_auc = 0.5
            for col in available:
                codes = pd.factorize(frame[col].astype(str), sort=True)[0]
                if len(np.unique(codes)) <= 1:
                    score = np.zeros(len(frame), dtype=float)
                else:
                    score = pd.Series(codes).rank(method="average").to_numpy(dtype=float)
                ap = family_average_precision(y_group, score)
                auc = safe_roc_auc(y_group, score)
                if ap > best_ap:
                    best_col = col
                    best_ap = ap
                    best_auc = auc
            records.append(
                {
                    "group": group_name,
                    "family": family,
                    "available_cols": len(available),
                    "best_col": best_col,
                    "best_ap": float(best_ap),
                    "best_auc": float(best_auc),
                }
            )
    return pd.DataFrame(records)


def select_group_candidates(frame: pd.DataFrame, min_score: float, min_support: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)
    filtered = frame[
        (pd.to_numeric(frame["group_score"], errors="coerce").fillna(0.0) >= float(min_score))
        & (pd.to_numeric(frame["support_count"], errors="coerce").fillna(0) >= int(min_support))
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=frame.columns)
    ranked = filtered.sort_values(["row_id", "group_score", "support_count"], ascending=[True, False, False])
    return ranked.groupby("row_id", as_index=False, sort=False).head(1).reset_index(drop=True)


def _f1_macro(y: np.ndarray, pred: np.ndarray, labels: list[int] | np.ndarray = ACTION_CLASSES) -> float:
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def _cap_token(churn: float) -> str:
    return f"{float(churn):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _candidate_tier(rec: dict[str, Any], diagnostic: bool) -> str:
    if diagnostic:
        return "diagnostic_only"
    if (
        float(rec["delta_vs_v173"]) > 0
        and float(rec["weak_mean_delta"]) > 0
        and float(rec["protected_mean_delta"]) >= 0
        and 3 <= int(rec["test_changed_rows"]) <= 25
    ):
        return "clean_probe"
    return "diagnostic_only"


def train_group_scores(oof: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [f"specialist_p_{a}" for a in TEACHER_ACTIONS] + [f"support_{a}" for a in TEACHER_ACTIONS] + ["anchor_action"]
    x = oof[feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    folds = oof["fold"].astype(int).to_numpy() if "fold" in oof else np.zeros(len(oof), dtype=int)
    out = pd.DataFrame(index=oof.index)
    y_true = oof["y_true_action"].astype(int).to_numpy()
    for group_name, actions in SPECIALIST_GROUPS.items():
        target = np.isin(y_true, actions).astype(int)
        scores = np.zeros(len(oof), dtype=float)
        fitted = 0
        for fold in sorted(np.unique(folds)):
            valid = folds == fold
            train = ~valid
            if len(np.unique(target[train])) < 2:
                continue
            clf = ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=288 + int(fold) + len(actions) * 17,
                n_jobs=1,
            )
            clf.fit(x.loc[train, feature_cols], target[train])
            scores[valid] = clf.predict_proba(x.loc[valid, feature_cols])[:, 1]
            fitted += 1
        if fitted == 0:
            scores[:] = float(target.mean()) if len(target) else 0.0
        out[f"group_p_{group_name}"] = scores
    return out


def build_oof_group_candidate_frame(oof: pd.DataFrame, group_scores: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    anchor = oof["anchor_action"].astype(int).to_numpy()
    for action in TEACHER_ACTIONS:
        group = group_for_action(action)
        score = (
            pd.to_numeric(group_scores[f"group_p_{group}"], errors="coerce").fillna(0.0).to_numpy()
            * pd.to_numeric(oof[f"specialist_p_{action}"], errors="coerce").fillna(0.0).to_numpy()
        )
        frame = pd.DataFrame(
            {
                "row_id": np.arange(len(oof), dtype=int),
                "group": group,
                "anchor_action": anchor,
                "candidate_action": int(action),
                "group_score": score,
                "support_count": pd.to_numeric(oof[f"support_{action}"], errors="coerce").fillna(0).to_numpy(),
            }
        )
        pieces.append(frame)
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["candidate_action"].astype(int).ne(out["anchor_action"].astype(int))]
    out = out[~out["anchor_action"].astype(int).isin(PROTECTED_ACTIONS + SERVE_ACTIONS)]
    return out.reset_index(drop=True)


def apply_row_cap(anchor: np.ndarray, row_candidates: pd.DataFrame, max_churn: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(anchor, dtype=int).copy()
    selected = np.zeros(len(pred), dtype=bool)
    max_rows = int(math.floor(len(pred) * float(max_churn)))
    if row_candidates.empty or max_rows <= 0:
        return pred, selected
    ranked = row_candidates.sort_values(["group_score", "support_count"], ascending=[False, False]).head(max_rows)
    ids = ranked["row_id"].astype(int).to_numpy()
    selected[ids] = True
    pred[ids] = ranked["candidate_action"].astype(int).to_numpy()
    return pred, selected


def filtered_test_prediction(anchor_test: np.ndarray, source_test: np.ndarray, allowed: set[int]) -> np.ndarray:
    pred = np.asarray(anchor_test, dtype=int).copy()
    source = np.asarray(source_test, dtype=int)
    changed = source != anchor_test
    keep = changed & np.isin(source, sorted(allowed)) & ~np.isin(anchor_test, PROTECTED_ACTIONS + SERVE_ACTIONS)
    pred[keep] = source[keep]
    return pred


def changed_row_report(anchor: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    anchor = np.asarray(anchor, dtype=int)
    pred = np.asarray(pred, dtype=int)
    changed = pred != anchor
    report: dict[str, int] = {"changed_rows": int(changed.sum())}
    for action in sorted(set(pred[changed].tolist())):
        report[f"changed_to_{int(action)}"] = int(np.sum(changed & (pred == int(action))))
    return report


def evaluate_variant(
    name: str,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    pred_oof: np.ndarray,
    anchor_test: np.ndarray,
    pred_test: np.ndarray,
    allowed: set[int],
    max_churn: float,
    diagnostic: bool,
) -> dict[str, Any]:
    base_macro = _f1_macro(y, anchor_oof)
    macro = _f1_macro(y, pred_oof)
    weak_actions = np.array(sorted({a for actions in SPECIALIST_GROUPS.values() for a in actions}), dtype=int)
    weak_base = _f1_macro(y, anchor_oof, weak_actions)
    weak = _f1_macro(y, pred_oof, weak_actions)
    prot_base = _f1_macro(y, anchor_oof, PROTECTED_ACTIONS)
    prot = _f1_macro(y, pred_oof, PROTECTED_ACTIONS)
    class_delta = {
        str(k): float(class_f1(y, pred_oof, ACTION_CLASSES)[k] - class_f1(y, anchor_oof, ACTION_CLASSES)[k])
        for k in ACTION_CLASSES
    }
    changed_test = pred_test != anchor_test
    rec = {
        "candidate": name,
        "allowed_actions": "/".join(str(x) for x in sorted(allowed)),
        "max_churn": float(max_churn),
        "action_macro_f1": float(macro),
        "delta_vs_v173": float(macro - base_macro),
        "weak_mean_delta": float(weak - weak_base),
        "protected_mean_delta": float(prot - prot_base),
        "test_changed_rows": int(changed_test.sum()),
        "test_churn": float(changed_test.mean()),
        "class_f1_delta_json": json.dumps(class_delta, sort_keys=True),
        **changed_row_report(anchor_test, pred_test),
    }
    rec["candidate_tier"] = _candidate_tier(rec, diagnostic=diagnostic)
    return rec


def write_submission(name: str, action: np.ndarray, anchor_sub: pd.DataFrame) -> Path:
    out = pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    return path


def build_feature_audit_input(oof: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if V286_STAGE1.exists():
        stage = pd.read_csv(V286_STAGE1)
        stage = stage[stage["next_actionId"].notna()].reset_index(drop=True)
        frame = build_basic_feature_frame(stage)
        y = stage["next_actionId"].astype(int).to_numpy()
        return frame, y
    frame = oof.copy()
    y = frame["y_true_action"].astype(int).to_numpy()
    return frame, y


def build_group_class_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    anchor_f1 = class_f1(y, anchor, ACTION_CLASSES)
    pred_f1 = class_f1(y, pred, ACTION_CLASSES)
    return pd.DataFrame(
        [
            {
                "action": int(action),
                "group": group_for_action(action),
                "is_protected": int(action in PROTECTED_ACTIONS),
                "anchor_f1": float(anchor_f1[action]),
                "v288_f1": float(pred_f1[action]),
                "delta": float(pred_f1[action] - anchor_f1[action]),
            }
            for action in ACTION_CLASSES
        ]
    )


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF file: {V286_OOF}")
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing anchor submission: {ANCHOR_SUBMISSION}")

    oof = pd.read_csv(V286_OOF)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()

    feature_frame, feature_y = build_feature_audit_input(oof)
    feature_report = feature_family_audit(feature_frame, feature_y)
    feature_report["family_enabled"] = feature_report["available_cols"].gt(0).astype(int)
    feature_report.to_csv(OUTDIR / "v288_feature_family_report.csv", index=False)

    group_scores = train_group_scores(oof)
    oof_candidates = build_oof_group_candidate_frame(oof, group_scores)
    rows = []
    generated = []
    predictions: dict[str, np.ndarray] = {}
    for name, group_name, allowed, churn, filename in EXPORT_VARIANTS:
        diagnostic = group_name not in {"fast_attack_57", "terminal_03", "bank_safe"}
        pool = oof_candidates[oof_candidates["candidate_action"].astype(int).isin(sorted(allowed))].copy()
        pool = select_group_candidates(pool, min_score=0.05, min_support=5)
        pred_oof, _selected = apply_row_cap(anchor_oof, pool, churn)
        source_path = V286_SUB_BY_CHURN[churn]
        source_sub = pd.read_csv(source_path)
        pred_test = filtered_test_prediction(anchor_test, source_sub["actionId"].astype(int).to_numpy(), allowed)
        rows.append(evaluate_variant(name, y, anchor_oof, pred_oof, anchor_test, pred_test, allowed, churn, diagnostic))
        generated.append(str(write_submission(filename, pred_test, anchor_sub).relative_to(ROOT)))
        predictions[name] = pred_oof

    search = pd.DataFrame(rows).sort_values(
        ["candidate_tier", "delta_vs_v173", "weak_mean_delta", "protected_mean_delta", "test_changed_rows"],
        ascending=[True, False, False, False, True],
    )
    search.to_csv(OUTDIR / "v288_group_search.csv", index=False)
    best = search.iloc[0].to_dict() if len(search) else {}
    best_pred = predictions.get(str(best.get("candidate", "")), anchor_oof)
    build_group_class_report(y, anchor_oof, best_pred).to_csv(OUTDIR / "v288_group_class_report.csv", index=False)

    changed_audit = search[
        ["candidate", "allowed_actions", "max_churn", "test_changed_rows", "test_churn", "changed_rows"]
    ].copy()
    changed_audit.to_csv(OUTDIR / "v288_changed_row_audit.csv", index=False)

    upload_recommendation = "DO_NOT_UPLOAD"
    clean = search[search["candidate_tier"].eq("clean_probe")].copy()
    if not clean.empty:
        candidate = clean.sort_values(["test_changed_rows", "delta_vs_v173"], ascending=[True, False]).iloc[0]
        if (
            float(candidate["delta_vs_v173"]) >= 0.001
            and float(candidate["weak_mean_delta"]) > 0
            and float(candidate["protected_mean_delta"]) >= 0
            and 3 <= int(candidate["test_changed_rows"]) <= 25
        ):
            upload_recommendation = "REVIEW_LOW_CHURN_V288_DIAGNOSTIC"

    report = {
        "version": "V288",
        "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
        "best_candidate": best,
        "generated_submissions": generated,
        "upload_recommendation": upload_recommendation,
    }
    (OUTDIR / "v288_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# V288 specialist feature discovery",
        "",
        f"Anchor: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        f"Best candidate: `{best.get('candidate', '')}`",
        f"OOF delta vs V173: {float(best.get('delta_vs_v173', 0.0)):.6f}",
        f"Weak/group mean delta: {float(best.get('weak_mean_delta', 0.0)):.6f}",
        f"Protected mean delta: {float(best.get('protected_mean_delta', 0.0)):.6f}",
        f"Test changed rows: {int(best.get('test_changed_rows', 0))}",
        f"Upload recommendation: {upload_recommendation}",
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v288_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "best_candidate": report["best_candidate"].get("candidate", ""),
                "best_delta_vs_v173": report["best_candidate"].get("delta_vs_v173", 0.0),
                "generated_submissions": len(report["generated_submissions"]),
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
