from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from analysis_v279_joint_action_point_candidate_pool import action_family, point_depth
from analysis_v282_joint_context_support_optimizer import (
    EXPECTED_ROWS,
    OUTDIR as V282_OUTDIR,
    REQUIRED_SUBMISSION_COLUMNS,
    TRAIN_PATH,
    UPLOAD_DIR,
    add_context_support,
    add_v282_utility,
    build_transition_tables,
    export_submission,
    load_anchor,
    transition_examples,
)


ROOT = Path(__file__).resolve().parent
V279_PATH = ROOT / "v279_joint_action_point_candidate_pool" / "v279_pair_candidates.csv"
OUTDIR = ROOT / "v283_pair_level_selector"
MAX_TRAIN_EXAMPLES = 15000


FEATURE_COLUMNS = [
    "compatibility_score",
    "action_agreement_count",
    "point_agreement_count",
    "pair_agreement_count",
    "action_source_count",
    "point_source_count",
    "action_changed",
    "point_changed",
    "candidate_action",
    "candidate_point",
    "candidate_family_code",
    "candidate_depth",
    "is_terminal_pair",
    "support_count",
    "pair_count",
    "pair_prob",
    "support_level_code",
    "v282_utility",
]


@dataclass(frozen=True)
class Profile:
    name: str
    max_rows: int
    margin: float
    require_both_changed: bool = False
    require_nonterminal: bool = False
    min_prob: float = 0.0


PROFILES = [
    Profile("v283_pairclf_churn0p005", 9, 0.03),
    Profile("v283_pairclf_churn0p010", 18, 0.04),
    Profile("v283_pairclf_churn0p020", 36, 0.05),
    Profile("v283_pairclf_both_churn0p010", 18, 0.03, require_both_changed=True),
    Profile("v283_pairclf_nonterminal_churn0p010", 18, 0.04, require_nonterminal=True),
]

FAMILY_CODES = {"Zero": 0, "Attack": 1, "Control": 2, "Defensive": 3, "Serve": 4}
SUPPORT_CODES = {
    "global": 0,
    "phase_family_depth": 1,
    "action_point": 2,
    "phase_action": 3,
    "phase_action_point": 4,
}


def build_pair_training_candidates(
    examples: pd.DataFrame,
    candidate_pairs: list[tuple[int, int]],
    max_negative_pairs: int = 8,
) -> pd.DataFrame:
    rows = []
    for ex in examples.itertuples(index=False):
        true_pair = (int(ex.candidate_action), int(ex.candidate_point))
        pairs = [true_pair]
        for pair in candidate_pairs:
            if pair != true_pair:
                pairs.append(pair)
            if len(pairs) >= max_negative_pairs + 1:
                break
        for action_id, point_id in pairs:
            rows.append(
                {
                    "rally_uid": int(ex.rally_uid),
                    "phase": str(ex.phase),
                    "last_action": int(ex.last_action),
                    "last_point": int(ex.last_point),
                    "candidate_action": int(action_id),
                    "candidate_point": int(point_id),
                    "label": int((int(action_id), int(point_id)) == true_pair),
                }
            )
    return pd.DataFrame(rows)


def top_candidate_pairs(examples: pd.DataFrame, n: int = 24) -> list[tuple[int, int]]:
    counts = (
        examples.groupby(["candidate_action", "candidate_point"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    return [
        (int(row.candidate_action), int(row.candidate_point))
        for row in counts.head(n).itertuples(index=False)
    ]


def compatibility_score(action_id: int, point_id: int) -> float:
    family = action_family(int(action_id))
    depth = point_depth(int(point_id))
    if family == "Serve":
        return 0.0
    if int(action_id) == 0:
        return 1.0 if int(point_id) == 0 else 0.0
    if int(point_id) == 0:
        return 0.90 if int(action_id) in {1, 2, 3, 13} else 0.55
    if family == "Attack":
        return {1: 0.68, 2: 0.78, 3: 0.95}.get(depth, 0.70)
    if family == "Control":
        return {1: 0.95, 2: 0.90, 3: 0.72}.get(depth, 0.70)
    if family == "Defensive":
        return {1: 0.45, 2: 0.65, 3: 0.90}.get(depth, 0.70)
    return 0.70


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["compatibility_score"] = [
        compatibility_score(a, p)
        for a, p in zip(out["candidate_action"], out["candidate_point"], strict=False)
    ]
    out["candidate_family_code"] = out["candidate_action"].map(lambda x: FAMILY_CODES[action_family(int(x))])
    out["candidate_depth"] = out["candidate_point"].map(lambda x: point_depth(int(x)))
    out["is_terminal_pair"] = out["candidate_point"].astype(int).eq(0).astype(int)
    return out


def add_train_support_features(train_candidates: pd.DataFrame, tables) -> pd.DataFrame:
    base = add_candidate_features(train_candidates)
    support_rows = []
    from analysis_v282_joint_context_support_optimizer import context_support_features

    for row in base.itertuples(index=False):
        support_rows.append(
            context_support_features(
                tables,
                phase=str(row.phase),
                last_action=int(row.last_action),
                last_point=int(row.last_point),
                candidate_action=int(row.candidate_action),
                candidate_point=int(row.candidate_point),
            )
        )
    support = pd.DataFrame(support_rows)
    out = pd.concat([base.reset_index(drop=True), support], axis=1)
    out["support_level_code"] = out["support_level"].map(SUPPORT_CODES).fillna(0).astype(int)
    out["action_agreement_count"] = 1
    out["point_agreement_count"] = 1
    out["pair_agreement_count"] = out["label"].astype(int)
    out["action_source_count"] = 1
    out["point_source_count"] = 1
    out["action_changed"] = 1
    out["point_changed"] = 1
    out["v282_utility"] = (
        out["compatibility_score"]
        + np.log1p(out["support_count"].astype(float)) * 0.1
        + out["pair_prob"].astype(float) * 8.0
        + np.log1p(out["pair_count"].astype(float)) * 0.35
    )
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_family_code" not in out:
        out["candidate_family_code"] = out["candidate_action"].map(lambda x: FAMILY_CODES[action_family(int(x))])
    if "candidate_depth" not in out:
        out["candidate_depth"] = out["candidate_point"].map(lambda x: point_depth(int(x)))
    if "is_terminal_pair" not in out:
        out["is_terminal_pair"] = out["candidate_point"].astype(int).eq(0).astype(int)
    if "support_level_code" not in out:
        out["support_level_code"] = out["support_level"].map(SUPPORT_CODES).fillna(0).astype(int)
    for col in FEATURE_COLUMNS:
        if col not in out:
            out[col] = 0
    return out[FEATURE_COLUMNS].astype(float)


def train_pair_classifier(training: pd.DataFrame) -> tuple[LogisticRegression, float]:
    groups = training["rally_uid"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=283)
    train_idx, valid_idx = next(splitter.split(training, training["label"], groups=groups))
    x_train = feature_matrix(training.iloc[train_idx])
    y_train = training.iloc[train_idx]["label"].astype(int)
    x_valid = feature_matrix(training.iloc[valid_idx])
    y_valid = training.iloc[valid_idx]["label"].astype(int)
    clf = LogisticRegression(
        max_iter=600,
        solver="liblinear",
        class_weight="balanced",
        random_state=283,
    )
    clf.fit(x_train, y_train)
    valid_prob = clf.predict_proba(x_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, valid_prob)) if y_valid.nunique() == 2 else float("nan")
    final = LogisticRegression(
        max_iter=600,
        solver="liblinear",
        class_weight="balanced",
        random_state=284,
    )
    final.fit(feature_matrix(training), training["label"].astype(int))
    return final, auc


def add_test_predictions(scored: pd.DataFrame, clf: LogisticRegression) -> pd.DataFrame:
    out = scored.copy()
    out["candidate_family_code"] = out["candidate_action"].map(lambda x: FAMILY_CODES[action_family(int(x))])
    out["candidate_depth"] = out["candidate_point"].map(lambda x: point_depth(int(x)))
    out["is_terminal_pair"] = out["candidate_point"].astype(int).eq(0).astype(int)
    out["support_level_code"] = out["support_level"].map(SUPPORT_CODES).fillna(0).astype(int)
    out["pred_correct_prob"] = clf.predict_proba(feature_matrix(out))[:, 1]
    anchor_prob = (
        out[~out["pair_changed"]][["rally_uid", "pred_correct_prob"]]
        .drop_duplicates("rally_uid")
        .rename(columns={"pred_correct_prob": "anchor_pred_correct_prob"})
    )
    out = out.merge(anchor_prob, on="rally_uid", how="left", validate="many_to_one")
    out["pred_margin_vs_anchor"] = out["pred_correct_prob"] - out["anchor_pred_correct_prob"].fillna(0.0)
    return out


def select_improvements(
    scored: pd.DataFrame,
    max_rows: int,
    margin: float,
    require_both_changed: bool = False,
    require_nonterminal: bool = False,
    min_prob: float = 0.0,
) -> pd.DataFrame:
    if "pred_margin_vs_anchor" not in scored.columns:
        anchor_prob = (
            scored[~scored["pair_changed"]][["rally_uid", "pred_correct_prob"]]
            .drop_duplicates("rally_uid")
            .rename(columns={"pred_correct_prob": "anchor_pred_correct_prob"})
        )
        scored = scored.merge(anchor_prob, on="rally_uid", how="left", validate="many_to_one")
        scored["pred_margin_vs_anchor"] = scored["pred_correct_prob"] - scored[
            "anchor_pred_correct_prob"
        ].fillna(0.0)
    rows = scored[scored["pair_changed"]].copy()
    rows = rows[~rows["candidate_action"].between(15, 18)]
    rows = rows[~((rows["candidate_point"] == 0) & (rows["anchor_point"] != 0))]
    rows = rows[rows["pred_margin_vs_anchor"] >= margin]
    rows = rows[rows["pred_correct_prob"] >= min_prob]
    if require_both_changed:
        rows = rows[rows["action_changed"] & rows["point_changed"]]
    if require_nonterminal:
        rows = rows[(rows["candidate_point"] != 0) & (rows["anchor_point"] != 0)]
    if rows.empty:
        return rows.head(0).copy()
    for col in ["v282_utility", "compatibility_score"]:
        if col not in rows.columns:
            rows[col] = 0.0
    rows = rows.sort_values(
        ["pred_margin_vs_anchor", "pred_correct_prob", "v282_utility", "compatibility_score"],
        ascending=[False, False, False, False],
    )
    return rows.drop_duplicates("rally_uid").head(max_rows).copy()


def summarize(profile: Profile, selected: pd.DataFrame) -> dict[str, object]:
    changed = int(len(selected))
    return {
        "candidate": profile.name,
        "changed_rows": changed,
        "action_changed_rows": int(selected["action_changed"].sum()) if changed else 0,
        "point_changed_rows": int(selected["point_changed"].sum()) if changed else 0,
        "both_changed_rows": int((selected["action_changed"] & selected["point_changed"]).sum()) if changed else 0,
        "point0_added_rows": int(((selected["candidate_point"] == 0) & (selected["anchor_point"] != 0)).sum()) if changed else 0,
        "serve_pred_count": int(selected["candidate_action"].between(15, 18).sum()) if changed else 0,
        "mean_pred_correct_prob": float(selected["pred_correct_prob"].mean()) if changed else 0.0,
        "min_pred_margin_vs_anchor": float(selected["pred_margin_vs_anchor"].min()) if changed else 0.0,
        "mean_pred_margin_vs_anchor": float(selected["pred_margin_vs_anchor"].mean()) if changed else 0.0,
        "action_distribution": json.dumps({str(k): int(v) for k, v in selected["candidate_action"].value_counts().sort_index().items()}) if changed else "{}",
        "point_distribution": json.dumps({str(k): int(v) for k, v in selected["candidate_point"].value_counts().sort_index().items()}) if changed else "{}",
        "verdict": "candidate_for_review" if changed >= 10 else "micro_or_empty",
    }


def write_submission(anchor: pd.DataFrame, selected: pd.DataFrame, name: str) -> None:
    # Reuse V282's tested exporter.
    export_submission(anchor, selected, f"submission_{name}__sr121.csv")
    src = V282_OUTDIR / f"submission_{name}__sr121.csv"
    dst = OUTDIR / f"submission_{name}__sr121.csv"
    shutil.copy2(src, dst)
    shutil.copy2(dst, UPLOAD_DIR / dst.name)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not V279_PATH.exists():
        (OUTDIR / "v283_report.md").write_text("status: waiting_for_v279\n", encoding="utf-8")
        print(json.dumps({"outdir": OUTDIR.name, "status": "waiting_for_v279"}))
        return

    train = pd.read_csv(TRAIN_PATH)
    examples = transition_examples(train)
    if len(examples) > MAX_TRAIN_EXAMPLES:
        examples = examples.sample(n=MAX_TRAIN_EXAMPLES, random_state=283).reset_index(drop=True)
    pairs = top_candidate_pairs(examples, n=28)
    training_candidates = build_pair_training_candidates(examples, pairs, max_negative_pairs=8)
    tables = build_transition_tables(train)
    training = add_train_support_features(training_candidates, tables)
    clf, valid_auc = train_pair_classifier(training)

    scored = pd.read_csv(V282_OUTDIR / "v282_scored_pair_candidates.csv")
    predicted = add_test_predictions(scored, clf)
    predicted.to_csv(OUTDIR / "v283_scored_pair_candidates.csv", index=False)

    anchor = load_anchor()
    search_rows = []
    for profile in PROFILES:
        selected = select_improvements(
            predicted,
            max_rows=profile.max_rows,
            margin=profile.margin,
            require_both_changed=profile.require_both_changed,
            require_nonterminal=profile.require_nonterminal,
            min_prob=profile.min_prob,
        )
        write_submission(anchor, selected, profile.name)
        search_rows.append(summarize(profile, selected))

    search = pd.DataFrame(search_rows)
    search["validation_auc"] = valid_auc
    search.to_csv(OUTDIR / "v283_pair_search.csv", index=False)
    recommendation = "none"
    viable = search[(search["changed_rows"] >= 10) & (search["point0_added_rows"] == 0) & (search["serve_pred_count"] == 0)]
    if not viable.empty:
        recommendation = str(viable.sort_values(["both_changed_rows", "mean_pred_margin_vs_anchor"], ascending=[False, False]).iloc[0]["candidate"])
    (OUTDIR / "v283_report.md").write_text(
        "\n".join(
            [
                "# V283 Pair-Level Selector",
                "",
                "status: ok",
                "",
                f"validation_auc: {valid_auc:.6f}",
                f"recommended_candidate: {recommendation}",
                "",
                "Notes:",
                "- Trains a pair-level classifier from train transitions.",
                "- Selects only candidates whose predicted correctness beats the anchor pair by a margin.",
                "- TTMATCH is not read.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"outdir": OUTDIR.name, "candidates": len(PROFILES), "validation_auc": valid_auc}, indent=2))


if __name__ == "__main__":
    main()
