from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V279_PATH = ROOT / "v279_joint_action_point_candidate_pool" / "v279_pair_candidates.csv"
OUTDIR = ROOT / "v280_joint_action_point_optimizer"
UPLOAD_DIR = ROOT / "upload_candidates_20260519"
ANCHOR_PATH = (
    ROOT
    / "v261_action_conditioned_point_residual"
    / "submission_v261_cap0p01__v173action_r121server.csv"
)
REQUIRED_SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845

ACTION_COUNTS: dict[int, int] = {}
POINT_COUNTS: dict[int, int] = {}


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    max_changed_rows: int
    high_confidence: bool = False
    require_both_changed: bool = False


PROFILES = [
    CandidateProfile("v280_joint_churn0p005", math.floor(EXPECTED_ROWS * 0.005)),
    CandidateProfile("v280_joint_churn0p010", math.floor(EXPECTED_ROWS * 0.010)),
    CandidateProfile("v280_joint_churn0p020", math.floor(EXPECTED_ROWS * 0.020)),
    CandidateProfile("v280_joint_highconfidence", 25, high_confidence=True),
    CandidateProfile("v280_joint_both_churn0p010", math.floor(EXPECTED_ROWS * 0.010), require_both_changed=True),
]


def class_rarity_weight(class_id: int, counts: dict[int, int]) -> float:
    return 1.0 / np.sqrt(max(counts.get(class_id, 1), 1))


def _numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def _bool(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if pd.isna(value):
        return False
    return bool(value)


def _rarity_bonus(class_id: int, counts: dict[int, int]) -> float:
    weights = [class_rarity_weight(k, counts) for k in counts] or [1.0]
    median_weight = float(np.median(weights))
    return max(0.0, class_rarity_weight(class_id, counts) - median_weight)


def action_utility(row: pd.Series) -> float:
    agreement = _numeric(row, "action_agreement_count", 1.0)
    source_count = max(_numeric(row, "action_source_count", 1.0), 1.0)
    changed = _bool(row, "action_changed")
    candidate_action = int(row["candidate_action"])

    support = agreement / source_count
    utility = 0.24 * agreement + 0.42 * support
    if changed:
        utility += 0.10
    else:
        utility -= 0.22

    rarity = _rarity_bonus(candidate_action, ACTION_COUNTS)
    if agreement >= 2:
        utility += 1.35 * rarity
    elif changed and rarity > 0:
        utility -= 0.20 + 0.75 * rarity

    if 15 <= candidate_action <= 18:
        utility -= 4.0
    if candidate_action == 0 and int(row["candidate_point"]) != 0:
        utility -= 3.0

    return float(utility)


def point_utility(row: pd.Series) -> float:
    agreement = _numeric(row, "point_agreement_count", 1.0)
    source_count = max(_numeric(row, "point_source_count", 1.0), 1.0)
    changed = _bool(row, "point_changed")
    candidate_point = int(row["candidate_point"])
    anchor_point = int(row["anchor_point"])

    support = agreement / source_count
    utility = 0.24 * agreement + 0.42 * support
    if changed:
        utility += 0.10
    else:
        utility -= 0.22

    rarity = _rarity_bonus(candidate_point, POINT_COUNTS)
    if agreement >= 2:
        utility += 1.20 * rarity
    elif changed and rarity > 0:
        utility -= 0.18 + 0.65 * rarity

    if changed and candidate_point == 0 and anchor_point != 0:
        utility -= 0.85
    elif changed and candidate_point != 0:
        utility += 0.12

    return float(utility)


def churn_penalty(row: pd.Series) -> float:
    penalty = 0.0
    action_changed = _bool(row, "action_changed")
    point_changed = _bool(row, "point_changed")
    if action_changed:
        penalty += 0.16
    if point_changed:
        penalty += 0.16
    if action_changed and point_changed:
        penalty += 0.10

    candidate_action = int(row["candidate_action"])
    candidate_point = int(row["candidate_point"])
    anchor_point = int(row["anchor_point"])
    compatibility = _numeric(row, "compatibility_score", 0.0)
    pair_agreement = _numeric(row, "pair_agreement_count", 0.0)

    if candidate_point == 0 and anchor_point != 0:
        penalty += 0.55
        if compatibility < 0.85 or pair_agreement < 2:
            penalty += 1.25
    if 15 <= candidate_action <= 18:
        penalty += 5.0
    if candidate_action in {10, 11, 12} and candidate_point == 0 and pair_agreement < 3:
        penalty += 0.60
    return float(penalty)


def pair_utility(row: pd.Series) -> float:
    pair_agreement = _numeric(row, "pair_agreement_count", 1.0)
    agreement_bonus = 0.18 * max(0.0, pair_agreement - 1.0)
    if pair_agreement >= 2:
        agreement_bonus += 0.24
    return float(
        0.4 * action_utility(row)
        + 0.4 * point_utility(row)
        + 0.2 * _numeric(row, "compatibility_score", 0.0)
        + agreement_bonus
        - churn_penalty(row)
    )


def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def write_waiting_report() -> None:
    ensure_outdir()
    report = "\n".join(
        [
            "# V280 Joint Action-Point Optimizer",
            "",
            "status: waiting_for_v279",
            "",
            f"Missing input: `{V279_PATH.relative_to(ROOT)}`",
            "",
            "No submissions were generated. Re-run this script after V279 writes the candidate pool.",
            "",
        ]
    )
    (OUTDIR / "v280_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"outdir": OUTDIR.name, "status": "waiting_for_v279"}))


def load_anchor() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    missing = [c for c in REQUIRED_SUBMISSION_COLUMNS if c not in anchor.columns]
    if missing:
        raise ValueError(f"Anchor missing columns: {missing}")
    anchor = anchor[REQUIRED_SUBMISSION_COLUMNS].copy()
    if len(anchor) != EXPECTED_ROWS:
        raise ValueError(f"Anchor row count {len(anchor)} != {EXPECTED_ROWS}")
    if anchor["rally_uid"].duplicated().any():
        raise ValueError("Anchor rally_uid contains duplicates")
    return anchor


def load_candidates() -> pd.DataFrame:
    candidates = pd.read_csv(V279_PATH)
    required = {
        "rally_uid",
        "anchor_action",
        "anchor_point",
        "candidate_action",
        "candidate_point",
        "compatibility_score",
        "action_changed",
        "point_changed",
        "pair_changed",
        "action_agreement_count",
        "point_agreement_count",
        "pair_agreement_count",
        "action_source_count",
        "point_source_count",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V279 candidate pool missing columns: {missing}")
    for column in [
        "anchor_action",
        "anchor_point",
        "candidate_action",
        "candidate_point",
        "action_agreement_count",
        "point_agreement_count",
        "pair_agreement_count",
        "action_source_count",
        "point_source_count",
    ]:
        candidates[column] = candidates[column].astype(int)
    for column in ["action_changed", "point_changed", "pair_changed"]:
        candidates[column] = candidates[column].astype(bool)
    return candidates


def add_utilities(candidates: pd.DataFrame) -> pd.DataFrame:
    global ACTION_COUNTS, POINT_COUNTS
    anchor_pairs = (
        candidates[["rally_uid", "anchor_action", "anchor_point"]]
        .drop_duplicates("rally_uid")
        .copy()
    )
    ACTION_COUNTS = anchor_pairs["anchor_action"].value_counts().astype(int).to_dict()
    POINT_COUNTS = anchor_pairs["anchor_point"].value_counts().astype(int).to_dict()

    scored = candidates.copy()
    scored["utility"] = scored.apply(pair_utility, axis=1)
    scored["action_utility"] = scored.apply(action_utility, axis=1)
    scored["point_utility"] = scored.apply(point_utility, axis=1)
    scored["churn_penalty"] = scored.apply(churn_penalty, axis=1)
    scored["point0_added"] = (
        (scored["candidate_point"] == 0) & (scored["anchor_point"] != 0)
    )
    scored["serve_pred"] = scored["candidate_action"].between(15, 18)
    return scored


def eligible_candidates(scored: pd.DataFrame, profile: CandidateProfile) -> pd.DataFrame:
    eligible = scored[scored["pair_changed"]].copy()
    if profile.require_both_changed:
        eligible = eligible[eligible["action_changed"] & eligible["point_changed"]]
    eligible = eligible[~eligible["candidate_action"].between(15, 18)]
    eligible = eligible[
        ~((eligible["candidate_action"] == 0) & (eligible["candidate_point"] != 0))
    ]
    eligible = eligible[
        ~(
            (eligible["candidate_point"] == 0)
            & (eligible["anchor_point"] != 0)
            & (
                (eligible["compatibility_score"] < 0.85)
                | (eligible["pair_agreement_count"] < 2)
            )
        )
    ]
    eligible = eligible[eligible["utility"] > 0]
    if profile.high_confidence:
        eligible = eligible[
            (eligible["pair_agreement_count"] >= 2)
            | (
                eligible["action_agreement_count"]
                + eligible["point_agreement_count"]
                >= 3
            )
        ]
    return eligible


def select_profile(scored: pd.DataFrame, profile: CandidateProfile) -> pd.DataFrame:
    eligible = eligible_candidates(scored, profile)
    if eligible.empty:
        return eligible.head(0).copy()
    eligible = eligible.sort_values(
        [
            "utility",
            "pair_agreement_count",
            "action_agreement_count",
            "point_agreement_count",
            "compatibility_score",
        ],
        ascending=[False, False, False, False, False],
    )
    best_per_rally = eligible.drop_duplicates("rally_uid", keep="first")
    return best_per_rally.head(profile.max_changed_rows).copy()


def export_submission(anchor: pd.DataFrame, selected: pd.DataFrame, filename: str) -> Path:
    submission = anchor.copy()
    if not selected.empty:
        replacements = selected.set_index("rally_uid")[
            ["candidate_action", "candidate_point"]
        ]
        mask = submission["rally_uid"].isin(replacements.index)
        replacement_rows = submission.loc[mask, "rally_uid"].map(replacements.to_dict("index"))
        submission.loc[mask, "actionId"] = replacement_rows.map(
            lambda value: int(value["candidate_action"])
        )
        submission.loc[mask, "pointId"] = replacement_rows.map(
            lambda value: int(value["candidate_point"])
        )
    submission = submission[REQUIRED_SUBMISSION_COLUMNS]
    if len(submission) != EXPECTED_ROWS:
        raise ValueError(f"{filename} row count {len(submission)} != {EXPECTED_ROWS}")
    if list(submission.columns) != REQUIRED_SUBMISSION_COLUMNS:
        raise ValueError(f"{filename} has wrong columns: {submission.columns.tolist()}")
    path = OUTDIR / filename
    submission.to_csv(path, index=False)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, UPLOAD_DIR / filename)
    return path


def compact_distribution(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    return json.dumps({str(int(k)): int(v) for k, v in counts.items()}, sort_keys=True)


def summarize_profile(profile: CandidateProfile, selected: pd.DataFrame) -> dict[str, object]:
    changed_rows = int(len(selected))
    if changed_rows == 0:
        mean_utility = 0.0
        min_utility = 0.0
    else:
        mean_utility = float(selected["utility"].mean())
        min_utility = float(selected["utility"].min())

    point0_added_rows = int(
        ((selected["candidate_point"] == 0) & (selected["anchor_point"] != 0)).sum()
    )
    serve_pred_count = int(selected["candidate_action"].between(15, 18).sum())
    if serve_pred_count:
        verdict = "reject_serve_predictions"
    elif point0_added_rows > 3:
        verdict = "reject_point0_heavy"
    elif changed_rows < 10:
        verdict = "micro_edit_not_material"
    elif changed_rows <= 80 and mean_utility > 0:
        verdict = "candidate_for_public_probe"
    else:
        verdict = "review_required"

    return {
        "candidate": profile.name,
        "changed_rows": changed_rows,
        "action_changed_rows": int(selected["action_changed"].sum()) if changed_rows else 0,
        "point_changed_rows": int(selected["point_changed"].sum()) if changed_rows else 0,
        "both_changed_rows": int((selected["action_changed"] & selected["point_changed"]).sum())
        if changed_rows
        else 0,
        "point0_added_rows": point0_added_rows,
        "serve_pred_count": serve_pred_count,
        "mean_utility": mean_utility,
        "min_utility": min_utility,
        "action_distribution": compact_distribution(selected["candidate_action"])
        if changed_rows
        else "{}",
        "point_distribution": compact_distribution(selected["candidate_point"])
        if changed_rows
        else "{}",
        "verdict": verdict,
    }


def write_report(search: pd.DataFrame) -> None:
    viable = search[
        (search["serve_pred_count"] == 0)
        & (search["point0_added_rows"] <= 3)
        & (search["changed_rows"] >= 10)
        & (search["mean_utility"] > 0)
    ].copy()
    if viable.empty:
        recommendation = "none"
    else:
        viable = viable.sort_values(
            ["mean_utility", "changed_rows"], ascending=[False, False]
        )
        recommendation = str(viable.iloc[0]["candidate"])

    rejected = []
    for row in search.to_dict("records"):
        reasons = []
        if row["serve_pred_count"] > 0:
            reasons.append("serve predictions present")
        if row["point0_added_rows"] > 3:
            reasons.append("point0 additions exceed clean threshold")
        if row["changed_rows"] < 10:
            reasons.append("micro-edit under 10 changed rows")
        if row["mean_utility"] <= 0:
            reasons.append("non-positive mean utility")
        if reasons:
            rejected.append(f"- {row['candidate']}: {', '.join(reasons)}")
    if not rejected:
        rejected.append("- none")

    best_utility = float(search["mean_utility"].max()) if len(search) else 0.0
    material = (
        "yes"
        if (recommendation != "none" and int(search["changed_rows"].max()) >= 10)
        else "no"
    )
    report = "\n".join(
        [
            "# V280 Joint Action-Point Optimizer",
            "",
            "status: ok",
            "",
            f"recommended_candidate: {recommendation}",
            "",
            "rejected_candidate_reasons:",
            *rejected,
            "",
            "materially_stronger_than_v277_micro_edit:",
            f"- {material}; best mean proxy utility = {best_utility:.6f}",
            "",
            "notes:",
            "- Utility is a macro-F1 proxy, not a claimed leaderboard estimate.",
            "- TTMATCH is not read or used by this script.",
            "- Final submissions use the V261/R121 anchor serverGetPoint values.",
            "",
        ]
    )
    (OUTDIR / "v280_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_outdir()
    if not V279_PATH.exists():
        write_waiting_report()
        return

    anchor = load_anchor()
    candidates = load_candidates()
    scored = add_utilities(candidates)
    scored.to_csv(OUTDIR / "v280_scored_pair_candidates.csv", index=False)

    search_rows = []
    for profile in PROFILES:
        selected = select_profile(scored, profile)
        filename = f"submission_{profile.name}__sr121.csv"
        export_submission(anchor, selected, filename)
        search_rows.append(summarize_profile(profile, selected))

    search = pd.DataFrame(search_rows)
    search.to_csv(OUTDIR / "v280_pair_search.csv", index=False)
    write_report(search)
    print(json.dumps({"outdir": OUTDIR.name, "candidates": len(PROFILES)}))


if __name__ == "__main__":
    main()
