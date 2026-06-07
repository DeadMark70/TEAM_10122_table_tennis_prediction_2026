from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v194_train_test_split_distribution_audit import phase_from_prefix_len
from analysis_v279_joint_action_point_candidate_pool import action_family, point_depth
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"
V279_PATH = ROOT / "v279_joint_action_point_candidate_pool" / "v279_pair_candidates.csv"
OUTDIR = ROOT / "v282_joint_context_support_optimizer"
UPLOAD_DIR = ROOT / "upload_candidates_20260519"
ANCHOR_PATH = (
    ROOT
    / "v261_action_conditioned_point_residual"
    / "submission_v261_cap0p01__v173action_r121server.csv"
)
REQUIRED_SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845


@dataclass(frozen=True)
class TransitionTables:
    tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    max_changed_rows: int
    require_both_changed: bool = False
    require_nonterminal: bool = False
    min_support_count: int = 5
    min_pair_prob: float = 0.0


PROFILES = [
    CandidateProfile("v282_support_churn0p010", math.floor(EXPECTED_ROWS * 0.010), min_support_count=8),
    CandidateProfile("v282_support_churn0p020", math.floor(EXPECTED_ROWS * 0.020), min_support_count=8),
    CandidateProfile(
        "v282_support_both_churn0p010",
        math.floor(EXPECTED_ROWS * 0.010),
        require_both_changed=True,
        min_support_count=5,
    ),
    CandidateProfile(
        "v282_support_nonterminal_churn0p010",
        math.floor(EXPECTED_ROWS * 0.010),
        require_nonterminal=True,
        min_support_count=8,
    ),
    CandidateProfile(
        "v282_support_highconfidence",
        25,
        min_support_count=10,
        min_pair_prob=0.02,
    ),
]

LEVEL_ORDER = [
    "phase_action_point",
    "phase_action",
    "action_point",
    "phase_family_depth",
    "global",
]


def transition_examples(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {sorted(missing)}")

    for _, rally in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        rally = rally.reset_index(drop=True)
        if len(rally) < 2:
            continue
        for idx in range(len(rally) - 1):
            cur = rally.iloc[idx]
            nxt = rally.iloc[idx + 1]
            last_action = int(cur["actionId"])
            last_point = int(cur["pointId"])
            next_action = int(nxt["actionId"])
            next_point = int(nxt["pointId"])
            rows.append(
                {
                    "rally_uid": int(cur["rally_uid"]),
                    "phase": phase_from_prefix_len(int(cur["strikeNumber"])),
                    "last_action": last_action,
                    "last_point": last_point,
                    "last_family": action_family(last_action),
                    "last_depth": point_depth(last_point),
                    "candidate_action": next_action,
                    "candidate_point": next_point,
                }
            )
    return pd.DataFrame(rows)


def _count_table(examples: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    group_cols = key_cols + ["candidate_action", "candidate_point"]
    pair_counts = examples.groupby(group_cols, dropna=False).size().reset_index(name="pair_count")
    support = examples.groupby(key_cols, dropna=False).size().reset_index(name="support_count")
    out = pair_counts.merge(support, on=key_cols, how="left")
    return out


def build_transition_tables(train: pd.DataFrame) -> TransitionTables:
    examples = transition_examples(train)
    if examples.empty:
        raise ValueError("No transition examples can be built from train.csv")
    global_examples = examples.copy()
    global_examples["global_key"] = "global"
    return TransitionTables(
        {
            "phase_action_point": _count_table(examples, ["phase", "last_action", "last_point"]),
            "phase_action": _count_table(examples, ["phase", "last_action"]),
            "action_point": _count_table(examples, ["last_action", "last_point"]),
            "phase_family_depth": _count_table(examples, ["phase", "last_family", "last_depth"]),
            "global": _count_table(global_examples, ["global_key"]),
        }
    )


def _lookup(
    table: pd.DataFrame,
    key_values: dict[str, object],
    candidate_action: int,
    candidate_point: int,
) -> tuple[int, int, float] | None:
    mask = np.ones(len(table), dtype=bool)
    for col, value in key_values.items():
        mask &= table[col].to_numpy() == value
    subset = table.loc[mask]
    if subset.empty:
        return None
    support_count = int(subset["support_count"].iloc[0])
    pair = subset[
        (subset["candidate_action"] == candidate_action)
        & (subset["candidate_point"] == candidate_point)
    ]
    pair_count = int(pair["pair_count"].iloc[0]) if not pair.empty else 0
    pair_prob = float(pair_count / support_count) if support_count else 0.0
    return support_count, pair_count, pair_prob


def context_support_features(
    tables: TransitionTables,
    phase: str,
    last_action: int,
    last_point: int,
    candidate_action: int,
    candidate_point: int,
) -> dict[str, object]:
    lookups = {
        "phase_action_point": {
            "phase": phase,
            "last_action": int(last_action),
            "last_point": int(last_point),
        },
        "phase_action": {"phase": phase, "last_action": int(last_action)},
        "action_point": {"last_action": int(last_action), "last_point": int(last_point)},
        "phase_family_depth": {
            "phase": phase,
            "last_family": action_family(int(last_action)),
            "last_depth": point_depth(int(last_point)),
        },
        "global": {"global_key": "global"},
    }
    for level in LEVEL_ORDER:
        found = _lookup(
            tables.tables[level],
            lookups[level],
            int(candidate_action),
            int(candidate_point),
        )
        if found is None:
            continue
        support_count, pair_count, pair_prob = found
        return {
            "support_level": level,
            "support_count": support_count,
            "pair_count": pair_count,
            "pair_prob": pair_prob,
        }
    raise RuntimeError("Global transition table lookup failed")


def latest_prefix_context(test: pd.DataFrame) -> pd.DataFrame:
    rows = (
        test[["rally_uid", "strikeNumber", "actionId", "pointId"]]
        .sort_values(["rally_uid", "strikeNumber"])
        .drop_duplicates("rally_uid", keep="last")
        .copy()
    )
    rows["phase"] = rows["strikeNumber"].map(phase_from_prefix_len)
    rows = rows.rename(columns={"actionId": "last_action", "pointId": "last_point"})
    return rows[["rally_uid", "phase", "last_action", "last_point"]]


def add_context_support(candidates: pd.DataFrame, tables: TransitionTables) -> pd.DataFrame:
    context = latest_prefix_context(pd.read_csv(TEST_PATH))
    out = candidates.merge(context, on="rally_uid", how="left", validate="many_to_one")
    if out[["phase", "last_action", "last_point"]].isna().any().any():
        raise ValueError("Missing test context after rally_uid merge")

    support_rows = [
        context_support_features(
            tables,
            phase=str(row.phase),
            last_action=int(row.last_action),
            last_point=int(row.last_point),
            candidate_action=int(row.candidate_action),
            candidate_point=int(row.candidate_point),
        )
        for row in out.itertuples(index=False)
    ]
    support_df = pd.DataFrame(support_rows)
    return pd.concat([out.reset_index(drop=True), support_df], axis=1)


def support_level_bonus(level: str) -> float:
    return {
        "phase_action_point": 1.00,
        "phase_action": 0.75,
        "action_point": 0.65,
        "phase_family_depth": 0.45,
        "global": 0.00,
    }.get(str(level), 0.0)


def add_v282_utility(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["support_bonus"] = (
        out["support_level"].map(support_level_bonus).astype(float)
        + np.log1p(out["support_count"].astype(float)) * 0.10
        + out["pair_prob"].astype(float) * 8.0
        + np.log1p(out["pair_count"].astype(float)) * 0.35
    )
    out["support_penalty"] = 0.0
    out.loc[out["pair_count"] == 0, "support_penalty"] += 2.0
    out.loc[(out["support_count"] < 5) & out["pair_changed"], "support_penalty"] += 0.7
    out.loc[(out["candidate_point"] == 0) & (out["anchor_point"] != 0), "support_penalty"] += 1.2
    out.loc[out["support_level"].eq("global") & out["pair_changed"], "support_penalty"] += 1.0
    out["v282_utility"] = out["utility"] + out["support_bonus"] - out["support_penalty"]
    return out


def compact_distribution(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)


def load_anchor() -> pd.DataFrame:
    anchor = pd.read_csv(ANCHOR_PATH)
    missing = [c for c in REQUIRED_SUBMISSION_COLUMNS if c not in anchor.columns]
    if missing:
        raise ValueError(f"Anchor missing columns: {missing}")
    anchor = anchor[REQUIRED_SUBMISSION_COLUMNS].copy()
    if len(anchor) != EXPECTED_ROWS:
        raise ValueError(f"Anchor row count {len(anchor)} != {EXPECTED_ROWS}")
    return anchor


def add_basic_utility(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    action_support = out["action_agreement_count"].astype(float) / out["action_source_count"].clip(lower=1).astype(float)
    point_support = out["point_agreement_count"].astype(float) / out["point_source_count"].clip(lower=1).astype(float)
    pair_support = out["pair_agreement_count"].astype(float)
    churn = out["action_changed"].astype(float) * 0.25 + out["point_changed"].astype(float) * 0.25
    both_bonus = (out["action_changed"] & out["point_changed"]).astype(float) * 0.20
    out["utility"] = (
        2.0 * action_support
        + 2.0 * point_support
        + 0.25 * pair_support
        + out["compatibility_score"].astype(float)
        + both_bonus
        - churn
    )
    out["point0_added"] = (out["candidate_point"] == 0) & (out["anchor_point"] != 0)
    return out


def eligible(scored: pd.DataFrame, profile: CandidateProfile) -> pd.DataFrame:
    rows = scored[scored["pair_changed"]].copy()
    rows = rows[~rows["candidate_action"].between(15, 18)]
    rows = rows[rows["point0_added"] == False]  # noqa: E712
    rows = rows[rows["support_count"] >= profile.min_support_count]
    rows = rows[rows["pair_prob"] >= profile.min_pair_prob]
    rows = rows[rows["pair_count"] > 0]
    rows = rows[rows["v282_utility"] > 0]
    if profile.require_both_changed:
        rows = rows[rows["action_changed"] & rows["point_changed"]]
    if profile.require_nonterminal:
        rows = rows[(rows["candidate_point"] != 0) & (rows["anchor_point"] != 0)]
    return rows


def select_profile(scored: pd.DataFrame, profile: CandidateProfile) -> pd.DataFrame:
    rows = eligible(scored, profile)
    if rows.empty:
        return rows.head(0).copy()
    rows = rows.sort_values(
        [
            "v282_utility",
            "pair_prob",
            "pair_count",
            "support_count",
            "pair_agreement_count",
            "compatibility_score",
        ],
        ascending=[False, False, False, False, False, False],
    )
    return rows.drop_duplicates("rally_uid").head(profile.max_changed_rows).copy()


def export_submission(anchor: pd.DataFrame, selected: pd.DataFrame, filename: str) -> Path:
    sub = anchor.copy()
    if not selected.empty:
        repl = selected.set_index("rally_uid")[["candidate_action", "candidate_point"]]
        repl_dict = repl.to_dict("index")
        mask = sub["rally_uid"].isin(repl.index)
        sub.loc[mask, "actionId"] = sub.loc[mask, "rally_uid"].map(
            lambda rid: int(repl_dict[int(rid)]["candidate_action"])
        )
        sub.loc[mask, "pointId"] = sub.loc[mask, "rally_uid"].map(
            lambda rid: int(repl_dict[int(rid)]["candidate_point"])
        )
    sub = sub[REQUIRED_SUBMISSION_COLUMNS]
    if len(sub) != EXPECTED_ROWS:
        raise ValueError(f"{filename} row count {len(sub)} != {EXPECTED_ROWS}")
    path = OUTDIR / filename
    sub.to_csv(path, index=False)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, UPLOAD_DIR / filename)
    return path


def summarize(profile: CandidateProfile, selected: pd.DataFrame) -> dict[str, object]:
    changed = int(len(selected))
    point0_added = int(((selected["candidate_point"] == 0) & (selected["anchor_point"] != 0)).sum()) if changed else 0
    serve_count = int(selected["candidate_action"].between(15, 18).sum()) if changed else 0
    both = int((selected["action_changed"] & selected["point_changed"]).sum()) if changed else 0
    if serve_count:
        verdict = "reject_serve"
    elif point0_added:
        verdict = "reject_point0_added"
    elif changed < 10:
        verdict = "micro_edit_not_material"
    elif both == 0 and "both" in profile.name:
        verdict = "reject_no_joint_rows"
    elif selected["support_level"].eq("global").any() if changed else False:
        verdict = "review_global_support"
    else:
        verdict = "candidate_for_review"
    return {
        "candidate": profile.name,
        "changed_rows": changed,
        "action_changed_rows": int(selected["action_changed"].sum()) if changed else 0,
        "point_changed_rows": int(selected["point_changed"].sum()) if changed else 0,
        "both_changed_rows": both,
        "point0_added_rows": point0_added,
        "serve_pred_count": serve_count,
        "mean_v282_utility": float(selected["v282_utility"].mean()) if changed else 0.0,
        "min_v282_utility": float(selected["v282_utility"].min()) if changed else 0.0,
        "mean_pair_prob": float(selected["pair_prob"].mean()) if changed else 0.0,
        "min_pair_prob": float(selected["pair_prob"].min()) if changed else 0.0,
        "min_support_count": int(selected["support_count"].min()) if changed else 0,
        "support_levels": compact_distribution(selected["support_level"].astype("category").cat.codes) if changed else "{}",
        "action_distribution": compact_distribution(selected["candidate_action"]) if changed else "{}",
        "point_distribution": compact_distribution(selected["candidate_point"]) if changed else "{}",
        "verdict": verdict,
    }


def write_report(search: pd.DataFrame) -> None:
    viable = search[
        (search["serve_pred_count"] == 0)
        & (search["point0_added_rows"] == 0)
        & (search["changed_rows"] >= 10)
        & (search["verdict"].eq("candidate_for_review"))
    ].copy()
    if viable.empty:
        rec = "none"
    else:
        viable = viable.sort_values(["both_changed_rows", "mean_v282_utility"], ascending=[False, False])
        rec = str(viable.iloc[0]["candidate"])
    lines = [
        "# V282 Joint Context-Support Optimizer",
        "",
        "status: ok",
        "",
        f"recommended_candidate: {rec}",
        "",
        "Notes:",
        "- Adds train-derived backoff transition support to V280 source agreement.",
        "- TTMATCH is not read.",
        "- Utility remains a proxy; public upload still requires review.",
        "",
    ]
    (OUTDIR / "v282_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not V279_PATH.exists():
        (OUTDIR / "v282_report.md").write_text("status: waiting_for_v279\n", encoding="utf-8")
        print(json.dumps({"outdir": OUTDIR.name, "status": "waiting_for_v279"}))
        return
    train = pd.read_csv(TRAIN_PATH)
    candidates = pd.read_csv(V279_PATH)
    candidates = add_basic_utility(candidates)
    tables = build_transition_tables(train)
    supported = add_context_support(candidates, tables)
    scored = add_v282_utility(supported)
    scored.to_csv(OUTDIR / "v282_scored_pair_candidates.csv", index=False)

    anchor = load_anchor()
    rows = []
    for profile in PROFILES:
        selected = select_profile(scored, profile)
        export_submission(anchor, selected, f"submission_{profile.name}__sr121.csv")
        rows.append(summarize(profile, selected))
    search = pd.DataFrame(rows)
    search.to_csv(OUTDIR / "v282_pair_search.csv", index=False)
    write_report(search)
    print(json.dumps({"outdir": OUTDIR.name, "candidates": len(PROFILES)}, indent=2))


if __name__ == "__main__":
    main()
