"""V220 train-backoff supported action filter.

V220 turns manual changed-row inspection into a reproducible filter.  It scans
the union of V217/V218/V219 action edits, compares each proposed action against
the V173 anchor using fold-free train next-action backoff statistics, and
exports only supported low-churn action changes.

Point remains V188 cap5 and server remains R121.  No external rows and no
TTMATCH are read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v220_action_backoff_support_filter")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v220_action_backoff_support_filter.py")
ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"

SOURCE_FILES = [
    UPLOAD_DIR / "submission_v217_macro_utility_churn0p005__pv188cap5__sr121.csv",
    UPLOAD_DIR / "submission_v217_macro_utility_churn0p01__pv188cap5__sr121.csv",
    UPLOAD_DIR / "submission_v218_weak_all_cap0p005__pv188cap5__sr121.csv",
    UPLOAD_DIR / "submission_v219_class_budget_s1p0__pv188cap5__sr121.csv",
    UPLOAD_DIR / "submission_v219_rare_budget_s1p0__pv188cap5__sr121.csv",
]

ACTION_NAMES = {
    0: "zero",
    1: "drive",
    2: "counter_drive",
    3: "smash",
    4: "backhand_twist",
    5: "fast_drive",
    6: "fast_push",
    7: "flip",
    8: "pimple_long_push",
    9: "pimple_fast_push",
    10: "long_push",
    11: "drop_shot",
    12: "chop",
    13: "block",
    14: "lob",
    15: "traditional_serve",
    16: "hook_serve",
    17: "reverse_serve",
    18: "squat_serve",
}


def point_depth(point_id: int) -> int:
    p = int(point_id)
    if p == 0:
        return 0
    if p in (1, 2, 3):
        return 1
    if p in (4, 5, 6):
        return 2
    if p in (7, 8, 9):
        return 3
    return -1


def phase_name(prefix_len: int) -> str:
    n = int(prefix_len)
    if n <= 1:
        return "receive"
    if n == 2:
        return "third_ball"
    if n == 3:
        return "fourth_ball"
    return "rally"


def build_next_action_examples(train: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, g in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        for i in range(len(g) - 1):
            lag = g.iloc[i]
            nxt = g.iloc[i + 1]
            records.append(
                {
                    "phase": phase_name(int(lag["strikeNumber"])),
                    "lag0_action": int(lag["actionId"]),
                    "lag0_point": int(lag["pointId"]),
                    "lag0_depth": point_depth(int(lag["pointId"])),
                    "lag0_spin": int(lag["spinId"]),
                    "lag0_strength": int(lag["strengthId"]),
                    "next_action": int(nxt["actionId"]),
                }
            )
    return pd.DataFrame(records)


def _counts_for(
    examples: pd.DataFrame,
    condition: dict,
    base_action: int,
    cand_action: int,
) -> dict:
    sub = examples
    for key, value in condition.items():
        sub = sub[sub[key].eq(value)]
    n = int(len(sub))
    if n == 0:
        return {
            "n": 0,
            "base_count": 0,
            "cand_count": 0,
            "base_rate": 0.0,
            "cand_rate": 0.0,
            "margin": 0.0,
            "top": "",
        }
    vc = sub["next_action"].value_counts()
    base_count = int(vc.get(int(base_action), 0))
    cand_count = int(vc.get(int(cand_action), 0))
    top = ",".join([f"{int(a)}:{int(c)}" for a, c in vc.head(5).items()])
    return {
        "n": n,
        "base_count": base_count,
        "cand_count": cand_count,
        "base_rate": base_count / n,
        "cand_rate": cand_count / n,
        "margin": (cand_count - base_count) / n,
        "top": top,
    }


def backoff_support_score(
    examples: pd.DataFrame,
    phase: str,
    lag0_action: int,
    lag0_point: int,
    lag0_depth: int,
    lag0_spin: int,
    lag0_strength: int,
    base_action: int,
    cand_action: int,
    min_support: int = 20,
) -> tuple[int, list[dict]]:
    """Score whether train backoff statistics support cand over base."""
    levels = [
        (
            "exact",
            {
                "phase": phase,
                "lag0_action": int(lag0_action),
                "lag0_point": int(lag0_point),
                "lag0_spin": int(lag0_spin),
                "lag0_strength": int(lag0_strength),
            },
        ),
        ("phase_action_point", {"phase": phase, "lag0_action": int(lag0_action), "lag0_point": int(lag0_point)}),
        ("phase_action_depth", {"phase": phase, "lag0_action": int(lag0_action), "lag0_depth": int(lag0_depth)}),
        ("action_point", {"lag0_action": int(lag0_action), "lag0_point": int(lag0_point)}),
        ("phase_action", {"phase": phase, "lag0_action": int(lag0_action)}),
    ]
    score = 0
    details = []
    for name, cond in levels:
        counts = _counts_for(examples, cond, base_action, cand_action)
        counts["level"] = name
        if counts["n"] >= int(min_support):
            if counts["cand_rate"] > counts["base_rate"]:
                score += 1
            elif counts["cand_rate"] < counts["base_rate"]:
                score -= 1
        details.append(counts)
    return score, details


def filter_supported_changes(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter scored changes by support strictness."""
    work = frame.copy()
    if mode == "strict":
        return work[(work["support_score"] >= 3) & (work["support_margin"] > 0.03)].copy()
    if mode == "balanced":
        return work[(work["support_score"] > 0) & (work["support_margin"] > 0.0)].copy()
    if mode == "extended":
        style_classes = {8, 9, 12, 14}
        return work[
            ((work["support_score"] > 0) & (work["support_margin"] >= 0.0))
            | ((work["cand_action"].astype(int).isin(style_classes)) & (work["support_score"] >= 0) & (work["support_margin"] > -0.02))
        ].copy()
    raise ValueError(f"unknown mode: {mode}")


def load_candidate_union(anchor: pd.DataFrame) -> pd.DataFrame:
    base = anchor[["rally_uid", "actionId"]].rename(columns={"actionId": "base_action"})
    pieces = []
    for path in SOURCE_FILES:
        if not path.exists():
            continue
        cand = pd.read_csv(path)[["rally_uid", "actionId"]].rename(columns={"actionId": "cand_action"})
        merged = base.merge(cand, on="rally_uid", how="inner")
        changed = merged[merged["base_action"].astype(int) != merged["cand_action"].astype(int)].copy()
        changed["source"] = path.name
        pieces.append(changed)
    if not pieces:
        return pd.DataFrame(columns=["rally_uid", "base_action", "cand_action", "source"])
    union = pd.concat(pieces, ignore_index=True)
    source_counts = union.groupby(["rally_uid", "base_action", "cand_action"]).size().reset_index(name="source_count")
    sources = union.groupby(["rally_uid", "base_action", "cand_action"])["source"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
    return source_counts.merge(sources, on=["rally_uid", "base_action", "cand_action"], how="left")


def score_test_candidates(candidates: pd.DataFrame, test: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    records = []
    test_sorted = test.sort_values(["rally_uid", "strikeNumber"])
    for row in candidates.itertuples(index=False):
        g = test_sorted[test_sorted["rally_uid"].eq(int(row.rally_uid))]
        if g.empty:
            continue
        last = g.iloc[-1]
        phase = phase_name(int(last["strikeNumber"]))
        score, details = backoff_support_score(
            examples,
            phase=phase,
            lag0_action=int(last["actionId"]),
            lag0_point=int(last["pointId"]),
            lag0_depth=point_depth(int(last["pointId"])),
            lag0_spin=int(last["spinId"]),
            lag0_strength=int(last["strengthId"]),
            base_action=int(row.base_action),
            cand_action=int(row.cand_action),
        )
        supported = [d for d in details if d["n"] >= 20]
        margin = float(np.mean([d["margin"] for d in supported])) if supported else 0.0
        best = supported[0] if supported else details[-1]
        records.append(
            {
                "rally_uid": int(row.rally_uid),
                "base_action": int(row.base_action),
                "cand_action": int(row.cand_action),
                "source_count": int(row.source_count),
                "sources": row.source,
                "prefix_len": int(len(g)),
                "phase": phase,
                "lag0_action": int(last["actionId"]),
                "lag0_point": int(last["pointId"]),
                "lag0_spin": int(last["spinId"]),
                "lag0_strength": int(last["strengthId"]),
                "support_score": int(score),
                "support_margin": margin,
                "best_level": best["level"],
                "best_n": int(best["n"]),
                "best_base_rate": float(best["base_rate"]),
                "best_cand_rate": float(best["cand_rate"]),
                "best_top": best["top"],
            }
        )
    return pd.DataFrame(records).sort_values(["support_score", "support_margin", "source_count"], ascending=[False, False, False])


def write_submission(name: str, anchor: pd.DataFrame, selected: pd.DataFrame) -> dict:
    out = anchor.copy()
    mapping = {int(r.rally_uid): int(r.cand_action) for r in selected.itertuples(index=False)}
    out["actionId"] = [mapping.get(int(rid), int(a)) for rid, a in zip(out["rally_uid"], out["actionId"])]
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected_path)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected_path)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    anchor = pd.read_csv(ANCHOR)
    test = pd.read_csv("test_new.csv")
    train = pd.read_csv("train.csv")
    examples = build_next_action_examples(train)
    union = load_candidate_union(anchor)
    scored = score_test_candidates(union, test, examples)
    scored.to_csv(OUTDIR / "v220_scored_candidate_changes.csv", index=False)

    generated = []
    records = []
    for mode in ["strict", "balanced", "extended"]:
        selected = filter_supported_changes(scored, mode)
        name = f"submission_v220_backoff_{mode}__pv188cap5__sr121.csv"
        info = write_submission(name, anchor, selected)
        info.update(
            {
                "mode": mode,
                "changed_rows": int(len(selected)),
                "changed_actions": json.dumps(selected["cand_action"].value_counts().sort_index().to_dict()) if len(selected) else "{}",
            }
        )
        generated.append(info)
        records.append(info)

    weak_targets = {0, 3, 5, 7, 8, 9, 12, 14}
    for mode in ["balanced", "extended"]:
        selected = filter_supported_changes(scored, mode)
        selected = selected[selected["cand_action"].astype(int).isin(weak_targets)].copy()
        name = f"submission_v220_backoff_{mode}_weakonly__pv188cap5__sr121.csv"
        info = write_submission(name, anchor, selected)
        info.update(
            {
                "mode": f"{mode}_weakonly",
                "changed_rows": int(len(selected)),
                "changed_actions": json.dumps(selected["cand_action"].value_counts().sort_index().to_dict()) if len(selected) else "{}",
            }
        )
        generated.append(info)
        records.append(info)

    summary = pd.DataFrame(records)
    summary.to_csv(OUTDIR / "v220_action_search.csv", index=False)
    report = {
        "verdict": "GENERATED_SUPPORT_FILTERED_TEST_CANDIDATES",
        "candidate_union_rows": int(len(union)),
        "scored_rows": int(len(scored)),
        "generated": generated,
        "notes": [
            "V220 filters V217/V218/V219 test action edits by train next-action backoff support.",
            "This is an audit/filter over existing model candidates, not manual label editing.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v220_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v220_report.md").write_text(
        "# V220 Action Backoff Support Filter\n\n"
        f"- Candidate union rows: `{len(union)}`\n"
        f"- Scored rows: `{len(scored)}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v220_action_backoff_support_filter.py", SRC_DEST)
    print(json.dumps({"generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
