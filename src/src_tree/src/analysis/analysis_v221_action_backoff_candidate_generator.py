"""V221 action backoff candidate generator.

V220 filters candidate rows proposed by existing models.  V221 expands the idea:
it generates action candidates directly from train next-action backoff statistics
for every row, validates thresholds fold-safely on OOF rows, then exports
low-churn test candidates.  The first production branch focuses on weak actions
to avoid broad drive-heavy rewrites.

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
from sklearn.metrics import f1_score

import analysis_v217_macro_f1_utility_reranker as v217
from analysis_r184_receiver_affordance_refiner import load_sub, rebuild_v173_best_actions
from analysis_v195_distribution_matched_point_gru import prepare_data
from analysis_v220_action_backoff_support_filter import phase_name, point_depth
from analysis_v216_terminal_action_tuner import POINT_ANCHOR, SERVER_ANCHOR
from baseline_lgbm import ACTION_CLASSES


OUTDIR = Path("v221_action_backoff_candidate_generator")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v221_action_backoff_candidate_generator.py")

WEAK_ACTIONS = {0, 3, 5, 7, 8, 9, 12, 14}
DIRECT_WEAK_GATES = {
    5: {"exact_n": 100, "exact_count": 30, "exact_rate": 0.25, "top_gap": 0.045, "support_score": 5, "support_margin": 0.05},
    7: {"exact_n": 100, "exact_count": 30, "exact_rate": 0.25, "top_gap": 0.050, "support_score": 5, "support_margin": 0.05},
    12: {"exact_n": 100, "exact_count": 35, "exact_rate": 0.30, "top_gap": 0.100, "support_score": 3, "support_margin": 0.05},
}
DRIVE_SURGICAL_GATES = {
    1: {"exact_n": 100, "exact_count": 50, "exact_rate": 0.50, "top_gap": 0.300, "support_score": 4, "support_margin": 0.20},
}
LEVELS = [
    ("exact", ("phase", "lag0_action", "lag0_point", "lag0_spin", "lag0_strength")),
    ("phase_action_point", ("phase", "lag0_action", "lag0_point")),
    ("phase_action_depth", ("phase", "lag0_action", "lag0_depth")),
    ("action_point", ("lag0_action", "lag0_point")),
    ("phase_action", ("phase", "lag0_action")),
]
SCHEMES = [
    {"name": "v221_direct_weak_exact_safecap", "gates": DIRECT_WEAK_GATES, "max_churn": 0.006},
    {"name": "v221_direct_weak_exact_tightcap", "gates": DIRECT_WEAK_GATES, "max_churn": 0.003},
    {"name": "v221_drive_surgical_diagnostic", "gates": DRIVE_SURGICAL_GATES, "max_churn": 0.004},
]


def macro_f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))


def rows_to_examples(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": rows["audit_phase"].astype(str).to_numpy(),
            "lag0_action": rows["lag0_actionId"].astype(int).to_numpy(),
            "lag0_point": rows["lag0_pointId"].astype(int).to_numpy(),
            "lag0_depth": rows["lag0_pointId"].astype(int).map(point_depth).to_numpy(),
            "lag0_spin": rows["lag0_spinId"].astype(int).to_numpy(),
            "lag0_strength": rows["lag0_strengthId"].astype(int).to_numpy(),
            "next_action": rows["next_actionId"].astype(int).to_numpy(),
        }
    )


def train_to_examples(train: pd.DataFrame, match_to_fold: dict[int, int] | None = None) -> pd.DataFrame:
    records = []
    for _, g in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) < 2:
            continue
        match = int(g.iloc[0]["match"])
        fold = int(match_to_fold.get(match, -1)) if match_to_fold is not None else -1
        for i in range(len(g) - 1):
            lag = g.iloc[i]
            nxt = g.iloc[i + 1]
            records.append(
                {
                    "match": match,
                    "fold": fold,
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


def test_to_context(test: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rid, g in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        last = g.sort_values("strikeNumber").iloc[-1]
        records.append(
            {
                "rally_uid": int(rid),
                "phase": phase_name(int(last["strikeNumber"])),
                "lag0_action": int(last["actionId"]),
                "lag0_point": int(last["pointId"]),
                "lag0_depth": point_depth(int(last["pointId"])),
                "lag0_spin": int(last["spinId"]),
                "lag0_strength": int(last["strengthId"]),
            }
        )
    return pd.DataFrame(records)


def build_support_tables(examples: pd.DataFrame) -> dict[str, dict[tuple, dict]]:
    tables: dict[str, dict[tuple, dict]] = {}
    for level_name, keys in LEVELS:
        table = {}
        grouped = examples.groupby(list(keys) + ["next_action"]).size().reset_index(name="count")
        for key_values, g in grouped.groupby(list(keys), sort=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            counts = {int(r.next_action): int(r.count) for r in g.itertuples(index=False)}
            total = int(sum(counts.values()))
            table[tuple(key_values)] = {"total": total, "counts": counts}
        tables[level_name] = table
    return tables


def choose_supported_candidate(
    tables: dict[str, dict[tuple, dict]],
    context: dict,
    base_action: int,
    allowed_actions: set[int],
    min_support: int,
    min_score: int,
    min_margin: float,
) -> dict | None:
    best = None
    allowed = {int(a) for a in allowed_actions if int(a) != int(base_action)}
    for cand in sorted(allowed):
        support_score = 0
        margins = []
        total_weight = 0
        detail_levels = []
        for level_name, keys in LEVELS:
            key = tuple(context[k] for k in keys)
            hit = tables.get(level_name, {}).get(key)
            if not hit or int(hit["total"]) < int(min_support):
                continue
            counts = hit["counts"]
            total = int(hit["total"])
            base_rate = counts.get(int(base_action), 0) / total
            cand_rate = counts.get(int(cand), 0) / total
            margin = cand_rate - base_rate
            margins.append(margin)
            total_weight += total
            if margin > 0:
                support_score += 1
            elif margin < 0:
                support_score -= 1
            detail_levels.append(f"{level_name}:{total}:{base_rate:.4f}>{cand_rate:.4f}")
        if not margins:
            continue
        avg_margin = float(np.mean(margins))
        if support_score < int(min_score) or avg_margin < float(min_margin):
            continue
        rec = {
            "candidate_action": int(cand),
            "support_score": int(support_score),
            "support_margin": avg_margin,
            "support_weight": int(total_weight),
            "support_details": "|".join(detail_levels),
        }
        if best is None or (rec["support_score"], rec["support_margin"], rec["support_weight"]) > (
            best["support_score"],
            best["support_margin"],
            best["support_weight"],
        ):
            best = rec
    return best


def exact_candidate_stats(tables: dict[str, dict[tuple, dict]], context: dict, cand: int, base_action: int) -> dict | None:
    key = tuple(context[k] for k in LEVELS[0][1])
    hit = tables.get("exact", {}).get(key)
    if not hit:
        return None
    counts = hit["counts"]
    total = int(hit["total"])
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_action, top_count = ordered[0]
    second_count = ordered[1][1] if len(ordered) > 1 else 0
    cand_count = int(counts.get(int(cand), 0))
    base_count = int(counts.get(int(base_action), 0))
    return {
        "exact_n": total,
        "exact_count": cand_count,
        "exact_rate": cand_count / total if total else 0.0,
        "exact_base_rate": base_count / total if total else 0.0,
        "top_gap": (top_count - second_count) / total if total else 0.0,
        "exact_top_action": int(top_action),
        "exact_is_top1": int(int(top_action) == int(cand)),
    }


def choose_exact_gated_candidate(
    tables: dict[str, dict[tuple, dict]],
    context: dict,
    base_action: int,
    gates: dict[int, dict],
) -> dict | None:
    """Choose candidate only when the exact bucket top-1 and backoff gates pass."""
    best = None
    for cand, gate in gates.items():
        if int(cand) == int(base_action):
            continue
        exact = exact_candidate_stats(tables, context, int(cand), int(base_action))
        if exact is None or exact["exact_is_top1"] != 1:
            continue
        support = choose_supported_candidate(
            tables,
            context=context,
            base_action=int(base_action),
            allowed_actions={int(cand)},
            min_support=int(gate["exact_n"]),
            min_score=int(gate["support_score"]),
            min_margin=float(gate["support_margin"]),
        )
        if support is None:
            continue
        if exact["exact_n"] < int(gate["exact_n"]):
            continue
        if exact["exact_count"] < int(gate["exact_count"]):
            continue
        if exact["exact_rate"] < float(gate["exact_rate"]):
            continue
        if exact["top_gap"] < float(gate["top_gap"]):
            continue
        rec = {**support, **exact}
        if best is None or (
            rec["support_score"],
            rec["support_margin"],
            rec["exact_rate"],
        ) > (
            best["support_score"],
            best["support_margin"],
            best["exact_rate"],
        ):
            best = rec
    return best


def generate_candidates(
    context: pd.DataFrame,
    base_actions: np.ndarray,
    tables: dict[str, dict[tuple, dict]],
    gates: dict[int, dict],
) -> pd.DataFrame:
    records = []
    for idx, row in enumerate(context.itertuples(index=False)):
        ctx = {
            "phase": row.phase,
            "lag0_action": int(row.lag0_action),
            "lag0_point": int(row.lag0_point),
            "lag0_depth": int(row.lag0_depth),
            "lag0_spin": int(row.lag0_spin),
            "lag0_strength": int(row.lag0_strength),
        }
        cand = choose_exact_gated_candidate(
            tables,
            context=ctx,
            base_action=int(base_actions[idx]),
            gates=gates,
        )
        if cand is None:
            continue
        records.append({"row_id": idx, **ctx, "base_action": int(base_actions[idx]), **cand})
    return pd.DataFrame(records)


def select_capped(candidates: pd.DataFrame, base_actions: np.ndarray, max_churn: float) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(base_actions, dtype=int).copy()
    selected = np.zeros(len(out), dtype=bool)
    if candidates.empty:
        return out, selected
    max_rows = int(np.floor(len(out) * float(max_churn)))
    if max_rows <= 0:
        return out, selected
    cand = candidates.copy()
    cand = cand.sort_values(["support_score", "support_margin", "support_weight"], ascending=[False, False, False]).head(max_rows)
    for row in cand.itertuples(index=False):
        selected[int(row.row_id)] = True
        out[int(row.row_id)] = int(row.candidate_action)
    return out, selected


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
    selected_path = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected_path)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected_path)}


def main() -> None:
    __main__.V3Tuning = v217.V3Tuning
    __main__.GrUTuning = v217.GrUTuning
    __main__.TransformerTuning = v217.TransformerTuning

    OUTDIR.mkdir(exist_ok=True)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    rows = data["rows"].copy()
    y = rows["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    base_score = macro_f1_score(y, v173_oof)
    match_to_fold = rows.groupby("match")["fold"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
    all_examples = train_to_examples(pd.read_csv("train.csv"), match_to_fold=match_to_fold)

    point = pd.read_csv(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    v173_test = point["actionId"].astype(int).to_numpy()
    test_context = test_to_context(pd.read_csv("test_new.csv")).set_index("rally_uid").loc[rally_uids].reset_index()

    search_records = [
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
    candidate_tables = []
    for scheme in SCHEMES:
        oof_pred = v173_oof.copy()
        oof_changed = np.zeros(len(oof_pred), dtype=bool)
        oof_candidates_parts = []
        for fold in sorted(rows["fold"].astype(int).unique()):
            valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
            train_examples = all_examples[~all_examples["fold"].astype(int).eq(int(fold))].copy()
            tables = build_support_tables(train_examples)
            valid_context = rows_to_examples(rows[valid]).drop(columns=["next_action"]).reset_index(drop=True)
            cands = generate_candidates(
                valid_context,
                v173_oof[valid],
                tables,
                gates=scheme["gates"],
            )
            pred_fold, selected_fold = select_capped(cands, v173_oof[valid], scheme["max_churn"])
            valid_idx = np.where(valid)[0]
            oof_pred[valid_idx] = pred_fold
            oof_changed[valid_idx] = selected_fold
            if not cands.empty:
                c = cands.copy()
                c["global_row_id"] = valid_idx[c["row_id"].astype(int).to_numpy()]
                c["scheme"] = scheme["name"]
                oof_candidates_parts.append(c)
        score = macro_f1_score(y, oof_pred)

        full_tables = build_support_tables(all_examples)
        test_cands = generate_candidates(
            test_context,
            v173_test,
            full_tables,
            gates=scheme["gates"],
        )
        test_pred, test_changed = select_capped(test_cands, v173_test, scheme["max_churn"])
        if not test_cands.empty:
            t = test_cands.copy()
            t["rally_uid"] = rally_uids[t["row_id"].astype(int).to_numpy()]
            t["scheme"] = scheme["name"]
            candidate_tables.append(t)
        rec = {
            "candidate": scheme["name"],
            "action_macro_f1": score,
            "delta_vs_v173_anchor": score - base_score,
            "action_churn_vs_v173_anchor": float(np.mean(oof_pred != v173_oof)),
            "changed_rows": int(oof_changed.sum()),
            "test_churn_vs_v173": float(np.mean(test_pred != v173_test)),
            "test_changed_rows": int(test_changed.sum()),
            "changed_actions": json.dumps(pd.Series(test_pred[test_changed]).value_counts().sort_index().to_dict()) if test_changed.any() else "{}",
            "available_test_candidates": int(len(test_cands)),
        }
        search_records.append(rec)
        info = write_submission(f"submission_{scheme['name']}__pv188cap5__sr121.csv", test_pred, point, server)
        info.update(rec)
        generated.append(info)

    search = pd.DataFrame(search_records).sort_values(["delta_vs_v173_anchor", "action_churn_vs_v173_anchor"], ascending=[False, True])
    search.to_csv(OUTDIR / "v221_action_search.csv", index=False)
    if candidate_tables:
        pd.concat(candidate_tables, ignore_index=True).to_csv(OUTDIR / "v221_test_candidates.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "generated": generated,
        "best": search.head(10).to_dict(orient="records"),
        "notes": [
            "V221 generates action candidates directly from fold-safe train backoff support.",
            "Weak branch is preferred; all-strict is diagnostic because it can become drive-heavy.",
            "Point is fixed at V188 cap5 and server is fixed at R121.",
            "No external rows and no TTMATCH are read.",
        ],
    }
    (OUTDIR / "v221_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v221_report.md").write_text(
        "# V221 Action Backoff Candidate Generator\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Best delta vs V173: `{best_delta:.6f}`\n"
        f"- Generated submissions: `{len(generated)}`\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v221_action_backoff_candidate_generator.py", SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
