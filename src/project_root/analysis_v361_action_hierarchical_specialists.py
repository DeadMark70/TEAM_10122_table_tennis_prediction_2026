"""V361 action hierarchical specialists.

Builds conservative action-only candidates around the V338 clean anchor:
V173 action, V338 point, and V300 server.  Source submissions are treated as
row-level evidence only; V361 never performs a full action replacement.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v361_action_hierarchical_specialists"
ANCHOR_PATH = ROOT / "v338_joint_moe_pack" / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
SERVE_LIKE_ACTIONS = {15, 16, 17, 18}
WEAK_ACTIONS = {3, 4, 5, 7, 8, 9, 12, 14}

SOURCE_DIRS = {
    "r166": ROOT / "r166_teacher_distillation",
    "v209": ROOT / "v209_action_selector_reranker",
    "v214": ROOT / "v214_shuttlenet_component_ablation",
    "v291": ROOT / "v291_weak_class_training_upgrade",
    "v332": ROOT / "v332_hierarchical_action_model",
    "v330": ROOT / "v330_action_weakclass_teacher_pool",
}

EXPLICIT_SOURCE_FILES = [
    SOURCE_DIRS["r166"] / "submission_r166__ar166_best_action__pr119_public_point__sr121_min_w0p2.csv",
    SOURCE_DIRS["v209"] / "submission_v209_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v209"] / "submission_v209_selector_churn0p01__pv188cap5__sr121.csv",
    SOURCE_DIRS["v214"] / "submission_v214_full_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v214"] / "submission_v214_no_beta_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_fast57_modelbank_c0p005__pv261cap1__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_terminal03_modelbank_c0p005__pv261cap1__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_bank_fast_terminal_c0p005__pv261cap1__sr121.csv",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def action_to_family(action: int) -> str:
    action_id = int(action)
    if action_id == 0:
        return "zero"
    if 1 <= action_id <= 7:
        return "attack"
    if 8 <= action_id <= 11:
        return "control"
    if 12 <= action_id <= 14:
        return "defensive"
    if 15 <= action_id <= 18:
        return "serve"
    return "unknown"


def point_to_depth(point: int) -> str:
    point_id = int(point)
    if point_id == 0:
        return "terminal"
    if 1 <= point_id <= 3:
        return "short"
    if 4 <= point_id <= 6:
        return "half"
    if 7 <= point_id <= 9:
        return "long"
    return "unknown"


def phase_from_prefix(prefix_len: int) -> str:
    value = int(prefix_len)
    if value <= 1:
        return "receive"
    if value == 2:
        return "third_ball"
    if value == 3:
        return "fourth_ball"
    return "rally"


def block_serve_like_actions(base: pd.Series, proposed: pd.Series) -> pd.Series:
    base_series = pd.Series(base).reset_index(drop=True).astype(int)
    proposed_series = pd.Series(proposed).reset_index(drop=True).astype(int)
    if len(base_series) != len(proposed_series):
        raise ValueError("base and proposed must have matching lengths")
    blocked = proposed_series.isin(SERVE_LIKE_ACTIONS) & ~base_series.isin(SERVE_LIKE_ACTIONS)
    out = proposed_series.copy()
    out.loc[blocked] = base_series.loc[blocked]
    return out.astype(int)


def package_action_candidate(anchor: pd.DataFrame, action_pred: pd.Series | np.ndarray) -> pd.DataFrame:
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"anchor columns {list(anchor.columns)} != {SUBMISSION_COLUMNS}")
    pred = pd.Series(action_pred).reset_index(drop=True).astype(int)
    if len(pred) != len(anchor):
        raise ValueError(f"action predictions {len(pred)} != anchor rows {len(anchor)}")
    out = anchor.copy()
    out["actionId"] = pred.to_numpy(dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["pointId"].equals(anchor["pointId"]):
        raise AssertionError("V361 export changed pointId")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("V361 export changed serverGetPoint")
    return out


def safe_output_path(filename: str) -> Path:
    path = OUTDIR / filename
    blocked = {"ttmatch", "oldserver", "old_server", "upload_candidates", "selected", "submissions"}
    lowered = [part.lower() for part in path.parts]
    if path.parent != OUTDIR:
        raise ValueError(f"V361 outputs must stay directly under {OUTDIR}: {path}")
    if any(any(token in part for token in blocked) for part in lowered):
        raise ValueError(f"refusing blocked V361 output path: {path}")
    return path


def load_anchor_submission() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing V338 anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{ANCHOR_PATH} columns {list(anchor.columns)} != {SUBMISSION_COLUMNS}")
    if len(anchor) != 1845:
        raise ValueError(f"{ANCHOR_PATH} rows {len(anchor)} != 1845")
    return anchor


def load_test_context(anchor: pd.DataFrame) -> pd.DataFrame:
    test_path = ROOT / "test_new.csv"
    if not test_path.exists():
        context = anchor[["rally_uid"]].copy()
        context["prefix_len"] = 0
        context["phase"] = "unknown"
        context["lag0_action_family"] = "unknown"
        context["lag0_point_depth"] = "unknown"
        return context
    test = pd.read_csv(test_path)
    test = test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", as_index=False).tail(1)
    context = anchor[["rally_uid"]].merge(test, on="rally_uid", how="left", validate="one_to_one")
    context["prefix_len"] = pd.to_numeric(context.get("strikeNumber"), errors="coerce").fillna(0).astype(int)
    context["phase"] = context["prefix_len"].map(phase_from_prefix)
    action_col = "actionId_y" if "actionId_y" in context else "actionId"
    context["lag0_action_family"] = (
        pd.to_numeric(context.get(action_col), errors="coerce").fillna(-1).astype(int).map(action_to_family)
        if action_col in context
        else "unknown"
    )
    point_col = "pointId_y" if "pointId_y" in context else "pointId"
    context["lag0_point_depth"] = pd.to_numeric(context.get(point_col), errors="coerce").fillna(-1).astype(int).map(point_to_depth)
    return context


def train_backoff_predictions(anchor: pd.DataFrame, context: pd.DataFrame) -> pd.Series:
    train_path = ROOT / "train.csv"
    if not train_path.exists():
        return anchor["actionId"].astype(int).copy()
    train = pd.read_csv(train_path)
    train = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    train["prev_action"] = train.groupby("rally_uid")["actionId"].shift(1)
    train["prev_point"] = train.groupby("rally_uid")["pointId"].shift(1)
    train["prev_spin"] = train.groupby("rally_uid")["spinId"].shift(1)
    train["prev_strength"] = train.groupby("rally_uid")["strengthId"].shift(1)
    train["prev_strike"] = train["strikeNumber"] - 1
    rows = train[train["prev_action"].notna()].copy()
    rows["phase"] = rows["prev_strike"].astype(int).map(phase_from_prefix)
    rows["lag0_action_family"] = rows["prev_action"].astype(int).map(action_to_family)
    rows["lag0_point_depth"] = rows["prev_point"].astype(int).map(point_to_depth)
    rows["spin_key"] = rows["prev_spin"].fillna(-1).astype(int)
    rows["strength_key"] = rows["prev_strength"].fillna(-1).astype(int)

    tables: list[dict[tuple[Any, ...], tuple[int, int]]] = []
    key_sets = [
        ["phase", "lag0_action_family", "lag0_point_depth", "spin_key", "strength_key"],
        ["phase", "lag0_action_family", "lag0_point_depth"],
        ["phase", "lag0_action_family"],
    ]
    for keys in key_sets:
        table: dict[tuple[Any, ...], tuple[int, int]] = {}
        for key, group in rows.groupby(keys, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            counts = group["actionId"].astype(int).value_counts()
            action = int(counts.index[0])
            support = int(counts.iloc[0])
            if len(group) >= 20 and support >= 8:
                table[key_tuple] = (action, int(len(group)))
        tables.append(table)

    test_context = context.copy()
    test_context["spin_key"] = pd.to_numeric(test_context.get("spinId"), errors="coerce").fillna(-1).astype(int)
    test_context["strength_key"] = pd.to_numeric(test_context.get("strengthId"), errors="coerce").fillna(-1).astype(int)
    out = anchor["actionId"].astype(int).copy()
    for idx, row in test_context.iterrows():
        keys_for_row = [
            (row.get("phase"), row.get("lag0_action_family"), row.get("lag0_point_depth"), row.get("spin_key"), row.get("strength_key")),
            (row.get("phase"), row.get("lag0_action_family"), row.get("lag0_point_depth")),
            (row.get("phase"), row.get("lag0_action_family")),
        ]
        for table, key in zip(tables, keys_for_row):
            if key in table:
                out.iloc[idx] = int(table[key][0])
                break
    return block_serve_like_actions(anchor["actionId"], out)


def source_weight(path: Path) -> float:
    name = path.as_posix().lower()
    if "v209_selector_churn0p005" in name:
        return 5.0
    if "v214_full_selector" in name:
        return 4.5
    if "v291_fast57" in name:
        return 4.0
    if "v291_terminal03" in name:
        return 3.8
    if "r166" in name:
        return 2.0
    return 1.0


def load_source_actions(anchor: pd.DataFrame) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[Path] = set()
    source_files = list(EXPLICIT_SOURCE_FILES)
    for directory in SOURCE_DIRS.values():
        if directory.exists():
            source_files.extend(sorted(directory.glob("submission*.csv")))
    for path in source_files:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        lower = path.name.lower()
        if "ttmatch" in lower or "oldserver" in lower or "old_server" in lower:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not {"rally_uid", "actionId"}.issubset(frame.columns):
            continue
        merged = anchor[["rally_uid", "actionId"]].merge(
            frame[["rally_uid", "actionId"]].rename(columns={"actionId": "source_action"}),
            on="rally_uid",
            how="left",
            validate="one_to_one",
        )
        if merged["source_action"].isna().any():
            continue
        proposed = block_serve_like_actions(anchor["actionId"], merged["source_action"])
        sources.append(
            {
                "name": path.stem,
                "path": relative_path(path),
                "weight": source_weight(path),
                "action": proposed.astype(int),
            }
        )
    return sources


def build_row_evidence(anchor: pd.DataFrame, context: pd.DataFrame, sources: list[dict[str, Any]]) -> pd.DataFrame:
    base = anchor["actionId"].astype(int).reset_index(drop=True)
    votes: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "sources": []})
    for source in sources:
        actions = pd.Series(source["action"]).reset_index(drop=True).astype(int)
        changed = actions.ne(base)
        for idx in np.where(changed.to_numpy())[0]:
            action = int(actions.iloc[idx])
            if action in SERVE_LIKE_ACTIONS and int(base.iloc[idx]) not in SERVE_LIKE_ACTIONS:
                continue
            key = (int(idx), action)
            votes[key]["score"] += float(source["weight"])
            votes[key]["sources"].append(str(source["name"]))

    backoff = train_backoff_predictions(anchor, context)
    backoff_changed = backoff.astype(int).ne(base)
    for idx in np.where(backoff_changed.to_numpy())[0]:
        action = int(backoff.iloc[idx])
        if action in SERVE_LIKE_ACTIONS and int(base.iloc[idx]) not in SERVE_LIKE_ACTIONS:
            continue
        key = (int(idx), action)
        votes[key]["score"] += 1.25
        votes[key]["sources"].append("train_backoff")

    records: list[dict[str, Any]] = []
    for (idx, action), data in votes.items():
        base_action = int(base.iloc[idx])
        row = context.iloc[idx] if idx < len(context) else pd.Series(dtype=object)
        family = action_to_family(action)
        base_family = action_to_family(base_action)
        source_counter = Counter(data["sources"])
        records.append(
            {
                "row_index": int(idx),
                "rally_uid": int(anchor["rally_uid"].iloc[idx]),
                "base_action": base_action,
                "proposed_action": int(action),
                "base_family": base_family,
                "proposed_family": family,
                "family_consistent": bool(base_family == family),
                "weak_class_related": bool(action in WEAK_ACTIONS or base_action in WEAK_ACTIONS),
                "phase": str(row.get("phase", "unknown")),
                "lag0_action_family": str(row.get("lag0_action_family", "unknown")),
                "lag0_point_depth": str(row.get("lag0_point_depth", "unknown")),
                "score": float(data["score"] + 0.10 * len(source_counter)),
                "source_count": int(len(source_counter)),
                "sources": ";".join(sorted(source_counter)),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "row_index",
                "rally_uid",
                "base_action",
                "proposed_action",
                "base_family",
                "proposed_family",
                "family_consistent",
                "weak_class_related",
                "phase",
                "score",
                "source_count",
                "sources",
            ]
        )
    evidence = pd.DataFrame(records)
    evidence = evidence.sort_values(
        ["score", "source_count", "family_consistent", "rally_uid"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return evidence


def select_nonconflicting(evidence: pd.DataFrame, mask: pd.Series, budget: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    used_rows: set[int] = set()
    candidates = evidence.loc[mask.to_numpy(dtype=bool)].copy()
    for _, row in candidates.iterrows():
        row_index = int(row["row_index"])
        if row_index in used_rows:
            continue
        selected.append(row)
        used_rows.add(row_index)
        if len(selected) >= int(budget):
            break
    if not selected:
        return evidence.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def apply_evidence(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    action = anchor["actionId"].astype(int).copy()
    for _, row in selected.iterrows():
        action.iloc[int(row["row_index"])] = int(row["proposed_action"])
    return block_serve_like_actions(anchor["actionId"], action)


def candidate_risk(churn: int, name: str) -> str:
    if "research" in name or int(churn) > 40:
        return "research"
    if int(churn) <= 10:
        return "safe"
    return "normal"


def action_distribution(values: pd.Series) -> str:
    counts = values.astype(int).value_counts().sort_index()
    return json.dumps({str(int(k)): int(v) for k, v in counts.items()}, sort_keys=True)


def export_candidate(anchor: pd.DataFrame, name: str, action: pd.Series, selected: pd.DataFrame) -> dict[str, Any]:
    out = package_action_candidate(anchor, action)
    churn = int(out["actionId"].astype(int).ne(anchor["actionId"].astype(int)).sum())
    filename = f"submission_{name}__v338point_v300server.csv"
    path = safe_output_path(filename)
    out.to_csv(path, index=False, float_format="%.8f")
    changed_actions = out.loc[out["actionId"].ne(anchor["actionId"]), "actionId"]
    return {
        "candidate": name,
        "path": relative_path(path),
        "changed_action_rows": churn,
        "risk": candidate_risk(churn, name),
        "point_preserved_from": "V338 point-only MoE no-p0-add budget24",
        "server_preserved_from": "V300",
        "serve_15_18_new_rows": int(
            (
                out["actionId"].astype(int).isin(SERVE_LIKE_ACTIONS)
                & ~anchor["actionId"].astype(int).isin(SERVE_LIKE_ACTIONS)
            ).sum()
        ),
        "changed_action_distribution": action_distribution(changed_actions) if churn else "{}",
        "evidence_rows": int(len(selected)),
        "evidence_source_count": int(selected["sources"].str.split(";").explode().nunique()) if len(selected) else 0,
        "mean_evidence_score": float(selected["score"].mean()) if len(selected) else 0.0,
    }


def build_candidates(anchor: pd.DataFrame, evidence: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if evidence.empty:
        return [
            ("v361_family_gate_low", evidence),
            ("v361_weakclass_specialist_b10", evidence),
            ("v361_weakclass_specialist_b25", evidence),
            ("v361_phase_specialist_receive_third", evidence),
            ("v361_research_action_hierarchy", evidence),
        ]

    normal_action = ~evidence["proposed_action"].astype(int).isin(SERVE_LIKE_ACTIONS)
    family_low = select_nonconflicting(evidence, normal_action & evidence["family_consistent"].astype(bool), 10)
    weak_b10 = select_nonconflicting(evidence, normal_action & evidence["weak_class_related"].astype(bool), 10)
    weak_b25 = select_nonconflicting(evidence, normal_action & evidence["weak_class_related"].astype(bool), 25)
    receive_third = select_nonconflicting(
        evidence,
        normal_action & evidence["phase"].isin(["receive", "third_ball"]),
        25,
    )
    research = select_nonconflicting(evidence, normal_action, 80)
    return [
        ("v361_family_gate_low", family_low),
        ("v361_weakclass_specialist_b10", weak_b10),
        ("v361_weakclass_specialist_b25", weak_b25),
        ("v361_phase_specialist_receive_third", receive_third),
        ("v361_research_action_hierarchy", research),
    ]


def write_reports(summary: pd.DataFrame, evidence: pd.DataFrame, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = safe_output_path("candidate_summary.csv")
    evidence_path = safe_output_path("row_evidence.csv")
    report_path = safe_output_path("search_report.json")
    summary.to_csv(summary_path, index=False)
    evidence.head(300).to_csv(evidence_path, index=False)
    report = {
        "version": "V361",
        "anchor_submission": relative_path(ANCHOR_PATH),
        "anchor_contract": {
            "action_anchor": "V173 carried by V338 package",
            "point_fixed_to": "V338 point-only MoE no-p0-add budget24",
            "server_fixed_to": "V300",
            "columns": SUBMISSION_COLUMNS,
            "expected_rows": 1845,
        },
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
            "block_new_serve_like_15_18": True,
            "research_if_action_churn_gt_40": True,
        },
        "source_count": len(source_rows),
        "sources": source_rows,
        "candidate_summary": relative_path(summary_path),
        "row_evidence": relative_path(evidence_path),
        "generated_submissions": summary.to_dict(orient="records"),
        "generated_submission_count": int(len(summary)),
    }
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor_submission()
    context = load_test_context(anchor)
    sources = load_source_actions(anchor)
    source_rows = [
        {
            "name": source["name"],
            "path": source["path"],
            "weight": float(source["weight"]),
            "changed_vs_anchor_after_serve_block": int(pd.Series(source["action"]).astype(int).ne(anchor["actionId"].astype(int)).sum()),
        }
        for source in sources
    ]
    evidence = build_row_evidence(anchor, context, sources)
    summary_rows: list[dict[str, Any]] = []
    for name, selected in build_candidates(anchor, evidence):
        action = apply_evidence(anchor, selected)
        summary_rows.append(export_candidate(anchor, name, action, selected))
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        risk_order = {"safe": 0, "normal": 1, "research": 2}
        summary["_risk_order"] = summary["risk"].map(risk_order).fillna(9).astype(int)
        summary = summary.sort_values(
            ["_risk_order", "changed_action_rows", "mean_evidence_score"],
            ascending=[True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        summary = summary.drop(columns=["_risk_order"])
    return write_reports(summary, evidence, source_rows)


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": relative_path(OUTDIR),
                "generated_submission_count": report["generated_submission_count"],
                "generated_submissions": [row["path"] for row in report["generated_submissions"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
