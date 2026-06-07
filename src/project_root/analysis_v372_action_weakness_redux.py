"""V372 action weakness redux.

Rebuild weak-action evidence as a row-level candidate bank over the clean
V338 package: V173 action, V338 point, and V300 server.  This module treats
historical action submissions as evidence only; it never performs a full
replacement from any source branch.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v372_action_weakness_redux"
ANCHOR_PATH = ROOT / "v338_joint_moe_pack" / "submission_v338_point_only_point_moe_no_p0_add_b24__v173action_v300server.csv"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
WEAK_ACTIONS = {3, 4, 5, 7, 8, 9, 12, 14}
SERVE_LIKE_ACTIONS = {15, 16, 17, 18}

SOURCE_DIRS = {
    "v209": ROOT / "v209_action_selector_reranker",
    "v214": ROOT / "v214_shuttlenet_component_ablation",
    "v291": ROOT / "v291_weak_class_training_upgrade",
    "v330": ROOT / "v330_action_weakclass_teacher_pool",
    "v361": ROOT / "v361_action_hierarchical_specialists",
}

EXPLICIT_SOURCE_FILES = [
    SOURCE_DIRS["v209"] / "submission_v209_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v209"] / "submission_v210_compat_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v209"] / "submission_v211_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v214"] / "submission_v214_full_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v214"] / "submission_v214_no_beta_selector_churn0p005__pv188cap5__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_fast57_modelbank_c0p005__pv261cap1__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_fast57_modelbank_c0p010__pv261cap1__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_bank_fast_terminal_c0p005__pv261cap1__sr121.csv",
    SOURCE_DIRS["v291"] / "submission_v291_terminal03_modelbank_c0p005__pv261cap1__sr121.csv",
    SOURCE_DIRS["v361"] / "submission_v361_weakclass_specialist_b10__v338point_v300server.csv",
    SOURCE_DIRS["v361"] / "submission_v361_family_gate_low__v338point_v300server.csv",
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


def weak_class_priority(action: int) -> int:
    action_id = int(action)
    if action_id in {5, 7}:
        return 4
    if action_id in {8, 9, 12, 14}:
        return 3
    if action_id in {3, 4}:
        return 2
    return 0


def action_family(action: int) -> str:
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


def point_depth(point: int) -> str:
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


def block_serve_like_changes(base: pd.Series, pred: pd.Series) -> pd.Series:
    base_series = pd.Series(base).reset_index(drop=True).astype(int)
    pred_series = pd.Series(pred).reset_index(drop=True).astype(int)
    if len(base_series) != len(pred_series):
        raise ValueError("base and pred must have matching lengths")
    blocked = pred_series.isin(SERVE_LIKE_ACTIONS) & ~base_series.isin(SERVE_LIKE_ACTIONS)
    out = pred_series.copy()
    out.loc[blocked] = base_series.loc[blocked]
    return out.astype(int)


def package_action_submission(anchor: pd.DataFrame, action_pred: pd.Series | np.ndarray | list[int]) -> pd.DataFrame:
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"anchor columns {list(anchor.columns)} != {SUBMISSION_COLUMNS}")
    pred = pd.Series(action_pred).reset_index(drop=True).astype(int)
    if len(pred) != len(anchor):
        raise ValueError(f"action predictions {len(pred)} != anchor rows {len(anchor)}")
    pred = block_serve_like_changes(anchor["actionId"], pred)
    out = anchor.copy()
    out["actionId"] = pred.to_numpy(dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if not out["pointId"].equals(anchor["pointId"]):
        raise AssertionError("V372 export changed pointId")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("V372 export changed serverGetPoint")
    return out


def safe_output_path(filename: str) -> Path:
    path = OUTDIR / filename
    blocked = {"ttmatch", "oldserver", "old_server", "upload_candidates", "selected", "submissions"}
    lowered = [part.lower() for part in path.parts]
    if path.parent != OUTDIR:
        raise ValueError(f"V372 outputs must stay directly under {OUTDIR}: {path}")
    if any(any(token in part for token in blocked) for part in lowered):
        raise ValueError(f"refusing blocked V372 output path: {path}")
    return path


def load_anchor() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing V338 anchor submission: {ANCHOR_PATH}")
    anchor = pd.read_csv(ANCHOR_PATH)
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{ANCHOR_PATH} columns {list(anchor.columns)} != {SUBMISSION_COLUMNS}")
    return anchor


def source_weight(name: str, path: Path | None = None) -> float:
    text = f"{name} {path.as_posix() if path is not None else ''}".lower()
    if "v361_weakclass_specialist_b10" in text:
        return 6.0
    if "v361_family_gate_low" in text:
        return 5.5
    if "v209_selector_churn0p005" in text:
        return 4.5
    if "v210_compat_selector_churn0p005" in text or "v211_selector_churn0p005" in text:
        return 3.5
    if "v214_full_selector" in text:
        return 2.5
    if "v214_no_beta_selector" in text:
        return 1.5
    if "v291_fast57_modelbank_c0p005" in text:
        return 4.0
    if "v291_fast57_modelbank_c0p010" in text:
        return 2.75
    if "v291_bank_fast_terminal" in text:
        return 1.75
    if "v291_terminal03" in text:
        return 1.25
    return 1.0


def _valid_source_name(path: Path) -> bool:
    lowered = path.as_posix().lower()
    if "ttmatch" in lowered or "oldserver" in lowered or "old_server" in lowered:
        return False
    if "v166" in lowered or "r166" in lowered:
        return False
    return True


def load_candidate_sources(anchor: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    anchor_frame = load_anchor() if anchor is None else anchor
    source_files: list[Path] = []
    source_files.extend(EXPLICIT_SOURCE_FILES)
    for key in ["v209", "v214", "v291", "v361"]:
        directory = SOURCE_DIRS[key]
        if directory.exists():
            source_files.extend(sorted(directory.glob("submission*.csv")))

    seen: set[Path] = set()
    sources: dict[str, pd.DataFrame] = {}
    for path in source_files:
        path = path.resolve()
        if path in seen or not path.exists() or not _valid_source_name(path):
            continue
        seen.add(path)
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not {"rally_uid", "actionId"}.issubset(frame.columns):
            continue
        merged = anchor_frame[["rally_uid", "actionId"]].merge(
            frame[["rally_uid", "actionId"]].rename(columns={"actionId": "candidate_action"}),
            on="rally_uid",
            how="left",
            validate="one_to_one",
        )
        if merged["candidate_action"].isna().any():
            continue
        out = anchor_frame[["rally_uid"]].copy()
        out["actionId"] = block_serve_like_changes(anchor_frame["actionId"], merged["candidate_action"]).to_numpy(dtype=int)
        sources[path.stem] = out
    return sources


def load_test_context(anchor: pd.DataFrame) -> pd.DataFrame:
    test_path = ROOT / "test_new.csv"
    context = anchor[["rally_uid", "actionId", "pointId"]].copy()
    if not test_path.exists():
        context["prefix_len"] = 0
        context["phase"] = "unknown"
        context["lag_action_family"] = "unknown"
        context["lag_point_depth"] = "unknown"
        return context
    test = pd.read_csv(test_path)
    last = test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", as_index=False).tail(1)
    context = context.merge(last, on="rally_uid", how="left", suffixes=("", "_observed"), validate="one_to_one")
    context["prefix_len"] = pd.to_numeric(context.get("strikeNumber"), errors="coerce").fillna(0).astype(int)
    context["phase"] = context["prefix_len"].map(phase_from_prefix)
    observed_action = "actionId_observed" if "actionId_observed" in context else "actionId"
    observed_point = "pointId_observed" if "pointId_observed" in context else "pointId"
    context["lag_action_family"] = pd.to_numeric(context.get(observed_action), errors="coerce").fillna(-1).astype(int).map(action_family)
    context["lag_point_depth"] = pd.to_numeric(context.get(observed_point), errors="coerce").fillna(-1).astype(int).map(point_depth)
    return context


def train_backoff_predictions(anchor: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    train_path = ROOT / "train.csv"
    if not train_path.exists():
        return anchor["actionId"].astype(int).copy(), pd.Series(np.zeros(len(anchor), dtype=int))
    train = pd.read_csv(train_path)
    train = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    train["prev_action"] = train.groupby("rally_uid")["actionId"].shift(1)
    train["prev_point"] = train.groupby("rally_uid")["pointId"].shift(1)
    rows = train[train["prev_action"].notna()].copy()
    rows["phase"] = (rows["strikeNumber"] - 1).astype(int).map(phase_from_prefix)
    rows["lag_action_family"] = rows["prev_action"].astype(int).map(action_family)
    rows["lag_point_depth"] = rows["prev_point"].astype(int).map(point_depth)

    tables: list[dict[tuple[Any, ...], tuple[int, int]]] = []
    key_sets = [
        ["phase", "lag_action_family", "lag_point_depth"],
        ["phase", "lag_action_family"],
        ["lag_action_family", "lag_point_depth"],
    ]
    for keys in key_sets:
        table: dict[tuple[Any, ...], tuple[int, int]] = {}
        for key, group in rows.groupby(keys, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            counts = group["actionId"].astype(int).value_counts()
            if len(group) >= 20 and int(counts.iloc[0]) >= 6:
                table[key_tuple] = (int(counts.index[0]), int(len(group)))
        tables.append(table)

    out = anchor["actionId"].astype(int).copy()
    support = pd.Series(np.zeros(len(anchor), dtype=int))
    for idx, row in context.iterrows():
        keys_for_row = [
            (row.get("phase"), row.get("lag_action_family"), row.get("lag_point_depth")),
            (row.get("phase"), row.get("lag_action_family")),
            (row.get("lag_action_family"), row.get("lag_point_depth")),
        ]
        for table, key in zip(tables, keys_for_row):
            if key in table:
                out.iloc[idx] = int(table[key][0])
                support.iloc[idx] = int(table[key][1])
                break
    return block_serve_like_changes(anchor["actionId"], out), support.astype(int)


def load_v371_support(anchor: pd.DataFrame) -> pd.Series:
    path = ROOT / "v371_joint_causal_consistency_lab" / "consistency_evidence.csv"
    support = pd.Series(np.zeros(len(anchor), dtype=int))
    if not path.exists():
        return support
    try:
        evidence = pd.read_csv(path)
    except Exception:
        return support
    if "rally_uid" not in evidence:
        return support
    uid_to_index = {int(uid): idx for idx, uid in enumerate(anchor["rally_uid"].astype(int))}
    for _, row in evidence.iterrows():
        uid = int(row["rally_uid"])
        if uid in uid_to_index:
            support.iloc[uid_to_index[uid]] += 1
    return support.astype(int)


def collect_action_candidates(anchor: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = anchor["actionId"].astype(int).reset_index(drop=True)
    votes: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"sources": [], "weight": 0.0})
    for name, source in sources.items():
        if not {"rally_uid", "actionId"}.issubset(source.columns):
            continue
        merged = anchor[["rally_uid", "actionId"]].merge(
            source[["rally_uid", "actionId"]].rename(columns={"actionId": "candidate_action"}),
            on="rally_uid",
            how="left",
            validate="one_to_one",
        )
        if merged["candidate_action"].isna().any():
            continue
        proposed = block_serve_like_changes(base, merged["candidate_action"])
        changed = proposed.ne(base)
        for idx in np.where(changed.to_numpy())[0]:
            action = int(proposed.iloc[idx])
            if action in SERVE_LIKE_ACTIONS and int(base.iloc[idx]) not in SERVE_LIKE_ACTIONS:
                continue
            key = (int(idx), action)
            votes[key]["sources"].append(str(name))
            votes[key]["weight"] += source_weight(str(name))

    records: list[dict[str, Any]] = []
    for (idx, action), data in votes.items():
        source_counter = Counter(data["sources"])
        support_count = int(len(source_counter))
        base_action = int(base.iloc[idx])
        weak_priority = max(weak_class_priority(action), weak_class_priority(base_action))
        score = float(data["weight"] + 0.35 * support_count + 0.75 * weak_priority)
        records.append(
            {
                "row_index": int(idx),
                "rally_uid": int(anchor["rally_uid"].iloc[idx]),
                "base_action": base_action,
                "candidate_action": int(action),
                "base_family": action_family(base_action),
                "candidate_family": action_family(action),
                "weak_priority": int(weak_priority),
                "support_count": support_count,
                "source_weight": float(data["weight"]),
                "score": score,
                "sources": ";".join(sorted(source_counter)),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "row_index",
                "rally_uid",
                "base_action",
                "candidate_action",
                "base_family",
                "candidate_family",
                "weak_priority",
                "support_count",
                "source_weight",
                "score",
                "sources",
            ]
        )
    return pd.DataFrame(records).sort_values(
        ["score", "support_count", "weak_priority", "rally_uid"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def add_backoff_and_consistency(bank: pd.DataFrame, anchor: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if bank.empty:
        out = bank.copy()
        out["train_backoff_agrees"] = pd.Series(dtype=bool)
        out["train_support_count"] = pd.Series(dtype=int)
        out["v371_support_count"] = pd.Series(dtype=int)
        return out
    backoff_pred, backoff_support = train_backoff_predictions(anchor, context)
    v371_support = load_v371_support(anchor)
    out = bank.copy()
    out["train_backoff_agrees"] = [
        int(backoff_pred.iloc[int(row.row_index)]) == int(row.candidate_action) for row in out.itertuples(index=False)
    ]
    out["train_support_count"] = [int(backoff_support.iloc[int(row.row_index)]) for row in out.itertuples(index=False)]
    out["v371_support_count"] = [int(v371_support.iloc[int(row.row_index)]) for row in out.itertuples(index=False)]
    out["score"] = (
        out["score"].astype(float)
        + out["train_backoff_agrees"].astype(int) * 1.25
        + np.log1p(out["train_support_count"].astype(float)).clip(0, 4) * 0.20
        + out["v371_support_count"].astype(float).clip(0, 3) * 0.75
    )
    out = out.sort_values(
        ["score", "support_count", "weak_priority", "rally_uid"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def select_nonconflicting(bank: pd.DataFrame, budget: int, *, research: bool = False) -> pd.DataFrame:
    if bank.empty:
        return bank.iloc[0:0].copy()
    selected: list[pd.Series] = []
    used_rows: set[int] = set()
    candidates = bank.copy()
    if not research:
        candidates = candidates[
            (candidates["candidate_action"].astype(int).map(weak_class_priority) > 0)
            | (candidates["base_action"].astype(int).map(weak_class_priority) > 0)
        ].copy()
        candidates = candidates[candidates["support_count"].astype(int) >= 2]
    candidates = candidates[~candidates["candidate_action"].astype(int).isin(SERVE_LIKE_ACTIONS)]
    for _, row in candidates.iterrows():
        row_index = int(row["row_index"])
        if row_index in used_rows:
            continue
        selected.append(row)
        used_rows.add(row_index)
        if len(selected) >= int(budget):
            break
    if not selected:
        return bank.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def apply_selected(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    action = anchor["actionId"].astype(int).copy()
    for _, row in selected.iterrows():
        action.iloc[int(row["row_index"])] = int(row["candidate_action"])
    return block_serve_like_changes(anchor["actionId"], action)


def action_distribution(values: pd.Series) -> str:
    counts = pd.Series(values).astype(int).value_counts().sort_index()
    return json.dumps({str(int(k)): int(v) for k, v in counts.items()}, sort_keys=True)


def local_evidence_decision(selected: pd.DataFrame, candidate_name: str) -> str:
    if selected.empty:
        return "DO_NOT_UPLOAD"
    mean_score = float(selected["score"].mean())
    mean_support = float(selected["support_count"].mean())
    backoff_rate = float(selected["train_backoff_agrees"].mean()) if "train_backoff_agrees" in selected else 0.0
    if "research" in candidate_name:
        return "DO_NOT_UPLOAD"
    if mean_score >= 18.0 and mean_support >= 4.0 and backoff_rate >= 0.20:
        return "HAS_UPLOAD_CANDIDATE"
    return "DO_NOT_UPLOAD"


def export_candidate(anchor: pd.DataFrame, name: str, selected: pd.DataFrame) -> dict[str, Any]:
    action = apply_selected(anchor, selected)
    out = package_action_submission(anchor, action)
    filename = f"submission_{name}__v338point_v300server.csv"
    path = safe_output_path(filename)
    out.to_csv(path, index=False, float_format="%.8f")
    churn = int(out["actionId"].astype(int).ne(anchor["actionId"].astype(int)).sum())
    changed_actions = out.loc[out["actionId"].ne(anchor["actionId"]), "actionId"]
    decision = local_evidence_decision(selected, name)
    return {
        "candidate": name,
        "path": relative_path(path),
        "changed_action_rows": churn,
        "risk": "research" if "research" in name else "safe",
        "decision": decision,
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
        "mean_support_count": float(selected["support_count"].mean()) if len(selected) else 0.0,
        "train_backoff_agree_rows": int(selected["train_backoff_agrees"].sum()) if "train_backoff_agrees" in selected else 0,
    }


def read_source_reports() -> list[dict[str, Any]]:
    report_paths = [
        SOURCE_DIRS["v209"] / "v209_action_search.csv",
        SOURCE_DIRS["v214"] / "v214_action_search.csv",
        SOURCE_DIRS["v291"] / "v291_candidate_search.csv",
        SOURCE_DIRS["v330"] / "v330_action_search.csv",
        SOURCE_DIRS["v361"] / "search_report.json",
    ]
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        if not path.exists() or not _valid_source_name(path):
            continue
        item: dict[str, Any] = {"path": relative_path(path)}
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                item["kind"] = "json"
                item["decision"] = data.get("decision", data.get("upload_recommendation", "UNKNOWN"))
                item["generated_submission_count"] = data.get("generated_submission_count")
            else:
                frame = pd.read_csv(path)
                item["kind"] = "csv"
                item["rows"] = int(len(frame))
                if "decision" in frame:
                    item["upload_like_rows"] = int(frame["decision"].astype(str).str.contains("UPLOAD", case=False, na=False).sum())
                if "upload_recommendation" in frame:
                    item["upload_like_rows"] = int(
                        frame["upload_recommendation"].astype(str).str.contains("UPLOAD|REVIEW", case=False, regex=True, na=False).sum()
                    )
                if "action_oof_delta" in frame:
                    item["best_action_oof_delta"] = float(pd.to_numeric(frame["action_oof_delta"], errors="coerce").max())
                elif "delta_vs_v173" in frame:
                    item["best_action_oof_delta"] = float(pd.to_numeric(frame["delta_vs_v173"], errors="coerce").max())
                elif "delta_vs_v173_anchor" in frame:
                    item["best_action_oof_delta"] = float(pd.to_numeric(frame["delta_vs_v173_anchor"], errors="coerce").max())
        except Exception as exc:
            item["read_error"] = str(exc)
        rows.append(item)
    return rows


def write_reports(summary: pd.DataFrame, bank: pd.DataFrame, source_rows: list[dict[str, Any]], loaded_sources: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary_path = safe_output_path("candidate_summary.csv")
    bank_path = safe_output_path("action_candidate_bank.csv")
    report_path = safe_output_path("search_report.json")
    summary.to_csv(summary_path, index=False)
    bank.to_csv(bank_path, index=False)
    has_upload = bool((summary.get("decision", pd.Series(dtype=str)) == "HAS_UPLOAD_CANDIDATE").any())
    top_candidate = summary.iloc[0].to_dict() if len(summary) else {}
    report = {
        "version": "V372",
        "decision": "HAS_UPLOAD_CANDIDATE" if has_upload else "DO_NOT_UPLOAD",
        "top_candidate": top_candidate,
        "anchor_submission": relative_path(ANCHOR_PATH),
        "anchor_contract": {
            "action_anchor": "V173 carried by V338 package",
            "point_fixed_to": "V338 point-only MoE no-p0-add budget24",
            "server_fixed_to": "V300",
            "columns": SUBMISSION_COLUMNS,
        },
        "policy": {
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_directory_writes": True,
            "manual_row_edits": False,
            "direct_v166_full_replacement": False,
            "block_new_serve_like_15_18": True,
            "weak_action_targets": sorted(WEAK_ACTIONS),
            "material_strength_gate": "safe candidates need mean score >=18, mean support >=4, and train backoff rate >=20%",
        },
        "loaded_source_count": int(len(loaded_sources)),
        "loaded_sources": sorted(loaded_sources),
        "source_reports": source_rows,
        "action_candidate_bank": relative_path(bank_path),
        "candidate_summary": relative_path(summary_path),
        "generated_submissions": summary.to_dict(orient="records"),
        "generated_submission_count": int(len(summary)),
        "sub_directions": [
            {
                "name": "weak_action_consensus_without_v166",
                "status": "continued",
                "result": "row-level evidence built from V209/V214/V291/V361 plus train backoff; V330 report present but no submission files found",
            }
        ],
    }
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return report


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor()
    context = load_test_context(anchor)
    sources = load_candidate_sources(anchor)
    bank = collect_action_candidates(anchor, sources)
    bank = add_backoff_and_consistency(bank, anchor, context)
    candidates = [
        ("v372_action_weak_safe_b05", select_nonconflicting(bank, 5)),
        ("v372_action_weak_safe_b10", select_nonconflicting(bank, 10)),
        ("v372_action_weak_research", select_nonconflicting(bank, 60, research=True)),
    ]
    summary_rows = [export_candidate(anchor, name, selected) for name, selected in candidates]
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        decision_order = {"HAS_UPLOAD_CANDIDATE": 0, "DO_NOT_UPLOAD": 1}
        risk_order = {"safe": 0, "research": 1}
        summary["_decision_order"] = summary["decision"].map(decision_order).fillna(9).astype(int)
        summary["_risk_order"] = summary["risk"].map(risk_order).fillna(9).astype(int)
        summary = summary.sort_values(
            ["_decision_order", "_risk_order", "changed_action_rows", "mean_evidence_score"],
            ascending=[True, True, True, False],
            kind="mergesort",
        ).drop(columns=["_decision_order", "_risk_order"]).reset_index(drop=True)
    return write_reports(summary, bank, read_source_reports(), sources)


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": relative_path(OUTDIR),
                "decision": report["decision"],
                "generated_submission_count": report["generated_submission_count"],
                "generated_submissions": [row["path"] for row in report["generated_submissions"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
