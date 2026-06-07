"""V323 action disagreement mining and source audit.

V323 keeps the clean V306 point + V300 server anchor fixed, audits existing
action sources against the V173 action line, and only emits local candidate
CSVs when a disagreement slice has strong fold-safe evidence.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
if not (ROOT / "train.csv").exists() and len(ROOT.parents) >= 2:
    ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_lgbm import ACTION_CLASSES  # noqa: E402


OUTDIR = ROOT / "v323_action_disagreement_mining"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V286_OOF = ROOT / "v286_weak_action_specialist_pretraining" / "v286_specialist_oof.csv"
V209_OOF = ROOT / "v209_action_selector_reranker" / "v209_v208_action_point_aux_oof.npy"
V209_TEST = ROOT / "v209_action_selector_reranker" / "v209_v208_action_point_aux_test.npy"
V312_SEARCH = ROOT / "v312_action_weak_complementarity" / "v312_action_search.csv"
V317_SEARCH = ROOT / "v317_action_specialist_ensemble" / "v317_action_search.csv"

N_ACTIONS = 19
MAX_EXPORT_CHANGED_ROWS = 20
MIN_ACTION_OOF_DELTA = 0.002
MIN_FAR_ABOVE_CHANGED_PRECISION = 0.45
MIN_DELTA_PATH_CHANGED_PRECISION = 0.30
MIN_OOF_CHANGED_ROWS = 20

ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "terminal": (0, 3),
    "attack": (2, 4, 5, 7, 10, 11, 13),
    "control": (1, 6, 8, 9),
    "defensive": (12, 14),
    "serve": (15, 16, 17, 18),
    "other": (),
}


@dataclass(frozen=True)
class ExportSpec:
    filename: str


@dataclass(frozen=True)
class ActionSource:
    name: str
    family: str
    oof_pred: np.ndarray | None
    test_pred: np.ndarray | None
    path: str
    note: str = ""


BEST_SLICE_SPEC = ExportSpec("submission_v323_best_disagreement_slice__v306point_v300server.csv")
CONSENSUS_SPEC = ExportSpec("submission_v323_v173_source_consensus_safe__v306point_v300server.csv")


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def action_group(action: int) -> str:
    action = int(action)
    for group, actions in ACTION_GROUPS.items():
        if action in actions:
            return group
    return "other"


def action_distribution(values: np.ndarray) -> str:
    unique, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return json.dumps({str(int(k)): int(v) for k, v in zip(unique, counts)}, sort_keys=True)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=list(ACTION_CLASSES), average="macro", zero_division=0))


def protected_output_path(outdir: Path, spec: ExportSpec) -> Path:
    root = Path(outdir)
    path = root / spec.filename
    parts = {part.lower() for part in path.parts}
    if any("upload_candidates" in part for part in parts) or "selected" in parts or "submissions" in parts:
        raise ValueError(f"refusing non-local V323 export path: {path}")
    if path.parent != root:
        raise ValueError(f"refusing non-local V323 export path: {path}")
    return path


def build_export_frame(anchor_sub: pd.DataFrame, action: np.ndarray) -> pd.DataFrame:
    pred = np.asarray(action, dtype=int)
    if len(anchor_sub) != len(pred):
        raise ValueError(f"action rows {len(pred)} != anchor submission rows {len(anchor_sub)}")
    return pd.DataFrame(
        {
            "rally_uid": anchor_sub["rally_uid"].astype(int),
            "actionId": pred,
            "pointId": anchor_sub["pointId"].astype(int),
            "serverGetPoint": anchor_sub["serverGetPoint"].astype(float),
        }
    )


def infer_phase(frame: pd.DataFrame) -> pd.Series:
    if "phase" in frame:
        return frame["phase"].astype(str)
    if "r184_phase" in frame:
        return frame["r184_phase"].astype(str)
    if "prefix_len" not in frame:
        return pd.Series(["unknown"] * len(frame), index=frame.index)
    prefix = pd.to_numeric(frame["prefix_len"], errors="coerce").fillna(-1).astype(int)
    return pd.Series(
        np.select(
            [prefix <= 1, prefix == 2, prefix == 3, prefix == 4, prefix > 4],
            ["serve", "receive", "third_ball", "fourth_ball", "rally"],
            default="unknown",
        ),
        index=frame.index,
    )


def infer_lag_action_family(frame: pd.DataFrame) -> pd.Series:
    if "lag_action_family" in frame:
        return frame["lag_action_family"].astype(str)
    if "lag0_actionId" not in frame:
        return pd.Series(["unknown"] * len(frame), index=frame.index)
    values = pd.to_numeric(frame["lag0_actionId"], errors="coerce").fillna(-999).astype(int)
    return values.map(action_group)


def build_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame(index=frame.index)
    meta["phase"] = infer_phase(frame).astype(str).to_numpy()
    meta["lag_action_family"] = infer_lag_action_family(frame).astype(str).to_numpy()
    return meta.reset_index(drop=True)


def slice_masks(meta: pd.DataFrame, target_group: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    masks: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(len(meta), dtype=bool))]
    for col in ["phase", "lag_action_family"]:
        if col in meta:
            for value in sorted(str(v) for v in pd.Series(meta[col]).dropna().unique()):
                masks.append((col, value, meta[col].astype(str).eq(value).to_numpy()))
    for value in sorted(str(v) for v in pd.Series(target_group).dropna().unique()):
        masks.append(("target_action_group", value, np.asarray(target_group, dtype=str) == value))
    return masks


def changed_row_precision_by_slice(
    source_name: str,
    y_true: np.ndarray,
    anchor: np.ndarray,
    source: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(anchor, dtype=int)
    pred = np.asarray(source, dtype=int)
    if not (len(y) == len(base) == len(pred) == len(metadata)):
        raise ValueError("y_true, anchor, source, and metadata must have matching lengths")
    base_score = macro_f1(y, base)
    target_group = np.array([action_group(v) for v in pred], dtype=object)
    rows: list[dict[str, Any]] = []
    for slice_type, slice_value, mask in slice_masks(metadata.reset_index(drop=True), target_group):
        changed = (pred != base) & np.asarray(mask, dtype=bool)
        changed_rows = int(changed.sum())
        changed_correct = int(np.sum(changed & (pred == y)))
        candidate = base.copy()
        candidate[changed] = pred[changed]
        rows.append(
            {
                "source": source_name,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "slice_rows": int(np.sum(mask)),
                "changed_rows": changed_rows,
                "changed_correct": changed_correct,
                "changed_row_oof_precision": float(changed_correct / changed_rows) if changed_rows else 0.0,
                "action_oof_delta": float(macro_f1(y, candidate) - base_score) if changed_rows else 0.0,
                "target_distribution": action_distribution(pred[changed]) if changed_rows else "{}",
            }
        )
    return pd.DataFrame(rows)


def evidence_passes(row: dict[str, Any] | pd.Series) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    changed = int(data.get("oof_changed_rows", data.get("changed_rows", 0)))
    test_changed = int(data.get("test_changed_rows", 0))
    precision = float(data.get("changed_row_oof_precision", 0.0))
    delta = float(data.get("action_oof_delta", 0.0))
    if changed < MIN_OOF_CHANGED_ROWS or test_changed <= 0 or test_changed > MAX_EXPORT_CHANGED_ROWS:
        return False
    if delta >= MIN_ACTION_OOF_DELTA and precision >= MIN_DELTA_PATH_CHANGED_PRECISION:
        return True
    return precision >= MIN_FAR_ABOVE_CHANGED_PRECISION


def load_anchor_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    if not ANCHOR_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing V306/V300 anchor submission: {ANCHOR_SUBMISSION}")
    if not V286_OOF.exists():
        raise FileNotFoundError(f"Missing V286 OOF labels: {V286_OOF}")
    from analysis_v290_shortcontrol411_specialist import load_anchor_frames as load_v290_anchor_frames

    rows, test_rows, rebuilt_y, rebuilt_anchor = load_v290_anchor_frames()
    oof = pd.read_csv(V286_OOF)
    anchor_sub = pd.read_csv(ANCHOR_SUBMISSION)
    rows = rows.reset_index(drop=True).copy()
    test_rows = test_rows.reset_index(drop=True).copy()
    if len(oof) != len(rows):
        raise ValueError(f"V286 OOF length {len(oof)} != train rows {len(rows)}")
    if len(anchor_sub) != len(test_rows):
        raise ValueError(f"anchor submission rows {len(anchor_sub)} != test rows {len(test_rows)}")
    y = oof["y_true_action"].astype(int).to_numpy()
    anchor_oof = oof["anchor_action"].astype(int).to_numpy()
    if len(rebuilt_y) == len(y) and not np.array_equal(np.asarray(rebuilt_y, dtype=int), y):
        raise ValueError("rebuilt action labels differ from V286 OOF labels")
    if len(rebuilt_anchor) == len(anchor_oof) and not np.array_equal(np.asarray(rebuilt_anchor, dtype=int), anchor_oof):
        raise ValueError("rebuilt V173 action anchor differs from V286 OOF anchor")
    return rows, test_rows, y, anchor_oof, anchor_sub


def load_npy_argmax_source() -> ActionSource | None:
    if not (V209_OOF.exists() and V209_TEST.exists()):
        return None
    oof = np.load(V209_OOF)
    test = np.load(V209_TEST)
    if oof.ndim != 2 or test.ndim != 2:
        return None
    return ActionSource(
        name="v209_v208_action_point_aux_argmax",
        family="v209_probability_artifact",
        oof_pred=oof.argmax(axis=1).astype(int),
        test_pred=test.argmax(axis=1).astype(int),
        path=str(V209_OOF.relative_to(ROOT)),
        note="Argmax of existing V209 V208 action-point auxiliary probability artifacts.",
    )


def load_r184_base_source() -> ActionSource | None:
    try:
        from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

        state = rebuild_v173_best_actions()
    except Exception:
        return None
    if not {"base_pred_oof", "base_pred_test"}.issubset(state):
        return None
    return ActionSource(
        name="r184_rebuilt_base_action",
        family="rebuilt_internal_base",
        oof_pred=np.asarray(state["base_pred_oof"], dtype=int),
        test_pred=np.asarray(state["base_pred_test"], dtype=int),
        path="analysis_r184_receiver_affordance_refiner.rebuild_v173_best_actions",
        note="Existing rebuilt pre-V173 base action source from R184 support audit.",
    )


def load_v286_specialist_source() -> ActionSource | None:
    if not V286_OOF.exists():
        return None
    oof = pd.read_csv(V286_OOF)
    score_cols = [col for col in oof.columns if col.startswith("specialist_p_")]
    if not score_cols:
        return None
    pred = oof["anchor_action"].astype(int).to_numpy()
    scores = oof[score_cols].to_numpy(dtype=float)
    actions = np.array([int(col.removeprefix("specialist_p_")) for col in score_cols], dtype=int)
    best = actions[np.nan_to_num(scores, nan=-np.inf).argmax(axis=1)]
    pred = best.astype(int)
    return ActionSource(
        name="v286_specialist_oof_argmax",
        family="v286_oof_only_specialist_scores",
        oof_pred=pred,
        test_pred=None,
        path=str(V286_OOF.relative_to(ROOT)),
        note="OOF-only argmax over existing V286 weak-action specialist score columns; audit only.",
    )


def source_submission_paths() -> list[Path]:
    roots = [
        ROOT / "v173_external_curriculum_pretrain",
        ROOT / "v191_v188_cap5_action_packager",
        ROOT / "r184_receiver_affordance_refiner",
        ROOT / "v197_action_teacher_surgery",
        ROOT / "v209_action_selector_reranker",
        ROOT / "v214_shuttlenet_component_ablation",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("submission*.csv")):
            text = str(path).lower()
            if "ttmatch" in text or "old_server" in text:
                continue
            paths.append(path)
    return paths


def load_submission_disagreement_table(anchor_sub: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_action = anchor_sub["actionId"].astype(int).to_numpy()
    records: list[dict[str, Any]] = []
    rows: list[pd.DataFrame] = []
    for path in source_submission_paths():
        try:
            sub = pd.read_csv(path)
        except Exception:
            continue
        if len(sub) != len(anchor_sub) or "actionId" not in sub:
            continue
        if "rally_uid" in sub and not sub["rally_uid"].astype(int).reset_index(drop=True).equals(
            anchor_sub["rally_uid"].astype(int).reset_index(drop=True)
        ):
            continue
        source_action = sub["actionId"].astype(int).to_numpy()
        changed = source_action != anchor_action
        source_name = path.stem
        records.append(
            {
                "source": source_name,
                "family": path.parent.name,
                "path": str(path.relative_to(ROOT)),
                "rows": int(len(sub)),
                "test_changed_rows": int(changed.sum()),
                "test_churn_vs_v173": float(changed.mean()) if len(changed) else 0.0,
                "test_changed_distribution": action_distribution(source_action[changed]) if changed.any() else "{}",
            }
        )
        if changed.any():
            part = pd.DataFrame(
                {
                    "source": source_name,
                    "rally_uid": anchor_sub["rally_uid"].astype(int),
                    "anchor_action": anchor_action,
                    "source_action": source_action,
                    "phase": test_meta["phase"],
                    "lag_action_family": test_meta["lag_action_family"],
                    "target_action_group": [action_group(v) for v in source_action],
                }
            )
            rows.append(part.loc[changed].reset_index(drop=True))
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.DataFrame(records).sort_values(["test_changed_rows", "source"], ascending=[False, True]), detail


def add_test_changed_counts(
    audit: pd.DataFrame,
    sources: list[ActionSource],
    anchor_test: np.ndarray,
    test_meta: pd.DataFrame,
) -> pd.DataFrame:
    if audit.empty:
        return audit
    out = audit.copy()
    out["test_changed_rows"] = 0
    for source in sources:
        if source.test_pred is None:
            continue
        pred = np.asarray(source.test_pred, dtype=int)
        if len(pred) != len(anchor_test):
            continue
        target_group = np.array([action_group(v) for v in pred], dtype=object)
        for idx, row in out[out["source"].eq(source.name)].iterrows():
            slice_type = str(row["slice_type"])
            slice_value = str(row["slice_value"])
            if slice_type == "overall":
                mask = np.ones(len(pred), dtype=bool)
            elif slice_type == "target_action_group":
                mask = target_group == slice_value
            elif slice_type in test_meta:
                mask = test_meta[slice_type].astype(str).eq(slice_value).to_numpy()
            else:
                mask = np.zeros(len(pred), dtype=bool)
            out.loc[idx, "test_changed_rows"] = int(np.sum((pred != anchor_test) & mask))
    out["evidence_pass"] = out.apply(lambda row: int(evidence_passes(row)), axis=1)
    return out


def load_reference_evidence() -> dict[str, Any]:
    ref = {
        "v312_best_delta": 0.0004923976482905656,
        "v312_best_changed_precision": 0.24691358024691357,
        "v317_min_changed_precision_gate": 0.30,
        "v317_best_delta": None,
        "v317_best_changed_precision": None,
    }
    if V312_SEARCH.exists():
        try:
            df = pd.read_csv(V312_SEARCH)
            ref["v312_best_delta"] = float(df["action_oof_delta"].max())
            ref["v312_best_changed_precision"] = float(df["changed_row_oof_precision"].max())
        except Exception:
            pass
    if V317_SEARCH.exists():
        try:
            df = pd.read_csv(V317_SEARCH)
            ref["v317_best_delta"] = float(df["action_oof_delta"].max())
            ref["v317_best_changed_precision"] = float(df["changed_row_oof_precision"].max())
        except Exception:
            pass
    return ref


def build_best_slice_prediction(
    audit: pd.DataFrame,
    sources: list[ActionSource],
    anchor_test: np.ndarray,
    test_meta: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    passed = audit[audit["evidence_pass"].astype(int).eq(1)].copy() if not audit.empty else pd.DataFrame()
    passed = passed.sort_values(
        ["action_oof_delta", "changed_row_oof_precision", "test_changed_rows"],
        ascending=[False, False, True],
    )
    source_map = {source.name: source for source in sources if source.test_pred is not None}
    for _, row in passed.iterrows():
        source = source_map.get(str(row["source"]))
        if source is None:
            continue
        pred = np.asarray(source.test_pred, dtype=int)
        target_group = np.array([action_group(v) for v in pred], dtype=object)
        slice_type = str(row["slice_type"])
        slice_value = str(row["slice_value"])
        if slice_type == "overall":
            mask = np.ones(len(pred), dtype=bool)
        elif slice_type == "target_action_group":
            mask = target_group == slice_value
        elif slice_type in test_meta:
            mask = test_meta[slice_type].astype(str).eq(slice_value).to_numpy()
        else:
            continue
        selected = (pred != anchor_test) & mask
        if 0 < int(selected.sum()) <= MAX_EXPORT_CHANGED_ROWS:
            out = anchor_test.copy()
            out[selected] = pred[selected]
            meta = row.to_dict()
            meta["export_changed_rows"] = int(selected.sum())
            meta["export_changed_distribution"] = action_distribution(out[selected])
            return out, meta
    return None, {}


def consensus_prediction(
    source_preds: list[np.ndarray],
    anchor: np.ndarray,
) -> np.ndarray:
    base = np.asarray(anchor, dtype=int)
    out = base.copy()
    if len(source_preds) < 2:
        return out
    stacked = np.vstack([np.asarray(pred, dtype=int) for pred in source_preds])
    for i in range(len(base)):
        vals, counts = np.unique(stacked[:, i], return_counts=True)
        order = np.argsort(-counts, kind="mergesort")
        best_val = int(vals[order[0]])
        best_count = int(counts[order[0]])
        if best_count >= 2 and best_val != int(base[i]):
            out[i] = best_val
    return out


def build_consensus_prediction(
    sources: list[ActionSource],
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    oof_sources = [np.asarray(source.oof_pred, dtype=int) for source in sources if source.oof_pred is not None]
    test_sources = [np.asarray(source.test_pred, dtype=int) for source in sources if source.test_pred is not None]
    oof_sources = [pred for pred in oof_sources if len(pred) == len(anchor_oof)]
    test_sources = [pred for pred in test_sources if len(pred) == len(anchor_test)]
    if len(oof_sources) < 2 or len(test_sources) < 2:
        return None, {}
    pred_oof = consensus_prediction(oof_sources, anchor_oof)
    pred_test = consensus_prediction(test_sources, anchor_test)
    changed = pred_oof != anchor_oof
    test_changed = pred_test != anchor_test
    rec = {
        "source": "source_consensus",
        "slice_type": "consensus",
        "slice_value": "two_or_more_sources",
        "oof_changed_rows": int(changed.sum()),
        "changed_rows": int(changed.sum()),
        "changed_correct": int(np.sum(changed & (pred_oof == y))),
        "changed_row_oof_precision": float(np.sum(changed & (pred_oof == y)) / int(changed.sum())) if changed.any() else 0.0,
        "action_oof_delta": float(macro_f1(y, pred_oof) - macro_f1(y, anchor_oof)) if changed.any() else 0.0,
        "test_changed_rows": int(test_changed.sum()),
        "evidence_pass": 0,
        "export_changed_distribution": action_distribution(pred_test[test_changed]) if test_changed.any() else "{}",
    }
    rec["evidence_pass"] = int(evidence_passes(rec))
    if rec["evidence_pass"] and 0 < int(test_changed.sum()) <= MAX_EXPORT_CHANGED_ROWS:
        return pred_test, rec
    return None, rec


def markdown_table(rows: pd.DataFrame, columns: list[str]) -> str:
    if rows.empty:
        return "_None_"

    def cell(value: Any) -> str:
        if isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        return text.replace("|", "\\|")

    records = rows[columns].to_dict(orient="records")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in records]
    return "\n".join([header, sep, *body])


def write_submission(spec: ExportSpec, action: np.ndarray, anchor_sub: pd.DataFrame) -> str:
    path = protected_output_path(OUTDIR, spec)
    out = build_export_frame(anchor_sub, action)
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path.relative_to(ROOT))


def run_pipeline() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTDIR.glob("submission_v323*.csv"):
        stale.unlink()

    rows, test_rows, y, anchor_oof, anchor_sub = load_anchor_frames()
    anchor_test = anchor_sub["actionId"].astype(int).to_numpy()
    train_meta = build_metadata(rows)
    test_meta = build_metadata(test_rows)

    sources = [source for source in [load_npy_argmax_source(), load_r184_base_source(), load_v286_specialist_source()] if source]
    valid_oof_sources = [src for src in sources if src.oof_pred is not None and len(src.oof_pred) == len(anchor_oof)]
    audit_parts = [
        changed_row_precision_by_slice(src.name, y, anchor_oof, np.asarray(src.oof_pred, dtype=int), train_meta)
        for src in valid_oof_sources
    ]
    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    if not audit.empty:
        audit = audit.rename(columns={"changed_rows": "oof_changed_rows"})
        audit["changed_rows"] = audit["oof_changed_rows"]
        audit = add_test_changed_counts(audit, sources, anchor_test, test_meta)
        audit = audit.sort_values(
            ["evidence_pass", "action_oof_delta", "changed_row_oof_precision", "test_changed_rows"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    source_summary, disagreement_detail = load_submission_disagreement_table(anchor_sub, test_meta)
    if not source_summary.empty:
        source_summary.to_csv(OUTDIR / "v323_submission_source_summary.csv", index=False)
    if not disagreement_detail.empty:
        disagreement_detail.to_csv(OUTDIR / "v323_test_disagreement_table.csv", index=False)
    if not audit.empty:
        audit.to_csv(OUTDIR / "v323_oof_slice_audit.csv", index=False)

    generated: list[str] = []
    best_pred, best_meta = build_best_slice_prediction(audit, sources, anchor_test, test_meta)
    if best_pred is not None:
        generated.append(write_submission(BEST_SLICE_SPEC, best_pred, anchor_sub))
    consensus_pred, consensus_meta = build_consensus_prediction(sources, y, anchor_oof, anchor_test)
    if consensus_pred is not None:
        generated.append(write_submission(CONSENSUS_SPEC, consensus_pred, anchor_sub))

    decision = "REVIEW_ACTION" if generated else "AUDIT_ONLY_DO_NOT_UPLOAD"
    top_audit = audit.head(12) if not audit.empty else pd.DataFrame()
    report = json_safe(
        {
            "version": "V323",
            "decision": decision,
            "anchor_submission": str(ANCHOR_SUBMISSION.relative_to(ROOT)),
            "action_anchor": "V173 action from V306 clean public-best submission",
            "point_fixed_to": "V306 p0 cap0p01 pointId",
            "server_fixed_to": "V300 serverGetPoint",
            "ttmatch_used": False,
            "old_server_used": False,
            "copied_to_upload_or_selected": False,
            "evidence_thresholds": {
                "min_action_oof_delta": MIN_ACTION_OOF_DELTA,
                "min_far_above_changed_precision": MIN_FAR_ABOVE_CHANGED_PRECISION,
                "min_delta_path_changed_precision": MIN_DELTA_PATH_CHANGED_PRECISION,
                "min_oof_changed_rows": MIN_OOF_CHANGED_ROWS,
                "max_export_changed_rows": MAX_EXPORT_CHANGED_ROWS,
            },
            "reference_evidence": load_reference_evidence(),
            "oof_sources": [
                {
                    "name": src.name,
                    "family": src.family,
                    "path": src.path,
                    "has_test_pred": src.test_pred is not None,
                    "note": src.note,
                }
                for src in sources
            ],
            "submission_sources_audited": int(len(source_summary)),
            "best_slice_candidate": best_meta,
            "consensus_candidate": consensus_meta,
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "top_audit_rows": top_audit.to_dict(orient="records") if not top_audit.empty else [],
        }
    )
    (OUTDIR / "v323_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    top_cols = [
        "source",
        "slice_type",
        "slice_value",
        "action_oof_delta",
        "changed_row_oof_precision",
        "oof_changed_rows",
        "test_changed_rows",
        "evidence_pass",
    ]
    md = [
        "# V323 action disagreement mining and source audit",
        "",
        f"Decision: `{decision}`",
        f"Anchor submission: `{ANCHOR_SUBMISSION.relative_to(ROOT)}`",
        "Point/server: fixed to V306 point and V300 server.",
        "",
        "## Top OOF slice audit rows",
        "",
        markdown_table(top_audit, top_cols) if not top_audit.empty else "_No OOF source rows available._",
        "",
        "## Generated local submissions",
        "",
        *[f"- `{name}`" for name in generated],
    ]
    (OUTDIR / "v323_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_pipeline()
    best = report.get("best_slice_candidate", {})
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "decision": report.get("decision"),
                "best_source": best.get("source", ""),
                "best_slice": f"{best.get('slice_type', '')}:{best.get('slice_value', '')}",
                "best_action_oof_delta": best.get("action_oof_delta", 0.0),
                "best_changed_row_oof_precision": best.get("changed_row_oof_precision", 0.0),
                "generated_submission_count": report.get("generated_submission_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
