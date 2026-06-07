"""V254 clean external acquisition/audit.

This script audits local external datasets before any new external pretraining.
It intentionally does not train a model and does not generate submissions.

Safety rules:
  - TTMATCH is listed as RED/banned and its contents are not read.
  - No old-server labels are used.
  - External exact labels are not mapped to AICUP actionId.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"
OUTDIR = ROOT / "v254_external_acquisition_audit"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v254_external_acquisition_audit.py"

BANNED_DATASETS = {"TTMATCH"}


@dataclass(frozen=True)
class DatasetPolicy:
    dataset: str
    tier: str
    allowed_use: str
    prohibited_use: str
    stage: str
    rationale: str


def should_read_content(dataset: str) -> bool:
    return dataset not in BANNED_DATASETS


def vector_length_from_string(value: Any) -> int:
    """Return vector length for strings like '[1.0 2.0 3.0]' or non-vectors."""
    if not isinstance(value, str):
        return 0
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return 0
    inner = text[1:-1].replace(",", " ").split()
    return len(inner)


def dataset_policy(dataset: str) -> DatasetPolicy:
    if dataset == "TTMATCH":
        return DatasetPolicy(
            dataset,
            "RED",
            "none",
            "Do not use for training, priors, lookup, blending, or validation.",
            "banned",
            "R178 marked TTMATCH high risk due overlap/AI-CUP-like provenance concerns.",
        )
    if dataset == "openttgames":
        return DatasetPolicy(
            dataset,
            "GREEN",
            "Coarse table-tennis event/sequence pretraining, phase/terminal/family priors, ball trajectory SSL.",
            "No direct mapping to AICUP exact actionId; no video/result lookup.",
            "V254/V255 pretraining",
            "Local repo has README/LICENSE and event/ball data; prior audits used it as clean external coarse signal.",
        )
    if dataset == "DeepMindrobottabletennis":
        return DatasetPolicy(
            dataset,
            "GREEN",
            "Physics/incoming-ball representation pretraining: velocity, spin, serve/rally state.",
            "No exact actionId supervision; robot domain cannot define player style.",
            "V255 physics SSL",
            "Robot table-tennis ball-state data with spin/velocity; low identity-overlap risk.",
        )
    if dataset == "CoachAI-Projects-main":
        return DatasetPolicy(
            dataset,
            "YELLOW",
            "Cross-domain sequence pretraining for phase, remaining, landing intent, and response uncertainty.",
            "No badminton shot-type to AICUP exact actionId mapping.",
            "V255 optional coarse pretraining",
            "Useful stroke-forecasting structure, but badminton is cross-domain.",
        )
    if dataset == "TT-MatchDynamics":
        return DatasetPolicy(
            dataset,
            "YELLOW",
            "Coarse landing/spin/forehand-backhand intent priors only.",
            "No direct receiver-relative pointId/actionId mapping; verify provenance before high weight.",
            "V255 low-weight auxiliary",
            "Small physics-like tabular data; license/provenance remains weaker than OpenTT/DeepMind.",
        )
    if dataset == "sonytabletennis":
        return DatasetPolicy(
            dataset,
            "YELLOW",
            "Event-level robot-vs-human table-tennis physics pretraining: shot/bounce/net, spin, velocity.",
            "No exact actionId mapping; no high-weight supervised fine-tune until license/provenance documented.",
            "V255 physics/event SSL",
            "New local data has event labels and spin/velocity; Readme exists but explicit license must be checked.",
        )
    if dataset == "TT3D":
        return DatasetPolicy(
            dataset,
            "YELLOW",
            "3D ball trajectory reconstruction SSL, depth/landing/trajectory representation only.",
            "No actionId/family supervision; no direct submission signal.",
            "V255 trajectory SSL",
            "Trajectory-only benchmark without shot labels; useful for physics representation, not action labels.",
        )
    return DatasetPolicy(
        dataset,
        "YELLOW",
        "Read-only audit only until schema/provenance are understood.",
        "No training use until promoted by V254 policy.",
        "audit-only",
        "Unknown external dataset folder.",
    )


def file_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(EXTERNAL.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(EXTERNAL)
        dataset = rel.parts[0]
        rows.append(
            {
                "dataset": dataset,
                "relative_path": str(path.relative_to(ROOT)),
                "suffix": path.suffix.lower(),
                "size_bytes": int(path.stat().st_size),
                "read_content_allowed": should_read_content(dataset),
            }
        )
    return pd.DataFrame(rows)


def dataset_inventory(files: pd.DataFrame) -> pd.DataFrame:
    if files.empty:
        return pd.DataFrame()
    rows = []
    for dataset, g in files.groupby("dataset", sort=True):
        suffix_counts = g["suffix"].value_counts().sort_index().to_dict()
        policy = dataset_policy(dataset)
        rows.append(
            {
                "dataset": dataset,
                "tier": policy.tier,
                "file_count": int(len(g)),
                "total_size_bytes": int(g["size_bytes"].sum()),
                "suffix_counts_json": json.dumps(suffix_counts, sort_keys=True),
                "read_content_allowed": bool(should_read_content(dataset)),
                "rationale": policy.rationale,
            }
        )
    return pd.DataFrame(rows)


def _safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(path, nrows=nrows, encoding="utf-8-sig")
        except Exception:
            return None


def _safe_read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def representative_files(files: pd.DataFrame) -> list[Path]:
    candidates: list[Path] = []
    preferred = [
        "external_data/sonytabletennis/data/match_data.csv",
        "external_data/TT3D/evaluation/3D_gt/001.csv",
        "external_data/TT3D/evaluation/README.md",
        "external_data/openttgames/processed/openttgames_events.csv",
        "external_data/openttgames/data/raw/game_data/train/game_1.json",
        "external_data/openttgames/data/raw/ball_data/train/train_1.json",
        "external_data/DeepMindrobottabletennis/rallies.json",
        "external_data/DeepMindrobottabletennis/serves.json",
        "external_data/TT-MatchDynamics/table_tennis_data.csv",
        "external_data/CoachAI-Projects-main/CoachAI-Challenge-IJCAI2023/ShuttleSet22/set/match.csv",
    ]
    for rel in preferred:
        p = ROOT / rel
        if p.exists():
            candidates.append(p)
    for dataset, g in files.groupby("dataset", sort=True):
        if not should_read_content(dataset):
            continue
        for suffix in [".csv", ".json", ".md", ".yaml"]:
            subset = g[g["suffix"] == suffix]
            if not subset.empty:
                candidates.append(ROOT / subset.iloc[0]["relative_path"])
    # stable unique order
    out: list[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def schema_audit(files: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(files["dataset"].unique()) if not files.empty else []:
        if not should_read_content(dataset):
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": "",
                    "kind": "banned",
                    "sample_rows": 0,
                    "columns": "",
                    "notes": "Content intentionally not read.",
                }
            )
    for path in representative_files(files):
        dataset = path.relative_to(EXTERNAL).parts[0]
        if not should_read_content(dataset):
            continue
        rel = str(path.relative_to(ROOT))
        suffix = path.suffix.lower()
        if suffix == ".csv":
            sample = _safe_read_csv(path, nrows=50)
            if sample is None:
                rows.append({"dataset": dataset, "relative_path": rel, "kind": "csv_error", "sample_rows": 0, "columns": "", "notes": "failed to read"})
                continue
            vector_cols = []
            for col in sample.columns:
                lens = sample[col].map(vector_length_from_string)
                if lens.max() >= 2:
                    vector_cols.append(f"{col}:len{int(lens.max())}")
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": rel,
                    "kind": "csv",
                    "sample_rows": int(len(sample)),
                    "columns": ",".join(map(str, sample.columns)),
                    "notes": "; ".join(vector_cols),
                }
            )
        elif suffix == ".json":
            data = _safe_read_json(path)
            if isinstance(data, list):
                keys = sorted({k for row in data[:20] if isinstance(row, dict) for k in row.keys()})
                rows.append(
                    {
                        "dataset": dataset,
                        "relative_path": rel,
                        "kind": "json_list",
                        "sample_rows": int(min(len(data), 20)),
                        "columns": ",".join(keys),
                        "notes": f"total_list_len={len(data)}",
                    }
                )
            elif isinstance(data, dict):
                rows.append(
                    {
                        "dataset": dataset,
                        "relative_path": rel,
                        "kind": "json_dict",
                        "sample_rows": 1,
                        "columns": ",".join(map(str, data.keys())),
                        "notes": "",
                    }
                )
        else:
            text = ""
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:1000]
            except Exception:
                pass
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": rel,
                    "kind": suffix.lstrip(".") or "file",
                    "sample_rows": 0,
                    "columns": "",
                    "notes": text.replace("\n", " ")[:240],
                }
            )
    return pd.DataFrame(rows)


def license_source_audit(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in files.groupby("dataset", sort=True):
        policy = dataset_policy(dataset)
        names = g["relative_path"].astype(str)
        license_files = [p for p in names if Path(p).name.lower() in {"license", "license.md", "licence", "licence.md"}]
        readmes = [p for p in names if Path(p).name.lower().startswith("readme")]
        rows.append(
            {
                "dataset": dataset,
                "tier": policy.tier,
                "license_files": "|".join(license_files[:5]),
                "readme_files": "|".join(readmes[:5]),
                "source_status": "local_files_present",
                "license_status": "explicit_license_file_present" if license_files else "needs_manual_verification",
                "notes": policy.rationale,
            }
        )
    return pd.DataFrame(rows)


def action_relevance_audit(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(files["dataset"].unique()) if not files.empty else []:
        policy = dataset_policy(dataset)
        if not should_read_content(dataset):
            rows.append(
                {
                    "dataset": dataset,
                    "tier": policy.tier,
                    "has_event_sequence": False,
                    "has_shot_or_stroke": False,
                    "has_ball_trajectory": False,
                    "has_spin_or_velocity": False,
                    "has_landing_or_area": False,
                    "has_player_context": False,
                    "has_exact_aicup_action_label": False,
                    "recommended_targets": "not_evaluated_banned_dataset",
                    "prohibited_targets": policy.prohibited_use,
                }
            )
            continue
        cols_text = " ".join(schema_audit(files).query("dataset == @dataset")["columns"].astype(str).tolist()).lower()
        path_text = " ".join(files.query("dataset == @dataset")["relative_path"].astype(str).head(200).tolist()).lower()
        combined = f"{cols_text} {path_text}"
        rows.append(
            {
                "dataset": dataset,
                "tier": policy.tier,
                "has_event_sequence": any(x in combined for x in ["event", "rally", "timestamp", "frame"]),
                "has_shot_or_stroke": any(x in combined for x in ["shot", "stroke", "hit", "bounce", "net"]),
                "has_ball_trajectory": any(x in combined for x in ["ball", "x", "y", "z", "pos", "trajectory"]),
                "has_spin_or_velocity": any(x in combined for x in ["spin", "vel", "velocity", "w_vel"]),
                "has_landing_or_area": any(x in combined for x in ["landing", "area", "x", "y", "point"]),
                "has_player_context": any(x in combined for x in ["player", "p1", "p2", "hitter"]),
                "has_exact_aicup_action_label": "actionid" in combined and dataset != "TTMATCH",
                "recommended_targets": policy.allowed_use,
                "prohibited_targets": policy.prohibited_use,
            }
        )
    return pd.DataFrame(rows)


def overlap_risk_audit(files: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    aicup_cols = {
        "rally_uid",
        "actionId",
        "pointId",
        "serverGetPoint",
        "strikeNumber",
        "scoreSelf",
        "scoreOther",
        "gamePlayerId",
        "gamePlayerOtherId",
    }
    rows = []
    for dataset in sorted(files["dataset"].unique()) if not files.empty else []:
        if dataset == "TTMATCH":
            rows.append(
                {
                    "dataset": dataset,
                    "risk": "RED",
                    "common_aicup_like_columns": "not_read",
                    "content_read": False,
                    "notes": "Banned by R178; no content read in V254.",
                }
            )
            continue
        cols = set()
        for text in schema.query("dataset == @dataset")["columns"].astype(str):
            cols.update(c.strip() for c in text.split(",") if c.strip())
        common = sorted(cols & aicup_cols)
        if {"rally_uid", "actionId", "pointId"} & set(common):
            risk = "HIGH"
        elif common:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        rows.append(
            {
                "dataset": dataset,
                "risk": risk,
                "common_aicup_like_columns": ",".join(common),
                "content_read": True,
                "notes": "Schema-only overlap check; no hidden target comparison.",
            }
        )
    return pd.DataFrame(rows)


def training_use_policy(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(files["dataset"].unique()) if not files.empty else []:
        p = dataset_policy(dataset)
        rows.append(
            {
                "dataset": p.dataset,
                "tier": p.tier,
                "allowed_use": p.allowed_use,
                "prohibited_use": p.prohibited_use,
                "stage": p.stage,
                "rationale": p.rationale,
            }
        )
    return pd.DataFrame(rows)


def write_report(outputs: dict[str, Any]) -> None:
    inv = outputs["dataset_inventory"]
    policy = outputs["training_use_policy"]
    lines = [
        "# V254 Clean External Acquisition/Audit",
        "",
        "Purpose: audit local external datasets before V255 clean external pretraining.",
        "",
        "Safety decisions:",
        "",
        "- TTMATCH is RED/banned and content was not read.",
        "- No old-server labels are used.",
        "- External labels must not map directly to AICUP exact actionId.",
        "",
        "Dataset tiers:",
        "",
    ]
    if not policy.empty:
        for _, row in policy.sort_values(["tier", "dataset"]).iterrows():
            lines.append(f"- `{row['dataset']}`: `{row['tier']}` - {row['allowed_use']}")
    lines.extend(
        [
            "",
            "Inventory summary:",
            "",
        ]
    )
    if not inv.empty:
        for _, row in inv.sort_values("dataset").iterrows():
            lines.append(
                f"- `{row['dataset']}`: {int(row['file_count'])} files, {int(row['total_size_bytes'])} bytes, tier `{row['tier']}`"
            )
    lines.extend(
        [
            "",
            "Recommended next step:",
            "",
            "Run V255 only on GREEN/YELLOW datasets according to `v254_training_use_policy.csv`; keep exact AICUP actionId supervision inside AICUP fine-tuning only.",
            "",
        ]
    )
    (OUTDIR / "v254_report.md").write_text("\n".join(lines), encoding="utf-8")

    report_json = {
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "datasets": inv.to_dict(orient="records") if not inv.empty else [],
        "policy": policy.to_dict(orient="records") if not policy.empty else [],
        "safety": {
            "ttmatch_banned": True,
            "old_server_used": False,
            "generates_submission": False,
            "trains_model": False,
        },
    }
    (OUTDIR / "v254_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = file_inventory()
    inv = dataset_inventory(files)
    schema = schema_audit(files)
    license_audit = license_source_audit(files)
    relevance = action_relevance_audit(files)
    overlap = overlap_risk_audit(files, schema)
    policy = training_use_policy(files)

    outputs = {
        "file_inventory": files,
        "dataset_inventory": inv,
        "schema_audit": schema,
        "license_source_audit": license_audit,
        "action_relevance_audit": relevance,
        "overlap_risk_audit": overlap,
        "training_use_policy": policy,
    }
    for name, df in outputs.items():
        df.to_csv(OUTDIR / f"v254_{name}.csv", index=False)
    write_report(outputs)

    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)

    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "datasets": int(inv["dataset"].nunique()) if not inv.empty else 0,
                "files": int(len(files)),
                "ttmatch_content_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
