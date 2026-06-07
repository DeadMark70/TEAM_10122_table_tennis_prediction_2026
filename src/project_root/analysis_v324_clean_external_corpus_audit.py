"""V324 clean external corpus feature audit.

Audits local external_data and existing clean OpenTT/CoachAI-style canonical
artifacts for usable coarse pretraining signal. This script writes reports only:
no model training, no submission files, no TTMATCH content reads, and no exact
external-label mapping to AICUP actionId/pointId labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"
OUTDIR = ROOT / "v324_clean_external_corpus_audit"
V255_CANONICAL = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"
V274_CANONICAL_SAMPLE = ROOT / "v274_clean_external_representation" / "v274_canonical_samples.csv"


@dataclass(frozen=True)
class ResourcePolicy:
    resource: str
    status: str
    content_readable: bool
    usable_for: str
    prohibited_use: str
    rationale: str


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resource_name_from_path(path: Path, external_root: Path = EXTERNAL) -> str:
    try:
        parts = path.relative_to(external_root).parts
    except ValueError:
        parts = path.parts
    return parts[0] if parts else ""


def resource_policy(resource: str) -> ResourcePolicy:
    name = str(resource)
    low = name.lower()
    if name.upper() == "TTMATCH":
        return ResourcePolicy(
            name,
            "banned",
            False,
            "none",
            "Do not read contents or use for training, priors, lookup, blending, validation, or packaging.",
            "Explicitly excluded by V324 scope and prior overlap/provenance risk.",
        )
    if low.startswith("opentt"):
        return ResourcePolicy(
            name,
            "clean",
            True,
            "Table-tennis coarse family, phase, response-style sequence, trajectory/side representation pretraining.",
            "No reverse lookup, video lookup, or mapping to exact AICUP actionId/pointId labels.",
            "Closest clean table-tennis event corpus with existing processed OpenTT resources.",
        )
    if low.startswith("coachai") or low.startswith("shuttleset"):
        return ResourcePolicy(
            name,
            "coarse_only",
            True,
            "Cross-domain coarse phase, sequence-length, landing-intent, and response-style contrastive signal.",
            "No badminton shot-type to exact AICUP actionId mapping.",
            "Useful sequence structure but badminton domain mismatch requires coarse-only use.",
        )
    if name in {"DeepMindrobottabletennis", "sonytabletennis", "TT3D", "TT-MatchDynamics"}:
        return ResourcePolicy(
            name,
            "auxiliary_only",
            True,
            "Physics, trajectory, spin/speed, landing, and generic table-tennis representation signal.",
            "No exact AICUP actionId/pointId supervision; no high-weight supervised fine-tune without separate validation.",
            "Clean enough for auxiliary features but weaker schema alignment than processed OpenTT events.",
        )
    return ResourcePolicy(
        name,
        "unknown_audit_only",
        True,
        "Read-only schema audit until provenance and labels are understood.",
        "No training use until promoted by a later policy.",
        "Unclassified external resource.",
    )


def should_read_content(path: Path | str, external_root: Path = EXTERNAL) -> bool:
    p = Path(path)
    text_parts = [part.upper() for part in p.parts]
    if "TTMATCH" in text_parts:
        return False
    return resource_policy(resource_name_from_path(p, external_root)).content_readable


def _safe_csv_schema(path: Path, sample_rows: int) -> tuple[int, str, str]:
    try:
        sample = pd.read_csv(path, nrows=sample_rows)
    except UnicodeDecodeError:
        sample = pd.read_csv(path, nrows=sample_rows, encoding="utf-8-sig")
    return int(len(sample)), "|".join(map(str, sample.columns)), "sampled_csv"


def _safe_json_schema(path: Path) -> tuple[int, str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        first = data[0] if data else {}
        columns = sorted(first.keys()) if isinstance(first, dict) else [type(first).__name__]
        return int(len(data)), "|".join(map(str, columns)), "sampled_json_list"
    if isinstance(data, dict):
        return int(len(data)), "|".join(map(str, sorted(data.keys()))), "sampled_json_dict"
    return 1, type(data).__name__, "sampled_json_scalar"


def schema_row(path: Path, external_root: Path = EXTERNAL, sample_rows: int = 25) -> dict[str, Any]:
    resource = resource_name_from_path(path, external_root)
    content_read = should_read_content(path, external_root)
    base = {
        "resource": resource,
        "relative_path": rel(path),
        "suffix": path.suffix.lower(),
        "size_bytes": int(path.stat().st_size),
        "content_read": bool(content_read),
        "sample_rows": 0,
        "columns": "",
        "notes": "",
    }
    if not content_read:
        base["notes"] = "content intentionally not read"
        return base
    if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
        base["notes"] = "schema sampling not applicable"
        return base
    try:
        if path.suffix.lower() == ".csv":
            rows, columns, notes = _safe_csv_schema(path, sample_rows)
        elif path.suffix.lower() == ".jsonl":
            lines = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for _, line in zip(range(sample_rows), f):
                    if line.strip():
                        lines.append(json.loads(line))
            keys = sorted({k for item in lines if isinstance(item, dict) for k in item})
            rows, columns, notes = len(lines), "|".join(map(str, keys)), "sampled_jsonl"
        else:
            rows, columns, notes = _safe_json_schema(path)
        base.update({"sample_rows": int(rows), "columns": columns, "notes": notes})
    except Exception as exc:  # keep audit robust across third-party files
        base["notes"] = f"schema_error: {type(exc).__name__}: {exc}"
    return base


def build_schema_inventory(external_root: Path = EXTERNAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    files: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    if not external_root.exists():
        return (
            pd.DataFrame(columns=["resource", "file_count", "total_size_bytes", "status"]),
            pd.DataFrame(
                columns=[
                    "resource",
                    "relative_path",
                    "suffix",
                    "size_bytes",
                    "content_read",
                    "sample_rows",
                    "columns",
                    "notes",
                ]
            ),
        )
    for path in sorted(external_root.rglob("*")):
        if not path.is_file():
            continue
        resource = resource_name_from_path(path, external_root)
        policy = resource_policy(resource)
        files.append(
            {
                "resource": resource,
                "relative_path": rel(path),
                "suffix": path.suffix.lower(),
                "size_bytes": int(path.stat().st_size),
                "content_readable": bool(policy.content_readable),
                "status": policy.status,
                "usable_for": policy.usable_for,
                "prohibited_use": policy.prohibited_use,
                "rationale": policy.rationale,
            }
        )
        schemas.append(schema_row(path, external_root))
    file_df = pd.DataFrame(files)
    if file_df.empty:
        inventory = pd.DataFrame(columns=["resource", "file_count", "total_size_bytes", "status"])
    else:
        inventory = (
            file_df.groupby("resource", dropna=False)
            .agg(
                file_count=("relative_path", "size"),
                total_size_bytes=("size_bytes", "sum"),
                status=("status", "first"),
                content_readable=("content_readable", "first"),
                usable_for=("usable_for", "first"),
                prohibited_use=("prohibited_use", "first"),
                rationale=("rationale", "first"),
            )
            .reset_index()
            .sort_values(["status", "resource"])
        )
    schema_df = pd.DataFrame(schemas)
    return inventory, schema_df


def _canonical_from_v274_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["source_dataset"] = df.get("source", pd.Series("", index=df.index)).replace(
        {"OpenTT": "openttgames", "ShuttleSet22": "CoachAI-Projects-main", "CoachAI": "CoachAI-Projects-main"}
    )
    out["sequence_id"] = df.get("sequence_id", pd.Series("", index=df.index))
    out["coarse_family"] = df.get("action_family", pd.Series(pd.NA, index=df.index))
    out["phase"] = df.get("phase", pd.Series(pd.NA, index=df.index))
    out["terminal_like"] = df.get("terminal_or_remaining", pd.Series("", index=df.index)).astype(str).eq("terminal")
    out["landing_x"] = df.get("landing_width_or_side", pd.Series(pd.NA, index=df.index))
    out["landing_y"] = df.get("landing_depth", pd.Series(pd.NA, index=df.index))
    out["speed"] = pd.NA
    out["spin"] = pd.NA
    out["risk_tier"] = "V274_SAMPLE"
    return out


def load_existing_canonical_corpus() -> pd.DataFrame:
    if V255_CANONICAL.exists():
        return pd.read_csv(V255_CANONICAL, low_memory=False)
    if V274_CANONICAL_SAMPLE.exists():
        return _canonical_from_v274_sample(V274_CANONICAL_SAMPLE)
    return pd.DataFrame()


def _availability(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return 0.0
    return float(df[column].notna().mean())


def _counts_json(series: pd.Series) -> str:
    counts = series.fillna("missing").astype(str).value_counts().sort_index().to_dict()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)


def _sequence_quantiles_json(group: pd.DataFrame) -> str:
    if "sequence_id" not in group.columns or group.empty:
        return json.dumps({}, sort_keys=True)
    lengths = group.groupby("sequence_id", dropna=False).size()
    quantiles = lengths.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    payload = {str(k): float(v) for k, v in quantiles.items()}
    payload["max"] = int(lengths.max()) if len(lengths) else 0
    return json.dumps(payload, sort_keys=True)


def _schema_compatibility(policy: ResourcePolicy, group: pd.DataFrame) -> str:
    if policy.status == "clean":
        return "high"
    if policy.status == "coarse_only":
        return "medium"
    if policy.status == "auxiliary_only":
        return "medium" if (_availability(group, "landing_y") > 0.2 or _availability(group, "speed") > 0.2) else "low"
    return "low"


def summarize_canonical_features(canonical: pd.DataFrame) -> dict[str, pd.DataFrame]:
    columns = [
        "resource",
        "status",
        "canonical_rows",
        "sequences",
        "action_family_counts_json",
        "phase_counts_json",
        "sequence_length_quantiles_json",
        "action_family_available_rate",
        "phase_available_rate",
        "landing_depth_available_rate",
        "landing_side_available_rate",
        "spin_available_rate",
        "speed_available_rate",
        "terminal_rate",
        "schema_compatibility",
        "usable_for",
    ]
    if canonical.empty:
        return {"resource_summary": pd.DataFrame(columns=columns)}
    source_col = "source_dataset" if "source_dataset" in canonical.columns else "source"
    rows: list[dict[str, Any]] = []
    for resource, group in canonical.groupby(source_col, dropna=False, sort=True):
        resource_text = str(resource)
        policy = resource_policy(resource_text)
        sequence_col = group["sequence_id"] if "sequence_id" in group.columns else pd.Series(np.arange(len(group)))
        family_col = group["coarse_family"] if "coarse_family" in group.columns else group.get("action_family", pd.Series(pd.NA, index=group.index))
        phase_col = group["phase"] if "phase" in group.columns else pd.Series(pd.NA, index=group.index)
        terminal_col = group["terminal_like"] if "terminal_like" in group.columns else pd.Series(False, index=group.index)
        rows.append(
            {
                "resource": resource_text,
                "status": policy.status,
                "canonical_rows": int(len(group)),
                "sequences": int(sequence_col.nunique(dropna=False)),
                "action_family_counts_json": _counts_json(family_col),
                "phase_counts_json": _counts_json(phase_col),
                "sequence_length_quantiles_json": _sequence_quantiles_json(group.assign(sequence_id=sequence_col)),
                "action_family_available_rate": float(family_col.notna().mean()) if len(group) else 0.0,
                "phase_available_rate": float(phase_col.notna().mean()) if len(group) else 0.0,
                "landing_depth_available_rate": _availability(group, "landing_y"),
                "landing_side_available_rate": _availability(group, "landing_x"),
                "spin_available_rate": _availability(group, "spin"),
                "speed_available_rate": _availability(group, "speed"),
                "terminal_rate": float(pd.Series(terminal_col).fillna(False).astype(bool).mean()) if len(group) else 0.0,
                "schema_compatibility": _schema_compatibility(policy, group),
                "usable_for": policy.usable_for,
            }
        )
    return {"resource_summary": pd.DataFrame(rows, columns=columns).sort_values(["canonical_rows", "resource"], ascending=[False, True])}


def _resource_score(row: pd.Series) -> float:
    rows_score = min(float(row.get("canonical_rows", 0)) / 10000.0, 2.0)
    sequence_score = min(float(row.get("sequences", 0)) / 50.0, 1.0)
    status_score = {"clean": 4.0, "coarse_only": 2.0, "auxiliary_only": 0.75}.get(
        str(row.get("status", "")), 0.0
    )
    compatibility = {"high": 2.0, "medium": 1.0, "low": 0.0}.get(str(row.get("schema_compatibility", "low")), 0.0)
    label_score = float(row.get("action_family_available_rate", 0)) + float(row.get("phase_available_rate", 0))
    landing_score = 0.5 * (
        float(row.get("landing_depth_available_rate", 0)) + float(row.get("landing_side_available_rate", 0))
    )
    return status_score + rows_score + sequence_score + compatibility + label_score + landing_score


def rank_recommendations(resource_summary: pd.DataFrame) -> pd.DataFrame:
    if resource_summary.empty:
        base_resources = "none"
        signal = "No canonical clean corpus rows found."
    else:
        scored = resource_summary.copy()
        scored["score"] = scored.apply(_resource_score, axis=1)
        usable = scored[~scored["status"].eq("banned")].sort_values(["score", "canonical_rows"], ascending=[False, False])
        base_resources = ", ".join(usable.head(3)["resource"].astype(str).tolist()) if not usable.empty else "none"
        signal = f"Top resources by coarse signal: {base_resources}."
    records = [
        {
            "rank": 1,
            "experiment": "masked family pretrain",
            "next_script": "analysis_v326_masked_family_pretrain.py",
            "resources": base_resources,
            "rationale": signal
            + " Highest priority because family/phase masking is clean, uses no exact labels, and can initialize weak action-family encoders.",
        },
        {
            "rank": 2,
            "experiment": "response-style contrastive",
            "next_script": "analysis_v327_response_style_contrastive.py",
            "resources": base_resources,
            "rationale": signal
            + " Use sequence neighborhoods, phase, side/depth, and terminal context to learn response-style embeddings without hidden labels.",
        },
        {
            "rank": 3,
            "experiment": "coarse-to-exact distillation",
            "next_script": "analysis_v328_coarse_to_exact_distillation.py",
            "resources": base_resources,
            "rationale": signal
            + " Defer until masked/contrastive embeddings exist; exact AICUP heads must be trained only on AICUP labels.",
        },
    ]
    return pd.DataFrame(records)


def write_outputs(
    outdir: Path,
    inventory: pd.DataFrame,
    schemas: pd.DataFrame,
    feature_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(outdir / "v324_external_file_inventory.csv", index=False)
    schemas.to_csv(outdir / "v324_schema_summary.csv", index=False)
    feature_summary.to_csv(outdir / "v324_canonical_feature_summary.csv", index=False)
    recommendations.to_csv(outdir / "v324_recommendations.csv", index=False)
    payload = {
        "submissions_written": 0,
        "ttmatch_content_rows_read": 0,
        "resources": inventory.to_dict(orient="records"),
        "feature_summary": feature_summary.to_dict(orient="records"),
        "recommendations": recommendations.to_dict(orient="records"),
    }
    (outdir / "v324_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# V324 Clean External Corpus Feature Audit",
        "",
        "- Submission files written: `0`",
        "- TTMATCH content rows read: `0`",
        "- Scope: clean/coarse external pretraining and feature-signal audit only.",
        "",
        "## Corpus Findings",
        "",
    ]
    if feature_summary.empty:
        lines.append("No existing canonical corpus artifact was found.")
    else:
        for row in feature_summary.itertuples(index=False):
            family_avail = float(getattr(row, "action_family_available_rate", 0.0))
            phase_avail = float(getattr(row, "phase_available_rate", 0.0))
            landing_depth_avail = float(getattr(row, "landing_depth_available_rate", 0.0))
            landing_side_avail = float(getattr(row, "landing_side_available_rate", 0.0))
            speed_avail = float(getattr(row, "speed_available_rate", 0.0))
            spin_avail = float(getattr(row, "spin_available_rate", 0.0))
            lines.append(
                f"- `{row.resource}` ({row.status}, {row.schema_compatibility} compatibility): "
                f"rows={int(row.canonical_rows)}, sequences={int(row.sequences)}, "
                f"family_avail={family_avail:.3f}, "
                f"phase_avail={phase_avail:.3f}, "
                f"landing_depth_avail={landing_depth_avail:.3f}, "
                f"landing_side_avail={landing_side_avail:.3f}, "
                f"speed_avail={speed_avail:.3f}, spin_avail={spin_avail:.3f}"
            )
    lines.extend(["", "## Ranked Next Trainable Experiments", ""])
    for row in recommendations.itertuples(index=False):
        lines.append(f"{int(row.rank)}. `{row.experiment}` -> `{row.next_script}`")
        lines.append(f"   Resources: {row.resources}")
        lines.append(f"   Rationale: {row.rationale}")
    lines.extend(
        [
            "",
            "## Safety Notes",
            "",
            "- `external_data/TTMATCH` is inventoried by file metadata only; contents are not sampled.",
            "- CoachAI/ShuttleSet rows are coarse-only and cannot supervise exact AICUP labels.",
            "- Coarse-to-exact distillation means external coarse encoders into AICUP-only exact heads, not external exact-label mapping.",
        ]
    )
    (outdir / "v324_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(external_root: Path = EXTERNAL, outdir: Path = OUTDIR) -> dict[str, Any]:
    inventory, schemas = build_schema_inventory(external_root)
    canonical = load_existing_canonical_corpus()
    feature_summary = summarize_canonical_features(canonical)["resource_summary"]
    recommendations = rank_recommendations(feature_summary)
    write_outputs(outdir, inventory, schemas, feature_summary, recommendations)
    return {
        "outdir": rel(outdir),
        "resources": int(len(inventory)),
        "schema_files": int(len(schemas)),
        "canonical_resources": int(len(feature_summary)),
        "submissions_written": 0,
        "ttmatch_content_rows_read": 0,
        "top_recommendation": recommendations.iloc[0]["experiment"] if not recommendations.empty else "",
    }


def main() -> None:
    print(json.dumps(run_audit(), ensure_ascii=True))


if __name__ == "__main__":
    main()
