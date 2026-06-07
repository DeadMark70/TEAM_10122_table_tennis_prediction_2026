"""V274 clean external representation audit.

Audits allowed external sequence sources and converts usable rows to coarse
canonical fields for future representation learning. This script never maps
external labels to exact AICUP actionId or pointId labels and never writes a
submission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"
OUTDIR = ROOT / "v274_clean_external_representation"

CANONICAL_FIELDS = [
    "action_family",
    "phase",
    "landing_depth",
    "landing_width_or_side",
    "terminal_or_remaining",
]

CANONICAL_COLUMNS = [
    "source",
    "source_path",
    "sequence_id",
    "event_index",
    "source_row",
    *CANONICAL_FIELDS,
    "raw_label",
]

INVENTORY_COLUMNS = [
    "source",
    "path",
    "file_type",
    "bytes",
    "rows",
    "columns",
    "usable_rows",
    "missing_action_family_rate",
    "missing_phase_rate",
    "missing_landing_depth_rate",
    "missing_landing_width_or_side_rate",
    "missing_terminal_or_remaining_rate",
    "status",
    "notes",
]

BADMINTON_FAMILY_MAP = {
    "serve": "Serve",
    "short service": "Serve",
    "long service": "Serve",
    "clear": "Defensive",
    "lob": "Defensive",
    "defensive clear": "Defensive",
    "smash": "Attack",
    "push/rush": "Attack",
    "drive": "Attack",
    "drop": "Control",
    "net shot": "Control",
    "net": "Control",
    "\u767c\u77ed\u7403": "Serve",
    "\u767c\u9577\u7403": "Serve",
    "\u9577\u7403": "Defensive",
    "\u6311\u7403": "Defensive",
    "\u9ede\u6263": "Attack",
    "\u6bba\u7403": "Attack",
    "\u5e73\u7403": "Attack",
    "\u5207\u7403": "Control",
    "\u7db2\u524d\u7403": "Control",
    "\u64cb\u5c0f\u7403": "Control",
    "\u653e\u5c0f\u7403": "Control",
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def forbid_ttmatch_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").upper()
    if "TTMATCH" in text:
        raise RuntimeError(f"TTMATCH is banned from clean V274 audit: {path}")


def allowed_source_root(path: Path) -> str | None:
    try:
        parts = path.relative_to(EXTERNAL).parts
    except ValueError:
        return None
    if not parts:
        return None
    top = parts[0]
    top_lower = top.lower()
    if top.upper() == "TTMATCH":
        return None
    if top_lower.startswith("opentt"):
        return "OpenTT"
    if top_lower.startswith("coachai"):
        if any("shuttleset22" in p.lower() for p in parts):
            return "ShuttleSet22"
        if any(p.lower() == "shuttleset" for p in parts):
            return "ShuttleSet"
        return "CoachAI"
    if top_lower.startswith("shuttleset"):
        return "ShuttleSet"
    if top == "TT-MatchDynamics":
        return "TT-MatchDynamics"
    return None


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def empty_canonical() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def phase_from_round(values: pd.Series) -> pd.Series:
    n = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [n.le(1), n.eq(2), n.eq(3), n.eq(4)],
            ["serve_like", "receive_like", "third_ball_like", "fourth_ball_like"],
            default="rally_like",
        ),
        index=values.index,
    )


def phase_from_event_index(values: pd.Series, terminal: pd.Series) -> pd.Series:
    n = pd.to_numeric(values, errors="coerce")
    phase = pd.Series(
        np.select(
            [n.le(0), n.eq(1), n.eq(2), n.eq(3)],
            ["serve_like", "receive_like", "third_ball_like", "fourth_ball_like"],
            default="rally_like",
        ),
        index=values.index,
    )
    return phase.mask(terminal.fillna(False).astype(bool), "terminal_like")


def family_from_opentt(value: Any, event_type: Any) -> str | None:
    fam = clean_text(value).lower()
    event = clean_text(event_type).lower()
    if "serve" in fam:
        return "Serve"
    if "attack" in fam:
        return "Attack"
    if "defensive" in fam:
        return "Defensive"
    if "control" in fam:
        return "Control"
    if event in {"net", "rally_ending"} or "ending" in event:
        return "Zero"
    if event == "bounce":
        return "Control"
    return None


def family_from_badminton(value: Any) -> str | None:
    key = clean_text(value).lower()
    if not key:
        return None
    return BADMINTON_FAMILY_MAP.get(key)


def depth_from_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = numeric.notna()
    if valid.sum() == 0:
        return out
    ranks = numeric[valid].rank(method="first", pct=True)
    out.loc[valid & ranks.le(1.0 / 3.0)] = "short"
    out.loc[valid & ranks.gt(1.0 / 3.0) & ranks.le(2.0 / 3.0)] = "mid"
    out.loc[valid & ranks.gt(2.0 / 3.0)] = "long"
    return out


def width_from_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = numeric.notna()
    if valid.sum() == 0:
        return out
    ranks = numeric[valid].rank(method="first", pct=True)
    out.loc[valid & ranks.le(1.0 / 3.0)] = "wide_left"
    out.loc[valid & ranks.gt(1.0 / 3.0) & ranks.le(2.0 / 3.0)] = "center"
    out.loc[valid & ranks.gt(2.0 / 3.0)] = "wide_right"
    return out


def terminal_from_columns(df: pd.DataFrame) -> pd.Series:
    terminal = pd.Series(False, index=df.index)
    for col in ["lose_reason", "win_reason", "getpoint_player", "flaw"]:
        if col in df.columns:
            text = df[col].astype("string").str.strip()
            terminal = terminal | (text.notna() & text.ne(""))
    if {"ball_round", "rally_length"}.issubset(df.columns):
        terminal = terminal | pd.to_numeric(df["ball_round"], errors="coerce").eq(
            pd.to_numeric(df["rally_length"], errors="coerce")
        )
    return pd.Series(np.where(terminal, "terminal", "remaining"), index=df.index)


def canonical_from_opentt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    terminal_bool = (
        pd.to_numeric(df.get("is_rally_ending", 0), errors="coerce").fillna(0).astype(int).eq(1)
        | pd.to_numeric(df.get("is_net", 0), errors="coerce").fillna(0).astype(int).eq(1)
    )
    event_index = df.groupby(["split", "video_id"], dropna=False).cumcount()
    out = pd.DataFrame(index=df.index)
    out["source"] = "OpenTT"
    out["source_path"] = rel(path)
    out["sequence_id"] = (
        df.get("split", "").astype(str) + ":" + df.get("video_id", "").astype(str)
    )
    out["event_index"] = event_index.astype(int)
    out["source_row"] = np.arange(len(df), dtype=int)
    out["action_family"] = [
        family_from_opentt(f, e) for f, e in zip(df.get("safe_action_family", ""), df.get("event_type", ""))
    ]
    out["phase"] = phase_from_event_index(event_index, terminal_bool)
    out["landing_depth"] = pd.NA
    out["landing_width_or_side"] = df.get("player_side", pd.Series(pd.NA, index=df.index)).replace("", pd.NA)
    out["terminal_or_remaining"] = np.where(terminal_bool, "terminal", "remaining")
    out["raw_label"] = df.get("event_raw_label", "").astype(str)
    return out[CANONICAL_COLUMNS]


def canonical_from_coachai_csv(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "type" not in df.columns or "ball_round" not in df.columns:
        return empty_canonical()
    out = pd.DataFrame(index=df.index)
    out["source"] = source
    out["source_path"] = rel(path)
    seq_cols = [c for c in ["match_id", "set", "rally_id", "rally"] if c in df.columns]
    if seq_cols:
        seq = df[seq_cols].astype("string").fillna("").agg(":".join, axis=1)
    else:
        seq = pd.Series(path.stem, index=df.index)
    out["sequence_id"] = seq
    out["event_index"] = pd.to_numeric(df["ball_round"], errors="coerce").fillna(0).astype(int) - 1
    out["source_row"] = np.arange(len(df), dtype=int)
    out["action_family"] = df["type"].map(family_from_badminton)
    out["phase"] = phase_from_round(df["ball_round"])
    out["landing_depth"] = depth_from_numeric(df["landing_y"]) if "landing_y" in df.columns else pd.NA
    out["landing_width_or_side"] = width_from_numeric(df["landing_x"]) if "landing_x" in df.columns else pd.NA
    out["terminal_or_remaining"] = terminal_from_columns(df)
    out["raw_label"] = df["type"].astype(str)
    return out[CANONICAL_COLUMNS]


def canonical_from_matchdynamics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Topspin/Backspin Indicator", "Forehand/Backhand Indicator", "Winning in First Three Strokes", "X", "Y"}
    if not required.issubset(df.columns):
        return empty_canonical()
    top = pd.to_numeric(df["Topspin/Backspin Indicator"], errors="coerce").fillna(0).astype(int)
    early_win = pd.to_numeric(df["Winning in First Three Strokes"], errors="coerce").fillna(0).astype(int)
    out = pd.DataFrame(index=df.index)
    out["source"] = "TT-MatchDynamics"
    out["source_path"] = rel(path)
    out["sequence_id"] = df.get("date", pd.Series("tt_matchdynamics", index=df.index)).astype(str)
    out["event_index"] = np.arange(len(df), dtype=int)
    out["source_row"] = np.arange(len(df), dtype=int)
    out["action_family"] = np.where(top.eq(1), "Attack", "Control")
    out["phase"] = np.where(early_win.eq(1), "third_ball_like", "rally_like")
    out["landing_depth"] = depth_from_numeric(df["Y"])
    out["landing_width_or_side"] = width_from_numeric(df["X"])
    out["terminal_or_remaining"] = np.where(early_win.eq(1), "terminal", "remaining")
    out["raw_label"] = (
        "topspin="
        + df["Topspin/Backspin Indicator"].astype(str)
        + ";fhbh="
        + df["Forehand/Backhand Indicator"].astype(str)
    )
    return out[CANONICAL_COLUMNS]


def is_track2_data_file(path: Path) -> bool:
    text = path.as_posix().lower()
    return "track 2_ stroke forecasting/data/" in text and path.suffix.lower() == ".csv"


def is_shuttleset_file(path: Path) -> bool:
    text = path.as_posix().lower()
    return "shuttleset" in text and path.name.lower().startswith("set") and path.suffix.lower() == ".csv"


def convert_file(path: Path, source: str) -> tuple[pd.DataFrame, str]:
    forbid_ttmatch_path(path)
    try:
        if source == "OpenTT" and path.as_posix().endswith("processed/openttgames_events.csv"):
            return canonical_from_opentt(path), "converted_opentt_events"
        if source in {"CoachAI", "ShuttleSet", "ShuttleSet22"} and path.suffix.lower() == ".csv":
            canonical = canonical_from_coachai_csv(path, source)
            if canonical.empty:
                return canonical, "not_a_known_sequence_schema"
            return canonical, "converted_coachai_strokes"
        if source == "TT-MatchDynamics" and path.name == "table_tennis_data.csv":
            return canonical_from_matchdynamics(path), "converted_matchdynamics"
    except Exception as exc:
        return empty_canonical(), f"conversion_error: {type(exc).__name__}: {exc}"
    return empty_canonical(), "not_a_known_sequence_schema"


def discover_allowed_files() -> list[tuple[str, Path]]:
    if not EXTERNAL.exists():
        return []
    files: list[tuple[str, Path]] = []
    for path in sorted(EXTERNAL.rglob("*")):
        if not path.is_file():
            continue
        source = allowed_source_root(path)
        if source is None:
            continue
        forbid_ttmatch_path(path)
        if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue
        files.append((source, path))
    return files


def inventory_row(source: str, path: Path, canonical: pd.DataFrame, status: str) -> dict[str, Any]:
    rows: int | float = np.nan
    columns = ""
    if path.suffix.lower() == ".csv":
        try:
            header = pd.read_csv(path, nrows=0)
            columns = "|".join(map(str, header.columns))
            rows = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
        except Exception as exc:
            status = f"{status}; inventory_error: {type(exc).__name__}: {exc}"
    elif path.suffix.lower() in {".json", ".jsonl"}:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = len(data) if hasattr(data, "__len__") else np.nan
            columns = "json"
        except Exception:
            columns = "json"

    usable = int(canonical[CANONICAL_FIELDS].notna().all(axis=1).sum()) if not canonical.empty else 0
    rates: dict[str, float] = {}
    for field in CANONICAL_FIELDS:
        rates[field] = float(canonical[field].isna().mean()) if not canonical.empty else 1.0
    return {
        "source": source,
        "path": rel(path),
        "file_type": path.suffix.lower().lstrip("."),
        "bytes": int(path.stat().st_size),
        "rows": rows,
        "columns": columns,
        "usable_rows": usable,
        "missing_action_family_rate": rates["action_family"],
        "missing_phase_rate": rates["phase"],
        "missing_landing_depth_rate": rates["landing_depth"],
        "missing_landing_width_or_side_rate": rates["landing_width_or_side"],
        "missing_terminal_or_remaining_rate": rates["terminal_or_remaining"],
        "status": status,
        "notes": "coarse canonical only; no AICUP exact actionId/pointId mapping",
    }


def build_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_records: list[dict[str, Any]] = []
    canonical_parts: list[pd.DataFrame] = []
    for source, path in discover_allowed_files():
        canonical, status = convert_file(path, source)
        if not canonical.empty:
            canonical_parts.append(canonical)
        inventory_records.append(inventory_row(source, path, canonical, status))
    inventory = pd.DataFrame(inventory_records, columns=INVENTORY_COLUMNS)
    canonical = pd.concat(canonical_parts, ignore_index=True) if canonical_parts else empty_canonical()
    return inventory, canonical[CANONICAL_COLUMNS]


def write_report(inventory: pd.DataFrame, canonical: pd.DataFrame) -> None:
    total_usable = int(inventory["usable_rows"].sum()) if not inventory.empty else 0
    source_summary = (
        inventory.groupby("source", dropna=False)
        .agg(
            files=("path", "size"),
            converted_files=("status", lambda s: int(s.astype(str).str.startswith("converted").sum())),
            rows=("rows", "sum"),
            usable_rows=("usable_rows", "sum"),
        )
        .reset_index()
        .sort_values(["usable_rows", "source"], ascending=[False, True])
        if not inventory.empty
        else pd.DataFrame(columns=["source", "files", "converted_files", "rows", "usable_rows"])
    )
    missing = {
        field: (float(canonical[field].isna().mean()) if not canonical.empty else 1.0)
        for field in CANONICAL_FIELDS
    }
    sources_ge_10k = source_summary[source_summary["usable_rows"].fillna(0).astype(float).ge(10000)]
    verdict = "V277_PRETRAINING_JUSTIFIED" if len(sources_ge_10k) > 0 else "AUDIT_ONLY_INSUFFICIENT_USABLE_ROWS"

    lines = [
        "# V274 Clean External Representation Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Allowed files inventoried: `{len(inventory)}`",
        f"- Canonical converted rows: `{len(canonical)}`",
        f"- Usable canonical rows: `{total_usable}`",
        "- Excluded root: `external_data/TTMATCH`",
        "- Submission files written: `0`",
        "",
        "External labels are converted only to coarse representation targets: "
        "`action_family`, `phase`, `landing_depth`, `landing_width_or_side`, "
        "`terminal_or_remaining`. They are not mapped to AICUP exact `actionId` "
        "or `pointId` labels.",
        "",
        "## Source Inventory",
        "",
    ]
    if source_summary.empty:
        lines.append("No allowed clean external source files were found.")
    else:
        for row in source_summary.itertuples(index=False):
            lines.append(
                f"- `{row.source}`: files={int(row.files)}, converted_files={int(row.converted_files)}, "
                f"rows={int(row.rows) if pd.notna(row.rows) else 0}, usable_rows={int(row.usable_rows)}"
            )
    lines.extend(["", "## Canonical Missing Rates", ""])
    for field, rate in missing.items():
        lines.append(f"- `{field}`: {rate:.4f}")
    lines.extend(["", "## Notes", ""])
    lines.append("- `TT-MatchDynamics` is treated as distinct from banned `external_data/TTMATCH`.")
    lines.append("- Files with unknown schemas remain in the inventory but contribute zero usable rows.")
    lines.append("- V274 prepares audit/cache artifacts only; actual representation pretraining belongs in V277.")
    (OUTDIR / "v274_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(inventory: pd.DataFrame, canonical: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTDIR / "v274_external_inventory.csv", index=False)
    sample = canonical.groupby("source", group_keys=False).head(2500).reset_index(drop=True) if not canonical.empty else canonical
    sample.to_csv(OUTDIR / "v274_canonical_samples.csv", index=False)
    write_report(inventory, canonical)


def main() -> None:
    inventory, canonical = build_audit()
    write_outputs(inventory, canonical)
    payload = {
        "outdir": rel(OUTDIR),
        "inventory_files": int(len(inventory)),
        "canonical_rows": int(len(canonical)),
        "usable_rows": int(inventory["usable_rows"].sum()) if not inventory.empty else 0,
        "sources": sorted(inventory["source"].dropna().unique().tolist()) if not inventory.empty else [],
        "ttmatch_rows": 0,
    }
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
