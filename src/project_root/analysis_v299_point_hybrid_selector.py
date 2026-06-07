"""V299 conservative point hybrid selector.

Combines V295/V297/V298 changed-row audits as row-level point votes and emits
low-churn point-only variants over the clean V261 anchor.  Action/server are
copied from the anchor; TTMATCH and old-server artifacts are not read.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v299_point_hybrid_selector"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v299_point_hybrid_selector.py"
ANCHOR_PATH = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
EXPECTED_ROWS = 1845
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
LONG = {7, 8, 9}

SOURCE_CONFIG = {
    "v295": {
        "audit": ROOT / "v295_true_oof_point_specialists" / "v295_changed_row_audit.csv",
        "search": ROOT / "v295_true_oof_point_specialists" / "v295_candidate_search.csv",
    },
    "v297": {
        "audit": ROOT / "v297_multisource_point_agreement" / "v297_changed_row_audit.csv",
        "search": ROOT / "v297_multisource_point_agreement" / "v297_candidate_search.csv",
    },
    "v298": {
        "audit": ROOT / "v298_action_point_support_prior" / "v298_changed_row_audit.csv",
        "search": ROOT / "v298_action_point_support_prior" / "v298_candidate_search.csv",
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_id",
            "candidate_point",
            "base_point",
            "score",
            "source_agreement_count",
            "sources",
            "source_candidate_names",
        ]
    )


def _is_long_vote(row: pd.Series) -> bool:
    mode = str(row.get("mode", "")).lower()
    specialist = str(row.get("specialist", "")).lower()
    reason = str(row.get("reason", "")).lower()
    candidate = str(row.get("candidate", "")).lower()
    return "long789" in mode or "long789" in specialist or "long789" in reason or "long789" in candidate


def _prepare_votes(audits: dict[str, pd.DataFrame], variant: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, audit in audits.items():
        if audit.empty:
            continue
        frame = audit.copy()
        if "row_id" not in frame.columns or "candidate_point" not in frame.columns:
            raise ValueError(f"{source} audit missing row_id/candidate_point")
        frame["source"] = source
        frame["row_id"] = pd.to_numeric(frame["row_id"], errors="coerce").astype("Int64")
        frame["candidate_point"] = pd.to_numeric(frame["candidate_point"], errors="coerce").astype("Int64")
        frame["score"] = pd.to_numeric(frame.get("score", 0.0), errors="coerce").fillna(0.0).astype(float)
        if "base_point" in frame.columns:
            frame["base_point"] = pd.to_numeric(frame["base_point"], errors="coerce").astype("Int64")
        else:
            frame["base_point"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
        if "candidate" not in frame.columns:
            frame["candidate"] = ""
        frame = frame.dropna(subset=["row_id", "candidate_point"])
        frame["row_id"] = frame["row_id"].astype(int)
        frame["candidate_point"] = frame["candidate_point"].astype(int)

        if variant == "no_point0":
            frame = frame[frame["candidate_point"] != 0]
        elif variant == "long789_only":
            is_long = frame.apply(_is_long_vote, axis=1)
            frame = frame[is_long & frame["source"].isin(["v295", "v297", "v298"])]
            frame = frame[frame["candidate_point"].isin(sorted(LONG))]
            with_base = frame["base_point"].notna()
            frame = frame[~with_base | frame["base_point"].astype(int).isin(sorted(LONG))]
        elif variant == "support_plus_agreement":
            pass
        elif variant != "agreement_2sources":
            raise ValueError(f"unknown variant kind: {variant}")
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    votes = pd.concat(frames, ignore_index=True)
    if votes.empty:
        return votes
    # Count a method once per row/candidate even if several source variants
    # wrote the same changed row into its audit.
    grouped = (
        votes.groupby(["source", "row_id", "candidate_point"], as_index=False)
        .agg(
            score=("score", "max"),
            base_point=("base_point", "first"),
            source_candidate_names=("candidate", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
        )
    )
    return grouped


def build_variant_candidates(audits: dict[str, pd.DataFrame], variant: str) -> pd.DataFrame:
    votes = _prepare_votes(audits, variant)
    if votes.empty:
        return _empty_candidates()

    rows: list[dict[str, Any]] = []
    for (row_id, point), group in votes.groupby(["row_id", "candidate_point"]):
        sources = sorted(group["source"].astype(str).unique().tolist())
        if variant == "support_plus_agreement":
            if "v298" not in sources or len([s for s in sources if s != "v298"]) < 1:
                continue
        elif len(sources) < 2:
            continue
        source_names = []
        for rec in group.sort_values("source").to_dict("records"):
            names = str(rec.get("source_candidate_names", ""))
            if names:
                source_names.append(f"{rec['source']}:{names}")
        base_values = group["base_point"].dropna()
        rows.append(
            {
                "row_id": int(row_id),
                "candidate_point": int(point),
                "base_point": int(base_values.iloc[0]) if not base_values.empty else pd.NA,
                "score": float(len(sources) + group["score"].astype(float).mean()),
                "source_agreement_count": int(len(sources)),
                "sources": "+".join(sources),
                "source_candidate_names": ";".join(source_names),
            }
        )
    if not rows:
        return _empty_candidates()
    return pd.DataFrame(rows).sort_values(
        ["source_agreement_count", "score", "row_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def apply_candidates(base: np.ndarray, candidates: pd.DataFrame, cap: float) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.asarray(base, dtype=int).copy()
    if candidates.empty or cap <= 0.0:
        return pred, candidates.head(0).copy()
    max_rows = len(pred) if cap >= 1.0 else int(math.floor(len(pred) * float(cap)))
    if max_rows <= 0:
        return pred, candidates.head(0).copy()
    ranked = candidates.copy()
    row_ids_all = ranked["row_id"].astype(int).to_numpy()
    if (row_ids_all < 0).any() or (row_ids_all >= len(pred)).any():
        raise ValueError("candidate row_id out of range")
    ranked = ranked[ranked["candidate_point"].astype(int).to_numpy() != pred[row_ids_all]]
    if ranked.empty:
        return pred, ranked.copy()
    ranked = (
        ranked.sort_values(["source_agreement_count", "score", "row_id"], ascending=[False, False, True])
        .drop_duplicates("row_id", keep="first")
        .head(max_rows)
        .copy()
    )
    row_ids = ranked["row_id"].astype(int).to_numpy()
    pred[row_ids] = ranked["candidate_point"].astype(int).to_numpy()
    return pred, ranked


def write_submission(path: Path, point_pred: np.ndarray, anchor: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> None:
    out = anchor.copy()
    out["pointId"] = np.asarray(point_pred, dtype=int)
    out = out.loc[:, SUBMISSION_COLUMNS]
    if len(out) != expected_rows:
        raise ValueError(f"bad submission row count: {len(out)}")
    if not out["pointId"].between(0, 9).all():
        raise ValueError("pointId out of range")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_audits() -> dict[str, pd.DataFrame]:
    audits: dict[str, pd.DataFrame] = {}
    for source, config in SOURCE_CONFIG.items():
        path = config["audit"]
        if not path.exists():
            raise FileNotFoundError(path)
        audits[source] = pd.read_csv(path)
    return audits


def load_source_search() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, config in SOURCE_CONFIG.items():
        path = config["search"]
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "candidate" not in frame.columns:
            continue
        delta_col = next(
            (col for col in ["delta_vs_v294_base", "delta_vs_aligned_base", "public_like_delta"] if col in frame.columns),
            None,
        )
        if delta_col is None:
            frame["local_delta"] = np.nan
        else:
            frame["local_delta"] = pd.to_numeric(frame[delta_col], errors="coerce")
        frame["source"] = source
        frames.append(frame[["source", "candidate", "local_delta"]].copy())
    if not frames:
        return pd.DataFrame(columns=["source", "candidate", "local_delta"])
    return pd.concat(frames, ignore_index=True)


def selected_source_delta(selected: pd.DataFrame, source_search: pd.DataFrame) -> float | None:
    if selected.empty or source_search.empty:
        return None
    names: set[str] = set()
    for value in selected.get("source_candidate_names", pd.Series(dtype=str)).astype(str):
        for block in value.split(";"):
            if ":" not in block:
                continue
            _source, candidates = block.split(":", 1)
            names.update(name for name in candidates.split("|") if name)
    if not names:
        return None
    deltas = source_search[source_search["candidate"].isin(names)]["local_delta"].dropna()
    if deltas.empty:
        return None
    return float(deltas.max())


def evaluate_variant(
    name: str,
    kind: str,
    cap: float,
    audits: dict[str, pd.DataFrame],
    anchor: pd.DataFrame,
    source_search: pd.DataFrame,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    base = anchor["pointId"].astype(int).to_numpy()
    candidates = build_variant_candidates(audits, kind)
    pred, selected = apply_candidates(base, candidates, cap)
    point0_delta = float(np.mean(pred == 0) - np.mean(base == 0))
    source_delta = selected_source_delta(selected, source_search)
    recommendation = "DO_NOT_UPLOAD"
    if source_delta is not None and source_delta >= 0.0015 and point0_delta <= 0.0:
        recommendation = "REVIEW_UPLOAD"
    rec = {
        "candidate": name,
        "variant": kind,
        "cap": cap,
        "test_changed_rows": int(len(selected)),
        "point_churn": float(len(selected) / len(base)),
        "point0_rate_delta": point0_delta,
        "source_agreement_count": int(selected["source_agreement_count"].min()) if not selected.empty else 0,
        "source_agreement_count_mean": float(selected["source_agreement_count"].mean()) if not selected.empty else 0.0,
        "available_source_local_delta": source_delta,
        "upload_recommendation": recommendation,
    }
    return rec, pred, selected


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR_PATH)
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"bad anchor columns: {list(anchor.columns)}")
    if len(anchor) != EXPECTED_ROWS:
        raise ValueError(f"bad anchor row count: {len(anchor)}")

    audits = load_audits()
    source_search = load_source_search()
    variants = [
        ("v299_agreement_2sources_cap0p0025", "agreement_2sources", 0.0025),
        ("v299_agreement_2sources_cap0p005", "agreement_2sources", 0.005),
        ("v299_no_point0_cap0p005", "no_point0", 0.005),
        ("v299_long789_only_cap0p005", "long789_only", 0.005),
        ("v299_support_plus_agreement_cap0p005", "support_plus_agreement", 0.005),
    ]

    records: list[dict[str, Any]] = []
    audits_out: list[pd.DataFrame] = []
    generated: list[str] = []
    for name, kind, cap in variants:
        rec, pred, selected = evaluate_variant(name, kind, cap, audits, anchor, source_search)
        filename = f"submission_{name}__v173action_r121server.csv"
        path = OUT_DIR / filename
        write_submission(path, pred, anchor)
        rec["path"] = str(path.relative_to(ROOT))
        records.append(rec)
        generated.append(str(path.relative_to(ROOT)))
        if not selected.empty:
            audit = selected.copy()
            row_ids = audit["row_id"].astype(int).to_numpy()
            audit["candidate"] = name
            audit["rally_uid"] = anchor.iloc[row_ids]["rally_uid"].to_numpy()
            audit["anchor_point"] = anchor.iloc[row_ids]["pointId"].astype(int).to_numpy()
            audit["new_point"] = audit["candidate_point"].astype(int).to_numpy()
            audits_out.append(audit)

    search = pd.DataFrame(records).sort_values(
        ["upload_recommendation", "available_source_local_delta", "test_changed_rows"],
        ascending=[False, False, True],
        na_position="last",
    )
    search.to_csv(OUT_DIR / "v299_candidate_search.csv", index=False)
    if audits_out:
        changed = pd.concat(audits_out, ignore_index=True)
    else:
        changed = _empty_candidates()
    changed.to_csv(OUT_DIR / "v299_changed_row_audit.csv", index=False)

    best = search.iloc[0].to_dict()
    upload_recommendation = (
        "REVIEW_UPLOAD" if search["upload_recommendation"].eq("REVIEW_UPLOAD").any() else "DO_NOT_UPLOAD"
    )
    report = _json_safe(
        {
            "version": "V299",
            "anchor_submission": str(ANCHOR_PATH.relative_to(ROOT)),
            "source_audits": {source: str(config["audit"].relative_to(ROOT)) for source, config in SOURCE_CONFIG.items()},
            "source_search_tables": {
                source: str(config["search"].relative_to(ROOT)) for source, config in SOURCE_CONFIG.items()
            },
            "generated_submissions": generated,
            "generated_submission_count": len(generated),
            "best_candidate": best,
            "upload_recommendation": upload_recommendation,
            "upload_gate": "DO_NOT_UPLOAD unless selected source-table local delta >= +0.0015 and point0_rate_delta <= 0.",
            "fixed_output": {
                "actionId": "copied exactly from V261 anchor",
                "serverGetPoint": "copied exactly from V261 anchor",
                "pointId": "V299 conservative hybrid selector",
            },
            "clean_rules": {
                "no_ttmatch": True,
                "no_old_server": True,
                "do_not_copy_to_upload_candidates": True,
            },
        }
    )
    (OUT_DIR / "v299_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# V299 point hybrid selector",
        "",
        f"Anchor: `{ANCHOR_PATH.relative_to(ROOT)}`",
        "Action/server fixed from anchor. TTMATCH/old-server not used.",
        f"Generated submissions: `{len(generated)}`",
        f"Upload recommendation: `{upload_recommendation}`",
        "",
        "## Candidate search",
        "",
    ]
    for row in search.to_dict("records"):
        local_delta = row.get("available_source_local_delta")
        local_delta_text = "NA" if pd.isna(local_delta) else f"{float(local_delta):.6f}"
        lines.append(
            f"- `{row['candidate']}`: changed={int(row['test_changed_rows'])}, "
            f"churn={float(row['point_churn']):.6f}, "
            f"point0_delta={float(row['point0_rate_delta']):.6f}, "
            f"agreement={int(row['source_agreement_count'])}, "
            f"source_delta={local_delta_text}, "
            f"recommendation=`{row['upload_recommendation']}`"
        )
    (OUT_DIR / "v299_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), SRC_DEST)
    return report


def main() -> None:
    report = run_pipeline()
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "outdir": str(OUT_DIR.relative_to(ROOT)),
                "best_candidate": best.get("candidate", ""),
                "best_test_changed_rows": best.get("test_changed_rows", 0),
                "best_point_churn": best.get("point_churn", 0.0),
                "best_point0_rate_delta": best.get("point0_rate_delta", 0.0),
                "generated_submissions": report["generated_submission_count"],
                "upload_recommendation": report["upload_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
