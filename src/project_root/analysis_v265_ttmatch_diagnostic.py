from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v265_ttmatch_diagnostic"
UPLOAD_DIR = ROOT / "upload_candidates_20260519"

TTMATCH_TRAIN = ROOT / "external_data" / "TTMATCH" / "train.csv"
AICUP_TEST_NEW = ROOT / "test_new.csv"
FALLBACK_CLEAN = ROOT / "v261_action_conditioned_point_residual" / "submission_v261_cap0p01__v173action_r121server.csv"
OLDSHARPEN = ROOT / "v249_current_anchor_server_diagnostic" / "submission_v249_v173_v188cap5_oldsharpen005095.csv"
R178_REPORT = ROOT / "r178_ttmatch_overlap_audit" / "r178_report.md"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
STRICT_COLUMNS = [
    "strikeNumber",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]
NOSTRIKE_COLUMNS = [c for c in STRICT_COLUMNS if c != "strikeNumber"]


def normalize_ttmatch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with TTMATCH typo columns normalized to AICUP names."""
    out = df.copy()
    rename = {}
    if "strickNumber" in out.columns and "strikeNumber" not in out.columns:
        rename["strickNumber"] = "strikeNumber"
    if "strickId" in out.columns and "strikeId" not in out.columns:
        rename["strickId"] = "strikeId"
    if rename:
        out = out.rename(columns=rename)
    return out


def sequence_key(group: pd.DataFrame, include_strike_number: bool = True) -> tuple[tuple[int, ...], ...]:
    """Build a hashable stroke-prefix signature from observed rows."""
    df = normalize_ttmatch_columns(group)
    columns = STRICT_COLUMNS if include_strike_number else NOSTRIKE_COLUMNS
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing sequence key columns: {missing}")
    values = []
    for _, row in df[columns].iterrows():
        values.append(tuple(_to_int(row[c]) for c in columns))
    return tuple(values)


def majority_vote(values) -> int:
    """Vote with deterministic smallest-value tie break."""
    counts = defaultdict(int)
    for value in values:
        counts[_to_int(value)] += 1
    if not counts:
        raise ValueError("majority_vote requires at least one value")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_next_stroke_lookup(ttmatch_train: pd.DataFrame, include_strike_number: bool = True) -> dict:
    """Map every TTMATCH observed prefix to the majority next-stroke target."""
    df = normalize_ttmatch_columns(ttmatch_train)
    _require_columns(df, ["rally_uid", "actionId", "pointId", "serverGetPoint"] + NOSTRIKE_COLUMNS)
    if include_strike_number:
        _require_columns(df, ["strikeNumber"])

    buckets = defaultdict(lambda: {"actionId": [], "pointId": [], "serverGetPoint": []})
    sort_cols = ["rally_uid", "strikeNumber"] if "strikeNumber" in df.columns else ["rally_uid"]
    for _, group in df.sort_values(sort_cols).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        for end in range(1, len(group)):
            key = sequence_key(group.iloc[:end], include_strike_number=include_strike_number)
            nxt = group.iloc[end]
            buckets[key]["actionId"].append(nxt["actionId"])
            buckets[key]["pointId"].append(nxt["pointId"])
            buckets[key]["serverGetPoint"].append(nxt["serverGetPoint"])

    lookup = {}
    for key, bucket in buckets.items():
        lookup[key] = {
            "actionId": majority_vote(bucket["actionId"]),
            "pointId": majority_vote(bucket["pointId"]),
            "serverGetPoint": majority_vote(bucket["serverGetPoint"]),
            "support": len(bucket["actionId"]),
        }
    return lookup


def apply_ttmatch_predictions(
    test_new: pd.DataFrame,
    strict_lookup: dict,
    nostrike_lookup: dict,
    fallback_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply strict TTMATCH prefix matches first, nostrike matches second, else fallback."""
    test_df = normalize_ttmatch_columns(test_new)
    fallback = fallback_submission[SUBMISSION_COLUMNS].copy()
    fallback_by_uid = fallback.set_index("rally_uid")

    predictions = []
    coverage = []
    for rally_uid, group in test_df.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=True):
        if rally_uid not in fallback_by_uid.index:
            raise KeyError(f"Fallback submission missing rally_uid={rally_uid}")
        base = fallback_by_uid.loc[rally_uid]
        strict_key = sequence_key(group, include_strike_number=True)
        nostrike_key = sequence_key(group, include_strike_number=False)
        match = strict_lookup.get(strict_key)
        match_type = "strict"
        if match is None:
            match = nostrike_lookup.get(nostrike_key)
            match_type = "nostrike"
        if match is None:
            match_type = "none"
            row = {
                "rally_uid": rally_uid,
                "actionId": _to_int(base["actionId"]),
                "pointId": _to_int(base["pointId"]),
                "serverGetPoint": float(base["serverGetPoint"]),
            }
            support = 0
        else:
            row = {
                "rally_uid": rally_uid,
                "actionId": _to_int(match["actionId"]),
                "pointId": _to_int(match["pointId"]),
                "serverGetPoint": float(match["serverGetPoint"]),
            }
            support = _to_int(match["support"])
        predictions.append(row)
        coverage.append(
            {
                "rally_uid": rally_uid,
                "match_type": match_type,
                "support": support,
                "pred_actionId": row["actionId"],
                "pred_pointId": row["pointId"],
                "pred_serverGetPoint": row["serverGetPoint"],
                "fallback_actionId": _to_int(base["actionId"]),
                "fallback_pointId": _to_int(base["pointId"]),
                "fallback_serverGetPoint": float(base["serverGetPoint"]),
                "prefix_len": len(group),
            }
        )

    pred_df = pd.DataFrame(predictions)[SUBMISSION_COLUMNS]
    coverage_df = pd.DataFrame(coverage)
    return pred_df, coverage_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ttmatch_train = pd.read_csv(TTMATCH_TRAIN)
    test_new = pd.read_csv(AICUP_TEST_NEW)
    clean = _load_submission(FALLBACK_CLEAN)
    oldsharpen = _load_submission(OLDSHARPEN)

    strict_lookup = build_next_stroke_lookup(ttmatch_train, include_strike_number=True)
    nostrike_lookup = build_next_stroke_lookup(ttmatch_train, include_strike_number=False)
    tt_pred, coverage = apply_ttmatch_predictions(test_new, strict_lookup, nostrike_lookup, clean)

    coverage_path = OUT_DIR / "v265_match_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    candidates = {}
    candidates["submission_v265_ttmatch_action_only__v261point_r121.csv"] = _compose_candidate(
        clean, tt_pred, replace_action=True, replace_point=False, server_source=clean
    )
    candidates["submission_v265_ttmatch_point_only__v173action_r121.csv"] = _compose_candidate(
        clean, tt_pred, replace_action=False, replace_point=True, server_source=clean
    )
    candidates["submission_v265_ttmatch_action_point__r121.csv"] = _compose_candidate(
        clean, tt_pred, replace_action=True, replace_point=True, server_source=clean
    )
    candidates["submission_v265_ttmatch_action_point__oldsharpen005095.csv"] = _compose_candidate(
        clean, tt_pred, replace_action=True, replace_point=True, server_source=oldsharpen
    )

    rows = []
    for name, candidate in candidates.items():
        _validate_submission(candidate, expected_rows=len(clean))
        out_path = OUT_DIR / name
        candidate.to_csv(out_path, index=False)
        if UPLOAD_DIR.exists():
            shutil.copy2(out_path, UPLOAD_DIR / name)
        rows.append(_candidate_metrics(name, candidate, clean, coverage))

    search = pd.DataFrame(rows)
    search_path = OUT_DIR / "v265_candidate_search.csv"
    search.to_csv(search_path, index=False)
    _write_report(search, coverage, strict_lookup, nostrike_lookup)


def _compose_candidate(
    clean: pd.DataFrame,
    tt_pred: pd.DataFrame,
    replace_action: bool,
    replace_point: bool,
    server_source: pd.DataFrame,
) -> pd.DataFrame:
    base = clean.set_index("rally_uid").copy()
    pred = tt_pred.set_index("rally_uid")
    server = server_source.set_index("rally_uid")
    if replace_action:
        base.loc[pred.index, "actionId"] = pred["actionId"].astype(int)
    if replace_point:
        base.loc[pred.index, "pointId"] = pred["pointId"].astype(int)
    base.loc[server.index, "serverGetPoint"] = server["serverGetPoint"].astype(float)
    return base.reset_index()[SUBMISSION_COLUMNS].sort_values("rally_uid").reset_index(drop=True)


def _candidate_metrics(name: str, candidate: pd.DataFrame, clean: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    merged = candidate.merge(clean, on="rally_uid", suffixes=("", "_clean"))
    action_changed = int((merged["actionId"] != merged["actionId_clean"]).sum())
    point_changed = int((merged["pointId"] != merged["pointId_clean"]).sum())
    server_mad = float((merged["serverGetPoint"] - merged["serverGetPoint_clean"]).abs().mean())
    matched = int((coverage["match_type"] != "none").sum())
    strict = int((coverage["match_type"] == "strict").sum())
    nostrike = int((coverage["match_type"] == "nostrike").sum())
    none = int((coverage["match_type"] == "none").sum())
    return {
        "candidate": name,
        "tier": "HIGH_RISK_DIAGNOSTIC",
        "rows": len(candidate),
        "matched_rows": matched,
        "strict_rows": strict,
        "nostrike_rows": nostrike,
        "no_match_rows": none,
        "match_rate": matched / len(coverage) if len(coverage) else 0.0,
        "action_changed_vs_clean": action_changed,
        "point_changed_vs_clean": point_changed,
        "server_mad_vs_clean": server_mad,
        "ttmatch_policy": "diagnostic_only_do_not_mix_with_clean_private_safe",
    }


def _write_report(search: pd.DataFrame, coverage: pd.DataFrame, strict_lookup: dict, nostrike_lookup: dict) -> None:
    counts = coverage["match_type"].value_counts().to_dict()
    r178_warning = R178_REPORT.read_text(encoding="utf-8", errors="ignore").splitlines()[2].strip()
    lines = [
        "# V265 TTMATCH Diagnostic",
        "",
        "Verdict: `HIGH_RISK_DIAGNOSTIC`",
        "",
        "This branch uses TTMATCH-derived prefix matches only as a diagnostic. It must not be mixed into clean/private-safe branches.",
        f"R178 reference: {r178_warning}",
        "",
        "## Lookup",
        f"- Strict lookup keys: {len(strict_lookup)}",
        f"- No-strike lookup keys: {len(nostrike_lookup)}",
        "",
        "## Coverage",
        f"- strict rows: {counts.get('strict', 0)}",
        f"- nostrike rows: {counts.get('nostrike', 0)}",
        f"- no-match rows: {counts.get('none', 0)}",
        f"- total rows: {len(coverage)}",
        "",
        "## Candidates",
        _dataframe_to_markdown(search),
        "",
        "## Caveat",
        "TTMATCH was previously audited as high-risk overlap data. These outputs are for public/ceiling diagnostics only.",
        "",
    ]
    (OUT_DIR / "v265_report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    _validate_submission(df)
    return df[SUBMISSION_COLUMNS].copy().sort_values("rally_uid").reset_index(drop=True)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.astype(object).itertuples(index=False, name=None)]

    def fmt(values):
        return "| " + " | ".join(str(value) for value in values) + " |"

    header = fmt(columns)
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [fmt(row) for row in rows]
    return "\n".join([header, sep] + body)


def _validate_submission(df: pd.DataFrame, expected_rows: int | None = 1845) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Unexpected submission columns: {list(df.columns)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if df["rally_uid"].duplicated().any():
        raise ValueError("Duplicate rally_uid in submission")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _to_int(value) -> int:
    if pd.isna(value):
        return -1
    return int(value)


if __name__ == "__main__":
    main()
