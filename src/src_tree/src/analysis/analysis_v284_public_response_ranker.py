from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v284_public_response_ranker"
R200_SUMMARY = ROOT / "r200_local_validation_dashboard" / "r200_candidate_summary.csv"
ANCHOR_PUBLIC = 0.3576720
ANCHOR_FILE = "submission_v261_cap0p01__v173action_r121server.csv"


KNOWN_PUBLIC_RESULTS = [
    {
        "candidate": "submission_v261_cap0p01__v173action_r121server.csv",
        "public_pl": 0.3576720,
        "note": "current clean best: V173 action + V261 cap1 point + R121 server",
    },
    {
        "candidate": "submission_v272_point_actioncond_cap0p010__v173action_r121server.csv",
        "public_pl": 0.3576159,
        "note": "action-conditioned point residual, slightly below current clean best",
    },
    {
        "candidate": "submission_v277_nonterminal_cap0p010__v173action_r121server.csv",
        "public_pl": 0.3574825,
        "note": "nonterminal point refinement, below current clean best",
    },
    {
        "candidate": "submission_v220_weakonly_churn0p005__pv188cap5__sr121.csv",
        "public_pl": 0.3542440,
        "note": "weak action repair failed public",
    },
    {
        "candidate": "submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv",
        "public_pl": 0.3509562,
        "note": "V166 full action replacement failed public",
    },
    {
        "candidate": "submission_v248_external_aug__pv188cap5__sr121.csv",
        "public_pl": 0.3554156,
        "note": "external augmentation action branch below clean best",
    },
    {
        "candidate": "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv",
        "public_pl": 0.3573932,
        "note": "previous clean point anchor before V261",
    },
]


def classify_family(candidate_name: str) -> str:
    name = Path(str(candidate_name)).name.lower()
    if "v273" in name:
        return "v273_action_style"
    if "v263" in name:
        return "v263_questionnaire_action"
    if "v267" in name:
        return "v267_longtail_action"
    if name.startswith("submission_v261") and "cap0p01" in name:
        return "v261_cap1_point"
    if "clean_v261cap1_r121" in name or "anchor_copy" in name:
        return "clean_anchor_copy"
    if "v272" in name:
        return "v272_point_actioncond"
    if "v277" in name:
        return "v277_point_refine"
    if "v191_v166" in name or "v166_best_action" in name:
        return "v191_v166_action"
    if "v220" in name:
        return "v220_action_repair"
    if "v248" in name:
        return "v248_action_aug"
    if "v280" in name:
        return "v280_joint_pair"
    if "v282" in name:
        return "v282_joint_support"
    if "v283" in name:
        return "v283_pair_selector"
    if "ttmatch" in name:
        return "ttmatch_diagnostic"
    if "oldhard" in name or "oldsharpen" in name or "oldrank" in name:
        return "old_server_diagnostic"
    if "server" in name and ("v266" in name or "v269" in name or "v270" in name or "v271" in name):
        return "clean_server_microblend"
    if "v188" in name and "cap0p05" in name:
        return "v188_cap5_point"
    return "unknown"


def family_public_delta(public_df: pd.DataFrame, anchor_public: float = ANCHOR_PUBLIC) -> dict[str, float]:
    public = public_df.copy()
    public["family"] = public["candidate"].map(classify_family)
    public["delta"] = public["public_pl"].astype(float) - float(anchor_public)
    grouped = public.groupby("family")["delta"].max()
    return {str(k): float(v) for k, v in grouped.items()}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(parsed):
        return default
    return parsed


def risk_penalty(
    action_churn: float,
    point_churn: float,
    server_mad: float,
    family_delta: float,
    decision: str,
) -> float:
    penalty = 0.0
    penalty += max(action_churn, 0.0) * 1.6
    penalty += max(point_churn, 0.0) * 1.1
    penalty += max(server_mad, 0.0) * 4.0
    if family_delta < 0:
        penalty += abs(family_delta) * 80.0
    if "REVIEW" in str(decision).upper():
        penalty += 0.05
    return float(penalty)


def score_candidate(row: pd.Series, family_deltas: dict[str, float]) -> float:
    family = classify_family(str(row.get("candidate", "")))
    family_delta = family_deltas.get(family, 0.0)
    action_churn = _to_float(row.get("action_churn_vs_anchor"))
    point_churn = _to_float(row.get("point_churn_vs_anchor"))
    server_mad = _to_float(row.get("server_mad_vs_anchor"))
    decision = str(row.get("decision", ""))
    score = 1.0
    score += family_delta * 120.0
    score -= risk_penalty(action_churn, point_churn, server_mad, family_delta, decision)
    if family == "v261_cap1_point":
        score += 0.15
    if family == "clean_anchor_copy":
        score -= 0.03
    if family in {"ttmatch_diagnostic", "old_server_diagnostic"}:
        score -= 1.0
    if family.startswith("v28"):
        score -= 0.10
    return float(score)


def load_r200_summary(path: Path = R200_SUMMARY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing R200 summary: {path}")
    df = pd.read_csv(path)
    if "candidate" not in df.columns:
        raise ValueError(f"R200 summary must include candidate column: {path}")
    return df


def build_known_public_results() -> pd.DataFrame:
    return pd.DataFrame(KNOWN_PUBLIC_RESULTS)


def ensure_anchor_row(df: pd.DataFrame) -> pd.DataFrame:
    if (df["candidate"] == ANCHOR_FILE).any():
        return df
    anchor_row = {
        "candidate": ANCHOR_FILE,
        "rows": 1845,
        "action_churn_vs_anchor": 0.0,
        "point_churn_vs_anchor": 0.0,
        "server_mad_vs_anchor": 0.0,
        "server_corr_vs_anchor": 1.0,
        "action_tier": "low",
        "point_tier": "safe_probe",
        "server_tier": "clean",
        "point0_rate": np.nan,
        "decision": "KEEP",
        "oof_source": "",
    }
    return pd.concat([df, pd.DataFrame([anchor_row])], ignore_index=True, sort=False)


def find_candidate_file(candidate: str) -> Path | None:
    name = Path(candidate).name
    search_roots = [
        ROOT / "upload_candidates_20260519",
        ROOT / "submissions" / "selected",
        ROOT,
    ]
    for base in search_roots:
        if not base.exists():
            continue
        direct = base / name
        if direct.exists():
            return direct
    matches = list(ROOT.glob(f"**/{name}"))
    return matches[0] if matches else None


def copy_shortlist(shortlist: pd.DataFrame) -> list[str]:
    copied: list[str] = []
    for target_dir in [ROOT / "upload_candidates_20260519", ROOT / "submissions" / "selected"]:
        target_dir.mkdir(parents=True, exist_ok=True)
    for _, row in shortlist.iterrows():
        candidate = str(row["candidate"])
        source = find_candidate_file(candidate)
        if source is None:
            continue
        for target_dir in [ROOT / "upload_candidates_20260519", ROOT / "submissions" / "selected"]:
            dest = target_dir / source.name
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            copied.append(str(dest.relative_to(ROOT)))
    return sorted(set(copied))


def rank_candidates(summary: pd.DataFrame, public_df: pd.DataFrame) -> pd.DataFrame:
    family_deltas = family_public_delta(public_df, ANCHOR_PUBLIC)
    ranked = ensure_anchor_row(summary).copy()
    ranked["family"] = ranked["candidate"].map(classify_family)
    ranked["known_family_public_delta"] = ranked["family"].map(family_deltas).fillna(0.0)
    ranked["v284_score"] = ranked.apply(lambda row: score_candidate(row, family_deltas), axis=1)
    ranked["diagnostic_only"] = ranked["family"].isin(["ttmatch_diagnostic", "old_server_diagnostic"])
    ranked["public_negative_family"] = ranked["known_family_public_delta"] < -0.00005
    ranked = ranked.drop_duplicates(subset=["candidate"], keep="first")
    ranked = ranked.sort_values(["diagnostic_only", "public_negative_family", "v284_score"], ascending=[True, True, False])
    return ranked


def write_report(ranked: pd.DataFrame, public_df: pd.DataFrame, shortlist: pd.DataFrame, copied: list[str]) -> None:
    lines = [
        "# V284 Public Response Ranker",
        "",
        f"anchor_public: {ANCHOR_PUBLIC:.7f}",
        f"ranked_candidates: {len(ranked)}",
        "",
        "## Recommended Clean Shortlist",
        "",
    ]
    if shortlist.empty:
        lines.append("No clean shortlist candidate passed filters.")
    else:
        for _, row in shortlist.iterrows():
            lines.append(
                f"- {row['candidate']} | score={row['v284_score']:.4f} | "
                f"family={row['family']} | public_delta={row['known_family_public_delta']:.7f}"
            )
    lines += [
        "",
        "## Known Public Results",
        "",
    ]
    for _, row in public_df.iterrows():
        delta = float(row["public_pl"]) - ANCHOR_PUBLIC
        lines.append(f"- {row['candidate']}: PL={float(row['public_pl']):.7f}, delta={delta:+.7f}")
    lines += [
        "",
        "## Copied Files",
        "",
    ]
    if copied:
        lines.extend(f"- {path}" for path in copied)
    else:
        lines.append("- none")
    lines += [
        "",
        "## Interpretation",
        "",
        "V284 is a selection guard, not a new model. It keeps the current clean anchor at the top unless a candidate is both low-risk and not from a known public-negative family.",
    ]
    (OUTDIR / "v284_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    public_df = build_known_public_results()
    summary = load_r200_summary()
    ranked = rank_candidates(summary, public_df)
    clean = ranked[
        (~ranked["diagnostic_only"])
        & (~ranked["public_negative_family"])
        & (ranked["decision"].fillna("").astype(str).str.upper().str.contains("KEEP"))
    ].head(8)
    copied = copy_shortlist(clean.head(3))

    public_df.to_csv(OUTDIR / "v284_known_public_results.csv", index=False)
    ranked.to_csv(OUTDIR / "v284_ranked_candidates.csv", index=False)
    clean.to_csv(OUTDIR / "v284_shortlist.csv", index=False)
    write_report(ranked, public_df, clean, copied)

    payload = {
        "outdir": str(OUTDIR.relative_to(ROOT)),
        "ranked_candidates": int(len(ranked)),
        "recommended_clean": None if clean.empty else str(clean.iloc[0]["candidate"]),
        "copied_files": copied,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
