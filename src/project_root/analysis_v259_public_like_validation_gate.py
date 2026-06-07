from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path("v259_public_like_validation_gate")
R200_SUMMARY = Path("r200_local_validation_dashboard") / "r200_candidate_summary.csv"

PUBLIC_RESULTS = [
    {
        "label": "current_anchor_v173_v188cap5_r121",
        "candidate_match": "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv",
        "public_pl": 0.3573932,
    },
    {
        "label": "v220_public_fail",
        "candidate_match": "v220",
        "public_pl": 0.3542440,
    },
    {
        "label": "v191_v166_public_fail",
        "candidate_match": "v191_v166",
        "public_pl": 0.3509562,
    },
    {
        "label": "v248_public_fail",
        "candidate_match": "v248",
        "public_pl": 0.3554156,
    },
    {
        "label": "v202_public_fail",
        "candidate_match": "v202",
        "public_pl": 0.3561381,
    },
]


def load_r200() -> pd.DataFrame:
    if not R200_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(R200_SUMMARY)


def best_r200_match(r200: pd.DataFrame, pattern: str) -> dict:
    if r200.empty:
        return {}
    mask = r200["candidate"].astype(str).str.contains(pattern, case=False, regex=False)
    subset = r200[mask].copy()
    if subset.empty:
        return {}
    for col in [
        "ordinary_action_delta_vs_anchor",
        "ordinary_point_macro_f1",
        "action_churn_vs_anchor",
        "point_churn_vs_anchor",
        "server_mad_vs_anchor",
    ]:
        if col not in subset:
            subset[col] = np.nan
    sort_col = "ordinary_action_delta_vs_anchor"
    if subset[sort_col].notna().any():
        subset = subset.sort_values(sort_col, ascending=False)
    return subset.iloc[0].to_dict()


def spearman_public_alignment(frame: pd.DataFrame, metric: str) -> float:
    sub = frame[["public_pl", metric]].dropna()
    if len(sub) < 3 or sub[metric].nunique() < 2:
        return float("nan")
    return float(sub["public_pl"].rank().corr(sub[metric].rank(), method="pearson"))


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    r200 = load_r200()
    rows = []
    for item in PUBLIC_RESULTS:
        rec = dict(item)
        match = best_r200_match(r200, item["candidate_match"])
        rec.update(
            {
                "local_candidate": match.get("candidate", ""),
                "ordinary_action_delta_vs_anchor": pd.to_numeric(match.get("ordinary_action_delta_vs_anchor", np.nan), errors="coerce"),
                "ordinary_point_macro_f1": pd.to_numeric(match.get("ordinary_point_macro_f1", np.nan), errors="coerce"),
                "action_churn_vs_anchor": pd.to_numeric(match.get("action_churn_vs_anchor", np.nan), errors="coerce"),
                "point_churn_vs_anchor": pd.to_numeric(match.get("point_churn_vs_anchor", np.nan), errors="coerce"),
                "server_mad_vs_anchor": pd.to_numeric(match.get("server_mad_vs_anchor", np.nan), errors="coerce"),
            }
        )
        rows.append(rec)
    backtest = pd.DataFrame(rows)
    metrics = [
        "ordinary_action_delta_vs_anchor",
        "ordinary_point_macro_f1",
        "action_churn_vs_anchor",
        "point_churn_vs_anchor",
        "server_mad_vs_anchor",
    ]
    alignment = {metric: spearman_public_alignment(backtest, metric) for metric in metrics}
    anchor_public = float(backtest.loc[backtest["label"].eq("current_anchor_v173_v188cap5_r121"), "public_pl"].iloc[0])
    failed_max = float(backtest.loc[~backtest["label"].eq("current_anchor_v173_v188cap5_r121"), "public_pl"].max())
    historical_order_pass = bool(anchor_public > failed_max)
    report = {
        "r200_rows": int(len(r200)),
        "historical_rows": int(len(backtest)),
        "historical_order_pass": historical_order_pass,
        "best_alignment_metric": max(
            alignment,
            key=lambda key: -999 if pd.isna(alignment[key]) else alignment[key],
        ),
        "alignment": alignment,
        "verdict": "VALIDATION_REFERENCE_READY" if historical_order_pass else "VALIDATION_REFERENCE_INCOMPLETE",
    }
    backtest.to_csv(OUTDIR / "v259_historical_backtest.csv", index=False)
    (OUTDIR / "v259_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v259_report.md").write_text(
        "# V259 Public-Like Validation Gate\n\n"
        f"- R200 rows: `{len(r200)}`\n"
        f"- Historical public probes: `{len(backtest)}`\n"
        f"- Anchor public > failed probes: `{historical_order_pass}`\n"
        f"- Best available alignment metric: `{report['best_alignment_metric']}`\n"
        f"- Verdict: `{report['verdict']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "verdict": report["verdict"], "historical_order_pass": historical_order_pass}))


if __name__ == "__main__":
    main()
