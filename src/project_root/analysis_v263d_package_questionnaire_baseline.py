"""Package V263 questionnaire baseline components.

This packager is deliberately conservative: it only combines V263A/B/C
components when their own search tables mark them as locally acceptable.  It
does not write upload candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v263_questionnaire_baseline_helpers import (
    EXPECTED_COLUMNS,
    OUTDIR,
    load_v261_cap1_anchor,
    write_local_submission,
)


ACTION_SEARCH = OUTDIR / "v263a_action_search.csv"
POINT_SEARCH = OUTDIR / "v263b_point_search.csv"
SERVER_SEARCH = OUTDIR / "v263c_server_search.csv"
PACKAGE_SEARCH = OUTDIR / "v263d_package_search.csv"


def _load_search(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "verdict" not in df.columns:
        return pd.DataFrame()
    verdict = df["verdict"].astype(str)
    accepted = df[verdict.str.contains("CANDIDATE|ACCEPT|LOCAL_POSITIVE", regex=True, na=False)].copy()
    if "path" in accepted.columns:
        accepted = accepted[accepted["path"].astype(str).map(lambda p: Path(p).exists())]
    return accepted


def pick_action(df: pd.DataFrame) -> dict | None:
    acc = _accepted_rows(df)
    if acc.empty:
        return None
    if "action_macro_f1" in acc.columns:
        # V263A may be positive only against a weak proxy base.  Do not accept
        # action candidates whose own OOF Macro-F1 is far below the established
        # V173-scale action range.
        acc = acc[pd.to_numeric(acc["action_macro_f1"], errors="coerce").ge(0.25)]
    if acc.empty:
        return None
    sort_cols = [c for c in ["delta_vs_proxy_base", "action_macro_f1"] if c in acc.columns]
    return acc.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else acc.iloc[0].to_dict()


def pick_point(df: pd.DataFrame) -> dict | None:
    acc = _accepted_rows(df)
    if acc.empty:
        return None
    if "point0_rate_test" in acc.columns:
        acc = acc[pd.to_numeric(acc["point0_rate_test"], errors="coerce").between(0.20, 0.35)]
    if acc.empty:
        return None
    sort_cols = [c for c in ["delta_vs_proxy_base", "point_macro_f1"] if c in acc.columns]
    return acc.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else acc.iloc[0].to_dict()


def pick_server(df: pd.DataFrame) -> dict | None:
    acc = _accepted_rows(df)
    if acc.empty:
        return None
    if "server_mad_vs_anchor" in acc.columns:
        acc = acc[pd.to_numeric(acc["server_mad_vs_anchor"], errors="coerce").le(0.02)]
    if acc.empty:
        return None
    sort_cols = [c for c in ["delta_vs_proxy_base", "server_auc"] if c in acc.columns]
    return acc.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else acc.iloc[0].to_dict()


def _read_component(row: dict | None) -> pd.DataFrame | None:
    if not row:
        return None
    path = Path(str(row.get("path", "")))
    if not path.exists():
        return None
    sub = pd.read_csv(path)
    if len(sub) != 1845 or list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Bad component submission: {path}")
    return sub


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    base = load_v261_cap1_anchor()
    action_row = pick_action(_load_search(ACTION_SEARCH))
    point_row = pick_point(_load_search(POINT_SEARCH))
    server_row = pick_server(_load_search(SERVER_SEARCH))

    chosen = {
        "action": action_row,
        "point": point_row,
        "server": server_row,
    }
    out = base.copy()
    action_sub = _read_component(action_row)
    point_sub = _read_component(point_row)
    server_sub = _read_component(server_row)
    if action_sub is not None:
        out["actionId"] = action_sub["actionId"].astype(int).to_numpy()
    if point_sub is not None:
        out["pointId"] = point_sub["pointId"].astype(int).to_numpy()
    if server_sub is not None:
        out["serverGetPoint"] = server_sub["serverGetPoint"].astype(float).to_numpy()

    accepted_components = [k for k, v in chosen.items() if v is not None]
    package_rows = []
    generated_path = ""
    verdict = "LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    if accepted_components:
        path = OUTDIR / "submission_v263d_best_combo__no_old.csv"
        write_local_submission(path, out)
        generated_path = str(path)
        verdict = "LOCAL_COMPONENTS_PACKAGED_REVIEW_BEFORE_UPLOAD"
        package_rows.append(
            {
                "candidate": "v263d_best_combo",
                "verdict": verdict,
                "accepted_components": ",".join(accepted_components),
                "path": generated_path,
                "action_changed_rows": int(np.sum(out["actionId"].to_numpy() != base["actionId"].to_numpy())),
                "point_changed_rows": int(np.sum(out["pointId"].to_numpy() != base["pointId"].to_numpy())),
                "server_mad_vs_anchor": float(np.mean(np.abs(out["serverGetPoint"].to_numpy(dtype=float) - base["serverGetPoint"].to_numpy(dtype=float)))),
            }
        )
    else:
        package_rows.append(
            {
                "candidate": "v263d_no_combo",
                "verdict": verdict,
                "accepted_components": "",
                "path": "",
                "action_changed_rows": 0,
                "point_changed_rows": 0,
                "server_mad_vs_anchor": 0.0,
            }
        )
    pd.DataFrame(package_rows).to_csv(PACKAGE_SEARCH, index=False)

    report = {
        "verdict": verdict,
        "upload_recommendation": "review_only_do_not_upload_automatically" if accepted_components else "do_not_upload",
        "accepted_components": accepted_components,
        "chosen": chosen,
        "generated_path": generated_path,
        "questionnaire_mapping": {
            "data_type": "sequential_prefix_features",
            "task_1_2_metric": "macro_f1_multiclass_with_balanced_models",
            "task_3_metric": "auc_probability_blend",
            "architecture": "shared_feature_builder_separate_heads_low_churn_residuals",
        },
    }
    (OUTDIR / "v263_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v263_report.md").write_text(
        "# V263 Questionnaire Baseline Rebuild\n\n"
        f"- Verdict: `{verdict}`\n"
        f"- Accepted components: `{','.join(accepted_components) if accepted_components else 'none'}`\n"
        f"- Generated combo: `{generated_path or 'none'}`\n"
        f"- Upload recommendation: `{report['upload_recommendation']}`\n\n"
        "## Interpretation\n\n"
        "V263 follows the questionnaire baseline idea: sequential safe features, class-balanced separate heads, Macro-F1/AUC aligned metrics, and low-churn residual export.\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "accepted_components": accepted_components, "generated_path": generated_path}, indent=2))


if __name__ == "__main__":
    main()
