import pandas as pd
from pathlib import Path

from analysis_v304_clean_decision_packager import (
    PUBLIC_BASELINE_SCORE,
    build_decision_table,
    public_results_frame,
)


def test_public_delta_vs_v261_uses_embedded_scores():
    public = public_results_frame()
    by_candidate = public.set_index("candidate")

    assert by_candidate.loc["V261 cap1", "public_delta_vs_v261"] == 0.0
    assert round(by_candidate.loc["V300 best_safe_repack", "public_delta_vs_v261"], 7) == round(
        0.3576975 - PUBLIC_BASELINE_SCORE,
        7,
    )
    assert round(by_candidate.loc["V272", "public_delta_vs_v261"], 7) == round(
        0.3576159 - PUBLIC_BASELINE_SCORE,
        7,
    )
    assert round(by_candidate.loc["V277", "public_delta_vs_v261"], 7) == round(
        0.3574825 - PUBLIC_BASELINE_SCORE,
        7,
    )
    assert round(by_candidate.loc["V291", "public_delta_vs_v261"], 7) == round(
        0.3559391 - PUBLIC_BASELINE_SCORE,
        7,
    )


def test_decision_ranking_prefers_v300_and_keeps_v302_unknown_placeholder():
    v300 = pd.DataFrame(
        [
            {
                "candidate": "submission_v300_best_safe_repack__v173action_v261point_server.csv",
                "path": "v300/submission.csv",
                "blend_kind": "best_safe_repack",
                "proxy_delta_vs_proxy_base": 0.0015,
                "server_mad_vs_anchor": 0.0001,
                "action_changed_vs_anchor": 0,
                "point_changed_vs_anchor": 0,
                "risk_tier": "safe",
                "verdict": "CANDIDATE_FOR_REVIEW",
            },
            {
                "candidate": "submission_v300_rankavg_w0p005__v173action_v261point_server.csv",
                "path": "v300/rankavg.csv",
                "blend_kind": "rankavg",
                "proxy_delta_vs_proxy_base": 0.0007,
                "server_mad_vs_anchor": 0.00001,
                "action_changed_vs_anchor": 0,
                "point_changed_vs_anchor": 0,
                "risk_tier": "safe",
                "verdict": "CANDIDATE_FOR_REVIEW",
            },
        ]
    )
    v301 = pd.DataFrame(
        [
            {
                "candidate": "support_pair_cap0p005",
                "action_churn": 0.0048,
                "point_churn": 0.0048,
                "pair_changed_rows": 9,
                "support_delta": 0.2,
                "recommendation": "DO_NOT_UPLOAD",
            }
        ]
    )
    v299 = pd.DataFrame(
        [
            {
                "candidate": "v299_no_point0_cap0p005",
                "point_churn": 0.0043,
                "available_source_local_delta": 0.0004,
                "upload_recommendation": "DO_NOT_UPLOAD",
            }
        ]
    )
    r200 = pd.DataFrame(
        [
            {
                "candidate": "submission_v300_best_safe_repack__v173action_v261point_server.csv",
                "action_churn_vs_anchor": 0.0,
                "point_churn_vs_anchor": 0.0,
                "server_mad_vs_anchor": 0.0015,
                "decision": "KEEP",
            }
        ]
    )

    table = build_decision_table(
        search_tables=[
            ("V300", Path("v300/v300_server_search.csv"), v300),
            ("V301", Path("v301/v301_pair_search.csv"), v301),
            ("V299", Path("v299/v299_candidate_search.csv"), v299),
        ],
        r200_summary=r200,
        workspace_root=Path("__v304_no_v302_workspace__"),
    )

    assert list(table.columns)[:8] == [
        "candidate",
        "task_changed",
        "public_score",
        "public_delta_vs_v261",
        "local_delta",
        "churn",
        "risk",
        "upload_priority",
    ]
    assert table.iloc[0]["candidate"] == "submission_v300_best_safe_repack__v173action_v261point_server.csv"
    assert table.iloc[0]["risk"] == "CURRENT_CLEAN_BEST"
    assert table.iloc[0]["upload_priority"] == 1

    v302 = table[table["candidate"].eq("V302 placeholder (absent)")].iloc[0]
    assert v302["risk"] == "UNKNOWN_ABSENT"
    assert v302["public_score"] is None

    weak = table[table["candidate"].isin(["support_pair_cap0p005", "v299_no_point0_cap0p005"])]
    assert set(weak["risk"]) == {"REJECT_POINT_ACTION_WEAK"}
    assert weak["upload_priority"].min() > v302["upload_priority"]
