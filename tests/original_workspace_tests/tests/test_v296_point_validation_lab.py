from pathlib import Path

import pandas as pd

from analysis_v296_point_validation_lab import (
    PUBLIC_BACKTEST_COLUMNS,
    RISK_COLUMNS,
    build_public_backtest,
    build_report_payload,
    build_risk_table,
    render_markdown_report,
    read_search_tables,
    risk_label,
)


def test_risk_label_helper_green_yellow_red_rules():
    assert (
        risk_label(
            specialist_group="long789",
            public_delta=0.0003,
            point_churn=0.004,
            point0_rate_delta=0.0,
            local_delta=0.001,
        )
        == "YELLOW"
    )
    assert (
        risk_label(
            specialist_group="rare134",
            public_delta=0.0003,
            point_churn=0.002,
            point0_rate_delta=0.0,
            local_delta=0.001,
        )
        == "RED"
    )
    assert (
        risk_label(
            specialist_group="point0",
            public_delta=0.0003,
            point_churn=0.002,
            point0_rate_delta=0.001,
            local_delta=0.001,
        )
        == "RED"
    )
    assert (
        risk_label(
            specialist_group="v261_like",
            public_delta=0.0003122,
            point_churn=0.004,
            point0_rate_delta=0.0,
            local_delta=0.001,
        )
        == "GREEN"
    )


def test_public_result_table_has_v261_and_v277():
    backtest = build_public_backtest([])
    candidates = set(backtest["candidate"])
    assert "V261 cap1" in candidates
    assert "V277" in candidates
    assert list(backtest.columns) == PUBLIC_BACKTEST_COLUMNS


def test_missing_search_files_do_not_crash():
    missing_root = Path("v296_point_validation_lab/__missing_workspace__")
    tables, manifest = read_search_tables(missing_root)
    assert tables == []
    assert manifest
    assert all(item["status"] == "missing" for item in manifest)


def test_report_schema_has_required_columns():
    search = pd.DataFrame(
        [
            {
                "candidate": "v293_long789_cap0p005",
                "specialist_group": "long789",
                "cap": 0.005,
                "delta_vs_v261": 0.0007218936821360156,
                "public_like_delta": 0.000692954396224188,
                "point_churn": 0.004991966949736057,
                "test_changed_rows": 9,
                "test_point0_rate_delta": 0.0,
                "long789_mean_delta": 0.0024063122737867835,
                "rare134_mean_delta": 0.0,
                "point0_f1_delta": 0.0,
                "path": "v293_point_weakclass_residual_lab/submission.csv",
            }
        ]
    )
    public_backtest = build_public_backtest([("v293_candidate_search.csv", search)])
    risk = build_risk_table(public_backtest)
    report = build_report_payload(public_backtest, risk, [{"path": "x", "status": "loaded"}])

    assert list(public_backtest.columns) == PUBLIC_BACKTEST_COLUMNS
    assert list(risk.columns) == RISK_COLUMNS
    assert set(report) >= {
        "version",
        "public_backtest_rows",
        "risk_rows",
        "public_backtest_columns",
        "risk_columns",
        "search_table_manifest",
        "key_risk_conclusions",
    }


def test_markdown_report_does_not_require_optional_tabulate():
    public_backtest = build_public_backtest([])
    risk = build_risk_table(public_backtest)
    report = build_report_payload(public_backtest, risk, [])
    markdown = render_markdown_report(report, risk)
    assert "# V296 Point Validation Lab" in markdown
    assert "V261 cap1" in markdown
