import json

import pandas as pd

from analysis_v389_synthetic_oof_proxy_lab import (
    compute_auc_like_separation,
    gate_ranked_rows,
    run_pipeline,
    score_proxy_pool,
)


def test_weighted_proxy_score_prefers_supported_allowed_safe_point_rows():
    pool = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "base_point": [8, 8],
            "candidate_point": [9, 0],
            "support_count": [24, 40],
            "source_family_count": [5, 8],
            "same_depth": [True, False],
            "same_side": [True, False],
            "is_point0_addition": [False, True],
            "sources": ["v338_joint_moe_pack|v362_point_hierarchical_specialists", "v306_point0_addition_probe"],
            "synthetic_allowed": [True, True],
            "synthetic_compatibility_score": [0.88, 0.95],
        }
    )

    ranked = score_proxy_pool(pool, kind="point")

    assert ranked.iloc[0]["rally_uid"] == 1
    assert ranked.loc[ranked["rally_uid"] == 1, "proxy_score"].item() > ranked.loc[
        ranked["rally_uid"] == 2, "proxy_score"
    ].item()


def test_historical_backtest_auc_like_separates_positive_and_negative_rows():
    backtest = pd.DataFrame(
        {
            "experiment": ["v338", "v362", "v191", "v220"],
            "historical_label": [1, 1, 0, 0],
            "proxy_score": [0.82, 0.76, 0.24, 0.31],
        }
    )

    assert compute_auc_like_separation(backtest) == 1.0


def test_gate_passes_only_safe_rows_above_threshold():
    rows = pd.DataFrame(
        {
            "rally_uid": [1, 2, 3],
            "proxy_score": [0.70, 0.90, 0.40],
            "is_point0_addition": [False, True, False],
        }
    )

    gated = gate_ranked_rows(rows, kind="point", threshold=0.62)

    assert gated["pass_gate"].tolist() == [True, False, False]


def test_run_pipeline_records_missing_v388_without_faking_candidate_pools(tmp_path):
    outdir = tmp_path / "v389_synthetic_oof_proxy_lab"

    report = run_pipeline(root=tmp_path, outdir=outdir)

    assert report["missing_v388"] is True
    assert report["validation_mode"] == "proxy_unavailable"
    assert report["point_rows_ranked"] == 0
    assert report["action_rows_ranked"] == 0
    stored = json.loads((outdir / "search_report.json").read_text())
    assert stored["missing_v388"] is True
    assert pd.read_csv(outdir / "ranked_point_pool.csv").empty
    assert pd.read_csv(outdir / "ranked_action_pool.csv").empty
