import json
import math
from pathlib import Path

import pandas as pd

from analysis_v406_public_response_meta_model import (
    build_candidate_feature_table,
    parse_public_pl_records,
    run_pipeline,
    score_candidates,
)


def test_parses_known_pl_records_from_fixture_log():
    log_text = """
| ID | Time | File | Public LB / PL | Rank | Notes |
| --- | --- | --- | ---: | ---: | --- |
| V338 | now | submission_v338_joint_moe.csv | 0.3590041 | 1 | clean |
| V341 | now | submission_v341_expand.csv | 0.3581101 | 2 | negative |

### V362 public probe result
PL = 0.3590124
V391 public-like proxy breakthrough did not transfer. PL = 0.3578818
"""

    table = parse_public_pl_records(log_text, include_fallback=False)

    by_key = {row.candidate_key: row.public_pl for row in table.itertuples(index=False)}
    assert by_key["v338"] == 0.3590041
    assert by_key["v341"] == 0.3581101
    assert any(abs(value - 0.3578818) < 1e-9 for value in by_key.values())
    assert {"pl_delta_vs_closest_anchor", "positive_transfer"}.issubset(table.columns)


def test_v391_like_proxy_candidates_are_high_risk(tmp_path):
    ranked_dir = tmp_path / "v391_oof_gated_submission_packager"
    ranked_dir.mkdir()
    pd.DataFrame(
        [
            {
                "rank": 1,
                "candidate": "v391_oof_point_top36",
                "path": "v391_oof_gated_submission_packager/submission_v391_oof_point_top36.csv",
                "selected_row_count": 32,
                "action_churn": 0,
                "point_churn": 32,
                "point0_additions": 0,
                "server_changed": 0,
            }
        ]
    ).to_csv(ranked_dir / "ranked_candidates.csv", index=False)

    candidates, _missing = build_candidate_feature_table(
        root=tmp_path,
        ranked_inputs=[("v391", ranked_dir / "ranked_candidates.csv")],
    )
    history = parse_public_pl_records("V362 PL = 0.3590124\nV391 PL = 0.3578818\n", include_fallback=False)
    scored, mode, _info = score_candidates(candidates, history)

    assert mode == "deterministic_risk_model"
    row = scored.iloc[0]
    assert row["risk"] == "high"
    assert row["synthetic_proxy_flag"] == 1
    assert "V391 public fail lineage" in row["score_reason"]


def test_no_fake_training_with_fewer_than_six_labels():
    candidates = pd.DataFrame(
        [
            {
                "candidate": "v400_public_agree_top9",
                "path": "submission_v400.csv",
                "source_family": "public_positive_agreement",
                "point_churn": 9,
                "action_churn": 0,
                "server_changed": 0,
                "point0_additions": 0,
                "point0_removals": 0,
                "longside_transition_count": 6,
                "half_boundary_transition_count": 3,
                "short_control_transition_count": 0,
                "synthetic_proxy_flag": 0,
                "public_positive_component_agreement_flag": 1,
                "posterior_model_flag": 0,
                "specialist_flag": 0,
                "rank": 1,
            }
        ]
    )
    history = parse_public_pl_records(
        """
| ID | File | Public LB / PL |
| --- | --- | ---: |
| V338 | submission_v338.csv | 0.3590041 |
| V341 | submission_v341.csv | 0.3581101 |
| V362 | submission_v362.csv | 0.3590124 |
""",
        include_fallback=False,
    )

    scored, mode, info = score_candidates(candidates, history)

    assert mode == "deterministic_risk_model"
    assert info["trained"] is False
    assert info["labeled_public_examples"] < 6
    assert scored.iloc[0]["model_mode"] == "deterministic_risk_model"


def test_pipeline_writes_finite_candidate_response_scores(tmp_path):
    v400_dir = tmp_path / "v400_public_component_recombination"
    v400_dir.mkdir()
    selected_path = v400_dir / "selected_rows_v400_public_agree_top9.csv"
    pd.DataFrame(
        [
            {"anchor_point": 7, "new_point": 8},
            {"anchor_point": 9, "new_point": 6},
            {"anchor_point": 2, "new_point": 5},
        ]
    ).to_csv(selected_path, index=False)
    pd.DataFrame(
        [
            {
                "candidate": "v400_public_agree_top9",
                "path": "v400_public_component_recombination/submission_v400_public_agree_top9.csv",
                "selected_rows": "v400_public_component_recombination/selected_rows_v400_public_agree_top9.csv",
                "selected_row_count": 3,
                "action_churn": 0,
                "point_churn": 3,
                "point0_additions": 0,
                "server_changed": 0,
                "risk": "safe",
                "evidence": "deterministic_public_source_agreement",
            }
        ]
    ).to_csv(v400_dir / "ranked_candidates.csv", index=False)
    (tmp_path / "experiments_log.md").write_text("V362 PL = 0.3590124\n", encoding="utf-8")

    # Exercise the core writer through the public run_pipeline contract in the
    # real workspace-owned outdir, then assert its numeric contract.
    report = run_pipeline(outdir=tmp_path / "v406_public_response_meta_model")
    scores = pd.read_csv(report["outputs"]["candidate_response_scores"])

    assert len(scores) > 0
    assert scores["response_score"].map(math.isfinite).all()
    parsed_report = json.loads(Path(report["outputs"]["search_report"]).read_text(encoding="utf-8"))
    assert parsed_report["policy"]["generated_submissions"] is False
