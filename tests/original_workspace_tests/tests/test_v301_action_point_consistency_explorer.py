import math
from pathlib import Path
from uuid import uuid4

import pandas as pd

from analysis_v301_action_point_consistency_explorer import (
    REQUIRED_SUBMISSION_COLUMNS,
    VARIANTS,
    apply_selected_changes,
    build_empirical_support,
    candidate_pool,
    max_changed_rows,
    run_pipeline,
    select_variant,
)


def _train_frame() -> pd.DataFrame:
    rows = []
    for rally_uid in range(1, 13):
        rows.extend(
            [
                {"rally_uid": rally_uid, "strikeNumber": 1, "actionId": 15, "pointId": 4},
                {"rally_uid": rally_uid, "strikeNumber": 2, "actionId": 1, "pointId": 8},
                {"rally_uid": rally_uid, "strikeNumber": 3, "actionId": 0, "pointId": 0},
            ]
        )
    for rally_uid in range(13, 25):
        rows.extend(
            [
                {"rally_uid": rally_uid, "strikeNumber": 1, "actionId": 16, "pointId": 5},
                {"rally_uid": rally_uid, "strikeNumber": 2, "actionId": 2, "pointId": 7},
                {"rally_uid": rally_uid, "strikeNumber": 3, "actionId": 1, "pointId": 8},
            ]
        )
    return pd.DataFrame(rows)


def _anchor_frame(n: int = 1000) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        if idx % 11 == 0:
            action_id, point_id = 0, 8
        elif idx % 13 == 0:
            action_id, point_id = 15, 4
        elif idx % 17 == 0:
            action_id, point_id = 2, 8
        else:
            action_id, point_id = 1, 7
        rows.append(
            {
                "rally_uid": idx + 1,
                "actionId": action_id,
                "pointId": point_id,
                "serverGetPoint": idx % 2,
            }
        )
    return pd.DataFrame(rows)


def test_max_changed_rows_respects_fraction_and_global_cap():
    assert max_changed_rows(1845, 0.0025) == math.floor(1845 * 0.0025)
    assert max_changed_rows(10000, 0.005) == 10
    assert max_changed_rows(100, 0.0025) == 0


def test_empirical_support_prefers_terminal_action0_point0_and_blocks_serve_next_prior():
    support = build_empirical_support(_train_frame())
    assert support.point_given_action(0, 0) > support.point_given_action(0, 8)
    assert support.action_given_point(0, 0) > support.action_given_point(1, 0)
    assert support.serve_next_prior(15) == 0.0
    assert support.serve_next_prior(1) > 0.0


def test_select_variant_caps_rows_and_filters_serve_actions():
    support = build_empirical_support(_train_frame())
    candidates = candidate_pool(_anchor_frame(1000), support)
    variant = next(v for v in VARIANTS if v.name == "support_pair_cap0p005")
    selected = select_variant(candidates, variant, row_count=1000)
    assert len(selected) <= 5
    assert not selected["candidate_action"].between(15, 18).any()


def test_apply_selected_changes_preserves_schema_and_server_column():
    anchor = _anchor_frame(1000)
    support = build_empirical_support(_train_frame())
    candidates = candidate_pool(anchor, support)
    variant = next(v for v in VARIANTS if v.name == "terminal_consistency_cap0p0025")
    selected = select_variant(candidates, variant, row_count=len(anchor))
    submission = apply_selected_changes(anchor, selected)
    assert submission.columns.tolist() == REQUIRED_SUBMISSION_COLUMNS
    assert submission["serverGetPoint"].equals(anchor["serverGetPoint"])
    changed_rows = (submission[["actionId", "pointId"]] != anchor[["actionId", "pointId"]]).any(axis=1).sum()
    assert changed_rows <= 2


def test_run_pipeline_writes_expected_outputs_without_server_churn():
    tmp_path = Path("v301_action_point_consistency_explorer") / f"pytest_{uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_path = tmp_path / "train.csv"
    anchor_path = tmp_path / "anchor.csv"
    out_dir = tmp_path / "v301"
    _train_frame().to_csv(train_path, index=False)
    _anchor_frame(1000).to_csv(anchor_path, index=False)

    report = run_pipeline(train_path=train_path, anchor_path=anchor_path, out_dir=out_dir, copy_to_src=False)

    assert report["upload_recommendation"] == "DO_NOT_UPLOAD"
    assert set(report["variants"]) == {v.name for v in VARIANTS}
    assert (out_dir / "v301_pair_search.csv").exists()
    assert (out_dir / "v301_changed_row_audit.csv").exists()
    for variant_name, metrics in report["variants"].items():
        submission = pd.read_csv(out_dir / f"submission_v301_{variant_name}.csv")
        anchor = pd.read_csv(anchor_path)
        assert submission.columns.tolist() == REQUIRED_SUBMISSION_COLUMNS
        assert submission["serverGetPoint"].equals(anchor["serverGetPoint"])
        assert int(metrics["pair_changed_rows"]) <= 10
