from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from analysis_v467_server_exhaustive_clean_sweep import (
    TabularModelConfig,
    aggregate_prefix_predictions,
    build_anchor_context,
    build_candidate_servers,
    build_specialist_masks,
    build_tabular_model_configs,
    package_server_only,
    run_pipeline,
    train_sequence_models,
)


def tiny_anchor(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": list(range(10, 10 + rows)),
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14][:rows],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9][:rows],
            "serverGetPoint": np.linspace(0.2, 0.8, rows),
        }
    )


def tiny_train(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": np.repeat(np.arange(1, 7), 2)[:rows],
            "match": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3][:rows],
            "numberGame": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2][:rows],
            "rally_id": list(range(1, rows + 1)),
            "scoreSelf": [0, 1, 2, 10, 0, 1, 9, 11, 0, 3, 10, 12][:rows],
            "scoreOther": [0, 0, 1, 10, 1, 1, 9, 10, 0, 2, 9, 11][:rows],
            "strikeNumber": [1, 2, 3, 6, 1, 4, 5, 7, 1, 2, 8, 9][:rows],
            "actionId": [0, 1, 8, 12, 15, 2, 9, 14, 0, 4, 10, 13][:rows],
            "pointId": [0, 1, 4, 7, 2, 5, 8, 9, 0, 3, 6, 9][:rows],
            "serverGetPoint": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1][:rows],
        }
    )


def tiny_test_new(anchor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in anchor.iterrows():
        for prefix in range(3 if idx % 2 == 0 else 1):
            rows.append(
                {
                    "rally_uid": row["rally_uid"],
                    "match": 1,
                    "numberGame": 1,
                    "rally_id": idx + 1,
                    "scoreSelf": idx + prefix,
                    "scoreOther": idx,
                    "strikeNumber": prefix + 1,
                    "actionId": row["actionId"],
                    "pointId": row["pointId"],
                }
            )
    return pd.DataFrame(rows)


def test_model_config_names_are_unique_and_include_available_optional_boosters():
    configs = build_tabular_model_configs(seed=467, runtime="fast")
    names = [config.name for config in configs]
    families = {config.family for config in configs}

    assert len(names) == len(set(names))
    assert {"linear", "tree", "boosting", "mlp"}.issubset(families)
    assert all(isinstance(config, TabularModelConfig) for config in configs)
    if importlib.util.find_spec("xgboost") is not None:
        assert any("xgboost" in name for name in names)
    if importlib.util.find_spec("catboost") is not None:
        assert any("catboost" in name for name in names)


def test_aggregation_modes_return_anchor_rows_and_late_weighted_differs_from_mean():
    anchor = tiny_anchor(3)
    test_new = pd.DataFrame(
        {
            "rally_uid": [10, 10, 10, 11, 12, 12],
            "strikeNumber": [1, 2, 3, 1, 3, 2],
        }
    )
    pred = np.array([0.1, 0.2, 0.9, 0.4, 0.7, 0.3])

    outputs = {mode: aggregate_prefix_predictions(test_new, anchor, pred, mode=mode) for mode in ["mean", "last", "max", "late_weighted"]}

    assert all(len(values) == len(anchor) for values in outputs.values())
    assert np.isclose(outputs["mean"][0], 0.4)
    assert np.isclose(outputs["last"][0], 0.9)
    assert outputs["late_weighted"][0] != outputs["mean"][0]


def test_specialist_masks_are_anchor_aligned():
    anchor = tiny_anchor(4).iloc[[2, 0, 3, 1]].reset_index(drop=True)
    test_new = tiny_test_new(anchor).sample(frac=1.0, random_state=467).reset_index(drop=True)

    context = build_anchor_context(anchor, test_new)
    masks = build_specialist_masks(context)

    assert context["rally_uid"].tolist() == anchor["rally_uid"].tolist()
    assert set(masks) == {"score_pressure", "phase_specialist", "terminal_like", "action_point_conditioned"}
    assert all(len(mask) == len(anchor) for mask in masks.values())
    assert all(mask.dtype == bool for mask in masks.values())


def test_candidate_packaging_preserves_v362_action_point_and_caps_mad():
    anchor = tiny_anchor(6)
    model_targets = {
        "tabular_global_rankmean": np.linspace(0.7, 0.3, len(anchor)),
        "tabular_boosting_rankmean": np.linspace(0.65, 0.35, len(anchor)),
        "tree_rankmean": np.linspace(0.6, 0.4, len(anchor)),
        "mlp_rankmean": np.linspace(0.45, 0.55, len(anchor)),
        "clean_source_rankmean": np.linspace(0.35, 0.65, len(anchor)),
    }
    masks = build_specialist_masks(build_anchor_context(anchor, tiny_test_new(anchor)))

    candidates = build_candidate_servers(anchor, model_targets, masks=masks)
    packaged = package_server_only(anchor, candidates[0].server, expected_rows=len(anchor))

    assert packaged["actionId"].tolist() == anchor["actionId"].tolist()
    assert packaged["pointId"].tolist() == anchor["pointId"].tolist()
    assert max(row.actual_mad for row in candidates if row.decision != "diagnostic_hold") <= 0.0100001
    assert any("full_exhaustive_ensemble" in row.candidate for row in candidates)


def test_sequence_models_skip_with_report_when_disabled():
    anchor = tiny_anchor(4)
    report: dict[str, object] = {}

    signals = train_sequence_models(tiny_train(12), tiny_test_new(anchor), anchor, runtime="test", enabled=False, report=report)

    assert signals == {}
    assert report["sequence_status"] == "skipped"
    assert "disabled" in str(report["sequence_skip_reason"])


def test_run_pipeline_writes_valid_server_only_submissions(tmp_path: Path):
    rows = 8
    anchor = tiny_anchor(rows)
    train = tiny_train(12)
    test_new = tiny_test_new(anchor)
    anchor_dir = tmp_path / "v362_point_hierarchical_specialists"
    anchor_dir.mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    test_new.to_csv(tmp_path / "test_new.csv", index=False)
    anchor.to_csv(anchor_dir / "submission_v362_depth_agree_only__v173action_v300server.csv", index=False)

    report = run_pipeline(root=tmp_path, outdir=tmp_path / "v467_server_exhaustive_clean_sweep", expected_rows=rows, runtime="test")
    board = pd.read_csv(tmp_path / "v467_server_exhaustive_clean_sweep" / "v467_server_search.csv")

    assert report["policy"]["no_upload_candidates_20260519"] is True
    assert set(report["aggregation_modes"]) == {"mean", "last", "max", "late_weighted"}
    assert not board.empty
    for path in board["path"].head(4):
        sub = pd.read_csv(path)
        assert list(sub.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
        assert len(sub) == rows
        assert sub["actionId"].equals(anchor["actionId"])
        assert sub["pointId"].equals(anchor["pointId"])
        assert sub["serverGetPoint"].between(0, 1).all()
