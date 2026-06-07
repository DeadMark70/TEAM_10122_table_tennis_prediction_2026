from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis_v332_hierarchical_action_model as v332


def test_action_family_mapping_is_complete_and_disjoint():
    seen = []
    for actions in v332.ACTION_FAMILIES.values():
        seen.extend(actions)

    assert sorted(seen) == list(range(19))
    assert len(seen) == len(set(seen))
    assert v332.ACTION_FAMILIES["zero"] == (0,)
    assert v332.ACTION_FAMILIES["attack"] == (1, 2, 3, 4, 5, 6, 7)
    assert v332.ACTION_FAMILIES["control"] == (8, 9, 10, 11)
    assert v332.ACTION_FAMILIES["defensive"] == (12, 13, 14)
    assert v332.ACTION_FAMILIES["serve"] == (15, 16, 17, 18)
    assert [v332.action_family_id(i) for i in range(19)] == [0] + [1] * 7 + [2] * 4 + [3] * 3 + [4] * 4


def test_soft_route_probabilities_normalize():
    family_prob = np.array(
        [
            [0.10, 0.35, 0.25, 0.20, 0.10],
            [0.70, 0.05, 0.10, 0.10, 0.05],
        ]
    )
    expert_probs = {
        1: np.tile(np.ones(7) / 7.0, (2, 1)),
        2: np.tile(np.ones(4) / 4.0, (2, 1)),
        3: np.tile(np.ones(3) / 3.0, (2, 1)),
        4: np.tile(np.ones(4) / 4.0, (2, 1)),
    }

    prob = v332.compose_action_probabilities(family_prob, expert_probs)

    assert prob.shape == (2, 19)
    np.testing.assert_allclose(prob.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(prob[:, 0], family_prob[:, 0])
    np.testing.assert_allclose(prob[:, 1:8].sum(axis=1), family_prob[:, 1])
    np.testing.assert_allclose(prob[:, 8:12].sum(axis=1), family_prob[:, 2])
    np.testing.assert_allclose(prob[:, 12:15].sum(axis=1), family_prob[:, 3])
    np.testing.assert_allclose(prob[:, 15:19].sum(axis=1), family_prob[:, 4])


def test_strict_v173_anchor_required(monkeypatch):
    outdir = Path("v332_hierarchical_action_model") / "pytest_tmp"
    outdir.mkdir(parents=True, exist_ok=True)
    anchor_path = outdir / "anchor.csv"
    pd.DataFrame(
        {
            "rally_uid": [11, 12],
            "actionId": [1, 3],
            "pointId": [4, 5],
            "serverGetPoint": [0.2, 0.8],
        }
    ).to_csv(anchor_path, index=False)

    def fake_rebuild():
        return {
            "rows": pd.DataFrame({"next_actionId": [1, 2], "fold": [0, 1]}),
            "test_rows": pd.DataFrame({"prefix_len": [1, 2]}),
            "rally_uids": np.array([11, 12]),
            "v173_pred_oof": np.array([1, 2]),
            "v173_pred_test": np.array([1, 2]),
        }

    monkeypatch.setattr(v332, "OUTDIR", outdir)
    monkeypatch.setattr(v332, "ANCHOR_SUBMISSION", anchor_path)
    monkeypatch.setattr(v332, "rebuild_strict_v173_anchor", fake_rebuild)

    with pytest.raises(ValueError, match="packaged V306 actionId"):
        v332.run_pipeline()

    assert not list(outdir.glob("submission_v332*.csv"))


def test_export_preserves_point_and_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [1, 5, 9],
            "pointId": [0, 7, 3],
            "serverGetPoint": [0.123456789, 0.5, 0.999999999],
        }
    )
    pred_action = np.array([0, 5, 12])

    out = v332.build_export_frame(anchor, pred_action)

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [0, 5, 12]
    assert out["pointId"].tolist() == anchor["pointId"].tolist()
    assert out["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()


def test_evidence_gate_blocks_low_precision():
    good = {
        "action_oof_delta": 0.0015,
        "changed_row_oof_precision": 0.45,
        "changed_action_rows": 5,
        "serve_action_rows": 0,
        "serve_count_delta": 0,
    }

    assert v332.evidence_passes(good)
    assert not v332.evidence_passes({**good, "action_oof_delta": 0.00149})
    assert not v332.evidence_passes({**good, "changed_row_oof_precision": 0.449})
    assert not v332.evidence_passes({**good, "changed_action_rows": 4})
    assert not v332.evidence_passes({**good, "changed_action_rows": 81})
    assert not v332.evidence_passes({**good, "serve_action_rows": 1})
    assert not v332.evidence_passes({**good, "serve_count_delta": 1})


def test_drop_leaky_feature_columns_removes_targets_and_next_labels():
    frame = pd.DataFrame(
        {
            "prefix_len": [1, 2],
            "lag0_actionId": [4, 5],
            "next_actionId": [9, 9],
            "next_pointId": [0, 8],
            "y_action_family": [1, 2],
            "true_action": [3, 4],
            "serverGetPoint": [0, 1],
        }
    )

    safe = v332.drop_leaky_feature_columns(frame)

    assert safe.columns.tolist() == ["prefix_len", "lag0_actionId"]


def test_protected_output_path_blocks_forbidden_dirs():
    path = v332.protected_output_path(Path("v332_hierarchical_action_model"), "submission_v332_soft_route_b10__v306point_v300server.csv")
    assert path.parent == Path("v332_hierarchical_action_model")

    with pytest.raises(ValueError):
        v332.protected_output_path(Path("upload_candidates"), "submission_v332_bad.csv")
