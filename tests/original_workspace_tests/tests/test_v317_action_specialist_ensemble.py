from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v317_action_specialist_ensemble import (
    CANDIDATE_SPECS,
    DEFAULT_SPECIALIST_GROUPS,
    ExportSpec,
    action_distribution,
    build_export_frame,
    evidence_passes,
    protected_output_path,
)


def test_specialist_groups_match_task2_focus_actions():
    assert DEFAULT_SPECIALIST_GROUPS["zero_terminal"] == (0,)
    assert DEFAULT_SPECIALIST_GROUPS["attack_finish_control"] == (3, 4, 5, 7)
    assert DEFAULT_SPECIALIST_GROUPS["rare_control_defense"] == (8, 9, 12, 14)
    assert set(CANDIDATE_SPECS) == {
        "submission_v317_zero_terminal_action_budget10__v306point_v300server.csv",
        "submission_v317_attack_finish_action_budget20__v306point_v300server.csv",
        "submission_v317_raredefense_action_budget15__v306point_v300server.csv",
        "submission_v317_specialist_union_safe__v306point_v300server.csv",
    }


def test_evidence_gate_is_stricter_than_v312_and_blocks_serve_explosion():
    assert evidence_passes(
        {
            "action_oof_delta": 0.0015,
            "changed_row_oof_precision": 0.30,
            "changed_action_rows": 20,
            "serve_action_rows": 0,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.00149,
            "changed_row_oof_precision": 0.30,
            "changed_action_rows": 20,
            "serve_action_rows": 0,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.002,
            "changed_row_oof_precision": 0.246914,
            "changed_action_rows": 20,
            "serve_action_rows": 0,
        }
    )
    assert not evidence_passes(
        {
            "action_oof_delta": 0.002,
            "changed_row_oof_precision": 0.40,
            "changed_action_rows": 20,
            "serve_action_rows": 1,
        }
    )


def test_build_export_frame_preserves_fixed_v306_point_and_v300_server():
    anchor = pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [1, 5, 9],
            "pointId": [0, 7, 12],
            "serverGetPoint": [0.123456789, 0.5, 0.999999999],
        }
    )
    pred_action = np.array([0, 5, 12])

    out = build_export_frame(anchor, pred_action)

    assert out.columns.tolist() == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert out["actionId"].tolist() == [0, 5, 12]
    assert out["pointId"].tolist() == anchor["pointId"].tolist()
    assert out["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()


def test_protected_output_path_blocks_upload_and_selected_writes():
    spec = ExportSpec(
        filename="submission_v317_zero_terminal_action_budget10__v306point_v300server.csv",
        group="zero_terminal",
        budget=10,
    )

    outdir = Path("v317_action_specialist_ensemble")

    path = protected_output_path(outdir, spec)

    assert path.parent == outdir
    assert "upload_candidates" not in str(path)
    assert "selected" not in str(path)


def test_action_distribution_returns_stable_json():
    text = action_distribution(np.array([12, 0, 12, 5, 0]))

    assert text == '{"0": 2, "5": 1, "12": 2}'
