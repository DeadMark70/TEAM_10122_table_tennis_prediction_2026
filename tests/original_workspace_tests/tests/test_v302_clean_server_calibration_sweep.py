import numpy as np
import pandas as pd
import pytest

from analysis_v302_clean_server_calibration_sweep import (
    apply_temperature,
    build_server_only_output,
    shrink_to_anchor,
    validate_submission_frame,
)


def test_shrink_to_anchor_moves_v300_server_toward_v261_anchor():
    v300 = np.array([0.2, 0.8, 0.5])
    anchor = np.array([0.4, 0.4, 0.1])

    shrunk = shrink_to_anchor(v300, anchor, strength=0.25)

    assert shrunk.tolist() == pytest.approx([0.25, 0.7, 0.4])


def test_temperature_transform_is_finite_in_range_and_directional():
    server = np.array([0.0, 0.25, 0.5, 0.75, 1.0, np.nan, np.inf, -np.inf])

    sharpened = apply_temperature(server, temperature=0.9)
    softened = apply_temperature(server, temperature=1.1)

    assert np.isfinite(sharpened).all()
    assert np.isfinite(softened).all()
    assert ((sharpened > 0.0) & (sharpened < 1.0)).all()
    assert ((softened > 0.0) & (softened < 1.0)).all()
    assert sharpened[1] < 0.25
    assert sharpened[3] > 0.75
    assert softened[1] > 0.25
    assert softened[3] < 0.75


def test_build_server_only_output_preserves_schema_action_and_point():
    base = pd.DataFrame(
        {
            "rally_uid": [101, 102, 103],
            "actionId": [4, 10, 13],
            "pointId": [8, 5, 0],
            "serverGetPoint": [0.2, 0.5, 0.8],
        }
    )
    server = np.array([0.3, 0.4, 0.9])

    output = build_server_only_output(base, server)

    assert list(output.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert output[["rally_uid", "actionId", "pointId"]].equals(
        base[["rally_uid", "actionId", "pointId"]]
    )
    assert output["serverGetPoint"].tolist() == pytest.approx([0.3, 0.4, 0.9])
    validate_submission_frame(output, expected_rows=3)
