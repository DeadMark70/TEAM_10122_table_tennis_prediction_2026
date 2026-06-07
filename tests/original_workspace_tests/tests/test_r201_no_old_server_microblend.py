import numpy as np

from analysis_r201_no_old_server_microblend import blend_server, server_mad


def test_blend_server_clips_to_unit_interval():
    out = blend_server(np.array([0.0, 1.0]), [np.array([2.0, -1.0])], [0.5])
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_server_mad_is_mean_absolute_difference():
    assert server_mad(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.5
