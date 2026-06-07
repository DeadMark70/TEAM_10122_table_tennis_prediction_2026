import numpy as np
import pandas as pd

from analysis_v255_external_pretraining_helpers import (
    action_family_from_id,
    canonical_phase_from_event,
    normalize_rows_safe,
    parse_vector_string,
    safe_family_prior_to_action_prob,
)


def test_parse_vector_string():
    assert parse_vector_string("[1 2 3]") == [1.0, 2.0, 3.0]
    assert parse_vector_string("[1, 2, 3]") == [1.0, 2.0, 3.0]
    assert parse_vector_string("bad") == []


def test_action_family_from_id():
    assert action_family_from_id(0) == "Zero"
    assert action_family_from_id(1) == "Attack"
    assert action_family_from_id(8) == "Control"
    assert action_family_from_id(12) == "Defensive"
    assert action_family_from_id(15) == "Serve"


def test_canonical_phase_from_event():
    assert canonical_phase_from_event("shot_p1", 1) == "receive_like"
    assert canonical_phase_from_event("bounce_p2", 2) == "rally_like"
    assert canonical_phase_from_event("net_p1", 4) == "terminal_like"


def test_normalize_rows_safe_no_nan():
    x = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    out = normalize_rows_safe(x)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()


def test_safe_family_prior_to_action_prob():
    prior = pd.DataFrame(
        {
            "Zero": [0.1],
            "Attack": [0.4],
            "Control": [0.3],
            "Defensive": [0.1],
            "Serve": [0.1],
        }
    )
    prob = safe_family_prior_to_action_prob(prior)
    assert prob.shape == (1, 19)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert prob[0, 15:19].sum() <= 0.1 + 1e-9
