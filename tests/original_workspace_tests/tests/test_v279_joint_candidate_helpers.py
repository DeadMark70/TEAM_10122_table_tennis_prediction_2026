import numpy as np
import pandas as pd

from analysis_v279_joint_action_point_candidate_pool import (
    action_family,
    point_depth,
    is_pair_compatible,
    expected_pair_columns,
)


def test_action_family_and_point_depth():
    assert action_family(0) == "Zero"
    assert all(action_family(i) == "Attack" for i in range(1, 8))
    assert all(action_family(i) == "Control" for i in range(8, 12))
    assert all(action_family(i) == "Defensive" for i in range(12, 15))
    assert all(action_family(i) == "Serve" for i in range(15, 19))
    assert point_depth(0) == 0
    assert [point_depth(i) for i in range(1, 10)] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_pair_compatibility_terminal_and_serve_rules():
    assert is_pair_compatible(0, 0)
    assert not is_pair_compatible(0, 8)
    assert not is_pair_compatible(15, 0)
    assert not is_pair_compatible(18, 9)
    assert is_pair_compatible(3, 0)
    assert is_pair_compatible(10, 2)
    assert is_pair_compatible(12, 8)


def test_candidate_table_schema_is_stable():
    cols = expected_pair_columns()
    required = {
        "rally_uid",
        "candidate_action",
        "candidate_point",
        "action_source",
        "point_source",
        "pair_key",
        "compatibility_score",
        "action_changed",
        "point_changed",
    }
    assert required.issubset(set(cols))
