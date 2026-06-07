import numpy as np
import pandas as pd

from analysis_v207_anchor_relative_ttselector import (
    build_anchor_relative_frame,
    select_by_slice_caps,
    transition_policy_mask,
)


def test_build_anchor_relative_frame_targets_improvement_over_anchor():
    rows = pd.DataFrame({"r184_phase": ["receive", "rally", "rally"]})
    anchor = np.array([8, 8, 0])
    candidate = np.array([0, 7, 8])
    truth = np.array([0, 8, 8])
    base_prob = np.full((3, 10), 0.1)
    cand_prob = np.full((3, 10), 0.1)
    frame = build_anchor_relative_frame(rows, anchor, candidate, truth, base_prob, cand_prob)
    assert frame["is_anchor_improvement"].tolist() == [1, 0, 1]
    assert frame["anchor_transition_kind"].tolist() == ["to_point0", "nonterminal", "from_point0"]


def test_transition_policy_mask_blocks_to_point0_when_not_strict():
    frame = pd.DataFrame(
        {
            "anchor_transition_kind": ["to_point0", "from_point0", "nonterminal"],
            "candidate_p0": [0.8, 0.2, 0.1],
            "prob_gain": [0.01, 0.1, 0.1],
        }
    )
    strict = transition_policy_mask(frame, point0_mode="strict")
    loose = transition_policy_mask(frame, point0_mode="loose")
    assert strict.tolist() == [False, True, True]
    assert loose.tolist() == [True, True, True]


def test_select_by_slice_caps_allocates_per_slice():
    labels = np.array([8, 8, 8, 8, 8, 8])
    candidate = np.array([0, 0, 7, 7, 9, 9])
    trust = np.array([0.9, 0.1, 0.8, 0.7, 0.6, 0.5])
    slices = pd.Series(["a", "a", "b", "b", "b", "b"])
    caps = {"a": 0.5, "b": 0.25}
    out, changed = select_by_slice_caps(labels, candidate, trust, slices, caps)
    assert changed.tolist() == [True, False, True, False, False, False]
    assert out.tolist() == [0, 8, 7, 8, 8, 8]
