import numpy as np
import pandas as pd
import pytest

from analysis_v316_nonterminal_point_correction import (
    EXPECTED_COLUMNS,
    build_best_nonterminal_candidates,
    count_point0_changes,
    ensure_local_output_path,
    select_nonterminal_replacements,
    validate_submission_frame,
)


def test_build_best_nonterminal_candidates_ignores_point0_and_base_label():
    base = np.array([7, 8, 4])
    prob = np.array(
        [
            [0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10, 0.70, 0.05],
            [0.50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.45, 0.44, 0.43],
            [0.40, 0.01, 0.01, 0.01, 0.30, 0.41, 0.39, 0.01, 0.01, 0.01],
        ]
    )

    cand, margin = build_best_nonterminal_candidates(base, prob, allowed_targets={4, 5, 6, 7, 8, 9})

    assert cand.tolist() == [8, 7, 5]
    assert np.allclose(margin, [0.60, 0.01, 0.11])
    assert not np.isin(cand, [0]).any()


def test_select_nonterminal_replacements_requires_positive_nonzero_confusion_rows():
    base = np.array([7, 0, 8, 9, 4, 5, 6])
    candidate = np.array([8, 7, 0, 9, 5, 3, 6])
    score = np.array([0.50, 0.99, 0.80, 0.70, 0.40, 0.30, 0.20])

    selected = select_nonterminal_replacements(
        base,
        candidate,
        score,
        budget=4,
        allowed_pairs={(7, 8), (4, 5), (5, 3)},
    )

    assert selected.tolist() == [True, False, False, False, True, False, False]


def test_select_nonterminal_replacements_is_stable_for_ties_and_respects_gate():
    base = np.array([7, 8, 9, 4])
    candidate = np.array([8, 9, 7, 5])
    score = np.array([0.30, 0.30, 0.20, 0.30])
    gate = np.array([True, True, True, False])

    selected = select_nonterminal_replacements(base, candidate, score, budget=2, gate=gate)

    assert selected.tolist() == [True, True, False, False]


def test_count_point0_changes_tracks_additions_and_removals():
    base = np.array([0, 7, 8, 0, 5])
    pred = np.array([0, 8, 0, 4, 6])

    counts = count_point0_changes(base, pred)

    assert counts == {"point0_additions": 1, "point0_removals": 1}


def test_validate_submission_frame_schema_ranges_and_row_count():
    frame = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [0, 18],
            "pointId": [0, 9],
            "serverGetPoint": [0.0, 1.0],
        }
    )

    out = validate_submission_frame(frame, expected_rows=2)

    assert list(out.columns) == EXPECTED_COLUMNS
    with pytest.raises(ValueError, match="columns"):
        validate_submission_frame(frame.assign(extra=1), expected_rows=2)
    with pytest.raises(ValueError, match="pointId"):
        validate_submission_frame(frame.assign(pointId=[0, 10]), expected_rows=2)


def test_ensure_local_output_path_rejects_upload_and_selected_targets():
    assert ensure_local_output_path("v316_nonterminal_point_correction/submission.csv").as_posix().endswith(
        "v316_nonterminal_point_correction/submission.csv"
    )

    with pytest.raises(ValueError, match="local-only"):
        ensure_local_output_path("upload_candidates_20260519/submission.csv")
    with pytest.raises(ValueError, match="local-only"):
        ensure_local_output_path("submissions/selected/submission.csv")
