import numpy as np

from analysis_v312_action_weak_complementarity import (
    DEFAULT_WEAK_GROUPS,
    changed_row_precision,
    decision_label,
    markdown_table,
    weak_group_masks,
)


def test_weak_group_masks_match_default_focus_groups():
    labels = np.array([0, 1, 3, 5, 7, 8, 9, 12, 14, 18])

    masks = weak_group_masks(labels)

    assert set(masks) == set(DEFAULT_WEAK_GROUPS)
    assert masks["terminal_03"].tolist() == [True, False, True, False, False, False, False, False, False, False]
    assert masks["fast_attack_57"].tolist() == [False, False, False, True, True, False, False, False, False, False]
    assert masks["style_control_89"].tolist() == [False, False, False, False, False, True, True, False, False, False]
    assert masks["defensive_1214"].tolist() == [False, False, False, False, False, False, False, True, True, False]


def test_changed_row_precision_counts_only_action_edits():
    y = np.array([0, 3, 5, 7, 8, 9])
    anchor = np.array([1, 3, 5, 0, 8, 2])
    candidate = np.array([0, 3, 7, 7, 8, 9])

    report = changed_row_precision(y, anchor, candidate)

    assert report["changed_rows"] == 4
    assert report["changed_correct"] == 3
    assert report["changed_precision"] == 0.75


def test_changed_row_precision_handles_no_edits():
    y = np.array([0, 3])
    anchor = np.array([0, 1])

    report = changed_row_precision(y, anchor, anchor)

    assert report["changed_rows"] == 0
    assert report["changed_correct"] == 0
    assert report["changed_precision"] == 0.0


def test_decision_label_uses_v312_thresholds():
    assert decision_label(0.0030, 80) == "REVIEW_AGGRESSIVE"
    assert decision_label(0.0029, 30) == "REVIEW_ACTION"
    assert decision_label(0.0015, 30) == "REVIEW_ACTION"
    assert decision_label(0.0030, 81) == "DO_NOT_UPLOAD"
    assert decision_label(0.0015, 31) == "DO_NOT_UPLOAD"
    assert decision_label(0.00149, 30) == "DO_NOT_UPLOAD"


def test_markdown_table_does_not_require_optional_tabulate():
    rows = [
        {"candidate": "a", "delta": 0.001234, "decision": "DO_NOT_UPLOAD"},
        {"candidate": "b", "delta": 0.010000, "decision": "REVIEW_ACTION"},
    ]

    text = markdown_table(rows, ["candidate", "delta", "decision"])

    assert text.splitlines()[0] == "| candidate | delta | decision |"
    assert "| --- | --- | --- |" in text
    assert "| a | 0.001234 | DO_NOT_UPLOAD |" in text
