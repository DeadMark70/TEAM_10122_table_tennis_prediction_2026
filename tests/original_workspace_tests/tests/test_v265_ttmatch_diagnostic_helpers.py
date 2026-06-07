import pandas as pd

from analysis_v265_ttmatch_diagnostic import (
    _dataframe_to_markdown,
    apply_ttmatch_predictions,
    build_next_stroke_lookup,
    majority_vote,
    normalize_ttmatch_columns,
    sequence_key,
)


def test_normalize_ttmatch_columns_renames_typos_without_mutating_input():
    src = pd.DataFrame({"strickNumber": [1], "strickId": [2], "actionId": [3]})

    out = normalize_ttmatch_columns(src)

    assert "strikeNumber" in out.columns
    assert "strikeId" in out.columns
    assert "strickNumber" in src.columns
    assert "strickId" in src.columns


def test_sequence_key_can_include_or_exclude_strike_number():
    group = pd.DataFrame(
        {
            "strikeNumber": [1, 2],
            "strikeId": [10, 20],
            "handId": [1, 2],
            "strengthId": [3, 4],
            "spinId": [5, 6],
            "pointId": [7, 8],
            "actionId": [9, 10],
            "positionId": [11, 12],
        }
    )

    strict = sequence_key(group, include_strike_number=True)
    nostrike = sequence_key(group, include_strike_number=False)

    assert strict[0] == (1, 10, 1, 3, 5, 7, 9, 11)
    assert nostrike[0] == (10, 1, 3, 5, 7, 9, 11)
    assert len(strict[0]) == len(nostrike[0]) + 1


def test_majority_vote_tie_breaks_to_smallest_value():
    assert majority_vote([3, 2, 3, 2]) == 2
    assert majority_vote([4, 4, 1]) == 4


def test_build_next_stroke_lookup_aggregates_duplicate_prefixes():
    ttmatch_train = pd.DataFrame(
        [
            _row(10, 1, 101, 1, 8, 0),
            _row(10, 2, 102, 2, 9, 1),
            _row(20, 1, 101, 1, 8, 0),
            _row(20, 2, 102, 3, 7, 0),
            _row(30, 1, 101, 1, 8, 0),
            _row(30, 2, 102, 2, 9, 1),
        ]
    )

    lookup = build_next_stroke_lookup(ttmatch_train, include_strike_number=True)
    key = sequence_key(ttmatch_train[ttmatch_train["rally_uid"] == 10].iloc[:1], True)

    assert lookup[key]["support"] == 3
    assert lookup[key]["actionId"] == 2
    assert lookup[key]["pointId"] == 9
    assert lookup[key]["serverGetPoint"] == 1


def test_apply_ttmatch_predictions_uses_strict_then_nostrike_then_fallback():
    fallback = pd.DataFrame(
        {
            "rally_uid": [100, 200, 300],
            "actionId": [1, 1, 1],
            "pointId": [8, 8, 8],
            "serverGetPoint": [0.25, 0.50, 0.75],
        }
    )
    test_new = pd.DataFrame(
        [
            _row(100, 1, 101, 1, 8, 0),
            _row(200, 99, 101, 1, 8, 0),
            _row(300, 1, 999, 1, 8, 0),
        ]
    )
    strict_key = sequence_key(test_new[test_new["rally_uid"] == 100], True)
    nostrike_key = sequence_key(test_new[test_new["rally_uid"] == 200], False)
    strict_lookup = {strict_key: {"actionId": 2, "pointId": 3, "serverGetPoint": 1, "support": 4}}
    nostrike_lookup = {nostrike_key: {"actionId": 4, "pointId": 5, "serverGetPoint": 0, "support": 2}}

    pred, coverage = apply_ttmatch_predictions(test_new, strict_lookup, nostrike_lookup, fallback)

    assert pred["actionId"].tolist() == [2, 4, 1]
    assert pred["pointId"].tolist() == [3, 5, 8]
    assert pred["serverGetPoint"].tolist() == [1.0, 0.0, 0.75]
    assert coverage["match_type"].tolist() == ["strict", "nostrike", "none"]


def test_dataframe_to_markdown_does_not_require_optional_dependencies():
    table = _dataframe_to_markdown(pd.DataFrame({"candidate": ["a"], "rows": [1845]}))

    assert "| candidate | rows |" in table
    assert "| a | 1845 |" in table


def _row(rally_uid, strike_number, strike_id, action_id, point_id, server_get_point):
    return {
        "rally_uid": rally_uid,
        "strikeNumber": strike_number,
        "strikeId": strike_id,
        "handId": 1,
        "strengthId": 1,
        "spinId": 1,
        "pointId": point_id,
        "actionId": action_id,
        "positionId": 1,
        "serverGetPoint": server_get_point,
    }
