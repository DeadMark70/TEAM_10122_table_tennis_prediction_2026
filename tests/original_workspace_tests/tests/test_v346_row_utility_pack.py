import pandas as pd

from analysis_v346_row_utility_pack import new_rows_beyond_v338, point0_additions, slug


def test_point0_additions_counts_nonzero_to_zero():
    assert point0_additions(pd.Series([8, 0, 7]), pd.Series([0, 0, 9])) == 1


def test_new_rows_beyond_v338_counts_only_new_candidate_edits():
    v306 = pd.Series([8, 8, 8, 8])
    v338 = pd.Series([7, 8, 9, 8])
    cand = pd.Series([7, 6, 8, 4])
    assert new_rows_beyond_v338(v306, v338, cand) == 2


def test_slug_sanitizes_filename_component():
    assert slug("V344 k=18 / point0") == "v344_k_18_point0"
