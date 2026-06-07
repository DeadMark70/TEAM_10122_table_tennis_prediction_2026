import pandas as pd

from analysis_v364_specialist_distill_packager import (
    candidate_name_allowed,
    prediction_signature,
)


def test_dedupe_equivalent_submissions_by_predictions():
    a = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "actionId": [10, 11],
            "pointId": [8, 9],
            "serverGetPoint": [0.4, 0.6],
        }
    )
    b = a.copy()

    assert prediction_signature(a) == prediction_signature(b)


def test_blocks_old_server_and_ttmatch_named_candidates():
    assert candidate_name_allowed("submission_v364_clean.csv") is True
    assert candidate_name_allowed("submission_v364_ttmatch_probe.csv") is False
    assert candidate_name_allowed("submission_v364_oldserver_probe.csv") is False
