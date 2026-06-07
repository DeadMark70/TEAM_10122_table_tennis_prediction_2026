import numpy as np
import pandas as pd
import pytest

from analysis_v319_clean_server_value_state import (
    ServerCandidate,
    _fit_oof_and_test_predictions,
    blend_to_target_mad,
    build_packaged_submission,
    build_value_state_features,
    decision_for_candidate,
    rank_normalize_to_anchor,
    summarize_candidate,
)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": [101, 102, 103, 104],
            "actionId": [3, 7, 12, 18],
            "pointId": [0, 8, 4, 0],
            "serverGetPoint": [0.20, 0.80, 0.50, 0.30],
        }
    )


def test_build_packaged_submission_is_server_only():
    anchor = _anchor()
    candidate = ServerCandidate(
        key="unit",
        server=np.array([0.21, 0.81, 0.51, 0.31]),
        source="unit_signal",
        oof_auc=0.62,
        anchor_auc=0.60,
        target_mad=0.01,
    )

    packaged = build_packaged_submission(anchor, candidate, expected_rows=4)

    assert packaged["rally_uid"].tolist() == anchor["rally_uid"].tolist()
    assert packaged["actionId"].tolist() == anchor["actionId"].tolist()
    assert packaged["pointId"].tolist() == anchor["pointId"].tolist()
    assert packaged["serverGetPoint"].tolist() == pytest.approx([0.21, 0.81, 0.51, 0.31])


def test_rank_normalize_and_target_mad_blend_are_tiny_and_ordered():
    anchor = np.array([0.2, 0.8, 0.5, 0.3])
    value_signal = np.array([40.0, 10.0, 30.0, 20.0])

    ranked = rank_normalize_to_anchor(value_signal, anchor)
    blended = blend_to_target_mad(anchor, ranked, target_mad=0.05)

    assert ranked.tolist() == pytest.approx([0.8, 0.2, 0.5, 0.3])
    assert float(np.mean(np.abs(blended - anchor))) == pytest.approx(0.05)
    assert blended[0] > anchor[0]
    assert blended[1] < anchor[1]
    assert blended[2] == pytest.approx(anchor[2])


def test_target_mad_blend_rejects_bad_inputs_and_handles_zero_delta():
    anchor = np.array([0.2, 0.8, 0.5])

    assert blend_to_target_mad(anchor, anchor, target_mad=0.005).tolist() == pytest.approx(anchor.tolist())
    with pytest.raises(ValueError, match="target_mad"):
        blend_to_target_mad(anchor, anchor, target_mad=-0.1)
    with pytest.raises(ValueError, match="length"):
        blend_to_target_mad(anchor, np.array([0.1, 0.2]), target_mad=0.005)


def test_value_state_features_include_prefix_score_phase_and_lags():
    frame = pd.DataFrame(
        {
            "rally_uid": [1, 1, 1, 2],
            "strikeNumber": [1, 2, 3, 1],
            "scoreSelf": [3, 3, 4, 10],
            "scoreOther": [2, 2, 2, 8],
            "actionId": [15, 10, 4, 15],
            "pointId": [9, 5, 8, 4],
            "positionId": [1, 0, 1, 1],
            "serverGetPoint": [0, 0, 1, 1],
        }
    )

    features, columns = build_value_state_features(frame)

    assert "prefix_len" in columns
    assert "score_margin" in columns
    assert "phase_code" in columns
    assert "lag1_actionId" in columns
    assert "anchor_actionId" in columns
    assert features.loc[0, "prefix_len"] == 1
    assert features.loc[2, "prefix_len"] == 3
    assert features.loc[1, "lag1_actionId"] == 15
    assert features.loc[0, "lag1_actionId"] == -1
    assert features.loc[2, "score_margin"] == 2


def test_oof_fitter_uses_single_thread_safe_logistic_model():
    x = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    x_test = pd.DataFrame({"a": [1, 4], "b": [0, 1]})

    oof, pred, auc, model_name = _fit_oof_and_test_predictions(x, y, x_test, random_state=319)

    assert model_name == "LogisticRegression"
    assert len(oof) == len(y)
    assert len(pred) == len(x_test)
    assert np.isfinite(auc)
    assert ((pred > 0.0) & (pred < 1.0)).all()


def test_summary_and_decision_require_server_only_auc_gain_and_mad_limit():
    anchor = _anchor()
    candidate = ServerCandidate(
        key="unit",
        server=np.array([0.21, 0.81, 0.51, 0.31]),
        source="unit_signal",
        oof_auc=0.62,
        anchor_auc=0.60,
        target_mad=0.01,
    )
    packaged = build_packaged_submission(anchor, candidate, expected_rows=4)

    row = summarize_candidate(
        candidate,
        anchor,
        packaged,
        "v319_clean_server_value_state/submission.csv",
    )

    assert row["server_mad_vs_v306_server"] == pytest.approx(0.01)
    assert row["oof_auc_delta_vs_anchor"] == pytest.approx(0.02)
    assert row["action_changed_rows_vs_anchor"] == 0
    assert row["point_changed_rows_vs_anchor"] == 0
    assert row["decision"] == "REVIEW_SERVER"

    assert decision_for_candidate(oof_auc=0.61, anchor_auc=0.60, mad=0.01, server_min=0.2, server_max=0.8) == "REVIEW_SERVER"
    assert decision_for_candidate(oof_auc=0.60, anchor_auc=0.60, mad=0.01, server_min=0.2, server_max=0.8) == "DIAGNOSTIC"
    assert decision_for_candidate(oof_auc=0.61, anchor_auc=0.60, mad=0.011, server_min=0.2, server_max=0.8) == "DIAGNOSTIC"
    assert decision_for_candidate(oof_auc=0.61, anchor_auc=0.60, mad=0.01, server_min=0.5, server_max=0.5) == "DIAGNOSTIC"
