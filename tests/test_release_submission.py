from pathlib import Path

import pandas as pd


def test_final_submission_schema_and_range():
    path = Path("artifacts/final_submission/submission_v362_depth_agree_only__v173action_v300server.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert len(df) == 1845
    assert df.isna().sum().sum() == 0
    assert df["serverGetPoint"].between(0, 1).all()
    assert df["actionId"].between(0, 18).all()
    assert df["pointId"].between(0, 9).all()
