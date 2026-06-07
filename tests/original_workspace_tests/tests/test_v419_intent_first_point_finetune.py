from pathlib import Path

import pandas as pd

from analysis_v335_moe_anchor_contract import SUBMISSION_COLUMNS
from analysis_v419_intent_first_point_finetune import build_test_rows, package_low_churn, run_pipeline


def test_v419_aligns_test_to_anchor_last_prefix():
    test = pd.DataFrame(
        {
            "rally_uid": ["a", "a", "b"],
            "strikeNumber": [1, 3, 2],
            "actionId": [1, 2, 3],
            "pointId": [4, 5, 6],
        }
    )
    anchor = pd.DataFrame(
        {
            "rally_uid": ["b", "a"],
            "actionId": [0, 0],
            "pointId": [0, 0],
            "serverGetPoint": [0.5, 0.5],
        }
    )

    aligned = build_test_rows(test, anchor)

    assert aligned["rally_uid"].tolist() == ["b", "a"]
    assert aligned["strikeNumber"].tolist() == [2, 3]


def test_v419_packager_blocks_point0_additions():
    anchor = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "actionId": [1, 1],
            "pointId": [5, 4],
            "serverGetPoint": [0.5, 0.5],
        }
    )
    pred = pd.DataFrame(
        {
            "rally_uid": ["r1", "r2"],
            "pred_actionId": [1, 1],
            "pred_pointId": [0, 7],
            "point_confidence": [9.0, 1.0],
        }
    )

    packed, report = package_low_churn(anchor, pred, point_limit=2, action_limit=0)

    assert packed["pointId"].tolist() == [5, 7]
    assert packed["serverGetPoint"].tolist() == [0.5, 0.5]
    assert report["blocked_point0_additions"] == 1


def _tiny_aicup_rows() -> pd.DataFrame:
    rows = []
    for idx, target_action, target_point in [
        (0, 6, 5),
        (1, 7, 6),
        (2, 6, 7),
        (3, 7, 8),
        (4, 6, 5),
        (5, 7, 6),
    ]:
        rally = f"tr{idx}"
        rows.append(
            {
                "rally_uid": rally,
                "match": idx,
                "sex": 1 + (idx % 2),
                "numberGame": 1,
                "strikeNumber": 1,
                "scoreSelf": idx % 5,
                "scoreOther": (idx + 1) % 5,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 2,
                "spinId": 3,
                "pointId": 4,
                "actionId": 2,
                "positionId": 1,
            }
        )
        rows.append(
            {
                "rally_uid": rally,
                "match": idx,
                "sex": 1 + (idx % 2),
                "numberGame": 1,
                "strikeNumber": 2,
                "scoreSelf": idx % 5,
                "scoreOther": (idx + 1) % 5,
                "strikeId": 2,
                "handId": 2,
                "strengthId": 1,
                "spinId": 2,
                "pointId": target_point,
                "actionId": target_action,
                "positionId": 2,
            }
        )
    return pd.DataFrame(rows)


def _tiny_test_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rally_uid": "te0",
                "sex": 1,
                "numberGame": 1,
                "strikeNumber": 1,
                "scoreSelf": 0,
                "scoreOther": 0,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 2,
                "spinId": 3,
                "pointId": 4,
                "actionId": 2,
                "positionId": 1,
            },
            {
                "rally_uid": "te0",
                "sex": 1,
                "numberGame": 1,
                "strikeNumber": 3,
                "scoreSelf": 1,
                "scoreOther": 0,
                "strikeId": 2,
                "handId": 2,
                "strengthId": 1,
                "spinId": 2,
                "pointId": 5,
                "actionId": 6,
                "positionId": 2,
            },
            {
                "rally_uid": "te1",
                "sex": 2,
                "numberGame": 1,
                "strikeNumber": 2,
                "scoreSelf": 2,
                "scoreOther": 1,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 3,
                "spinId": 4,
                "pointId": 4,
                "actionId": 2,
                "positionId": 1,
            },
        ]
    )


def _tiny_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token": [
                "fam=badminton_drive",
                "phase=rally",
                "depth=half",
                "side=left",
                "side=middle",
                "speed=medium",
                "spin=medium",
                "terminal=nonterminal",
            ],
            "emb_0": [0.4, 0.2, 0.6, 0.1, 0.3, 0.5, 0.7, 0.9],
            "emb_1": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6],
        }
    )


def test_v419_pipeline_generates_schema_safe_low_churn_candidates(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test_new.csv"
    token_path = tmp_path / "token_embeddings.csv"
    anchor_path = tmp_path / "anchor.csv"
    outdir = tmp_path / "v419"
    anchor = pd.DataFrame(
        {
            "rally_uid": ["te1", "te0"],
            "actionId": [2, 6],
            "pointId": [4, 5],
            "serverGetPoint": [0.25, 0.75],
        }
    )
    _tiny_aicup_rows().to_csv(train_path, index=False)
    _tiny_test_rows().to_csv(test_path, index=False)
    _tiny_embeddings().to_csv(token_path, index=False)
    anchor.to_csv(anchor_path, index=False)

    report = run_pipeline(
        train_path=train_path,
        test_path=test_path,
        token_embedding_path=token_path,
        anchor_path=anchor_path,
        outdir=outdir,
        expected_rows=len(anchor),
    )

    summary = pd.read_csv(outdir / "candidate_summary.csv")
    assert set(summary["candidate"]) == {"intent_point_top5", "intent_point_top10", "joint_top10"}
    assert report["fallback_used"] is False
    for filename in [
        "submission_v419_intent_point_top5__v362anchor.csv",
        "submission_v419_intent_point_top10__v362anchor.csv",
        "submission_v419_joint_top10__v362anchor.csv",
    ]:
        frame = pd.read_csv(outdir / filename)
        assert list(frame.columns) == SUBMISSION_COLUMNS
        assert frame["rally_uid"].tolist() == ["te1", "te0"]
        assert frame["serverGetPoint"].tolist() == anchor["serverGetPoint"].tolist()
        assert not ((frame["pointId"].astype(int).eq(0)) & (anchor["pointId"].astype(int).ne(0))).any()
    assert (summary["point0_additions"] == 0).all()
