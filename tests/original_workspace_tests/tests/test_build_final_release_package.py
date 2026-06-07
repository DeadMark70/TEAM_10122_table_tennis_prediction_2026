from pathlib import Path

import pandas as pd

from tools.build_final_release_package import build_release_package


def test_build_release_package_copies_final_submission_and_docs(tmp_path: Path):
    root = Path.cwd()
    out = tmp_path / "release_final"

    report = build_release_package(root=root, outdir=out)

    final_submission = (
        out
        / "artifacts"
        / "final_submission"
        / "submission_v362_depth_agree_only__v173action_v300server.csv"
    )
    assert final_submission.exists()
    df = pd.read_csv(final_submission)
    assert list(df.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert len(df) == 1845
    assert df["serverGetPoint"].between(0, 1).all()
    assert (out / "README.md").exists()
    assert (out / "docs" / "external_resources.md").exists()
    assert any(
        "submission_v362_depth_agree_only__v173action_v300server.csv" in path
        for path in report["copied_files"]
    )


def test_build_release_package_does_not_copy_raw_competition_data(tmp_path: Path):
    root = Path.cwd()
    out = tmp_path / "release_final"

    build_release_package(root=root, outdir=out)

    forbidden = {"train.csv", "test_new.csv", "test_old.csv"}
    copied_names = {p.name for p in out.rglob("*") if p.is_file()}
    assert forbidden.isdisjoint(copied_names)
