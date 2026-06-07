import json
import uuid
from pathlib import Path

import pandas as pd

from analysis_v347_v338_v341_diff_audit import (
    build_row_diff,
    build_slice_summary,
    build_transition_summary,
    discover_v341_paths,
    load_candidate_bank_counts,
    run_pipeline,
)


def _submission(points):
    return pd.DataFrame(
        {
            "rally_uid": ["r0", "r1", "r2"],
            "actionId": [1, 1, 1],
            "pointId": points,
            "serverGetPoint": [0.1, 0.2, 0.3],
        }
    )


def _case_dir(name: str) -> Path:
    path = Path("v347_v338_v341_diff_audit") / "unit_test_workspace" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_build_row_diff_groups_exact_transitions_and_flags():
    base = _submission([8, 4, 6])
    v338 = _submission([7, 4, 6])
    v341_a = _submission([7, 5, 6])
    v341_b = _submission([7, 6, 0])
    v345 = _submission([8, 5, 6])
    v344 = _submission([8, 4, 0])
    bank_counts = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "old_point": [8, 4, 6],
            "new_point": [7, 5, 0],
            "source_count_from_candidate_bank": [3, 2, 4],
        }
    )

    out = build_row_diff(
        base,
        [
            ("v338", "v338.csv", v338),
            ("v341", "v341_a.csv", v341_a),
            ("v341", "v341_b.csv", v341_b),
            ("v345_b36", "v345.csv", v345),
            ("v344_k12", "v344.csv", v344),
        ],
        bank_counts=bank_counts,
    )

    row0 = out[(out["row_id"] == 0) & (out["transition"] == "8->7")].iloc[0]
    assert bool(row0["in_v338"]) is True
    assert bool(row0["in_v341"]) is True
    assert int(row0["v341_candidate_count"]) == 2
    assert int(row0["source_count_from_candidate_bank"]) == 3

    row1_45 = out[(out["row_id"] == 1) & (out["transition"] == "4->5")].iloc[0]
    assert bool(row1_45["in_v341"]) is True
    assert bool(row1_45["in_v345_b36"]) is True
    assert bool(row1_45["same_depth"]) is True

    row2 = out[(out["row_id"] == 2) & (out["transition"] == "6->0")].iloc[0]
    assert bool(row2["in_v344_k12"]) is True
    assert bool(row2["same_depth"]) is False
    assert int(row2["new_depth"]) == -1


def test_summaries_are_stable_for_empty_and_nonempty():
    empty = build_transition_summary(pd.DataFrame())
    assert list(empty.columns)[0] == "transition"

    row_diff = pd.DataFrame(
        {
            "row_id": [0, 1],
            "transition": ["8->7", "4->0"],
            "in_v338": [True, False],
            "in_v341": [True, False],
            "in_v345_b36": [False, False],
            "in_v344_k12": [False, True],
            "same_depth": [True, False],
            "old_point": [8, 4],
            "new_point": [7, 0],
            "source_count_from_candidate_bank": [2, 1],
        }
    )
    transition = build_transition_summary(row_diff)
    assert transition["rows"].sum() == 2
    slices = build_slice_summary(row_diff)
    assert set(slices["slice"]) == {"v338|v341", "v344_k12"}


def test_load_candidate_bank_counts_counts_unique_sources():
    case_dir = _case_dir("bank_counts")
    path = case_dir / "candidate_bank.csv"
    pd.DataFrame(
        {
            "row_id": [1, 1, 1],
            "anchor_value": [8, 8, 8],
            "candidate_value": [7, 7, 6],
            "source": ["a", "a", "b"],
        }
    ).to_csv(path, index=False)
    out = load_candidate_bank_counts(path)
    count = out[(out["row_id"] == 1) & (out["new_point"] == 7)]["source_count_from_candidate_bank"].iloc[0]
    assert int(count) == 1


def test_discover_v341_paths_skips_banned_names():
    case_dir = _case_dir("discover")
    good = case_dir / "submission_good.csv"
    banned_old = case_dir / "submission_old_server.csv"
    banned_tt = case_dir / "submission_ttmatch.csv"
    for path in (good, banned_old, banned_tt):
        path.write_text("x", encoding="utf-8")
    assert discover_v341_paths(case_dir) == [good]


def test_run_pipeline_writes_reports_only(monkeypatch):
    root = _case_dir("pipeline")
    v306 = root / "v306.csv"
    v338 = root / "v338.csv"
    v345 = root / "v345.csv"
    v344 = root / "v344.csv"
    v341_dir = root / "v341"
    v341_dir.mkdir()
    v341 = v341_dir / "submission_v341.csv"

    _submission([8, 4, 6]).to_csv(v306, index=False)
    _submission([7, 4, 6]).to_csv(v338, index=False)
    _submission([7, 5, 6]).to_csv(v341, index=False)
    _submission([8, 5, 6]).to_csv(v345, index=False)
    _submission([8, 4, 0]).to_csv(v344, index=False)

    import analysis_v347_v338_v341_diff_audit as mod

    monkeypatch.setattr(mod, "V306_ANCHOR", v306)
    monkeypatch.setattr(mod, "V338_PUBLIC_POSITIVE", v338)
    monkeypatch.setattr(mod, "V341_DIR", v341_dir)
    monkeypatch.setattr(mod, "V345_B36", v345)
    monkeypatch.setattr(mod, "V344_K12", v344)
    monkeypatch.setattr(mod, "V343_BANK", root / "missing_bank.csv")
    monkeypatch.setattr(mod, "TEST_FEATURES", root / "missing_features.csv")

    outdir = root / "out"
    report = run_pipeline(outdir=outdir, expected_rows=3)
    assert report["decision"] == "REPORTS_EXPORTED"
    assert (outdir / "row_diff.csv").exists()
    assert (outdir / "transition_summary.csv").exists()
    assert (outdir / "slice_summary.csv").exists()
    payload = json.loads((outdir / "search_report.json").read_text(encoding="utf-8"))
    assert payload["policy"]["submission_exports"] is False
    assert not list(outdir.glob("submission*.csv"))
