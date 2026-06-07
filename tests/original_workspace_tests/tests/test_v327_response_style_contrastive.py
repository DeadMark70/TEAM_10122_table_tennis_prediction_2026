import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_v327_response_style_contrastive as v327


def scratch_dir(name: str) -> Path:
    path = v327.OUTDIR / f"pytest_{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_load_event_table_prefers_v326_and_excludes_ttmatch():
    root = scratch_dir("root")
    v326 = root / "v326_masked_family_pretrain"
    v326.mkdir()
    pd.DataFrame(
        [
            {
                "corpus": "OpenTT",
                "sequence_id": "a",
                "step_idx": 0,
                "phase_code": "serve_like",
                "coarse_family": "Serve",
                "landing_depth": "short",
                "landing_side": "center",
                "has_spin": 0,
                "has_speed": 0,
                "source_weight": 1.0,
                "source_path": "external_data/openttgames/ok.csv",
            },
            {
                "corpus": "TTMATCH",
                "sequence_id": "bad",
                "step_idx": 0,
                "phase_code": "rally_like",
                "coarse_family": "Attack",
                "landing_depth": "long",
                "landing_side": "left",
                "has_spin": 0,
                "has_speed": 0,
                "source_weight": 1.0,
                "source_path": "external_data/TTMATCH/banned.csv",
            },
        ]
    ).to_csv(v326 / "v326_external_event_table.csv", index=False)

    events, source = v327.load_event_table(root)

    assert source == "v326_csv"
    assert len(events) == 1
    assert events.iloc[0]["coarse_family"] == "Serve"
    assert not events.astype(str).agg("|".join, axis=1).str.upper().str.contains("TTMATCH").any()


def test_response_pairs_and_context_matrix_have_smoothed_probabilities():
    events = pd.DataFrame(
        [
            {"corpus": "OpenTT", "sequence_id": "s1", "step_idx": 0, "phase_code": "serve_like", "coarse_family": "Serve", "landing_depth": "short", "landing_side": "left"},
            {"corpus": "OpenTT", "sequence_id": "s1", "step_idx": 1, "phase_code": "receive_like", "coarse_family": "Attack", "landing_depth": "long", "landing_side": "right"},
            {"corpus": "OpenTT", "sequence_id": "s2", "step_idx": 0, "phase_code": "serve_like", "coarse_family": "Serve", "landing_depth": "short", "landing_side": "left"},
            {"corpus": "OpenTT", "sequence_id": "s2", "step_idx": 1, "phase_code": "receive_like", "coarse_family": "Control", "landing_depth": "mid", "landing_side": "center"},
        ]
    )

    pairs = v327.build_response_pairs(events)
    model = v327.build_response_style_model(pairs, n_components=3)

    assert list(pairs["incoming_family"]) == ["Serve", "Serve"]
    assert set(model.matrix.columns) == set(v327.FAMILY_CLASSES)
    row = model.matrix.loc["Serve|serve_like|short|left"]
    assert np.isclose(float(row.sum()), 1.0)
    assert float(row["Attack"]) > float(row["Serve"])
    assert model.embeddings.shape[0] == 1


def test_project_aicup_prefixes_uses_only_observed_prefix_context():
    events = pd.DataFrame(
        [
            {"corpus": "OpenTT", "sequence_id": "s1", "step_idx": 0, "phase_code": "serve_like", "coarse_family": "Serve", "landing_depth": "missing", "landing_side": "left"},
            {"corpus": "OpenTT", "sequence_id": "s1", "step_idx": 1, "phase_code": "receive_like", "coarse_family": "Attack", "landing_depth": "missing", "landing_side": "missing"},
        ]
    )
    model = v327.build_response_style_model(v327.build_response_pairs(events), n_components=2)
    prefixes = pd.DataFrame(
        [
            {
                "rally_uid": 10,
                "match": 1,
                "prefix_len": 1,
                "lag0_actionId": 15,
                "lag0_positionId": 1,
                "lag0_spinId": 3,
                "lag0_strengthId": 2,
            }
        ]
    )

    features, coverage = v327.project_prefix_features(prefixes, model, split="train")

    assert features.loc[0, "split"] == "train"
    assert features.loc[0, "v327_context_key"] == "Serve|serve_like|missing|left"
    assert features.loc[0, "v327_context_covered"] == 1
    assert coverage["covered_rows"] == 1
    assert "next_actionId" not in features.columns


def test_write_outputs_creates_expected_artifacts():
    outdir = scratch_dir("outputs")
    events = pd.DataFrame(
        [
            {"corpus": "OpenTT", "sequence_id": "s1", "step_idx": 0, "phase_code": "serve_like", "coarse_family": "Serve", "landing_depth": "missing", "landing_side": "missing"},
            {"corpus": "OpenTT", "sequence_id": "s1", "step_idx": 1, "phase_code": "receive_like", "coarse_family": "Attack", "landing_depth": "missing", "landing_side": "missing"},
        ]
    )
    model = v327.build_response_style_model(v327.build_response_pairs(events), n_components=2)
    train_features = pd.DataFrame({"split": ["train"], "rally_uid": [1], "prefix_len": [1], "v327_context_key": ["Serve|serve_like|missing|missing"], "v327_context_covered": [1]})
    test_features = pd.DataFrame({"split": ["test"], "rally_uid": [2], "prefix_len": [1], "v327_context_key": ["Serve|serve_like|missing|missing"], "v327_context_covered": [1]})
    report = v327.build_report(
        events=events,
        pairs=v327.build_response_pairs(events),
        model=model,
        source="unit",
        train_coverage={"rows": 1, "covered_rows": 1, "coverage_rate": 1.0},
        test_coverage={"rows": 1, "covered_rows": 1, "coverage_rate": 1.0},
    )

    v327.write_outputs(outdir, train_features, test_features, model, report)

    expected = {
        "v327_aicup_response_style_features.csv",
        "v327_context_response_matrix.csv",
        "v327_report.json",
        "v327_report.md",
    }
    assert expected.issubset({p.name for p in outdir.iterdir()})
    payload = json.loads((outdir / "v327_report.json").read_text(encoding="utf-8"))
    assert payload["submissions_written"] == 0
