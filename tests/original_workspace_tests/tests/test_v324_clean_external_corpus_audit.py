import json
from pathlib import Path

import pandas as pd

import analysis_v324_clean_external_corpus_audit as v324


def test_ttmatch_is_banned_and_never_content_readable():
    policy = v324.resource_policy("TTMATCH")

    assert policy.status == "banned"
    assert not policy.content_readable
    assert not v324.should_read_content(Path("external_data/TTMATCH/train.csv"))
    assert v324.resource_policy("openttgames").status == "clean"


def test_schema_inventory_reads_allowed_headers_and_skips_banned_content():
    external = v324.ROOT / "v324_clean_external_corpus_audit" / "test_workspace" / "external_data"
    opentt = external / "openttgames" / "processed"
    banned = external / "TTMATCH"
    opentt.mkdir(parents=True, exist_ok=True)
    banned.mkdir(parents=True, exist_ok=True)
    (opentt / "openttgames_events.csv").write_text(
        "video_id,event_type,safe_action_family,is_rally_ending,player_side\n"
        "g1,stroke,Attack,0,left\n",
        encoding="utf-8",
    )
    (banned / "train.csv").write_text("actionId,pointId\n1,2\n", encoding="utf-8")

    inventory, schemas = v324.build_schema_inventory(external)

    assert set(inventory["resource"]) == {"openttgames", "TTMATCH"}
    banned_row = schemas[schemas["resource"].eq("TTMATCH")].iloc[0]
    assert not bool(banned_row["content_read"])
    assert banned_row["columns"] == ""
    allowed_row = schemas[schemas["resource"].eq("openttgames")].iloc[0]
    assert bool(allowed_row["content_read"])
    assert "safe_action_family" in allowed_row["columns"].split("|")
    assert int(allowed_row["sample_rows"]) == 1


def test_feature_summary_captures_family_phase_and_availability():
    canonical = pd.DataFrame(
        [
            {
                "source_dataset": "openttgames",
                "sequence_id": "a",
                "coarse_family": "Attack",
                "phase": "serve_like",
                "terminal_like": False,
                "landing_x": 1.0,
                "landing_y": 2.0,
                "speed": 5.0,
                "spin": None,
                "risk_tier": "GREEN",
            },
            {
                "source_dataset": "openttgames",
                "sequence_id": "a",
                "coarse_family": "Control",
                "phase": "rally_like",
                "terminal_like": False,
                "landing_x": None,
                "landing_y": 3.0,
                "speed": 6.0,
                "spin": 9.0,
                "risk_tier": "GREEN",
            },
            {
                "source_dataset": "CoachAI-Projects-main",
                "sequence_id": "b",
                "coarse_family": "Defensive",
                "phase": "rally_like",
                "terminal_like": True,
                "landing_x": None,
                "landing_y": None,
                "speed": None,
                "spin": None,
                "risk_tier": "YELLOW",
            },
        ]
    )

    summary = v324.summarize_canonical_features(canonical)["resource_summary"]
    opentt = summary[summary["resource"].eq("openttgames")].iloc[0]

    assert int(opentt["canonical_rows"]) == 2
    assert int(opentt["sequences"]) == 1
    assert opentt["action_family_counts_json"] == json.dumps(
        {"Attack": 1, "Control": 1}, sort_keys=True
    )
    assert opentt["phase_counts_json"] == json.dumps(
        {"rally_like": 1, "serve_like": 1}, sort_keys=True
    )
    assert float(opentt["landing_depth_available_rate"]) == 1.0
    assert float(opentt["landing_side_available_rate"]) == 0.5
    assert float(opentt["spin_available_rate"]) == 0.5


def test_ranked_recommendations_name_next_trainable_scripts():
    summary = pd.DataFrame(
        [
            {
                "resource": "openttgames",
                "status": "clean",
                "canonical_rows": 5000,
                "sequences": 30,
                "action_family_available_rate": 1.0,
                "phase_available_rate": 1.0,
                "landing_depth_available_rate": 0.0,
                "landing_side_available_rate": 1.0,
                "spin_available_rate": 0.0,
                "speed_available_rate": 0.0,
                "schema_compatibility": "high",
            },
            {
                "resource": "CoachAI-Projects-main",
                "status": "coarse_only",
                "canonical_rows": 12000,
                "sequences": 80,
                "action_family_available_rate": 1.0,
                "phase_available_rate": 1.0,
                "landing_depth_available_rate": 0.7,
                "landing_side_available_rate": 0.7,
                "spin_available_rate": 0.0,
                "speed_available_rate": 0.0,
                "schema_compatibility": "medium",
            },
        ]
    )

    ranked = v324.rank_recommendations(summary)

    assert ranked.iloc[0]["experiment"] == "masked family pretrain"
    assert ranked.iloc[0]["next_script"] == "analysis_v326_masked_family_pretrain.py"
    assert str(ranked.iloc[0]["resources"]).startswith("openttgames")
    assert set(ranked["experiment"]) == {
        "masked family pretrain",
        "response-style contrastive",
        "coarse-to-exact distillation",
    }
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_write_outputs_creates_audit_artifacts():
    outdir = v324.ROOT / "v324_clean_external_corpus_audit" / "test_workspace" / "write_outputs"
    inventory = pd.DataFrame(
        [{"resource": "openttgames", "file_count": 1, "total_size_bytes": 10, "status": "clean"}]
    )
    schemas = pd.DataFrame(
        [
            {
                "resource": "openttgames",
                "relative_path": "external_data/openttgames/processed/openttgames_events.csv",
                "suffix": ".csv",
                "size_bytes": 10,
                "content_read": True,
                "sample_rows": 1,
                "columns": "event_type|safe_action_family",
                "notes": "sampled",
            }
        ]
    )
    feature_summary = pd.DataFrame(
        [
            {
                "resource": "openttgames",
                "status": "clean",
                "canonical_rows": 1,
                "sequences": 1,
                "action_family_counts_json": "{}",
                "phase_counts_json": "{}",
                "sequence_length_quantiles_json": "{}",
                "landing_depth_available_rate": 0.0,
                "landing_side_available_rate": 0.0,
                "spin_available_rate": 0.0,
                "speed_available_rate": 0.0,
                "schema_compatibility": "high",
            }
        ]
    )
    recommendations = v324.rank_recommendations(feature_summary)

    v324.write_outputs(outdir, inventory, schemas, feature_summary, recommendations)

    expected = {
        "v324_external_file_inventory.csv",
        "v324_schema_summary.csv",
        "v324_canonical_feature_summary.csv",
        "v324_recommendations.csv",
        "v324_report.json",
        "v324_report.md",
    }
    assert expected.issubset({p.name for p in outdir.iterdir()})
    payload = json.loads((outdir / "v324_report.json").read_text(encoding="utf-8"))
    assert payload["submissions_written"] == 0
    assert payload["ttmatch_content_rows_read"] == 0
