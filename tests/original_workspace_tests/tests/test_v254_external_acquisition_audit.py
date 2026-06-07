import importlib


v254 = importlib.import_module("analysis_v254_external_acquisition_audit")


def test_ttmatch_is_banned_and_not_readable():
    assert not v254.should_read_content("TTMATCH")
    policy = v254.dataset_policy("TTMATCH")
    assert policy.tier == "RED"
    assert "Do not use" in policy.prohibited_use


def test_known_clean_external_policies():
    assert v254.dataset_policy("openttgames").tier == "GREEN"
    assert v254.dataset_policy("DeepMindrobottabletennis").tier == "GREEN"
    assert v254.dataset_policy("sonytabletennis").tier == "YELLOW"
    assert v254.dataset_policy("TT3D").tier == "YELLOW"


def test_vector_length_from_string():
    assert v254.vector_length_from_string("[ 1.0 2.0 3.0]") == 3
    assert v254.vector_length_from_string("[1, 2, 3]") == 3
    assert v254.vector_length_from_string("not a vector") == 0


def test_banned_relevance_is_not_evaluated():
    import pandas as pd

    files = pd.DataFrame(
        [
            {
                "dataset": "TTMATCH",
                "relative_path": "external_data/TTMATCH/train.csv",
                "suffix": ".csv",
                "size_bytes": 1,
                "read_content_allowed": False,
            }
        ]
    )
    out = v254.action_relevance_audit(files)
    row = out.iloc[0]
    assert row["recommended_targets"] == "not_evaluated_banned_dataset"
    assert not bool(row["has_exact_aicup_action_label"])
