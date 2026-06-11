from pathlib import Path


def test_final_pipeline_dependency_files_are_packaged():
    required = [
        "scripts/train_action_teacher.py",
        "scripts/train_point_residual.py",
        "scripts/train_server_model.py",
        "scripts/train_full_pipeline.py",
        "scripts/build_final_submission.py",
        "scripts/run_release_checks.py",
        "scripts/pipeline_utils.py",
        "configs/action_teacher_v173.yaml",
        "configs/point_v362.yaml",
        "configs/server_v300.yaml",
        "configs/full_training.yaml",
        "docs/artifact_provenance.md",
        "docs/full_training_reproduction.md",
        "docs/model_components.md",
        "src/project_root/analysis_v173_external_curriculum_pretrain.py",
        "src/project_root/analysis_v165_combined_external_pretrain_proxy.py",
        "src/project_root/analysis_v160_v163_task_pretrain_distill.py",
        "src/project_root/analysis_v188_point_intent_gru.py",
        "src/project_root/analysis_v188_export_r186_w005_cap005.py",
        "src/project_root/analysis_v300_clean_server_blend_recycler.py",
        "src/project_root/analysis_v338_joint_moe_pack.py",
        "src/project_root/analysis_v362_point_hierarchical_specialists.py",
        "src/src_tree/src/analysis/analysis_v300_clean_server_blend_recycler.py",
    ]
    for rel in required:
        assert Path(rel).exists(), rel


def test_teacher_provenance_docs_link_teachers_to_scripts():
    text = Path("docs/artifact_provenance.md").read_text(encoding="utf-8")
    required_pairs = {
        "V173 action teacher": "analysis_v173_external_curriculum_pretrain.py",
        "V300 server blend": "analysis_v300_clean_server_blend_recycler.py",
        "V362 point residual": "analysis_v362_point_hierarchical_specialists.py",
    }
    for component, script in required_pairs.items():
        assert component in text
        assert script in text


def test_training_wrappers_are_importable():
    import ast

    for rel in [
        "scripts/pipeline_utils.py",
        "scripts/train_action_teacher.py",
        "scripts/train_point_residual.py",
        "scripts/train_server_model.py",
        "scripts/train_full_pipeline.py",
        "scripts/build_final_submission.py",
        "scripts/run_release_checks.py",
    ]:
        ast.parse(Path(rel).read_text(encoding="utf-8"), filename=rel)


def test_release_does_not_package_raw_competition_or_external_data():
    forbidden_names = {"train.csv", "test_new.csv", "test_old.csv"}
    copied_names = {p.name for p in Path(".").rglob("*") if p.is_file()}
    assert forbidden_names.isdisjoint(copied_names)
    assert not Path("external_data").exists()
    assert not Path("data/raw").exists()
