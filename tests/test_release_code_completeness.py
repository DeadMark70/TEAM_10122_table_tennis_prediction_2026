from pathlib import Path


def test_final_pipeline_dependency_files_are_packaged():
    required = [
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


def test_release_does_not_package_raw_competition_or_external_data():
    forbidden_names = {"train.csv", "test_new.csv", "test_old.csv"}
    copied_names = {p.name for p in Path(".").rglob("*") if p.is_file()}
    assert forbidden_names.isdisjoint(copied_names)
    assert not Path("external_data").exists()
    assert not Path("data/raw").exists()
