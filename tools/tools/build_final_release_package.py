from __future__ import annotations

import json
import shutil
from pathlib import Path


FINAL_SUBMISSION = "submission_v362_depth_agree_only__v173action_v300server.csv"


def copy_file(root: Path, outdir: Path, src_rel: str, dst_rel: str, copied: list[str]) -> None:
    src = root / src_rel
    dst = outdir / dst_rel
    if not src.exists():
        raise FileNotFoundError(f"Missing required release file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst_rel.replace("\\", "/"))


def write_text(outdir: Path, rel: str, text: str, copied: list[str]) -> None:
    dst = outdir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    copied.append(rel.replace("\\", "/"))


def copy_python_sources(root: Path, outdir: Path, copied: list[str]) -> None:
    """Copy runnable project Python code without raw data or bulky artifacts."""
    source_specs = [
        ("*.py", "src/project_root"),
        ("src/**/*.py", "src/src_tree"),
        ("tools/**/*.py", "tools"),
        ("tests/*.py", "tests/original_workspace_tests"),
    ]
    for pattern, dst_prefix in source_specs:
        for src in sorted(root.glob(pattern)):
            if "__pycache__" in src.parts:
                continue
            if src.name.endswith(".pyc"):
                continue
            if src.resolve().is_relative_to(outdir):
                continue
            rel = src.relative_to(root)
            dst = outdir / dst_prefix / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str((Path(dst_prefix) / rel).as_posix()))


def build_release_package(root: Path, outdir: Path) -> dict:
    root = root.resolve()
    outdir = outdir.resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    copied: list[str] = []

    copy_python_sources(root, outdir, copied)

    copy_file(
        root,
        outdir,
        "v362_point_hierarchical_specialists/submission_v362_depth_agree_only__v173action_v300server.csv",
        f"artifacts/final_submission/{FINAL_SUBMISSION}",
        copied,
    )

    required_files = [
        ("analysis_v173_external_curriculum_pretrain.py", "src/analysis/analysis_v173_external_curriculum_pretrain.py"),
        ("analysis_v188_point_intent_gru.py", "src/analysis/analysis_v188_point_intent_gru.py"),
        ("analysis_v188_export_r186_w005_cap005.py", "src/analysis/analysis_v188_export_r186_w005_cap005.py"),
        ("analysis_v300_clean_server_blend_recycler.py", "src/analysis/analysis_v300_clean_server_blend_recycler.py"),
        ("src/analysis/analysis_v300_clean_server_blend_recycler.py", "src/analysis/core/analysis_v300_clean_server_blend_recycler.py"),
        ("analysis_v335_moe_anchor_contract.py", "src/analysis/analysis_v335_moe_anchor_contract.py"),
        ("analysis_v338_joint_moe_pack.py", "src/analysis/analysis_v338_joint_moe_pack.py"),
        ("analysis_v362_point_hierarchical_specialists.py", "src/analysis/analysis_v362_point_hierarchical_specialists.py"),
        ("v173_external_curriculum_pretrain/v173_report.md", "artifacts/reports/v173_report.md"),
        ("v173_external_curriculum_pretrain/v173_report.json", "artifacts/reports/v173_report.json"),
        ("v173_external_curriculum_pretrain/v173_action_curriculum_search.csv", "artifacts/reports/v173_action_curriculum_search.csv"),
        ("v173_external_curriculum_pretrain/v173_opentt_prior_table.csv", "artifacts/reports/v173_opentt_prior_table.csv"),
        ("v173_external_curriculum_pretrain/v173_coachai_transition_stats.csv", "artifacts/reports/v173_coachai_transition_stats.csv"),
        ("v338_joint_moe_pack/search_report.json", "artifacts/reports/v338_search_report.json"),
        ("v338_joint_moe_pack/joint_summary.csv", "artifacts/reports/v338_joint_summary.csv"),
        ("v300_clean_server_blend_recycler/v300_report.md", "artifacts/reports/v300_report.md"),
        ("v300_clean_server_blend_recycler/v300_report.json", "artifacts/reports/v300_report.json"),
        ("v300_clean_server_blend_recycler/v300_server_search.csv", "artifacts/reports/v300_server_search.csv"),
        ("v362_point_hierarchical_specialists/search_report.json", "artifacts/reports/v362_search_report.json"),
        ("v362_point_hierarchical_specialists/candidate_summary.csv", "artifacts/reports/v362_candidate_summary.csv"),
        ("v362_point_hierarchical_specialists/scored_candidates.csv", "artifacts/reports/v362_scored_candidates.csv"),
        ("v362_point_hierarchical_specialists/depth_backoff_table.csv", "artifacts/reports/v362_depth_backoff_table.csv"),
        ("v362_point_hierarchical_specialists/source_votes.csv", "artifacts/reports/v362_source_votes.csv"),
        ("v411_external_inventory_lockfile/dataset_summary.csv", "artifacts/external_audit/dataset_summary.csv"),
        ("v411_external_inventory_lockfile/license_summary.csv", "artifacts/external_audit/license_summary.csv"),
        ("v411_external_inventory_lockfile/search_report.json", "artifacts/external_audit/v411_search_report.json"),
        ("v413_external_license_overlap_guard/allowed_sources.csv", "artifacts/external_audit/allowed_sources.csv"),
        ("v413_external_license_overlap_guard/blocked_sources.csv", "artifacts/external_audit/blocked_sources.csv"),
        ("v413_external_license_overlap_guard/license_guard_report.json", "artifacts/external_audit/license_guard_report.json"),
        ("v430_external_audit_canonical_expander/external_source_audit.csv", "artifacts/external_audit/external_source_audit.csv"),
        ("v430_external_audit_canonical_expander/license_overlap_report.json", "artifacts/external_audit/license_overlap_report.json"),
    ]
    for src_rel, dst_rel in required_files:
        copy_file(root, outdir, src_rel, dst_rel, copied)

    write_text(outdir, "README.md", README_TEXT, copied)
    write_text(outdir, "requirements.txt", REQUIREMENTS_TEXT, copied)
    write_text(outdir, ".gitignore", GITIGNORE_TEXT, copied)
    write_text(outdir, "configs/final_v362.yaml", FINAL_CONFIG_TEXT, copied)
    write_text(outdir, "scripts/verify_submission.py", VERIFY_SUBMISSION_TEXT, copied)
    write_text(outdir, "scripts/reproduce_final.py", REPRODUCE_FINAL_TEXT, copied)
    write_text(outdir, "scripts/reproduce_final.ps1", REPRODUCE_FINAL_PS1_TEXT, copied)
    write_text(outdir, "docs/method_summary.md", METHOD_SUMMARY_TEXT, copied)
    write_text(outdir, "docs/external_resources.md", EXTERNAL_RESOURCES_TEXT, copied)
    write_text(outdir, "docs/ai_usage_disclosure.md", AI_USAGE_TEXT, copied)
    write_text(outdir, "docs/old_overlap_diagnostic_note.md", OLD_OVERLAP_NOTE_TEXT, copied)
    write_text(outdir, "docs/report_outline_mapping.md", REPORT_OUTLINE_MAPPING_TEXT, copied)
    write_text(outdir, "docs/report_asset_checklist.md", REPORT_ASSET_CHECKLIST_TEXT, copied)
    write_text(outdir, "docs/references_apa.md", REFERENCES_APA_TEXT, copied)
    write_text(outdir, "tests/test_release_submission.py", TEST_RELEASE_SUBMISSION_TEXT, copied)
    write_text(outdir, "tests/test_external_resource_docs.py", TEST_EXTERNAL_RESOURCE_DOCS_TEXT, copied)
    write_text(outdir, "tests/test_release_code_completeness.py", TEST_RELEASE_CODE_COMPLETENESS_TEXT, copied)

    report = {
        "outdir": str(outdir),
        "copied_files": copied,
        "final_submission": FINAL_SUBMISSION,
    }
    (outdir / "release_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


README_TEXT = """# AI CUP 2026 Spring Table Tennis Prediction Final Code

This repository contains the final clean pipeline used for the submission:

`submission_v362_depth_agree_only__v173action_v300server.csv`

Final clean submission: `submission_v362_depth_agree_only__v173action_v300server.csv`

Final score: `0.3750309`

Rank: `20/423`

## Pipeline

1. Action prediction: external curriculum and table-tennis tactical priors, V173 action teacher.
2. Point prediction: conservative depth-agreement point specialist, V362.
3. Rally outcome: conservative clean server model, V300.

## Environment

```powershell
python -m pip install -r requirements.txt
```

## Reproduce final submission check

```powershell
python scripts/reproduce_final.py
python -m pytest tests -q -p no:cacheprovider
```

`scripts/reproduce_final.py` verifies the final submission schema and copies it to `outputs/final_submission.csv`.

## Data placement for retraining

Official competition files are not redistributed. To retrain, place:

```text
data/raw/train.csv
data/raw/test_new.csv
data/raw/sample_submission.csv
```

Reference old test data was used only for diagnostic analysis and not for the final clean submission.

## External resources

External datasets are documented in `docs/external_resources.md` and audited under `artifacts/external_audit/`.
"""

REQUIREMENTS_TEXT = """numpy
pandas
scikit-learn
torch
lightgbm
catboost
xgboost
pytest
pyyaml
"""

GITIGNORE_TEXT = """__pycache__/
*.pyc
.pytest_cache/
data/raw/
external_data/
*.pkl
*.pt
*.pth
*.npy
outputs/
"""

FINAL_CONFIG_TEXT = """final_submission: artifacts/final_submission/submission_v362_depth_agree_only__v173action_v300server.csv
final_score: 0.3750309
rank: 20
teams: 423
action_component: V173 external curriculum action teacher
point_component: V362 depth-agree-only point specialist
server_component: V300 clean conservative server blend
"""

VERIFY_SUBMISSION_TEXT = """from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def verify_submission(path: Path) -> None:
    df = pd.read_csv(path)
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != 1845:
        raise ValueError(f"Expected 1845 rows, got {len(df)}")
    if df.isna().sum().sum() != 0:
        raise ValueError("Submission contains NaN values")
    if not df["serverGetPoint"].between(0, 1).all():
        raise ValueError("serverGetPoint must be in [0, 1]")
    print(f"OK: {path}")


if __name__ == "__main__":
    verify_submission(Path(sys.argv[1]))
"""

REPRODUCE_FINAL_TEXT = """from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_submission import verify_submission


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "final_submission" / "submission_v362_depth_agree_only__v173action_v300server.csv"
DST = ROOT / "outputs" / "final_submission.csv"


def main() -> None:
    verify_submission(SRC)
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    verify_submission(DST)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
"""

REPRODUCE_FINAL_PS1_TEXT = """python scripts/reproduce_final.py
"""

METHOD_SUMMARY_TEXT = """# Method Summary

The final system uses three task-specific components:

1. **Action branch**: an external-curriculum action teacher that combines table-tennis tactical priors, coarse external sequence priors, player-response style signals, and supervised AICUP action teachers.
2. **Point branch**: a conservative point-residual system. Neural point models were not used as raw argmax decoders; only low-risk depth-agreement and high-confidence nonterminal changes were accepted.
3. **Server branch**: a clean, low-variance server probability model. Old-test direct server labels were not used in the final clean submission.

The final submission is `submission_v362_depth_agree_only__v173action_v300server.csv`.
"""

EXTERNAL_RESOURCES_TEXT = """# External Resources

External resources were used for coarse pretraining, priors, or feature learning. Raw external files are not redistributed in this repository.

| Resource | Use | Redistribution | Notes |
|---|---|---|---|
| OpenTTGames | Coarse table-tennis priors / transition statistics | Not redistributed | Used for action-family and tactical priors |
| TT3D | Trajectory / landing / physical priors | Not redistributed | Used only for coarse auxiliary signals |
| AIMY | Physics / spin / trajectory auxiliary information | Not redistributed | Used only where local files existed and license allowed |
| SpinDOE | Spin/physics auxiliary information | Not redistributed | Used only where local files existed and license allowed |
| CoachAI/ShuttleSet | Badminton sequence pretraining concepts / coarse shot-family priors | Not redistributed | No direct mapping to AICUP exact actionId |
| TT-MatchDynamics | Optional clean external sequence source where license permits | Not redistributed | Any overlapping or high-risk usage was disclosed and separated from final clean submission |

See `artifacts/external_audit/` for source and license audit tables.

Reference details for report writing are listed in `docs/references_apa.md`.
"""

AI_USAGE_TEXT = """# Generative AI Usage Disclosure

During the competition, generative AI tools were used to assist:

- experiment planning and hypothesis generation;
- code drafting and debugging;
- writing helper scripts and tests;
- summarizing experiment logs;
- drafting and organizing the final report.

All submitted predictions were produced by automated machine-learning/deep-learning pipelines. No submitted row was manually edited after inspecting its target label. The final clean submission did not use old-test server labels as direct replacements.
"""

OLD_OVERLAP_NOTE_TEXT = """# Old-Overlap Diagnostic

We tested an old-test overlap server diagnostic to estimate how much the Public score could be improved by directly aligning old-test `serverGetPoint` values where `rally_uid` overlapped.

Diagnostic result:

- `submission_v472_old_overlap_hard_server__v362anchor.csv`
- Public score: `0.4273695`

This diagnostic was not selected as the final clean submission because the official notice warned that overreliance on old-test server labels may overfit the Public leaderboard and may not generalize to the newly released Private data. The final selected submission was:

- `submission_v362_depth_agree_only__v173action_v300server.csv`
- Final score: `0.3750309`
- Rank: `20/423`
"""

REPORT_OUTLINE_MAPPING_TEXT = """# Official Report Outline Mapping

壹、環境: use README, requirements, external resources.

貳、演算方法與模型架構: include final pipeline diagram.

參、創新性: external curriculum action teacher, conservative point residual gating, and clean server blend.

肆、資料處理: official data parsing, full-prefix features, external-source audit, and no manual row correction.

伍、訓練方式: action teacher, point residual specialists, server model, and validation.

陸、分析與結論: include final score, public/private jump, successful/failed directions, and figures from reports.

柒、程式碼: provide GitHub link and attached release package.

捌、使用的外部資源與參考文獻: use external resources document and APA citations.
"""

REPORT_ASSET_CHECKLIST_TEXT = """# Report Asset Checklist

Required official sections:

- 壹、環境: use README, requirements, external resources.
- 貳、演算方法與模型架構: include final pipeline diagram.
- 參、創新性: external curriculum action teacher, conservative point residual gating, clean server blend.
- 肆、資料處理: official data parsing, full-prefix features, external-source audit, no manual correction.
- 伍、訓練方式: action teacher, point residual specialists, server model, validation.
- 陸、分析與結論: include at least one figure and successful/failed examples.
- 柒、程式碼: GitHub URL plus attached code package.
- 捌、使用的外部資源與參考文獻: APA-style references.

Figures to prepare:

1. Overall pipeline diagram: action, point, server branches.
2. Public vs final/private score comparison.
3. Final point changes distribution or depth-agreement gate analysis.
4. External data audit and clean-vs-diagnostic separation.
"""

REFERENCES_APA_TEXT = """# References And External Resource Citations

Use APA style in the final Word report. This file is a working citation checklist; verify author names and URLs against the original dataset pages before submission.

- AI CUP 2026 Spring. (2026). *基於時序資料之桌球戰術與結果預測競賽*. AIdea.
- OpenTTGames contributors. (n.d.). *OpenTTGames table tennis dataset / resources*.
- CoachAI contributors. (n.d.). *CoachAI projects and ShuttleSet badminton stroke forecasting resources*.
- TT3D contributors. (n.d.). *TT3D table tennis trajectory dataset*.
- AIMY contributors. (n.d.). *AIMY table tennis trajectory / robot dataset*.
- SpinDOE contributors. (n.d.). *SpinDOE table tennis spin / physics dataset*.
- TT-MatchDynamics contributors. (n.d.). *TT-MatchDynamics dataset*. Kaggle.

The code package includes audit tables under `artifacts/external_audit/` to document which resources were allowed, blocked, or used only as coarse auxiliary priors.
"""

TEST_RELEASE_SUBMISSION_TEXT = """from pathlib import Path

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
"""

TEST_EXTERNAL_RESOURCE_DOCS_TEXT = """from pathlib import Path


def test_external_docs_and_audit_tables_exist():
    assert Path("docs/external_resources.md").exists()
    assert Path("docs/ai_usage_disclosure.md").exists()
    assert Path("artifacts/external_audit/license_summary.csv").exists()
    assert Path("artifacts/external_audit/allowed_sources.csv").exists()
    assert Path("artifacts/external_audit/external_source_audit.csv").exists()


def test_old_overlap_note_separates_diagnostic_from_final():
    text = Path("docs/old_overlap_diagnostic_note.md").read_text(encoding="utf-8")
    assert "not selected as the final clean submission" in text
    assert "submission_v362_depth_agree_only__v173action_v300server.csv" in text
"""

TEST_RELEASE_CODE_COMPLETENESS_TEXT = """from pathlib import Path


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
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_release_package(root=root, outdir=root / "release_final")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
