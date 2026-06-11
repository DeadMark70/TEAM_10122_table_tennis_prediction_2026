# Full Training Reproduction Guide

The repository is organized so that another reviewer can inspect and rerun the
main final pipeline components after cloning the GitHub repository.

## 1. Install environment

```powershell
python -m pip install -r requirements.txt
```

## 2. Place official data

Official AI CUP files are not redistributed. Place them as:

```text
data/raw/train.csv
data/raw/test_new.csv
data/raw/sample_submission.csv
```

The wrapper scripts copy these files to root-level legacy names
(`train.csv`, `test_new.csv`, `sample_submission.csv`) because many original
experiment scripts were written before the release package was normalized.
Those copied files are ignored by git.

## 3. Place optional external data

For the closest reproduction of the V173 external-curriculum action teacher,
place external datasets under `external_data/` using the source names documented
in `docs/external_resources.md`. If external datasets are omitted, scripts that
depend on them may fail or fall back depending on the original experiment code.

## 4. Run component training

```powershell
python scripts/train_action_teacher.py
python scripts/train_server_model.py
python scripts/train_point_residual.py
```

Or run the documented sequence:

```powershell
python scripts/train_full_pipeline.py
```

For a non-destructive dependency check:

```powershell
python scripts/train_full_pipeline.py --dry-run
```

## 5. Build and verify the final packaged submission

```powershell
python scripts/build_final_submission.py
python scripts/run_release_checks.py
```

`build_final_submission.py` verifies the packaged final artifact and writes
`outputs/final_submission.csv`.

## Reproducibility scope

This release supports final artifact verification and component-level reruns.
It does not claim to be a one-command reconstruction of every historical
experiment, upload probe, or intermediate checkpoint. The final score was
obtained by selecting the clean final artifact listed in `configs/final_v362.yaml`.
