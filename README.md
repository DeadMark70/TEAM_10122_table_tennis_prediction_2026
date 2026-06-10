# AI CUP 2026 Spring Table Tennis Prediction Final Code

GitHub: https://github.com/DeadMark70/TEAM_10122_table_tennis_prediction_2026

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
It is a release-artifact verification script, not a one-command full retraining pipeline.

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
