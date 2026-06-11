# Model Components

## Action branch

The final action branch uses a V173 external-curriculum teacher. The teacher is
centered on AICUP training data and uses external sources only as coarse
tactical priors: action family, response pattern, phase, spin/physics hints, and
landing intent. It is not a direct row-level label transfer from external data.

Entry point:

```powershell
python scripts/train_action_teacher.py
```

## Point branch

The final point branch is a conservative residual specialist. It does not trust
raw neural `pointId` predictions directly. Instead, it scores candidate point
changes and exports only depth-agree/high-confidence corrections while
preserving the action and server branches.

Entry point:

```powershell
python scripts/train_point_residual.py
```

## Server branch

The final server branch uses clean probability/rank blending and avoids old-test
server labels in the clean submission. Old-overlap experiments were diagnostic
only and are documented separately.

Entry point:

```powershell
python scripts/train_server_model.py
```
