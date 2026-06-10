# External Resources

External resources were used for coarse pretraining, priors, or feature learning. Raw external files are not redistributed in this repository.

| Resource | Use | Redistribution | Notes |
|---|---|---|---|
| OpenTTGames | Coarse table-tennis priors / transition statistics | Not redistributed | Used for action-family and tactical priors |
| TT3D | Trajectory / landing / physical priors | Not redistributed | Used only for coarse auxiliary signals |
| AIMY | Inventory / license audit only | Not redistributed | Did not enter the final clean canonical corpus because local conversion conditions were insufficient |
| SpinDOE | Spin/physics auxiliary information | Not redistributed | Used only as auxiliary prior/reference where local audit allowed |
| CoachAI/ShuttleSet | Badminton sequence pretraining concepts / coarse shot-family priors | Not redistributed | No direct mapping to AICUP exact actionId |
| TT-MatchDynamics | Clean external sequence source with Apache 2.0 metadata | Not redistributed | Used only as a coarse sequence/reference source; no `test_new` row-wise alignment, no manual label alignment, and no row-wise final post-processing |

See `artifacts/external_audit/` for source and license audit tables.

Reference details for report writing are listed in `docs/references_apa.md`.
