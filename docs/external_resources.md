# External Resources

External resources were used for coarse pretraining, priors, or feature learning. Raw external files are not redistributed in this repository. Reviewers should download them from the original source URLs and place them under the documented `external_data/` paths.

| Resource | Source URL | License recorded | Local path | Use | Redistribution |
|---|---|---|---|---|---|
| OpenTTGames / Extended OpenTTGames | https://arxiv.org/abs/2512.19327 | CC BY-NC-SA 4.0 | `external_data/openttgames` | Coarse table-tennis action-family / phase / transition priors | Not redistributed |
| TT3D | https://arxiv.org/abs/2504.10035 | CC BY 4.0 | `external_data/TT3D` | Trajectory, landing, and physical priors | Not redistributed |
| AIMY | https://arxiv.org/abs/2210.06048 | DL-DE/BY-2.0 in local audit | `external_data/AIMY` | Inventory / license audit only | Not redistributed |
| SpinDOE | https://arxiv.org/abs/2303.03879 | CC BY-SA 4.0 | `external_data/spindoe` | Spin / physics auxiliary reference | Not redistributed |
| CoachAI/ShuttleSet | https://github.com/wywyWang/CoachAI-Projects | MIT repository license; project citations required | `external_data/CoachAI-Projects-main` | Badminton sequence concepts / coarse shot-family priors | Not redistributed |
| TT-MatchDynamics | https://www.kaggle.com/datasets/guangliangyang/tt-matchdynamics | Apache 2.0 metadata on Kaggle | `external_data/TT-MatchDynamics` | Clean external sequence/coarse reference only | Not redistributed |
| DeepMind robot table tennis | https://arxiv.org/abs/2408.03906 | Apache 2.0 software; CC BY 4.0 materials | `external_data/DeepMindrobottabletennis` | Ball-state physics, spin, speed, and landing priors | Not redistributed |
| Sony table tennis | local source README / original publication source | CC BY-NC-ND 4.0 | `external_data/sonytabletennis` | Audit-only diagnostic reference | Not redistributed |

See `artifacts/external_audit/` for source and license audit tables.

Machine-readable source metadata is stored in `configs/external_sources.yaml`. Run:

```powershell
python scripts/check_external_sources.py
```

to print the expected source URLs, licenses, local placement paths, and whether the data is present locally.

Reference details for report writing are listed in `docs/references_apa.md`.
