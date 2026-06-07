# Old-Overlap Diagnostic

We tested an old-test overlap server diagnostic to estimate how much the Public score could be improved by directly aligning old-test `serverGetPoint` values where `rally_uid` overlapped.

Diagnostic result:

- `submission_v472_old_overlap_hard_server__v362anchor.csv`
- Public score: `0.4273695`

This diagnostic was not selected as the final clean submission because the official notice warned that overreliance on old-test server labels may overfit the Public leaderboard and may not generalize to the newly released Private data. The final selected submission was:

- `submission_v362_depth_agree_only__v173action_v300server.csv`
- Final score: `0.3750309`
- Rank: `20/423`
