# Method Summary

The final system uses three task-specific components:

1. **Action branch**: an external-curriculum action teacher that combines table-tennis tactical priors, coarse external sequence priors, player-response style signals, and supervised AICUP action teachers.
2. **Point branch**: a conservative point-residual system. Neural point models were not used as raw argmax decoders; only low-risk depth-agreement and high-confidence nonterminal changes were accepted.
3. **Server branch**: a clean, low-variance server probability model. Old-test direct server labels were not used in the final clean submission.

The final submission is `submission_v362_depth_agree_only__v173action_v300server.csv`.
