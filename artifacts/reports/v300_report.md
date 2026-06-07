# V300 Clean Server Blend Recycler

Clean server-only recycler. Action and point stay fixed to the V261/V173 clean anchor.

## Policy

- No TTMATCH input.
- No old-server input.
- No copying to upload_candidates.

## Sources

- Raw source rows read: `13`
- Unique server sources used: `11`
- Anchor: `upload_candidates_20260519\submission_v261_cap0p01__v173action_r121server.csv`

## Candidates

| candidate | kind | weight | sources | proxy_auc | proxy_delta | MAD | corr | risk |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `submission_v300_rankavg_w0p005__v173action_v261point_server.csv` | rankavg | 0.005 | 11 | 0.550670 | 0.007974 | 0.000003 | 1.000000 | `safe` |
| `submission_v300_rankavg_w0p01__v173action_v261point_server.csv` | rankavg | 0.010 | 11 | 0.550670 | 0.007974 | 0.000006 | 1.000000 | `safe` |
| `submission_v300_rankavg_w0p02__v173action_v261point_server.csv` | rankavg | 0.020 | 11 | 0.550670 | 0.007974 | 0.000013 | 1.000000 | `safe` |
| `submission_v300_mean_w0p005__v173action_v261point_server.csv` | mean | 0.005 | 11 | 0.550670 | 0.007974 | 0.000005 | 1.000000 | `safe` |
| `submission_v300_mean_w0p01__v173action_v261point_server.csv` | mean | 0.010 | 11 | 0.550670 | 0.007974 | 0.000010 | 1.000000 | `safe` |
| `submission_v300_mean_w0p02__v173action_v261point_server.csv` | mean | 0.020 | 11 | 0.550670 | 0.007974 | 0.000020 | 1.000000 | `safe` |
| `submission_v300_best_safe_repack__v173action_v261point_server.csv` | best_safe_repack | 1.000 | 1 | nan | 0.012409 | 0.001533 | 0.999929 | `safe` |

Search CSV: `v300_clean_server_blend_recycler\v300_server_search.csv`
