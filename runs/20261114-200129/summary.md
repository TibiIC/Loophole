# Evaluation summary: `20261114-200129`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `verda/deepseek-4-1-flash-1017` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| opencode | loophole-alpha__BpQxf4N | 0.004 | no | 198 | 0 | 21.61 | 21.29 | 1 |
| opencode | loophole-alpha__jAVD75d | 0.005 | no | 198 | 0 | 21.61 | 21.26 | 1 |
| opencode | loophole-alpha__wZquaF5 | 0.007 | no | 179 | 0 | 21.61 | 21.10 | 1 |

## Scores

| run | score |
|---|---|
| oracle | – |
| noop | – |
| stop-early | – |
| opencode `verda/deepseek-4-1-flash-1017` (n=3) | 0.005 ± 0.001 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).
