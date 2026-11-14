# Evaluation summary: `20261114-204530`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `verda/kimi-k3-1-pro` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| opencode | loophole-alpha__4BanhcW | 0.005 | no | 35 | 0 | 21.61 | 21.28 | 1 |
| opencode | loophole-alpha__D9VAB6s | 0.000 | no | 3 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__UpevmKp | 0.000 | no | 182 | 0 | 21.61 | 21.61 | 1 |

## Scores

| run | score |
|---|---|
| oracle | – |
| noop | – |
| stop-early | – |
| opencode `verda/kimi-k3-1-pro` (n=3) | 0.002 ± 0.003 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).
