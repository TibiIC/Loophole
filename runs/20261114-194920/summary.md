# Evaluation summary: `20261114-194920`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `gemini/gemini-2.5-flash` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| opencode | loophole-alpha__8RcG9VQ | 0.000 | no | 0 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__cDA8RbC | 0.000 | no | 0 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__j7MwLox | 0.000 | no | 0 | 0 | 21.61 | 21.61 | 1 |

## Scores

| run | score |
|---|---|
| oracle | – |
| noop | – |
| stop-early | – |
| opencode `gemini/gemini-2.5-flash` (n=3) | 0.000 ± 0.000 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).
