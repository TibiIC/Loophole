# Evaluation summary: `20261114-192911`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `gemini/gemini-3.5-flash-lite` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| opencode | loophole-alpha__UofRoUL | 0.000 | no | 54 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__o6W9Upt | 0.000 | no | 55 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__tyxJ2PC | 0.000 | no | 198 | 0 | 21.61 | 21.61 | 1 |

## Scores

| run | score |
|---|---|
| oracle | – |
| noop | – |
| stop-early | – |
| opencode `gemini/gemini-3.5-flash-lite` (n=3) | 0.000 ± 0.000 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).
