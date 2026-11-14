# Evaluation summary: `20261114-111735`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `gemini/gemini-4-carbon` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| oracle | loophole-alpha__fVDVriY | 0.851 | yes | 59 | 0 | 21.61 | 6.06 | 1 |
| noop | loophole-alpha__pjvF4jM | 0.000 | no | 0 | 0 | 21.61 | 21.61 | 1 |
| stop-early | loophole-alpha__H67A5CT | 0.000 | no | 6 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__AbJs9Px | 0.884 | yes | 46 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__qxQuETk | 0.823 | yes | 70 | 0 | 21.61 | 6.45 | 1 |
| opencode | loophole-alpha__sbMy6Ld | 0.886 | yes | 45 | 0 | 21.61 | 21.61 | 1 |

## Scores

| run | score |
|---|---|
| oracle | 0.851 |
| noop | 0.000 |
| stop-early | 0.000 |
| opencode `gemini/gemini-4-carbon` (n=3) | 0.864 ± 0.036 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).
