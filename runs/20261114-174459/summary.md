# Evaluation summary: `20261114-174459`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `gemini/gemini-3.8-flash` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| opencode | loophole-alpha__QrsMh3s | 0.006 | no | 189 | 0 | 21.61 | 21.19 | 1 |
| opencode | loophole-alpha__sHE2fBv | 0.000 | no | 196 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__snypBd4 | 0.019 | no | 76 | 0 | 21.61 | 20.27 | 1 |

## Scores

| run | score |
|---|---|
| oracle | – |
| noop | – |
| stop-early | – |
| opencode `gemini/gemini-3.8-flash` (n=3) | 0.008 ± 0.010 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).
