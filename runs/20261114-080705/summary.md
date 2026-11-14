# Evaluation summary: `20261114-080705`

Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `gemini/gemini-4-carbon` (OpenCode).

| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |
|---|---|---|---|---|---|---|---|---|
| oracle | loophole-alpha__Xx78e6N | 0.801 | yes | 79 | 0 | 21.61 | 0.23 | 1 |
| noop | loophole-alpha__qSYWQW4 | 0.000 | no | 0 | 0 | 21.61 | 21.61 | 1 |
| stop-early | loophole-alpha__Hh5ATY8 | 0.000 | no | 6 | 0 | 21.61 | 21.61 | 1 |
| opencode | loophole-alpha__5oFTZb8 | 0.838 | yes | 64 | 0 | 21.61 | 1.65 | pose-level (see rescored.json) |
| opencode | loophole-alpha__Kt5DJ3h | 0.139 | no | 157 | 0 | 21.61 | 11.57 | pose-level (see rescored.json) |
| opencode | loophole-alpha__wRU3ykj | 0.000 | no | 0 | 0 | 21.61 | 21.61 | pose-level (see rescored.json) |

## Scores

| run | score |
|---|---|
| oracle | 0.801 |
| noop | 0.000 |
| stop-early | 0.000 |
| opencode `gemini/gemini-4-carbon` (n=3) | 0.326 ± 0.449 |

Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).

## Notes

* **Interface.** This run used the *original* tool set and goal: min distance > 0.4 m;
  `get_state` exposed the goal distance and roadmap progress. The redesigned task
  (free-escape goal, `sensors` profile) is in `runs/20261114-093603-v2/`.
* **Agent scores were re-scored.** Harbor rebuilds the verifier image from the working
  tree, and the tree changed during this eval. So Harbor's verifier rejected the agent logs
  (`reward.txt` = 0, "different puzzle file"); those files are kept unchanged. Each agent log
  was then re-scored by a full replay in a Linux image of the code the simulator ran (see
  `verifier/rescored.json` and `reverified.json`). Every pose is within 5×10⁻⁷ m of the log,
  every per-action outcome (blocked / success / violations) matches, and each final score
  equals the simulator sidecar's own final record. The bitwise state hashes differ slightly
  (small code drift in that tree), so these are *pose-level*, not bitwise, verifications.
  `scripts/eval.py` now refuses to run with uncommitted sim-image changes.
* **Trials.**
  * `5oFTZb8` **solved** the puzzle in 64 actions (0.838). It decoded the PNGs itself over
    raw HTTP, space-carved both tubes from 50+ renders, planned in C++ on the
    reconstruction, then executed 63 moves without a single block.
  * `Kt5DJ3h` reached 46 % of the way out (held cost-to-go 11.6 of 21.6) in 157 actions,
    mostly by trial and error with `get_state` (82 calls, 36 blocked moves).
  * `wRU3ykj` spent its time on 194 renders, then stopped on a Gemini API error ("Requests
    ending with a model turn are not supported", HTTP 400) before acting.
