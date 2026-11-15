Harbor runs of tasks/loophole-rattrap, 2026-11-14, before the release-after-block fix (rig_block_mode halt, every
substep audited) and before the closed-loop reference replay. Linux (arm64) simulator container, 2 CPUs.

| job | reward | notes |
|---|---|---|
| rattrap-oracle | 0.270000 | the fixed 284-command conversion of Tibi's solve: 18 / 20 checkpoints, unsolved. The same commands solve on macOS (0.833): cross-platform numerical divergence (docs/RATTRAP_DESIGN.md) |
| rattrap-noop | 0.000000 | no action |
| rattrap-stop-early | 0.000000 | grab + 5 random moves |
| rattrap-isolation-probe | 0.000000 | loophole/harbor_probe.py: agent-container isolation (docs/EXPLOITS.md R1-R5) |

Closed-loop reference, 2026-11-15, commit 7f8094b (release settle audit on, rig_block_mode halt; Linux arm64 simulator
container, 2 CPUs; total runtime 1 h 26 min, of which the verifier's replay 42 min):

| job | reward | notes |
|---|---|---|
| rattrap-oracle-closedloop | **0.833333** | Tibi's recorded human solve, replayed closed-loop from its fixed waypoints (`solution/waypoints.jsonl`, `solution/follow_waypoints.py`): solved, 284 actions, 20 / 20 checkpoints, log consistent; the verifier's independent replay agrees |
