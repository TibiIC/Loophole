Harbor runs of tasks/loophole-rattrap, 2026-11-14, before the release-after-block fix (rig_block_mode halt, every
substep audited) and before the closed-loop reference replay. Linux (arm64) simulator container, 2 CPUs.

| job | reward | notes |
|---|---|---|
| rattrap-oracle | 0.270000 | the fixed 284-command conversion of Tibi's solve: 18 / 20 checkpoints, unsolved. The same commands solve on macOS (0.833): cross-platform numerical divergence (docs/RATTRAP_DESIGN.md) |
| rattrap-noop | 0.000000 | no action |
| rattrap-stop-early | 0.000000 | grab + 5 random moves |
| rattrap-isolation-probe | 0.000000 | loophole/harbor_probe.py: agent-container isolation (docs/EXPLOITS.md R1-R5) |
