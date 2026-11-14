# Loophole

An RL environment (Harbor + MCP) where an LLM agent solves 3D **disentanglement puzzles**
in a MuJoCo simulation. Primary puzzle: an **Alpha-style puzzle**, two identical
alpha-shaped tubes interlocked through each other's loops; tube A is fixed, the agent
moves tube B with a grab handle and must separate them. The agent only talks to the
simulator over MCP (sidecar container), so it never sees the simulator, solver or
reference.

## Quick start

```bash
make dev                         # python3.11 venv + editable install (.[mesh,dev])
make test                        # pytest (all green; artifact-dependent tests skip with a reason)
.venv/bin/python scripts/demo.py # loads the Alpha puzzle, makes a few moves, writes runs/demo/*.png
```

Run the MCP server (streamable HTTP on :8000/mcp; admin options are CLI/env only, never MCP tools):

```bash
.venv/bin/python -m loophole.mcp_server --puzzle alpha-medium --port 8000 --log-dir runs/episode
# or in Docker (OSMesa headless rendering):
make build && docker run --rm -p 8000:8000 -e LOOPHOLE_PUZZLE=alpha-medium loophole-sim:local
```

MCP tools: `get_state`, `render` (PNG + caption; free), `grab`, `move`, `release`, `wait`
(one action each). Every call is logged to `episode.jsonl`.

Harbor (Docker + `uv tool install harbor`); every target builds the sim image first:

```bash
make reference  TASKS=tasks/loophole-alpha-medium                       # oracle replays the stored reference
make agent      TASKS=tasks/loophole-alpha-medium MODEL=anthropic/claude-sonnet-5 [AGENT=opencode] [RUNS=3]
make noop       TASKS=tasks/loophole-alpha-medium                       # built-in `nop` agent
make stop-early TASKS=tasks/loophole-alpha-medium                       # oracle + LOOPHOLE_SOLUTION=stop_early
make eval MODEL=...                                                     # all of the above over tasks/
```

## The Alpha puzzle (MVP)

* **Geometry: our own, procedural.** The canonical OMPL.app Alpha meshes are under the
  Rice University licence (non-commercial, *no redistribution*), so they are **never
  committed or shipped**. `loophole/puzzles/alpha/generate.py` builds two identical
  tubes by sweeping a sphere (radius r = 1 cm) along an alpha centre-line, the nodal
  cubic `S·(t²−1, t³−t, k·t)` (S = 0.15 m). The twist k opens a gap at the crossing,
  with nominal clearance `c = 2kS − 4r` (6 mm for `alpha-medium`). The start pose is a
  seeded search for the most clearance among poses whose loops are linked (Gauss linking
  number ±1). Inspired by the Alpha puzzle (B. Yamrom; Parasol Lab / N. Amato; OMPL.app
  benchmark).
* `make fetch-alpha` downloads the canonical OMPL.app files at a pinned commit (sha256
  checked) into `.ompl_alpha/` for **local** side-by-side comparison only
  (`scripts/compare_alpha.py`). This directory is gitignored and dockerignored.
* **Units:** metres. Puzzle YAMLs carry `unit_scale` (1.0 here), applied in exactly one
  place at load time (`sim/spec.py: apply_unit_scale`); `to_native()` converts back.
  The round trip is tested.
* **Collision geometry:** a chain of capsules along a polyline, which is an *exact* convex
  decomposition of the swept tube up to the polyline sagitta (0.3 mm, ≤ 0.2 × clearance,
  tested). We use this instead of CoACD because it is simpler and more accurate for a
  sweep; the CoACD importer (`sim/mesh_import.py`) remains for arbitrary meshes.
* **Sim:** gravity off, no floor, frictionless contacts, dt = 1 ms, contact margin 0.
  One grab handle (mocap + predeclared weld) with clamped, velocity-capped moves (≤ 5 cm
  and ≤ 0.4 rad per call) that stall when the held tube is obstructed. A per-substep audit
  (penetration > 4 mm, per-substep displacement > 3 mm, integrator divergence) voids
  success.
* **One collision checker everywhere:** `sim/collision.py` (`is_valid`, `is_separated`,
  `clearance`, `segment_valid`) uses MuJoCo's own collision pipeline on the same geoms as
  the sim. Note: MuJoCo's midphase ignores geom margins, so the checker disables it;
  otherwise margin queries silently miss pairs. This is tested against an independent
  numpy distance.
* **Reward (MVP):** success = min distance(A, B) > 6 cm (`d_sep`) with zero audit
  violations; score = success × (1 − 0.5 × actions_used / max_actions); noop = 0.
  `reward/progress.py: progress_potential(env)` returns 0 unless the WIP roadmap reward
  is switched on with `reward.roadmap: true`.
* **Verifier:** a separate Harbor container replays the logged actions in a fresh
  simulator and recomputes the score. It checks the puzzle hash, config, roadmap hash,
  initial state and every per-action state hash; any divergence or forgery scores 0.

## Status

| Component | Status | Notes |
|---|---|---|
| Alpha geometry generator (3 clearances) | done | `make alpha` regenerates; `alpha-medium` is the MVP variant |
| Sim: fixed A / free B, gravity off, frictionless, capsule-chain collision | done | |
| Grab handle, velocity-capped moves, stall on contact | done | `rig.num_handles: 1` for Alpha |
| Penetration / speed / instability audit | done | violations void success |
| Shared collision checker (`sim/collision.py`) | done | margin queries exact (midphase off) |
| MCP server (get_state, render, grab, move, release, wait) | done | streamable HTTP, OSMesa in Docker |
| MVP reward + independent replay verifier | done | |
| Tests: start valid, no tunnelling at max speed, noop = 0, teleport-apart = success | done | `tests/test_alpha.py` |
| `scripts/demo.py` | done | |
| Harbor task `tasks/loophole-alpha-medium` | in progress | generated, lint-clean; Harbor run not yet re-verified for Alpha |
| RRT-Connect reference (`loophole/planning/`, `scripts/build_alpha.py`) | in progress | plans, validates densely and replays for all 3 variants; easy/hard artifacts being regenerated |
| Cost-to-go roadmap reward (`reward/roadmap.py`) | in progress | built and tested; behind `reward.roadmap` (off); ctg-along-reference test xfail until the roadmap-shortest-path reference lands |
| Topological diagnostics (linking integral, loop words) | in progress | logged per action; `scripts/episode_report.py` plots them |
| Alpha easy / hard variants | in progress | generated, not yet exposed as tasks |
| Rope puzzles (easy/medium/hard) + Harbor tasks | done | earlier milestones; oracle 0.82 / nop 0 / stop-early 0 via Harbor |
| Pose stream / recorder / replay | done | |
| Canonical OMPL comparison (`make fetch-alpha`) | done | local only, never committed |

## Next steps for teammates

1. **Roadmap reward**: switch the reference to the roadmap's shortest path
   (`build_alpha.reference_from_roadmap`), flip `reward.roadmap` on for Alpha, and remove
   the xfail.
2. **Reference solution**: commit regenerated reference + stop-early trajectories for
   every variant; verify `harbor run -a oracle` on the Alpha task(s).
3. **Harbor tasks**: `tasks/loophole-alpha-{easy,medium,hard}` via `scripts/sync_tasks.py`;
   run reference / agent / nop / stop-early.
4. **More variants**: clearance ladder via `generate.py VARIANTS`; check difficulty with planning time.
5. Rope puzzles: keep as a second track (already working end to end).

## Layout

```
loophole/sim/        spec (YAML schema, unit_scale), builder (MJCF), rig, audit, collision checker, env, render
loophole/puzzles/    rope puzzles (*.yaml), alpha/ (generator, generated YAML + OBJ, roadmaps)
loophole/planning/   SE(3) RRT-Connect, shortcutting, path -> rig commands
loophole/reward/     components + composer, topology tracker, roadmap, progress stub, replay verifier
loophole/mcp_server/ FastMCP server + finalize collect hook
loophole/reference/  recorder, replay, MCP replay client, pose stream, trajectories/
tasks/               Harbor task dirs (generated by scripts/sync_tasks.py)
verifier/test.sh     verifier entrypoint baked into the sim image
```

Adding a puzzle: write a YAML in `loophole/puzzles/` (primitives, `tube`, `capsule_ring`,
or `mesh` with automatic CoACD decomposition), add a reference trajectory, then run
`python scripts/sync_tasks.py`.
