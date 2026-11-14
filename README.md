# Loophole

An RL environment (Harbor + MCP) where an LLM agent solves 3D **disentanglement puzzles**
in a MuJoCo simulation. Primary puzzle: an **Alpha-style puzzle**, two identical
alpha-shaped tubes interlocked through each other's loops; tube A is fixed, the agent
moves tube B with a grab handle and must separate them. The agent only talks to the
simulator over MCP (sidecar container), so it never sees the simulator, solver or
reference.

## The Alpha puzzle task (`tasks/loophole-alpha`)

**What the agent must do.** Two identical rigid tubes, each bent into an alpha, are
interlocked: each loop threads the other. Tube A is fixed; the agent grabs tube B with
one handle and moves it with bounded steps (≤ 5 cm and ≤ 0.1 rad per `move`,
velocity-limited) until the minimum distance between the tubes exceeds **0.4 m**,
within a fixed action budget. It sees the world only through the MCP tools `get_state`,
`render`, `grab`, `move`, `release` and `wait`.

**Why it is hard.** The Alpha puzzle is a classic narrow-passage problem in the 6-D
configuration space SE(3). The way out exists, but it is a thin tunnel of
configurations that requires coordinated translation *and* rotation. Uniform sampling
almost never hits it, and greedy "move away" motions stall against the other tube. The
puzzle was designed by Boris Yamrom (GE Corporate R&D) and became a standard
motion-planning benchmark through Nancy Amato's Parasol Lab and OMPL. On why such
puzzles are hard and how their C-space tunnels can be found, see X. Zhang, R. Belfer,
P. G. Kry, E. Vouga, *C-Space Tunnel Discovery for Puzzle Path Planning*, ACM
Transactions on Graphics 39(4) (SIGGRAPH 2020).

**Geometry: canonical, fetched, never committed.** We use the OMPL.app Alpha files
(`resources/3D/` at commit `958884d`): `alpha_env-1.5.dae` (tube A), `alpha_robot.dae`
(tube B), `alpha-1.5.cfg` (start/goal/volume) and `alpha-1.5.path` (OMPL's own solution,
used only as a test). Licence: *Rice University Software Distribution License*
(non-commercial; redistribution not permitted). They are downloaded with sha256 checks
by `make fetch-alpha` (locally, into gitignored `.ompl_alpha/`) and at Docker build time.
Derived meshes and hulls are generated there and never committed. The repo contains only
poses and parameters (`loophole/puzzles/alpha_ompl/alpha-1.5.yaml`).
* Frame convention, taken from OMPL.app's source: Assimp converts the Z-up Collada files
  to Y-up, (x, y, z) → (x, z, −y), and the robot mesh is shifted by the mean of its
  vertices before the `.cfg` pose is applied. With this convention OMPL's own path is
  collision-free under FCL on the raw triangles (0/103 states); without the up-axis step
  ~40% of it collides.
* `unit_scale: 0.01` (native units → metres, scene ≈ ±2 m) is applied once at load time;
  start-pose round-trip to the `.cfg` values is tested to be exact.
* **Convex decomposition: face wedges rather than CoACD.** CoACD's threshold is relative
  to the 2.8 m bounding diagonal, so even threshold 0.03 gave hulls bulging ~8 cm. That is
  far larger than the narrowest clearance on OMPL's path (1.8 mm). We therefore decompose
  each canonical triangle into a convex wedge extruded inward, with the inward direction
  found by ray casting: the outside is exactly the canonical surface. Measured against
  FCL on the raw triangles along OMPL's path, the max distance error is **0.15 mm**
  (≈ 8% of the 1.8 mm narrowest clearance). Start clearance is 32.9 mm in both. CoACD
  stays available (`decompose: coacd`) for other meshes.
* `d_sep = 0.4 m`: hill-climbing the clearance from the start with collision-free motions
  saturates at ≈ 0.22 m, so 0.4 m cannot be reached while the tubes are still interlocked.

**Scoring.** The verifier writes one float to `/logs/verifier/reward.txt`:
`score = success × (1 − 0.5 × actions_used / max_actions)`. Success means the min
distance exceeded `d_sep` and the per-substep penetration/speed audit recorded zero
violations. Doing nothing or stopping early scores 0. Per action, the log also records
(unscored) the min distance, the relative pose of B w.r.t. A (metres and native units)
and the `progress_potential` hook (cost-to-go roadmap; flagged off).

**Anti-exploit setup.** Harbor supports multi-container environments, so the MuJoCo MCP
server runs in a **separate sidecar container** (`loophole-sim`). The agent's container
has no simulator code, meshes, solver or episode log; it can only reach the MCP
endpoint. After the agent finishes, a collect hook in the sidecar finalises the episode
(the agent container is already stopped). The sidecar's episode directory goes to a
**separate verifier container** (sim image, no network). That container replays every
logged action in a fresh simulator and recomputes the score, checking puzzle hash, config,
initial state and every per-action state hash. Nothing the agent writes is read.

**One-command reproduction** (Docker + `uv tool install harbor`; the image build fetches
the meshes):
```bash
harbor run -p ./tasks/loophole-alpha -a oracle                       # reference solution
harbor run -p ./tasks/loophole-alpha -a nop                          # do-nothing baseline
harbor run -p ./tasks/loophole-alpha -a oracle --ae LOOPHOLE_SOLUTION=stop_early
OPENAI_API_KEY=... harbor run -p ./tasks/loophole-alpha -a opencode -m openai/<model>
```

> **Deprecated:** the procedural Alpha-*style* tubes (`loophole/puzzles/alpha/`,
> `tasks/loophole-alpha-{easy,medium,hard}`) are not the real Alpha geometry. They are
> kept for reference and tests but are no longer the active task.

### Model API keys (for `-a opencode`)

For Gemini with OpenCode, set **`GOOGLE_GENERATIVE_AI_API_KEY`**: that's the only name
OpenCode's Google provider reads. Harbor forwards whichever of `GOOGLE_API_KEY`,
`GEMINI_API_KEY` and `GOOGLE_GENERATIVE_AI_API_KEY` it finds first, so leave the other
two unset in that shell. Models are `gemini/<id>` (list yours with the models API).
OpenAI: `OPENAI_API_KEY`, models `openai/<id>`. Keep keys in a gitignored `.env`
(`.env*` is ignored), then:

```bash
set -a; source .env; set +a
harbor run -p ./tasks/loophole-alpha -a opencode -m gemini/gemini-4-carbon
```

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
# all Alpha tasks: add HARBOR_FLAGS via `harbor run -p tasks -i 'loophole-alpha-*' ...`, or one task:
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
| **Canonical Alpha 1.5 task `tasks/loophole-alpha`** | done | harbor: oracle **0.876**, nop **0**, stop-early **0** (runs/harbor/) |
| Canonical geometry pipeline (fetch, Assimp frame, face-wedge hulls) | done | OMPL's own path collision-free; hull error 0.15 mm vs 1.8 mm clearance |
| Canonical oracle (RRT-Connect, 20 min box) | done | uniform sampling failed in 20 min; orientation-focused solved in 406 s |
| Real model run (`-a opencode`) | not started | waiting for an API key (see "Model API keys") |
| Canonical Alpha 1.2 / 1.0 | not started | `write_puzzle("1.2")` + `build_alpha_ompl.py --variant 1.2` |
| Alpha geometry generator (3 clearances) | done | `make alpha` regenerates; `alpha-medium` is the MVP variant |
| Sim: fixed A / free B, gravity off, frictionless, capsule-chain collision | done | |
| Grab handle, velocity-capped moves, stall on contact | done | `rig.num_handles: 1` for Alpha |
| Penetration / speed / instability audit | done | violations void success |
| Shared collision checker (`sim/collision.py`) | done | margin queries exact (midphase off) |
| MCP server (get_state, render, grab, move, release, wait) | done | streamable HTTP, OSMesa in Docker |
| MVP reward + independent replay verifier | done | |
| Tests: start valid, no tunnelling at max speed, noop = 0, teleport-apart = success | done | `tests/test_alpha.py` |
| `scripts/demo.py` | done | |
| Harbor tasks `tasks/loophole-alpha-{easy,medium,hard}` | done | harbor: oracle 0.938 / 0.903 / 0.935, nop 0, stop-early 0 (all logs independently replayed) |
| RRT-Connect reference (`loophole/planning/`, `scripts/build_alpha.py`) | done | all 3 variants: dense-valid, replays to separation (easy 15 / medium 29 / hard 26 actions) |
| Cost-to-go roadmap reward (`reward/roadmap.py`) | in progress | ~8k nodes / ~67k edges per variant, ctg monotone along the reference, wall guard tested; behind `reward.roadmap` (off by default) |
| Topological diagnostics (linking integral, loop words) | in progress | logged per action; `scripts/episode_report.py` plots them |
| Alpha easy / hard variants | done | gap clearance 12 / 6 / 3 mm; start ctg 1.12 / 2.60 / 2.42 |
| Rope puzzles (easy/medium/hard) + Harbor tasks | done | earlier milestones; oracle 0.82 / nop 0 / stop-early 0 via Harbor |
| Pose stream / recorder / replay | done | |
| Canonical OMPL comparison (`make fetch-alpha`) | done | local only, never committed |

## Next steps for teammates

1. **Roadmap reward**: flip `reward.roadmap` on for the Alpha tasks once an agent run
   confirms the progress signal is useful; optionally switch the reference to the roadmap
   shortest path (`build_alpha.reference_from_roadmap`).
2. **Agent runs**: `make agent TASKS=tasks MODEL=...` on the Alpha tasks; check whether
   OpenCode passes MCP images to the model (the text state is sufficient on its own).
3. **Roadmap reward in tasks**: `python scripts/sync_tasks.py --roadmap-reward` wires
   `reward.roadmap` into both the sim sidecar and the verifier.
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
