# Loophole

An RL environment (Harbor + MCP) where an LLM agent solves 3D **disentanglement puzzles**
in a MuJoCo simulation. Primary puzzle: an **Alpha-style puzzle**, two identical
alpha-shaped tubes interlocked through each other's loops; tube A is fixed, the agent
moves tube B with a grab handle and must separate them. The agent only talks to the
simulator over MCP (sidecar container), so it never sees the simulator, solver or
reference.

## The Alpha puzzle task (`tasks/loophole-alpha`): Alpha 1.0

**What the agent must do.** Two identical rigid tubes, each bent into an alpha, are
interlocked. Tube A is fixed; the agent grabs tube B with one handle and moves it with
bounded steps (≤ 10 cm and ≤ 0.2 rad per `move`, velocity-limited) until the minimum
distance between the tubes exceeds **0.4 m**, within 198 actions. It sees the world only
through the MCP tools `get_state`, `render`, `grab`, `move`, `release` and `wait`.

**Why it is hard.** The way out is a narrow tunnel in the 6-D configuration space SE(3):
a coordinated sequence of translations *and* rotations with millimetre clearance (the
published solution grazes contact). Greedy "move away" motions stall against the other
tube, and Euclidean closeness is deceptive across tube walls. Sampling planners need
targeted tunnel discovery [ZBKV20]; even the easier Alpha 1.5 has two homotopy classes of
solution paths [JSCP15]. The puzzle was designed by Boris Yamrom (GE Corporate R&D) and
became a standard benchmark through Nancy Amato's Parasol Lab and OMPL.

**Geometry and oracle: from the paper, fetched, never committed.** Alpha 1.0 exactly as
used in Zhang et al. [ZBKV20]: two congruent tube meshes (volumes 2870.33 / 2869.76), the
OMPL problem file and the authors' solution, from their puzzle repository
(`gitlab.com/xinyazhang/puzzle-geometry` @ `cd05c5d`, byte-identical to the paper's
supplementary). The geometry derives from the OMPL.app Alpha puzzle (Rice licence:
non-commercial, no redistribution), so `make fetch-alpha-paper` and the Docker build
fetch it with sha256 checks; only poses and parameters are committed.
* OMPL conventions (robot shifted by its vertex mean): all 214 states of the authors'
  path are collision-free under FCL on the raw meshes, and 66,679 interpolated states at
  0.8 mm too; our MuJoCo collision model (exact-outside convex face wedges) matches FCL to
  5.8 µm. `unit_scale: 0.04` (tubes ≈ 2.5 m); the start pose round-trips to the `.cfg`.
* Oracle (`solution/`): the authors' simplified 22-pose solution, densely re-validated
  in our simulator (28,189 states, 0 collisions) and replayed through the agent's rig:
  separates in 79/198 actions, 0 violations, score 0.80. Watch it: `make view-alpha`.

**Scoring** (`loophole/reward/literature_rewards.py: episode_score`). The verifier
writes one float in [0, 1] to `/logs/verifier/reward.txt`:

| outcome | score |
|---|---|
| any penetration/tunnelling violation (episode ends immediately) | 0 |
| solved (min distance > 0.4 m) | 1 − 0.5 · a/T, in [0.5, 1] |
| unsolved | 0.3 · p, in [0, 0.3] |

with `a` = actions used, `T` = max actions, and progress
`p = clip(1 − best_ctg / start_ctg, 0, 1)`. `ctg` is the cost-to-go of a roadmap over the
free configuration space (shortest collision-free path length to separation), matched only
through roadmap nodes reachable by a collision-free straight segment; `best_ctg` only
updates when the match holds for ≥ 3 consecutive actions. Every solve beats every
non-solve; doing nothing scores 0; going back and forth cannot raise `best_ctg`.

**Training signals** (logged per action in the episode JSONL, never in the score):
`cspace_tunnel` (primary, potential-based shaping Φ = −ctg with γ = 1 [NHR99]; its
terminal is `episode_score`), `clearance_shaping` (potential-based clearance kernel adapted
from IndustReal [TLA+23], weight 0) and `gauss_linking_diagnostic` (open-curve Gauss
integral [KL00, RN11], logged next to cost-to-go, no reward). Citations, verified, are
in [REFERENCES.md](REFERENCES.md).

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
> `tasks/loophole-alpha-{easy,medium,hard}`) are not the real Alpha geometry, and the
> OMPL.app Alpha 1.5 pipeline (`loophole/puzzles/alpha_ompl/`) has non-identical tubes
> (OMPL widens tube A). Both are kept for reference and tests; the active task is Alpha 1.0.

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
