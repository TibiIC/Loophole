# Loophole

An RL environment (Harbor + MCP) where an LLM agent solves 3D **disentanglement puzzles**
in a MuJoCo simulation. Primary puzzle: an **Alpha-style puzzle**, two identical
alpha-shaped tubes interlocked through each other's loops; tube A is fixed, the agent
moves tube B with a grab handle and must separate them. The agent only talks to the
simulator over MCP (sidecar container), so it never sees the simulator, solver or
reference.

## The two tasks and where their reference solutions come from

| Task | Reference (`solution/`) | Provenance | Solver anywhere in the pipeline? |
|---|---|---|---|
| `tasks/loophole-rattrap`: the Rat Trap | fixed waypoints (`solution/waypoints.jsonl`) | **A human solve.** Tibi's teleoperated solve, recorded in our simulator (`loophole/reference/rattrap-solve.jsonl`). Its waypoints are fixed and committed; at replay a controller moves the gripper toward the next waypoint from its actual position. | **None**, not at runtime, offline or as a fallback. |
| `tasks/loophole-alpha`: Alpha 1.0 | 59 fixed commands | **A published, planner-generated path.** The solution path of Zhang et al. 2020 ([ZBKV20]), produced by their C-space tunnel-discovery planning pipeline, taken from their supplementary data (`alpha-1.0.simplified.txt`), converted once into agent commands and committed. | None of ours at runtime. The path was computed by the paper authors' planner, not by us. |

For both, the "oracle" is Harbor's name for the agent that replays `solution/`; it searches
for nothing. The difference matters: the Rat Trap shows a human can solve the environment as
built, while the Alpha reference shows the authors' planner's path is feasible in our
simulator.

Our own RRT-Connect planner (`loophole/planning/`) produced references only for the
procedural Alpha variants (`alpha-easy` / `medium` / `hard`, not submitted) and for an
earlier OMPL.app Alpha 1.5 task (superseded). An attempt to plan an independent Alpha 1.0
reference did not succeed, so none is committed.

## The Alpha puzzle task (`tasks/loophole-alpha`): Alpha 1.0

**What the agent must do.** Two identical rigid tubes, each bent into an alpha, are
interlocked. Tube A is fixed; the agent grabs tube B with one handle and moves it with
bounded steps (≤ 10 cm and ≤ 0.2 rad per `move`, velocity-limited) until tube B **can
slide away to infinity along some straight line without touching tube A** (free escape
by translation, see [Goal](#goal-free-escape-by-translation)), within T = 198 actions.

**Why it is hard.** The way out is a narrow tunnel in the 6-D configuration space SE(3):
a coordinated sequence of translations *and* rotations with millimetre clearance (the
published solution grazes contact). Greedy "move away" motions stall against the other
tube, and Euclidean closeness is deceptive across tube walls. Sampling planners need
targeted tunnel discovery [ZBKV20]; even the easier Alpha 1.5 has two homotopy classes of
solution paths [JSCP15]. The puzzle was designed by Boris Yamrom (GE Corporate R&D) and
became a standard benchmark through Nancy Amato's Parasol Lab and OMPL.

**Geometry and oracle: from the paper, fetched, never committed.** Alpha 1.0 exactly as
used in Zhang et al. [ZBKV20]: two congruent tube meshes, the
OMPL problem file and the authors' solution, from their puzzle repository
(`gitlab.com/xinyazhang/puzzle-geometry` @ `cd05c5d`, byte-identical to the paper's
supplementary). The geometry derives from the OMPL.app Alpha puzzle (Rice licence:
non-commercial, no redistribution), so `make fetch-alpha-paper` and the Docker build
fetch it with sha256 checks; only poses and parameters are committed.
* OMPL conventions (robot shifted by its vertex mean). In our MuJoCo collision model
  (exact-outside convex face wedges), the authors' 22 keyframe poses, densified to 28,189
  states, are all collision-free (`loophole/puzzles/alpha_paper/alpha-1.0-paper-build.json`:
  `dense_states_checked`, `invalid: 0`). `unit_scale: 0.04` (tubes ≈ 2.5 m).
* Oracle (`solution/`): the authors' simplified 22-pose solution, densely re-validated
  in our simulator (28,189 states, 0 collisions) and replayed through the agent's rig:
  free after 59/198 actions, 0 violations, score **0.851**. Watch it: `make view-alpha`.

### Input space: what the agent observes

The agent is an off-the-shelf coding agent (OpenCode) talking to the simulator **only over
MCP**. Its observations are tools. Observation tools are free; each action tool costs one
action. Which observation tools exist is set by the task's **observation profile**
(`observation.profile` in the sim config). The same actions on the same seed give the same
score under every profile (tested).

| tool | modality | returns | profiles |
|---|---|---|---|
| `render(view, azimuth, elevation, distance, lookat, modalities, width, height, depth_format)` | **vision** | PNG images: `rgb`, `depth` (8-bit "visual", or 16-bit millimetres "metric") and/or `segmentation` (background / tube A / tube B / handle), 64–1280 × 64–960 px, any viewpoint; plus a JSON caption with the **full camera model** (position, forward/up/right axes, fov, intrinsics f<sub>x</sub>, f<sub>y</sub>, c<sub>x</sub>, c<sub>y</sub>) | all |
| `get_state()` | **proprioception** | pose of tube B (position + quaternion, in tube A's frame), the handle (holding what, where), the last action's outcome (executed translation/rotation, `blocked`, `clamped`), action budget, `done`/`success`, rig limits | all |
| `get_contacts()` | **touch** | current A–B contacts: point, normal (into tube B), penetration depth, normal force; total contact force/torque; grip lag (how far the held tube trails the handle) | all |
| `check_motion(...)` | planning query | dry run of `move` with identical clamping and kinematics: collision-free? which fraction is free? end pose | `planner` |
| `check_poses(poses)` | planning query | collision-free flag for up to 256 absolute poses of tube B (a state-validity checker) | `planner` |
| `get_geometry()` | planning query | triangle meshes of both tubes (body frames, metres) and their poses | `planner` |

Profiles: **`sensors`** = vision + proprioception + touch (task `tasks/loophole-alpha`);
**`planner`** = `sensors` + planning queries (task `tasks/loophole-alpha-planner`);
**`full`** additionally shows privileged fields (goal test, cost-to-go progress, scene
geometry) and is for debugging only. Under `sensors`, nothing about the goal, the
distance between the tubes or the progress measure is shown (tested).

Depth pixels back-project exactly to 3-D points (≤ 0.5 mm from the true surface, tested):
`point = position + depth · (forward + (u + ½ − c_x)/f_x · right − (v + ½ − c_y)/f_y · up)`,
with (u, v) = (column, row) from the top-left and depth measured along the optical axis.
So an agent can do multi-view reconstruction or segmentation-guided pose estimation. The
agent container has Python 3.11 with numpy, scipy, Pillow, OpenCV and a C/C++ compiler (no
internet during the run), so it may write its own perception or planning code and call the
MCP server from it. One Gemini 4 Carbon agent did exactly that, space-carving the tubes from
52 renders in C++.

### Output space: what the agent does

| tool | effect |
|---|---|
| `grab(handle=0, target="tube_b", point?)` | weld handle 0 to tube B (grasp point within 5 cm of the tube; default its reference point) |
| `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)` | relative SE(3) motion of the handle and the held tube; translation clamped to 0.1 m, rotation (XYZ Euler about the grasp point, `world` or `local` axes) to 0.2 rad; executed at ≤ 0.1 m/s and 0.1 rad/s; stops early (`blocked`) when the tube is obstructed |
| `release(handle)` | open the handle |
| `wait(steps)` | 1–4000 physics substeps (1 ms) without moving |

Formally the action space is A = {grab, release} × {0} ∪ {move} × B<sub>0.1</sub>(ℝ³) ×
B<sub>0.2</sub>(so(3)) × {world, local} ∪ {wait} × {1, …, 4000}. Invalid arguments
(NaN, ±inf, wrong types, out of range) are rejected before any physics and cost nothing.
Nothing can pass through anything else: a per-substep audit (penetration > 4 mm,
per-substep displacement > 3 mm, integrator divergence) ends the episode with score 0.

### Goal: free escape by translation

Let q = (t, R) ∈ SE(3) be tube B's pose in tube A's frame, B(q) the region tube B
occupies, and A tube A's region. The goal set is

  G = { q : ∃ u ∈ S², ∀ s ≥ 0 : B(t + s·u, R) ∩ A = ∅ },

tube B can be slid to infinity along some straight line without touching tube A. This is
the removability-by-translation notion of assembly planning [WL94]. Unlike a distance
threshold, it has no tunable target: it is exactly "no longer caged".

*Implementation* (`sim/collision.py: escape_direction`): 64 fixed directions on a
Fibonacci sphere. Each is swept from the current pose until the bounding spheres of the two
tubes are disjoint, after which contact is impossible. Far from tube A the sweep jumps by
the separation distance (conservative advancement [Mir96]: a translation shorter than the
distance cannot touch), using a convex-chunk proxy whose distance is a lower bound. Near
tube A it falls back to the simulator's exact penetration test every 4 mm. The check is
**sound**: a reported direction is free (a dense 1 mm exact sweep confirms this in the
tests). It is **conservative**: a direction that needs more than 20 cm of sliding contact
is treated as blocked, which can only delay success. The check costs about 0.3 s per action
and is re-run by the verifier.

### Rewards: formal definitions, intuition, sources

The complete reference for every reward component in both environments (scored, shaping and diagnostic, each with
its definition, a worked example from a committed run and its provenance) is [docs/rewards.md](docs/rewards.md).

**Episode score: the only number the verifier writes** (`/logs/verifier/reward.txt`;
`reward/literature_rewards.py: episode_score`). With a = actions used, T = 198, V = audit
violations:

| outcome | S | range |
|---|---|---|
| V > 0 (forced through; episode ends immediately) | 0 | 0 |
| solved: q<sub>t</sub> ∈ G at some action t | 1 − α · a / T | [0.5, 1] |
| unsolved | κ · p | [0, 0.3] |

with α = 0.5, κ = 0.3 and progress p = clip(1 − c\* / c<sub>0</sub>, 0, 1). Here:
* **c(q)**, the cost-to-go, is the length of the shortest collision-free path from q to
  the separated set, estimated on a roadmap of N = 8044 free configurations and 60,260
  collision-checked edges: c(q) = min<sub>n ∈ V(q)</sub> [ d(q, n) + ctg(n) ]. ctg(n) is
  the Dijkstra distance from node n to the 847 roadmap nodes that are clearly apart
  (minimum distance > 0.4 m). V(q) is the set of the 10
  nearest nodes that q can reach by a collision-free straight segment, and
  d(q₁, q₂) = ‖t₁ − t₂‖ + w · ∠(R₁ᵀR₂) with w = 1.54 m (tube B's radius).
* **c\***, the best *held* cost-to-go, is min<sub>t</sub> max(c<sub>t</sub>, c<sub>t+1</sub>,
  c<sub>t+2</sub>) over windows of 3 consecutive actions that all have a visible roadmap
  node (`StableMin`). c<sub>0</sub> = c(q<sub>0</sub>) = 21.6.

*Intuition.* Forcing the tubes together is worthless. Any solve beats any non-solve, and
faster solves score more. A non-solve earns credit for how far along the shortest
collision-free way out it got. Distance through a tube wall does not count: visibility
blocks it. A lucky single reading does not count: it must be held for 3 actions. Wiggling
does not count either: c\* only reaches a level after 3 actions at or below it. Doing
nothing scores 0. The roadmap measures *progress along a way out* rather than "distance to
one reference path", because a puzzle can have several homotopy classes of solutions
[JSCP15]. Tunnel structure of C-space is the core difficulty of this puzzle [ZBKV20].

**Training signals** (logged per action in the episode JSONL for RL / distillation; never
part of S):

1. `cspace_tunnel` (primary, weight 1). Potential-based shaping on the C-space navigation
   function, Φ(q) = −c(q), γ = 1:
   r<sub>t</sub> = (c(q<sub>t−1</sub>) − c(q<sub>t</sub>)) / c<sub>0</sub>, and its
   terminal reward is exactly S. *Intuition:* a dense signal for "moved along the way out".
   Potential-based shaping leaves optimal policies unchanged and telescopes, so going back
   and forth nets nothing [NHR99]. Source of the tunnel view: [ZBKV20].
2. `clearance_shaping` (weight 0, logged). Φ(s) = exp(−max(0, d<sub>sep</sub> − d(s)) / σ)
   with d(s) the A–B minimum distance, d<sub>sep</sub> = 0.4 m, σ = 0.08 m:
   r = Φ(s′) − Φ(s) − w<sub>pen</sub> · max(0, pen − 2 mm) − w<sub>act</sub> · (‖Δt‖² + 0.1 ‖Δθ‖²),
   using the motion actually executed. *Intuition:* "more room is better, don't press,
   don't flail". It is adapted from IndustReal's SDF-based dense reward [TLA+23] (built on
   Factory [NSA+22]), which measures alignment with a goal pose. Clearance can grow while
   the tubes stay caged, hence weight 0.
3. `gauss_linking_diagnostic` (log only). The open-curve Gauss linking integral of the two
   tube centre-lines, Lk = (1/4π) ∮∮ (r₁ − r₂) · (dr₁ × dr₂) / ‖r₁ − r₂‖³, computed with
   the exact segment-pair formula [KL00] (background [RN11]). *Intuition:* how wound around
   each other the tubes are. For open curves it is real-valued and changes under rotation
   in place, so it is never rewarded. (Topological models of *string* puzzles: [PS02].)

All citations were checked against their DOI, arXiv or publisher records; see
[REFERENCES.md](REFERENCES.md).

**Anti-exploit setup.** Harbor supports multi-container environments, so the MuJoCo MCP
server runs in a **separate sidecar container** (`loophole-sim`). The agent's container
has no simulator code, meshes, solver or episode log; it can only reach the MCP
endpoint. After the agent finishes, a collect hook in the sidecar finalises the episode
(the agent container is already stopped). The sidecar's episode directory goes to a
**separate verifier container** (sim image, no network). That container replays every
logged action in a fresh simulator and recomputes the score, checking puzzle hash, config,
initial state and every per-action state hash. Nothing the agent writes is read.

**Reproducing a score: re-run, don't replay an archived Alpha log.**
* A run is verified inside the image build that recorded it. The verifier replays every action and requires every
  per-action state hash to match, bit for bit.
* **The Alpha image is not bit-reproducible across builds.**
  * Its collision geometry is regenerated at every build. In two clean builds of the same commit, 22 of 5,592 convex
    pieces differ, by about 1.4e-5 in vertex coordinates.
  * The differences are confined to face-wedge pieces, whose inward depth is set by ray casting in the face-wedge
    decomposition. Python packages and compiled libraries are identical across the two builds.
  * Replays then drift in the last bits: the same 59 actions, the same 0.851010, poses agreeing to six decimal places.
    But the state hash differs from action 2, so an archived Alpha log re-verified in a different build scores 0: it
    fails closed.
  * Of the 31 committed Alpha trials, only the 5 zero-action episodes re-verify in a rebuilt image.
* **The Rat Trap is reproducible across builds.** Its geometry is boxes and capsules, with no mesh decomposition. Its
  archived closed-loop oracle log (`runs/harbor-rattrap/rattrap-oracle-closedloop/`) re-verifies in an independently
  rebuilt image: 0.833333, solved, 20 / 20, every per-action state hash matching.
* **What holds:**
  * Every archived score was computed at run time by Harbor's separate, no-network verifier, with
    `log_consistent: 1`.
  * Only cross-build re-verification of archived Alpha logs fails.
  * A fresh evaluation reproduces. From a clean HTTPS clone of main and a cleared Docker cache, the Alpha reference
    scored exactly 0.851010 and the do-nothing baseline 0 (build 132 s, reference 248 s, noop 35 s). The Rat Trap
    reference scored 0.833333, solved, 20 / 20.
* Judges therefore reproduce a result by re-running it (`harbor run -p ./tasks/loophole-alpha ...`), not by
  replaying an archived Alpha log.

**One-command reproduction** (Docker + `uv tool install harbor`; the image build fetches
the meshes):
```bash
harbor run -p ./tasks/loophole-alpha -a oracle                       # reference solution (0.851)
harbor run -p ./tasks/loophole-alpha -a nop                          # do-nothing baseline
harbor run -p ./tasks/loophole-alpha -a oracle --ae LOOPHOLE_SOLUTION=stop_early
OPENAI_API_KEY=... harbor run -p ./tasks/loophole-alpha -a opencode -m openai/<model>
harbor run -p ./tasks/loophole-alpha-planner -a opencode -m <model>  # + planning queries
python scripts/sync_tasks.py --alpha                                 # regenerate both Alpha tasks
```

> **Deprecated:** the procedural Alpha-*style* tubes (`loophole/puzzles/alpha/`,
> `tasks/loophole-alpha-{easy,medium,hard}`) are not the real Alpha geometry, and the
> OMPL.app Alpha 1.5 pipeline (`loophole/puzzles/alpha_ompl/`) has non-identical tubes
> (OMPL widens tube A). Both are kept for reference and tests; the active task is Alpha 1.0.

### Before reading a failed run: harness findings

Two settings decide whether some models get to act at all. Both fixes are built into the
tasks; details and evidence are in [docs/HARNESS_NOTES.md](docs/HARNESS_NOTES.md#running-agents-configuration-failures-that-look-like-capability-failures).

* **Gemini 4 Carbon needs more than OpenCode's 32,000-token output cap on the Rat Trap.**
  * Its opening plan takes 50,324 reasoning tokens. At the default cap the turn is cut off and
    returns nothing, so a score of 0 there is a configuration result, not a capability one.
  * The Rat Trap task raises the cap to 65,536 (`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` and
    `limit.output`), keeps OpenCode's default thinking level, and streams the reasoning
    (`includeThoughts`). All of this is in `tasks/loophole-rattrap/environment/docker-compose.yaml`.
    An earlier pin to `thinkingLevel: low` made the model act, but with almost no reasoning; it
    has been replaced.
  * The Alpha task keeps OpenCode's defaults: no Alpha turn has reached the cap, and the reported
    Alpha results were produced at the default.
* **Nemotron 3.5 Super VL truncates at 8k output tokens.** Our function-calling harness now
  allows 32k and re-prompts a cut-off reply instead of ending the run.

`scripts/harness_audit.py` checks every run for this class of failure: output-cap
truncation, permission blocks, unknown tools, API errors, and runs that never acted. Excluded
on that basis: 3 Alpha trials of `gemini-2.5-flash` (the model is no longer available to new
API users) and one Gemini 4 Carbon Alpha trial ended by an API 400.

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

## The Rat Trap: a cord-and-frame puzzle

A closed string loop, with a mouse bead tied on, starts wrapped around a dowel of a wooden
frame. The frame has posts, dowels through holes, a ring and a sphere. The agent frees the
loop with two grippers: `grab(node)`, `move(dx, dy, dz)` (at most 30 mm per call, halting when
blocked) and `release`, in MuJoCo with an inextensible capsule-chain cord.
Geometry: `loophole/puzzles/rattrap/rattrap.yaml`; every dimension is provisional.

**Reward** (full derivation, evidence and sources: [docs/RATTRAP_DESIGN.md](docs/RATTRAP_DESIGN.md)):
* **Violation → 0.** Any contact pressed deeper than 1 mm ends the episode.
* **Solved → 1 − 0.5·a/T**, T = 852. Solved means the frozen cord, mouse included, can slide
  free in a straight line.
* **Unsolved → 0.3·p**, where p is the fraction of the 20 checkpoints of a reference solve
  reached in order. A checkpoint is reached when the max per-node displacement is under
  ε = 16.6 mm, calibrated on the reference.
* **Training shaping** is potential-based on the checkpoint index plus the distance to the
  next checkpoint, and is never scored.
* **Why staged geometric milestones:** in the rope and cable literature, topological
  signatures guide planning (Bhattacharya et al., RSS 2011; Mitrano & Berenson, ICRA 2024),
  while cable-routing RL uses staged subtasks with geometric success tests (SILO,
  arXiv:2607.04616). Our own topological word / Przytycki–Sikora norm turned out unsound
  here and is kept as a logged diagnostic only.
* **Reference solve: Tibi's recorded human solve, replayed closed-loop.**
  * The waypoints are fixed and committed (`solution/waypoints.jsonl`): the recording's
    grabs and releases, and its hand path simplified to 0.1 mm.
  * At replay a controller (`solution/follow_waypoints.py`) moves the gripper toward the next
    waypoint from where it actually is. That is the same correction the offline conversion
    performed, done live, so cross-platform numerical differences do not accumulate.
  * Why: replayed as a fixed list of 284 commands, the solve reached 20 / 20 checkpoints on
    macOS but only 18 / 20 on Linux (unsolved, 0.27), because the physics is not
    bit-identical across platforms.
  * Closed-loop, it solves on both: 284 actions, 20 / 20, 0.833, and through Harbor on Linux.
  * There is **no solver, planner or search anywhere in the Rat Trap pipeline**: not at
    runtime, not offline, not as a fallback.

**Through Harbor** (`tasks/loophole-rattrap`, the same structure as the Alpha task):
* a sim sidecar serves the MCP tools;
* the agent container has no simulator;
* a separate, no-network verifier replays the logged commands in a fresh simulator, checking
  a state hash per action.

```bash
harbor run -p ./tasks/loophole-rattrap -a oracle                          # the recorded human solve, replayed closed-loop: 0.833
harbor run -p ./tasks/loophole-rattrap -a nop                             # 0
harbor run -p ./tasks/loophole-rattrap -a opencode -m gemini/gemini-4-carbon   # gemini/, not google/ (Harbor forwards the key only for gemini/)
make reference TASKS=tasks/loophole-rattrap    # or: make noop / make agent TASKS=tasks/loophole-rattrap MODEL=...
```

**Baselines** (in-process, no Docker; after `make dev`):

```bash
make rattrap-oracle   # the reference solve as agent commands: solved in 284 actions, p = 1, score 0.833
make rattrap-noop     # no action: score 0
make rattrap-random   # random grabs / moves / releases to the full budget: p stays at 0
```

**Agents.** Both harnesses offer the same tools, scene text and scoring
(`loophole/puzzles/rattrap/agent_tools.py`):

```bash
# OpenAI-compatible function calling (Verda: DeepSeek, Kimi, Nemotron; keys in .env)
PYTHONPATH=$PWD .venv/bin/python scripts/agent_rattrap.py --model deepseek
# OpenCode with a native provider, over MCP (OpenCode's own tools disabled), as for Alpha
.venv/bin/python scripts/opencode_rattrap.py --model google/gemini-4-carbon --runs 3
```

<!-- rattrap-results:start -->
| Round | Model (provider, harness) | Reward | Score | Solved | Actions (of T) | Checkpoints reached | Blocked moves | Closest to c₁ |
|---|---|---|---|---|---|---|---|---|
| 1 | DeepSeek V4.1 Flash (Verda, function calling) | sparse, T = 400; re-scored² | 0.0 | no | 322 / 400 | 0 / 20 | 42 | 97.7 mm |
| 1 | Kimi K3 Pro (Verda, function calling) | sparse, T = 400; re-scored² | 0.0 | no | 357 / 400 | 0 / 20 | 47 | 70.1 mm |
| 1 | Nemotron 3.5 Super VL (Verda, function calling, with images) | sparse, T = 400; re-scored² | 0.0 | no | 343 / 400 | 0 / 20 | 35 | 110.8 mm |
| 1 | Gemini 4 Carbon (Google OpenAI-compatible API) | sparse, T = 400 | 0.0 | no | 0 / 400 | – | – | – |
| 2 | DeepSeek V4.1 Flash (Verda, function calling) | checkpoints, T = 852 | 0.0 | no | 493 / 852 | 0 / 20 | 43 | 88.1 mm |
| 2 | Kimi K3 Pro (Verda, function calling) | checkpoints, T = 852 | 0.0 | no | 852 / 852 | 0 / 20 | 141 | 63.4 mm |
| 2 | Nemotron 3.5 Super VL (Verda, function calling, with images) | **excluded³** (spurious release violation) | – | – | 501 / 852 | 0 / 20 | 48 | 94.3 mm |
| 2 | Gemini 4 Carbon (OpenCode, native Google provider), run 1 | **excluded³** (spurious release violation) | – | – | 89 / 852 | 0 / 20 | 17 | – |
| 2 | Gemini 4 Carbon (OpenCode, native Google provider), runs 0 and 2 | **excluded⁴** (reasoning capped by our `thinkingLevel: low` pin) | – | – | 298 / 852, 248 / 852 (OpenCode error / model stopped) | 0 / 20 | 80 / 59 | – |
| 2 | Gemini 4 Carbon (Harbor, `thinkingLevel: low`) | **excluded⁴** (0 reasoning tokens; 1,654 of 1,695 turns empty; timed out) | – | – | 25 / 852 | 0 / 20 | 6 | – |
| – | reference solve (oracle) | checkpoints, T = 852 | **0.833** | yes | 284 / 852 | 20 / 20 | – | 0 |
| – | noop | checkpoints, T = 852 | 0.0 | no | 0 | 0 / 20 | – | 110.8 mm (the start) |

² Round 1's command logs were replayed offline under the checkpoint reward (`scripts/rescore_rattrap.py`;
same physics, no API calls). **All three unsolved 300+-action runs land at p = 0.** None came
within ε = 16.6 mm of the first checkpoint: the closest was 70 mm, and Nemotron never got closer
than where it started. This is evidence both of the task's difficulty and of the reward's
integrity: hundreds of plausible-looking moves buy no partial credit.

³ **Excluded, not capability results.** Both ended the same way: a move was blocked with the
cord strained against the frame, and the agent's next `release` registered a 1.006 mm
penetration for a few substeps (tolerance 1 mm) before the cord settled at 0.1 mm. That is an
environment defect, found and fixed (docs/HARNESS_NOTES.md), not an agent error.

⁴ **Excluded, not capability results.** These runs had their reasoning capped by our own pin,
`thinkingLevel: low`: about 12 reasoning tokens per turn, and none at all in the Harbor run. The pin was a
workaround for OpenCode's 32,000-token output cap. The task now raises the cap to the endpoint's
real ceiling, 65,536, and keeps the default level (docs/HARNESS_NOTES.md §1). The runs are kept as the
evidence: `runs/rattrap-agents/`, `runs/harbor-rattrap/rattrap-gemini-thinking-low/`.

The Gemini 4 Carbon row in round 1 is a harness/API issue, not a verdict on the model. Through
Google's OpenAI-compatible endpoint it spent its whole 32k-token output budget on hidden
reasoning after the first tool result and returned nothing, with or without images;
`reasoning_effort` and `thinking_budget` were both ignored. Round 2 runs it through OpenCode's
native Google provider, the path that solves Alpha.
**The reference reproduces on Linux.** Replayed closed-loop from its fixed waypoints, Tibi's solve scores 0.833
(solved, 20 / 20 checkpoints, 284 actions) through Harbor in the Linux simulator container
(`runs/harbor-rattrap/rattrap-oracle-closedloop/`). Replayed as a fixed command list it had diverged there, to
18 / 20 checkpoints and 0.27 (`runs/harbor-rattrap/rattrap-oracle/`).
<!-- rattrap-results:end -->

## Quick start

**The whole evaluation of one task, one command** (Docker must be running; model key in the environment or `.env`):

```bash
./run_evaluation.sh alpha   [gemini/gemini-4-carbon]    # or: make judge TASK=alpha
./run_evaluation.sh rattrap [gemini/gemini-4-carbon]    # or: make judge TASK=rattrap
```

It installs Harbor 0.25.0 (and `uv`) if missing, builds the simulator image, and runs through Harbor:
* the reference (`oracle`, replaying `solution/`);
* the stop-early baseline;
* the do-nothing baseline (`nop`);
* three agent runs (OpenCode with the given model).

It prints one score per run (`jobs/judge-<task>-<time>/summary.md`).

Wall-clock on one loaded laptop (m = measured from the committed run logs; e = estimate):

| | Alpha | Rat Trap |
|---|---|---|
| first image build | 10–15 min (e; fetches the meshes) | same image |
| reference | 2.9 min (m) | ~1.5 h (e, from a 30-action Harbor run at ~9 s/action, replayed again by the verifier) |
| each baseline | ≤ 0.5 min (m) | noop 0.6 min (m) |
| each agent run | 55–74 min (m, Gemini 4 Carbon) | up to 4 h of agent time, plus up to ~2 h of verification (e) |

**If an agent stops acting (repeated empty responses).**
* **Seen with:** Gemini 4 Carbon through OpenCode's Google provider, on the Rat Trap, while the task pinned
  `thinkingLevel: low`. That pin was a workaround for OpenCode's 32k output cap. The task now raises the cap instead
  and keeps the default level. In a 50-step probe with the raised cap there were 0 empty steps (docs/HARNESS_NOTES.md
  §1).
* **How to spot it:**
  * in the agent transcript (`agent/opencode.txt` in the trial directory), step after step ends with
    `"reason":"unknown"`, 0 output tokens and 0 reasoning tokens, and there is no error event;
  * the action count stops rising (`artifacts/episode/commands.jsonl`, or `actions_used` in `get_state`);
  * the simulator sidecar sits idle, waiting for a request that never comes.
* **It is the model's behaviour, not the environment.**
  * In one Rat Trap run, 288 consecutive empty steps followed 25 actions.
  * Earlier Rat Trap runs had bursts of 20, 34 and 86 and recovered. The runs that stalled were the ones in which the
    model produced no reasoning tokens at all.
  * The stall is not caused by context size. On Alpha, at OpenCode's default level, the same model ran with 0 empty
    steps up to 176k tokens of context; the Rat Trap stalls at 26k–34k.
* **Remedy:** restart the run. A run in progress either recovers on its own or reaches the agent timeout and is scored
  on the actions it took.
* **Model id:** use `-m gemini/gemini-4-carbon` with Harbor, not `google/...`. Harbor forwards the key only for
  `gemini/`, and with `google/` the agent exits at once ("Google Generative AI API key is missing"). Harbor still
  starts OpenCode with its `google` provider, so the task's OpenCode settings apply.

Step by step, for development:

```bash
make dev                         # python3.11 venv + editable install (.[mesh,dev])
make test                        # pytest (artifact-dependent tests skip with a reason)
.venv/bin/python scripts/demo.py # loads the Alpha puzzle, makes a few moves, writes runs/demo/*.png
```

Run the MCP server (streamable HTTP on :8000/mcp; admin options are CLI/env only, never MCP tools):

```bash
.venv/bin/python -m loophole.mcp_server --puzzle alpha-medium --port 8000 --log-dir runs/episode
# or in Docker (OSMesa headless rendering):
make build && docker run --rm -p 8000:8000 -e LOOPHOLE_PUZZLE=alpha-medium loophole-sim:local
```

MCP tools: observation tools per profile (see [Input space](#input-space-what-the-agent-observes);
free), `grab`, `move`, `release`, `wait` (one action each). Every call is logged to `episode.jsonl`.

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

Each row points at the artifact behind it.

| Component | Status | Evidence |
|---|---|---|
| **Alpha 1.0 task `tasks/loophole-alpha`** (Zhang et al. 2020) | done | `runs/20261114-111735/`: oracle **0.851** (the authors' published path), nop **0**, stop-early **0**, Gemini 4 Carbon 0.884 / 0.823 / 0.886 |
| **Rat Trap task `tasks/loophole-rattrap`** | done | `runs/harbor-rattrap/`: Harbor oracle **0.833** (Tibi's recorded human solve, replayed closed-loop: solved, 20 / 20 checkpoints, Linux), nop **0**, stop-early **0**; random agent **0**: `runs/rattrap-random/` |
| *Earlier:* Alpha 1.5 task on OMPL.app geometry | superseded | its oracle was our own RRT-Connect plan: `runs/harbor/canon-oracle/` (0.876) |
| Real model runs (`-a opencode`) | done | Alpha: docs/RESULTS.md; Rat Trap: the README results table and `runs/rattrap-agents/` |
| Cost-to-go roadmap (Alpha partial credit) | done | `loophole/puzzles/alpha_paper/roadmap-alpha-1.0-paper.npz` (8,044 nodes, 60,260 edges) |
| Independent replay verifiers (Alpha, Rat Trap) | done | `loophole/reward/verify.py`, `loophole/puzzles/rattrap/verify.py`; tests `test_reward.py`, `test_rattrap_verify.py` |
| Penetration / speed / instability audit; swept-motion certificate | done | `loophole/sim/audit.py`, `loophole/sim/certify.py`; `test_shaping_scaffold.py` |
| Procedural Alpha variants, rope puzzles, OMPL comparison | kept, not submitted | code and tasks remain; no results are claimed for them |

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
6. **Deterministic Alpha geometry**: make the face-wedge decomposition deterministic (fixed ray-casting order), with a
   test that two clean builds produce byte-identical convex pieces. Archived Alpha logs would then re-verify in any build
   of the same commit.

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
