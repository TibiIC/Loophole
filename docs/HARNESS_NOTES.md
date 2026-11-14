# Harness notes

## Task and puzzle

The submission task is `tasks/loophole-alpha`: **Alpha 1.0**, two identical interlocked
alpha tubes. Only this variant is in scope. Other Alpha variants are not built as tasks
or evaluated; legacy code for them is kept but unused.

## Provenance and licence of the Alpha 1.0 geometry and solution

| item | source | committed? |
|---|---|---|
| tube meshes `alpha-1.0.dt.tcp.obj`, `alpha_env-1.0.dt.tcp.obj` | `gitlab.com/xinyazhang/puzzle-geometry` @ `cd05c5da442a526b13b0abb1c072e6e1c061aede`, `puzzles/alpha/` (byte-identical to the paper's supplementary `supp-material/puzzles/alpha/`) | **no**: fetched with sha256 checks by `scripts/fetch_alpha_paper.sh` (locally into `.alpha_paper/`, and at Docker build time) |
| problem file `alpha-1.0.cfg`, full solution path `alpha-1.0.ompl.txt` | same | **no**: fetched |
| simplified solution `alpha-1.0.simplified.txt` (22 poses) | paper supplementary `supp-material.zip` (sha256 `00dff255…07c`), https://xinyazhang.gitlab.io/puzzletunneldiscovery/ | the 22 poses **yes** (`loophole/puzzles/alpha_paper/alpha-1.0-paper-reference.npz`), plus derived rig commands (`loophole/reference/trajectories/alpha-1.0-paper.jsonl`, `tasks/loophole-alpha/solution/reference.jsonl`) and the roadmap seeded from them |

* Paper: X. Zhang, R. Belfer, P. G. Kry, E. Vouga, *C-Space Tunnel Discovery for Puzzle
  Path Planning*, ACM TOG 39(4), Article 104 (SIGGRAPH 2020),
  doi:10.1145/3386569.3392468. Attribution is in `REFERENCES.md` and in the header of
  every oracle file.
* **Licence findings (checked 2026-11-14):**
  * `github.com/xinyazhang/PuzzleTunnelDiscovery` (the authors' code) has
    `LICENSE.GPL2` (GPL-2.0). It does not contain the Alpha 1.0 solution.
  * `gitlab.com/xinyazhang/puzzle-geometry` (the puzzle files) has **no LICENSE file**
    (only `README.md`, "Placeholder", and `.gitignore`); its GitLab licence field is empty.
  * The supplementary archive states no licence.
  * The meshes derive from the OMPL.app Alpha puzzle, under the Rice University Software
    Distribution License (non-commercial; *no redistribution*). Hence they are never
    committed.
* Decision (project owner): commit the 22 published solution poses and our derived
  oracle and roadmap, with attribution. **Revisit before the repository goes public:**
  without a stated licence, redistribution of the poses is not explicitly granted. Ask
  the authors, or switch the oracle to fetching the supplementary at run time.

## The oracle runs in our simulator, not only in the authors' planner

The published path was checked in **our** pipeline:
1. FCL on the raw meshes: 0/214 states in collision, and 0/66,679 interpolated states at
   0.8 mm.
2. Our MuJoCo collision model: 0/28,189 dense states of the 22-pose path; MuJoCo vs FCL
   distances agree to 5.8 µm on the 214 states.
3. A **Harbor oracle run**: `solution/solve.sh` replays the 79 rig commands through the
   MCP tools (`grab`/`move`) into the simulator sidecar, with the per-substep penetration
   audit active. The separate verifier then replays the logged actions in a fresh
   simulator and recomputes the score.

Result (old distance goal, min distance > 0.4 m): reward 0.8005, 79 actions, 0 violations,
log consistent, sidecar agrees. Under the current goal (free escape by translation,
`goal.criterion: escape`) the same commands free tube B after **59 actions: score 0.851**
(local replay; the trajectory header records `expected_score` 0.851, `distance_goal_score`
0.8005).

## What is withheld from the agent: a solution oracle, not task state

The agent gets everything needed to reason about the task:
* cameras: rgb, metric depth, segmentation and the full camera model, from any viewpoint;
* tube B's pose relative to tube A, and its own handle;
* contact points, normals, penetration and forces;
* the outcome of every action (executed motion, `blocked`, `clamped`);
* the action budget and the rig limits.

What it does not get is any **solution signal**: the roadmap cost-to-go, the derived progress
value, the goal test or escape direction. These stay verifier-side. Exposing them turns the
task into black-box descent on our own planner's value function: on the original interface,
one run (`Kt5DJ3h`) did exactly that by finite differences and failed (docs/ANOMALY.md).

**Proximity (minimum distance between the tubes) is deliberately excluded**, in any form,
including a proximity value in `get_contacts`. It is a hand-made shaping signal rather than
something a robot senses. The old goal was defined through it (min distance > 0.4 m), and the
puzzle's real obstruction is topological and geometric, not metric. Contact feedback (actual
touch) is provided.

## Observation profiles and images

The task's MCP tools come from `observation.profile` in the sim config
(`loophole/configs/alpha_training.yaml` → `sensors`, `alpha_planner.yaml` → `planner`):
* `sensors` (task `loophole-alpha`): `render` (rgb / depth / segmentation + camera model),
  `get_state` (proprioception), `get_contacts` (touch). No goal, distance or progress
  values are exposed (`test_sensors_profile_hides_privileged_information`).
* `planner` (task `loophole-alpha-planner`): adds `check_motion`, `check_poses`,
  `get_geometry`. These queries change nothing and cost nothing
  (`test_check_motion_agrees_with_move`).

**Do images reach the model? Not with OpenCode + `gemini/gemini-4-carbon`.** OpenCode stores
an MCP `ImageContent` result as a tool-part attachment (a data URL) in its session, but it
does not forward it to this model. The solving agent of the 2026-11-14 eval (trial
`5oFTZb8`) said so in its final message: "Direct image attachments from `render` could not be
displayed directly because this model configuration does not accept image inputs". Most
likely OpenCode's model registry has no image-input capability for this model id; not
verified in OpenCode's source.

The agents worked around it. They called the MCP server over raw HTTP from Python, decoded
the PNGs with a hand-written decoder, and did vision in code. The solving agent:
1. rendered 50+ calibrated views;
2. reconstructed both tubes by space carving in C++;
3. planned a path with its own sampling planner on the reconstruction's distance field;
4. verified and smoothed the path offline;
5. executed it: 63 moves, never blocked, free after 64 actions.

This is why the redesigned `render` returns pixel-exact numeric channels (metric depth,
segmentation, full camera model) and the agent image ships numpy/scipy/Pillow/OpenCV: vision
works through code even when the harness does not pass images to the model. A model/harness
pair that does accept images gets the same PNGs directly.

## Reproducibility

* **Pinned:** both base images by digest (`python:3.11-slim-bookworm@sha256:a83ba18a…`);
  direct Python dependencies in `pyproject.toml`; transitive ones in
  `docker/constraints.txt` (frozen from the sim image); the agent image's compute stack
  (numpy 2.4.6, scipy 1.17.1, Pillow 13.0.0, OpenCV 5.0.0.93, mcp 1.30.0); the geometry by
  commit + sha256; OpenCode 1.19.5 via Harbor 0.25.0. Apt packages follow Debian bookworm
  and are not pinned.
* **Seeds:** the episode seed is `LOOPHOLE_SEED=0` (task.toml + compose); the roadmap was
  built with seed 1 (stored in the `.npz`); the escape directions are a deterministic
  Fibonacci lattice. Two fresh simulators given the same actions reach bit-identical
  states (`test_replay_is_deterministic`), which is what the verifier's per-action
  state-hash check relies on.
* **One episode per session:** the MCP server runs exactly one episode per process. After
  it ends (solved, budget, violation, or `/admin/finalize`), every action tool is refused
  for every client, including new MCP sessions. There is no reset tool and no reset route
  (`test_no_second_episode_without_admin_reset`).

## Network

* **Docker build needs network access:** `docker/sim.Dockerfile` downloads the Alpha 1.0
  meshes, `.cfg` and solution (pinned commit, sha256) and pip packages.
* **Agent runs have no internet.** `[agent] network_mode = "allowlist"` with
  `allowed_hosts = ["loophole-sim"]`; Harbor additionally auto-allowlists the agent's own
  model endpoint. Agent installation (`agent.setup()`) still uses the public baseline.
* The simulator sidecar declares `networks: [default]`. That keeps it out of Harbor's
  egress-control namespace, so it keeps its own DNS name and network namespace, and its
  loopback-only `/admin/*` routes are unreachable from the agent container. Without this,
  Harbor would place it in the agent's network namespace (`loophole-sim` would not
  resolve, and the admin route would become reachable over shared loopback).
* The verifier runs in a separate container with `network_mode = "no-network"`.

## Model access

OpenCode with Gemini reads only `GOOGLE_GENERATIVE_AI_API_KEY`. Harbor forwards the first
of `GOOGLE_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` it finds, so set
only that one. Models: `gemini/<id>` (e.g. `gemini/gemini-4-carbon`).

## Baselines

* **noop**: Harbor's built-in `nop` agent (`harbor run -p ./tasks/loophole-alpha -a nop`):
  no actions; score 0.
* **stop-early**: a scripted agent. The oracle harness runs `solution/solve.sh` with
  `LOOPHOLE_SOLUTION=stop_early`, which replays `solution/stop_early.jsonl` over MCP: `grab`
  plus 5 seeded random valid moves (each within the rig caps), then stops
  (`harbor run -p ./tasks/loophole-alpha -a oracle --ae LOOPHOLE_SOLUTION=stop_early`).
  Score 0: it never holds a lower cost-to-go for 3 consecutive actions.
