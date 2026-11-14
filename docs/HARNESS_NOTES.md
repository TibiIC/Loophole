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

## Observation profiles and images

The task's MCP tools come from `observation.profile` in the sim config
(`loophole/configs/alpha_training.yaml` → `sensors`, `alpha_planner.yaml` → `planner`):
* `sensors` (task `loophole-alpha`): `render` (rgb / depth / segmentation + camera model),
  `get_state` (proprioception), `get_contacts` (touch). No goal, distance or progress
  values are exposed (`test_sensors_profile_hides_privileged_information`).
* `planner` (task `loophole-alpha-planner`): adds `check_motion`, `check_poses`,
  `get_geometry`. These queries change nothing and cost nothing
  (`test_check_motion_agrees_with_move`).

**Do images reach the model?** OpenCode stores an MCP `ImageContent` result as a
tool-part attachment (a data URL) in its session. In the Gemini 4 Carbon runs of
2026-11-14 (old tool set), agents called `render` and then also fetched the PNGs over raw
HTTP and decoded them in Python. One agent wrote a PNG decoder, then a C++ space-carving
reconstruction from 52 renders. The trajectories do not show whether the model saw the
pixels natively; the agents behaved as if they needed numeric access. We therefore
provide pixel-exact numeric channels (metric depth + camera model + segmentation) and a
compute stack in the agent image, so the vision task does not depend on the harness
forwarding images.

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
