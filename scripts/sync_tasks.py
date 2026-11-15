#!/usr/bin/env python3
"""Generate the Harbor task directories tasks/loophole-<puzzle>/ from one template.

    python scripts/sync_tasks.py            # all puzzles that have a reference trajectory

Each task:
  environment/Dockerfile          agent container ("main"): python + mcp client only
  environment/docker-compose.yaml sim sidecar "loophole-sim" built from the repo root
  solution/                       oracle: replays a stored trajectory over MCP
  tests/test.sh                   documentation only: the real verifier is baked into
                                  the loophole-sim image (/tests/test.sh)
"""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUZZLES = ROOT / "loophole" / "puzzles"
TRAJ = ROOT / "loophole" / "reference" / "trajectories"
TASKS = ROOT / "tasks"

# Rat Trap only: agent-container environment with an OpenCode overlay (JSON, merged by OpenCode over Harbor's
# opencode.json). Alpha keeps OpenCode's defaults: its reported Gemini results were produced at the default level.
RATTRAP_AGENT_ENV = """\
    environment:
      # OpenCode agent settings (docs/HARNESS_NOTES.md §1). OpenCode caps each turn at
      # min(limit.output, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX), default 32,000 tokens. Gemini 4 Carbon's opening plan
      # on this task takes ~50k reasoning tokens, so at the default cap it is truncated and returns nothing.
      OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX: "{output_token_max}"
      # Inline config, read with the highest priority over the opencode.json Harbor writes (MCP servers, provider).
      # The thinking level is OpenCode's default (not pinned); includeThoughts streams the reasoning, without which a
      # turn longer than ~30k tokens outlasts OpenCode's 5-minute wait for the response headers.
      OPENCODE_CONFIG_CONTENT: '{opencode_overlay}'
"""

OUTPUT_TOKEN_MAX = 65536  # the Gemini endpoint's real per-turn ceiling for gemini-4-carbon (measured: a 131,072 request stops at 65,536)
# limit.output keeps the cap when OpenCode loads the models.dev catalog (which lists 32,768 for this model). OpenCode
# rejects a limit without `context`, and context also sets when it compacts history, so it is the model's real input
# window: 1,048,576 tokens, the inputTokenLimit from Gemini's models.get (models.dev lists the same).
CONTEXT_WINDOW = 1048576
OPENCODE_OVERLAY = json.dumps({"provider": {"google": {"models": {"gemini-4-carbon": {
    "options": {"thinkingConfig": {"includeThoughts": True}},
    "limit": {"context": CONTEXT_WINDOW, "output": OUTPUT_TOKEN_MAX}}}}}}, separators=(",", ":"))

TASK_TOML = """\
schema_version = "1.4"
# The sim sidecar's episode directory is collected after the agent phase and handed
# to the separate verifier, which replays it from scratch.
artifacts = [{{ source = "/var/loophole/episode", destination = "episode", service = "loophole-sim" }}]

[task]
name = "loophole/{name}"
authors = [{{ name = "Loophole team" }}]
description = "{description}"
keywords = ["mujoco", "physics", "spatial-reasoning", "disentanglement-puzzle", "mcp"]

[metadata]
difficulty = "{difficulty}"
category = "spatial-reasoning"
max_actions = {max_actions}

[agent]
timeout_sec = {agent_timeout}

[verifier]
timeout_sec = 600.0
env = {{ LOOPHOLE_PUZZLE = "{puzzle}", LOOPHOLE_SEED = "0" }}

# Runs inside the sim sidecar after the agent container is stopped: ends the
# episode and writes the final record next to the log.
[[verifier.collect]]
service = "loophole-sim"
command = "python -m loophole.mcp_server.finalize"
timeout_sec = 120.0

# Separate verifier built from tests/docker-compose.yaml: the same sim image (which
# bakes the verifier in at /tests), no network.
[verifier.environment]
network_mode = "no-network"
cpus = 2
memory_mb = 4096

[environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 4096

[[environment.mcp_servers]]
name = "loophole"
transport = "streamable-http"
url = "http://loophole-sim:8000/mcp"
"""

COMPOSE = """\
# Harbor merges this over its base compose; "main" is the agent container.
services:
  main:
    depends_on:
      loophole-sim:
        condition: service_healthy

  loophole-sim:
    # Built from the repository root (the agent container never sees this source).
    image: loophole-sim:local
    build:
      context: ../../..
      dockerfile: docker/sim.Dockerfile
    pull_policy: build
    environment:
      LOOPHOLE_PUZZLE: "{puzzle}"
      LOOPHOLE_SEED: "0"
      MUJOCO_GL: "${{LOOPHOLE_MUJOCO_GL:-osmesa}}"
    # Explicit networking keeps the simulator OUT of Harbor's egress-control namespace (only
    # the agent container is egress-controlled): it keeps its own DNS name and network
    # namespace, so its loopback-only admin routes stay unreachable from the agent.
    networks: [default]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
      interval: 2s
      timeout: 5s
      retries: 60
      start_period: 5s
"""

AGENT_DOCKERFILE = """\
# Agent container ("main"). Deliberately contains no simulator, puzzle or solver:
# the agent can only reach the environment through the MCP server.
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
RUN apt-get update \\
 && apt-get install -y --no-install-recommends curl ca-certificates git \\
 && rm -rf /var/lib/apt/lists/*
# MCP client library (also used by the oracle's trajectory replayer) and the compute stack
# an agent may use to process images or plan (the run itself has no internet)
RUN apt-get update \\
 && apt-get install -y --no-install-recommends build-essential \\
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mcp==1.30.0 numpy==2.4.6 scipy==1.17.1 pillow==13.0.0 opencv-python-headless==5.0.0.93
WORKDIR /app
"""

SOLVE_SH = """\
#!/bin/bash
# Oracle: replay a stored trajectory through the MCP server, exactly like an agent.
#   LOOPHOLE_SOLUTION=reference   (default) scripted reference solution
#   LOOPHOLE_SOLUTION=stop_early  stop-early baseline: a few plausible moves, then stop
#   LOOPHOLE_SOLUTION=noop        do nothing (same as `-a nop`)
set -euo pipefail
case "${LOOPHOLE_SOLUTION:-reference}" in
  reference)  TRAJ=/solution/reference.jsonl ;;
  stop_early) TRAJ=/solution/stop_early.jsonl ;;
  noop)       echo "noop baseline: no actions"; exit 0 ;;
  *) echo "unknown LOOPHOLE_SOLUTION=${LOOPHOLE_SOLUTION}" >&2; exit 2 ;;
esac
python3 /solution/replay_mcp.py "$TRAJ" --url "${LOOPHOLE_MCP_URL:-http://loophole-sim:8000/mcp}"
"""

TESTS_SH = """\
#!/bin/bash
# Reference copy only. The verifier image (tests/docker-compose.yaml) is the
# loophole-sim image, which bakes in /tests/test.sh from verifier/test.sh at the repository root and
# overrides this file. That script replays the episode log collected from the sim
# sidecar in a fresh simulator and writes /logs/verifier/reward.json.
exec python -m loophole.reward.verify --log /var/loophole/episode/episode.jsonl --out /logs/verifier/reward.json
"""

TESTS_COMPOSE = """\
# Verifier image = the simulator image (same Dockerfile, so the build is a cache hit);
# it bakes the verifier in at /tests/test.sh.
services:
  main:
    build:
      context: ../../..
      dockerfile: docker/sim.Dockerfile
"""

INSTRUCTION_ALPHA = """\
# Loophole: {title}

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is a 3D *disentanglement puzzle* in the spirit of the classic Alpha puzzle:

- two identical rigid **tubes**, each bent into an "alpha" (a loop whose two ends cross
  with a small gap between them, then continue as tails),
- **`tube_a`** (grey) is fixed in space; **`{target}`** (blue) is free. There is no gravity.
- The tubes are interlocked: each loop threads the other.

## Goal

Separate the tubes: move `{target}` so that its minimum distance to `tube_a` exceeds
**{d_sep} m** (`get_state` reports it under `goal`). The episode ends as soon as this holds.

Nothing can pass through anything else: the simulator detects and fails any attempt to
force the tubes through each other. The way out is narrow (a few millimetres of clearance
for the harder variants), so you have to find it, not force it.

## Tools (MCP server `loophole`)

Observation tools are free:
- `get_state()`: JSON with the pose of `{target}`, both tube centre-lines (world frame),
  the tube radius, the loop discs, goal distance, action budget, and `progress`: a
  normalised estimate of how far along the shortest collision-free way out you are
  (0 = start, 1 = separated). It tells you whether you are getting closer, not how.
- `render(view, azimuth, elevation, distance, lookat)`: PNG image + caption.

Each of these costs **one action** from a budget of **{max_actions}**:
- `grab(handle=0, target="{target}")`: grasp the tube (the handle then sits at the tube's
  own origin, which is the point between its two crossing strands).
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion of the grasped
  tube about the handle, at most {max_lin} m / {max_ang} rad per call (larger requests are
  clamped). Motion stops early (`blocked`) if the tube is obstructed.
- `release(handle)`, `wait(steps)`.

There is one handle (0).

## Scoring

Separate the tubes and you score 1 - 0.5 × (actions used / {max_actions}).
If you don't, you get at most 0.3 × the best `progress` you reached.
Plan with the free observation tools before acting. The simulator's source is not
available to you; the score is computed by independently replaying your actions.
When you are done (or out of ideas), simply stop.
"""

INSTRUCTION_CANON = """\
# Loophole: the Alpha puzzle

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is the classic **Alpha puzzle** (a motion-planning benchmark): two identical
rigid tubes, each bent into the shape of the Greek letter alpha, interlocked so that each
one threads the other.

- **`tube_a`** (grey) is fixed in space; **`{target}`** (blue) is free. There is no gravity
  and contacts are frictionless.
- Units are metres and radians; the world frame is `tube_a`'s frame. The tubes are about
  2 m across.

## Goal

Free `{target}`: get it to a pose from which it could **slide away to infinity along some
straight line without touching `tube_a`**. There is no distance target. The simulator checks
this after every action and the episode ends as soon as it holds (`success: true`).

Nothing can pass through anything else. The simulator audits every physics step, and an
episode in which the tubes are forced into each other scores 0.

## Tools (MCP server `loophole`)

Free (no action cost):
- `render(view, azimuth, elevation, distance, lookat, modalities, width, height, depth_format)`:
  **cameras**. Any viewpoint; `modalities` is any of `rgb`, `depth`, `segmentation`. The
  caption is JSON with the full camera model (position, forward/up/right axes, intrinsics),
  so depth pixels back-project to 3-D points:
  `point = position + depth * (forward + (u+0.5-cx)/fx * right - (v+0.5-cy)/fy * up)`.
  `depth_format="metric"` gives a 16-bit PNG in millimetres.
- `get_state()`: **proprioception**. The pose of `{target}` (position + quaternion w,x,y,z),
  your handle, the outcome of your last action, the action budget and the rig limits.
- `get_contacts()`: **touch**. Current contact points, normals and forces between the tubes.
{planner_tools}
Each costs one action from a budget of **{max_actions}**:
- `grab(handle=0, target="{target}")`: grasp the tube. The handle sits at the tube's
  reference point, and motions are applied about that point.
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion of the grasped
  tube. At most **{max_lin} m** of translation and **{max_ang} rad** of rotation per call;
  larger requests are clamped. Execution is velocity-limited ({lin_speed} m/s,
  {ang_speed} rad/s) and stops early (`blocked: true`) if the tube is obstructed.
  `frame` is `"world"` or `"local"` (the tube's own axes).
- `release(handle)`, `wait(steps)`.

There is one handle (0). The workspace is bounded to the box reported in `get_state`.
Your container has Python 3.11 with numpy, scipy, Pillow and OpenCV, and a C/C++ compiler,
and no internet access. You may write and run your own code (for example to process
images or to plan a path) and call the MCP server from it.

## Scoring

- Free the tube: score = 1 - 0.5 × (actions used / {max_actions}), between 0.5 and 1.
- Otherwise: partial credit up to 0.3 for how far along a collision-free way out you got
  (measured by the simulator; held for at least 3 consecutive actions).
- Forcing the tubes into each other: 0.

The simulator's source is not available to you; the score is computed by independently
replaying your actions. When you are done (or out of ideas), simply stop.
"""

PLANNER_TOOLS = """\
- `check_motion(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: **planning query**. Dry
  run of `move` with the same arguments: is the motion collision-free, and how far is free?
- `check_poses(poses)`: **planning query**. Is each absolute pose `[x, y, z, qw, qx, qy, qz]`
  of `{target}` collision-free? (Up to 256 per call.)
- `get_geometry()`: **planning query**. Triangle meshes of both tubes and their poses.
"""

INSTRUCTION = """\
# Loophole: {title}

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is a 3D *disentanglement puzzle*:

- a rigid **frame** (grey) fixed to the table,
- a **rope** (orange) whose two ends are anchored to the frame. It is flexible, cannot
  stretch, and slides without friction,
- a free piece, the **`{target}`** (blue), which is entangled with the rope and the frame.

{description}

## Goal

Free the `{target}`: move it farther from **every rope anchor than the rope could
ever reach** and clear of the frame (`get_state` reports the exact distances under
`goal`). The episode ends as soon as this holds.

Nothing can pass through anything else: the simulator detects and fails any
attempt to force pieces through each other. You have to find the real way out.

## Tools (MCP server `loophole`)

Observation tools are free:
- `get_state()`: JSON with all poses, rope node positions, anchors, holes, handles,
  action budget, goal distances and a description of the frame.
- `render(view, azimuth, elevation, distance, lookat)`: PNG image + caption
  (presets: iso, front, back, side, top).

Each of these costs **one action** from a budget of **{max_actions}**:
- `grab(handle, target, point?, segment?)`: grasp `{target}` or a rope segment (`target="rope"`).
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion, at most
  {max_lin} m / {max_ang} rad per call (larger requests are clamped). Motion stops early
  (`blocked`) if the held body is obstructed.
- `release(handle)`, `wait(steps)`.

You have two handles (0 and 1) that can hold things at the same time.

## Scoring

Score = 1 - 0.5 × (actions used / {max_actions}) if you free the `{target}`, otherwise 0.
Any solve beats any non-solve, and faster solves score higher. Plan with the free
observation tools before acting. The simulator's source is not available to you;
the score is computed by independently replaying your actions. When you are done
(or out of ideas), simply stop.
"""


ROADMAP_CFG = "/opt/loophole/loophole/configs/roadmap_reward.yaml"  # path inside the sim image


def make_task(puzzle: str, roadmap_reward: bool = False, task_name: str | None = None,
              config_file: str | None = None, agent_timeout: float | None = None, agent_no_internet: bool = False) -> Path:
    from loophole.sim.spec import resolve_puzzle

    spec = yaml.safe_load(resolve_puzzle(puzzle).read_text())
    ref = TRAJ / f"{puzzle}.jsonl"
    se = TRAJ / f"{puzzle}_stop_early.jsonl"
    if not ref.exists():
        raise FileNotFoundError(f"no reference trajectory for {puzzle}; run scripts/make_reference.py")
    from loophole.config import load_config

    from loophole.config import effective_config
    from loophole.sim.spec import load_puzzle

    cfg = effective_config(load_config(), load_puzzle(resolve_puzzle(puzzle)))
    d = TASKS / (task_name or f"loophole-{puzzle}")
    if d.exists():
        shutil.rmtree(d)
    (d / "environment").mkdir(parents=True)
    (d / "solution").mkdir()
    (d / "tests").mkdir()
    ctx = {
        "name": puzzle,
        "puzzle": puzzle,
        "title": f"{spec.get('difficulty', puzzle)} disentanglement puzzle",
        "description": spec.get("description", "").replace('"', "'"),
        "difficulty": spec.get("difficulty", "easy"),
        "max_actions": spec.get("max_actions", 60),
        "target": spec["target"],
        "agent_timeout": agent_timeout or {"easy": 1800.0, "medium": 2700.0, "hard": 3600.0}.get(spec.get("difficulty"), 1800.0),
        "max_lin": cfg.rig.max_linear_step,
        "max_ang": cfg.rig.max_angular_step,
        "d_sep": round(spec.get("goal", {}).get("d_sep", 0) * spec.get("unit_scale", 1.0), 6),
        "lin_speed": cfg.rig.linear_speed,
        "ang_speed": cfg.rig.angular_speed,
        "planner_tools": "",
    }
    if config_file:  # the task's own observation profile decides which tools the instruction lists
        local = ROOT / "loophole" / "configs" / Path(config_file).name
        tools = load_config(local).observation.tool_names()
        if "check_motion" in tools:
            ctx["planner_tools"] = PLANNER_TOOLS.format(target=spec["target"])
    # canonical (mesh-based) Alpha: OMPL.app or the Zhang et al. 2020 puzzle set
    canon = any(g.get("type") == "mesh" for b in spec.get("frame", []) for g in b.get("geoms", []))
    alpha = spec.get("goal", {}).get("type") == "separation"
    if canon:
        ctx["name"] = task_name or puzzle
    if alpha:
        ctx["title"] = f"Alpha-style tubes ({ctx['difficulty']})"
    toml = TASK_TOML.format(**ctx)
    compose = COMPOSE.format(**ctx)  # Alpha: OpenCode defaults (the reported Alpha results were run at the default)
    cfg_path = config_file or (ROADMAP_CFG if roadmap_reward else None)
    if cfg_path:  # same config for the sim sidecar and the verifier
        toml = toml.replace('LOOPHOLE_SEED = "0" }', f'LOOPHOLE_SEED = "0", LOOPHOLE_CONFIG = "{cfg_path}" }}')
        compose = compose.replace('      LOOPHOLE_SEED: "0"\n', f'      LOOPHOLE_SEED: "0"\n      LOOPHOLE_CONFIG: "{cfg_path}"\n')
    if agent_no_internet:
        # no internet during agent.run(): egress denied except the agent's own inference
        # endpoint, which Harbor auto-allowlists; setup (installing the agent) stays public
        toml = toml.replace(f"[agent]\ntimeout_sec = {ctx['agent_timeout']}\n",
                            f"[agent]\ntimeout_sec = {ctx['agent_timeout']}\nnetwork_mode = \"allowlist\"\n"
                            f"allowed_hosts = [\"loophole-sim\"]  # the MCP sidecar only (+ the auto-allowlisted model endpoint)\n")
    (d / "task.toml").write_text(toml)
    (d / "instruction.md").write_text((INSTRUCTION_CANON if canon else INSTRUCTION_ALPHA if alpha else INSTRUCTION).format(**ctx))
    (d / "environment" / "Dockerfile").write_text(AGENT_DOCKERFILE)
    (d / "environment" / "docker-compose.yaml").write_text(compose)
    (d / "tests" / "docker-compose.yaml").write_text(TESTS_COMPOSE)
    for name, body in (("solution/solve.sh", SOLVE_SH), ("tests/test.sh", TESTS_SH)):
        p = d / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copy(ROOT / "loophole" / "reference" / "replay_mcp.py", d / "solution" / "replay_mcp.py")
    shutil.copy(ref, d / "solution" / "reference.jsonl")
    if se.exists():
        shutil.copy(se, d / "solution" / "stop_early.jsonl")
    return d


ALPHA_TASKS = {  # task dir -> sim config (inside the sim image); Alpha 1.0 (Zhang et al. 2020) only
    "loophole-alpha": "/opt/loophole/loophole/configs/alpha_training.yaml",  # vision + proprioception + touch
    "loophole-alpha-planner": "/opt/loophole/loophole/configs/alpha_planner.yaml",  # + planning queries
}


def make_alpha_tasks() -> list[Path]:
    return [make_task("alpha-1.0-paper", task_name=name, config_file=cfg, agent_timeout=7200.0, agent_no_internet=True)
            for name, cfg in ALPHA_TASKS.items()]


RATTRAP_TOML = """\
schema_version = "1.4"
# The sim sidecar's episode directory is collected after the agent phase and handed
# to the separate verifier, which replays it from scratch.
artifacts = [{{ source = "/var/loophole/episode", destination = "episode", service = "loophole-sim" }}]

[task]
name = "loophole/loophole-rattrap"
authors = [{{ name = "Loophole team" }}]
description = "The Rat Trap: free a closed string loop, with a mouse bead tied on, from a wooden frame of posts, dowels, a ring and a sphere."
keywords = ["mujoco", "physics", "spatial-reasoning", "string-puzzle", "deformable", "mcp"]

[metadata]
difficulty = "hard"
category = "spatial-reasoning"
max_actions = {T}
reference_actions = {oracle}

[agent]
timeout_sec = {agent_timeout}
network_mode = "allowlist"
allowed_hosts = ["loophole-sim"]  # the MCP sidecar only (+ the auto-allowlisted model endpoint)

[verifier]
# the verifier replays every action in a fresh simulator: measured ~9 s per action in a 2-CPU container on a loaded
# machine, so ~2.1 h for a full-budget (T = 852) episode; 6 h leaves room for slower hosts
timeout_sec = 21600.0
env = {{ LOOPHOLE_TASK = "rattrap" }}

# Runs inside the sim sidecar after the agent container is stopped: writes the final record next to the log.
[[verifier.collect]]
service = "loophole-sim"
command = "python -m loophole.puzzles.rattrap.finalize"
timeout_sec = 120.0

# Separate verifier built from tests/docker-compose.yaml: the same sim image (which
# bakes the verifier in at /tests), no network.
[verifier.environment]
network_mode = "no-network"
cpus = 2
memory_mb = 4096

[environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 4096

[[environment.mcp_servers]]
name = "loophole"
transport = "streamable-http"
url = "http://loophole-sim:8000/mcp"
"""

RATTRAP_INSTRUCTION = """\
# Loophole: the Rat Trap

You are manipulating a physical string puzzle in a physics simulator through the **`loophole` MCP server**
(tools `get_state`, `grab`, `move`, `release`, `give_up`). The simulator's source is not available to you; the score
is computed afterwards by independently replaying your actions. When you are done (or out of ideas), simply stop.

{scene}
"""


RATTRAP_SOLVE_SH = """\
#!/bin/bash
# Oracle for the Rat Trap. The reference is Tibi's recorded human solve: fixed, committed waypoints
# (solution/waypoints.jsonl), followed closed-loop from the gripper's actual position, the same correction the offline
# conversion made, done live so cross-platform numerical differences do not accumulate. No search, no planning.
#   LOOPHOLE_SOLUTION=reference   (default) the recorded human solve
#   LOOPHOLE_SOLUTION=stop_early  stop-early baseline: grab + five random moves, then stop
#   LOOPHOLE_SOLUTION=noop        do nothing (same as `-a nop`)
set -euo pipefail
URL="${LOOPHOLE_MCP_URL:-http://loophole-sim:8000/mcp}"
case "${LOOPHOLE_SOLUTION:-reference}" in
  reference)  exec python3 /solution/follow_waypoints.py /solution/waypoints.jsonl --url "$URL" ;;
  stop_early) exec python3 /solution/replay_mcp.py /solution/stop_early.jsonl --url "$URL" ;;
  noop)       echo "noop baseline: no actions"; exit 0 ;;
  *) echo "unknown LOOPHOLE_SOLUTION=${LOOPHOLE_SOLUTION}" >&2; exit 2 ;;
esac
"""


def make_rattrap_task() -> Path:
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    from loophole.puzzles.rattrap.agent_tools import scene_text
    from loophole.puzzles.rattrap.reward import RatTrapEpisode
    from loophole.puzzles.rattrap.spec import load_spec

    spec = load_spec()
    ep = RatTrapEpisode(spec)
    oracle = sum(1 for l in (ROOT / "tasks" / "loophole-rattrap" / "solution" / "reference.jsonl").read_text().splitlines()
                 if l.strip() and not json.loads(l).get("type"))
    d = TASKS / "loophole-rattrap"
    for sub in ("environment", "tests", "solution"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(RATTRAP_TOML.format(T=ep.T, oracle=oracle, agent_timeout=14400.0))
    (d / "instruction.md").write_text(RATTRAP_INSTRUCTION.format(scene=scene_text(spec, ep)))
    (d / "environment" / "Dockerfile").write_text(AGENT_DOCKERFILE)
    compose = COMPOSE.format(puzzle="rattrap").replace("        condition: service_healthy\n", "        condition: service_healthy\n" + RATTRAP_AGENT_ENV.format(opencode_overlay=OPENCODE_OVERLAY, output_token_max=OUTPUT_TOKEN_MAX), 1)
    compose = compose.replace('      LOOPHOLE_PUZZLE: "rattrap"\n      LOOPHOLE_SEED: "0"\n', '      LOOPHOLE_TASK: "rattrap"\n')
    compose = compose.replace("    networks: [default]\n", '    command: ["python", "-m", "loophole.puzzles.rattrap.mcp_server", "--http"]\n    networks: [default]\n')
    (d / "environment" / "docker-compose.yaml").write_text(compose)
    (d / "tests" / "docker-compose.yaml").write_text(TESTS_COMPOSE)
    solve = RATTRAP_SOLVE_SH
    # stop-early baseline, as for Alpha: grab (the reference's first grab), five seeded random moves, stop
    import random

    rng = random.Random(0)
    first = next(json.loads(l) for l in (d / "solution" / "reference.jsonl").read_text().splitlines() if l.strip() and json.loads(l).get("tool") == "grab")
    se = [{"type": "trajectory", "puzzle": "rattrap", "source": "baseline:stop_early(5 random moves)"},
          {"tool": "grab", "args": first["args"]}]
    se += [{"tool": "move", "args": {"handle": first["args"]["handle"], **{k: round(rng.uniform(-25, 25), 3) for k in ("dx", "dy", "dz")}}} for _ in range(5)]
    (d / "solution" / "stop_early.jsonl").write_text("\n".join(json.dumps(r) for r in se) + "\n")
    for name, body in (("solution/solve.sh", solve), ("tests/test.sh", TESTS_SH.replace(
            "exec python -m loophole.reward.verify --log /var/loophole/episode/episode.jsonl --out /logs/verifier/reward.json",
            "exec python -m loophole.puzzles.rattrap.verify --log /var/loophole/episode/commands.jsonl --out /logs/verifier/reward.txt"))):
        p = d / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copy(ROOT / "loophole" / "reference" / "replay_mcp.py", d / "solution" / "replay_mcp.py")
    shutil.copy(ROOT / "loophole" / "reference" / "follow_waypoints.py", d / "solution" / "follow_waypoints.py")
    return d


if __name__ == "__main__":
    import sys

    if "--rattrap" in sys.argv:
        print("wrote", make_rattrap_task().relative_to(ROOT))
        sys.exit(0)

    if "--alpha" in sys.argv:
        for t in make_alpha_tasks():
            print("wrote", t.relative_to(ROOT))
        sys.exit(0)
    roadmap_reward = "--roadmap-reward" in sys.argv
    puzzles = sorted(p.stem for p in [*PUZZLES.glob("*.yaml"), *PUZZLES.glob("alpha/*.yaml")] if (TRAJ / f"{p.stem}.jsonl").exists())
    for pz in puzzles:
        print("wrote", make_task(pz, roadmap_reward and pz.startswith("alpha")).relative_to(ROOT))
    print(json.dumps({"tasks": puzzles}))
