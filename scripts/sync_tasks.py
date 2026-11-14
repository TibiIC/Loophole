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
    expose:
      - "8000"
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
FROM python:3.11-slim-bookworm
RUN apt-get update \\
 && apt-get install -y --no-install-recommends curl ca-certificates git \\
 && rm -rf /var/lib/apt/lists/*
# MCP client library (also used by the oracle's trajectory replayer)
RUN pip install --no-cache-dir mcp==1.30.0
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


def make_task(puzzle: str, roadmap_reward: bool = False) -> Path:
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
    d = TASKS / f"loophole-{puzzle}"
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
        "agent_timeout": {"easy": 1800.0, "medium": 2700.0, "hard": 3600.0}.get(spec.get("difficulty"), 1800.0),
        "max_lin": cfg.rig.max_linear_step,
        "max_ang": cfg.rig.max_angular_step,
        "d_sep": spec.get("goal", {}).get("d_sep"),
    }
    alpha = spec.get("goal", {}).get("type") == "separation"
    if alpha:
        ctx["title"] = f"Alpha-style tubes ({ctx['difficulty']})"
    toml = TASK_TOML.format(**ctx)
    compose = COMPOSE.format(**ctx)
    if roadmap_reward:  # same config for the sim sidecar and the verifier
        toml = toml.replace('LOOPHOLE_SEED = "0" }', f'LOOPHOLE_SEED = "0", LOOPHOLE_CONFIG = "{ROADMAP_CFG}" }}')
        compose = compose.replace('      LOOPHOLE_SEED: "0"\n', f'      LOOPHOLE_SEED: "0"\n      LOOPHOLE_CONFIG: "{ROADMAP_CFG}"\n')
    (d / "task.toml").write_text(toml)
    (d / "instruction.md").write_text((INSTRUCTION_ALPHA if alpha else INSTRUCTION).format(**ctx))
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


if __name__ == "__main__":
    import sys

    roadmap_reward = "--roadmap-reward" in sys.argv
    puzzles = sorted(p.stem for p in [*PUZZLES.glob("*.yaml"), *PUZZLES.glob("alpha/*.yaml")] if (TRAJ / f"{p.stem}.jsonl").exists())
    for pz in puzzles:
        print("wrote", make_task(pz, roadmap_reward and pz.startswith("alpha")).relative_to(ROOT))
    print(json.dumps({"tasks": puzzles}))
