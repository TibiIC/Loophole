"""Local dev viewer (outside Docker): watch scripted episodes or replay episode logs.

    mjpython -m loophole.viewer --trajectory loophole/reference/trajectories/easy.jsonl
    mjpython -m loophole.viewer --episode runs/episode.jsonl        # an MCP/Harbor episode log
    mjpython -m loophole.viewer --puzzle easy                        # just look at a puzzle

On macOS the passive viewer needs `mjpython` (ships with the mujoco wheel);
on Linux plain `python` works.
"""

from __future__ import annotations

import argparse
import json
import time

import mujoco
import mujoco.viewer

from .config import load_config
from .reference.recorder import load_trajectory
from .sim.env import PHYSICS_TOOLS, LoopholeEnv
from .sim.rig import RigError


def view_replay(env: LoopholeEnv, steps: list[tuple[str, dict]], speed: float = 1.0, sync_every: int = 8) -> None:
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = (env.frame_aabb()[0] + env.frame_aabb()[1]) / 2
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, 135, -20
        inner = env._substep
        counter = {"n": 0}
        dt = env.model.opt.timestep

        def substep():
            inner()
            counter["n"] += 1
            if counter["n"] % sync_every == 0:
                v.sync()
                time.sleep(sync_every * dt / speed)

        env._substep = substep
        env.rig.substep = substep
        time.sleep(1.0)
        for tool, args in steps:
            if not v.is_running():
                return
            if tool not in PHYSICS_TOOLS or env.done:
                continue
            try:
                r = env.act(tool, dict(args))
                print(tool, args, "->", {k: r.get(k) for k in ("action", "blocked", "done", "success")})
            except (RigError, ValueError) as e:
                print(tool, args, "-> rejected:", e)
        print("final:", {k: env.finalize()[k] for k in ("score", "success", "actions_used")})
        while v.is_running():
            v.sync()
            time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--episode", default=None, help="episode.jsonl written by the MCP server / Harbor")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    steps: list[tuple[str, dict]] = []
    puzzle, seed, cfg = a.puzzle, 0, load_config()
    if a.episode:
        recs = [json.loads(l) for l in open(a.episode) if l.strip()]
        hdr = recs[0]
        puzzle = puzzle or hdr["puzzle"]
        seed = hdr.get("seed", 0)
        cfg = cfg.model_validate(hdr["config"])
        steps = [(r["tool"], r["args"]) for r in recs if r.get("type") == "action"]
    elif a.trajectory:
        steps = load_trajectory(a.trajectory)
        if puzzle is None:
            puzzle = json.loads(open(a.trajectory).readline()).get("puzzle")
    env = LoopholeEnv(puzzle or "easy", config=cfg, seed=seed)
    view_replay(env, steps, speed=a.speed)


if __name__ == "__main__":
    main()
