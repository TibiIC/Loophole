"""Deterministic local replay of a trajectory through the same rig the agent uses.

    python -m loophole.reference.replay loophole/reference/trajectories/easy.jsonl --puzzle easy
    python -m loophole.reference.replay traj.jsonl --puzzle easy --log runs/ep.jsonl --view   # watch it

Observation calls in the trajectory are skipped; invalid actions are reported
and skipped (exactly as the MCP server would do).
"""

from __future__ import annotations

import argparse
import json
import sys

from ..config import load_config
from ..sim.env import PHYSICS_TOOLS, LoopholeEnv
from ..sim.rig import RigError
from .recorder import load_trajectory


def replay(env: LoopholeEnv, steps: list[tuple[str, dict]], on_step=None, stop_when_done: bool = True) -> dict:
    for i, (tool, args) in enumerate(steps):
        if tool not in PHYSICS_TOOLS:
            continue
        if env.done and stop_when_done:
            break
        try:
            r = env.act(tool, dict(args))
        except (RigError, ValueError, TypeError) as e:
            r = {"error": str(e)}
        if on_step:
            on_step(i, tool, args, r)
    return env.finalize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("--puzzle", default=None, help="defaults to the trajectory header's puzzle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--log", default=None, help="write an episode log here")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer (local only)")
    a = ap.parse_args(argv)
    puzzle = a.puzzle
    if puzzle is None:
        with open(a.trajectory) as f:
            puzzle = json.loads(f.readline()).get("puzzle")
    env = LoopholeEnv(puzzle, config=load_config(a.config), seed=a.seed, log_path=a.log)
    steps = load_trajectory(a.trajectory)

    def show(i, tool, args, r):
        brief = {k: r.get(k) for k in ("action", "blocked", "done", "success", "violations", "error") if k in r}
        print(f"[{i}] {tool} {json.dumps(args)} -> {brief}")

    if a.view:
        from ..viewer import view_replay

        view_replay(env, steps)
        return 0
    final = replay(env, steps, on_step=show)
    print(json.dumps({k: final[k] for k in ("score", "success", "actions_used", "done_reason")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
