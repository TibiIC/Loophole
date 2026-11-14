"""Rat Trap over MCP (stdio), for OpenCode and any other MCP client: the same tools, scene text and scoring as
scripts/agent_rattrap.py (agent_tools.py), so a model can be run through its native provider in OpenCode.

    python -m loophole.puzzles.rattrap.mcp_server --out RUN_DIR [--sparse]

Writes RUN_DIR/commands.jsonl (replayable), tools.jsonl (every call and result) and result.json (rewritten after
every call, so a session that is cut off still leaves its score). The agent sees no score, progress or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .agent_tools import Harness, scene_text
from .reward import RatTrapEpisode
from .spec import load_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    overrides = {"score_progress_cap": 0.0} if a.sparse else {}
    spec = load_spec(overrides=overrides)
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, a.out, vision=False, overrides=overrides)
    log = open(a.out / "tools.jsonl", "a", buffering=1)
    t0 = time.time()
    (a.out / "scene.txt").write_text(scene_text(spec, ep))

    def finish() -> None:
        res = {"model": a.label, "harness": "opencode+mcp", "score": ep.score(), "solved": ep.solved, "violation": ep.violation,
               "actions": ep.actions, "max_actions": ep.T, "gave_up": hz.gave_up, "reward": "sparse" if a.sparse else "checkpoints",
               "progress": ep.progress(), "checkpoints_reached": None if ep.tracker is None else ep.tracker.k,
               "checkpoints": None if ep.tracker is None else ep.tracker.cps.K, "word": ep.word(), "uncertified_substeps": ep.uncertified,
               "minutes": round((time.time() - t0) / 60, 1),
               "end": "solved" if ep.solved else "violation" if ep.violation else "budget" if ep.done else "gave up" if hz.gave_up else "running/stopped"}
        (a.out / "result.json").write_text(json.dumps(res, indent=1))

    def call(name: str, args: dict) -> str:
        try:
            res, _ = hz.call(name, args)
        except Exception as e:  # a bad call must not kill the server
            res = {"error": f"{type(e).__name__}: {e}"}
        log.write(json.dumps({"t": round(time.time() - t0, 1), "name": name, "args": args, "result": res}) + "\n")
        finish()
        return json.dumps(res)

    mcp = FastMCP("rattrap")

    @mcp.tool()
    def get_state() -> str:
        """Current state: string nodes (every 6th, mm), grippers, mouse, actions left."""
        return call("get_state", {})

    @mcp.tool()
    def grab(handle: int, node: int) -> str:
        """Close gripper `handle` (0 or 1) on the string at node index `node`."""
        return call("grab", {"handle": handle, "node": node})

    @mcp.tool()
    def move(handle: int, dx: float, dy: float, dz: float) -> str:
        """Move gripper `handle` by (dx, dy, dz) millimetres (each |d| <= 30). Stops early if blocked."""
        return call("move", {"handle": handle, "dx": dx, "dy": dy, "dz": dz})

    @mcp.tool()
    def release(handle: int) -> str:
        """Open gripper `handle`."""
        return call("release", {"handle": handle})

    @mcp.tool()
    def give_up() -> str:
        """End the episode now."""
        return call("give_up", {})

    finish()
    mcp.run("stdio")


if __name__ == "__main__":
    main()
