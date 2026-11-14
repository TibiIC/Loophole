"""Rat Trap over MCP, for OpenCode and any other MCP client: the same tools, scene text and scoring as
scripts/agent_rattrap.py (agent_tools.py).

    python -m loophole.puzzles.rattrap.mcp_server --out RUN_DIR                  # stdio (scripts/opencode_rattrap.py)
    python -m loophole.puzzles.rattrap.mcp_server --http [--port 8000]           # streamable HTTP at /mcp (Harbor sidecar,
                                                                                 #   tasks/loophole-rattrap); --out defaults
                                                                                 #   to $LOOPHOLE_LOG_DIR
Writes OUT/commands.jsonl (header with the YAML / checkpoint / start-state hashes, then one record per action with the
resulting state hash: what the verifier replays, loophole/puzzles/rattrap/verify.py), tools.jsonl (every call and
result) and result.json (rewritten after every call, so a session that is cut off still leaves its score).
The agent sees no score, progress or checkpoints. HTTP mode also serves GET /healthz and a loopback-only
POST /admin/finalize (the Harbor collect hook), like the Alpha server.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .agent_tools import Harness, scene_text
from .reward import RatTrapEpisode
from .spec import load_spec

INSTRUCTIONS = "The Rat Trap string puzzle: free the closed string loop (with the mouse bead) from the wooden frame. See get_state."


def build(out: Path, sparse: bool = False, label: str = "", host: str = "0.0.0.0", port: int = 8000) -> FastMCP:
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"score_progress_cap": 0.0} if sparse else {}
    spec = load_spec(overrides=overrides)
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, out, vision=False, overrides=overrides)
    log = open(out / "tools.jsonl", "a", buffering=1)
    lock = threading.Lock()
    t0 = time.time()
    (out / "scene.txt").write_text(scene_text(spec, ep))

    def result() -> dict:
        res = {"model": label, "harness": "mcp", "score": ep.score(), "solved": ep.solved, "violation": ep.violation,
               "actions": ep.actions, "max_actions": ep.T, "gave_up": hz.gave_up, "reward": "sparse" if sparse else "checkpoints",
               "progress": ep.progress(), "checkpoints_reached": None if ep.tracker is None else ep.tracker.k,
               "checkpoints": None if ep.tracker is None else ep.tracker.cps.K, "word": ep.word(), "uncertified_substeps": ep.uncertified,
               "minutes": round((time.time() - t0) / 60, 1),
               "end": "solved" if ep.solved else "violation" if ep.violation else "budget" if ep.done else "gave up" if hz.gave_up else "running/stopped"}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        return res

    def call(name: str, args: dict) -> str:
        with lock:
            try:
                res, _ = hz.call(name, args)
            except Exception as e:  # a bad call must not kill the server
                res = {"error": f"{type(e).__name__}: {e}"}
            log.write(json.dumps({"t": round(time.time() - t0, 1), "name": name, "args": args, "result": res}) + "\n")
            result()
        return json.dumps(res)

    from mcp.server.transport_security import TransportSecuritySettings

    mcp = FastMCP("rattrap", instructions=INSTRUCTIONS, host=host, port=port, stateless_http=True, json_response=True,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    @mcp.tool()
    def get_state() -> str:
        """Current state: string nodes (every 6th, mm), grippers, mouse, actions left. Free (no action)."""
        return call("get_state", {})

    @mcp.tool()
    def grab(handle: int, node: int) -> str:
        """Close gripper `handle` (0 or 1) on the string at node index `node`. One action."""
        return call("grab", {"handle": handle, "node": node})

    @mcp.tool()
    def move(handle: int, dx: float, dy: float, dz: float) -> str:
        """Move gripper `handle` by (dx, dy, dz) millimetres (each |d| <= 30). Stops early if blocked. One action."""
        return call("move", {"handle": handle, "dx": dx, "dy": dy, "dz": dz})

    @mcp.tool()
    def release(handle: int) -> str:
        """Open gripper `handle`. One action."""
        return call("release", {"handle": handle})

    @mcp.tool()
    def give_up() -> str:
        """End the episode now."""
        return call("give_up", {})

    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse

    @mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(request: Request):
        return PlainTextResponse("ok")

    @mcp.custom_route("/admin/finalize", methods=["POST"], include_in_schema=False)
    async def finalize(request: Request):
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        with lock:
            return JSONResponse(result())

    result()
    return mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("LOOPHOLE_LOG_DIR", "/var/loophole/episode")))
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--http", action="store_true", help="streamable HTTP at /mcp instead of stdio")
    ap.add_argument("--host", default=os.environ.get("LOOPHOLE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LOOPHOLE_PORT", "8000")))
    a = ap.parse_args()
    mcp = build(a.out, a.sparse, a.label, a.host, a.port)
    mcp.run("streamable-http" if a.http else "stdio")


if __name__ == "__main__":
    main()
