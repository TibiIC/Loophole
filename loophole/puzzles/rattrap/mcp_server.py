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
import sys
import threading
import time
from pathlib import Path

import anyio
from mcp.server.fastmcp import Context, FastMCP

from .agent_tools import Harness, scene_text
from .reward import RatTrapEpisode
from .spec import load_spec

INSTRUCTIONS = "The Rat Trap string puzzle: free the closed string loop (with the mouse bead) from the wooden frame. See get_state."


def build(out: Path, sparse: bool = False, label: str = "", host: str = "0.0.0.0", port: int = 8000,
          live_view: int | None = None, live_host: str = "127.0.0.1") -> FastMCP:
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"score_progress_cap": 0.0} if sparse else {}
    spec = load_spec(overrides=overrides)
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, out, vision=False, overrides=overrides)
    if live_view:  # off unless --live-view: a separate HTTP port, never MCP (live_view.py, docs/watching.md)
        from .live_view import LiveView

        lv = LiveView(out, live_view, host=live_host).attach(hz)
        print(f"live view: {lv.url}", file=sys.stderr, flush=True)  # stderr: stdout is the MCP stdio channel
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

    async def acall(name: str, args: dict, ctx: Context | None) -> str:
        """Run the (blocking) simulation step in a worker thread and, while it runs, send a log notification every
        2 s on the request's stream: a physics step can take tens of seconds, and a silent connection can be cut by
        a proxy on the way (Harbor's egress control sits between the agent and this sidecar)."""
        done = anyio.Event()
        out: dict = {}

        async def work():
            out["r"] = await anyio.to_thread.run_sync(call, name, args)
            done.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(work)
            while not done.is_set():
                with anyio.move_on_after(2.0):
                    await done.wait()
                if not done.is_set() and ctx is not None:
                    try:
                        await ctx.info(f"{name}: simulating")
                    except Exception:
                        pass
        return out["r"]

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

    mcp = FastMCP("rattrap", instructions=INSTRUCTIONS, host=host, port=port, stateless_http=True, json_response=False,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    @mcp.tool()
    async def get_state(ctx: Context) -> str:
        """Current state: every string node (mm), grippers, mouse, actions left. Free (no action)."""
        return await acall("get_state", {}, ctx)

    @mcp.tool()
    async def grab(handle: int, node: int, ctx: Context) -> str:
        """Close gripper `handle` (0 or 1) on the string at node index `node`. One action."""
        return await acall("grab", {"handle": handle, "node": node}, ctx)

    @mcp.tool()
    async def move(handle: int, dx: float, dy: float, dz: float, ctx: Context) -> str:
        """Move gripper `handle` by (dx, dy, dz) millimetres (each |d| <= 30). Stops early if blocked. One action."""
        return await acall("move", {"handle": handle, "dx": dx, "dy": dy, "dz": dz}, ctx)

    @mcp.tool()
    async def release(handle: int, ctx: Context) -> str:
        """Open gripper `handle`. One action."""
        return await acall("release", {"handle": handle}, ctx)

    @mcp.tool()
    async def give_up(ctx: Context) -> str:
        """End the episode now."""
        return await acall("give_up", {}, ctx)

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
    ap.add_argument("--live-view", type=int, default=int(os.environ.get("LOOPHOLE_LIVE_VIEW", "0")) or None, metavar="PORT",
                    help="serve a live run viewer on this port (off by default; for us, never through MCP)")
    ap.add_argument("--live-host", default=os.environ.get("LOOPHOLE_LIVE_HOST", "127.0.0.1"),
                    help="viewer bind address (127.0.0.1; 0.0.0.0 only inside the standalone container of docker/live-view.compose.yaml)")
    a = ap.parse_args()
    mcp = build(a.out, a.sparse, a.label, a.host, a.port, a.live_view, a.live_host)
    mcp.run("streamable-http" if a.http else "stdio")


if __name__ == "__main__":
    main()
