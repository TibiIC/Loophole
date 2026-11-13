"""Agent-facing MCP server (FastMCP, streamable-http).

Exposes only: get_state, render, grab, move, release, wait. Reset, puzzle
selection and seeding are fixed at process start (CLI / env vars) and are never
reachable over MCP. Every call is appended to the JSONL episode log.

Admin-only HTTP routes (not MCP tools, loopback clients only):
* GET  /healthz           liveness for the compose healthcheck
* POST /admin/finalize    end the episode, write the final record; returns it
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from ..config import load_config
from ..sim.env import EpisodeOver, LoopholeEnv
from ..sim.rig import RigError

log = logging.getLogger("loophole.mcp")

INSTRUCTIONS = """\
Loophole: a 3D disentanglement puzzle in a physics simulator.
A rigid frame (fixed), a rope whose ends are anchored to the frame, and a free
target piece. Free the target piece: carry it farther from every rope anchor
than the rope could ever reach, and clear of the frame.

Observation tools (free): get_state, render.
Action tools (each costs one action from a fixed budget): grab, move, release, wait.
You control up to two handles (0 and 1). A handle grabs one body (the target
piece, or a rope segment) and moves it with bounded steps; motion stops early
("blocked") when the held body is obstructed. Nothing can be pushed through
anything else. The score is higher the fewer actions you use; failing scores 0.
Coordinates are metres, world frame, z up.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
ACTION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def build_server(env: LoopholeEnv, host: str = "0.0.0.0", port: int = 8000) -> FastMCP:
    lock = threading.Lock()
    mcp = FastMCP(
        "loophole",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    def physics(tool: str, args: dict) -> dict:
        with lock:
            try:
                return env.act(tool, args)
            except (RigError, EpisodeOver, ValueError, TypeError) as e:
                raise ToolError(str(e)) from None

    @mcp.tool(annotations=READ_ONLY)
    def get_state() -> dict:
        """Full simulator state as JSON: object poses, rope node positions (nodes[k] starts
        segment k), anchors, holes, handle status, action count / budget, done flag,
        goal distances, a static description of the frame, rig limits and (if enabled)
        the topological crossing words. Does not cost an action."""
        with lock:
            env.log_observation("get_state", {})
            return env.get_state()

    @mcp.tool(annotations=READ_ONLY)
    def render(
        view: Annotated[Literal["iso", "front", "back", "side", "top"], Field(description="named preset camera")] = "iso",
        azimuth: Annotated[float | None, Field(description="degrees; overrides the preset")] = None,
        elevation: Annotated[float | None, Field(description="degrees, negative looks down")] = None,
        distance: Annotated[float | None, Field(description="metres from lookat")] = None,
        lookat: Annotated[list[float] | None, Field(description="[x, y, z] point the camera looks at")] = None,
    ) -> list:
        """Render the scene to a PNG image plus a one-line text caption. Camera moves
        never cost actions. Colours: frame grey, rope orange, target blue,
        handle 0 red, handle 1 green."""
        with lock:
            args = {"view": view, "azimuth": azimuth, "elevation": elevation, "distance": distance, "lookat": lookat}
            env.log_observation("render", args)
            try:
                png, caption = env.render(view, azimuth, elevation, distance, lookat)
            except ValueError as e:
                raise ToolError(str(e)) from None
        return [Image(data=png, format="png"), caption]

    @mcp.tool(annotations=ACTION)
    def grab(
        handle: Annotated[int, Field(description="handle index: 0 or 1")],
        target: Annotated[str, Field(description="body to grasp: the target piece name, or 'rope' (with segment or point)")],
        point: Annotated[list[float] | None, Field(description="optional [x, y, z] grasp point on the body's surface")] = None,
        segment: Annotated[int | None, Field(description="rope segment index (0 = start anchor end)")] = None,
    ) -> dict:
        """Grasp a body with a free handle. The handle jumps to the grasp point (the
        body's centre, the segment midpoint, or `point`). Costs one action."""
        return physics("grab", {"handle": handle, "target": target, "point": point, "segment": segment})

    @mcp.tool(annotations=ACTION)
    def move(
        handle: Annotated[int, Field(description="handle index: 0 or 1")],
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        droll: Annotated[float, Field(description="radians about x")] = 0.0,
        dpitch: Annotated[float, Field(description="radians about y")] = 0.0,
        dyaw: Annotated[float, Field(description="radians about z")] = 0.0,
        frame: Annotated[Literal["world", "local"], Field(description="axes for the displacement / rotation")] = "world",
    ) -> dict:
        """Move a handle (and what it holds) by a relative displacement (metres) and
        rotation (radians, applied about the grasp point). Requests larger than the
        rig limits are clamped; execution is velocity-limited and stops early if the
        held body is blocked. Returns the new state summary and any clamping. Costs one action."""
        return physics("move", {"handle": handle, "dx": dx, "dy": dy, "dz": dz, "droll": droll, "dpitch": dpitch, "dyaw": dyaw, "frame": frame})

    @mcp.tool(annotations=ACTION)
    def release(handle: Annotated[int, Field(description="handle index: 0 or 1")]) -> dict:
        """Open a handle, letting go of whatever it holds. Costs one action."""
        return physics("release", {"handle": handle})

    @mcp.tool(annotations=ACTION)
    def wait(steps: Annotated[int, Field(description="physics substeps (1 ms each), clamped to [1, 4000]")] = 200) -> dict:
        """Let physics run without moving the handles (e.g. to let things settle). Costs one action."""
        return physics("wait", {"steps": steps})

    @mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(request: Request):
        return PlainTextResponse("ok")

    @mcp.custom_route("/admin/finalize", methods=["POST"], include_in_schema=False)
    async def finalize(request: Request):
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        with lock:
            final = env.finalize()
        return JSONResponse(final)

    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Loophole MCP server (admin options are not exposed over MCP)")
    ap.add_argument("--puzzle", default=os.environ.get("LOOPHOLE_PUZZLE", "easy"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("LOOPHOLE_SEED", "0")))
    ap.add_argument("--config", default=os.environ.get("LOOPHOLE_CONFIG"))
    ap.add_argument("--log-dir", default=os.environ.get("LOOPHOLE_LOG_DIR", "/var/loophole/episode"))
    ap.add_argument("--host", default=os.environ.get("LOOPHOLE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LOOPHOLE_PORT", "8000")))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log_path = Path(a.log_dir) / "episode.jsonl"
    env = LoopholeEnv(a.puzzle, config=load_config(a.config), seed=a.seed, log_path=log_path)
    log.info("puzzle=%s seed=%s log=%s initial_hash=%s", a.puzzle, a.seed, log_path, env.initial_hash)
    (Path(a.log_dir) / "server.json").write_text(json.dumps({"puzzle": a.puzzle, "seed": a.seed, "pid": os.getpid()}))
    build_server(env, a.host, a.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
