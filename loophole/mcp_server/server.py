"""Agent-facing MCP server (FastMCP, streamable-http).

Action tools: grab, move, release, wait. Observation tools depend on the
configured observation profile (config.observation): `sensors` = get_state
(proprioception), render (rgb / depth / segmentation cameras), get_contacts
(touch); `planner` adds check_motion, check_poses, get_geometry; `full` also
shows privileged goal/progress fields (debugging). Reset, puzzle
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

INSTRUCTIONS_ROPE = """\
Loophole: a 3D disentanglement puzzle in a physics simulator.
A rigid frame (fixed), a rope whose ends are anchored to the frame, and a free
target piece. Free the target piece: carry it farther from every rope anchor
than the rope could ever reach, and clear of the frame.

Action tools (each costs one action from a fixed budget): grab, move, release, wait.
All other tools are free observations. You control up to two handles (0 and 1).
A handle grabs one body (the target piece, or a rope segment) and moves it with
bounded steps; motion stops early ("blocked") when the held body is obstructed.
Nothing can be pushed through anything else. Coordinates are metres, world frame, z up.
"""

INSTRUCTIONS_SEPARATION = """\
Loophole: a rigid two-piece disentanglement puzzle in a physics simulator.
One piece is fixed; the other is free and can be grasped with handle 0. The
puzzle is solved as soon as the free piece can slide away to infinity along
some straight line without touching the fixed piece.

Action tools (each costs one action from a fixed budget): grab, move, release, wait.
All other tools are free observations: cameras (render), the robot's own state
(get_state) and touch (get_contacts), plus planning queries if enabled.
Nothing can be pushed through anything else. Coordinates are metres, world frame, z up.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
ACTION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def build_server(env: LoopholeEnv, host: str = "0.0.0.0", port: int = 8000, training: dict | None = None) -> FastMCP:
    """`training` (admin only, off by default; never set by the Harbor tasks): enables a `reset` tool
    that starts a new episode from a sampled *training* start (never the canonical one) with its
    own log file, and adds the per-step shaping reward to action results. Evaluation mode keeps
    one episode per process and exposes no shaping."""
    lock = threading.Lock()
    mcp = FastMCP(
        "loophole",
        instructions=INSTRUCTIONS_SEPARATION if env.checker is not None else INSTRUCTIONS_ROPE,
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    def physics(tool: str, args: dict) -> dict:
        with lock:
            try:
                out = env.act(tool, args)
            except (RigError, EpisodeOver, ValueError, TypeError) as e:
                raise ToolError(str(e)) from None
            if training is not None:
                out["shaping_reward"] = round(env.last_step_reward, 6)
                out["rewards"] = env.composer.last_step
                if env.done:
                    out["score"] = env.finalize()["score"]
            return out

    tools = set(env.tools)

    def observe(name: str, args: dict, fn):
        with lock:
            env.log_observation(name, args)
            try:
                return fn()
            except (RigError, EpisodeOver, ValueError, TypeError) as e:
                raise ToolError(str(e)) from None

    def register(name: str, fn, doc: str):
        if name in tools:
            fn.__doc__ = doc
            mcp.tool(name=name, annotations=READ_ONLY)(fn)

    def get_state() -> dict:
        return observe("get_state", {}, env.get_state)

    register("get_state", get_state, """Proprioception: the pose of each piece (position + quaternion w,x,y,z; world = the
        fixed piece's frame), your handle (holding what, where), the outcome of your last action
        (executed motion, blocked, clamped), the action budget, done/success, and the rig limits.
        Free (no action cost).""" if not env.privileged else """Full simulator state as JSON (privileged
        debugging profile): poses, handles, budget, goal metrics, progress, scene geometry. Free.""")

    def render(
        view: Annotated[Literal["iso", "front", "back", "side", "top"], Field(description="named preset camera")] = "iso",
        azimuth: Annotated[float | None, Field(description="degrees; overrides the preset")] = None,
        elevation: Annotated[float | None, Field(description="degrees, negative looks down")] = None,
        distance: Annotated[float | None, Field(description="metres from lookat, 0.05..20")] = None,
        lookat: Annotated[list[float] | None, Field(description="[x, y, z] point the camera looks at")] = None,
        modalities: Annotated[list[Literal["rgb", "depth", "segmentation"]] | None,
                              Field(description="images to return, default ['rgb']")] = None,
        width: Annotated[int | None, Field(description="pixels, 64..1280 (default 480)")] = None,
        height: Annotated[int | None, Field(description="pixels, 64..960 (default 360)")] = None,
        depth_format: Annotated[Literal["visual", "metric"], Field(
            description="visual: 8-bit grey, white = near; metric: 16-bit PNG in millimetres")] = "visual",
    ) -> list:
        args = {"view": view, "azimuth": azimuth, "elevation": elevation, "distance": distance, "lookat": lookat,
                "modalities": modalities, "width": width, "height": height, "depth_format": depth_format}
        imgs, caption = observe("render", args, lambda: env.render_views(
            view, azimuth, elevation, distance, lookat, modalities or ["rgb"], width, height, depth_format))
        return [*(Image(data=png, format="png") for _, png in imgs), caption]

    register("render", render, """Camera: render the scene from any viewpoint as PNG images (rgb, depth and/or
        segmentation) plus a JSON caption with the full camera model (position, forward/up/right
        axes, intrinsics fx fy cx cy), so pixels can be back-projected to 3-D points:
        point = position + depth * (forward + (u+0.5-cx)/fx * right - (v+0.5-cy)/fy * up).
        Fixed piece grey, free piece blue, handle red. Free (no action cost).""")

    def get_contacts() -> dict:
        return observe("get_contacts", {}, env.get_contacts)

    register("get_contacts", get_contacts, """Touch: the contacts between the free piece and the fixed one
        right now (position, normal pointing into the free piece, penetration depth, normal force),
        the total contact force/torque, and how far the held piece lags its handle. Free.""")

    def check_motion(
        handle: Annotated[int, Field(description="handle index (must be holding the piece)")] = 0,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
        droll: Annotated[float, Field(description="radians about x")] = 0.0,
        dpitch: Annotated[float, Field(description="radians about y")] = 0.0,
        dyaw: Annotated[float, Field(description="radians about z")] = 0.0,
        frame: Annotated[Literal["world", "local"], Field(description="axes for the displacement / rotation")] = "world",
    ) -> dict:
        args = {"handle": handle, "dx": dx, "dy": dy, "dz": dz, "droll": droll, "dpitch": dpitch, "dyaw": dyaw, "frame": frame}
        return observe("check_motion", args, lambda: env.check_motion(**args))

    register("check_motion", check_motion, """Planning query: dry run of `move` with identical arguments, clamping
        and kinematics. Reports whether the straight-line motion would be collision-free, how much of
        it is free, and the resulting pose. Changes nothing and costs no action.""")

    def check_poses(poses: Annotated[list[list[float]], Field(description="1..256 poses [x, y, z, qw, qx, qy, qz] of the free piece")]) -> dict:
        return observe("check_poses", {"n": len(poses) if isinstance(poses, list) else None}, lambda: env.check_poses(poses))

    register("check_poses", check_poses, """Planning query: for each absolute pose of the free piece, is it
        collision-free? (A state-validity checker for your own motion planner.) Free.""")

    def get_geometry() -> dict:
        return observe("get_geometry", {}, env.get_geometry)

    register("get_geometry", get_geometry, """Planning query: triangle meshes of both pieces in their own body
        frames (metres) and their current poses. Large (~2-3k triangles per piece). Free.""")

    if training is not None:
        import numpy as np

        from ..training.starts import StartSampler

        sampler = StartSampler(env, training["q0"])
        episodes = {"n": 0}

        @mcp.tool(annotations=ACTION)
        def reset(
            seed: Annotated[int, Field(description="episode seed (start sampling and simulator)")],
            start: Annotated[Literal["perturb", "curriculum"], Field(description="training start distribution")] = "perturb",
            curriculum_distance: Annotated[float | None, Field(description="curriculum: 0 = just before free, 1 = as far as the canonical start")] = None,
            curriculum_source: Annotated[Literal["oracle", "roadmap"], Field(description="curriculum: poses along the oracle path or roadmap nodes")] = "oracle",
        ) -> dict:
            """TRAINING MODE ONLY. End the current episode and start a new one from a sampled training
            start (never the canonical start). Returns the new episode's start and state."""
            with lock:
                if env.finalized is None:
                    env.finalize()
                kw = {"mode": start, "curriculum_source": curriculum_source}
                if curriculum_distance is not None:
                    kw["curriculum_distance"] = curriculum_distance
                try:
                    q, info = sampler.sample(np.random.default_rng(seed), **kw)
                except (ValueError, RuntimeError) as e:
                    raise ToolError(str(e)) from None
                episodes["n"] += 1
                env.start_pose, env.episode_meta = q, {**info, "mcp_training": True}
                env.reset(seed=seed, log_path=Path(training["log_dir"]) / f"episode-{episodes['n']:05d}-seed{seed}.jsonl")
                return {"episode": episodes["n"], "start": info, "state": env.get_state()}

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
    ap.add_argument("--training", action="store_true", default=os.environ.get("LOOPHOLE_TRAINING") == "1",
                    help="admin: training mode (reset tool with sampled starts, shaping reward in results); never for evaluation")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log_path = None if a.training else Path(a.log_dir) / "episode.jsonl"
    env = LoopholeEnv(a.puzzle, config=load_config(a.config), seed=a.seed, log_path=log_path)
    if a.training:  # the canonical start is held out: no actions until reset() samples a training start
        env.done, env.done_reason, env.finalized = True, "training mode: call reset(seed) to start an episode", {}
    log.info("puzzle=%s seed=%s log=%s initial_hash=%s", a.puzzle, a.seed, log_path, env.initial_hash)
    (Path(a.log_dir) / "server.json").write_text(json.dumps({"puzzle": a.puzzle, "seed": a.seed, "pid": os.getpid()}))
    training = None
    if a.training:
        log.warning("TRAINING MODE: reset tool and shaping rewards exposed; not an evaluation server")
        training = {"q0": env.target_pose().copy(), "log_dir": a.log_dir}
    build_server(env, a.host, a.port, training).run(transport="streamable-http")


if __name__ == "__main__":
    main()
