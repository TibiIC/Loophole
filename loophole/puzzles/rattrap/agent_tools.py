"""Rat Trap agent interface, shared by every harness (scripts/agent_rattrap.py: OpenAI-compatible function calling;
loophole/puzzles/rattrap/mcp_server.py: MCP, for OpenCode and other MCP clients): the scene description, the tool
schemas and their execution against a RatTrapEpisode. Units at the interface: millimetres; moves capped at 30 mm.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .reward import RatTrapEpisode

MM = 1000.0
STRIDE = 6  # cord nodes shown in get_state: every 6th


def mm(v) -> list:
    return [round(float(x) * MM, 1) for x in v]


def scene_text(spec, ep: RatTrapEpisode) -> str:
    r, lm = spec.raw, spec.landmarks
    W, H = r["frame_inner_width"], r["frame_inner_height"]
    n = ep.sim.n_nodes
    mouse = ep.escape.spheres[-1][0] if r.get("mouse_enabled") else None
    ap = {k: f"centre {mm(a.center)} mm, passage axis {[int(x) for x in a.axis]}" for k, a in spec.apertures.items()}
    return f"""\
You are manipulating a physical string puzzle ("the Rat Trap") in a physics simulator, through tools.

Coordinates in millimetres: x = left -> right, y = front -> back (the front is -y), z = up. A rectangular wooden FRAME
stands upright in the plane y = 0 on a base plank (top of the base at z = {r['base_thickness'] * MM:.0f}). The frame's
opening is {W * MM:.0f} mm wide (x from {-W / 2 * MM:.0f} to {W / 2 * MM:.0f}) and {H * MM:.0f} mm tall (z from {lm['inner_floor_z'] * MM:.0f} to
{lm['inner_ceiling_z'] * MM:.0f}); its rails are {r['rail_thickness'] * MM:.0f} mm thick. Inside the opening:
- UPPER POST: a horizontal bar ({r['post_thickness'] * MM:.0f} mm square) from the RIGHT rail leftwards, {r['post_upper_length'] * MM:.0f} mm long, at z = {lm['post_upper_z'] * MM:.1f}.
- LOWER POST: the same from the LEFT rail rightwards, at z = {lm['post_lower_z'] * MM:.1f}.
- LEFT DOWEL: a thin vertical rod (radius {r['dowel_radius'] * MM:.0f} mm) at x = {lm['dowel_left_x'] * MM:.0f} hanging from the top rail, passing down through
  a hole (radius {r['hole_radius'] * MM:.1f} mm) in the upper post, ending in a SPHERE (radius {r['sphere_radius'] * MM:.0f} mm) centred at z = {lm['sphere_center_z'] * MM:.0f}.
- RIGHT DOWEL: a thin vertical rod at x = {lm['dowel_right_x'] * MM:.0f} rising from the bottom rail, up through a hole in the lower post, ending in a
  vertical RING (major radius {r['ring_major_radius'] * MM:.0f} mm, tube {r['ring_tube_radius'] * MM:.0f} mm) centred at {mm(lm['ring_center'])}, whose axis runs
  left-right; the upper post passes through the ring.
Named passages (gaps the string can go through): {json.dumps(ap)}.

The STRING is a closed loop of length {sum(ep.sim.built.rope.seg_lengths) * MM:.0f} mm, radius {r['cord_radius'] * MM:.1f} mm, modelled as {n} nodes (indices 0..{n - 1};
node 0 and node {n - 1} are the same point, the loop is closed). A MOUSE (a bead of radius {r['mouse_radius'] * MM:.0f} mm) is tied to node {mouse}; it is too big
for the dowel holes, the ring and the gaps beside the sphere, but fits through the gaps at the free ends of the posts.
At the start the loop is wrapped around the right dowel (between the bottom rail and the lower post), both strands
running out over the front of the base into a wide loop in front of the frame.

GOAL: free the string loop (with the mouse) completely from the wooden structure. It counts as solved when the loop,
frozen in its current shape, could slide away to infinity in some straight direction without touching anything.
Checked automatically after every action. Score: solved -> 1 - 0.5 * actions / {ep.T}; not solved -> {r['score_progress_cap']:g} x the
fraction of the milestones of a reference solution you reached, in order (0 if none). Pressing the string into the
wood (more than {r['penetration_tolerance'] * MM:.0f} mm) ends the episode with 0.

TOOLS: two grippers (handle 0 and 1). grab(handle, node) takes the string at a node; move(handle, dx, dy, dz) moves
that gripper by up to 30 mm per call (the string follows; the motion stops early if blocked); release(handle).
get_state() shows the string. render() shows images (if available). Budget: {ep.T} grab/move/release actions in total;
get_state / render are free. Plan, then act. The string cannot pass through the wood or through itself."""


TOOLS = [
    {"type": "function", "function": {"name": "get_state", "description": "Current state: string nodes (every 6th, mm), grippers, mouse, actions left.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "grab", "description": "Close a gripper on the string at a node index.",
                                      "parameters": {"type": "object", "properties": {"handle": {"type": "integer", "enum": [0, 1]}, "node": {"type": "integer"}},
                                                     "required": ["handle", "node"]}}},
    {"type": "function", "function": {"name": "move", "description": "Move a gripper by (dx, dy, dz) millimetres (each |d| <= 30). Stops early if blocked.",
                                      "parameters": {"type": "object", "properties": {"handle": {"type": "integer", "enum": [0, 1]}, "dx": {"type": "number"},
                                                                                      "dy": {"type": "number"}, "dz": {"type": "number"}},
                                                     "required": ["handle", "dx", "dy", "dz"]}}},
    {"type": "function", "function": {"name": "release", "description": "Open a gripper.",
                                      "parameters": {"type": "object", "properties": {"handle": {"type": "integer", "enum": [0, 1]}}, "required": ["handle"]}}},
    {"type": "function", "function": {"name": "give_up", "description": "End the episode now.", "parameters": {"type": "object", "properties": {}}}},
]
RENDER = {"type": "function", "function": {"name": "render", "description": "Images of the scene (iso and front views), attached as a user message.",
                                           "parameters": {"type": "object", "properties": {}}}}


def spec_sha256(spec) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(spec.raw, sort_keys=True, default=str).encode()).hexdigest()


def checkpoints_sha256(ep: RatTrapEpisode) -> str | None:
    import hashlib

    from .reward import DEFAULT_FILE

    name = ep.spec.raw.get("checkpoints_file")
    path = DEFAULT_FILE if not name else DEFAULT_FILE.parent / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def episode_header(spec, ep: RatTrapEpisode, overrides: dict) -> dict:
    """What the verifier checks before replaying: the same YAML, checkpoints and start state, and no overrides."""
    return {"type": "header", "puzzle": "rattrap", "spec": "rattrap.yaml", "overrides": overrides, "spec_sha256": spec_sha256(spec),
            "checkpoints_sha256": checkpoints_sha256(ep), "max_actions": ep.T, "initial_state_hash": ep.sim.state_hash()}


class Harness:
    def __init__(self, spec, ep: RatTrapEpisode, out: Path, vision: bool, overrides: dict):
        self.spec, self.ep, self.out, self.vision = spec, ep, out, vision
        self.held = {0: None, 1: None}
        self.cmds = open(out / "commands.jsonl", "w", buffering=1)
        self.cmds.write(json.dumps(episode_header(spec, ep, overrides)) + "\n")
        self.gave_up = False

    def state(self) -> dict:
        sim = self.ep.sim
        P = sim.node_positions()
        hs = {}
        for h in (0, 1):
            pos = sim.rig.handle_pose(sim.rig.handles[h])[0]
            hs[h] = {"holding_node": self.held[h], "position_mm": mm(pos)}
        mouse = self.ep.escape.spheres[-1][0] if self.spec.raw.get("mouse_enabled") else None
        return {"actions_used": self.ep.actions, "actions_left": self.ep.T - self.ep.actions,
                "nodes_mm": {int(i): mm(P[i]) for i in range(0, len(P), STRIDE)},
                "mouse_node": mouse, "mouse_mm": mm(P[mouse]) if mouse is not None else None,
                "grippers": hs, "solved": self.ep.solved, "done": self.ep.done}

    def render_png(self) -> bytes:
        from PIL import Image

        tiles = [self.ep.sim.render("iso", distance=0.45, width=512, height=384), self.ep.sim.render("front", distance=0.4, width=512, height=384)]
        buf = io.BytesIO()
        Image.fromarray(np.concatenate(tiles, axis=1)).save(buf, "PNG")
        return buf.getvalue()

    def call(self, name: str, a: dict) -> tuple[dict, bytes | None]:
        ep = self.ep
        if name == "get_state":
            return self.state(), None
        if name == "render":
            return ({"images": "attached in the next user message"}, self.render_png()) if self.vision else ({"error": "no images for this model"}, None)
        if name == "give_up":
            self.gave_up = True
            return {"ended": True}, None
        if ep.done:
            return {"error": "the episode is over"}, None
        h = int(a.get("handle", -1))
        if h not in (0, 1):
            return {"error": "handle must be 0 or 1"}, None
        if name == "grab":
            k = int(a["node"])
            if not 0 <= k < ep.sim.n_nodes:
                return {"error": f"node must be in 0..{ep.sim.n_nodes - 1}"}, None
            tool, args = "grab_node", {"handle": h, "node": k}
        elif name == "move":
            if self.held[h] is None:
                return {"error": f"gripper {h} is not holding the string; grab first"}, None
            d = [float(a.get(k, 0.0)) for k in ("dx", "dy", "dz")]
            if not all(np.isfinite(d)):
                return {"error": "non-finite move"}, None
            d = [float(np.clip(x, -30.0, 30.0)) / MM for x in d]
            tool, args = "move", {"handle": h, "dx": d[0], "dy": d[1], "dz": d[2]}
        elif name == "release":
            tool, args = "release", {"handle": h}
        else:
            return {"error": f"unknown tool {name}"}, None
        res = ep.act(tool, args)
        self.cmds.write(json.dumps({"tool": tool, "args": args, "i": ep.actions, "state_hash": ep.sim.state_hash()}) + "\n")
        if tool == "grab_node":
            self.held[h] = args["node"]
        elif tool == "release":
            self.held[h] = None
        out = {k: res[k] for k in ("solved", "violation", "done")}
        sim = ep.sim
        out["gripper_mm"] = mm(sim.rig.handle_pose(sim.rig.handles[h])[0])
        if tool == "move":
            out.update(blocked=res.get("blocked"), blocked_reason=res.get("blocked_reason"), executed_fraction=res.get("executed_fraction"))
        out["actions_left"] = ep.T - ep.actions
        return out, None


