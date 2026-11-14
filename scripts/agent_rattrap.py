#!/usr/bin/env python3
"""Let an LLM agent try the Rat Trap: in-process simulator, OpenAI-compatible function calling, no Harbor / Docker.

    PYTHONPATH=$PWD .venv/bin/python scripts/agent_rattrap.py --model deepseek [--max-minutes 90]
    models: deepseek, kimi, nemotron (Verda endpoints), gemini-4-carbon (Google's OpenAI-compatible endpoint); keys from the main repo's .env, never printed)

In-process, not the Harbor harness. Reward: the Rat Trap score (docs/RATTRAP_DESIGN.md): solved (the frozen cord,
mouse included, slides off the structure: reward.FrozenEscape) -> 1 - 0.5 a / T; unsolved -> 0.3 * p, p = checkpoints of
the reference solve reached in order / K; any penetration beyond tolerance -> 0. --sparse: the first round's
solved-or-0 reward (score_progress_cap = 0).
Writes runs/rattrap-agents/<model>-<time>/: transcript.jsonl (every request / reply / tool result),
commands.jsonl (the simulator commands, replayable with teleop.replay), result.json, final.png.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.reward import RatTrapEpisode  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402

MODELS = {
    "deepseek": dict(base="https://inference.verda.com/v1", model="deepseek-4-1-flash-1017", key="DEEPSEEK_AI_API_KEY", vision=False),
    "kimi": dict(base="https://inference.verda.com/v1", model="kimi-k3-1-pro", key="KIMI_AI_API_KEY", vision=False),
    "nemotron": dict(base="https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1", model="nemotron-3.5-super-vl",
                     key="NVIDIA_NEMOTRON_AI_API_KEY", vision=True),
    "gemini-4-carbon": dict(base="https://generativelanguage.googleapis.com/v1beta/openai", model="gemini-4-carbon",
                            key="GOOGLE_GENERATIVE_AI_API_KEY", vision=True, max_tokens=32768,
                            extra={"reasoning_effort": "medium"}),  # unbounded, it thinks past 32k tokens and answers nothing
    # with an image in context Gemini 4 Carbon reasons past 32k tokens and returns nothing (even at medium effort): text only
    "gemini-4-carbon-text": dict(base="https://generativelanguage.googleapis.com/v1beta/openai", model="gemini-4-carbon",
                                 key="GOOGLE_GENERATIVE_AI_API_KEY", vision=False, max_tokens=32768,
                                 # reasoning_effort is ignored: it then thinks 32k tokens per turn and answers nothing
                                 extra={"extra_body": {"google": {"thinking_config": {"thinking_budget": 4096}}}}),
}
ENV_FILES = [ROOT / ".env", ROOT.parent / "Loophole" / ".env"]
MM = 1000.0
STRIDE = 6  # cord nodes shown in get_state: every 6th


def load_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for f in ENV_FILES:
        if f.exists():
            for line in f.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip().removeprefix("export ").strip() == name:
                    return v.strip().strip('"').strip("'")
    sys.exit(f"{name} not found in the environment or .env")


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


class Harness:
    def __init__(self, spec, ep: RatTrapEpisode, out: Path, vision: bool, overrides: dict):
        self.spec, self.ep, self.out, self.vision = spec, ep, out, vision
        self.held = {0: None, 1: None}
        self.cmds = open(out / "commands.jsonl", "w", buffering=1)
        self.cmds.write(json.dumps({"type": "header", "spec": "rattrap.yaml", "overrides": overrides}) + "\n")
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
        self.cmds.write(json.dumps({"tool": tool, "args": args}) + "\n")
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


def trim(msgs: list, budget_chars: int) -> list:
    """Keep the system and first user message plus the most recent turns that fit, starting at an assistant turn."""
    head, tail = msgs[:2], msgs[2:]
    size = 0
    keep = len(tail)
    for i in range(len(tail) - 1, -1, -1):
        size += len(json.dumps(tail[i])) if not isinstance(tail[i].get("content"), list) else 2000
        if size > budget_chars:
            break
        keep = i
    while keep < len(tail) and tail[keep]["role"] != "assistant":
        keep += 1
    dropped = keep
    out = head + ([{"role": "user", "content": f"({dropped} earlier messages trimmed; call get_state to re-sync.)"}] if dropped else []) + tail[keep:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--max-turns", type=int, default=1000)
    ap.add_argument("--sparse", action="store_true", help="solved-or-0 (the first round's reward)")
    ap.add_argument("--context-chars", type=int, default=150_000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    m = MODELS[a.model]
    key = load_key(m["key"])
    out = a.out or ROOT / "runs" / "rattrap-agents" / f"{a.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"score_progress_cap": 0.0} if a.sparse else {}
    spec = load_spec(overrides=overrides)
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, out, m["vision"], overrides)
    tools = TOOLS + ([RENDER] if m["vision"] else [])
    msgs = [{"role": "system", "content": scene_text(spec, ep)},
            {"role": "user", "content": "Begin. Call get_state first" + (" and render" if m["vision"] else "") + ", then work towards freeing the loop."}]
    log = open(out / "transcript.jsonl", "w", buffering=1)
    t0, usage, turns, nudges, err = time.time(), {"prompt_tokens": 0, "completion_tokens": 0}, 0, 0, None
    client = httpx.Client(timeout=600.0, headers={"Authorization": f"Bearer {key}"})
    try:
        while not (ep.done or hz.gave_up) and turns < a.max_turns and time.time() - t0 < a.max_minutes * 60:
            turns += 1
            body = {"model": m["model"], "messages": trim(msgs, a.context_chars), "tools": tools, "tool_choice": "auto", "max_tokens": m.get("max_tokens", 8192), **m.get("extra", {})}
            for attempt in range(4):
                try:
                    r = client.post(f"{m['base']}/chat/completions", json=body)
                    if r.status_code == 200:
                        break
                    log.write(json.dumps({"type": "http_error", "status": r.status_code, "body": r.text[:2000]}) + "\n")
                except httpx.HTTPError as e:
                    log.write(json.dumps({"type": "http_error", "error": repr(e)}) + "\n")
                time.sleep(10 * (attempt + 1))
            else:
                err = "API failed 4 times"
                break
            resp = r.json()
            for k in usage:
                usage[k] += int((resp.get("usage") or {}).get(k) or 0)
            msg = resp["choices"][0]["message"]
            log.write(json.dumps({"type": "reply", "turn": turns, "t": round(time.time() - t0, 1), "message": msg, "usage": resp.get("usage"),
                                  "finish_reason": resp["choices"][0].get("finish_reason")}) + "\n")
            calls = msg.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                nudges += 1
                if nudges > 3:
                    break
                msgs.append({"role": "user", "content": "Continue by calling tools (get_state, grab, move, release" + (", render" if m["vision"] else "") + "), or call give_up to stop."})
                continue
            images = []
            for c in calls:
                fn = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = None
                if args is None:
                    res, img = {"error": "arguments are not valid JSON"}, None
                else:
                    try:
                        res, img = hz.call(fn, args)
                    except Exception as e:  # a bad tool call must not end the run
                        res, img = {"error": f"{type(e).__name__}: {e}"}, None
                log.write(json.dumps({"type": "tool", "turn": turns, "name": fn, "args": args, "result": res}) + "\n")
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res)})
                if img:
                    images.append(img)
            if images:
                msgs.append({"role": "user", "content": [{"type": "text", "text": "render(): left = iso view, right = front view."}] +
                             [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}} for b in images[-1:]]})
    except Exception:
        err = traceback.format_exc()
    result = {"model": a.model, "model_id": m["model"], "score": ep.score() if not err or ep.actions else 0.0, "solved": ep.solved,
              "violation": ep.violation, "actions": ep.actions, "max_actions": ep.T, "turns": turns, "gave_up": hz.gave_up,
              "reward": "sparse" if a.sparse else "checkpoints", "progress": ep.progress(),
              "checkpoints_reached": None if ep.tracker is None else ep.tracker.k, "checkpoints": None if ep.tracker is None else ep.tracker.cps.K,
              "word": ep.word(), "uncertified_substeps": ep.uncertified, "minutes": round((time.time() - t0) / 60, 1), "usage": usage,
              "end": "solved" if ep.solved else "violation" if ep.violation else "budget" if ep.done else "gave up" if hz.gave_up
              else "error" if err else "stopped talking" if nudges > 3 else "time/turn limit", "error": err}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    try:
        (out / "final.png").write_bytes(hz.render_png())
    except Exception:
        pass
    print(json.dumps({k: v for k, v in result.items() if k != "error"}), flush=True)


if __name__ == "__main__":
    main()
