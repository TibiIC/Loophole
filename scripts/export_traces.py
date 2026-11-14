#!/usr/bin/env python3
"""Export simulator episode logs as chat-SFT traces (OpenAI-style messages with tool calls).

    .venv/bin/python scripts/export_traces.py runs/<ts>/**/episode.jsonl -o traces/alpha.jsonl
    .venv/bin/python scripts/export_traces.py runs/rollouts/*/episode-*.jsonl -o traces/rollouts.jsonl --min-score 0.5

Input: episode logs in the simulator's format (header / observe / action / rejected /
final): what the MCP server writes in evaluation, and what scripts/collect_rollouts.py
writes in training. Each log is replayed in a fresh simulator (puzzle, config, seed and,
for training starts, the start pose from the header), and every logged tool call, free
observations included, is re-executed to regenerate its output. Image outputs are
written as PNG files next to the JSONL (or inlined as data URLs with --inline-images).

One JSONL line per episode:
  {"id", "source", "score", "success", "actions_used", "start_pose", "tools": [function schemas],
   "messages": [system (task instruction), user, {assistant tool_call, tool result}*, ...]}
The episode log holds tool calls only; an agent's free-text reasoning (e.g. OpenCode's own
trajectory) is not in it and is not exported.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.config import EnvConfig  # noqa: E402
from loophole.sim.env import PHYSICS_TOOLS, EpisodeOver, LoopholeEnv  # noqa: E402
from loophole.sim.rig import RigError  # noqa: E402

DEFAULT_INSTRUCTION = ROOT / "tasks" / "loophole-alpha" / "instruction.md"
USER_PROMPT = "Solve the puzzle using the loophole tools. Stop when you are done."


def tool_schemas(env: LoopholeEnv) -> list[dict]:
    """The exact MCP tool list of this env's profile, as OpenAI function schemas."""
    from loophole.mcp_server.server import build_server

    tools = asyncio.run(build_server(env).list_tools())
    return [{"type": "function", "function": {"name": t.name, "description": " ".join((t.description or "").split()),
                                              "parameters": t.inputSchema}} for t in tools]


def observe(env: LoopholeEnv, tool: str, args: dict, img_dir: Path, stem: str, inline: bool) -> list[dict] | str:
    """Re-run a free observation tool; text, or a list of content parts for images."""
    if tool == "get_state":
        return json.dumps(env.get_state())
    if tool == "get_contacts":
        return json.dumps(env.get_contacts())
    if tool == "render":
        a = {k: v for k, v in args.items() if v is not None}
        imgs, caption = env.render_views(a.get("view", "iso"), a.get("azimuth"), a.get("elevation"), a.get("distance"), a.get("lookat"),
                                         a.get("modalities") or ["rgb"], a.get("width"), a.get("height"), a.get("depth_format", "visual"))
        parts = []
        for k, (kind, png) in enumerate(imgs):
            if inline:
                url = "data:image/png;base64," + base64.b64encode(png).decode()
            else:
                f = img_dir / f"{stem}_{kind}.png"
                f.write_bytes(png)
                url = str(f.relative_to(img_dir.parent))
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts + [{"type": "text", "text": caption}]
    if tool == "check_motion":
        return json.dumps(env.check_motion(**args))
    if tool == "check_poses":
        return json.dumps({"note": "poses not logged; re-run not possible"})
    if tool == "get_geometry":
        return json.dumps({"note": "meshes omitted from traces"})
    return json.dumps({"error": f"unknown tool {tool}"})


def export(path: Path, out_dir: Path, instruction: str, inline: bool) -> dict | None:
    recs = [json.loads(l) for l in path.open() if l.strip()]
    if not recs or recs[0].get("type") != "header":
        print(f"skip {path}: no header", file=sys.stderr)
        return None
    hdr = recs[0]
    env = LoopholeEnv(hdr["puzzle"], config=EnvConfig.model_validate(hdr["config"]), seed=int(hdr.get("seed", 0)),
                      start_pose=hdr.get("start_pose"))
    physics = (hdr.get("training") or {}).get("physics")
    if physics:  # training episode with randomised contact parameters: replay with the same ones
        from loophole.training import randomize

        env.episode_meta = hdr["training"]
        randomize.apply_physics(env, physics, randomize.defaults(env))
        env.reset(seed=int(hdr.get("seed", 0)))
    if env.initial_hash != hdr.get("initial_state_hash"):
        print(f"skip {path}: initial state does not reproduce", file=sys.stderr)
        return None
    ep_id = f"{path.parent.name}-{path.stem}"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "system", "content": instruction}, {"role": "user", "content": USER_PROMPT}]
    consistent = True
    n = 0
    for r in recs[1:]:
        if r["type"] not in ("observe", "action", "rejected"):
            continue
        tool, args = r["tool"], r.get("args") or {}
        n += 1
        cid = f"call_{n}"
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]})
        if r["type"] == "observe":
            content = observe(env, tool, args, img_dir, f"{ep_id}_{n}", inline)
        else:
            try:
                res = env.act(tool, args) if tool in PHYSICS_TOOLS else None
                content = json.dumps(res)
                if r["type"] == "action" and env.state_hash() != r.get("state_hash"):
                    consistent = False
            except (RigError, EpisodeOver, ValueError, TypeError) as e:
                content = json.dumps({"error": str(e)})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": content})
    final = env.finalize()
    out = {"id": ep_id, "source": str(path), "score": final["score"], "success": final["success"],
           "actions_used": final["actions_used"], "done_reason": final["done_reason"], "start_pose": hdr.get("start_pose"),
           "training": hdr.get("training"), "replay_consistent": consistent, "tools": tool_schemas(env), "messages": msgs}
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="episode logs (globs allowed)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    ap.add_argument("--inline-images", action="store_true")
    ap.add_argument("--min-score", type=float, default=None, help="keep only episodes scoring at least this")
    a = ap.parse_args()
    paths = sorted({Path(p) for g in a.logs for p in (glob.glob(g, recursive=True) or [g])})
    instruction = a.instruction.read_text() if a.instruction.exists() else ""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with a.out.open("w") as f:
        for p in paths:
            ep = export(p, a.out.parent, instruction, a.inline_images)
            if ep is None or (a.min_score is not None and ep["score"] < a.min_score):
                continue
            if not ep["replay_consistent"]:
                print(f"warning {p}: replay diverged from the logged state hashes", file=sys.stderr)
            f.write(json.dumps(ep) + "\n")
            kept += 1
            print(f"{p}: score {ep['score']:.3f}, {len(ep['messages'])} messages", flush=True)
    print(json.dumps({"out": str(a.out), "episodes": kept, "of": len(paths)}))


if __name__ == "__main__":
    main()
