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
from loophole.puzzles.rattrap.agent_tools import RENDER, TOOLS, Harness, scene_text  # noqa: E402

MODELS = {
    "deepseek": dict(base="https://inference.verda.com/v1", model="deepseek-4-1-flash-1017", key="DEEPSEEK_AI_API_KEY", vision=False),
    "kimi": dict(base="https://inference.verda.com/v1", model="kimi-k3-1-pro", key="KIMI_AI_API_KEY", vision=False),
    "nemotron": dict(base="https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1", model="nemotron-3.5-super-vl",
                     key="NVIDIA_NEMOTRON_AI_API_KEY", vision=True, max_tokens=32768),  # its reasoning overruns 8k
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
    t0, usage, turns, nudges, cutoffs, err = time.time(), {"prompt_tokens": 0, "completion_tokens": 0}, 0, 0, 0, None
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
            if not calls and resp["choices"][0].get("finish_reason") == "length":
                cutoffs += 1  # the reply was cut off mid-reasoning: not a decision to stop
                if cutoffs > 10:
                    err = "10 replies cut off at the output limit"
                    break
                msgs.append({"role": "user", "content": "Your last reply was cut off before any tool call. Think less and reply with a tool call now."})
                continue
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
