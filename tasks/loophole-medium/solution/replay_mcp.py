#!/usr/bin/env python3
"""Replay a trajectory file through the Loophole MCP server, exactly as an agent would.

Standalone on purpose (depends only on the `mcp` package): it is copied into
each Harbor task's solution/ directory and run by the oracle agent.

Trajectory format (JSONL): an optional header line {"type": "trajectory", ...}
followed by one {"tool": ..., "args": {...}} object per call. Observation
calls (get_state / render) are allowed and cost nothing.

    python replay_mcp.py reference.jsonl --url http://loophole-sim:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def load_trajectory(path: str) -> list[dict]:
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") in ("trajectory", "header"):
                continue
            steps.append({"tool": rec["tool"], "args": rec.get("args", {})})
    return steps


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def wait_for_server(url: str, timeout: float = 120.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            async with streamable_http_client(url) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    return
        except Exception:
            if asyncio.get_event_loop().time() > deadline:
                raise
            await asyncio.sleep(1.0)


async def replay(url: str, steps: list[dict], limit: int | None = None, verbose: bool = True) -> dict | None:
    last = None
    async with streamable_http_client(url) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for i, st in enumerate(steps[:limit] if limit is not None else steps):
                res = await s.call_tool(st["tool"], st.get("args", {}))
                txt = _text(res)
                if verbose:
                    print(f"[{i}] {st['tool']} {json.dumps(st.get('args', {}))} -> {'ERROR ' if res.isError else ''}{txt[:300]}", flush=True)
                if res.isError:
                    continue
                if st["tool"] in ("grab", "move", "release", "wait"):
                    try:
                        last = json.loads(txt)
                    except json.JSONDecodeError:
                        last = None
                    if last and last.get("done"):
                        break
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("--url", default="http://loophole-sim:8000/mcp")
    ap.add_argument("--limit", type=int, default=None, help="only replay the first N calls")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    steps = load_trajectory(a.trajectory)
    asyncio.run(wait_for_server(a.url))
    last = asyncio.run(replay(a.url, steps, a.limit, verbose=not a.quiet))
    if last:
        print(json.dumps({k: last.get(k) for k in ("action", "done", "done_reason", "success", "violations")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
