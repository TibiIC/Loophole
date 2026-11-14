#!/usr/bin/env python3
"""Rat Trap reference replay: follow the fixed waypoints of Tibi's recorded solve, closed-loop.

    python follow_waypoints.py waypoints.jsonl --url http://loophole-sim:8000/mcp       (Harbor oracle, solution/solve.sh)
    follow(events, call)                                                                (in-process, tests)

waypoints.jsonl is fixed and committed: the recorded solve's grabs and releases, and for each stretch of one gripper's
recorded motion, the recorded hand positions kept by Ramer-Douglas-Peucker simplification at 0.1 mm (exactly the
points the offline conversion followed). At replay, each waypoint is approached from where the gripper actually is
(as reported by the last tool result), in moves of at most 30 mm; a blocked move ends that waypoint and the next one is
aimed from wherever the gripper stopped. This is the same correction the offline conversion performed, done live, so
small cross-platform numerical differences do not accumulate. It searches for nothing: no planning, no
solution-finding, no feedback from the score.

Standalone on purpose (only the `mcp` package for the MCP mode): it is copied into the task's solution/ directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math

CAP_MM = 30.0
EPS_MM = 0.01  # a waypoint closer than this counts as reached
MAX_MOVES_PER_WAYPOINT = 64


def load(path: str) -> list[dict]:
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("type") not in ("header", "trajectory"):
                out.append(r)
    return out


async def afollow(events: list[dict], acall, verbose: bool = False) -> dict | None:
    """Replay `events` through `await acall(tool, args) -> dict` (a tool result). Returns the last action result."""
    pos: dict[int, list[float]] = {}
    last = None
    for ev in events:
        if ev["kind"] == "grab":
            last = await acall("grab", {"handle": ev["handle"], "node": ev["node"]})
            pos[ev["handle"]] = last.get("gripper_mm")
        elif ev["kind"] == "release":
            last = await acall("release", {"handle": ev["handle"]})
        else:  # waypoint: approach it from where the gripper actually is
            h, wp = ev["handle"], ev["mm"]
            for _ in range(MAX_MOVES_PER_WAYPOINT):
                cur = pos.get(h)
                if cur is None:
                    break
                d = [w - c for w, c in zip(wp, cur)]
                L = math.sqrt(sum(x * x for x in d))
                if L < EPS_MM:
                    break
                sc = 1.0 if L <= CAP_MM else CAP_MM / L
                last = await acall("move", {"handle": h, "dx": d[0] * sc, "dy": d[1] * sc, "dz": d[2] * sc})
                pos[h] = last.get("gripper_mm", cur)
                if verbose:
                    print(f"move h{h} -> {wp}: at {pos[h]} blocked={last.get('blocked')}", flush=True)
                if last.get("blocked") or last.get("done") or last.get("error"):
                    break
        if last is not None and (last.get("done") or last.get("solved")):
            break
    return last


def follow(events: list[dict], call, verbose: bool = False) -> dict | None:
    """Synchronous form, for an in-process `call(tool, args) -> dict` (tests): the same code path as the oracle."""
    async def acall(tool, args):
        return call(tool, args)

    return asyncio.run(afollow(events, acall, verbose))


async def _mcp(url: str, events: list[dict]) -> dict | None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()

            async def acall(tool: str, args: dict) -> dict:
                res = await s.call_tool(tool, args)
                txt = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    return {"error": txt}

            return await afollow(events, acall, verbose=True)


async def _wait(url: str, timeout: float = 120.0) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("waypoints")
    ap.add_argument("--url", default="http://loophole-sim:8000/mcp")
    a = ap.parse_args()
    events = load(a.waypoints)
    asyncio.run(_wait(a.url))
    last = asyncio.run(_mcp(a.url, events))
    print(json.dumps({"final": last}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
