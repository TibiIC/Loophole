#!/usr/bin/env python3
"""Collect N seeded training episodes with any policy; write simulator episode logs + an index.

    # in-process Gym policies
    .venv/bin/python scripts/collect_rollouts.py --policy random --episodes 20 --start perturb
    .venv/bin/python scripts/collect_rollouts.py --policy gym:my_pkg.policies:policy --episodes 50 --start curriculum --curriculum-distance 0.3
    # any MCP client (e.g. an LLM agent) against a training-mode MCP server; {url} and {seed} are substituted
    .venv/bin/python scripts/collect_rollouts.py --policy 'mcp:python my_agent.py --mcp {url}' --episodes 5

Policies
  random | noop               uniform random / zero actions in [-1, 1]^6 (loophole.gym.AlphaEnv)
  gym:<module>:<attr>         <attr> is a callable obs -> action, or a class whose instances are
                              (constructed with the env); `act(obs)` is used if present
  mcp:<command>               starts `python -m loophole.mcp_server --training`, calls reset(seed)
                              for each episode, then runs <command> (which talks to {url}) until it exits

Output (default runs/rollouts/<timestamp>/): one simulator episode log per episode (the format
the MCP server writes, with header.start_pose and header.training) and rollouts.jsonl, one
line per episode {seed, log, score, success, actions_used, done_reason, start, policy}. Feed the
logs to scripts/export_traces.py for chat-SFT traces. Training starts only: the canonical start
is never used here (evaluate on it with the Harbor task).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "loophole" / "configs" / "alpha_training.yaml"


def load_policy(spec: str, env, rng: np.random.Generator):
    if spec == "random":
        return lambda obs: rng.uniform(-1.0, 1.0, 6)
    if spec == "noop":
        return lambda obs: np.zeros(6)
    if spec.startswith("gym:"):
        mod, attr = spec[4:].rsplit(":", 1)
        obj = getattr(importlib.import_module(mod), attr)
        if isinstance(obj, type):
            obj = obj(env)
        return getattr(obj, "act", obj)
    raise SystemExit(f"unknown policy {spec!r}")


def final_record(log: Path) -> dict:
    last = {}
    for line in log.open():
        r = json.loads(line)
        if r.get("type") == "final":
            last = r
    return last


def run_gym(a, out: Path) -> list[dict]:
    from loophole.gym import AlphaEnv

    env = AlphaEnv(config=a.config, start=a.start, curriculum_distance=a.curriculum_distance, curriculum_source=a.curriculum_source,
                   randomize_orientation=a.randomize_orientation, randomize_friction=a.randomize_friction,
                   randomize_stiffness=a.randomize_stiffness, obs_noise=a.obs_noise)
    rows = []
    for seed in range(a.seed0, a.seed0 + a.episodes):
        rng = np.random.default_rng(seed)
        policy = load_policy(a.policy, env, rng)
        log = out / f"episode-seed{seed}.jsonl"
        obs, info = env.reset(seed=seed, options={"log_path": log})
        ret, t0 = 0.0, time.time()
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            ret += r
            if term or trunc or (a.max_steps and env.env.actions_used >= a.max_steps):
                break
        final = env.env.finalize()
        rows.append({"seed": seed, "log": log.name, "score": final["score"], "success": final["success"],
                     "actions_used": final["actions_used"], "done_reason": final["done_reason"], "return": round(ret, 6),
                     "start": env.start_info, "policy": a.policy, "wall_s": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    env.close()
    return rows


async def _mcp_reset(url: str, seed: int, a) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    args = {"seed": seed, "start": a.start, "curriculum_source": a.curriculum_source}
    if a.start == "curriculum":
        args["curriculum_distance"] = a.curriculum_distance
    async with streamable_http_client(url + "/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("reset", args)
            if res.isError:
                raise RuntimeError(res.content[0].text)
            return json.loads(res.content[0].text)


def run_mcp(a, out: Path) -> list[dict]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    url = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([sys.executable, "-m", "loophole.mcp_server", "--training", "--puzzle", "alpha-1.0-paper",
                            "--config", str(a.config), "--host", "127.0.0.1", "--port", str(port), "--log-dir", str(out)],
                           stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT, env={**os.environ})
    rows = []
    try:
        for _ in range(480):
            try:
                urllib.request.urlopen(url + "/healthz", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        cmd = a.policy[4:]
        for k, seed in enumerate(range(a.seed0, a.seed0 + a.episodes), start=1):
            start = asyncio.run(_mcp_reset(url, seed, a))
            t0 = time.time()
            subprocess.run(shlex.split(cmd.format(url=url + "/mcp", seed=seed)), timeout=a.timeout, check=False)
            if k == a.episodes:  # the last episode is finalised explicitly; earlier ones by the next reset
                urllib.request.urlopen(urllib.request.Request(url + "/admin/finalize", data=b"{}", method="POST"), timeout=60)
            rows.append({"seed": seed, "episode": start["episode"], "start": start["start"], "policy": a.policy,
                         "wall_s": round(time.time() - t0, 1)})
    finally:
        srv.terminate()
        srv.wait(30)
    for row in rows:  # scores from the logs' final records (written at the next reset / finalize)
        log = next(out.glob(f"episode-{row['episode']:05d}-seed{row['seed']}.jsonl"))
        f = final_record(log)
        row.update(log=log.name, score=f.get("score"), success=f.get("success"), actions_used=f.get("actions_used"),
                   done_reason=f.get("done_reason"))
        print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--start", choices=["perturb", "curriculum"], default="perturb")
    ap.add_argument("--curriculum-distance", type=float, default=0.5)
    ap.add_argument("--curriculum-source", choices=["oracle", "roadmap"], default="oracle")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--randomize-orientation", action="store_true")
    ap.add_argument("--randomize-friction", action="store_true")
    ap.add_argument("--randomize-stiffness", action="store_true")
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop an episode early after this many actions")
    ap.add_argument("--timeout", type=float, default=3600, help="mcp policies: seconds per episode")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or ROOT / "runs" / "rollouts" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=1))
    rows = run_mcp(a, out) if a.policy.startswith("mcp:") else run_gym(a, out)
    with (out / "rollouts.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sc = [r["score"] for r in rows if r.get("score") is not None]
    print(json.dumps({"out": str(out), "episodes": len(rows), "mean_score": round(float(np.mean(sc)), 4) if sc else None}))


if __name__ == "__main__":
    main()
