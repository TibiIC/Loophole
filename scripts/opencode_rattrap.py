#!/usr/bin/env python3
"""Run a model on the Rat Trap through OpenCode and its native provider (the path Gemini uses on Alpha), with the
Rat Trap tools served over MCP (loophole/puzzles/rattrap/mcp_server.py).

    .venv/bin/python scripts/opencode_rattrap.py --model google/gemini-4-carbon [--runs 3] [--max-minutes 180]
        [--opencode PATH]   (default: `opencode` on PATH; tested with opencode-ai 1.19.5, as in the Alpha runs)

Each run gets its own directory under runs/rattrap-agents/ with an isolated OpenCode config / data / state (the user's
own OpenCode setup is never read or written). OpenCode's built-in tools (bash, file read / write / edit, search, web,
tasks) are disabled, so the model has exactly the Rat Trap tools the other harness offers and cannot read the
repository (or the reference solve). Same prompt (the scene text) and scoring as scripts/agent_rattrap.py.
Keys come from the environment or the .env files (GOOGLE_GENERATIVE_AI_API_KEY for google/...); never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# OpenCode 1.19.5's built-in tools (an unknown name in `tools`, or a `permission` block, made it fail with "Unexpected
# server error" before the first model call)
BUILTIN = ["bash", "edit", "write", "read", "grep", "glob", "list", "patch", "todowrite", "todoread", "webfetch", "task"]
KEYS = {"google": "GOOGLE_GENERATIVE_AI_API_KEY"}
ENV_FILES = [ROOT / ".env", ROOT.parent / "Loophole" / ".env"]


def key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    for f in ENV_FILES:
        if f.exists():
            for line in f.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip().removeprefix("export ").strip() == name:
                    return v.strip().strip('"').strip("'")
    return None


def one_run(a, i: int) -> Path:
    from loophole.puzzles.rattrap.agent_tools import scene_text
    from loophole.puzzles.rattrap.reward import RatTrapEpisode
    from loophole.puzzles.rattrap.spec import load_spec

    tag = a.model.replace("/", "-")
    run = ROOT / "runs" / "rattrap-agents" / f"{tag}-opencode-{time.strftime('%Y%m%d-%H%M%S')}-{i}"
    run.mkdir(parents=True, exist_ok=True)
    # OpenCode's home and working directory live outside the repository (inside it, OpenCode adopts the repo as its
    # project); the session database is copied into the run directory afterwards
    home = Path(tempfile.mkdtemp(prefix="rattrap-opencode-"))
    for d in ("config", "data", "state", "cache", "work"):
        (home / d).mkdir(parents=True, exist_ok=True)
    py = str(ROOT / ".venv" / "bin" / "python")
    server = [py, "-m", "loophole.puzzles.rattrap.mcp_server", "--out", str(run), "--label", a.model] + (["--sparse"] if a.sparse else [])
    cfg = {"$schema": "https://opencode.ai/config.json",
           "mcp": {"rattrap": {"type": "local", "command": server, "enabled": True, "environment": {"PYTHONPATH": str(ROOT)}}},
           "tools": {t: False for t in BUILTIN}}
    if a.model.startswith("google/") and a.thinking_level:
        # Gemini 4 Carbon at OpenCode's default thinking level (and at "medium") spends its whole 32k-token output
        # budget reasoning after the first get_state and answers nothing; "low" acts. (Google refuses a thinking
        # budget alongside the level OpenCode sets, so the level is what can be changed.)
        mid = a.model.split("/", 1)[1]
        cfg["provider"] = {"google": {"models": {mid: {"options": {"thinkingConfig": {"thinkingLevel": a.thinking_level, "includeThoughts": True}}}}}}
    (home / "config" / "opencode.json").write_text(json.dumps(cfg, indent=1))
    spec = load_spec(overrides={"score_progress_cap": 0.0} if a.sparse else {})
    prompt = (scene_text(spec, RatTrapEpisode(spec)) + "\n\nYour only tools are the Rat Trap tools (get_state, grab, move, release, give_up). "
              "Begin: call get_state first, then work towards freeing the loop. Keep calling tools until the string is free or the "
              "action budget is used up; nobody will answer questions.")
    (run / "prompt.txt").write_text(prompt)
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(XDG_CONFIG_HOME=str(home / "config"), XDG_DATA_HOME=str(home / "data"), XDG_STATE_HOME=str(home / "state"),
               XDG_CACHE_HOME=str(home / "cache"), OPENCODE_CONFIG=str(home / "config" / "opencode.json"))
    provider = a.model.split("/", 1)[0]
    if provider in KEYS:
        k = key(KEYS[provider])
        if not k:
            sys.exit(f"{KEYS[provider]} not set")
        env[KEYS[provider]] = k
    cmd = [a.opencode, f"--model={a.model}", "run", "--format=json", "--thinking", "--dangerously-skip-permissions", "--", prompt]
    with open(run / "opencode.jsonl", "w") as out, open(run / "opencode.stderr", "w") as err:
        try:
            p = subprocess.run(cmd, cwd=home / "work", env=env, stdout=out, stderr=err, timeout=a.max_minutes * 60)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    shutil.copytree(home, run / "opencode-home", dirs_exist_ok=True)
    from scrub_logs import scrub_tree  # keys and environment dumps out, OpenCode's internal state dropped

    scrub_tree(run)
    res = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {"error": "no result (the MCP server never started?)"}
    res.update(thinking_level=a.thinking_level if a.model.startswith("google/") else None, opencode_exit=rc, opencode_version=subprocess.run([a.opencode, "--version"], capture_output=True, text=True).stdout.strip())
    (run / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"run": run.name, **{k: res.get(k) for k in ("score", "solved", "actions", "checkpoints_reached", "end", "opencode_exit")}}), flush=True)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-4-carbon")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--opencode", default=shutil.which("opencode") or "opencode")
    ap.add_argument("--parallel", action="store_true", help="start all runs at once")
    ap.add_argument("--thinking-level", default="low", help="Google models: thinkingConfig.thinkingLevel ('' = OpenCode's default)")
    a = ap.parse_args()
    if a.parallel:
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(a.runs) as ex:
            list(ex.map(lambda i: (time.sleep(2 * i), one_run(a, i))[1], range(a.runs)))
    else:
        for i in range(a.runs):
            one_run(a, i)


if __name__ == "__main__":
    main()
