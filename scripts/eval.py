#!/usr/bin/env python3
"""Full evaluation of tasks/loophole-alpha: oracle, 3 OpenCode runs, noop, stop-early.

    make eval MODEL=gemini/gemini-4-carbon
    .venv/bin/python scripts/eval.py --model gemini/gemini-4-carbon [--runs 3] [--task tasks/loophole-alpha]

Everything (Harbor CLI logs, trial dirs incl. agent trajectories, episode logs from the
simulator, verifier reports, rewards) is copied to runs/<timestamp>/, and
runs/<timestamp>/summary.md gets a score table (mean ± std across the agent runs).
API keys come from environment variables (e.g. GOOGLE_GENERATIVE_AI_API_KEY for gemini/*,
OPENAI_API_KEY for openai/*); nothing is read from or written to the repo's .env here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARBOR = os.environ.get("HARBOR", shutil.which("harbor") or str(Path.home() / ".local/bin/harbor"))


def run(name: str, args: list[str], out: Path, task: str) -> list[dict]:
    job = f"{out.name}-{name}"
    cmd = [HARBOR, "run", "-p", task, *args, "-o", str(ROOT / "jobs"), "--job-name", job]
    print(f"[eval] {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(out / f"{name}.harbor.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
    results = []
    for trial in sorted((ROOT / "jobs" / job).glob("*__*")):
        dest = out / name / trial.name
        shutil.copytree(trial, dest, dirs_exist_ok=True)
        rep_path = trial / "verifier" / "report.json"
        rew_path = trial / "verifier" / "reward.txt"
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
        reward = float(rew_path.read_text().strip()) if rew_path.exists() else None
        final = rep.get("final") or {}
        results.append({
            "run": name, "trial": trial.name, "reward": reward, "success": bool(rep.get("success")),
            "actions": rep.get("actions_used"), "violations": rep.get("violations"), "reason": rep.get("reason"),
            "log_consistent": rep.get("log_consistent"),
            "ctg": (final.get("cost_to_go") or {}),
        })
    shutil.copy(ROOT / "jobs" / job / "result.json", out / name / "job_result.json") if (ROOT / "jobs" / job / "result.json").exists() else None
    print(f"[eval] {name}: {[r['reward'] for r in results]} ({time.time() - t0:.0f}s)", flush=True)
    return results


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def summary(out: Path, model: str, rows: list[dict]) -> str:
    lines = [f"# Evaluation summary: `{out.name}`", "", f"Task `tasks/loophole-alpha` (Alpha 1.0). Agent model: `{model}` (OpenCode).", "",
             "| run | trial | score | solved | actions | violations | start ctg | best held ctg | log consistent |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["ctg"]
        lines.append(f"| {r['run']} | {r['trial']} | {fmt(r['reward'])} | {'yes' if r['success'] else 'no'} | {r['actions']} | "
                     f"{r['violations']} | {fmt(c.get('start'), 2)} | {fmt(c.get('stable_best'), 2)} | {r['log_consistent']} |")
    agents = [r["reward"] for r in rows if r["run"] == "opencode" and r["reward"] is not None]
    lines += ["", "## Scores", "", "| run | score |", "|---|---|"]
    for name in ("oracle", "noop", "stop-early"):
        sc = [r["reward"] for r in rows if r["run"] == name]
        lines.append(f"| {name} | {fmt(sc[0]) if sc else '–'} |")
    if agents:
        sd = statistics.stdev(agents) if len(agents) > 1 else 0.0
        lines.append(f"| opencode `{model}` (n={len(agents)}) | {statistics.mean(agents):.3f} ± {sd:.3f} |")
    lines += ["", "Score: solved → 1 − 0.5·a/T; unsolved → 0.3·p with p = 1 − best held cost-to-go / start "
              "cost-to-go; any violation → 0 (see loophole/reward/literature_rewards.py).", ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--task", default="./tasks/loophole-alpha")
    ap.add_argument("--agent-timeout-multiplier", type=float, default=1.0)
    ap.add_argument("--allow-dirty", action="store_true", help="run even if files baked into the sim image have uncommitted changes")
    a = ap.parse_args()
    # Harbor rebuilds the sim image from the working tree for every sidecar and every verifier, so
    # editing loophole/ during an eval changes the code mid-run (verifier != simulator -> score 0).
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "loophole", "docker", "verifier", "scripts/fetch_alpha_paper.sh",
                            "pyproject.toml"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty and not a.allow_dirty:
        sys.exit(f"uncommitted changes in files baked into the sim image (commit first, or --allow-dirty):\n{dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"model": a.model, "runs": a.runs, "task": a.task, "harbor": HARBOR, "commit": commit,
                                                 "dirty": bool(dirty)}, indent=1))
    print("[eval] do not edit loophole/, docker/ or verifier/ until the eval finishes (the image is rebuilt per trial)", flush=True)
    rows = []
    rows += run("oracle", ["-a", "oracle"], out, a.task)
    rows += run("noop", ["-a", "nop"], out, a.task)
    rows += run("stop-early", ["-a", "oracle", "--ae", "LOOPHOLE_SOLUTION=stop_early"], out, a.task)
    rows += run("opencode", ["-a", "opencode", "-m", a.model, "-k", str(a.runs), "-n", str(a.runs),
                             "--agent-timeout-multiplier", str(a.agent_timeout_multiplier)], out, a.task)
    print(summary(out, a.model, rows))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
