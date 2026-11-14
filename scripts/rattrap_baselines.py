#!/usr/bin/env python3
"""Rat Trap baselines through the agent interface (RatTrapEpisode), scored by the real reward.

    .venv/bin/python scripts/rattrap_baselines.py oracle    # the reference solve's commands: solved, p = 1, score 0.833
    .venv/bin/python scripts/rattrap_baselines.py noop      # no action: score 0
    .venv/bin/python scripts/rattrap_baselines.py random [--seed 0]   # random grabs / moves / releases to the full budget

Prints one JSON line (score, solved, actions, checkpoints reached) and exits non-zero if the result contradicts the
reward's guarantees (oracle not solved or not at p = 1; noop or random above zero partial credit by more than 1/K).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.reward import RatTrapEpisode  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import load_commands  # noqa: E402

CMDS = ROOT / "loophole" / "reference" / "rattrap-solve.commands.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["oracle", "noop", "random"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = load_spec()
    if a.which == "random":
        from loophole.puzzles.rattrap.random_agent import run

        r = run(a.seed, spec)
        ok = not r["solved"] and (r["checkpoints_reached"] or 0) <= 1
    else:
        ep = RatTrapEpisode(spec)
        if a.which == "oracle":
            for tool, args in load_commands(CMDS):
                if ep.done:
                    break
                ep.act(tool, args)
        r = {"baseline": a.which, "score": round(ep.score(), 6), "solved": ep.solved, "violation": ep.violation, "actions": ep.actions,
             "T": ep.T, "p": ep.progress(), "checkpoints_reached": None if ep.tracker is None else ep.tracker.k,
             "K": None if ep.tracker is None else ep.tracker.cps.K}
        ok = (ep.solved and ep.progress() == 1.0) if a.which == "oracle" else ep.score() == 0.0
    print(json.dumps(r), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
