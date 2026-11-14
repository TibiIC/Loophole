#!/usr/bin/env python3
"""Re-score a logged Rat Trap agent run offline under the current reward (no API calls): replay its commands.jsonl
through RatTrapEpisode (deterministic, same physics) and report score, p, furthest checkpoint and blocked moves.

    PYTHONPATH=$PWD .venv/bin/python scripts/rescore_rattrap.py runs/rattrap-agents/<run>/ [--T 852]

The budget is the current YAML's T unless --T is given (round 1 ran with T = 400; its runs used fewer actions than
either, so only the solved branch, which none reached, would differ).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.checkpoints import max_dev  # noqa: E402
from loophole.puzzles.rattrap.reward import RatTrapEpisode  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402


def rescore(run: Path, T: int | None = None) -> dict:
    lines = [json.loads(l) for l in (run / "commands.jsonl").read_text().splitlines() if l.strip()]
    cmds = [(r["tool"], r["args"]) for r in lines if r.get("type") != "header"]
    spec = load_spec(overrides={"max_actions": T} if T else None)
    ep = RatTrapEpisode(spec)
    blocked = 0
    furthest_at = None
    closest_c1 = max_dev(ep.sim.node_positions(), ep.tracker.cps.states[0]) if ep.tracker else None
    for i, (tool, a) in enumerate(cmds, 1):
        if ep.done:
            break
        out = ep.act(tool, a)
        blocked += bool(out.get("blocked"))
        if ep.tracker is not None:
            closest_c1 = min(closest_c1, max_dev(ep.sim.node_positions(), ep.tracker.cps.states[0]))
        if ep.tracker is not None and ep.tracker.reached_at and furthest_at != ep.tracker.reached_at[-1]:
            furthest_at = ep.tracker.reached_at[-1]
    return {"run": run.name, "actions": ep.actions, "solved": ep.solved, "violation": ep.violation, "score": ep.score(),
            "p": ep.progress(), "checkpoints_reached": ep.tracker.k if ep.tracker else None, "K": ep.tracker.cps.K if ep.tracker else None,
            "furthest_checkpoint_at_action": furthest_at, "blocked_moves": blocked,
            "moves": sum(1 for t, _ in cmds[: ep.actions] if t == "move"),
            "closest_to_c1_mm": None if closest_c1 is None else round(closest_c1 * 1000, 1),
            "eps_mm": None if ep.tracker is None else round(ep.tracker.cps.eps * 1000, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--T", type=int, default=None)
    a = ap.parse_args()
    for r in a.runs:
        res = rescore(r, a.T)
        (r / "rescore.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
