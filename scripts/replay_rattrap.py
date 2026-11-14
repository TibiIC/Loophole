#!/usr/bin/env python3
"""Replay a recorded Rat Trap solve headlessly from its converted commands and check it reaches the same state.

    .venv/bin/python scripts/replay_rattrap.py [--mode commands|exact] [loophole/reference/rattrap-solve.commands.jsonl]

--mode commands (default): replay the converted discrete commands (the agent's tool interface).
--mode exact: re-execute the raw recording step for step (the determinism check; also how a recorded solve is
              re-validated after the model changes, e.g. with the mouse tied on).
Asserts that the final state is free (the frozen cord assembly can slide out), that no node ends more than --tol
from the live final position, and that nothing penetrated beyond the audit tolerance. A failure in "commands" mode
with a pass in "exact" mode means the conversion is broken, not the determinism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import load_commands, replay  # noqa: E402
from loophole.puzzles.rattrap.topology import build_discs, read_word  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("commands", type=Path, nargs="?", default=ROOT / "loophole" / "reference" / "rattrap-solve.commands.jsonl")
    ap.add_argument("--tol", type=float, default=0.01, help="max node deviation from the live final state (m)")
    ap.add_argument("--mode", choices=["commands", "exact"], default="commands")
    a = ap.parse_args()
    hdr = json.loads(a.commands.open().readline())
    live = hdr.get("live_final") or {}
    spec = load_spec()
    if a.mode == "exact":
        from loophole.puzzles.rattrap.teleop import load_records, replay_exact

        raw = a.commands.with_name(a.commands.name.replace(".commands.jsonl", ".jsonl"))
        cmds = []
        sim = replay_exact(load_records(raw), spec)
    else:
        cmds = load_commands(a.commands)
        sim = replay(cmds, spec)
    nodes = sim.node_positions()
    word = read_word(nodes, build_discs(spec), closed=bool(spec["cord_closed"]))
    from loophole.puzzles.rattrap.reward import FrozenEscape

    free = FrozenEscape(spec, sim).escape_direction(nodes) is not None
    dev = float(np.max(np.linalg.norm(nodes - np.asarray(live["nodes"]), axis=1))) if live.get("nodes") else None
    out = {"mode": a.mode, "commands": len(cmds), "free": free, "replay_word": word, "live_word": live.get("word"),
           "max_node_deviation_m": dev, "max_penetration_m": sim.max_penetration}
    print(json.dumps(out, indent=1))
    ok = free and (dev is None or dev <= a.tol) and sim.max_penetration < spec["penetration_tolerance"]
    if not ok:
        sys.exit("REPLAY DIVERGES from the live solve: the conversion or the determinism is broken")
    print("replay reaches the live solve's final state")


if __name__ == "__main__":
    main()
