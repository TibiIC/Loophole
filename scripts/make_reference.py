#!/usr/bin/env python3
"""Regenerate stored reference / stop-early trajectories from the scripted solutions
(and check that each reference really solves its puzzle)."""

import sys
from pathlib import Path

from loophole.baselines import stop_early
from loophole.reference.recorder import write_trajectory
from loophole.reference.replay import replay
from loophole.reference.scripted import SCRIPTS
from loophole.sim.env import LoopholeEnv

OUT = Path(__file__).resolve().parents[1] / "loophole" / "reference" / "trajectories"

ok = True
for puzzle, steps in SCRIPTS.items():
    final = replay(LoopholeEnv(puzzle), steps)
    print(f"{puzzle}: score={final['score']:.3f} success={final['success']} actions={final['actions_used']}")
    ok &= bool(final["success"])
    write_trajectory(OUT / f"{puzzle}.jsonl", puzzle, steps, source="scripted", meta={"expected_score": final["score"]})
    se = stop_early.steps(puzzle)
    write_trajectory(OUT / f"{puzzle}_stop_early.jsonl", puzzle, se, source="baseline:stop_early")
sys.exit(0 if ok else 1)
