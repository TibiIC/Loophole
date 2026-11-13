"""Stop-early baseline: a few plausible moves from the reference, then stop.

In Harbor: `harbor run -a oracle --ae LOOPHOLE_SOLUTION=stop_early ...` replays
the stored stop-early trajectory. Locally:  python -m loophole.baselines.stop_early --puzzle easy
"""

from __future__ import annotations

import argparse
import json

from ..reference.replay import replay
from ..reference.scripted import SCRIPTS
from ..sim.env import LoopholeEnv

N_STEPS = 3  # e.g. easy: grab the bight tip and lift it twice, then give up


def steps(puzzle: str) -> list[tuple[str, dict]]:
    return SCRIPTS[puzzle][:N_STEPS]


def run(puzzle: str, log_path=None) -> dict:
    return replay(LoopholeEnv(puzzle, log_path=log_path), steps(puzzle))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default="easy")
    print(json.dumps({k: v for k, v in run(ap.parse_args().puzzle).items() if k in ("score", "success", "actions_used")}))
