"""Do-nothing baseline: no actions at all. Must score exactly 0.

In Harbor use the built-in `nop` agent (`harbor run -a nop ...`); this module
runs the same baseline locally:  python -m loophole.baselines.noop --puzzle easy
"""

from __future__ import annotations

import argparse
import json

from ..sim.env import LoopholeEnv

STEPS: list[tuple[str, dict]] = []


def run(puzzle: str, log_path=None) -> dict:
    env = LoopholeEnv(puzzle, log_path=log_path)
    return env.finalize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default="easy")
    print(json.dumps({k: v for k, v in run(ap.parse_args().puzzle).items() if k in ("score", "success", "actions_used")}))
