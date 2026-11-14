#!/usr/bin/env python3
"""Demo: load the Alpha puzzle, grab tube B, make a few moves, save renders.

    .venv/bin/python scripts/demo.py [--out runs/demo]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loophole.sim.env import LoopholeEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--puzzle", default="alpha-medium")
ap.add_argument("--out", default="runs/demo")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

env = LoopholeEnv(a.puzzle)
png, cap = env.render("iso")
(out / "0_start.png").write_bytes(png)
print("start:", cap)
env.act("grab", {"handle": 0, "target": "tube_b"})
for mv in ({"dz": 0.05}, {"dx": 0.05}, {"dyaw": 0.4}, {"dz": -0.05}):
    r = env.act("move", {"handle": 0, **mv})
    print(json.dumps({"move": mv, "blocked": r["blocked"], "clamped": r["clamped"], "min_distance": r["goal"]["target_min_distance_to_frame"]}))
png, cap = env.render("iso")
(out / "1_after_moves.png").write_bytes(png)
print("after:", cap)
f = env.finalize()
print("score", f["score"], "success", f["success"], "actions", f["actions_used"], "violations", f["violations"]["count"])
print("renders in", out)
