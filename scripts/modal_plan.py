#!/usr/bin/env python3
"""Plan an independent Alpha 1.0 reference on Modal: one container per seed.

    pip install modal==1.6.4 && modal token new          # once
    modal run scripts/modal_plan.py --seeds 50 --budget 7200 [--planner rrt-connect] [--seed0 1000]

* Image: built from docker/sim.Dockerfile, the simulator / verifier image (Image.from_dockerfile, context =
  repository root, .dockerignore respected). The Alpha 1.0 meshes are fetched during the image build from
  the pinned commit with the same sha256 checks as now (scripts/fetch_alpha_paper.sh). They live only in
  the Modal image, never in a committed artifact. Hull decomposition and MuJoCo are therefore identical to
  the verifier's Linux environment.
* One container per seed (Function.map): RRT-Connect on our MuJoCo collision check on our hulls
  (scripts/plan_reference.py: _solve), `budget` seconds, then heavy shortcutting (rotation-first partial,
  translation partial, full), validated at 1 mm, truncated at the first free pose.
* Each seed's result (metrics, shortest path, platform info) is written to the Modal Volume
  `loophole-plan-seeds`, and pulled down to runs/plan-seeds/ by the local entrypoint, with summary.json
  (path-length distribution, corridor clustering, the shortest path).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = modal.App("loophole-plan-reference")
VOLUME = modal.Volume.from_name("loophole-plan-seeds", create_if_missing=True)
image = (
    modal.Image.from_dockerfile(str(ROOT / "docker" / "sim.Dockerfile"), context_dir=str(ROOT))
    .env({"PYTHONPATH": "/opt/loophole:/opt/loophole/scripts"})
    .add_local_file(str(ROOT / "scripts" / "plan_reference.py"), "/opt/loophole/scripts/plan_reference.py")
    .add_local_file(str(ROOT / "scripts" / "smooth_oracle.py"), "/opt/loophole/scripts/smooth_oracle.py")
)


@APP.function(image=image, timeout=600)
def setup() -> dict:
    import os

    os.chdir("/opt/loophole")
    from loophole.config import load_config
    from loophole.sim.env import LoopholeEnv

    env = LoopholeEnv("alpha-1.0-paper", config=load_config("/opt/loophole/loophole/configs/alpha_training.yaml"))
    lo, hi = env.frame_aabb()
    out = {"start": env.target_pose().tolist(), "lo": (lo - 1.0).tolist(), "hi": (hi + 1.0).tolist(), "w": float(env.roadmap.w)}
    env.close()
    return out


@APP.function(image=image, cpu=1.0, memory=6144, timeout=24 * 3600, volumes={"/out": VOLUME}, retries=0)
def plan_seed(seed: int, budget: float, start: list, lo: list, hi: list, w: float, planner: str) -> dict:
    import os
    import sys

    os.chdir("/opt/loophole")
    sys.path[:0] = ["/opt/loophole", "/opt/loophole/scripts"]
    import plan_reference as pr

    pr._init()
    r = pr._solve((seed, budget, start, lo, hi, w, planner))
    r["where"] = "modal"
    (Path("/out") / f"seed-{seed}.json").write_text(json.dumps(r))
    VOLUME.commit()
    return {k: v for k, v in r.items() if k != "path"}


@APP.local_entrypoint()
def main(seeds: int = 50, budget: float = 7200.0, planner: str = "rrt-connect", seed0: int = 1000):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from plan_reference import summarise

    s = setup.remote()
    jobs = [(seed0 + k, budget, s["start"], s["lo"], s["hi"], s["w"], planner) for k in range(seeds)]
    for r in plan_seed.starmap(jobs, order_outputs=False, return_exceptions=True):
        print(r if isinstance(r, Exception) else {k: r.get(k) for k in ("seed", "solved", "plan_s", "shortcut", "platform")}, flush=True)
    out = ROOT / "runs" / "plan-seeds"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in VOLUME.listdir("/"):
        if entry.path.startswith("seed-") and entry.path.endswith(".json"):
            data = b"".join(VOLUME.read_file(entry.path))
            (out / entry.path).write_bytes(data)
            rows.append(json.loads(data))
    summary = summarise(rows, s["w"], out)
    print(json.dumps(summary, indent=1))
