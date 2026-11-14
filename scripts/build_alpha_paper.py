#!/usr/bin/env python3
"""Build the Alpha 1.0 oracle from the published solution of Zhang et al. (SIGGRAPH 2020).

    make fetch-alpha-paper
    .venv/bin/python scripts/build_alpha_paper.py --supp path/to/supp-material.zip

Inputs (never committed): the paper's meshes + .cfg (scripts/fetch_alpha_paper.sh) and
the authors' simplified solution path `supp-material/puzzles/alpha/alpha-1.0.simplified.txt`
from the paper's supplementary archive
(https://xinyazhang.gitlab.io/puzzletunneldiscovery/assets/supp-material.zip).

Steps: write the puzzle YAML; densely validate the authors' path in our simulator
(MuJoCo, margin 0, <= 0.8 mm point motion between checks); convert it to rig commands;
replay through the simulator (must separate with zero violations); write the reference
trajectory and the 22 solution poses (poses only) with their provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.planning.commands import path_to_commands  # noqa: E402
from loophole.puzzles import paper_alpha as pa  # noqa: E402
from loophole.puzzles.ompl_alpha import parse_cfg  # noqa: E402
from loophole.reference.recorder import write_trajectory  # noqa: E402
from loophole.reference.replay import replay  # noqa: E402
from loophole.sim.collision import CollisionChecker, interpolate, pose_metric  # noqa: E402
from loophole.sim.env import LoopholeEnv  # noqa: E402
from loophole.sim.spec import load_puzzle  # noqa: E402

SUPP_SHA256 = "00dff255b3ca002240f0be157ec70431377bff2c187dafe13659be706815e07c"
SIMPLIFIED = "supp-material/puzzles/alpha/alpha-1.0.simplified.txt"
TRAJ = ROOT / "loophole" / "reference" / "trajectories"
NAME = "alpha-1.0-paper"
MAX_LIN, MAX_ANG = 0.1, 0.2  # per-move caps (velocity limits are what prevent tunnelling)


def config_for(cfg: dict) -> dict:
    s = pa.UNIT_SCALE
    return {
        "reset_settle_steps": 20,
        "rig": {
            "num_handles": 1, "max_linear_step": MAX_LIN, "max_angular_step": MAX_ANG, "linear_speed": 0.1, "angular_speed": 0.1,
            "weld_solref": [0.002, 1.0], "lag_limit": 0.006, "angle_lag_limit": 0.02,
            "workspace_min": [v * s for v in cfg["volume"]["min"]], "workspace_max": [v * s for v in cfg["volume"]["max"]],
            "grasp_tolerance": 3.0,
        },
        "topology": {"enabled": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supp", required=True, help="the paper's supp-material.zip")
    ap.add_argument("--resolution", type=float, default=0.0008, help="dense check: max point motion between checks (m)")
    a = ap.parse_args()
    raw = Path(a.supp).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUPP_SHA256:
        sys.exit("supp-material.zip sha256 mismatch")
    with zipfile.ZipFile(a.supp) as z:
        text = z.read(SIMPLIFIED).decode()
    rows = np.array([list(map(float, l.split())) for l in text.splitlines() if l.strip()])
    native = np.concatenate([rows[:, :3], rows[:, 6:7], rows[:, 3:6]], axis=1)  # x y z qw qx qy qz
    path = native.copy()
    path[:, :3] *= pa.UNIT_SCALE

    cfg = parse_cfg(pa.paper_dir() / pa.FILES["cfg"])
    s = cfg["start"]
    assert np.allclose(native[0, :3], s["xyz"]), "solution must start at the .cfg start"
    cmds_probe = path_to_commands(list(path), "tube_b", MAX_LIN, MAX_ANG)
    max_actions = int(np.ceil(2.5 * len(cmds_probe)))
    p = load_puzzle(pa.write_puzzle(config_for(cfg), max_actions=max_actions, name=NAME))
    ck = CollisionChecker(p)
    rob = pa.load_obj(pa.paper_dir() / pa.FILES["robot"])
    w = float(np.max(np.linalg.norm(rob.vertices - np.asarray(rob.vertices).mean(0), axis=1))) * pa.UNIT_SCALE

    n = bad = 0
    for x, y in zip(path[:-1], path[1:]):
        k = max(1, int(np.ceil(float(pose_metric(x, y, w)) / a.resolution)))
        for i in range(k):
            n += 1
            bad += not ck.is_valid(interpolate(x, y, i / k))
    print(f"authors' path: {len(path)} poses; dense check {n} states @{a.resolution * 1000:.1f} mm -> invalid {bad}")
    if bad:
        sys.exit("the published path is not valid in our simulator")
    env = LoopholeEnv(p)
    cmds = path_to_commands(list(path), p.target, env.cfg.rig.max_linear_step, env.cfg.rig.max_angular_step)
    final = replay(env, cmds)
    print(f"replay: success={final['success']} actions={final['actions_used']}/{max_actions} score={final['score']:.3f} violations={final['violations']['count']}")
    if not final["success"] or final["violations"]["count"]:
        sys.exit("replay failed")
    write_trajectory(TRAJ / f"{NAME}.jsonl", NAME, cmds, source="Zhang et al. 2020 supplementary: alpha-1.0.simplified.txt",
                     meta={"expected_score": final["score"], "supp_sha256": SUPP_SHA256})
    rng = np.random.default_rng(1)
    se = [("grab", {"handle": 0, "target": p.target})] + [
        ("move", {"handle": 0, "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2])}) for d in rng.uniform(-0.03, 0.03, (5, 3))
    ]
    write_trajectory(TRAJ / f"{NAME}_stop_early.jsonl", NAME, se, source="baseline:stop_early(5 random moves)")
    np.savez(pa.PUZZLE_OUT / f"{NAME}-reference.npz", path=path, poses_native=native, w=w)
    rep = {"source": "Zhang, Belfer, Kry, Vouga 2020 (supp-material/puzzles/alpha/alpha-1.0.simplified.txt)", "poses": len(path),
           "dense_states_checked": n, "invalid": bad, "reference_actions": final["actions_used"], "max_actions": max_actions,
           "reference_score": final["score"]}
    (pa.PUZZLE_OUT / f"{NAME}-build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
