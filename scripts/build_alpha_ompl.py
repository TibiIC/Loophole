#!/usr/bin/env python3
"""Offline build for the canonical OMPL.app Alpha puzzle (default: Alpha 1.5).

    make fetch-alpha && .venv/bin/python scripts/build_alpha_ompl.py [--variant 1.5] [--time 1200]

1. Writes loophole/puzzles/alpha_ompl/alpha-<v>.yaml (poses + parameters only).
2. RRT-Connect (numpy, SE(3)) from the .cfg start to the .cfg goal (pushed further out
   along the same direction until the separation criterion holds), using the shared
   MuJoCo collision checker on the decomposed hulls. Time-boxed.
3. Shortcut, dense re-validation (margin 0), conversion to grab/move commands,
   replay through the simulator rig: must succeed with zero violations.
4. Writes loophole/reference/trajectories/alpha-<v>.jsonl (+ _stop_early) and the
   path (poses only) to loophole/puzzles/alpha_ompl/alpha-<v>-reference.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.planning.commands import path_to_commands  # noqa: E402
from loophole.planning.rrt import PlannerConfig, path_length, rrt_connect, shortcut  # noqa: E402
from loophole.puzzles.ompl_alpha import PUZZLE_OUT, UNIT_SCALE, axis_angle_quat, load_scene_mesh, ompl_dir, parse_cfg, robot_center, write_puzzle  # noqa: E402
from loophole.reference.recorder import write_trajectory  # noqa: E402
from loophole.reference.replay import replay  # noqa: E402
from loophole.sim.collision import CollisionChecker, interpolate, pose_metric  # noqa: E402
from loophole.sim.env import LoopholeEnv  # noqa: E402
from loophole.sim.spec import load_puzzle  # noqa: E402

TRAJ = ROOT / "loophole" / "reference" / "trajectories"
D_SEP_NATIVE = 40.0  # 0.4 m: above the largest clearance reachable while still interlocked (measured)


def config_for(cfg: dict) -> dict:
    vmin = [v * UNIT_SCALE for v in cfg["volume"]["min"]]
    vmax = [v * UNIT_SCALE for v in cfg["volume"]["max"]]
    return {
        "reset_settle_steps": 20,
        "rig": {
            "num_handles": 1,
            "max_linear_step": 0.05,
            "max_angular_step": 0.1,
            "linear_speed": 0.1,
            "angular_speed": 0.1,
            "weld_solref": [0.002, 1.0],
            "lag_limit": 0.006,
            "angle_lag_limit": 0.02,
            "workspace_min": vmin,
            "workspace_max": vmax,
            "grasp_tolerance": 3.0,
        },
        "topology": {"enabled": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="1.5")
    ap.add_argument("--time", type=float, default=1200.0, help="planning time box (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=0, help="0: 4x the reference length")
    ap.add_argument("--focus-rot", type=float, default=0.0, help="if > 0: focused samples stay within this rotation std (rad) of the start orientation")
    a = ap.parse_args()
    v = a.variant
    cfg = parse_cfg(ompl_dir() / f"alpha-{v}.cfg")
    yaml_path = write_puzzle(v, d_sep_native=D_SEP_NATIVE, config=config_for(cfg))
    p = load_puzzle(yaml_path)
    ck = CollisionChecker(p)
    B = load_scene_mesh(ompl_dir() / "alpha_robot.dae", assimp_frame=True)
    w = float(np.max(np.linalg.norm(B.vertices - robot_center(B), axis=1))) * UNIT_SCALE
    start = np.array([*p.objects[0].pos, *p.objects[0].quat])
    g = cfg["goal"]
    goal = np.array([*(np.asarray(g["xyz"]) * UNIT_SCALE), *axis_angle_quat(g["theta"], g["axis"])])
    direction = goal[:3] - start[:3]
    direction /= np.linalg.norm(direction)
    print(f"start valid={ck.is_valid(start)} clearance={ck.clearance(start, 0.2):.4f} m | cfg goal clearance={ck.clearance(goal, 1.0):.4f} m separated={ck.is_separated(goal)}")
    while not ck.is_separated(goal):  # same direction, further out, until the success criterion holds
        goal[:3] += 0.05 * direction
    print(f"planning goal: cfg goal + {np.linalg.norm(goal[:3] - (np.asarray(g['xyz']) * UNIT_SCALE)):.2f} m along the cfg start->goal direction")
    vmin = np.array(p.config["rig"]["workspace_min"])
    vmax = np.array(p.config["rig"]["workspace_max"])
    pc = PlannerConfig(w=w, step=0.08, resolution=0.002, margin=0.0, lo=vmin, hi=vmax, max_time=a.time, seed=a.seed,
                       focus=np.array([start[:3]]), focus_sigma=0.3, focus_prob=0.5,
                       focus_quat=start[3:] if a.focus_rot > 0 else None, focus_rot_sigma=a.focus_rot or 0.5)
    t0 = time.time()
    path = rrt_connect(ck, start, goal, pc, log=lambda m: print("   ", m, flush=True))
    if path is None:
        print(f"PLANNING FAILED within {a.time:.0f}s (geometry unchanged)")
        sys.exit(2)
    raw = path_length(path, w)
    path = shortcut(ck, path, w, 0.002, 0.0, iters=1500, seed=a.seed)
    dense = []
    for x, y in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(float(pose_metric(x, y, w)) / 0.0005)))
        dense += [interpolate(x, y, i / n) for i in range(n)]
    dense.append(path[-1])
    dense = np.array(dense)
    ok = all(ck.is_valid(x) for x in dense)
    print(f"path: {len(path)} waypoints, metric length {raw:.2f} -> {path_length(path, w):.2f}; dense ({len(dense)} states @0.5 mm) valid={ok}; {time.time() - t0:.0f}s")
    if not ok:
        sys.exit(3)
    env = LoopholeEnv(p)
    cmds = path_to_commands(path, p.target, env.cfg.rig.max_linear_step, env.cfg.rig.max_angular_step)
    max_actions = a.max_actions or int(4 * len(cmds))
    write_puzzle(v, d_sep_native=D_SEP_NATIVE, config=config_for(cfg), max_actions=max_actions)
    p = load_puzzle(yaml_path)
    final = replay(LoopholeEnv(p), cmds)
    print(f"replay: success={final['success']} actions={final['actions_used']}/{max_actions} score={final['score']:.3f} violations={final['violations']['count']}")
    if not final["success"] or final["violations"]["count"]:
        sys.exit(4)
    name = f"alpha-{v}"
    write_trajectory(TRAJ / f"{name}.jsonl", name, cmds, source="rrt-connect (canonical OMPL.app geometry)", meta={"expected_score": final["score"], "seed": a.seed})
    rng = np.random.default_rng(a.seed + 1)
    se = [("grab", {"handle": 0, "target": p.target})] + [
        ("move", {"handle": 0, "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2])}) for d in rng.uniform(-0.03, 0.03, (5, 3))
    ]
    write_trajectory(TRAJ / f"{name}_stop_early.jsonl", name, se, source="baseline:stop_early(5 random moves)")
    np.savez(PUZZLE_OUT / f"{name}-reference.npz", path=np.array(path), dense=dense, w=w)
    rep = {"variant": v, "reference_actions": final["actions_used"], "max_actions": max_actions, "reference_score": final["score"],
           "planning_s": round(time.time() - t0, 1), "waypoints": len(path), "min_clearance_m": round(min(ck.clearance(x, 0.05) for x in dense[::5]), 5)}
    (PUZZLE_OUT / f"{name}-build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
