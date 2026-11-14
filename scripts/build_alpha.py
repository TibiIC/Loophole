#!/usr/bin/env python3
"""Offline build for the Alpha-style puzzles: reference plan + roadmap.

    python scripts/build_alpha.py                 # all variants, reference + roadmap
    python scripts/build_alpha.py --only alpha-hard --skip-roadmap

Per variant:
1. RRT-Connect from the start pose to a separated pose, using the shared
   collision checker with the largest planning margin that succeeds
   (margin ladder; the achieved margin is a measured lower bound on half the
   narrow-passage width), then shortcut.
2. Dense re-validation of the path at margin 0 (0.5 mm metric resolution).
3. Conversion to rig commands; replay through the simulator (same rig as the
   agent) must reach separation with zero violations.
4. Writes loophole/reference/trajectories/<variant>.jsonl (+ _stop_early), the
   path in puzzles/alpha/<variant>-reference.npz, and the roadmap
   (puzzles/alpha/roadmap-<variant>.npz, see reward/roadmap.py).
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
from loophole.planning.se3 import char_length  # noqa: E402
from loophole.reference.recorder import write_trajectory  # noqa: E402
from loophole.reference.replay import replay  # noqa: E402
from loophole.sim.collision import CollisionChecker, interpolate, pose_metric  # noqa: E402
from loophole.sim.env import LoopholeEnv  # noqa: E402
from loophole.sim.spec import load_puzzle, resolve_puzzle  # noqa: E402

ALPHA = ROOT / "loophole" / "puzzles" / "alpha"
TRAJ = ROOT / "loophole" / "reference" / "trajectories"
VARIANTS = ["alpha-easy", "alpha-medium", "alpha-hard"]


def start_pose(p) -> np.ndarray:
    o = next(o for o in p.objects if o.name == p.target)
    return np.array([*o.pos, *o.quat], float)


def weight(p) -> float:
    o = next(o for o in p.objects if o.name == p.target)
    g = next(g for g in o.geoms if g.type == "tube")
    return char_length(np.asarray(g.points)) + g.tube_radius


def separated_goal(ck: CollisionChecker, start: np.ndarray, margin: float) -> np.ndarray:
    for d in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
        g = start.copy()
        g[:3] += 0.6 * np.asarray(d, float)
        if ck.is_valid(g, margin) and ck.is_separated(g):
            return g
    raise RuntimeError("no separated goal pose found")


def densify(path: list[np.ndarray], w: float, res: float) -> np.ndarray:
    out = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(float(pose_metric(a, b, w)) / res)))
        out += [interpolate(a, b, i / n) for i in range(n)]
    out.append(path[-1])
    return np.array(out)


def plan(variant: str, seed: int, time_per_margin: float) -> dict:
    p = load_puzzle(resolve_puzzle(variant))
    ck = CollisionChecker(p)
    w = weight(p)
    c = p.meta["nominal_gap_clearance"]
    start = start_pose(p)
    assert ck.is_valid(start), "canonical start must be collision-free"
    for frac in (0.4, 0.3, 0.2, 0.15, 0.1):
        margin = frac * c
        try:
            goal = separated_goal(ck, start, margin)
        except RuntimeError:
            continue
        if not ck.is_valid(start, margin):
            continue
        cfg = PlannerConfig(
            w=w, step=0.05, resolution=c / 2, margin=margin, lo=np.full(3, -0.6), hi=np.full(3, 0.6),
            max_time=time_per_margin, seed=seed, focus=np.array([[0, 0, 0], start[:3]]), focus_sigma=0.04, focus_prob=0.3,
        )
        print(f"[{variant}] planning with margin {margin * 1000:.2f} mm ({frac:.2f} x clearance)")
        t0 = time.time()
        path = rrt_connect(ck, start, goal, cfg, log=lambda m: print("   ", m))
        if path is None:
            continue
        raw_len = path_length(path, w)
        path = shortcut(ck, path, w, c / 2, margin, iters=800, seed=seed)
        dense = densify(path, w, 0.0005)
        ok = all(ck.is_valid(x) for x in dense)
        clear = min(ck.clearance(x, 0.05) for x in dense)
        print(f"[{variant}] path: {len(path)} waypoints, length {raw_len:.3f} -> {path_length(path, w):.3f}, dense valid={ok}, min clearance {clear * 1000:.2f} mm, {time.time() - t0:.0f}s")
        if ok:
            return {"puzzle": p, "path": path, "margin": margin, "w": w, "min_clearance": clear, "dense": dense}
    raise RuntimeError(f"{variant}: planning failed at every margin")


def build_reference(variant: str, seed: int, time_per_margin: float) -> dict:
    res = plan(variant, seed, time_per_margin)
    p = res["puzzle"]
    env = LoopholeEnv(p)
    cmds = path_to_commands(res["path"], p.target, env.cfg.rig.max_linear_step, env.cfg.rig.max_angular_step)
    final = replay(env, cmds)
    print(f"[{variant}] replay: success={final['success']} actions={final['actions_used']} score={final['score']:.3f} violations={final['violations']['count']}")
    if not final["success"] or final["violations"]["count"]:
        raise RuntimeError(f"{variant}: reference replay failed")
    write_trajectory(TRAJ / f"{variant}.jsonl", variant, cmds, source="rrt-connect", meta={"expected_score": final["score"], "planning_margin": res["margin"], "seed": seed})
    # stop-early baseline: grab + 5 random (seeded) moves, then stop
    rng = np.random.default_rng(seed + 1)
    lin, ang = env.cfg.rig.max_linear_step, env.cfg.rig.max_angular_step
    se = [("grab", {"handle": 0, "target": p.target})] + [
        ("move", {"handle": 0, **dict(zip(("dx", "dy", "dz"), (rng.uniform(-lin, lin, 3) / np.sqrt(3)).tolist())), **dict(zip(("droll", "dpitch", "dyaw"), (rng.uniform(-ang, ang, 3) / np.sqrt(3)).tolist()))})
        for _ in range(5)
    ]
    write_trajectory(TRAJ / f"{variant}_stop_early.jsonl", variant, se, source="baseline:stop_early(5 random moves)")
    np.savez(ALPHA / f"{variant}-reference.npz", path=np.array(res["path"]), dense=res["dense"], margin=res["margin"], w=res["w"])
    return {
        "planning_margin_mm": round(res["margin"] * 1000, 3),
        "path_min_clearance_mm": round(res["min_clearance"] * 1000, 3),
        "reference_actions": final["actions_used"],
        "reference_score": final["score"],
        "waypoints": len(res["path"]),
    }


def reference_from_roadmap(variant: str, seed: int) -> dict:
    """Replace the RRT reference by the roadmap's shortest path (shortcut, densely
    validated, replayed through the rig). Keeps the RRT path if the replay fails."""
    from loophole.reward.roadmap import Roadmap

    p = load_puzzle(resolve_puzzle(variant))
    ck = CollisionChecker(p)
    rm = Roadmap.for_puzzle(p)
    start = start_pose(p)
    c = p.meta["nominal_gap_clearance"]
    _, n = rm.cost_to_go(start, ck)
    adj = [[] for _ in rm.nodes]
    for i, j in rm.edges:
        adj[i].append(j)
        adj[j].append(i)
    path = [start, rm.nodes[n]]
    cur = n
    while rm.cost[cur] > 0:
        cur = min(adj[cur], key=lambda j: float(pose_metric(rm.nodes[cur], rm.nodes[j], rm.w)) + rm.cost[j])
        path.append(rm.nodes[cur])
    # carry on a little beyond the goal boundary so the replay clearly separates
    end = path[-1].copy()
    away = end[:3] - start[:3]
    end[:3] += 0.05 * away / max(np.linalg.norm(away), 1e-9)
    if ck.is_valid(end) and ck.segment_valid(path[-1], end, rm.w, c / 2):
        path.append(end)
    raw = path_length(path, rm.w)
    path = shortcut(ck, path, rm.w, c / 2, 0.1 * c, iters=1500, seed=seed)
    dense = densify(path, rm.w, 0.0005)
    if not all(ck.is_valid(x) for x in dense):
        print(f"[{variant}] roadmap path invalid when densified; keeping RRT reference")
        return {"reference_source": "rrt-connect"}
    env = LoopholeEnv(p)
    cmds = path_to_commands(path, p.target, env.cfg.rig.max_linear_step, env.cfg.rig.max_angular_step)
    final = replay(env, cmds)
    if not final["success"] or final["violations"]["count"]:
        print(f"[{variant}] roadmap path did not replay cleanly ({final['done_reason']}); keeping RRT reference")
        return {"reference_source": "rrt-connect"}
    old = dict(np.load(ALPHA / f"{variant}-reference.npz"))
    np.savez(ALPHA / f"{variant}-reference.npz", path=np.array(path), dense=dense, margin=0.0, w=rm.w,
             seed_path=old["path"], seed_dense=old["dense"], seed_margin=old["margin"])
    write_trajectory(TRAJ / f"{variant}.jsonl", variant, cmds, source="roadmap-shortest-path", meta={"expected_score": final["score"], "seed": seed})
    clear = min(ck.clearance(x, 0.05) for x in dense)
    print(f"[{variant}] roadmap reference: length {raw:.3f} -> {path_length(path, rm.w):.3f}, min clearance {clear * 1000:.2f} mm, replay actions={final['actions_used']} score={final['score']:.3f}")
    return {
        "reference_source": "roadmap-shortest-path",
        "path_min_clearance_mm": round(clear * 1000, 3),
        "reference_actions": final["actions_used"],
        "reference_score": final["score"],
        "waypoints": len(path),
        "reference_length": round(path_length(path, rm.w), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-per-margin", type=float, default=300.0)
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--skip-roadmap", action="store_true")
    ap.add_argument("--skip-final", action="store_true", help="keep the RRT reference instead of the roadmap shortest path")
    a = ap.parse_args()
    report = {}
    for v in [a.only] if a.only else VARIANTS:
        report[v] = {}
        if not a.skip_reference:
            report[v].update(build_reference(v, a.seed, a.time_per_margin))
        if not a.skip_roadmap:
            from loophole.reward.roadmap import build_roadmap

            report[v].update(build_roadmap(v, seed=a.seed))
        if not a.skip_final and not a.skip_roadmap:
            report[v].update(reference_from_roadmap(v, a.seed))
        print(json.dumps({v: report[v]}))
    out = ALPHA / "build_report.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(report)
    out.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
