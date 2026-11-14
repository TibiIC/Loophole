#!/usr/bin/env python3
"""Plan our own Alpha 1.0 reference from scratch: N independent RRT-Connect solves, shortcut, keep the shortest.

    .venv/bin/python scripts/plan_reference.py --seeds 50 --budget 1800 --workers 8 [--out runs/reference-planning]

* Planner: loophole.planning.rrt.rrt_connect on our MuJoCo collision check (CollisionChecker on our
  hulls, margin 0, edges at 4 mm of point motion). Start = the canonical start; goal = the start
  pose moved 3.5 m up (trivially free; the tree from it covers the free region). Uniform sampling in the
  workspace box, no use of the published path or the roadmap. One process per seed.
* Each solution is shortcut heavily (rotation-first partial shortcuts, then translation, then full;
  scripts/smooth_oracle.py) and validated at 1 mm; the solution is truncated at the first pose that is free
  (free escape by translation), since the episode ends there.
* Reported per seed: solved, planning time, raw / shortcut path length (translation m, rotation rad, L).
Results: <out>/seeds.jsonl, <out>/summary.json, <out>/best_path.npy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NAME = "alpha-1.0-paper"
_G = {}


def _init():
    from loophole.puzzles import paper_alpha as pa  # noqa: F401
    from loophole.sim.collision import CollisionChecker, fibonacci_directions
    from loophole.sim.spec import load_puzzle, resolve_puzzle

    pz = load_puzzle(resolve_puzzle(NAME))
    _G["ck"] = CollisionChecker(pz)
    _G["dirs"] = fibonacci_directions(64)


def truncate_at_free(ck, dirs, path, w, step=0.01):
    """Cut the path at the first (densely sampled) pose from which the tube can slide out."""
    from smooth_oracle import densify

    dense = densify(path, w, step)
    lo, hi = 0, len(dense) - 1
    if ck.escape_direction(dense[hi], dirs) is None:
        return path
    while hi - lo > 1:  # free-ness is monotone enough along a solution's tail for bisection
        mid = (lo + hi) // 2
        if ck.escape_direction(dense[mid], dirs) is None:
            lo = mid
        else:
            hi = mid
    return dense[: hi + 1]


def smooth(ck, path, w, rng, rounds=4, tries=600, res=0.001):
    from smooth_oracle import densify, shortcut_pass

    for _ in range(rounds):
        dense = densify(path, w, 0.02)
        changed = 0
        for kind in ("rotation", "translation", "full"):
            dense, c = shortcut_pass(ck, dense, w, res, kind, tries, rng, lambda s: None)
            changed += c
        path = dense
        if not changed:
            break
    simp = [path[0]]
    for k in range(1, len(path) - 1):
        if not ck.segment_valid(simp[-1], path[k + 1], w, res):
            simp.append(path[k])
    simp.append(path[-1])
    return simp


def rrt_connect_retract(ck, start, goal, cfg, rng_seed, retract_tries=12, retract_sigma=(0.02, 0.06),
                        frontier_prob=0.5, cell=0.1):
    """RRT-Connect with two standard narrow-passage aids (the same ideas the winning agent's planner used):
    * retraction: when an extension's new node collides (or its edge does), try `retract_tries` random
      perturbations of it (translation / rotation sigma) and keep the valid one closest to the target that the
      tree node can reach by a collision-free segment;
    * frontier sampling: with probability `frontier_prob`, sample around a node of the tree in one of its
      least-populated translation cells (density balancing), instead of uniformly.
    Collision checks: CollisionChecker.is_valid / segment_valid on our exact hulls, margin 0."""
    from loophole.planning.rrt import Tree
    from loophole.planning.se3 import random_quat, steer
    from loophole.reward.roadmap import _rotate_quat

    rng = np.random.default_rng(rng_seed)
    w, res, step = cfg.w, cfg.resolution, cfg.step
    ta, tb = Tree(), Tree()
    ta.add(start, -1)
    tb.add(goal, -1)
    cells = [dict(), dict()]  # per tree: cell -> node ids

    def add(tree, ci, y, parent):
        j = tree.add(y, parent)
        cells[ci].setdefault(tuple(np.floor(y[:3] / cell).astype(int)), []).append(j)
        return j

    cells[0][tuple(np.floor(start[:3] / cell).astype(int))] = [0]
    cells[1][tuple(np.floor(goal[:3] / cell).astype(int))] = [0]

    def sample(ci, tree):
        if rng.random() < frontier_prob and len(tree.nodes) > 1:
            keys = list(cells[ci])
            counts = np.array([len(cells[ci][k]) for k in keys], float)
            k = keys[int(rng.choice(len(keys), p=(1 / counts) / (1 / counts).sum()))]
            b = tree.nodes[int(rng.choice(cells[ci][k]))]
            return np.concatenate([b[:3] + rng.normal(0, 0.1, 3), _rotate_quat(b[3:], rng.normal(0, 0.4, 3))])
        return np.concatenate([cfg.lo + rng.random(3) * (cfg.hi - cfg.lo), random_quat(rng)])

    def extend(tree, ci, x):
        i = tree.nearest(x, w)
        y = steer(tree.nodes[i], x, w, step)
        if ck.is_valid(y) and ck.segment_valid(tree.nodes[i], y, w, res):
            return ("reached" if np.allclose(y, x) else "advanced"), add(tree, ci, y, i)
        best = None
        for _ in range(retract_tries):  # retraction around the blocked node
            z = np.concatenate([y[:3] + rng.normal(0, retract_sigma[0], 3), _rotate_quat(y[3:], rng.normal(0, retract_sigma[1], 3))])
            if ck.is_valid(z) and ck.segment_valid(tree.nodes[i], z, w, res):
                d = float(pose_metric_np(z, x, w))
                if best is None or d < best[0]:
                    best = (d, z)
        if best is None:
            return "trapped", -1
        return "advanced", add(tree, ci, best[1], i)

    def connect(tree, ci, x):
        while True:
            s_, j = extend(tree, ci, x)
            if s_ != "advanced":
                return s_, j

    t0, a_is_start = time.time(), True
    ia, ib = 0, 1
    while time.time() - t0 < cfg.max_time:
        x = sample(ia, ta)
        s_, j = extend(ta, ia, x)
        if s_ != "trapped":
            s2, k = connect(tb, ib, ta.nodes[j])
            if s2 == "reached":
                pa_, pb_ = ta.path_to_root(j)[::-1], tb.path_to_root(k)
                path = pa_ + pb_[1:]
                return path if a_is_start else path[::-1]
        ta, tb, ia, ib = tb, ta, ib, ia
        a_is_start = not a_is_start
    return None


def pose_metric_np(a, b, w):
    from loophole.sim.collision import pose_metric

    return pose_metric(a, np.asarray(b)[None], w)[0]


def _solve(job):
    from loophole.planning.rrt import PlannerConfig, rrt_connect
    from smooth_oracle import stats

    seed, budget, start, lo, hi, w, planner = job
    ck = _G["ck"]
    start = np.asarray(start)
    goal = start.copy()
    goal[:3] += np.array([0.0, 0.0, 3.5])
    cfg = PlannerConfig(w=w, step=0.15, resolution=0.004, margin=0.0, lo=np.asarray(lo), hi=np.maximum(np.asarray(hi), goal[:3] + 0.5),
                        max_time=budget, seed=seed)
    t0 = time.time()
    path = rrt_connect(ck, start, goal, cfg, log=lambda s: None) if planner == "rrt-connect" else rrt_connect_retract(ck, start, goal, cfg, seed)
    import platform

    import mujoco

    out = {"seed": seed, "planner": planner, "solved": path is not None, "plan_s": round(time.time() - t0, 1),
           "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                        "mujoco": mujoco.__version__, "where": "local"},
           "budget_s": budget}
    if path is None:
        return out
    rng = np.random.default_rng(seed)
    raw = truncate_at_free(ck, _G["dirs"], path, w)
    sm = smooth(ck, raw, w, rng)
    sm = truncate_at_free(ck, _G["dirs"], sm, w)
    bad = sum(not ck.segment_valid(a, b, w, 0.001) for a, b in zip(sm[:-1], sm[1:]))
    out.update(raw=stats(raw, w), shortcut=stats(sm, w), invalid_segments_at_1mm=bad, total_s=round(time.time() - t0, 1),
               path=[p.tolist() for p in sm])
    return out


def _resample(path, w, n=60):
    from smooth_oracle import densify

    d = np.asarray(densify([np.asarray(p) for p in path], w, 0.01))
    seg = np.array([pose_metric_np(a, b, w) for a, b in zip(d[:-1], d[1:])])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    idx = np.searchsorted(s, np.linspace(0, s[-1], n)).clip(0, len(d) - 1)
    return d[idx]


def corridors(paths, w, threshold=0.5):
    """Cluster paths: two paths share a corridor if their arc-length-resampled poses are on average within
    `threshold` (SE(3) metric) of each other; single linkage."""
    R = [_resample(p, w) for p in paths]
    n = len(R)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.mean([pose_metric_np(a, b, w) for a, b in zip(R[i], R[j])]))
    label = list(range(n))
    for i in range(n):
        for j in range(n):
            if D[i, j] < threshold and label[i] != label[j]:
                old, new = label[j], label[i]
                label = [new if x == old else x for x in label]
    return label, D


def summarise(rows, w, out: Path, reference_paths: dict | None = None) -> dict:
    """Path-length distribution, corridor clusters (optionally against named reference paths), shortest path."""
    from smooth_oracle import stats

    solved = [r for r in rows if r.get("solved") and r.get("invalid_segments_at_1mm") == 0]
    summary = {"seeds": len(rows), "solved": sum(bool(r.get("solved")) for r in rows), "valid_after_shortcut": len(solved),
               "platforms": sorted({json.dumps(r.get("platform"), sort_keys=True) for r in rows})}
    if not solved:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary
    L = np.array([r["shortcut"]["L"] for r in solved])
    summary["L"] = {"min": float(L.min()), "median": float(np.median(L)), "max": float(L.max()), "sorted": sorted(L.round(3).tolist())}
    refs = reference_paths or {}
    paths = [np.asarray(r["path"]) for r in solved] + [np.asarray(p) for p in refs.values()]
    label, _ = corridors(paths, w)
    names = [f"seed {r['seed']}" for r in solved] + list(refs)
    groups: dict = {}
    for nm, lb, p in zip(names, label, paths):
        groups.setdefault(lb, []).append((nm, stats(list(p), w)["L"]))
    summary["corridors"] = [{"members": [m for m, _ in g], "L_range": [min(x for _, x in g), max(x for _, x in g)]}
                            for g in sorted(groups.values(), key=lambda g: min(x for _, x in g))]
    best = solved[int(L.argmin())]
    np.save(out / "best_path.npy", np.asarray(best["path"]))
    summary["best"] = {k: best.get(k) for k in ("seed", "planner", "plan_s", "raw", "shortcut", "platform")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=1800.0, help="RRT-Connect seconds per seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "reference-planning")
    ap.add_argument("--planner", choices=["rrt-connect", "rrt-connect-retract"], default="rrt-connect")
    a = ap.parse_args()
    from loophole.config import load_config
    from loophole.sim.env import LoopholeEnv

    env = LoopholeEnv(NAME, config=load_config(ROOT / "loophole/configs/alpha_training.yaml"))
    start, w = env.target_pose().copy(), float(env.roadmap.w)
    lo, hi = env.frame_aabb()
    lo, hi = lo - 1.0, hi + 1.0
    env.close()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.seed0 + k, a.budget, start.tolist(), lo.tolist(), hi.tolist(), w, a.planner) for k in range(a.seeds)]
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init) as pool, (a.out / "seeds.jsonl").open("w") as f:
        for r in pool.imap_unordered(_solve, jobs):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            (a.out / f"seed-{r['seed']}.json").write_text(json.dumps(r))
            s = r.get("shortcut") or {}
            print(f"[{len(rows)}/{a.seeds}] seed {r['seed']} solved={r['solved']} plan {r['plan_s']}s "
                  f"L={s.get('L')} rot={s.get('rotation_rad')} ({time.time() - t0:.0f}s)", flush=True)
    summary = summarise(rows, w, a.out)
    summary.update(budget_s=a.budget, planner=a.planner, wall_s=round(time.time() - t0, 1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
