#!/usr/bin/env python3
"""Training roadmap for Alpha 1.0: the evaluation roadmap + interlocked configurations of both ties.

    .venv/bin/python scripts/build_training_roadmap.py [--n-inter 6000] [--workers 6]

The evaluation roadmap (roadmap-alpha-1.0-paper.npz, a tube around the published solution) is
left untouched: it defines the task's partial credit. This writes a separate
roadmap-alpha-1.0-paper-train.npz for training configs (reward.roadmap_file), adding:
  * interlocked nodes from loophole.training.starts (random loop-around-tube placements, both
    ties: linking number -1 = the canonical tie, +1 = the flipped tie), collision-free;
  * local Gaussian expansions around them (to bridge narrow regions);
  * free, unlinked poses near tube A (the outer region that leads to separation);
then validates k-nearest-neighbour edges between new and all nodes (4 mm resolution, margin 0,
parallel workers) and recomputes cost-to-go by Dijkstra from the separated nodes.

Connectivity is reported per tie: the fraction of interlocked nodes with a finite cost-to-go.
A flipped-tie node connected to the separated set proves (up to the 4 mm edge resolution)
that the flipped tie can be solved; the graph contains the path.
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

from loophole.config import load_config  # noqa: E402
from loophole.puzzles import paper_alpha as pa  # noqa: E402
from loophole.reward.roadmap import Roadmap, _dijkstra, _knn  # noqa: E402
from loophole.sim.collision import CollisionChecker, pose_metric  # noqa: E402
from loophole.sim.env import LoopholeEnv  # noqa: E402
from loophole.sim.spec import load_puzzle, resolve_puzzle  # noqa: E402
from loophole.training.starts import StartSampler  # noqa: E402

NAME = "alpha-1.0-paper"
_CK = None


def _init():
    global _CK
    _CK = CollisionChecker(load_puzzle(resolve_puzzle(NAME)))


def _check(job):
    i, j, a, b, w, res = job
    return i, j, _CK.segment_valid(np.asarray(a), np.asarray(b), w, res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inter", type=int, default=6000)
    ap.add_argument("--n-expand", type=int, default=3000)
    ap.add_argument("--n-outer", type=int, default=1500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-edge", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    env = LoopholeEnv(NAME, config=load_config(ROOT / "loophole" / "configs" / "alpha_training.yaml"))
    base = env.roadmap
    ck = env.checker
    w, res = float(base.w), float(base.resolution)
    s = StartSampler(env, env.target_pose().copy(), mode="interlocked")
    rng = np.random.default_rng(a.seed)

    new, tie = [], []
    while len(new) < a.n_inter:  # interlocked, both ties (no escapability / ctg requirement for nodes)
        q = s.interlocked_candidate(rng, "both")
        if ck.is_valid(q):
            lk = s.linking(q)
            if abs(abs(lk) - 1) < 0.1:
                new.append(q)
                tie.append(s.chirality_of(lk))
    print(f"interlocked nodes: {len(new)} ({tie.count('canonical')} canonical tie, {tie.count('flipped')} flipped) {time.time() - t0:.0f}s", flush=True)
    from loophole.reward.roadmap import _rotate_quat

    seeds = list(new)
    while len(new) < a.n_inter + a.n_expand:  # local expansions (narrow regions)
        b = seeds[rng.integers(len(seeds))]
        sc = rng.choice([0.3, 1.0])
        q = np.concatenate([b[:3] + rng.normal(0, 0.05 * sc, 3), _rotate_quat(b[3:], rng.normal(0, 0.25 * sc, 3))])
        if ck.is_valid(q):
            lk = s.linking(q)
            new.append(q)
            tie.append(s.chirality_of(lk) if abs(abs(lk) - 1) < 0.1 else "unlinked")
    lo = env.frame_aabb()[0] - 1.0
    hi = env.frame_aabb()[1] + 1.0
    from loophole.planning.se3 import random_quat

    while len(new) < a.n_inter + a.n_expand + a.n_outer:  # free poses around tube A
        q = np.concatenate([lo + rng.random(3) * (hi - lo), random_quat(rng)])
        if ck.is_valid(q) and abs(s.linking(q)) < 0.1:
            new.append(q)
            tie.append("unlinked")
    new = np.array(new)
    print(f"new nodes: {len(new)} {time.time() - t0:.0f}s", flush=True)

    old_nodes = np.asarray(base.nodes)
    nodes = np.vstack([old_nodes, new])
    n_old, N = len(old_nodes), len(old_nodes) + len(new)
    old_edges = {(int(i), int(j)) for i, j in np.asarray(np.load(base.path)["edges"])}
    nbrs = _knn(nodes, w, a.k)
    cand = {(min(i, int(j)), max(i, int(j))) for i in range(n_old, N) for j in nbrs[i]}
    cand |= {(min(i, int(j)), max(i, int(j))) for i in range(n_old) for j in nbrs[i] if int(j) >= n_old}
    cand = sorted((i, j) for i, j in cand - old_edges if float(pose_metric(nodes[i], nodes[j][None], w)[0]) <= a.max_edge)
    print(f"candidate edges: {len(cand)} {time.time() - t0:.0f}s", flush=True)
    edges = set(old_edges)
    jobs = [(i, j, nodes[i].tolist(), nodes[j].tolist(), w, res) for i, j in cand]
    with Pool(a.workers, initializer=_init) as pool:
        for n, (i, j, ok) in enumerate(pool.imap_unordered(_check, jobs, chunksize=64)):
            if ok:
                edges.add((i, j))
            if n % 10000 == 0:
                print(f"  edges checked {n}/{len(jobs)} valid {len(edges) - len(old_edges)} ({time.time() - t0:.0f}s)", flush=True)
    edges = np.array(sorted(edges))
    goal = np.array([i for i in range(N) if ck.is_separated(nodes[i])])
    adj = [[] for _ in range(N)]
    for i, j in edges:
        wgt = float(pose_metric(nodes[i], nodes[j][None], w)[0])
        adj[i].append((int(j), wgt))
        adj[j].append((int(i), wgt))
    cost = _dijkstra(N, adj, goal)
    reach = np.isfinite(cost)
    tie_all = np.array(["eval"] * n_old + tie)
    report = {"nodes": N, "edges": int(len(edges)), "new_edges": int(len(edges) - len(old_edges)), "goal_nodes": int(len(goal)),
              "seconds": round(time.time() - t0, 1)}
    for t in ("canonical", "flipped", "unlinked"):
        m = tie_all == t
        report[f"{t}_nodes"] = int(m.sum())
        report[f"{t}_connected_to_goal"] = int((m & reach).sum())
    remap = -np.ones(N, int)
    remap[reach] = np.arange(reach.sum())
    e2 = remap[edges]
    e2 = e2[(e2 >= 0).all(axis=1)]
    out = pa.PUZZLE_OUT / f"roadmap-{NAME}-train.npz"
    np.savez_compressed(out, nodes=nodes[reach], cost=cost[reach], edges=e2, w=w, resolution=res, k_query=10,
                        puzzle_sha256=env.puzzle.source_sha256 or "", seed=a.seed, tie=tie_all[reach])
    rm = Roadmap.load(out)
    report["canonical_start_ctg"] = rm.cost_to_go(env.target_pose(), ck)[0]
    report["flipped_start_ctg"] = rm.cost_to_go(s.flipped_start(), ck)[0]
    (pa.PUZZLE_OUT / f"{NAME}-train-roadmap.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
