"""Cost-to-go roadmap: an approximate navigation function on C-free.

The state that matters is the target's pose relative to the fixed piece, q in
SE(3), with the metric d(q1, q2) = ||t1 - t2|| + w * angle(R1^T R2) (w = the
target's characteristic length, so d bounds the motion of every point of it).

Offline (`build_roadmap`): a PRM over C-free built with the shared collision
checker. Nodes are the densified reference path, heavy Gaussian sampling in a
tube around it (uniform sampling misses the narrow passage) and uniform samples
for coverage; edges join k nearest neighbours and are validated densely. Every
node inside the goal set (separated) is a goal node; multi-source Dijkstra from
the goal set gives each node its cost-to-go. Saved to roadmap-<variant>.npz.

Online (`Roadmap.cost_to_go`): ctg(q) = min over the k nearest nodes n that are
*visible* from q (straight SE(3) segment collision-free) of d(q, n) + ctg(n).
The visibility test is what stops "the nearest node is through the wall".
If no candidate is visible, None is returned and the caller keeps its previous
value (and logs the fallback).
"""

from __future__ import annotations

import hashlib
import heapq
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.collision import CollisionChecker, interpolate, pose_metric
from ..sim.spec import Puzzle, load_puzzle, resolve_puzzle


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Roadmap:
    nodes: np.ndarray  # (N, 7)
    cost: np.ndarray  # (N,) cost-to-go of each node
    edges: np.ndarray  # (E, 2)
    w: float
    resolution: float
    k_query: int = 10
    path: str | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Roadmap":
        z = np.load(path, allow_pickle=False)
        return cls(
            nodes=z["nodes"], cost=z["cost"], edges=z["edges"], w=float(z["w"]), resolution=float(z["resolution"]),
            k_query=int(z["k_query"]), path=str(path), sha256=file_sha256(path),
        )

    @classmethod
    def for_puzzle(cls, puzzle: Puzzle) -> "Roadmap | None":
        if not puzzle.roadmap or not puzzle.source_path:
            return None
        p = Path(puzzle.source_path).parent / puzzle.roadmap
        return cls.load(p) if p.exists() else None

    def cost_to_go(self, q: np.ndarray, ck: CollisionChecker) -> tuple[float | None, int | None]:
        """(cost-to-go, node index) through the best *visible* nearby node, or (None, None)."""
        q = np.asarray(q, float)
        d = pose_metric(q, self.nodes, self.w)
        k = min(self.k_query, len(d))
        near = np.argpartition(d, k - 1)[:k]
        order = near[np.argsort(d[near] + self.cost[near])]
        for n in order:
            if ck.segment_valid(q, self.nodes[n], self.w, self.resolution):
                return float(d[n] + self.cost[n]), int(n)
        return None, None


# --------------------------------------------------------------------- building
def _knn(nodes: np.ndarray, w: float, k: int) -> list[np.ndarray]:
    out = []
    for i in range(len(nodes)):
        d = pose_metric(nodes[i], nodes, w)
        d[i] = np.inf
        out.append(np.argpartition(d, k)[:k])
    return out


def _dijkstra(n: int, adj: list[list[tuple[int, float]]], sources: np.ndarray) -> np.ndarray:
    dist = np.full(n, np.inf)
    heap = []
    for s in sources:
        dist[s] = 0.0
        heap.append((0.0, int(s)))
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, wgt in adj[u]:
            nd = d + wgt
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _rotate_quat(q: np.ndarray, rv: np.ndarray) -> np.ndarray:
    from ..sim.geometry import quat_from_rotvec, quat_mul, quat_normalize

    return quat_normalize(quat_mul(quat_from_rotvec(rv), q))


def build_roadmap(
    variant: str,
    seed: int = 1,
    n_tube: int = 5000,
    n_uniform: int = 2500,
    n_goal: int = 300,
    k: int = 12,
    sigma_t: float = 0.015,
    sigma_r: float = 0.15,
    log=print,
) -> dict:
    from ..planning.se3 import random_quat

    t0 = time.time()
    puzzle = load_puzzle(resolve_puzzle(variant))
    ck = CollisionChecker(puzzle)
    here = Path(puzzle.source_path).parent
    ref = np.load(here / f"{variant}-reference.npz")
    w = float(ref["w"])
    c = float(puzzle.meta["nominal_gap_clearance"])
    resolution = c / 2
    rng = np.random.default_rng(seed)

    # 1. densified reference path (consecutive nodes are connected: validated densely already)
    dense = ref["dense"]
    keep = [0]
    for i in range(1, len(dense)):
        if pose_metric(dense[keep[-1]], dense[i], w) >= 0.01:
            keep.append(i)
    keep.append(len(dense) - 1)
    ref_nodes = dense[sorted(set(keep))]
    nodes = [x for x in ref_nodes]
    ref_edges = [(i, i + 1) for i in range(len(ref_nodes) - 1)]

    # 2. Gaussian tube around the reference path
    tries = 0
    while len(nodes) < len(ref_nodes) + n_tube and tries < 40 * n_tube:
        tries += 1
        base = dense[rng.integers(len(dense))]
        s = rng.choice([0.25, 1.0, 3.0])  # mixture of tube widths
        x = np.concatenate([base[:3] + rng.normal(0, sigma_t * s, 3), _rotate_quat(base[3:], rng.normal(0, sigma_r * s, 3))])
        if ck.is_valid(x):
            nodes.append(x)
    n_t = len(nodes)

    # 3. uniform coverage + 4. explicit goal-set samples around the separated end of the path
    lo, hi = np.full(3, -0.6), np.full(3, 0.6)
    tries = 0
    while len(nodes) < n_t + n_uniform and tries < 40 * n_uniform:
        tries += 1
        x = np.concatenate([lo + rng.random(3) * (hi - lo), random_quat(rng)])
        if ck.is_valid(x):
            nodes.append(x)
    goal_c = dense[-1]
    tries = 0
    n_before = len(nodes)
    while len(nodes) < n_before + n_goal and tries < 40 * n_goal:
        tries += 1
        x = np.concatenate([goal_c[:3] + rng.normal(0, 0.1, 3), random_quat(rng)])
        if ck.is_valid(x) and ck.is_separated(x):
            nodes.append(x)
    nodes = np.array(nodes)
    N = len(nodes)
    log(f"[{variant}] roadmap nodes: ref={len(ref_nodes)} tube={n_t - len(ref_nodes)} uniform+goal={N - n_t} ({time.time() - t0:.0f}s)")

    # edges: k nearest neighbours, validated densely at margin 0
    edges = set(ref_edges)
    nbrs = _knn(nodes, w, k)
    cand = {(min(i, int(j)), max(i, int(j))) for i in range(N) for j in nbrs[i]} - edges
    for i, j in sorted(cand):
        if ck.segment_valid(nodes[i], nodes[j], w, resolution):
            edges.add((i, j))
    edges = np.array(sorted(edges))
    log(f"[{variant}] roadmap edges: {len(edges)} valid of {len(cand) + len(ref_edges)} candidates ({time.time() - t0:.0f}s)")

    goal = np.array([i for i in range(N) if ck.is_separated(nodes[i])])
    adj: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    for i, j in edges:
        wgt = float(pose_metric(nodes[i], nodes[j], w))
        adj[i].append((int(j), wgt))
        adj[j].append((int(i), wgt))
    cost = _dijkstra(N, adj, goal)
    reach = np.isfinite(cost)
    # keep only nodes connected to the goal set (others have no defined cost-to-go)
    remap = -np.ones(N, int)
    remap[reach] = np.arange(reach.sum())
    e2 = remap[edges]
    e2 = e2[(e2 >= 0).all(axis=1)]
    out = here / f"roadmap-{variant}.npz"
    np.savez_compressed(
        out, nodes=nodes[reach], cost=cost[reach], edges=e2, w=w, resolution=resolution, k_query=10,
        puzzle_sha256=puzzle.source_sha256 or "", seed=seed,
    )
    rm = Roadmap.load(out)
    start = np.asarray(ref["path"][0])
    ctg0, _ = rm.cost_to_go(start, ck)
    info = {
        "roadmap_nodes": int(reach.sum()),
        "roadmap_edges": int(len(e2)),
        "roadmap_goal_nodes": int(len(goal)),
        "roadmap_unreachable_dropped": int((~reach).sum()),
        "start_cost_to_go": None if ctg0 is None else round(ctg0, 4),
        "roadmap_build_s": round(time.time() - t0, 1),
    }
    log(f"[{variant}] {info}")
    return info


__all__ = ["Roadmap", "build_roadmap", "interpolate"]
