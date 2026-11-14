"""Certified start bank: interlocked training starts, each with a found collision-free way out.

A start is *certified* when a planner connects it to the evaluation roadmap (whose nodes all
have a finite cost-to-go to the separated set). The connecting path, shortcut and densified,
becomes a corridor of nodes and edges in a separate **training roadmap**, so cost-to-go
shaping and partial credit work from every bank start. The evaluation roadmap and the
task's scoring are never modified.

Planner: single-tree RRT from the start with goal bias toward roadmap nodes; after every new
tree node, the roadmap's own visibility test (a collision-free straight segment to one of the
10 nearest roadmap nodes) decides whether the tree has reached the roadmap. Edges are checked
at the roadmap's resolution (4 mm of motion of any point), margin 0. A failure within the time
budget does not prove a start unsolvable; it is recorded as uncertified and not used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..planning.se3 import random_quat, steer
from ..reward.roadmap import Roadmap, _dijkstra
from ..sim.collision import interpolate, pose_metric

BANK_FILE = "start-bank-{name}.npz"
TRAIN_ROADMAP_FILE = "roadmap-{name}-train.npz"


@dataclass
class ConnectResult:
    path: list[np.ndarray] | None  # start ... last tree node, then the visible roadmap node
    node: int | None  # roadmap node reached
    seconds: float
    tree_size: int


def connect_to_roadmap(ck, rm: Roadmap, start: np.ndarray, rng: np.random.Generator, lo, hi, time_budget: float = 90.0,
                       step: float = 0.15, goal_bias: float = 0.3, bias_sigma_t: float = 0.1, bias_sigma_r: float = 0.3) -> ConnectResult:
    from ..reward.roadmap import _rotate_quat

    t0 = time.time()
    w, res = float(rm.w), float(rm.resolution)
    nodes, parent = [np.asarray(start, float)], [-1]
    c, n = rm.cost_to_go(nodes[0], ck)
    if c is not None:
        return ConnectResult([nodes[0], np.asarray(rm.nodes[n])], n, time.time() - t0, 1)
    R = np.asarray(rm.nodes)
    while time.time() - t0 < time_budget:
        if rng.random() < goal_bias:
            b = R[rng.integers(len(R))]
            x = np.concatenate([b[:3] + rng.normal(0, bias_sigma_t, 3), _rotate_quat(b[3:], rng.normal(0, bias_sigma_r, 3))])
        else:
            x = np.concatenate([lo + rng.random(3) * (hi - lo), random_quat(rng)])
        d = pose_metric(x, np.asarray(nodes), w)
        i = int(np.argmin(d))
        y = steer(nodes[i], x, w, step)
        if not ck.is_valid(y) or not ck.segment_valid(nodes[i], y, w, res):
            continue
        nodes.append(y)
        parent.append(i)
        c, n = rm.cost_to_go(y, ck)
        if c is not None:
            path, j = [], len(nodes) - 1
            while j >= 0:
                path.append(nodes[j])
                j = parent[j]
            return ConnectResult(path[::-1] + [np.asarray(rm.nodes[n])], n, time.time() - t0, len(nodes))
    return ConnectResult(None, None, time.time() - t0, len(nodes))


def shortcut_path(ck, path: list[np.ndarray], w: float, res: float, rng: np.random.Generator, iters: int = 200) -> list[np.ndarray]:
    path = list(path)
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i >= 2 and ck.segment_valid(path[i], path[j], w, res):
            path = path[: i + 1] + path[j:]
    return path


def densify(path: list[np.ndarray], w: float, spacing: float = 0.05) -> np.ndarray:
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        m = max(1, int(np.ceil(float(pose_metric(a, b[None], w)[0]) / spacing)))
        out += [interpolate(a, b, k / m) for k in range(1, m + 1)]
    return np.array(out)


def roadmap_route(rm: Roadmap, node: int) -> np.ndarray:
    """Shortest roadmap route from `node` to the separated set (steepest descent on cost-to-go)."""
    edges = np.asarray(np.load(rm.path)["edges"])
    nbr: dict[int, list[int]] = {}
    for i, j in edges:
        nbr.setdefault(int(i), []).append(int(j))
        nbr.setdefault(int(j), []).append(int(i))
    cost, nodes = np.asarray(rm.cost), np.asarray(rm.nodes)
    route, i = [int(node)], int(node)
    while cost[i] > 0:
        i = min(nbr[i], key=lambda j: cost[j] + float(pose_metric(nodes[i], nodes[j][None], float(rm.w))[0]))
        route.append(i)
    return nodes[route]


def merge_corridors(rm: Roadmap, corridors: list[np.ndarray], targets: list[int], out: Path, **meta) -> Roadmap:
    """Training roadmap = evaluation roadmap + corridors (each corridor's last node is a roadmap node)."""
    base = np.load(rm.path)
    nodes = [np.asarray(base["nodes"])]
    edges = [np.asarray(base["edges"])]
    w = float(rm.w)
    n = len(nodes[0])
    for cor, tgt in zip(corridors, targets):
        cor = cor[:-1]  # the last pose is the roadmap node itself
        k = len(cor)
        nodes.append(cor)
        e = [(n + i, n + i + 1) for i in range(k - 1)] + [(n + k - 1, int(tgt))]
        edges.append(np.array(e))
        n += k
    nodes = np.vstack(nodes)
    edges = np.vstack(edges)
    N = len(nodes)
    goal = np.flatnonzero(np.asarray(base["cost"]) == 0.0)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    for i, j in edges:
        wgt = float(pose_metric(nodes[i], nodes[j][None], w)[0])
        adj[i].append((int(j), wgt))
        adj[j].append((int(i), wgt))
    cost = _dijkstra(N, adj, goal)
    np.savez_compressed(out, nodes=nodes, cost=cost, edges=edges, w=w, resolution=float(rm.resolution), k_query=10,
                        puzzle_sha256=str(base["puzzle_sha256"]), seed=int(base["seed"]), **meta)
    return Roadmap.load(out)


class StartBank:
    """Certified starts (pose, tie, linking number, cost-to-go on the training roadmap)."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.path = Path(path)
        self.starts = z["starts"]
        self.tie = z["tie"]
        self.ctg = z["ctg"]
        self.linking = z["linking"]
        self.flipped_path = z["flipped_path"] if "flipped_path" in z.files and len(z["flipped_path"]) else None

    def indices(self, tie: str = "both") -> np.ndarray:
        return np.arange(len(self.starts)) if tie == "both" else np.flatnonzero(self.tie == tie)

    def sample(self, rng: np.random.Generator, tie: str = "both") -> tuple[int, np.ndarray]:
        idx = self.indices(tie)
        if idx.size == 0:
            raise ValueError(f"start bank has no certified {tie}-tie starts")
        i = int(rng.choice(idx))
        return i, np.asarray(self.starts[i], float)
