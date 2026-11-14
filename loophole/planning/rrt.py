"""RRT-Connect over SE(3) (Kuffner & LaValle 2000), numpy, using the shared collision checker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..sim.collision import CollisionChecker
from .se3 import pose_metric, random_quat, steer


@dataclass
class Tree:
    nodes: list[np.ndarray] = field(default_factory=list)
    parent: list[int] = field(default_factory=list)
    _arr: np.ndarray | None = None

    def add(self, x: np.ndarray, parent: int) -> int:
        self.nodes.append(x)
        self.parent.append(parent)
        self._arr = None
        return len(self.nodes) - 1

    def nearest(self, x: np.ndarray, w: float) -> int:
        if self._arr is None or len(self._arr) != len(self.nodes):
            self._arr = np.array(self.nodes)
        return int(np.argmin(pose_metric(x, self._arr, w)))

    def path_to_root(self, i: int) -> list[np.ndarray]:
        out = []
        while i != -1:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out


@dataclass
class PlannerConfig:
    w: float  # rotation weight (characteristic length)
    step: float  # max metric length of a tree edge
    resolution: float  # collision-check resolution along edges (metric)
    margin: float  # planning margin (inflation) for robustness
    lo: np.ndarray  # translation sampling box
    hi: np.ndarray
    max_time: float = 600.0
    seed: int = 0
    focus: np.ndarray | None = None  # optional Gaussian focus points for translation sampling
    focus_sigma: float = 0.03
    focus_prob: float = 0.0
    focus_quat: np.ndarray | None = None  # if set, focused samples also stay near this orientation
    focus_rot_sigma: float = 0.5  # rad


def rrt_connect(ck: CollisionChecker, start: np.ndarray, goal: np.ndarray, cfg: PlannerConfig, log=print) -> list[np.ndarray] | None:
    rng = np.random.default_rng(cfg.seed)
    valid = lambda x: ck.is_valid(x, cfg.margin)
    seg_ok = lambda a, b: ck.segment_valid(a, b, cfg.w, cfg.resolution, cfg.margin)
    if not valid(start) or not valid(goal):
        raise ValueError("start or goal invalid at the planning margin")
    ta, tb = Tree(), Tree()
    ta.add(start, -1)
    tb.add(goal, -1)
    t0 = time.time()
    it = 0

    def sample():
        if cfg.focus is not None and rng.random() < cfg.focus_prob:
            c = cfg.focus[rng.integers(len(cfg.focus))]
            t = c + rng.normal(0, cfg.focus_sigma, 3)
            if cfg.focus_quat is not None:
                from ..sim.geometry import quat_from_rotvec, quat_mul, quat_normalize

                return np.concatenate([t, quat_normalize(quat_mul(quat_from_rotvec(rng.normal(0, cfg.focus_rot_sigma, 3)), cfg.focus_quat))])
        else:
            t = cfg.lo + rng.random(3) * (cfg.hi - cfg.lo)
        return np.concatenate([t, random_quat(rng)])

    def extend(tree: Tree, x: np.ndarray):
        i = tree.nearest(x, cfg.w)
        y = steer(tree.nodes[i], x, cfg.w, cfg.step)
        if valid(y) and seg_ok(tree.nodes[i], y):
            j = tree.add(y, i)
            return ("reached" if np.allclose(y, x) else "advanced"), j
        return "trapped", -1

    def connect(tree: Tree, x: np.ndarray):
        while True:
            s, j = extend(tree, x)
            if s != "advanced":
                return s, j

    a_is_start = True
    while time.time() - t0 < cfg.max_time:
        it += 1
        x = sample()
        s, j = extend(ta, x)
        if s != "trapped":
            s2, k = connect(tb, ta.nodes[j])
            if s2 == "reached":
                pa, pb = ta.path_to_root(j)[::-1], tb.path_to_root(k)
                path = pa + pb[1:]
                if not a_is_start:
                    path = path[::-1]
                log(f"rrt-connect: solved after {it} iterations, {len(ta.nodes) + len(tb.nodes)} nodes, {time.time() - t0:.1f}s")
                return path
        ta, tb = tb, ta
        a_is_start = not a_is_start
        if it % 2000 == 0:
            log(f"  it={it} nodes={len(ta.nodes)}+{len(tb.nodes)} checks={ck.calls} t={time.time() - t0:.0f}s")
    log(f"rrt-connect: no solution within {cfg.max_time}s ({it} iterations)")
    return None


def shortcut(ck: CollisionChecker, path: list[np.ndarray], w: float, resolution: float, margin: float, iters: int = 400, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    path = list(path)
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if ck.segment_valid(path[i], path[j], w, resolution, margin):
            path = path[: i + 1] + path[j:]
    return path


def path_length(path: list[np.ndarray], w: float) -> float:
    return float(sum(pose_metric(a, b, w) for a, b in zip(path[:-1], path[1:])))
