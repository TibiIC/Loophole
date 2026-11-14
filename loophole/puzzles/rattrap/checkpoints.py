"""Rat Trap progress: ordered checkpoint matching against the reference solve.

Checkpoints c_1..c_K are cord states of the recorded solve, sampled at its grab / release boundaries (and its final,
free state), each a full set of cord node positions (the mouse is welded at a node, so the nodes carry it too).
A checkpoint that lies within `min_separation` of the previous kept one (or of the start) is dropped, so neighbouring
checkpoints are distinguishable and the start cannot match c_1.

    reached:   checkpoint k is reached at time t if  max_i |x_i(t) - c_k,i| < eps  (max over nodes, not the mean: it
               cannot be bought by getting most of the cord roughly right) and checkpoints 1..k-1 were each reached at
               strictly earlier times. One evaluation advances at most one checkpoint, so none can be skipped.
    progress:  p = k_reached / K   (monotone: reached checkpoints stay reached)
    potential: Phi = -[(K - k) + min(1, d(x, c_{k+1}) / D_{k+1})],   D_{k+1} = d(c_k, c_{k+1})  (c_0 = the start),
               Phi = 0 once k = K. Continuous across a checkpoint: just before reaching c_{k+1}, Phi ~ -(K - k);
               just after, the distance to c_{k+2} is ~ D_{k+2}, so Phi ~ -(K - k - 1) - 1.

eps is calibrated by running the reference solve, as discrete agent commands, through this matcher: the smallest
eps at which every checkpoint registers, times a safety factor (scripts/build_rattrap_reference.py).

Why staged geometric milestones and not a topological signature as the reward: in the rope / cable literature,
topological signatures (h-signatures, grasp-loop signatures) guide planning, while cable-routing RL is driven by
staged decompositions with simple geometric success tests. See docs/RATTRAP_DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "reference" / "rattrap-checkpoints.npz"


def max_dev(a: np.ndarray, b: np.ndarray) -> float:
    """max over nodes of |a_i - b_i| (both (n, 3))."""
    return float(np.max(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=1)))


def select_checkpoints(start: np.ndarray, candidates: list[np.ndarray], min_separation: float) -> list[int]:
    """Indices of the candidates kept: each at least `min_separation` from the previously kept one (or the start).
    The last candidate (the solve's final state) is always kept: if it is too close to the last kept one, it replaces it."""
    kept, prev = [], start
    for i, c in enumerate(candidates):
        if max_dev(c, prev) >= min_separation:
            kept.append(i)
            prev = c
    last = len(candidates) - 1
    if last >= 0 and (not kept or kept[-1] != last):
        if kept and max_dev(candidates[last], candidates[kept[-1]]) < min_separation:
            kept[-1] = last
        elif max_dev(candidates[last], start) >= min_separation:
            kept.append(last)
    return kept


def calibrate_eps(trajectory: list[np.ndarray], checkpoints: list[np.ndarray]) -> float:
    """The smallest eps at which a greedy, ordered, one-per-step matcher registers every checkpoint along `trajectory`
    (states at the evaluation times). Exact up to the bisection tolerance: greedy earliest matching is optimal for
    ordered matching, and feasibility is monotone in eps."""
    D = np.array([[max_dev(x, c) for c in checkpoints] for x in trajectory])  # (T, K)

    def ok(eps: float) -> bool:
        k = 0
        for t in range(len(trajectory)):
            if k < len(checkpoints) and D[t, k] < eps:
                k += 1
        return k == len(checkpoints)

    lo, hi = 0.0, float(D.max()) + 1e-9
    if not ok(hi):
        return float("inf")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


@dataclass
class Checkpoints:
    start: np.ndarray
    states: np.ndarray  # (K, n, 3)
    eps: float
    meta: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.states)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_FILE) -> "Checkpoints":
        import json

        z = np.load(path, allow_pickle=False)
        return cls(start=z["start"], states=z["states"], eps=float(z["eps"]), meta=json.loads(str(z["meta"])))

    def save(self, path: Path | str = DEFAULT_FILE) -> None:
        import json

        np.savez_compressed(path, start=self.start, states=self.states, eps=np.float64(self.eps), meta=json.dumps(self.meta))

    def segment_lengths(self) -> np.ndarray:
        prev = [self.start] + list(self.states[:-1])
        return np.array([max(max_dev(a, b), 1e-9) for a, b in zip(prev, self.states)])


class CheckpointTracker:
    """Ordered matching during one episode. `update(nodes)` after every action; `k` = checkpoints reached."""

    def __init__(self, cps: Checkpoints):
        self.cps = cps
        self.k = 0
        self.D = cps.segment_lengths()
        self.reached_at: list[int] = []
        self.t = 0

    def update(self, nodes: np.ndarray) -> int:
        self.t += 1
        if self.k < self.cps.K and max_dev(nodes, self.cps.states[self.k]) < self.cps.eps:
            self.k += 1
            self.reached_at.append(self.t)
        return self.k

    def progress(self) -> float:
        return self.k / self.cps.K if self.cps.K else 0.0

    def distance_to_next(self, nodes: np.ndarray) -> float | None:
        """Normalised distance to the next checkpoint, min(1, d / D); None once all are reached."""
        if self.k >= self.cps.K:
            return None
        return min(1.0, max_dev(nodes, self.cps.states[self.k]) / float(self.D[self.k]))

    def potential(self, nodes: np.ndarray) -> float:
        dn = self.distance_to_next(nodes)
        return 0.0 if dn is None else -float((self.cps.K - self.k) + dn)
