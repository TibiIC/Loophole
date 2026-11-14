"""Training start states for the Alpha puzzle (the canonical start is the held-out test).

Modes
-----
* ``perturb``: the canonical start q0 plus Gaussian noise (translation sigma_t, rotation
  sigma_r as a random rotation vector), accepted only if the pose is collision-free, still
  interlocked (no free escape by translation) and its roadmap cost-to-go is at least
  ``min_ctg_frac`` of the canonical start's.
* ``curriculum`` (reverse curriculum): a pose at a chosen distance ``d`` in [0, 1] from the
  goal, sampled along the oracle path (``source="oracle"``: d = path length still to go to
  the first free pose on the path / that pose's path length from the start, small jitter) or from roadmap nodes (``source="roadmap"``: cost-to-go within
  ``band`` of d * c0); also required to be collision-free and not yet free.

In every mode, poses within ``exclude_radius`` of q0 (in the SE(3) metric
||dt|| + w * angle) are rejected, so the canonical start is never sampled in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.collision import interpolate, pose_metric
from ..sim.geometry import quat_from_rotvec, quat_mul, quat_normalize


class StartSamplingError(RuntimeError):
    pass


@dataclass
class StartSampler:
    env: object  # LoopholeEnv with a checker and a roadmap
    q0: np.ndarray  # canonical start pose (x y z qw qx qy qz)
    mode: str = "perturb"
    sigma_t: float = 0.05  # m
    sigma_r: float = 0.10  # rad
    min_ctg_frac: float = 0.5  # perturbed starts: ctg >= this * c0 (still far from free)
    exclude_radius: float = 0.05  # never within this SE(3) distance of the canonical start
    curriculum_distance: float = 0.5  # 0 = at the goal, 1 = as far as the canonical start
    curriculum_source: str = "oracle"
    band: float = 0.05  # roadmap curriculum: |ctg / c0 - d| <= band
    jitter_t: float = 0.01
    jitter_r: float = 0.02
    max_tries: int = 500
    _path: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.q0 = np.asarray(self.q0, float)
        ck, rm = self.env.checker, self.env.roadmap
        if ck is None or rm is None:
            raise ValueError("start sampling needs a separation puzzle with the roadmap enabled (reward.roadmap: true)")
        self.w = float(rm.w)
        self.c0, _ = rm.cost_to_go(self.q0, ck)
        if not self.c0:
            raise ValueError("canonical start has no visible roadmap node")
        if self.mode not in ("perturb", "curriculum"):
            raise ValueError("mode must be 'perturb' or 'curriculum' (the canonical start is never a training start)")

    # ------------------------------------------------------------------ checks
    def ctg(self, q) -> float | None:
        return self.env.roadmap.cost_to_go(q, self.env.checker)[0]

    def acceptable(self, q: np.ndarray, min_ctg: float | None = None) -> tuple[bool, float | None]:
        ck = self.env.checker
        if float(pose_metric(q, self.q0[None], self.w)[0]) < self.exclude_radius:
            return False, None
        if not ck.is_valid(q):
            return False, None
        c = self.ctg(q)
        if c is None or (min_ctg is not None and c < min_ctg):
            return False, c
        if self.env.checker.escape_direction(q, self.env._escape_dirs) is not None:  # already free
            return False, c
        return True, c

    def _jitter(self, q, rng, st, sr) -> np.ndarray:
        t = q[:3] + rng.normal(0.0, st, 3)
        r = quat_normalize(quat_mul(quat_from_rotvec(rng.normal(0.0, sr, 3)), q[3:]))
        return np.concatenate([t, r])

    def oracle_path(self) -> np.ndarray:
        if self._path is None:
            from ..puzzles import paper_alpha as pa

            self._path = np.load(pa.PUZZLE_OUT / f"{self.env.puzzle.name}-reference.npz")["path"]
        return self._path

    def _path_arclength(self):
        P = self.oracle_path()
        seg = np.array([float(pose_metric(a, b[None], self.w)[0]) for a, b in zip(P[:-1], P[1:])])
        return P, seg, np.concatenate([[0.0], np.cumsum(seg)])

    def _at(self, s: float) -> np.ndarray:
        P, seg, L = self._path_arclength()
        k = int(np.clip(np.searchsorted(L, s, side="right") - 1, 0, len(seg) - 1))
        f = 0.0 if seg[k] == 0 else (s - L[k]) / seg[k]
        return interpolate(P[k], P[k + 1], float(np.clip(f, 0.0, 1.0)))

    def free_arclength(self) -> float:
        """Path length from the start to the first pose on the oracle path that is free (bisection, 1 mm)."""
        if getattr(self, "_s_free", None) is None:
            _, _, L = self._path_arclength()
            dirs = self.env._escape_dirs
            lo, hi = 0.0, float(L[-1])
            while hi - lo > 1e-3:
                mid = (lo + hi) / 2
                if self.env.checker.escape_direction(self._at(mid), dirs) is None:
                    lo = mid
                else:
                    hi = mid
            self._s_free = hi
        return self._s_free

    def _along_path(self, d: float) -> np.ndarray:
        return self._at((1.0 - float(np.clip(d, 0.0, 1.0))) * self.free_arclength())

    # ------------------------------------------------------------------ sample
    def sample(self, rng: np.random.Generator, **override) -> tuple[np.ndarray, dict]:
        mode = override.get("mode", self.mode)
        d = float(override.get("curriculum_distance", self.curriculum_distance))
        source = override.get("curriculum_source", self.curriculum_source)
        for tries in range(1, self.max_tries + 1):
            if mode == "perturb":
                q = self._jitter(self.q0, rng, self.sigma_t, self.sigma_r)
                ok, c = self.acceptable(q, self.min_ctg_frac * self.c0)
            elif mode == "curriculum" and source == "oracle":
                q = self._jitter(self._along_path(d + rng.uniform(-0.02, 0.02)), rng, self.jitter_t, self.jitter_r)
                ok, c = self.acceptable(q)
            elif mode == "curriculum" and source == "roadmap":
                rm = self.env.roadmap
                idx = np.flatnonzero(np.abs(rm.cost / self.c0 - d) <= self.band)
                if idx.size == 0:
                    raise StartSamplingError(f"no roadmap node with ctg/c0 within {self.band} of {d}")
                q = np.asarray(rm.nodes[rng.choice(idx)], float)
                ok, c = self.acceptable(q)
            else:
                raise ValueError(f"unknown start mode/source {mode}/{source}")
            if ok:
                return q, {"start_mode": mode, "curriculum_distance": d if mode == "curriculum" else None,
                           "curriculum_source": source if mode == "curriculum" else None,
                           "start_ctg": None if c is None else round(c, 5), "canonical_ctg": round(self.c0, 5), "tries": tries}
        raise StartSamplingError(f"no acceptable {mode} start in {self.max_tries} tries")
