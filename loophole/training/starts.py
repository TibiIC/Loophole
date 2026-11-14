"""Training start states for the Alpha puzzle (the canonical start is the held-out test).

Modes
-----
* ``perturb``: deviations of a reference start: Gaussian noise (translation sigma_t, rotation
  sigma_r as a random rotation vector) on the canonical start (``chirality="canonical"``) or on
  the turned-over canonical start (``"flipped"``; ``"both"`` picks one per episode), accepted
  only if collision-free, still interlocked with the same tie, not yet free, and its roadmap
  cost-to-go is at least ``min_ctg_frac`` of the reference start's. The flipped tie needs a
  certified way out (start bank) for its cost-to-go; until then it raises.
* ``curriculum`` (reverse curriculum, per tie; default canonical): a pose at a chosen distance
  ``d`` in [0, 1] from the goal, sampled along the tie's reference path (canonical: the published
  solution; flipped: the start bank's certified path of the turned-over start) (``source="oracle"``: d = path length still to go to
  the first free pose on the path / that pose's path length from the start, small jitter) or from roadmap nodes (``source="roadmap"``: cost-to-go within
  ``band`` of d * c0); also required to be collision-free and not yet free.

* ``interlocked``: genuinely different configurations, not perturbations. Tube B's loop is
  placed around tube A's loop tube at a random point (loop axis roughly along A's tube there,
  random tilt, random spin about the axis, random centre offset), which links the loops by
  construction; ~45 % of candidates are collision-free. ``chirality`` picks the tie:
  "both" (default), "canonical" (same linking sign as the canonical start) or "flipped".
* ``bank``: a certified start from the start bank (loophole/training/bank.py: each has a
  planner-found way out and a corridor in the training roadmap), plus small jitter;
  ``chirality`` selects the tie. Only ties with certified starts can be used.
* ``flipped``: the canonical start turned over (B rotated 180 degrees about an axis in its
  loop plane through the loop centre), which ties the loops the *other* way (linking number
  +1 instead of -1). The collision-free turn-over with the most clearance is used, plus jitter.

"Loops connected" is tested topologically: the Gauss linking number of the two alphas' loops,
each closed across its own gap, is +-1 when interlocked and 0 when not (it changes only when
one tube passes through the other's gap). Every training start must have |Lk| = 1, be
collision-free and not yet free (no free escape by translation).

In every mode, poses within ``exclude_radius`` of q0 (in the SE(3) metric
||dt|| + w * angle) are rejected, so the canonical start is never sampled in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.collision import interpolate, pose_metric
from ..sim.geometry import mat_to_quat, quat_from_axis_angle, quat_from_rotvec, quat_mul, quat_normalize, quat_to_mat


def loop_frame(loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centre, unit normal) of a closed loop polyline (vector-area normal)."""
    c = loop[:-1].mean(0)
    n = np.zeros(3)
    for p, q in zip(loop[:-1], loop[1:]):
        n += np.cross(p - c, q - c)
    return c, n / np.linalg.norm(n)


def _rotation_taking(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)  # 180 degrees about p
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


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
    chirality: str = "both"  # interlocked mode: "both", "canonical" or "flipped"
    tilt_sigma: float = 0.35  # interlocked: tilt of B's loop axis away from A's tube direction (rad)
    offset_sigma: float = 0.15  # interlocked: offset of B's loop centre from A's tube axis (m)
    require_ctg: bool = False  # interlocked / flipped: also require a visible roadmap node (partial credit)
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
        if self.mode not in ("perturb", "curriculum", "interlocked", "flipped", "bank"):
            raise ValueError("mode must be perturb, curriculum, interlocked, flipped or bank (the canonical start is never a training start)")
        if self.chirality not in ("both", "canonical", "flipped"):
            raise ValueError("chirality must be both, canonical or flipped")
        loops = self.env.loops
        self.loop_a = loops[self.env.built.frame_bodies[0]]
        self.loop_b = loops[self.env.puzzle.target]
        self.cb, self.nb = loop_frame(self.loop_b)
        self.lk0 = self.linking(self.q0)
        if abs(abs(self.lk0) - 1) > 0.1:
            raise ValueError(f"canonical start is not interlocked (Lk = {self.lk0:.3f})")
        self._flipped = None
        self._bank = None

    def bank(self):
        if self._bank is None:
            from ..puzzles import paper_alpha as pa
            from .bank import BANK_FILE, StartBank

            path = pa.PUZZLE_OUT / BANK_FILE.format(name=self.env.puzzle.name)
            if not path.exists():
                raise StartSamplingError(f"no start bank at {path}: run scripts/build_start_bank.py")
            self._bank = StartBank(path)
        return self._bank

    # ------------------------------------------------------------------ checks
    def ctg(self, q) -> float | None:
        return self.env.roadmap.cost_to_go(q, self.env.checker)[0]

    def linking(self, q: np.ndarray) -> float:
        """Gauss linking number of the two loops (each closed across its own gap): +-1 = interlocked."""
        from ..puzzles.alpha.geometry import gauss_linking

        return float(gauss_linking(self.loop_a, q[:3] + self.loop_b @ quat_to_mat(q[3:]).T))

    def chirality_of(self, lk: float) -> str:
        return "canonical" if np.sign(lk) == np.sign(self.lk0) else "flipped"

    def acceptable(self, q: np.ndarray, min_ctg: float | None = None, need_ctg: bool = True,
                   chirality: str | None = "both") -> tuple[bool, float | None, float | None]:
        """(ok, cost-to-go, linking number). Rejects: near canonical, colliding, loops not
        connected (|Lk| != 1) or of the wrong tie, no visible roadmap node (if needed), too close
        to free (min_ctg), already free. chirality=None skips the linking test (reverse curriculum:
        near the goal the loops are already unlinked while the tube is still caged)."""
        ck = self.env.checker
        if float(pose_metric(q, self.q0[None], self.w)[0]) < self.exclude_radius:
            return False, None, None
        if not ck.is_valid(q):
            return False, None, None
        lk = self.linking(q)
        if chirality is not None and (abs(abs(lk) - 1) > 0.1 or (chirality != "both" and self.chirality_of(lk) != chirality)):
            return False, None, lk
        c = self.ctg(q)
        if (need_ctg and c is None) or (min_ctg is not None and (c is None or c < min_ctg)):
            return False, c, lk
        if ck.escape_direction(q, self.env._escape_dirs) is not None:  # already free
            return False, c, lk
        return True, c, lk

    def interlocked_candidate(self, rng: np.random.Generator, chirality: str) -> np.ndarray:
        """B's loop around A's loop tube at a random point (links the loops by construction)."""
        A = self.loop_a
        i = int(rng.integers(len(A) - 1))
        tangent = A[(i + 1) % (len(A) - 1)] - A[i - 1]
        tangent /= np.linalg.norm(tangent)
        sign = {"both": rng.choice([-1.0, 1.0]), "canonical": 1.0, "flipped": -1.0}[chirality]
        # the tie direction follows the sign of B's loop axis along A's tube; calibrate once
        sign *= self._axis_sign()
        axis = sign * tangent + rng.normal(0.0, self.tilt_sigma, 3)
        axis /= np.linalg.norm(axis)
        R = quat_to_mat(quat_from_axis_angle(axis, rng.uniform(0, 2 * np.pi))) @ _rotation_taking(self.nb, axis)
        off = rng.normal(0.0, self.offset_sigma, 3)
        off -= (off @ axis) * axis
        return np.concatenate([A[i] + off - R @ self.cb, mat_to_quat(R)])

    def _axis_sign(self) -> float:
        """+1 if loop-axis-along-tangent gives the canonical tie, else -1 (estimated once)."""
        if getattr(self, "_sign", None) is None:
            rng = np.random.default_rng(12345)
            votes = []
            self._sign = 1.0
            while len(votes) < 15:
                q = self.interlocked_candidate(rng, "canonical")
                lk = self.linking(q)
                if abs(abs(lk) - 1) < 0.1:
                    votes.append(self.chirality_of(lk) == "canonical")
            self._sign = 1.0 if np.mean(votes) >= 0.5 else -1.0
        return self._sign

    def flipped_start(self) -> np.ndarray:
        """The canonical start turned over (180 degrees about an in-plane loop axis), opposite tie."""
        if self._flipped is None:
            R0 = quat_to_mat(self.q0[3:])
            cw = self.q0[:3] + R0 @ self.cb
            u = np.cross(self.nb, [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(self.nb, u)
            best = None
            for ang in np.linspace(0, np.pi, 72, endpoint=False):
                aw = R0 @ (np.cos(ang) * u + np.sin(ang) * v)
                dq = quat_from_axis_angle(aw, np.pi)
                q = np.concatenate([cw + quat_to_mat(dq) @ (self.q0[:3] - cw), quat_normalize(quat_mul(dq, self.q0[3:]))])
                if not self.env.checker.is_valid(q) or self.chirality_of(self.linking(q)) != "flipped":
                    continue
                clr = self.env.checker.clearance(q, 0.05)
                if best is None or clr > best[0]:
                    best = (clr, q)
            if best is None:
                raise StartSamplingError("no collision-free turn-over of the canonical start")
            self._flipped = best[1]
        return self._flipped

    def _jitter(self, q, rng, st, sr) -> np.ndarray:
        t = q[:3] + rng.normal(0.0, st, 3)
        r = quat_normalize(quat_mul(quat_from_rotvec(rng.normal(0.0, sr, 3)), q[3:]))
        return np.concatenate([t, r])

    # ------------------------------------------------------------------ per-tie references
    def base_pose(self, tie: str) -> np.ndarray:
        """The reference start of a tie: the canonical start, or the canonical start turned over."""
        return self.q0 if tie == "canonical" else self.flipped_start()

    def base_ctg(self, tie: str) -> float:
        """Cost-to-go of the tie's reference start; flipped needs a certified corridor (start bank)."""
        if tie == "canonical":
            return self.c0
        c = self.ctg(self.flipped_start())
        if c is None:
            raise StartSamplingError("the flipped tie has no certified way out yet (no cost-to-go): build the start bank "
                                     "(scripts/build_start_bank.py); until then only the canonical tie can be used here")
        return c

    def flipped_ready(self) -> bool:
        """True once the start bank certified the turned-over start (cost-to-go and reference path exist)."""
        if getattr(self, "_flipped_ready", None) is None:
            try:
                self._flipped_ready = self.ctg(self.flipped_start()) is not None and self.bank().flipped_path is not None
            except StartSamplingError:
                self._flipped_ready = False
        return self._flipped_ready

    def _pick_tie(self, chir: str, rng: np.random.Generator) -> str:
        """`both` = every usable tie: canonical always, flipped once certified (option 1: no uncertified ties)."""
        if chir != "both":
            return chir
        return str(rng.choice(["canonical", "flipped"])) if self.flipped_ready() else "canonical"

    def reference_path(self, tie: str = "canonical") -> np.ndarray:
        """Canonical tie: the published solution. Flipped tie: the certified path of the turned-over
        start from the start bank (its planner corridor followed by the roadmap's shortest route)."""
        cache = self.__dict__.setdefault("_paths", {})
        if tie not in cache:
            if tie == "canonical":
                from ..puzzles import paper_alpha as pa

                cache[tie] = np.load(pa.PUZZLE_OUT / f"{self.env.puzzle.name}-reference.npz")["path"]
            else:
                fp = self.bank().flipped_path
                if fp is None:
                    raise StartSamplingError("no certified path for the flipped start in the start bank")
                cache[tie] = fp
        return cache[tie]

    def oracle_path(self) -> np.ndarray:  # backwards compatible name
        return self.reference_path("canonical")

    def _path_arclength(self, tie: str = "canonical"):
        P = self.reference_path(tie)
        seg = np.array([float(pose_metric(a, b[None], self.w)[0]) for a, b in zip(P[:-1], P[1:])])
        return P, seg, np.concatenate([[0.0], np.cumsum(seg)])

    def _at(self, s: float, tie: str = "canonical") -> np.ndarray:
        P, seg, L = self._path_arclength(tie)
        k = int(np.clip(np.searchsorted(L, s, side="right") - 1, 0, len(seg) - 1))
        f = 0.0 if seg[k] == 0 else (s - L[k]) / seg[k]
        return interpolate(P[k], P[k + 1], float(np.clip(f, 0.0, 1.0)))

    def free_arclength(self, tie: str = "canonical") -> float:
        """Path length from the start to the first free pose on the tie's reference path (bisection, 1 mm)."""
        cache = self.__dict__.setdefault("_s_free_by_tie", {})
        if tie not in cache:
            _, _, L = self._path_arclength(tie)
            dirs = self.env._escape_dirs
            lo, hi = 0.0, float(L[-1])
            while hi - lo > 1e-3:
                mid = (lo + hi) / 2
                if self.env.checker.escape_direction(self._at(mid, tie), dirs) is None:
                    lo = mid
                else:
                    hi = mid
            cache[tie] = hi
        return cache[tie]

    def _along_path(self, d: float, tie: str = "canonical") -> np.ndarray:
        return self._at((1.0 - float(np.clip(d, 0.0, 1.0))) * self.free_arclength(tie), tie)

    def node_ties(self) -> np.ndarray:
        """Tie label of every node of the current roadmap (canonical / flipped / unlinked), cached."""
        rm = self.env.roadmap
        key = getattr(rm, "sha256", id(rm))
        if getattr(self, "_node_ties_key", None) != key:
            if hasattr(rm, "tie") and rm.tie is not None:
                self._node_ties = np.asarray(rm.tie)
            else:
                lab = []
                for q in np.asarray(rm.nodes):
                    lk = self.linking(q)
                    lab.append(self.chirality_of(lk) if abs(abs(lk) - 1) < 0.1 else "unlinked")
                self._node_ties = np.array(lab)
            self._node_ties_key = key
        return self._node_ties

    # ------------------------------------------------------------------ sample
    def sample(self, rng: np.random.Generator, **override) -> tuple[np.ndarray, dict]:
        mode = override.get("mode", self.mode)
        d = float(override.get("curriculum_distance", self.curriculum_distance))
        source = override.get("curriculum_source", self.curriculum_source)
        chir = override.get("chirality", self.chirality)
        for tries in range(1, self.max_tries + 1):
            tie = self._pick_tie(chir, rng)
            if mode == "perturb":  # deviations of the tie's reference start, same tie, still far from free
                base = self.base_pose(tie)
                q = self._jitter(base, rng, self.sigma_t, self.sigma_r)
                ok, c, lk = self.acceptable(q, self.min_ctg_frac * self.base_ctg(tie), chirality=tie)
            elif mode == "curriculum" and source == "oracle":
                q = self._jitter(self._along_path(d + rng.uniform(-0.02, 0.02), tie), rng, self.jitter_t, self.jitter_r)
                ok, c, lk = self.acceptable(q, chirality=None)
            elif mode == "curriculum" and source == "roadmap":
                rm = self.env.roadmap
                ref = self.base_ctg(tie)
                idx = np.flatnonzero((np.abs(np.asarray(rm.cost) / ref - d) <= self.band) & np.isin(self.node_ties(), [tie, "unlinked"]))  # near the goal the loops are unlinked
                if idx.size == 0:
                    raise StartSamplingError(f"no {tie}-tie roadmap node with ctg/ctg_ref within {self.band} of {d}")
                q = np.asarray(rm.nodes[rng.choice(idx)], float)
                ok, c, lk = self.acceptable(q, chirality=None)
            elif mode == "interlocked":
                q = self.interlocked_candidate(rng, chir)
                ok, c, lk = self.acceptable(q, need_ctg=self.require_ctg, chirality=chir)
            elif mode == "bank":
                i, base = self.bank().sample(rng, chir)
                q = self._jitter(base, rng, self.jitter_t, self.jitter_r) if tries <= self.max_tries // 2 else base
                ok, c, lk = self.acceptable(q, need_ctg=True, chirality=str(self.bank().tie[i]))
            elif mode == "flipped":
                q = self._jitter(self.flipped_start(), rng, self.jitter_t, self.jitter_r) if tries > 1 or self.jitter_t > 0 else self.flipped_start()
                ok, c, lk = self.acceptable(q, need_ctg=self.require_ctg, chirality="flipped")
            else:
                raise ValueError(f"unknown start mode/source {mode}/{source}")
            if ok:
                return q, {"start_mode": mode, "curriculum_distance": d if mode == "curriculum" else None,
                           "curriculum_source": source if mode == "curriculum" else None,
                           "linking_number": round(lk, 4), "tie": self.chirality_of(lk) if abs(abs(lk) - 1) < 0.1 else "unlinked",
                           "start_ctg": None if c is None else round(c, 5), "canonical_ctg": round(self.c0, 5), "tries": tries}
        raise StartSamplingError(f"no acceptable {mode} start in {self.max_tries} tries")
