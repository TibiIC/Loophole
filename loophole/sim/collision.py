"""The one collision checker used by the planner, the roadmap, the reward and the verifier.

It owns a *query* MuJoCo model compiled from the same puzzle spec by the same
builder as the simulation (identical geoms), poses the target body there and
asks MuJoCo's own collision pipeline (``mj_forward``) for contacts between the
target and the frame. Nothing here can disagree with the physics because it is
the physics' collision detection.

Poses are the target's pose relative to the fixed frame, which sits at the
world origin: ``pose = (x, y, z, qw, qx, qy, qz)``.

* ``is_valid(pose, margin=0)``: no frame-target pair closer than ``margin``
  (0 = no penetration). A positive margin inflates the target (planning slack).
* ``is_separated(pose)``: ``is_valid(pose, margin=d_sep)``, i.e. the minimum
  distance exceeds the goal threshold.
* ``clearance(pose, dmax)``: minimum frame-target distance, capped at ``dmax``.
  For mesh puzzles, distances beyond ``near`` (5 cm) come from a conservative
  proxy (convex hulls of surface chunks, a superset of the shapes), so they are
  lower bounds; ``is_separated`` uses the same proxy and can therefore never
  report separation falsely. Distances below ``near`` are exact.
* ``segment_valid(p, q, step)``: dense straight-line check in SE(3).
"""

from __future__ import annotations

import mujoco
import numpy as np

from .builder import build_model
from .geometry import quat_normalize, quat_slerp
from .spec import Puzzle

Pose = np.ndarray  # (7,) x y z qw qx qy qz


def pose_metric(p: np.ndarray, q: np.ndarray, w: float) -> np.ndarray:
    """d = ||t1 - t2|| + w * angle(R1^T R2); vectorised over leading dims of q."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    dt = np.linalg.norm(q[..., :3] - p[:3], axis=-1)
    dot = np.abs(np.sum(q[..., 3:] * p[3:], axis=-1))
    ang = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
    return dt + w * ang


def interpolate(p: np.ndarray, q: np.ndarray, f: float) -> np.ndarray:
    """Straight line in SE(3): lerp translation, slerp rotation (same as the rig's moves)."""
    return np.concatenate([p[:3] + f * (q[:3] - p[:3]), quat_slerp(p[3:], q[3:], f)])


class CollisionChecker:
    NEAR = 0.05  # m: exact distances up to here; the far-field proxy beyond

    def __init__(self, puzzle: Puzzle, _proxy: bool = True):
        self.puzzle = puzzle
        self.proxy = None
        heavy = any(g.type == "mesh" and g.decompose == "facewedge" for b in puzzle.frame + puzzle.objects for g in b.geoms)
        if heavy and _proxy:
            def coarse(b):
                return b.model_copy(update={"geoms": [g.model_copy(update={"decompose": "chunkhull"}) if g.type == "mesh" else g for g in b.geoms]})
            pz = puzzle.model_copy(update={"frame": [coarse(b) for b in puzzle.frame], "objects": [coarse(b) for b in puzzle.objects]})
            self.proxy = CollisionChecker(pz, _proxy=False)
        built = build_model(puzzle, num_handles=0)
        self.model = mujoco.MjModel.from_xml_string(built.xml)
        # MuJoCo's body-level midphase culls pairs on un-inflated bounding volumes, so
        # margin queries (is_separated, clearance, planning margins) silently miss nearby
        # pairs (tens of mm). It is therefore disabled for every margin > 0 query, and kept
        # for plain penetration checks (margin 0), where it is exact and much faster on
        # meshes with thousands of convex pieces.
        # Regression test: tests/test_alpha.py::test_checker_margin_queries_exact
        self.data = mujoco.MjData(self.model)
        m = self.model
        tid = m.body(puzzle.target).id
        frame_ids = {m.body(b).id for b in built.frame_bodies}
        self.target_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == tid])
        self._is_target = np.zeros(m.ngeom, bool)
        self._is_target[self.target_geoms] = True
        self._is_frame = np.array([m.geom_bodyid[g] in frame_ids for g in range(m.ngeom)])
        self._qadr = m.jnt_qposadr[m.body(puzzle.target).jntadr[0]]
        # only frame-vs-target pairs matter: switch every other geom out of collision
        for g in range(m.ngeom):
            if not (self._is_target[g] or self._is_frame[g]):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        self._margin = None
        self._set_margin(0.0)
        self.calls = 0

    def _set_margin(self, margin: float) -> None:
        if margin != self._margin:
            self.model.geom_margin[self.target_geoms] = margin
            if margin > 0:
                self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
            else:
                self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            self._margin = margin

    def _pairs(self, pose: Pose, margin: float) -> np.ndarray:
        """Distances of all frame-target contacts within `margin` at this pose."""
        self.calls += 1
        self._set_margin(margin)
        d = self.data
        d.qpos[self._qadr : self._qadr + 3] = pose[:3]
        d.qpos[self._qadr + 3 : self._qadr + 7] = quat_normalize(pose[3:7])
        mujoco.mj_forward(self.model, d)
        n = d.ncon
        if n == 0:
            return np.empty(0)
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        mask = (self._is_target[g1] & self._is_frame[g2]) | (self._is_target[g2] & self._is_frame[g1])
        return d.contact.dist[:n][mask]

    def is_valid(self, pose: Pose, margin: float = 0.0) -> bool:
        dist = self._pairs(np.asarray(pose, float), margin)
        return bool(np.all(dist >= margin))

    def is_separated(self, pose: Pose) -> bool:
        if self.proxy is not None:  # conservative: proxy distance <= true distance (+1 mm for
            # MuJoCo's convex-distance tolerance, measured <= 0.6 mm against FCL)
            return self.proxy.is_valid(pose, margin=self.puzzle.goal.d_sep + 0.001)
        return self.is_valid(pose, margin=self.puzzle.goal.d_sep)

    def clearance(self, pose: Pose, dmax: float) -> float:
        if self.proxy is not None and dmax > self.NEAR:
            near = self._exact(pose, self.NEAR)
            if near < self.NEAR:
                return near
            return float(max(self.NEAR, self.proxy._exact(pose, dmax)))  # lower bound beyond NEAR
        return self._exact(pose, dmax)

    def _exact(self, pose: Pose, dmax: float) -> float:
        dist = self._pairs(np.asarray(pose, float), dmax)
        return float(min(dmax, dist.min())) if dist.size else float(dmax)

    def segment_valid(self, p: Pose, q: Pose, w: float, step: float, margin: float = 0.0) -> bool:
        """Check the straight SE(3) segment p -> q at metric resolution `step` (endpoints included)."""
        p = np.asarray(p, float)
        q = np.asarray(q, float)
        n = max(1, int(np.ceil(float(pose_metric(p, q, w)) / step)))
        # coarse-to-fine order finds collisions early
        order = sorted(range(n + 1), key=lambda i: (bin(i)[::-1], i)) if n > 4 else range(n + 1)
        for i in order:
            if not self.is_valid(interpolate(p, q, i / n), margin):
                return False
        return True
