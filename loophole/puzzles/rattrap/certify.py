"""Swept-motion certificate for the Rat Trap cord, alongside the per-substep penetration audit.

Between two physics substeps every cord node is taken to move in a straight line; sim/certify.py's polyline
certificate then proves (or fails to prove) that no point of the cord, or of a body carried on it (the mouse),
comes closer to the rigid structure than -slack anywhere in between. The cord is 1.2 mm in radius and the hole
walls 1.5 mm thick, so a per-state audit alone could miss a strand crossing a wall between two checks.

Clearance queries reuse FrozenEscape's query model (the static structure plus one capsule per segment and the
carried spheres): the capsules are re-posed vectorised, and answered at margin 0 (penetrations only) whenever the
sweep bound is below the slack, which is nearly always. Not covered: cord-against-cord (self) passage; that stays
with the physics and the audit.

Adapted from `motion_safe` in [HO26] (see sim/certify.py and REFERENCES.md).
"""

from __future__ import annotations

import numpy as np

from ...sim.certify import CertificateLog, polyline_motion_safe
from .reward import FrozenEscape


def _z_to(u: np.ndarray) -> np.ndarray:
    """Quaternions (w, x, y, z) rotating +z onto each unit row of u (spin about the axis is irrelevant for a capsule)."""
    q = np.c_[1.0 + u[:, 2], -u[:, 1], u[:, 0], np.zeros(len(u))]
    flip = u[:, 2] < -1 + 1e-12
    q[flip] = [0.0, 1.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class CordMotionCertifier:
    def __init__(self, sim, slack: float | None = None, max_depth: int = 12, escape: FrozenEscape | None = None):
        self.sim = sim
        self.esc = escape or FrozenEscape(sim.spec, sim)
        self.slack = float(sim.spec["penetration_tolerance"] if slack is None else slack)
        self.max_depth = max_depth
        self.log = CertificateLog()
        self._prev = None
        self.steps = 0

    def _pose(self, nodes: np.ndarray) -> None:
        e, P = self.esc, np.asarray(nodes, float)
        a, b = P[:-1], P[1:]
        seg = b - a
        u = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-15)
        g = e.ag[: e.n]
        e.m.geom_pos[g] = (a + b) / 2
        e.m.geom_quat[g] = _z_to(u)
        for k, (node, _) in enumerate(e.spheres):
            e.m.geom_pos[e.ag[e.n + k]] = P[node]

    def gap_at_least(self, nodes, need: float) -> bool:
        self._pose(nodes)
        return self.esc._min_dist(np.zeros(3), max(need, 0.0)) >= need

    def prime(self) -> None:
        self._prev = self.sim.node_positions()

    def after_substep(self) -> None:
        self.steps += 1
        cur = self.sim.node_positions()
        if self._prev is not None:
            ok = polyline_motion_safe(self.gap_at_least, self._prev, cur, slack=self.slack, max_depth=self.max_depth)
            self.log.record(ok, self.steps, "" if ok else f"substep {self.steps}: cord sweep not certified within {self.slack} m")
        self._prev = cur
