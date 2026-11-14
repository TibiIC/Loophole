"""Swept-motion certificate: a conservative proof that a whole motion interval stays clear of the structure.

The penetration audit (sim/audit.py) checks the state *at* every physics substep; a fast or thin body could
still cross a thin part *between* two checks (the audit's speed cap only bounds that indirectly). This module
certifies the continuous interval between two consecutive states instead of sampling it: it runs *alongside*
the audit, never instead of it.

Method (recursive bisection against a sweep bound). For an interval with midpoint configuration m, let b be an
upper bound on how far any point of the moving body travels between m and either end of the interval. If the
clearance at m exceeds b, nothing can touch the structure anywhere in the interval. Otherwise, split the interval
in two and certify each half, up to `max_depth` levels. A midpoint that is already in contact, or the depth
limit, ends the attempt as *uncertified*. The test is conservative: it can fail to certify a safe motion, but it
never certifies an unsafe one.

    rigid body, translate dp and rotate rv (world rotation vector) about its reference point p:
        b = |dp| / 2 + r_out * |rv| / 2        r_out = the body's outer radius about p
        (a point at distance <= r_out from p, rotated by angle |rv|/2, moves by at most r_out * |rv|/2)
    polyline / capsule chain whose nodes move linearly (the cord between two substeps):
        b = max_i |dx_i| / 2
        (a point on a segment is a convex combination of its two end nodes, so it moves no more than they do)

`slack` admits a declared penetration tolerance (soft contacts sit a little inside each other). The certificate
then proves that no point goes deeper than `slack` during the interval: the continuous-time version of the
audit's per-substep threshold.

Provenance: the recursive-bisection certificate and its rigid sweep bound are adapted from `motion_safe` in a
colleague's ring-on-wire disentanglement environment (puzzle-handoff-v1, src/disentanglement_env/core.py;
shared with us, no licence stated; see REFERENCES.md). Changes here: a clearance predicate instead of a clearance
value (so MuJoCo can answer most queries at margin 0, which is exact and fast), the `slack` for soft contacts,
and the polyline variant for the deformable cord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geometry import quat_from_rotvec, quat_mul, quat_normalize, quat_conj, quat_to_rotvec

GapAtLeast = Callable[[object, float], bool]  # (configuration, need) -> clearance(configuration) >= need ?


def _certify(start, delta, interp, halve, bound, gap_at_least: GapAtLeast, slack: float, max_depth: int, tol: float,
             depth: int = 0) -> bool:
    mid = interp(start, delta, 0.5)
    b = bound(delta)
    if gap_at_least(mid, b - slack + tol):
        return True
    if depth >= max_depth or not gap_at_least(mid, -slack + tol):
        return False
    half = halve(delta)
    return (_certify(start, half, interp, halve, bound, gap_at_least, slack, max_depth, tol, depth + 1)
            and _certify(mid, half, interp, halve, bound, gap_at_least, slack, max_depth, tol, depth + 1))


def rigid_motion_safe(gap_at_least: GapAtLeast, p, q, dp, rv, outer_radius: float, slack: float = 0.0,
                      max_depth: int = 12, tol: float = 1e-9) -> bool:
    """Certify the rigid motion (p, q) -> (p + dp, R(rv) q): translation lerped, world rotation about p by f * rv.
    Configurations passed to `gap_at_least` are 7-vectors (x, y, z, qw, qx, qy, qz)."""
    p, q, dp, rv = (np.asarray(v, float) for v in (p, q, dp, rv))

    def interp(c, d, f):
        return np.concatenate([c[:3] + f * d[0], quat_normalize(quat_mul(quat_from_rotvec(f * d[1]), c[3:]))])

    return _certify(np.concatenate([p, quat_normalize(q)]), (dp, rv), interp, lambda d: (d[0] / 2, d[1] / 2),
                    lambda d: float(np.linalg.norm(d[0]) / 2 + outer_radius * np.linalg.norm(d[1]) / 2),
                    gap_at_least, slack, max_depth, tol)


def rigid_pose_motion_safe(gap_at_least: GapAtLeast, pose0, pose1, outer_radius: float, **kw) -> bool:
    """Certify the motion between two poses (x, y, z, qw, qx, qy, qz) of a rigid body about its origin."""
    pose0, pose1 = np.asarray(pose0, float), np.asarray(pose1, float)
    rel = quat_mul(quat_normalize(pose1[3:]), quat_conj(quat_normalize(pose0[3:])))
    return rigid_motion_safe(gap_at_least, pose0[:3], pose0[3:], pose1[:3] - pose0[:3], quat_to_rotvec(rel), outer_radius, **kw)


def polyline_motion_safe(gap_at_least: GapAtLeast, nodes0, nodes1, slack: float = 0.0, max_depth: int = 12,
                         tol: float = 1e-9) -> bool:
    """Certify the motion of a capsule chain whose nodes move linearly from `nodes0` to `nodes1`.
    Configurations passed to `gap_at_least` are (n, 3) node arrays."""
    a = np.asarray(nodes0, float)
    d = np.asarray(nodes1, float) - a
    return _certify(a, d, lambda c, dd, f: c + f * dd, lambda dd: dd / 2,
                    lambda dd: float(np.max(np.linalg.norm(dd, axis=1)) / 2), gap_at_least, slack, max_depth, tol)


@dataclass
class CertificateLog:
    """Running tally of certified / uncertified substep intervals (kept apart from the audit's violations)."""

    checked: int = 0
    failures: list[dict] = field(default_factory=list)
    max_records: int = 50
    failed: int = 0

    def record(self, ok: bool, step: int, detail: str = "") -> None:
        self.checked += 1
        if not ok:
            self.failed += 1
            if len(self.failures) < self.max_records:
                self.failures.append({"step": step, "detail": detail})

    def summary(self) -> dict:
        return {"checked": self.checked, "uncertified": self.failed, "records": list(self.failures)}


class RigidMotionCertifier:
    """Certify every substep interval of one rigid body of a LoopholeEnv against its frame (CollisionChecker)."""

    def __init__(self, env, slack: float | None = None, max_depth: int = 12):
        self.env = env
        self.checker = env.checker
        self.slack = float(env.cfg.audit.penetration_threshold if slack is None else slack)
        self.max_depth = max_depth
        _, _, self.outer_radius = self.checker.bounding_spheres()
        self.log = CertificateLog()
        self._prev = None

    def gap_at_least(self, pose, need: float) -> bool:
        if need <= 0.0:  # contacts at margin 0 are exact (midphase on); only penetrations can violate need <= 0
            dist = self.checker._pairs(np.asarray(pose, float), 0.0)
            return bool(dist.size == 0 or dist.min() >= need)
        return self.checker.is_valid(pose, margin=need)

    def prime(self) -> None:
        self._prev = np.asarray(self.env.target_pose(), float).copy()

    def after_substep(self, step: int) -> None:
        cur = np.asarray(self.env.target_pose(), float).copy()
        if self._prev is not None:
            ok = rigid_pose_motion_safe(self.gap_at_least, self._prev, cur, self.outer_radius, slack=self.slack, max_depth=self.max_depth)
            self.log.record(ok, step, "" if ok else f"substep {step}: swept motion not certified within {self.slack} m")
        self._prev = cur
