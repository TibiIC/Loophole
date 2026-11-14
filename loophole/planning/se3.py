"""SE(3) helpers for planning: poses are (x, y, z, qw, qx, qy, qz) relative to the fixed piece."""

from __future__ import annotations

import numpy as np

from ..sim.collision import interpolate, pose_metric
from ..sim.geometry import quat_conj, quat_mul, quat_normalize

__all__ = ["interpolate", "pose_metric", "random_quat", "steer", "char_length", "rel_rotation"]


def random_quat(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Uniform random unit quaternions (Shoemake), w >= 0."""
    m = 1 if n is None else n
    u1, u2, u3 = rng.random((3, m))
    q = np.stack(
        [np.sqrt(u1) * np.cos(2 * np.pi * u3), np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(u1) * np.sin(2 * np.pi * u3)],
        axis=1,
    )
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q[0] if n is None else q


def steer(p: np.ndarray, q: np.ndarray, w: float, max_dist: float) -> np.ndarray:
    d = float(pose_metric(p, q, w))
    return q.copy() if d <= max_dist else interpolate(p, q, max_dist / d)


def char_length(points: np.ndarray) -> float:
    """Characteristic length w of a body: max distance of its centre-line points from
    the body origin, plus the tube radius is added by the caller. With this w the
    metric upper-bounds the displacement of any point of the body."""
    return float(np.max(np.linalg.norm(points, axis=1)))


def rel_rotation(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """World-frame rotation taking q0 to q1 (q1 = r * q0)."""
    return quat_normalize(quat_mul(q1, quat_conj(q0)))
