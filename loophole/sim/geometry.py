"""Small vector / quaternion helpers (quaternions are MuJoCo order: w, x, y, z)."""

from __future__ import annotations

import numpy as np


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def basis_perp(n) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors u, v such that (u, v, n) is right-handed and orthonormal."""
    n = unit(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(helper, n))
    v = np.cross(n, u)
    return u, v


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    if abs(angle) < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = unit(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), *(axis * s)])


def quat_to_axis_angle(q) -> tuple[np.ndarray, float]:
    q = quat_normalize(q)
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return q[1:] / s, 2 * np.arctan2(s, q[0])


def quat_from_rotvec(rv) -> np.ndarray:
    rv = np.asarray(rv, dtype=float)
    a = np.linalg.norm(rv)
    if a < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_from_axis_angle(rv / a, a)


def quat_to_rotvec(q) -> np.ndarray:
    axis, ang = quat_to_axis_angle(q)
    return axis * ang


def quat_slerp(a, b, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = np.asarray(b, dtype=float)
    if np.dot(a, b) < 0:
        b = -b
    rel = quat_mul(quat_conj(a), b)
    return quat_normalize(quat_mul(a, quat_from_rotvec(quat_to_rotvec(rel) * t)))


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_euler_xyz(rpy) -> np.ndarray:
    """Extrinsic X then Y then Z rotations (roll, pitch, yaw) in radians."""
    r, p, y = rpy
    qx = quat_from_axis_angle([1, 0, 0], r)
    qy = quat_from_axis_angle([0, 1, 0], p)
    qz = quat_from_axis_angle([0, 0, 1], y)
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def quat_between(a, b) -> np.ndarray:
    """Shortest rotation taking direction a onto direction b."""
    a, b = unit(a), unit(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-12:
        u, _ = basis_perp(a)
        return quat_from_axis_angle(u, np.pi)
    axis = np.cross(a, b)
    return quat_normalize(np.array([1.0 + c, *axis]))


def frame_from_x(direction, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion of a frame whose x axis is `direction` (roll fixed by `up_hint`)."""
    x = unit(direction)
    up = np.asarray(up_hint, dtype=float)
    if abs(np.dot(x, unit(up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = unit(np.cross(up, x))
    z = np.cross(x, y)
    mat = np.stack([x, y, z], axis=1)
    return mat_to_quat(mat)


def mat_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return quat_normalize(q)


def polyline_length(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_arclength(pts, n: int) -> np.ndarray:
    """n+1 nodes equally spaced in arclength along the polyline (exact endpoints)."""
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for s in np.linspace(0.0, cum[-1], n + 1):
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(pts) - 2))
        t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-15)
        out.append(pts[i] + t * (pts[i + 1] - pts[i]))
    out[0], out[-1] = pts[0], pts[-1]
    return np.array(out)
