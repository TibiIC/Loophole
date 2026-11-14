"""Procedural Alpha-style tube geometry (our own design, inspired by the Alpha puzzle).

Centre-line: the nodal cubic, lifted by a linear twist so the curve does not
meet itself at its crossing:

    c(t) = S * (t^2 - 1,  t^3 - t,  k t),    t in [-T, T]

For t in [-1, 1] it traces the loop of an "alpha"; at t = +-1 both strands pass
over the origin, separated vertically by 2kS (the open "gap" of the loop); for
|t| > 1 the two tails cross and diverge. The tube is a sphere of radius r swept
along the centre-line. Another tube of the same radius fits through the gap with
nominal clearance

    c = 2kS - 4r    =>    k = (c + 4r) / (2S)
"""

from __future__ import annotations

import numpy as np


def centerline(S: float, k: float, T: float, n: int) -> np.ndarray:
    t = np.linspace(-T, T, n)
    return np.stack([S * (t**2 - 1), S * (t**3 - t), S * k * t], axis=1)


def twist_for_clearance(S: float, r: float, clearance: float) -> float:
    return (clearance + 4 * r) / (2 * S)


def resample_by_sagitta(S: float, k: float, T: float, max_sagitta: float, n_dense: int = 20000) -> np.ndarray:
    """Polyline whose deviation from the smooth curve (sagitta) is <= max_sagitta."""
    dense = centerline(S, k, T, n_dense)
    out = [dense[0]]
    i = 0
    while i < len(dense) - 1:
        j = i + 1
        # extend the chord while every dense point stays within max_sagitta of it
        while j + 1 < len(dense):
            a, b = dense[i], dense[j + 1]
            seg = b - a
            L2 = float(seg @ seg)
            pts = dense[i + 1 : j + 1]
            tt = np.clip((pts - a) @ seg / L2, 0, 1)
            dev = np.linalg.norm(pts - (a + tt[:, None] * seg), axis=1).max()
            if dev > max_sagitta:
                break
            j += 1
        out.append(dense[j])
        i = j
    return np.array(out)


def polyline_sagitta(poly: np.ndarray, S: float, k: float, T: float, n_dense: int = 20000) -> float:
    """Max distance from the smooth centre-line to the polyline (decomposition error)."""
    dense = centerline(S, k, T, n_dense)
    a, b = poly[:-1], poly[1:]
    seg = b - a
    best = np.full(len(dense), np.inf)
    for s in range(len(seg)):
        L2 = float(seg[s] @ seg[s])
        tt = np.clip((dense - a[s]) @ seg[s] / L2, 0, 1)
        d = np.linalg.norm(dense - (a[s] + tt[:, None] * seg[s]), axis=1)
        best = np.minimum(best, d)
    return float(best.max())


def segment_distances(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """All pairwise distances between segments of polylines P and Q (N-1 x M-1)."""
    p0, p1 = P[:-1, None, :], P[1:, None, :]
    q0, q1 = Q[None, :-1, :], Q[None, 1:, :]
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    denom = a * e - b * b
    s = np.where(denom > 1e-15, np.clip((b * f - c * e) / np.where(denom > 1e-15, denom, 1), 0, 1), 0.0)
    t = (b * s + f) / e
    t_c = np.clip(t, 0, 1)
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = t_c
    diff = p0 + d1 * s[..., None] - (q0 + d2 * t[..., None])
    return np.linalg.norm(diff, axis=-1)


def gauss_linking(P: np.ndarray, Q: np.ndarray) -> float:
    """Gauss linking integral of two polylines (exact per segment pair, Klenin & Langowski 2000).
    Integer for closed curves; real-valued for open ones."""
    total = 0.0
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        c, d = Q[:-1], Q[1:]
        r13, r14, r23, r24 = c - a, d - a, c - b, d - b
        r12, r34 = b - a, d - c
        faces = [(r13, r14), (r14, r24), (r24, r23), (r23, r13)]
        n = [np.cross(u, v) for u, v in faces]
        n = [x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-300) for x in n]
        omega = sum(np.arcsin(np.clip(np.sum(n[j] * n[(j + 1) % 4], axis=1), -1, 1)) for j in range(4))
        sign = np.sign(np.sum(np.cross(r34, r12[None, :]) * r13, axis=1))
        total += float(np.sum(omega * sign))
    return total / (4 * np.pi)
