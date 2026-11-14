"""Aperture-passage alignment: a shaping potential for threading one thing through another.

    Phi(s) = -w * exp(-(d / sigma)^2) * max(0, 1 - (n . tau)^2)

n:     the aperture's axis (unit; a ring's plane normal, a hole's axis). A plane normal is defined only up to
       sign, and spinning the aperture about n changes nothing physical, so only (n . tau)^2 may enter;
       an unsquared n . tau would penalise physically identical states.
tau:   the passer's local tangent at the aperture (unit, also sign-free: a closed cord has no direction), from
       `local_tangent` over the passer's polyline (a cord's nodes, a wire's centre line).
d:     distance from the aperture centre to the passer; the Gaussian gate switches the term off far away,
       where the orientation of the passer relative to the hole means nothing.
w, sigma: `weight`, `gate_distance`.

The penalty is 0 when the passer runs straight along the axis (ready to pass) and w when it lies across the
aperture next to it. It serves both directions of the same geometry: a rigid ring threaded on a cord (n = the
ring's normal, the passer = the cord) and a cord through a fixed hole (n = the hole's axis, the passer = the
cord). As a potential it only ever enters through PotentialShaping, so it is training shaping, never score.

Provenance: the orientation term and `local_tangent` are adapted from a colleague's ring-on-wire
disentanglement environment (puzzle-handoff-v1, src/disentanglement_env/core.py: `potential_terms`,
`local_tangent`; shared with us, no licence stated; see REFERENCES.md). There, n is the third column of the
ring's rotation matrix and tau the wire's smoothed tangent at the ring centre. Changes here: the aperture and
the passer are inputs rather than one fixed ring and wire; `local_tangent` flips each segment direction to agree
with the nearest segment before averaging, because the two strands of a doubled cord near a hole run
antiparallel and would otherwise cancel; the zero-length case falls back to the nearest segment. Their
straight-line distance-to-goal potential is deliberately not ported (it failed in their tests and ignores that
the body must travel along the constraint; our cost-to-go and free-group norm exist for exactly that).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .components import PotentialShaping, RewardState, register_component


def point_segment_distances(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from point p to every segment a[i] -> b[i]."""
    d = b - a
    L2 = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", p - a, d) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    return np.linalg.norm(p - a - t[:, None] * d, axis=1)


def local_tangent(polyline, p, length_scale: float = 0.01, closed: bool = False) -> tuple[np.ndarray, float]:
    """(unit tangent, distance) of a polyline near point p: a smooth, distance-weighted average of the segment
    directions, weights exp(-(d_i^2 - d_min^2) / length_scale^2), each direction sign-aligned with the nearest
    segment first. The tangent's sign is arbitrary (only its line matters)."""
    P = np.asarray(polyline, float)
    if closed:
        P = np.vstack([P, P[:1]])
    a, b = P[:-1], P[1:]
    seg = b - a
    L = np.linalg.norm(seg, axis=1)
    keep = L > 1e-12
    if not keep.any():
        raise ValueError("polyline has no segment of positive length")
    a, seg, L = a[keep], seg[keep], L[keep]
    dirs = seg / L[:, None]
    dist = point_segment_distances(np.asarray(p, float), a, a + seg)
    k = int(np.argmin(dist))
    w = np.exp(-(dist**2 - dist[k] ** 2) / length_scale**2)
    sgn = np.where(dirs @ dirs[k] < 0, -1.0, 1.0)
    t = np.sum((w * sgn)[:, None] * dirs, axis=0)
    n = float(np.linalg.norm(t))
    return (dirs[k].copy() if n < 1e-12 else t / n), float(dist[k])


def alignment_penalty(n, tau, d: float, gate_distance: float) -> float:
    """exp(-(d / sigma)^2) * max(0, 1 - (n . tau)^2) with n, tau normalised: in [0, 1], even in n and in tau."""
    n = np.asarray(n, float)
    tau = np.asarray(tau, float)
    c = float(n @ tau) / (np.linalg.norm(n) * np.linalg.norm(tau))
    return float(np.exp(-((d / gate_distance) ** 2)) * max(0.0, 1.0 - c * c))


def ring_normal(rotation_matrix) -> np.ndarray:
    """A ring's plane normal: the third column of its rotation matrix (the ring lies in its local xy plane)."""
    return np.asarray(rotation_matrix, float)[:, 2].copy()


Resolver = Callable[[RewardState], Any]


class AperturePassageAlignment(PotentialShaping):
    """Phi = -weight * alignment_penalty(n, tau, d, gate_distance), shaped through PotentialShaping.

    aperture_normal: a key into `RewardState.geometry` holding {"center": (3,), "normal": (3,)}, or a callable
                     state -> (center, normal).
    passer_tangent:  a key into `RewardState.geometry` holding the passer's polyline ((n, 3) nodes), or a
                     callable (state, center) -> (tau, d).
    Missing geometry (None) gives no potential, hence no shaping, for that step.
    """

    name = "aperture_alignment"

    def __init__(self, aperture_normal: str | Resolver, passer_tangent: str | Resolver, gate_distance: float, weight: float = 0.25,
                 tangent_length_scale: float = 0.01, closed_passer: bool = False, gamma: float = 0.99, lam: float = 1.0,
                 label: str | None = None, **kw):
        super().__init__(gamma=gamma, lam=lam, gate_distance=gate_distance, weight=weight, **kw)
        if gate_distance <= 0 or weight < 0:
            raise ValueError("need gate_distance > 0 and weight >= 0")
        self.aperture_normal = aperture_normal
        self.passer_tangent = passer_tangent
        self.sigma = float(gate_distance)
        self.weight = float(weight)
        self.ell = float(tangent_length_scale)
        self.closed = bool(closed_passer)
        if label:
            self.name = f"aperture_alignment:{label}"
        self.last_terms: dict | None = None

    def terms(self, s: RewardState) -> dict | None:
        if callable(self.aperture_normal):
            ap = self.aperture_normal(s)
        else:
            ap = (s.geometry or {}).get(self.aperture_normal)
            ap = None if ap is None else (ap["center"], ap["normal"])
        if ap is None:
            return None
        center, normal = (np.asarray(v, float) for v in ap)
        if callable(self.passer_tangent):
            pt = self.passer_tangent(s, center)
        else:
            poly = (s.geometry or {}).get(self.passer_tangent)
            pt = None if poly is None else local_tangent(poly, center, self.ell, self.closed)
        if pt is None:
            return None
        tau, d = pt
        pen = alignment_penalty(normal, tau, d, self.sigma)
        return {"distance": float(d), "penalty": pen, "abs_cos": abs(float(np.asarray(normal) @ tau)) / float(np.linalg.norm(normal))}

    def phi(self, s: RewardState) -> float | None:
        t = self.terms(s)
        self.last_terms = t
        return None if t is None else -self.weight * t["penalty"]


register_component("aperture_alignment", AperturePassageAlignment)
