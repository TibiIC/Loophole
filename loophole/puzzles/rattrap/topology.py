"""Topological state of the Rat Trap cord: signed crossings of the cord through spanning discs, freely reduced.

Each independent passage of the rigid structure is spanned by a disc and named by a generator letter
(declared in rattrap.yaml: `topology_alphabet`). A crossing of the cord through a disc in its positive
direction reads as the upper-case letter, in the negative direction as the lower-case (inverse) letter.

Reading: walk along the cord and list its signed crossings in order, then reduce freely (x x^-1 -> e).
For a closed loop the word is a cyclic word: it is also cyclically reduced, and put in a canonical form
(lexicographically smallest rotation of the word or of its inverse, since a loop has no start and no
direction). The cord cannot pass through a disc's boundary (solid structure), so this reading changes exactly
when a crossing is created or removed, i.e. it equals appending signed crossing events and reducing.

Approach: crossing words of a flexible string through the holes of a rigid frame, after Przytycki & Sikora,
"Topological insights from the Chinese Rings" (Proc. AMS 130(3), 2002; arXiv:math/0007134).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def inverse(letter: str) -> str:
    return letter.lower() if letter.isupper() else letter.upper()


def free_reduce(word) -> list[str]:
    out: list[str] = []
    for x in word:
        if out and out[-1] == inverse(x):
            out.pop()
        else:
            out.append(x)
    return out


def cyclic_reduce(word) -> list[str]:
    w = free_reduce(word)
    while len(w) >= 2 and w[0] == inverse(w[-1]):
        w = w[1:-1]
    return w


def word_inverse(word) -> list[str]:
    return [inverse(x) for x in reversed(word)]


def canonical_cyclic(word) -> str:
    """Canonical form of a closed loop's word: cyclically reduced, smallest rotation of it or of its inverse."""
    w = cyclic_reduce(word)
    if not w:
        return ""
    cands = []
    for v in (w, word_inverse(w)):
        cands += ["".join(v[i:] + v[:i]) for i in range(len(v))]
    return min(cands)


@dataclass
class Disc:
    """A planar disc: origin, positive normal, in-plane axes (u, v), and a shape in (u, v) coordinates:
    `include` / `exclude` lists of ("rect", u0, u1, v0, v1) | ("circle", r) | ("annulus", r_in, r_out)."""

    letter: str
    name: str
    origin: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    include: list = field(default_factory=list)
    exclude: list = field(default_factory=list)

    def _in(self, shape, a: float, b: float) -> bool:
        kind = shape[0]
        if kind == "rect":
            return shape[1] <= a <= shape[2] and shape[3] <= b <= shape[4]
        r = (a * a + b * b) ** 0.5
        if kind == "circle":
            return r <= shape[1]
        if kind == "annulus":
            return shape[1] <= r <= shape[2]
        raise ValueError(f"unknown disc shape {kind!r}")

    def contains(self, p: np.ndarray) -> bool:
        d = np.asarray(p, float) - self.origin
        a, b = float(d @ self.u), float(d @ self.v)
        return any(self._in(s, a, b) for s in self.include) and not any(self._in(s, a, b) for s in self.exclude)


def piercings(nodes: np.ndarray, discs: list[Disc]) -> list[tuple[int, float, str]]:
    """Signed crossings of the polyline `nodes` through the discs, in order along the cord:
    (segment index, parameter along the segment, letter)."""
    out = []
    P = np.asarray(nodes, float)
    for d in discs:
        s = (P - d.origin) @ d.normal
        pos = s >= 0  # a node exactly on the plane counts as the positive side (consistent between frames)
        for i in np.flatnonzero(pos[:-1] != pos[1:]):
            t = float(s[i] / (s[i] - s[i + 1]))
            p = P[i] + t * (P[i + 1] - P[i])
            if d.contains(p):
                out.append((int(i), t, d.letter if s[i + 1] > s[i] else d.letter.lower()))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def read_word(nodes: np.ndarray, discs: list[Disc], closed: bool) -> str:
    letters = [x[2] for x in piercings(nodes, discs)]
    return canonical_cyclic(letters) if closed else "".join(free_reduce(letters))


class TopologyTracker:
    """Live tracker: the reduced word of the current cord shape, plus the crossing events since the last update
    (letters that appeared / disappeared in the unreduced reading)."""

    def __init__(self, discs: list[Disc], closed: bool):
        self.discs = discs
        self.closed = closed
        self.word = ""
        self._raw: list[str] = []
        self.history: list[str] = []  # reduced words, each time the word changed

    def update(self, nodes: np.ndarray) -> dict:
        raw = [x[2] for x in piercings(nodes, self.discs)]
        word = canonical_cyclic(raw) if self.closed else "".join(free_reduce(raw))
        events = {"added": _multiset_diff(raw, self._raw), "removed": _multiset_diff(self._raw, raw)}
        changed = word != self.word
        self._raw, self.word = raw, word
        if changed or not self.history:
            self.history.append(word)
        return {"word": word, "changed": changed, **events}


def _multiset_diff(a, b) -> list[str]:
    from collections import Counter

    return sorted((Counter(a) - Counter(b)).elements())


def build_discs(spec) -> list[Disc]:
    """The discs named in the YAML's `topology_alphabet`, with geometry from the spec's landmarks."""
    r, lm = spec.raw, spec.landmarks
    W = r["frame_inner_width"]
    z0, ztop = lm["inner_floor_z"], lm["inner_ceiling_z"]
    xL, xR = lm["dowel_left_x"], lm["dowel_right_x"]
    z_up, z_lo = lm["post_upper_z"], lm["post_lower_z"]
    ps = r["post_thickness"] / 2
    X, Y, Z = np.eye(3)
    B_rect = ("rect", xL, W / 2, z_up + ps, ztop)  # above the upper post, right of the left dowel (in x, z)
    C_rect = ("rect", -W / 2, xR, z0, z_lo - ps)  # below the lower post, left of the right dowel
    hex_corner = r["hole_radius"] / np.cos(np.pi / int(r["hole_sides"]))
    geometry = {
        "frame_main": dict(origin=np.zeros(3), normal=Y, u=X, v=Z, include=[("rect", -W / 2, W / 2, z0, ztop)], exclude=[B_rect, C_rect]),
        "frame_upper_right": dict(origin=np.zeros(3), normal=Y, u=X, v=Z, include=[B_rect]),
        "frame_lower_left": dict(origin=np.zeros(3), normal=Y, u=X, v=Z, include=[C_rect]),
        "hole_upper": dict(origin=np.array([xL, 0, z_up]), normal=Z, u=X, v=Y, include=[("annulus", r["dowel_radius"], hex_corner)]),
        "hole_lower": dict(origin=np.array([xR, 0, z_lo]), normal=Z, u=X, v=Y, include=[("annulus", r["dowel_radius"], hex_corner)]),
        "ring": dict(origin=np.array([xR, 0, z_up]), normal=X, u=Y, v=Z,
                     include=[("circle", r["ring_major_radius"] - r["ring_tube_radius"])], exclude=[("rect", -ps, ps, -ps, ps)]),
    }
    discs = []
    for letter, d in r["topology_alphabet"].items():
        if not (len(letter) == 1 and letter.isalpha() and letter.isupper()):
            raise ValueError(f"topology_alphabet: generator {letter!r} must be one upper-case letter")
        g = geometry.get(d["disc"])
        if g is None:
            raise ValueError(f"topology_alphabet: unknown disc {d['disc']!r} (known: {sorted(geometry)})")
        discs.append(Disc(letter=letter, name=d["disc"], **{k: (np.asarray(v, float) if isinstance(v, np.ndarray) else v) for k, v in g.items()}))
    return discs
