"""Topological bookkeeping: signed crossings of discs spanning the puzzle's holes.

Every hole is modelled as a disc (centre, unit normal, radius). Crossing a disc
along +normal emits the hole's lowercase letter, crossing along -normal emits
the uppercase letter. Words are freely reduced (``aA`` and ``Aa`` cancel), so
going back and forth through a hole leaves no trace.

Two kinds of words are tracked:

* ``target``: an *event* word. After every physics substep the motion of the
  target piece's centre is tested against every disc not attached to the target
  itself and the crossing symbols are appended (then reduced).
* ``rope``: a *state* word. The rope polyline is walked from its start anchor to
  its end and the signed crossings with every disc (including discs attached to
  moving objects, e.g. the target ring's own hole) are listed in order.
  ``rope_raw`` is the unreduced list, ``rope`` its free reduction.

``goal_distance(word) -> int`` is a pluggable hook: plug in a puzzle-specific
model (e.g. Przytycki–Sikora) with :func:`set_goal_distance`. The default stub
returns the reduced word length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def inverse(sym: str) -> str:
    return sym.lower() if sym.isupper() else sym.upper()


def free_reduce(word: str) -> str:
    """Freely reduce a word over letters and their inverses (case-swapped)."""
    stack: list[str] = []
    for s in word:
        if stack and stack[-1] == inverse(s):
            stack.pop()
        else:
            stack.append(s)
    return "".join(stack)


def segment_disc_crossing(p0, p1, center, normal, radius: float) -> int:
    """+1 / -1 if segment p0->p1 crosses the disc along +/- normal, else 0.

    Touching the plane without crossing is not a crossing. A segment that ends
    exactly on the plane counts (half-open convention: s0 < 0 <= s1 or reverse),
    so a point that stops on the disc and then continues is counted once.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    c = np.asarray(center, dtype=float)
    n = np.asarray(normal, dtype=float)
    s0 = float(np.dot(p0 - c, n))
    s1 = float(np.dot(p1 - c, n))
    if s0 < 0 <= s1:
        sign = 1
    elif s1 < 0 <= s0:
        sign = -1
    else:
        return 0
    t = s0 / (s0 - s1)
    x = p0 + t * (p1 - p0)
    r = x - c
    r -= np.dot(r, n) * n
    return sign if float(np.dot(r, r)) <= radius * radius else 0


def polyline_crossings(points: np.ndarray, center, normal, radius: float) -> list[tuple[float, int]]:
    """Signed crossings of a polyline with a disc, as (arclength-ish index, sign).

    Vectorised over segments; the position along the polyline is the segment
    index plus the fractional crossing parameter so crossings from several
    discs can be merged in order.
    """
    P = np.asarray(points, dtype=float)
    c = np.asarray(center, dtype=float)
    n = np.asarray(normal, dtype=float)
    s = (P - c) @ n
    s0, s1 = s[:-1], s[1:]
    up = (s0 < 0) & (s1 >= 0)
    down = (s0 >= 0) & (s1 < 0)
    idx = np.nonzero(up | down)[0]
    out = []
    for i in idx:
        t = s0[i] / (s0[i] - s1[i])
        x = P[i] + t * (P[i + 1] - P[i])
        r = x - c
        r -= np.dot(r, n) * n
        if float(np.dot(r, r)) <= radius * radius:
            out.append((float(i + t), 1 if up[i] else -1))
    return out


@dataclass
class Disc:
    name: str
    body: str  # "world" or a body name
    center_local: np.ndarray
    normal_local: np.ndarray
    radius: float
    center: np.ndarray | None = None  # world, updated each substep
    normal: np.ndarray | None = None

    def symbol(self, sign: int) -> str:
        return self.name if sign > 0 else self.name.upper()


def default_goal_distance(word: str) -> int:
    """Stub: length of the freely reduced word. Replace per puzzle."""
    return len(free_reduce(word))


_GOAL_DISTANCE: dict[str, Callable[[str], int]] = {}


def set_goal_distance(puzzle_name: str, fn: Callable[[str], int]) -> None:
    """Register a puzzle-specific goal distance (e.g. a Przytycki–Sikora model)."""
    _GOAL_DISTANCE[puzzle_name] = fn


def get_goal_distance(puzzle_name: str) -> Callable[[str], int]:
    return _GOAL_DISTANCE.get(puzzle_name, default_goal_distance)


class TopologyTracker:
    """Tracks crossing words for one episode. Pure geometry: fed positions by the env."""

    def __init__(self, discs: list[Disc], target_name: str | None = None, primary: str = "rope"):
        self.discs = discs
        self.target_name = target_name
        self.primary = primary
        self.reset()

    def reset(self) -> None:
        self.target_word = ""
        self.rope_raw = ""
        self._last_target: np.ndarray | None = None

    # -- disc poses -----------------------------------------------------
    def set_disc_pose(self, disc: Disc, pos: np.ndarray, mat: np.ndarray) -> None:
        disc.center = pos + mat @ disc.center_local
        n = mat @ disc.normal_local
        disc.normal = n / np.linalg.norm(n)

    # -- updates --------------------------------------------------------
    def update_target(self, pos: np.ndarray) -> None:
        """Call after every substep with the target centre (event word)."""
        pos = np.asarray(pos, dtype=float).copy()
        if self._last_target is not None:
            for d in self.discs:
                if d.body == self.target_name:
                    continue
                s = segment_disc_crossing(self._last_target, pos, d.center, d.normal, d.radius)
                if s:
                    self.target_word = free_reduce(self.target_word + d.symbol(s))
        self._last_target = pos

    def update_rope(self, nodes: np.ndarray) -> None:
        """Recompute the rope state word from its node polyline (start -> end)."""
        events: list[tuple[float, str]] = []
        for d in self.discs:
            for pos, sign in polyline_crossings(nodes, d.center, d.normal, d.radius):
                events.append((pos, d.symbol(sign)))
        events.sort()
        self.rope_raw = "".join(s for _, s in events)

    # -- outputs --------------------------------------------------------
    @property
    def words(self) -> dict[str, str]:
        return {"rope": free_reduce(self.rope_raw), "rope_raw": self.rope_raw, "target": self.target_word}

    @property
    def word(self) -> str:
        return self.words[self.primary]

    def summary(self) -> dict:
        return {
            **self.words,
            "primary": self.primary,
            "holes": {d.name: d.body for d in self.discs},
        }
