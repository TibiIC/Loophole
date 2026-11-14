"""The norm ||w|| of Przytycki & Sikora (arXiv:math/0007134, Section 3, Theorem 3.2).

For a word w in the free group and a set P of "passable" generators (those whose conjugates may be inserted or
deleted at cost 1: physically, holes the cord with everything it carries can pass through), ||w|| is the minimal
number k with w = prod_{i=1..k} p_i g_{a_i}^{+-1} p_i^{-1}, g_{a_i} in P, and infinity if no such presentation
exists. Theorem 3.2: ||w|| = |w|_c, the minimum over all *connections* on w - non-crossing arcs in the upper half
plane joining letters g with g^-1 - of the number of unconnected letters, where only letters of P may be left
unconnected. Computed by an O(L^3) interval dynamic programme.

Distance (Proposition 3.1): (x, y) = ||x y^-1||, the minimal number of passes turning x into y.
"""

from __future__ import annotations

import math
from functools import lru_cache

from .topology import inverse, word_inverse

INF = math.inf


def norm(word, passable) -> float:
    """||w||: minimal number of unpaired letters over non-crossing pairings g ... g^-1, only `passable`
    generators (upper-case letters) may stay unpaired. INF when no connection exists. O(L^3)."""
    w = list(word)
    P = {g.upper() for g in passable}
    L = len(w)

    @lru_cache(maxsize=None)
    def f(i: int, j: int) -> float:  # min unpaired letters in w[i..j] (inclusive); empty -> 0
        if i > j:
            return 0
        best = (1 + f(i + 1, j)) if w[i].upper() in P else INF  # w[i] left unconnected
        for k in range(i + 1, j + 1):  # w[i] connected to w[k] (an inverse letter); arcs cannot cross
            if w[k] == inverse(w[i]):
                c = f(i + 1, k - 1)
                if c < best:
                    best = min(best, c + f(k + 1, j))
        return best

    return f(0, L - 1) if L else 0


def distance(x, y, passable) -> float:
    """(x, y) = ||x y^-1||: minimal number of passes through passable holes turning x into y."""
    return norm(list(x) + word_inverse(list(y)), passable)
