"""The Przytycki-Sikora norm (Theorem 3.2) against the paper's own results (arXiv:math/0007134, Sections 2-3).

Generators g1, g2, ... are written A, B, C, ...; lower case is the inverse.
"""

import math

import pytest

from loophole.puzzles.rattrap.norm import distance, norm
from loophole.puzzles.rattrap.topology import free_reduce, inverse, word_inverse

G = "ABCDEFG"


def a(n: int) -> list[str]:
    """a_1 = g1; a_{n+1}: replace every g_n^{+-1} in a_n by g_{n+1} g_n^{+-1} g_{n+1}^{-1} (paper, Section 2)."""
    w = [G[0]]
    for k in range(1, n):
        gk, gn = G[k - 1], G[k]
        out = []
        for x in w:
            out += [gn, x, inverse(gn)] if x.upper() == gk else [x]
        w = out
    return w


def test_paper_presentations():
    assert "".join(a(1)) == "A"
    assert "".join(a(2)) == "BAb"  # g2 g1 g2^-1
    assert a(3) == list("CBcACbc")  # g3 g2 g3^-1 g1 g3 g2^-1 g3^-1


@pytest.mark.parametrize("n", range(1, 7))
def test_proposition_2_1(n):
    """a_n is reduced and g_n^{+-1} appears 2^(n-1) times in it."""
    w = a(n)
    assert free_reduce(w) == w
    assert sum(x.upper() == G[n - 1] for x in w) == 2 ** (n - 1)


@pytest.mark.parametrize("n", range(1, 7))
def test_proposition_2_2_prime(n):
    """The minimal number of passes through the n-th hole turning a_n into a_{n-1} is 2^(n-1):
    (a_n, a_{n-1}) = ||a_n a_{n-1}^-1|| with only g_n passable (a_0 = e)."""
    prev = a(n - 1) if n > 1 else []
    assert distance(a(n), prev, passable={G[n - 1]}) == 2 ** (n - 1)


def test_figure_10():
    """P = g1 gn g1^-1 gn g1 gn^-1 g1^-1 has connections of norm 3 and of norm 1; the minimum is 1."""
    P = list("AZaZAza")
    assert norm(P, passable={"Z"}) == 1


def test_norm_axioms_and_literal_values():
    assert norm([], {"A"}) == 0
    assert norm(list("A"), {"A"}) == 1 and norm(list("Aa"), set()) == 0
    assert norm(list("AB"), {"A"}) == math.inf  # B can be neither paired nor left unpaired
    w = list("BAbCa")
    assert norm(w, set("ABC")) == norm(word_inverse(w), set("ABC"))  # ||w^-1|| = ||w||
    # the literal ||a_n|| is NOT 2^(n-1): with only g_n passable a_n is not in the kernel (infinite) for n >= 2,
    # and with every generator passable it is 1 (only g1 is left unpaired). The paper's 2^(n-1) is (a_n, a_{n-1}).
    for n in range(2, 6):
        assert norm(a(n), {G[n - 1]}) == math.inf
        assert norm(a(n), set(G[:n])) == 1
