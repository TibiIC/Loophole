"""Rat Trap topological tracker: crossing detection and free / cyclic reduction on synthetic sequences."""

import numpy as np

from loophole.puzzles.rattrap.spec import load_spec
from loophole.puzzles.rattrap.topology import (Disc, TopologyTracker, build_discs, canonical_cyclic, cyclic_reduce, free_reduce,
                                               piercings, read_word)

X, Y, Z = np.eye(3)
SQUARE = Disc("A", "test", np.zeros(3), Y, X, Z, include=[("rect", -1, 1, -1, 1)])
HOLE = Disc("U", "hole", np.zeros(3), Z, X, Y, include=[("annulus", 0.1, 0.3)])


def test_free_reduction():
    assert free_reduce("AaB") == ["B"]
    assert free_reduce("ABba") == []
    assert free_reduce("ABbCa") == ["A", "C", "a"]
    assert free_reduce("") == []


def test_cyclic_reduction_and_canonical_form():
    assert cyclic_reduce("aBCA") == ["B", "C"]
    assert canonical_cyclic("aBCA") == canonical_cyclic("CB") == canonical_cyclic("bc")  # rotation and reversal agree
    assert canonical_cyclic("ABab") != canonical_cyclic("")  # a commutator is not trivial
    assert canonical_cyclic("Aa") == ""


def test_signed_crossing_detection():
    line = np.array([[0, -1, 0], [0, 1, 0]], float)
    assert [x[2] for x in piercings(line, [SQUARE])] == ["A"]
    assert [x[2] for x in piercings(line[::-1], [SQUARE])] == ["a"]
    outside = line + np.array([5, 0, 0])
    assert piercings(outside, [SQUARE]) == []  # crossing the plane outside the disc does not count
    through_hole = np.array([[0.2, 0, -1], [0.2, 0, 1]], float)
    through_dowel = np.array([[0.0, 0, -1], [0.0, 0, 1]], float)  # inside the annulus' inner radius (the dowel)
    assert [x[2] for x in piercings(through_hole, [HOLE])] == ["U"] and piercings(through_dowel, [HOLE]) == []


def test_crossings_are_ordered_along_the_cord_and_reduced():
    zigzag = np.array([[0, -1, 0], [0, 1, 0], [0.5, -1, 0], [0.5, 1, 0]], float)  # through, back, through
    assert [x[2] for x in piercings(zigzag, [SQUARE])] == ["A", "a", "A"]
    assert read_word(zigzag, [SQUARE], closed=False) == "A"


def test_tracker_events_on_a_moving_cord():
    tr = TopologyTracker([SQUARE], closed=False)
    before = np.array([[0, -1, 0], [0, -0.5, 0]], float)
    assert tr.update(before)["word"] == ""
    ev = tr.update(before + np.array([0, 0.75, 0]))  # the segment now straddles the disc: a new crossing
    assert ev["changed"] and ev["word"] == "A" and ev["added"] == ["A"]
    ev = tr.update(before)  # pulled back out
    assert ev["word"] == "" and ev["removed"] == ["A"] and tr.history == ["", "A", ""]


def test_alphabet_is_declared_and_start_word_is_entangled():
    spec = load_spec()
    discs = build_discs(spec)
    assert sorted(d.letter for d in discs) == sorted(spec["topology_alphabet"])
    from loophole.puzzles.rattrap.build import build  # noqa: F401  (geometry consistency only)
    from loophole.puzzles.rattrap.sim import RatTrapSim

    s = RatTrapSim(spec)
    w = read_word(s.node_positions(), discs, closed=bool(spec["cord_closed"]))
    assert w != ""  # the loop around the right dowel is not the free state
    s.close()
