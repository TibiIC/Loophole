"""Rat Trap reward: P&S norm potential, frozen-shape free escape, scoring (verifier side), shaping (training only)."""

from pathlib import Path

import numpy as np
import pytest

from loophole.puzzles.rattrap.reward import FrozenEscape, RatTrapEpisode, check_passability, word_norm
from loophole.puzzles.rattrap.spec import load_spec

SPEC = load_spec()
SOLVE = Path(__file__).resolve().parents[1] / "reference" / "rattrap-solve.commands.jsonl"


@pytest.fixture(scope="module")
def ep0():
    return RatTrapEpisode(SPEC)


def test_start_norm_is_finite_and_positive(ep0):
    assert ep0.word0 != "" and 0 < ep0.norm0 < float("inf")
    assert word_norm("", SPEC) == 0


def test_passability_is_consistent_with_the_clearance_table():
    assert check_passability(SPEC) == []
    alpha = {g: {**d, "passable": True} for g, d in SPEC["topology_alphabet"].items()}  # declare every hole passable
    wrong = load_spec(overrides={"mouse_enabled": True, "topology_alphabet": alpha})  # ... but the mouse blocks U, L, R
    assert sorted(x[0] for x in check_passability(wrong)) == ["L", "R", "U"]


def test_frozen_escape(ep0):
    fe: FrozenEscape = ep0.escape
    P = ep0.sim.node_positions()
    assert fe.escape_direction(P) is None  # the loop encircles the right dowel
    assert fe.escape_direction(P + np.array([0.0, -0.03, 0.0])) is not None  # the same shape, off the dowel
    u = fe.escape_direction(P + np.array([0.0, -0.6, 0.0]))
    assert u is not None and abs(np.linalg.norm(u) - 1) < 1e-9


def test_noop_scores_zero():
    ep = RatTrapEpisode(SPEC)
    assert ep.score() == 0.0 and not ep.solved


def test_score_ranges():
    ep = RatTrapEpisode(SPEC)
    ep.solved, ep.actions = True, ep.T
    worst_solve = ep.score()
    ep.solved, ep.norms = False, [ep.norm0, 0.0]  # the best possible unsolved progress
    best_unsolved = ep.score()
    assert worst_solve == pytest.approx(0.5) and best_unsolved == pytest.approx(0.3) and worst_solve > best_unsolved


def test_penetration_scores_zero():
    ep = RatTrapEpisode(SPEC)
    P = ep.sim.node_positions()
    ep.act("grab_node", {"handle": 0, "node": int(np.argmin(P[:, 1]))})
    ep.sim.max_penetration = 1.0  # inject an audited penetration (the rig itself never forces one)
    out = ep.act("move", {"handle": 0, "dx": 0.0, "dy": 0.0, "dz": 0.001})
    assert out["violation"] and ep.done and ep.score() == 0.0


def test_oscillation_through_a_hole_earns_nothing():
    """A strand node pushed back through the frame opening and out again: the word is unchanged, so the potential-based
    shaping nets zero and the progress (min ||w_t||) does not move."""
    ep = RatTrapEpisode(SPEC, training=True)
    P = ep.sim.node_positions()
    k = int(np.argmin(np.linalg.norm(P - np.array([0.04, -0.05, 0.0222]), axis=1)))  # on the right strand, in front
    ep.act("grab_node", {"handle": 0, "node": k})
    words = []
    for d in ([0, 0, 0.03], [0, 0.03, 0], [0, 0.03, 0], [0, 0.02, 0], [0, -0.02, 0], [0, -0.03, 0], [0, -0.03, 0], [0, 0, -0.03]) * 2:
        out = ep.act("move", {"handle": 0, "dx": d[0], "dy": d[1], "dz": d[2]})
        words.append(out["word"])
        assert not out["violation"]
    assert ep.shaping_total == pytest.approx(0.0, abs=1e-9)
    assert ep.progress() == 0.0 and ep.score() == 0.0
    assert set(words) == {ep.word0}  # a bight through a hole and back reduces away


@pytest.mark.skipif(not SOLVE.exists(), reason="no recorded solve yet (scripts/teleop_rattrap.py)")
def test_recorded_solve_scores_highest():
    from loophole.puzzles.rattrap.teleop import load_commands

    ep = RatTrapEpisode(SPEC, training=True)
    for tool, a in load_commands(SOLVE):
        if ep.done:
            break
        ep.act(tool, a)
    assert ep.solved and ep.score() >= 0.5
    # ||w_t|| along the solve: report where it is not non-increasing (the doc explains each such step)
    rises = [i for i in range(1, len(ep.norms)) if ep.norms[i] > ep.norms[i - 1]]
    print(f"norm sequence changes: {[(i, ep.norms[i]) for i in range(1, len(ep.norms)) if ep.norms[i] != ep.norms[i - 1]]}; rises at {rises}")
