"""Rat Trap reward: frozen-shape free escape, scoring (verifier side), shaping (training only); the P&S norm as a diagnostic."""

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
    ep.solved = False
    if ep.tracker is not None:
        ep.tracker.k = ep.tracker.cps.K  # the best possible unsolved progress: every checkpoint reached
        best_unsolved = ep.score()
    else:
        best_unsolved = float(SPEC["score_progress_cap"])
    assert worst_solve == pytest.approx(0.5) and best_unsolved == pytest.approx(0.3) and worst_solve > best_unsolved


def test_penetration_scores_zero():
    ep = RatTrapEpisode(SPEC)
    P = ep.sim.node_positions()
    ep.act("grab_node", {"handle": 0, "node": int(np.argmin(P[:, 1]))})
    ep.sim.max_penetration = 1.0  # inject an audited penetration (the rig itself never forces one)
    out = ep.act("move", {"handle": 0, "dx": 0.0, "dy": 0.0, "dz": 0.001})
    assert out["violation"] and ep.done and ep.score() == 0.0


def test_oscillation_through_a_hole_earns_nothing():
    """A strand node pushed back through the frame opening and out again: the potential-based shaping telescopes to
    Phi(end) - Phi(start), which is ~0 because the cord ends near where it started, and no checkpoint is reached."""
    ep = RatTrapEpisode(SPEC, training=True)
    phi0 = ep._rs.geometry["checkpoint_potential"]
    P = ep.sim.node_positions()
    k = int(np.argmin(np.linalg.norm(P - np.array([0.04, -0.05, 0.0222]), axis=1)))  # on the right strand, in front
    ep.act("grab_node", {"handle": 0, "node": k})
    words = []
    for d in ([0, 0, 0.03], [0, 0.03, 0], [0, 0.03, 0], [0, 0.02, 0], [0, -0.02, 0], [0, -0.03, 0], [0, -0.03, 0], [0, 0, -0.03]) * 2:
        out = ep.act("move", {"handle": 0, "dx": d[0], "dy": d[1], "dz": d[2]})
        words.append(out["word"])
        assert not out["violation"]
    if phi0 is not None:
        assert ep.shaping_total == pytest.approx(ep._rs.geometry["checkpoint_potential"] - phi0, abs=1e-9)  # telescopes
        assert abs(ep.shaping_total) < 0.5  # well under one checkpoint's worth
    assert ep.progress() == 0.0 and ep.score() == 0.0
    assert set(words) == {ep.word0}  # diagnostics: a bight through a hole and back reduces away


@pytest.mark.skipif(not SOLVE.exists(), reason="no recorded solve yet (scripts/teleop_rattrap.py)")
def test_recorded_solve_scores_highest():
    from loophole.puzzles.rattrap.teleop import load_commands

    ep = RatTrapEpisode(SPEC, training=True)
    for tool, a in load_commands(SOLVE):
        if ep.done:
            break
        ep.act(tool, a)
    assert ep.solved and not ep.violation and ep.score() >= 0.5
    assert ep.progress() == 1.0  # every checkpoint of the reference reached, in order, before it came free
    assert ep.score() > float(SPEC["score_progress_cap"])  # above any non-solve
    # diagnostics only: the word empties long before the cord is free (docs/RATTRAP_DESIGN.md)
    empties = next((i for i, n in enumerate(ep.norms) if n == 0), None)
    print(f"oracle: {ep.actions} actions, score {ep.score():.4f}; ||w|| first 0 after action {empties}")
