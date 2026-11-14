"""Rat Trap partial credit by ordered checkpoint matching (checkpoints.py), and the score built on it."""

import numpy as np
import pytest

from loophole.puzzles.rattrap.checkpoints import DEFAULT_FILE, Checkpoints, CheckpointTracker, calibrate_eps, max_dev, select_checkpoints
from loophole.puzzles.rattrap.reward import RatTrapEpisode
from loophole.puzzles.rattrap.spec import load_spec

SPEC = load_spec()
have_file = DEFAULT_FILE.exists()
needs_file = pytest.mark.skipif(not have_file, reason="no checkpoint file (scripts/build_rattrap_reference.py)")


def _toy(K=4, n=5, gap=0.05, eps=0.01):
    start = np.zeros((n, 3))
    states = np.array([start + [gap * (k + 1), 0, 0] for k in range(K)])
    return Checkpoints(start=start, states=states, eps=eps)


# ---------------------------------------------------------------- matcher
def test_checkpoints_cannot_be_skipped():
    cps = _toy()
    t = CheckpointTracker(cps)
    t.update(cps.states[1])  # c_2 before c_1: nothing
    t.update(cps.states[3])
    assert t.k == 0
    t.update(cps.states[0])
    assert t.k == 1
    t.update(cps.states[2])  # c_3 before c_2: still 1
    assert t.k == 1
    for k in (1, 2, 3):
        t.update(cps.states[k])
    assert t.k == 4 and t.progress() == 1.0


def test_one_checkpoint_per_evaluation():
    """Two checkpoints within eps of one state: one evaluation registers only the first (strictly earlier times)."""
    start = np.zeros((3, 3))
    cps = Checkpoints(start=start, states=np.array([start + [0.05, 0, 0], start + [0.052, 0, 0]]), eps=0.01)
    t = CheckpointTracker(cps)
    t.update(start + [0.051, 0, 0])
    assert t.k == 1
    t.update(start + [0.051, 0, 0])
    assert t.k == 2


def test_max_norm_not_mean():
    """Most nodes exactly on the checkpoint and one node far off: not reached."""
    cps = _toy(K=1)
    x = cps.states[0].copy()
    x[2] += [0, 0.05, 0]
    t = CheckpointTracker(cps)
    t.update(x)
    assert t.k == 0
    assert max_dev(x, cps.states[0]) == pytest.approx(0.05)


def test_potential_is_continuous_across_a_checkpoint_and_zero_at_the_end():
    cps = _toy(K=3, gap=0.05, eps=0.01)
    t = CheckpointTracker(cps)
    near = cps.states[0] + [0.001, 0, 0]
    before = t.potential(near)
    t.update(near)
    after = t.potential(near)
    assert before == pytest.approx(-3.02, abs=1e-9) and after == pytest.approx(-3.0, abs=0.03)
    assert abs(after - before) <= cps.eps / 0.05 + 1e-9
    for k in (1, 2):
        t.update(cps.states[k])
    assert t.potential(cps.states[2]) == 0.0
    assert all(-cps.K - 1 <= v <= 0 for v in [before, after])  # bounded: no -1e6 branch anywhere


def test_select_and_calibrate():
    start = np.zeros((4, 3))
    cands = [start + [0.001, 0, 0], start + [0.05, 0, 0], start + [0.051, 0, 0], start + [0.1, 0, 0]]
    idx = select_checkpoints(start, cands, 0.02)
    assert idx == [1, 3]  # the start-like candidate and the near-duplicate are dropped
    traj = [start, start + [0.045, 0, 0], start + [0.07, 0, 0], start + [0.103, 0, 0]]
    eps = calibrate_eps(traj, [cands[i] for i in idx])
    assert eps == pytest.approx(0.005, abs=1e-6)


# ---------------------------------------------------------------- episode / score
@needs_file
def test_noop_scores_zero():
    ep = RatTrapEpisode(SPEC)
    assert ep.progress() == 0.0 and ep.score() == 0.0


@needs_file
def test_stop_early_scores_zero():
    from loophole.puzzles.rattrap.teleop import load_commands

    ep = RatTrapEpisode(SPEC)
    for tool, a in load_commands(DEFAULT_FILE.parent / "rattrap-solve.commands.jsonl")[:3]:
        ep.act(tool, a)
    assert ep.progress() == 0.0 and ep.score() == 0.0


@needs_file
def test_jiggling_near_the_start_earns_nothing():
    ep = RatTrapEpisode(SPEC, training=True)
    P = ep.sim.node_positions()
    k = int(np.argmin(np.linalg.norm(P - np.array([0.04, -0.05, 0.0222]), axis=1)))
    ep.act("grab_node", {"handle": 0, "node": k})
    for d in ([0, 0, 0.01], [0, 0, -0.01], [0.01, 0, 0], [-0.01, 0, 0], [0, -0.01, 0], [0, 0.01, 0]) * 2:
        ep.act("move", {"handle": 0, "dx": d[0], "dy": d[1], "dz": d[2]})
    assert ep.tracker.k == 0 and ep.progress() == 0.0 and ep.score() == 0.0


@needs_file
def test_start_is_not_a_checkpoint():
    cps = Checkpoints.load()
    assert max_dev(cps.start, cps.states[0]) >= 2 * cps.eps


def test_every_solve_beats_every_non_solve_on_a_grid():
    T = int(SPEC["max_actions"])
    ep = RatTrapEpisode(SPEC)
    ep.tracker = CheckpointTracker(_toy(K=10))  # the score only reads k / K
    solves, non = [], []
    for a in [*range(0, T + 1, max(1, T // 40)), T]:
        ep.actions = a
        ep.solved = True
        solves.append(ep.score())
        ep.solved = False
        for k in range(11):
            ep.tracker.k = k
            non.append(ep.score())
    ep.violation = True
    ep.solved = True
    assert ep.score() == 0.0
    assert min(solves) == pytest.approx(0.5) and max(non) == pytest.approx(0.3) and min(solves) > max(non)


@needs_file
@pytest.mark.skipif(not __import__("os").environ.get("RATTRAP_SLOW"), reason="full-budget random walk (~45 min): RATTRAP_SLOW=1")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_agent_to_the_full_budget_earns_almost_nothing(seed):
    """The adversarial probe: random grabs, moves and releases until the episode ends (T actions unless it is
    solved or crashes). Partial credit must stay near zero."""
    from loophole.puzzles.rattrap.random_agent import run

    r = run(seed)
    print(r)
    assert not r["solved"]
    assert r["checkpoints_reached"] <= 1 and r["score"] <= 0.3 / 20 + 1e-9
