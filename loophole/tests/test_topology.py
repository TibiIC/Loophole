"""Milestone 5: crossing detection, free reduction, tracker, potential-based shaping."""

import numpy as np
import pytest

from loophole.config import load_config
from loophole.reward.components import RewardComposer, RewardState
from loophole.reward.topology import (
    Disc,
    TopologyTracker,
    default_goal_distance,
    free_reduce,
    get_goal_distance,
    polyline_crossings,
    segment_disc_crossing,
    set_goal_distance,
)
from loophole.sim.env import LoopholeEnv

from .conftest import FIXTURES, move_to

Z = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "word,reduced",
    [("", ""), ("aA", ""), ("Aa", ""), ("abBA", ""), ("abAB", "abAB"), ("aAb", "b"), ("aabBA", "a"), ("abcCBd", "ad"), ("AAaa", "")],
)
def test_free_reduce(word, reduced):
    assert free_reduce(word) == reduced


def test_segment_crossing_signs_and_radius():
    c = np.zeros(3)
    assert segment_disc_crossing([0, 0, -1], [0, 0, 1], c, Z, 0.1) == 1
    assert segment_disc_crossing([0, 0, 1], [0, 0, -1], c, Z, 0.1) == -1
    assert segment_disc_crossing([0.2, 0, -1], [0.2, 0, 1], c, Z, 0.1) == 0  # outside the disc
    assert segment_disc_crossing([0, 0, 1], [0, 0, 2], c, Z, 0.1) == 0  # no crossing
    assert segment_disc_crossing([-1, 0, 0.5], [1, 0, 0.5], c, Z, 0.1) == 0  # parallel
    # oblique crossing hits the disc plane at x = 0.05 -> inside
    assert segment_disc_crossing([0, 0, -1], [0.1, 0, 1], c, Z, 0.1) == 1


def test_crossing_counted_once_when_stopping_on_plane():
    c = np.zeros(3)
    a, b, d = [0, 0, -1], [0, 0, 0], [0, 0, 1]
    assert segment_disc_crossing(a, b, c, Z, 0.1) + segment_disc_crossing(b, d, c, Z, 0.1) == 1
    # touching the plane from above and going back up is not a crossing
    assert segment_disc_crossing([0, 0, 1], [0, 0, 0], c, Z, 0.1) + segment_disc_crossing([0, 0, 0], [0, 0, 1], c, Z, 0.1) == 0


def _tracker():
    discs = [
        Disc("a", "world", np.array([0.0, 0, 0]), Z, 0.1),
        Disc("b", "world", np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), 0.1),
    ]
    t = TopologyTracker(discs, target_name="ring")
    for d in discs:
        t.set_disc_pose(d, np.zeros(3), np.eye(3))
    return t


def _feed(t, pts, n=10):
    pts = [np.asarray(p, float) for p in pts]
    for p, q in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, n + 1)[1:]:
            t.update_target(p + s * (q - p))


def test_tracker_event_word_reduces_back_and_forth():
    t = _tracker()
    t.update_target([0, 0, -0.5])
    _feed(t, [[0, 0, -0.5], [0, 0, 0.5], [0, 0, -0.5]])  # up through a, back down
    assert t.target_word == ""
    _feed(t, [[0, 0, -0.5], [0, 0, 0.5]])  # up through a
    assert t.target_word == "a"
    _feed(t, [[0, 0, 0.5], [0.5, 0, 0], [1.5, 0, 0]])  # then +x through b
    assert t.target_word == "ab"
    _feed(t, [[1.5, 0, 0], [0.5, 0, 0]])  # back through b
    assert t.target_word == "a"


def test_rope_state_word_orders_crossings_along_rope():
    t = _tracker()
    # a rope that goes down through a, up again, then crosses b in -x
    rope = np.array([[0, 0, 1], [0, 0, -1], [0.02, 0, -1], [0.02, 0, 1], [2, 0, 0], [0.5, 0, 0]])
    t.update_rope(rope)
    assert t.rope_raw == "AaB" and t.words["rope"] == "B"


def test_polyline_crossings_positions():
    rope = np.array([[0, 0, 1], [0, 0, -1], [0, 0, 1]])
    out = polyline_crossings(rope, np.zeros(3), Z, 0.1)
    assert [s for _, s in out] == [-1, 1] and out[0][0] == pytest.approx(0.5)


def test_moving_disc_follows_its_body():
    d = Disc("r", "ring", np.zeros(3), Z, 0.05)
    t = TopologyTracker([d])
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # body z axis -> world -y
    t.set_disc_pose(d, np.array([1.0, 2.0, 3.0]), R)
    assert np.allclose(d.center, [1, 2, 3]) and np.allclose(d.normal, [0, -1, 0])


def _shaping_composer(gamma=1.0):
    cfg = load_config(overrides={"reward": {"components": [{"type": "topology_shaping", "weight": 1.0, "params": {"gamma": gamma}}]}}).reward
    return RewardComposer.from_config(cfg)


def _rs(word, n):
    return RewardState(puzzle="synthetic", actions_used=n, max_actions=10, success=False, violations=0, done=False, topology_word=word)


def test_potential_shaping_oscillation_earns_nothing():
    comp = _shaping_composer()
    comp.reset(_rs("ab", 0))
    for i, w in enumerate(["a", "ab", "a", "ab"], 1):  # oscillate across one hole
        comp.step(_rs(w, i))
    comp.terminal(_rs("ab", 4))
    assert comp.score() == 0.0
    comp.reset(_rs("ab", 0))
    comp.step(_rs("a", 1))
    comp.step(_rs("", 2))
    comp.terminal(_rs("", 2))
    assert comp.score() == pytest.approx(1.0)  # (d0 - dT) / d0


def test_goal_distance_is_pluggable():
    assert get_goal_distance("synthetic") is default_goal_distance
    set_goal_distance("synthetic", lambda w: 7 if w else 0)
    try:
        comp = _shaping_composer()
        comp.reset(_rs("x", 0))
        assert comp.step(_rs("", 1)) == pytest.approx(1.0)
    finally:
        from loophole.reward import topology

        topology._GOAL_DISTANCE.pop("synthetic")


def test_env_target_word_through_real_hole():
    env = LoopholeEnv(FIXTURES / "hollow.yaml")
    env.act("grab", {"handle": 0, "target": "small"})
    move_to(env, 0, [-0.15, 0, 0.245])
    move_to(env, 0, [0.15, 0, 0.245])
    assert env.tracker.target_word == "a"
    move_to(env, 0, [-0.15, 0, 0.245])
    assert env.tracker.target_word == ""
    assert env.get_state()["topology"]["target"] == ""


def test_shaping_enabled_in_env_never_beats_a_solve():
    cfg = load_config(overrides={"reward": {"components": [
        {"type": "base", "weight": 1.0, "params": {"alpha": 0.5}},
        {"type": "topology_shaping", "weight": 0.2, "enabled": True, "params": {"gamma": 1.0}},
    ]}})
    from loophole.reference.replay import replay
    from loophole.reference.scripted import SCRIPTS

    env = LoopholeEnv("easy", config=cfg)
    final = replay(env, SCRIPTS["easy"][:5])
    assert 0.0 <= final["score"] <= 0.2
