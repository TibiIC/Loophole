"""Ported pieces (REFERENCES.md [HO26]): the PotentialShaping scaffold's terminal / truncation rule, the
aperture-passage alignment term's symmetries, local_tangent, and the swept-motion certificate."""

import numpy as np
import pytest

from loophole.reward.alignment import AperturePassageAlignment, alignment_penalty, local_tangent, ring_normal
from loophole.reward.components import PotentialShaping, RewardState
from loophole.sim.certify import polyline_motion_safe, rigid_motion_safe
from loophole.sim.geometry import quat_from_axis_angle, quat_to_mat


def _rs(**kw):
    base = dict(puzzle="x", actions_used=1, max_actions=10, success=False, violations=0, done=False)
    return RewardState(**{**base, **kw})


class _Phi(PotentialShaping):
    name = "phi_test"

    def phi(self, s):
        return -float(s.topology_word.count("a"))


# ---------------------------------------------------------------- scaffold
def test_terminal_potential_is_zero_and_truncation_keeps_it():
    c = _Phi(gamma=0.99, lam=1.0)
    prev = _rs(topology_word="aaa")
    assert c.step(prev, _rs(topology_word="aa"), {}) == pytest.approx(0.99 * -2 + 3)
    assert c.step(prev, _rs(topology_word="aa", success=True, done=True), {}) == pytest.approx(3.0)  # solved: Phi' = 0
    assert c.step(prev, _rs(topology_word="aa", violations=1, done=True), {}) == pytest.approx(3.0)  # violation: Phi' = 0
    trunc = _rs(topology_word="aa", actions_used=10, done=True)  # budget exhausted: external truncation
    assert trunc.truncated and not trunc.terminated
    assert c.step(prev, trunc, {}) == pytest.approx(0.99 * -2 + 3)  # Phi' retained (the learner bootstraps)
    assert _Phi(gamma=0.99, lam=0.5).step(prev, _rs(topology_word="aa"), {}) == pytest.approx(0.5 * (0.99 * -2 + 3))


def test_discounted_shaping_depends_only_on_the_endpoints():
    """gamma < 1: sum_t gamma^t F_t = gamma^T Phi(s_T) - Phi(s_0). An oscillation that returns to the start gains
    nothing over standing still for the same number of steps (the oscillation property, for gamma < 1)."""
    g = 0.99
    c = _Phi(gamma=g)

    def ret(words):
        states = [_rs(topology_word=w) for w in words]
        return sum(g**t * c.step(a, b, {}) for t, (a, b) in enumerate(zip(states, states[1:])))

    osc = ret(["aa", "a", "aaa", "a", "aa"])
    still = ret(["aa"] * 5)
    assert osc == pytest.approx(still) == pytest.approx(g**4 * -2 + 2)
    g1 = _Phi(gamma=1.0)
    states = [_rs(topology_word=w) for w in ["aa", "a", "aaa", "a", "aa"]]
    assert sum(g1.step(a, b, {}) for a, b in zip(states, states[1:])) == pytest.approx(0.0)  # gamma = 1: telescopes


def test_unavailable_potential_gives_no_shaping():
    class _None(PotentialShaping):
        def phi(self, s):
            return None

    assert _None().step(_rs(), _rs(), {}) == 0.0


# ---------------------------------------------------------------- alignment
def _rot(axis, ang):
    return quat_to_mat(quat_from_axis_angle(axis, ang))


def test_alignment_invariant_under_normal_flip_and_spin_about_the_axis():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = _rot(rng.normal(size=3), rng.uniform(0, np.pi))
        tau = rng.normal(size=3)
        d = rng.uniform(0, 0.05)
        n = ring_normal(R)
        ref = alignment_penalty(n, tau, d, 0.02)
        assert alignment_penalty(-n, tau, d, 0.02) == pytest.approx(ref, abs=1e-12)  # n -> -n
        assert alignment_penalty(n, -tau, d, 0.02) == pytest.approx(ref, abs=1e-12)  # tau -> -tau (unoriented cord)
        for th in rng.uniform(-np.pi, np.pi, 4):
            spun = R @ _rot([0, 0, 1], th)  # the ring spun about its own axis: physically the same state
            assert np.allclose(ring_normal(spun), n)
            assert alignment_penalty(ring_normal(spun), tau, d, 0.02) == pytest.approx(ref, abs=1e-12)
        flipped = R @ _rot([1, 0, 0], np.pi)  # the ring turned over: the same plane, normal reversed
        assert alignment_penalty(ring_normal(flipped), tau, d, 0.02) == pytest.approx(ref, abs=1e-12)


def test_alignment_component_symmetries_and_shape():
    """Through the component: a cord threading a hole along its axis vs across it, with the aperture normal
    flipped, the hole spun about its axis, and the cord's node order reversed."""
    axis = np.array([0.0, 0.0, 1.0])
    along = np.c_[np.zeros(9), np.zeros(9), np.linspace(-0.02, 0.02, 9)]
    across = np.c_[np.linspace(-0.02, 0.02, 9), np.zeros(9), np.full(9, 0.002)]
    comp = AperturePassageAlignment("hole", "cord", gate_distance=0.01, weight=0.25, gamma=1.0)

    def phi(normal, cord):
        return comp.phi(_rs(geometry={"hole": {"center": np.zeros(3), "normal": normal}, "cord": cord}))

    assert phi(axis, along) == pytest.approx(0.0, abs=1e-12)  # aligned: no penalty
    assert phi(axis, across) == pytest.approx(-0.25 * np.exp(-(0.002 / 0.01) ** 2))
    for cord in (along, across):
        ref = phi(axis, cord)
        assert phi(-axis, cord) == pytest.approx(ref, abs=1e-12)
        assert phi(axis, cord[::-1]) == pytest.approx(ref, abs=1e-12)
        for th in (0.3, 1.7, -2.4):
            assert phi(_rot(axis, th) @ axis, cord) == pytest.approx(ref, abs=1e-12)
    far = across + np.array([0.0, 0.2, 0.0])
    assert phi(axis, far) == pytest.approx(0.0, abs=1e-9)  # gated off far from the aperture


def test_local_tangent_survives_antiparallel_strands():
    """A doubled cord through a hole: two strands running in opposite directions. A plain average of segment
    directions cancels; the sign-aligned average keeps the common line."""
    up = np.c_[np.full(6, -0.001), np.zeros(6), np.linspace(-0.01, 0.01, 6)]
    loop = np.vstack([up, up[::-1] + np.array([0.002, 0.0, 0.0])])
    tau, d = local_tangent(loop, np.zeros(3), length_scale=0.01)
    assert abs(tau[2]) > 0.99 and d == pytest.approx(0.001, abs=1e-9)
    seg = np.diff(loop, axis=0)
    plain = np.sum(seg / np.linalg.norm(seg, axis=1, keepdims=True), axis=0)  # the unaligned average
    assert abs(plain[2]) < 1e-9  # the two strands cancel exactly along the axis


# ---------------------------------------------------------------- certificate
WALL = 0.001  # a thin wall: the slab |x| <= WALL


def _point_gap(r):
    return lambda c, need: (abs(float(np.asarray(c)[0])) - WALL - r) >= need


def test_rigid_certificate_catches_a_jump_through_a_thin_wall():
    r = 0.005
    gap = _point_gap(r)
    p0, q0 = np.array([-0.01, 0, 0]), np.array([1.0, 0, 0, 0])
    jump = np.array([0.02, 0, 0])  # both ends are clear of the wall: a per-state audit sees nothing
    assert gap(p0, 0.0) and gap(p0 + jump, 0.0)
    assert not rigid_motion_safe(gap, p0, q0, jump, np.zeros(3), outer_radius=r)
    assert rigid_motion_safe(gap, p0, q0, np.array([0.0, 0.02, 0.0]), np.zeros(3), outer_radius=r)  # parallel to it
    assert rigid_motion_safe(gap, p0, q0, np.array([0.003, 0, 0]), np.zeros(3), outer_radius=r)  # stops short


def test_rigid_certificate_bounds_rotation_sweep():
    """A bar of half-length 0.02 about its centre at x = -0.012, swinging its tip through the wall: the bound uses
    the outer radius, so a pure rotation that sweeps the tip across x = 0 is not certified."""
    L = 0.02

    def gap(c, need):
        c = np.asarray(c)
        R = quat_to_mat(c[3:])
        tips = c[:3] + np.array([[L, 0, 0], [-L, 0, 0]]) @ R.T
        xs = np.linspace(tips[0, 0], tips[1, 0], 41)
        dist = np.min(np.abs(xs)) - WALL if np.all(np.sign(xs) == np.sign(xs[0])) else -1.0
        return dist >= need

    p, q = np.array([-0.012, 0, 0]), quat_from_axis_angle([0, 0, 1], np.pi / 2)  # bar along y: clear of the wall
    assert gap(np.r_[p, q], 0.0)
    assert not rigid_motion_safe(gap, p, q, np.zeros(3), np.array([0, 0, np.pi / 2]), outer_radius=L)  # tip crosses
    assert rigid_motion_safe(gap, p, q, np.zeros(3), np.array([0, 0, 0.2]), outer_radius=L)  # small swing: clear


def test_polyline_certificate_and_slack():
    nodes0 = np.c_[np.full(5, -0.004), np.linspace(0, 0.01, 5), np.zeros(5)]

    def gap(c, need):
        return float(np.min(np.abs(np.asarray(c)[:, 0])) - WALL) >= need

    assert not polyline_motion_safe(gap, nodes0, nodes0 + [0.008, 0, 0])  # the whole chain hops the wall
    moved = nodes0.copy()
    moved[2, 0] += 0.0028  # one node nudged to 0.2 mm from the wall: certified
    assert polyline_motion_safe(gap, nodes0, moved)
    into = nodes0.copy()
    into[:, 0] += 0.0035  # 0.5 mm into the wall's clearance zone: only certified with slack >= 0.5 mm
    assert not polyline_motion_safe(gap, nodes0, into)
    assert polyline_motion_safe(gap, nodes0, into, slack=0.0006)


def test_topology_progress_in_score_is_zero_after_a_violation_and_without_progress():
    """topology_shaping's terminal value counts towards the score when enabled: it is the endpoint progress
    (d_start - d_T) / max(1, d_start), 0 after a violation (it used to keep the progress made before the
    violation), and 0 when the start is already at goal distance 0."""
    from loophole.config import RewardConfig
    from loophole.reward.components import RewardComposer

    cfg = RewardConfig.model_validate({"components": [{"type": "base", "weight": 1.0, "params": {"alpha": 0.5}},
                                                      {"type": "topology_shaping", "weight": 0.2, "params": {"gamma": 1.0}}]})

    def score(words, violation=False, puzzle="easy"):
        comp = RewardComposer.from_config(cfg)
        kw = dict(puzzle=puzzle, max_actions=60, success=False)
        comp.reset(_rs(actions_used=0, topology_word=words[0], **kw))
        for i, w in enumerate(words[1:], 1):
            last = i == len(words) - 1
            comp.step(_rs(actions_used=i, topology_word=w, violations=int(violation and last), done=last, **kw))
        comp.terminal(_rs(actions_used=len(words) - 1, topology_word=words[-1], violations=int(violation), done=True, **kw))
        return comp.score()

    assert score(["ab", "a", ""]) == pytest.approx(0.2)  # unsolved, full topological progress: the 0.2 cap
    assert score(["ab", "a", ""], violation=True) == 0.0
    assert score(["", "", ""]) == 0.0  # no progress possible, none credited
