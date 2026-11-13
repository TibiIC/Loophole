"""Milestone 1: model building and physics invariants."""

import mujoco
import numpy as np

from loophole.sim.builder import build_model
from loophole.sim.spec import load_puzzle, resolve_puzzle

from .conftest import EASY_SOLUTION, make_env, move_to, run


def adjacent_distances(env):
    nodes = env.rope_nodes()
    return np.linalg.norm(np.diff(nodes, axis=0), axis=1)


def test_mjcf_builds_from_yaml():
    built = build_model(load_puzzle(resolve_puzzle("easy")))
    m = mujoco.MjModel.from_xml_string(built.xml)
    assert m.nmocap == 2
    # one weld per (handle, graspable body): ring + every rope segment
    assert len(built.weld_names) == 2 * (1 + len(built.rope.body_names))
    assert all(m.eq_active0[m.equality(n).id] == 0 for n in built.weld_names.values())
    # ball joints along the rope, a free joint on the ring
    assert sum(m.jnt_type == mujoco.mjtJoint.mjJNT_BALL) == len(built.rope.body_names)
    assert sum(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1


def test_initial_state_is_stable_and_threaded(easy_env):
    st = easy_env.get_state()
    assert st["violations"] == 0
    # rope passes through ring (r) and loop (a) and back
    assert st["topology"]["rope_raw"] == "RAar"
    ring = np.array(st["objects"]["ring"]["pos"])
    assert abs(ring[2] - 0.2627) < 0.01  # resting on top of the loop


def test_rope_cannot_stretch_under_extreme_pull():
    # Disable the stall safety and stiffen the weld to yank as hard as possible.
    env = make_env(rig={"lag_limit": 10.0, "angle_lag_limit": 10.0, "weld_solref": [0.004, 1.0]})
    seg = np.array(env.built.rope.seg_lengths)
    env.act("grab", {"handle": 0, "target": "rope", "segment": 10})
    for _ in range(8):
        if env.done:
            break
        env.act("move", {"handle": 0, "dx": 0.1, "dy": -0.1})
        # joint coordinates: segment lengths are exact at all times
        assert np.max(np.abs(adjacent_distances(env) - seg)) < 1e-9
    # the only soft link is the closing connect; overpowering it is caught by the audit
    kinds = {v["kind"] for v in env.audit.summary()["records"]}
    assert env.audit.count == 0 or kinds & {"anchor_drift", "penetration", "speed"}


def test_rope_pull_with_default_rig_keeps_rope_intact(easy_env):
    env = easy_env
    env.act("grab", {"handle": 0, "target": "rope", "segment": 20})
    for _ in range(6):
        r = env.act("move", {"handle": 0, "dx": -0.1, "dz": 0.05})
    assert r["blocked"]  # the rope is anchored: the handle stalls instead of tearing it
    assert env.audit.count == 0
    assert np.max(np.abs(adjacent_distances(env) - np.array(env.built.rope.seg_lengths))) < 1e-9


def test_rope_passes_through_capsule_loop(easy_env):
    env = easy_env
    run(env, EASY_SOLUTION[:5])  # lift the bight out of the loop
    assert env.tracker.rope_raw == ""  # rope no longer crosses the loop or the ring
    assert env.audit.count == 0


def test_small_object_passes_through_hollow_loop(hollow_env):
    env = hollow_env
    env.act("grab", {"handle": 0, "target": "small"})
    move_to(env, 0, [-0.15, 0, 0.245])
    assert env.tracker.target_word == ""
    for _ in range(3):
        env.act("move", {"handle": 0, "dx": 0.1})
    x = env.data.xpos[env.model.body("small").id][0]
    assert x > 0.1, "small ball should pass through the hollow capsule loop"
    assert env.tracker.target_word == "a"
    assert env.audit.count == 0


def test_large_object_cannot_pass_through_loop(hollow_env):
    env = hollow_env
    env.act("grab", {"handle": 0, "target": "large"})
    move_to(env, 0, [-0.15, 0, 0.245])
    blocked = []
    for _ in range(4):
        blocked.append(env.act("move", {"handle": 0, "dx": 0.1})["blocked"])
    x = env.data.xpos[env.model.body("large").id][0]
    # seated against the loop: centre at -sqrt((0.05 + 0.008)^2 - 0.045^2) = -0.0366
    assert x < -0.03 and any(blocked)
    assert env.tracker.target_word == ""
    assert env.audit.count == 0


def test_solid_object_cannot_pass_through_wall(hollow_env):
    env = hollow_env
    env.act("grab", {"handle": 0, "target": "small"})
    move_to(env, 0, [-0.05, 0, 0.12])
    for _ in range(4):
        env.act("move", {"handle": 0, "dy": 0.1})
    y = env.data.xpos[env.model.body("small").id][1]
    assert y < 0.1 - 0.006 - 0.02 + 0.003
    assert env.audit.count == 0


def test_determinism_same_commands_same_hash():
    hashes = []
    for _ in range(2):
        env = make_env()
        h0 = env.initial_hash
        per_action = [env.act(t, dict(a)) and env.state_hash() for t, a in EASY_SOLUTION[:8]]
        hashes.append((h0, per_action))
        env.close()
    assert hashes[0] == hashes[1]


def test_seed_changes_initial_state_only_when_noise_enabled():
    p = load_puzzle(resolve_puzzle("easy"))
    from loophole.sim.env import LoopholeEnv

    a = LoopholeEnv(p, seed=1).initial_hash
    b = LoopholeEnv(p, seed=2).initial_hash
    assert a == b
    p2 = p.model_copy(update={"init_noise": 0.002})
    assert LoopholeEnv(p2, seed=1).initial_hash != LoopholeEnv(p2, seed=2).initial_hash
