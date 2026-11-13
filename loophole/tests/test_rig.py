"""Milestone 2: grab/move/release, two handles, velocity cap, tunnelling audit."""

import numpy as np
import pytest

from loophole.sim.rig import RigError

from .conftest import make_env


def test_grab_release_and_two_handles(easy_env):
    env = easy_env
    r0 = env.act("grab", {"handle": 0, "target": "rope", "segment": 5})
    r1 = env.act("grab", {"handle": 1, "target": "ring"})
    assert r0["grasped"] == "rope_5" and r1["grasped"] == "ring"
    st = env.get_state()["handles"]
    assert st[0]["holding"] == "rope_5" and st[1]["holding"] == "ring"
    env.act("release", {"handle": 0})
    assert env.get_state()["handles"][0]["holding"] is None
    assert env.actions_used == 3


def test_grab_by_point_selects_nearest_segment(easy_env):
    env = easy_env
    nodes = env.rope_nodes()
    p = (nodes[7] + nodes[8]) / 2
    r = env.act("grab", {"handle": 0, "target": "rope", "point": p.tolist()})
    assert r["grasped"] == "rope_7"


def test_invalid_requests_do_not_cost_actions(easy_env):
    env = easy_env
    h = env.state_hash()
    for tool, args in [
        ("grab", {"handle": 3, "target": "ring"}),
        ("grab", {"handle": 0, "target": "base"}),
        ("grab", {"handle": 0, "target": "ring", "point": [5, 5, 5]}),
        ("release", {"handle": 0}),
        ("move", {"handle": 0, "frame": "banana"}),
        ("teleport", {}),
    ]:
        with pytest.raises((RigError, ValueError)):
            env.act(tool, args)
    assert env.actions_used == 0 and env.state_hash() == h


def test_oversized_move_is_clamped(easy_env):
    env = easy_env
    env.act("grab", {"handle": 0, "target": "ring"})
    p0 = env.rig.handle_pose(env.rig.handles[0])[0]
    r = env.act("move", {"handle": 0, "dx": 5.0, "dyaw": 3.0})
    assert any("translation" in c for c in r["clamped"]) and any("rotation" in c for c in r["clamped"])
    p1 = env.rig.handle_pose(env.rig.handles[0])[0]
    assert np.linalg.norm(p1 - p0) <= env.cfg.rig.max_linear_step + 1e-9


def test_workspace_is_bounded(easy_env):
    env = easy_env
    env.act("grab", {"handle": 0, "target": "ring"})
    for _ in range(4):
        r = env.act("move", {"handle": 0, "dz": -0.1})
    assert r["handle_pos"][2] >= env.cfg.rig.workspace_min[2] - 1e-9


def test_max_speed_moves_into_frame_do_not_tunnel(easy_env):
    env = easy_env
    env.act("grab", {"handle": 0, "target": "ring"})
    blocked = []
    for mv in [{"dz": -0.1}] * 3 + [{"droll": 0.6}] * 2 + [{"dx": 0.1, "dz": -0.1}] * 3:
        blocked.append(env.act("move", {"handle": 0, **mv})["blocked"])
    assert any(blocked)
    assert env.audit.count == 0 and env.audit.max_penetration < env.cfg.audit.penetration_threshold
    # still above the loop: it never went through
    assert env.tracker.target_word == ""


def test_broken_rig_cannot_tunnel_unnoticed():
    # Remove every rig safety and use the stiffest weld MuJoCo allows: pressing the
    # ring down into the (too small) loop must either fail or be flagged.
    env = make_env(rig={"lag_limit": 10.0, "angle_lag_limit": 10.0, "stall_penetration": 10.0, "stall_anchor_drift": 10.0, "weld_solref": [0.002, 1.0]})
    env.act("grab", {"handle": 0, "target": "ring"})
    for _ in range(3):
        if env.done:
            break
        env.act("move", {"handle": 0, "dz": -0.1})
    passed_through = env.tracker.target_word != ""
    assert not passed_through or env.audit.count > 0


def _ring_qpos_adr(env):
    return env.model.jnt_qposadr[env.model.body("ring").jntadr[0]]


def test_audit_flags_teleport_through_loop(easy_env):
    # Simulate a physics exploit: the ring jumps 4 cm down, through the loop, in one substep.
    env = easy_env
    env.data.qpos[_ring_qpos_adr(env) + 2] -= 0.04
    env.act("wait", {"steps": 10})
    assert env.audit.count > 0
    assert {"speed"} <= {v["kind"] for v in env.audit.summary()["records"]}
    assert env.done and not env.success and env.finalize()["score"] == 0.0


def test_audit_flags_deep_overlap(easy_env):
    # Ring forced into the loop tube: a deep contact is a violation.
    env = easy_env
    adr = _ring_qpos_adr(env)
    env.data.qpos[adr : adr + 3] = [0.0, 0.0, 0.252]  # ring tube now overlaps the loop tube
    env.audit._prev_pos = None  # isolate the penetration check from the speed check
    env.act("wait", {"steps": 1})
    assert "penetration" in {v["kind"] for v in env.audit.summary()["records"]}
