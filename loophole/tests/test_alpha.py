"""Alpha MVP: start pose, no tunnelling, reward basics."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from loophole.puzzles.alpha.generate import loop_polyline
from loophole.puzzles.alpha.geometry import gauss_linking, segment_distances
from loophole.reference.recorder import load_trajectory
from loophole.reference.replay import replay
from loophole.sim.collision import CollisionChecker
from loophole.sim.env import LoopholeEnv
from loophole.sim.geometry import quat_to_mat
from loophole.sim.spec import load_puzzle, resolve_puzzle, to_native

PUZZLE = "alpha-medium"
TRAJ = Path(__file__).resolve().parents[1] / "reference" / "trajectories"


def _start(p):
    o = p.objects[0]
    return np.array([*o.pos, *o.quat])


def test_start_pose_is_collision_free_and_interlocked():
    p = load_puzzle(resolve_puzzle(PUZZLE))
    ck = CollisionChecker(p)
    s = _start(p)
    assert ck.is_valid(s) and not ck.is_separated(s)
    # independent check: numpy capsule-chain distance agrees with MuJoCo
    P = np.asarray(p.frame[0].geoms[0].points)
    r = p.frame[0].geoms[0].tube_radius
    R = quat_to_mat(s[3:])
    d_np = segment_distances(P, P @ R.T + s[:3]).min() - 2 * r
    assert d_np > 0 and ck.clearance(s, 0.1) == pytest.approx(d_np, abs=1e-6)
    # the two loops are linked (Gauss linking number of the closed loops = +-1)
    L = loop_polyline(p.meta["k"])
    assert abs(abs(gauss_linking(L, L @ R.T + s[:3])) - 1) < 0.05
    env = LoopholeEnv(PUZZLE)
    assert env.get_state()["violations"] == 0 and np.allclose(env.target_pose(), s, atol=1e-6)


def test_checker_margin_queries_exact():
    """Guards a MuJoCo pitfall: the midphase ignores geom margins unless disabled."""
    from loophole.planning.se3 import random_quat

    p = load_puzzle(resolve_puzzle(PUZZLE))
    ck = CollisionChecker(p)
    P = np.asarray(p.frame[0].geoms[0].points)
    r = p.frame[0].geoms[0].tube_radius
    rng = np.random.default_rng(0)
    s = _start(p)
    for _ in range(200):
        x = np.concatenate([s[:3] + rng.normal(0, 0.06, 3), random_quat(rng)])
        d_np = segment_distances(P, P @ quat_to_mat(x[3:]).T + x[:3]).min() - 2 * r
        assert ck.clearance(x, 0.08) == pytest.approx(min(d_np, 0.08), abs=1e-6)
        assert ck.is_separated(x) == (d_np >= p.goal.d_sep)


def test_b_cannot_tunnel_through_a_at_max_speed():
    rng = np.random.default_rng(0)
    for _ in range(12):
        env = LoopholeEnv(PUZZLE)
        env.act("grab", {"handle": 0, "target": "tube_b"})
        d = rng.normal(size=3)
        d *= 5 * env.cfg.rig.max_linear_step / np.linalg.norm(d)  # oversized: gets clamped to the cap
        w = rng.normal(size=3)
        w *= env.cfg.rig.max_angular_step / np.linalg.norm(w) / 2
        for _ in range(4):
            env.act("move", {"handle": 0, "dx": d[0], "dy": d[1], "dz": d[2], "droll": w[0], "dpitch": w[1], "dyaw": w[2]})
        assert env.audit.count == 0
        assert env.audit.max_penetration <= env.cfg.rig.stall_penetration + 0.001
        assert env.checker.is_valid(env.target_pose(), margin=-0.003)
        assert not env.success


def test_noop_scores_zero():
    assert LoopholeEnv(PUZZLE).finalize()["score"] == 0.0


def test_teleport_apart_counts_as_success():
    """Test-only state edit (never reachable over MCP): put B far away, then step."""
    env = LoopholeEnv(PUZZLE)
    adr = env.model.jnt_qposadr[env.model.body("tube_b").jntadr[0]]
    env.data.qpos[adr : adr + 3] += [0.0, 0.0, 0.8]
    mujoco.mj_forward(env.model, env.data)
    env.audit.prime(env.data)  # isolate the goal check from the teleport-speed audit
    r = env.act("wait", {"steps": 10})
    assert r["success"] and r["done"] and r["goal"]["goal_reached"]
    final = env.finalize()
    assert final["score"] == pytest.approx(1 - 0.5 * 1 / env.puzzle.max_actions)


def test_teleport_is_flagged_when_not_test_primed():
    env = LoopholeEnv(PUZZLE)
    adr = env.model.jnt_qposadr[env.model.body("tube_b").jntadr[0]]
    env.data.qpos[adr + 2] += 0.8
    env.act("wait", {"steps": 10})
    assert env.audit.count > 0 and not env.success and env.finalize()["score"] == 0.0


def test_reference_replay_separates():
    final = replay(LoopholeEnv(PUZZLE), load_trajectory(TRAJ / f"{PUZZLE}.jsonl"))
    assert final["success"] and final["violations"]["count"] == 0 and final["score"] > 0.5


def test_unit_scale_roundtrip(tmp_path):
    spec = {
        "name": "units", "unit_scale": 0.01, "gravity": [0, 0, 0], "floor": False,
        "frame": [{"name": "a", "geoms": [{"type": "tube", "tube_radius": 2.5, "points": [[0, 0, 0], [10, 0, 0]]}]}],
        "objects": [{"name": "b", "pos": [-21.91, -11.11, -14.14], "quat": [1, 0, 0, 0], "graspable": True,
                     "geoms": [{"type": "sphere", "size": [3.0]}]}],
        "target": "b", "goal": {"type": "separation", "d_sep": 5.0},
    }
    f = tmp_path / "u.yaml"
    f.write_text(yaml.safe_dump(spec))
    p = load_puzzle(f)
    assert p.objects[0].pos == pytest.approx((-0.2191, -0.1111, -0.1414)) and p.goal.d_sep == pytest.approx(0.05)
    assert to_native(list(p.objects[0].pos), p) == [-21.91, -11.11, -14.14]


def test_verifier_accepts_default_alpha_episode(tmp_path):
    """Regression (found by a Harbor oracle run): with the roadmap flag off the verifier
    must not expect a roadmap even though the puzzle ships one."""
    from loophole.reward.verify import verify_episode

    log = tmp_path / "episode.jsonl"
    final = replay(LoopholeEnv(PUZZLE, log_path=log), load_trajectory(TRAJ / f"{PUZZLE}.jsonl"))
    res = verify_episode(log, PUZZLE)
    assert res["log_consistent"] == 1 and res["success"] == 1 and res["reward"] == pytest.approx(final["score"])
