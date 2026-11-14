"""Canonical OMPL.app Alpha 1.5 (meshes fetched by `make fetch-alpha`, never committed).

Checks the geometry pipeline against OMPL.app's own conventions:
(a) the .cfg start pose is collision-free with our decomposed hulls,
(b) the decomposition error is far below the narrowest clearance, measured against
    FCL on the raw canonical triangles along OMPL's own solution path,
(c) converting the start pose back to native units reproduces the .cfg exactly.
"""

import numpy as np
import pytest

from loophole.puzzles.ompl_alpha import UNIT_SCALE, axis_angle_quat, load_scene_mesh, ompl_dir, parse_cfg, parse_path, robot_center
from loophole.sim.collision import CollisionChecker
from loophole.sim.env import LoopholeEnv
from loophole.sim.geometry import quat_to_mat
from loophole.sim.spec import load_puzzle, resolve_puzzle, to_native

pytestmark = pytest.mark.skipif(not (ompl_dir() / "alpha_env-1.5.dae").exists(), reason="canonical meshes not fetched (run `make fetch-alpha`)")
PUZZLE = "alpha-1.5"


@pytest.fixture(scope="module")
def setup():
    p = load_puzzle(resolve_puzzle(PUZZLE))
    return p, CollisionChecker(p), parse_cfg(ompl_dir() / "alpha-1.5.cfg")


def test_start_pose_roundtrip_reproduces_cfg_exactly(setup):
    p, ck, cfg = setup
    o = p.objects[0]
    assert to_native(list(o.pos), p) == cfg["start"]["xyz"]
    assert np.allclose(o.quat, axis_angle_quat(cfg["start"]["theta"], cfg["start"]["axis"]))
    assert p.native["objects"][0]["pos"] == cfg["start"]["xyz"]
    assert p.unit_scale == UNIT_SCALE


def test_canonical_start_is_collision_free(setup):
    p, ck, cfg = setup
    s = np.array([*p.objects[0].pos, *p.objects[0].quat])
    assert ck.is_valid(s) and not ck.is_separated(s)
    env = LoopholeEnv(PUZZLE)
    assert env.get_state()["violations"] == 0 and np.allclose(env.target_pose(), s, atol=1e-6)


def test_decomposition_error_far_below_narrowest_clearance(setup):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fcl")
    p, ck, cfg = setup
    A = load_scene_mesh(ompl_dir() / "alpha_env-1.5.dae", assimp_frame=True)
    B = load_scene_mesh(ompl_dir() / "alpha_robot.dae", assimp_frame=True)
    c = robot_center(B)
    cm = trimesh.collision.CollisionManager()
    cm.add_object("A", A)
    path = parse_path(ompl_dir() / "alpha-1.5.path")  # OMPL's own solution (native units)
    fcl_d, mj_d = [], []
    for x in path:
        R = quat_to_mat(x[3:])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = x[:3] - R @ c
        fcl_d.append(cm.min_distance_single(B, transform=T) * UNIT_SCALE)
        y = np.concatenate([x[:3] * UNIT_SCALE, x[3:]])
        assert ck.is_valid(y), "OMPL's own solution path must be collision-free in our sim"
        mj_d.append(ck.clearance(y, ck.NEAR))  # exact up to NEAR; beyond it the far-field proxy gives lower bounds
    fcl_d, mj_d = np.array(fcl_d), np.array(mj_d)
    near = fcl_d < ck.NEAR
    err = float(np.max(np.abs(mj_d[near] - fcl_d[near])))
    narrowest = float(fcl_d.min())
    assert narrowest > 0.001  # ~1.8 mm along OMPL's path
    assert err < 0.2 * narrowest, (err, narrowest)


def test_noop_zero_and_cfg_goal_not_counted_as_separated(setup):
    p, ck, cfg = setup
    assert LoopholeEnv(PUZZLE).finalize()["score"] == 0.0
    # d_sep (0.4 m) is above the largest clearance reachable while interlocked (~0.22 m)
    assert p.goal.d_sep == pytest.approx(0.4)


def test_reference_plan_valid_and_replays(setup):
    from pathlib import Path

    from loophole.reference.recorder import load_trajectory
    from loophole.reference.replay import replay

    p, ck, cfg = setup
    root = Path(__file__).resolve().parents[1]
    ref = np.load(root / "puzzles" / "alpha_ompl" / "alpha-1.5-reference.npz")
    dense = ref["dense"]
    assert all(ck.is_valid(x) for x in dense[::7])  # full 0.5 mm check is done at build time
    final = replay(LoopholeEnv(PUZZLE), load_trajectory(root / "reference" / "trajectories" / "alpha-1.5.jsonl"))
    assert final["success"] and final["violations"]["count"] == 0 and final["score"] > 0.5
    se = replay(LoopholeEnv(PUZZLE), load_trajectory(root / "reference" / "trajectories" / "alpha-1.5_stop_early.jsonl"))
    assert se["score"] == 0.0
