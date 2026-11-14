"""Alpha 1.0 as used in Zhang et al. 2020, with the authors' published solution.

Skips unless the paper's files were fetched (`make fetch-alpha-paper`).
"""

from pathlib import Path

import numpy as np
import pytest

from loophole.puzzles import paper_alpha as pa
from loophole.puzzles.ompl_alpha import axis_angle_quat, parse_cfg
from loophole.reference.recorder import load_trajectory
from loophole.reference.replay import replay
from loophole.sim.collision import CollisionChecker, interpolate, pose_metric
from loophole.sim.env import LoopholeEnv
from loophole.sim.geometry import quat_to_mat
from loophole.sim.spec import load_puzzle, resolve_puzzle, to_native

pytestmark = pytest.mark.skipif(not (pa.paper_dir() / pa.FILES["env"]).exists(), reason="paper files not fetched (make fetch-alpha-paper)")
NAME = "alpha-1.0-paper"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    p = load_puzzle(resolve_puzzle(NAME))
    ref = np.load(ROOT / "puzzles" / "alpha_paper" / f"{NAME}-reference.npz")
    return p, CollisionChecker(p), ref


def test_pieces_are_identical():
    import trimesh

    a = pa.load_obj(pa.paper_dir() / pa.FILES["env"]).copy()
    b = pa.load_obj(pa.paper_dir() / pa.FILES["robot"]).copy()
    for m in (a, b):
        m.merge_vertices(merge_tex=True, merge_norm=True)
        assert m.is_watertight
    assert abs(a.volume - b.volume) / b.volume < 1e-3 and abs(a.area - b.area) / b.area < 1e-3


def test_start_roundtrip_and_valid(setup):
    p, ck, _ = setup
    cfg = parse_cfg(pa.paper_dir() / pa.FILES["cfg"])
    o = p.objects[0]
    assert to_native(list(o.pos), p) == cfg["start"]["xyz"]
    assert np.allclose(o.quat, axis_angle_quat(cfg["start"]["theta"], cfg["start"]["axis"]))
    s = np.array([*o.pos, *o.quat])
    assert ck.is_valid(s) and not ck.is_separated(s)


def test_published_solution_states_match_fcl(setup):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fcl")
    p, ck, _ = setup
    A = pa.load_obj(pa.paper_dir() / pa.FILES["env"])
    B = pa.load_obj(pa.paper_dir() / pa.FILES["robot"])
    c = np.asarray(B.vertices).mean(axis=0)
    cm = trimesh.collision.CollisionManager()
    cm.add_object("A", A)
    path = pa.solution_path_native()  # the authors' full 214-state path
    errs = []
    for x in path:
        R = quat_to_mat(x[3:])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = x[:3] - R @ c
        assert not cm.in_collision_single(B, transform=T)  # valid on the raw meshes (OMPL convention)
        y = np.concatenate([x[:3] * pa.UNIT_SCALE, x[3:]])
        assert ck.is_valid(y)
        errs.append(abs(min(cm.min_distance_single(B, transform=T) * pa.UNIT_SCALE, 0.04) - ck.clearance(y, 0.04)))
    assert max(errs) < 1e-4  # decomposition error << any clearance


def test_reference_path_dense_valid(setup):
    p, ck, ref = setup
    path, w = ref["path"], float(ref["w"])
    for x, y in zip(path[:-1], path[1:]):
        k = max(1, int(np.ceil(float(pose_metric(x, y, w)) / 0.002)))  # build checks at 0.8 mm
        assert all(ck.is_valid(interpolate(x, y, i / k)) for i in range(k))
    assert ck.is_separated(path[-1])


def test_oracle_noop_stop_early(setup):
    final = replay(LoopholeEnv(NAME), load_trajectory(ROOT / "reference" / "trajectories" / f"{NAME}.jsonl"))
    assert final["success"] and final["violations"]["count"] == 0 and final["score"] > 0.5
    assert LoopholeEnv(NAME).finalize()["score"] == 0.0
    assert replay(LoopholeEnv(NAME), load_trajectory(ROOT / "reference" / "trajectories" / f"{NAME}_stop_early.jsonl"))["score"] == 0.0


def test_separation_check_never_false_positive(setup):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fcl")
    from loophole.planning.se3 import random_quat

    p, ck, ref = setup
    A = pa.load_obj(pa.paper_dir() / pa.FILES["env"])
    B = pa.load_obj(pa.paper_dir() / pa.FILES["robot"])
    c = np.asarray(B.vertices).mean(axis=0)
    cm = trimesh.collision.CollisionManager()
    cm.add_object("A", A)
    rng = np.random.default_rng(0)
    for _ in range(80):
        y = np.concatenate([ref["path"][-1][:3] + rng.normal(0, 0.15, 3), random_quat(rng)])
        x = y.copy()
        x[:3] /= pa.UNIT_SCALE
        R = quat_to_mat(x[3:])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = x[:3] - R @ c
        if cm.in_collision_single(B, transform=T):
            continue
        if ck.is_separated(y):
            assert cm.min_distance_single(B, transform=T) * pa.UNIT_SCALE > p.goal.d_sep
