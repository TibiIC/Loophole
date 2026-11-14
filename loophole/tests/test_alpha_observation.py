"""Alpha 1.0: the free-escape goal and the observation profiles (sensors / planner)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from loophole.config import load_config
from loophole.puzzles import paper_alpha as pa
from loophole.sim.collision import fibonacci_directions
from loophole.sim.env import LoopholeEnv
from loophole.sim.geometry import quat_to_mat

pytestmark = pytest.mark.skipif(not (pa.paper_dir() / pa.FILES["env"]).exists(), reason="paper files not fetched (make fetch-alpha-paper)")
ROOT = Path(__file__).resolve().parents[1]
NAME = "alpha-1.0-paper"
REF = pa.PUZZLE_OUT / f"{NAME}-reference.npz"


def _env(profile="sensors"):
    return LoopholeEnv(NAME, config=load_config(ROOT / "configs" / "alpha_training.yaml", overrides={"observation": {"profile": profile}}))


@pytest.fixture(scope="module")
def sensors():
    e = _env("sensors")
    yield e
    e.close()


@pytest.fixture(scope="module")
def planner():
    e = _env("planner")
    yield e
    e.close()


def test_fibonacci_directions_are_unit_and_spread():
    D = fibonacci_directions(64)
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)
    assert np.abs(D.mean(axis=0)).max() < 0.05  # no preferred hemisphere


def test_escape_goal_along_the_authors_path(sensors):
    """Interlocked at the start, escapable at the end; a reported direction really is free."""
    ck = sensors.checker
    D = fibonacci_directions(64)
    path = np.load(REF)["path"]
    assert ck.escape_direction(path[0], D) is None
    assert ck.escape_direction(path[len(path) // 2], D) is None
    u = ck.escape_direction(path[-1], D)
    assert u is not None
    x = path[-1].copy()
    for s in np.arange(0.0, 4.0, 0.002):  # dense exact sweep along the reported direction
        x[:3] = path[-1][:3] + s * u
        assert ck.is_valid(x)


def test_sensors_profile_hides_privileged_information(sensors):
    st = sensors.get_state()
    assert set(st["tools"]) == {"get_state", "render", "get_contacts"}
    flat = json.dumps(st)
    for leak in ("min_distance", "progress", "cost_to_go", "geometry", "centerline", "goal", "escape"):
        assert leak not in flat, leak
    assert st["objects"]["tube_b"]["pos"] and st["objects"]["tube_b"]["quat"]
    _, cap = sensors.render_views("iso")
    for leak in ("min_distance", "progress", "goal", "escape"):
        assert leak not in cap, leak


def test_depth_back_projects_onto_the_true_surfaces(planner):
    from PIL import Image
    import trimesh

    imgs, cap = planner.render_views("top", modalities=["depth", "segmentation"], depth_format="metric")
    cam = json.loads(cap)["camera"]
    K = cam["intrinsics"]
    depth = np.asarray(Image.open(io.BytesIO(imgs[0][1]))).astype(float) / 1000.0
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    F, U, R, P = (np.asarray(cam[k]) for k in ("forward", "up", "right", "position"))
    pts = P + z[:, None] * (F + ((u + 0.5 - K["cx"]) / K["fx"])[:, None] * R - ((v + 0.5 - K["cy"]) / K["fy"])[:, None] * U)
    g = planner.get_geometry()
    meshes = []
    for name, m in g["meshes"].items():
        pose = np.asarray(g["poses"][name])
        meshes.append(trimesh.Trimesh(pose[:3] + np.asarray(m["vertices"]) @ quat_to_mat(pose[3:]).T, m["faces"]))
    sel = pts[:: max(1, len(pts) // 300)]
    d = np.min([np.abs(trimesh.proximity.signed_distance(m, sel)) for m in meshes], axis=0)
    assert np.percentile(d, 95) < 0.003  # metres
    seg = np.asarray(Image.open(io.BytesIO(imgs[1][1])))
    assert (seg[..., 2] > 200).any() and ((seg[..., 0] == 150) & (seg[..., 1] == 150)).any()  # both pieces labelled


def test_render_rejects_bad_arguments(sensors):
    for kw in ({"modalities": ["thermal"]}, {"modalities": ["rgb", "rgb"]}, {"width": 10}, {"height": 5000},
               {"width": 64.5}, {"depth_format": "raw"}):
        with pytest.raises(ValueError):
            sensors.render_views("iso", **kw)


def test_contacts_report_touch():
    e = _env("sensors")
    path = np.load(REF)["path"]
    from loophole.planning.commands import path_to_commands
    from loophole.reference.replay import replay

    cmds = path_to_commands(list(path[:6]), "tube_b", 0.1, 0.2)
    replay(e, cmds, stop_when_done=False)
    c = e.get_contacts()
    assert {"contacts", "total_contact_force", "grip_lag", "in_contact"} <= set(c)
    for k in c["contacts"]:
        assert np.isclose(np.linalg.norm(k["normal_into_free_piece"]), 1.0, atol=1e-3)
    e.close()


def test_check_motion_agrees_with_move():
    e = _env("planner")
    e.act("grab", {"handle": 0, "target": "tube_b"})
    free = e.check_motion(0, dx=0.1)
    hit = e.check_motion(0, dz=0.1)
    assert free["free"] and not hit["free"] and 0 < hit["free_fraction"] < 1
    h = e.state_hash()
    e.check_poses([e.target_pose().tolist()])
    assert e.state_hash() == h and e.actions_used == 1  # queries change nothing and cost nothing
    r = e.act("move", {"handle": 0, "dz": 0.1})
    assert r["blocked"]
    r = e.act("move", {"handle": 0, "dx": 0.1})
    assert not r["blocked"]
    e.close()


def test_profiles_change_only_observations():
    """The score must not depend on the observation profile: same actions, same log hashes."""
    from loophole.reference.recorder import load_trajectory
    from loophole.reference.replay import replay

    steps = load_trajectory(ROOT / "reference" / "trajectories" / "alpha-1.0-paper_stop_early.jsonl")
    out = []
    for prof in ("sensors", "planner"):
        e = _env(prof)
        f = replay(e, steps)
        out.append((f["score"], f["final_state_hash"]))
        e.close()
    assert out[0] == out[1]
