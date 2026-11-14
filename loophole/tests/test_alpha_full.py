"""Alpha beyond the MVP: all variants, reference paths, cost-to-go roadmap reward (WIP).

Tests needing build artifacts (puzzles/alpha/*-reference.npz, roadmap-*.npz,
reference/trajectories/alpha-*.jsonl, produced by `make alpha`) skip with a
reason when those files are missing. The roadmap reward is behind
`reward.roadmap` (off by default); these tests switch it on explicitly.
"""

from pathlib import Path

import numpy as np
import pytest

from loophole.config import load_config
from loophole.puzzles.alpha.generate import loop_polyline
from loophole.puzzles.alpha.geometry import gauss_linking, polyline_sagitta
from loophole.reference.recorder import load_trajectory
from loophole.reference.replay import replay
from loophole.reward.roadmap import Roadmap
from loophole.sim.collision import CollisionChecker, pose_metric
from loophole.sim.env import LoopholeEnv
from loophole.sim.geometry import quat_to_mat
from loophole.sim.spec import load_puzzle, resolve_puzzle

ALPHA = Path(__file__).resolve().parents[1] / "puzzles" / "alpha"
TRAJ = Path(__file__).resolve().parents[1] / "reference" / "trajectories"
VARIANTS = [v for v in ("alpha-easy", "alpha-medium", "alpha-hard") if (ALPHA / f"{v}.yaml").exists()]
ROADMAP_ON = load_config(overrides={"reward": {"roadmap": True}})


def _start(p):
    o = p.objects[0]
    return np.array([*o.pos, *o.quat])


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} not built yet (run `make alpha`)")
    return path


@pytest.mark.parametrize("v", VARIANTS)
def test_decomposition_error_and_margins_vs_clearance(v):
    p = load_puzzle(resolve_puzzle(v))
    m = p.meta
    c = m["nominal_gap_clearance"]
    assert polyline_sagitta(np.asarray(p.frame[0].geoms[0].points), m["S"], m["k"], m["T"]) <= 0.2 * c
    env = LoopholeEnv(v)
    assert np.all(env.model.geom_margin == 0)
    per_substep = (env.cfg.rig.linear_speed + env.cfg.rig.angular_speed * 0.31) * env.model.opt.timestep
    assert per_substep < 0.1 * c


@pytest.mark.parametrize("v", VARIANTS)
def test_start_valid_and_linked(v):
    p = load_puzzle(resolve_puzzle(v))
    ck = CollisionChecker(p)
    s = _start(p)
    assert ck.is_valid(s) and not ck.is_separated(s)
    L = loop_polyline(p.meta["k"])
    assert abs(abs(gauss_linking(L, L @ quat_to_mat(s[3:]).T + s[:3])) - 1) < 0.05


@pytest.mark.parametrize("v", VARIANTS)
def test_reference_path_valid_at_fine_resolution(v):
    p = load_puzzle(resolve_puzzle(v))
    ck = CollisionChecker(p)
    dense = np.load(_need(ALPHA / f"{v}-reference.npz"))["dense"]  # 0.5 mm metric resolution
    assert np.allclose(dense[0], _start(p))
    assert all(ck.is_valid(x) for x in dense) and ck.is_separated(dense[-1])


@pytest.mark.parametrize("v", VARIANTS)
def test_reference_replay_separates_in_sim(v):
    final = replay(LoopholeEnv(v), load_trajectory(_need(TRAJ / f"{v}.jsonl")))
    assert final["success"] and final["violations"]["count"] == 0 and final["score"] > 0.5


@pytest.fixture(scope="module", params=VARIANTS)
def rm(request):
    v = request.param
    _need(ALPHA / f"roadmap-{v}.npz")
    p = load_puzzle(resolve_puzzle(v))
    return v, p, CollisionChecker(p), Roadmap.for_puzzle(p)


def test_roadmap_ctg_decreases_along_reference(rm):
    v, p, ck, road = rm
    dense = np.load(_need(ALPHA / f"{v}-reference.npz"))["dense"]
    idx = np.linspace(0, len(dense) - 1, 60).astype(int)
    ctg = [road.cost_to_go(dense[i], ck)[0] for i in idx]
    assert all(c is not None for c in ctg)
    tol = 0.03 * ctg[0]
    assert all(b <= a + tol for a, b in zip(ctg[:-1], ctg[1:])), np.round(ctg, 3)


def test_roadmap_does_not_leak_cost_through_walls(rm):
    """Poses near the reference often have a metrically close, much cheaper node on the
    other side of tube A. Without the visibility check they would inherit its cost."""
    from loophole.sim.geometry import quat_from_rotvec, quat_mul, quat_normalize

    v, p, ck, road = rm
    dense = np.load(_need(ALPHA / f"{v}-reference.npz"))["dense"]
    rng = np.random.default_rng(0)
    found = tried = 0
    while tried < 250:
        b = dense[rng.integers(len(dense))]
        q = np.concatenate([b[:3] + rng.normal(0, 0.02, 3), quat_normalize(quat_mul(quat_from_rotvec(rng.normal(0, 0.3, 3)), b[3:]))])
        if not ck.is_valid(q):
            continue
        tried += 1
        ctg_q, node = road.cost_to_go(q, ck)
        if ctg_q is None:
            continue
        d = pose_metric(q, road.nodes, road.w)
        for n in np.argsort(d)[:10]:
            through_wall = d[n] + road.cost[n]
            if through_wall < ctg_q - 0.05 and not ck.segment_valid(q, road.nodes[n], road.w, road.resolution):
                found += 1
                assert node != n and ctg_q > through_wall  # the occluded cheap node was not used
    assert found > 0, "expected occluded cheap nodes near the reference (the guard is exercised)"


def test_roadmap_goal_nodes_are_really_separated(rm):
    v, p, ck, road = rm
    goal = np.nonzero(road.cost == 0)[0]
    assert all(ck.is_separated(road.nodes[i]) for i in goal[:: max(1, len(goal) // 200)])


@pytest.mark.parametrize("v", VARIANTS)
def test_roadmap_reward_integrity(v):
    _need(ALPHA / f"roadmap-{v}.npz")
    noop = LoopholeEnv(v, config=ROADMAP_ON).finalize()["score"]
    stop = replay(LoopholeEnv(v, config=ROADMAP_ON), load_trajectory(_need(TRAJ / f"{v}_stop_early.jsonl")))["score"]
    ref = replay(LoopholeEnv(v, config=ROADMAP_ON), load_trajectory(_need(TRAJ / f"{v}.jsonl")))["score"]
    assert noop == 0.0 and stop < 0.05 and ref > max(noop, stop) and ref >= 0.5

    def oscillate(n):
        env = LoopholeEnv(v, config=ROADMAP_ON)
        env.act("grab", {"handle": 0, "target": "tube_b"})
        for _ in range(n):
            for dz in (0.02, -0.02):
                env.act("move", {"handle": 0, "dz": dz})
        f = env.finalize()
        return f["score"], f["breakdown"]["cost_to_go_shaping"]["step_sum"]

    s2, _ = oscillate(2)
    s8, sh8 = oscillate(8)
    assert s8 <= s2 + 1e-9 and abs(sh8) < 0.02


def test_progress_potential_follows_flag():
    v = "alpha-medium"
    assert LoopholeEnv(v).get_state()["progress_potential"] == 0.0
    _need(ALPHA / f"roadmap-{v}.npz")
    st = LoopholeEnv(v, config=ROADMAP_ON).get_state()
    assert st["progress_potential"] < 0 and st["progress"]["progress"] == pytest.approx(0.0, abs=1e-6)
