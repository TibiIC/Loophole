"""Alpha 1.0 score: violation 0; solved 1 - 0.5 a/T in [0.5, 1]; unsolved 0.3 p in [0, 0.3]."""

from pathlib import Path

import numpy as np
import pytest

from loophole.config import load_config
from loophole.puzzles import paper_alpha as pa
from loophole.reference.recorder import load_trajectory
from loophole.reference.replay import replay
from loophole.reward.components import RewardState, StableMin
from loophole.reward.literature_rewards import ClearanceShapingReward, episode_score
from loophole.sim.env import LoopholeEnv

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "alpha_training.yaml"
NAME = "alpha-1.0-paper"
TRAJ = ROOT / "reference" / "trajectories"
have_paper = (pa.paper_dir() / pa.FILES["env"]).exists()
have_roadmap = (pa.PUZZLE_OUT / f"roadmap-{NAME}.npz").exists()
needs_env = pytest.mark.skipif(not (have_paper and have_roadmap), reason="needs fetched paper files and the built roadmap")


def _rs(**kw):
    base = dict(puzzle="x", actions_used=50, max_actions=200, success=False, violations=0, done=True, start_cost_to_go=2.0)
    return RewardState(**{**base, **kw})


def test_every_solve_beats_every_non_solve_on_a_grid():
    T = 200
    solves = [episode_score(_rs(success=True, actions_used=a, max_actions=T)) for a in range(0, T + 1, 5)]
    non_solves = [episode_score(_rs(actions_used=a, max_actions=T, stable_best_cost_to_go=2.0 * (1 - p)))
                  for p in np.linspace(0, 1, 41) for a in range(0, T + 1, 25)]
    assert min(solves) == pytest.approx(0.5) and max(solves) == pytest.approx(1.0)
    assert max(non_solves) == pytest.approx(0.3) and min(non_solves) == 0.0
    assert min(solves) > max(non_solves)
    assert all(0.0 <= s <= 1.0 for s in solves + non_solves)
    assert episode_score(_rs(success=True, violations=1)) == 0.0


def test_single_glitch_cannot_spike_progress():
    m = StableMin(10.0, hold=3)
    for c in [9.0, 0.1, 9.0, 9.0]:
        m.update(c)
    assert m.best == pytest.approx(9.0)
    for c in [2.0, 2.0, None, 2.0, 2.0]:
        m.update(c)
    assert m.best == pytest.approx(9.0)
    m.update(2.0)
    assert m.best == pytest.approx(2.0)


def test_clearance_shaping_zero_without_motion():
    r = ClearanceShapingReward()
    s0 = _rs(min_distance=0.05)
    total = sum(r.step(s0, s0, {"args": {"steps": 100}, "last_penetration": 0.0}) for _ in range(20))
    assert total == 0.0


@needs_env
def test_clearance_shaping_sum_over_noop_episode_is_zero():
    env = LoopholeEnv(NAME, config=load_config(CFG))
    for _ in range(5):
        env.act("wait", {"steps": 50})  # physics runs, nothing moves
    f = env.finalize()
    assert f["breakdown"]["clearance_shaping"]["step_sum"] == pytest.approx(0.0, abs=1e-9)
    assert f["score"] == 0.0


@needs_env
def test_noop_stop_early_oracle_ordering():
    cfg = load_config(CFG)
    noop = LoopholeEnv(NAME, config=cfg).finalize()["score"]
    stop = replay(LoopholeEnv(NAME, config=cfg), load_trajectory(TRAJ / f"{NAME}_stop_early.jsonl"))["score"]
    oracle = replay(LoopholeEnv(NAME, config=cfg), load_trajectory(TRAJ / f"{NAME}.jsonl"))
    assert noop == 0.0
    assert stop < 0.05
    assert oracle["success"] and oracle["score"] > 0.5 and oracle["score"] > max(noop, stop, 0.3)


@needs_env
def test_oscillating_near_tunnel_mouth():
    cfg = load_config(CFG)
    steps = load_trajectory(TRAJ / f"{NAME}.jsonl")
    env = LoopholeEnv(NAME, config=cfg)
    for tool, args in steps[: int(0.6 * len(steps))]:  # follow the oracle to the tunnel mouth
        env.act(tool, dict(args))
    assert not env.done
    before = env.composer.step_totals[0]  # cspace_tunnel
    pose0 = env.target_pose()
    for k in range(20):  # wiggle back and forth there
        sign = 1.0 if k % 2 == 0 else -1.0
        env.act("move", {"handle": 0, "dx": 0.02 * sign, "dyaw": 0.05 * sign})
        if env.done:
            break
    net_shaping = env.composer.step_totals[0] - before
    returned = float(np.linalg.norm(env.target_pose()[:3] - pose0[:3]))
    final = env.finalize()
    assert not final["success"] and final["score"] < 0.3
    assert abs(net_shaping) < 0.05 + returned  # telescopes: ~0 when the pose returns


@needs_env
def test_injected_penetration_scores_zero_and_ends_episode():
    env = LoopholeEnv(NAME, config=load_config(CFG))
    env.act("grab", {"handle": 0, "target": "tube_b"})
    env.audit.inject(env.substeps)
    r = env.act("move", {"handle": 0, "dx": 0.05})
    assert r["done"] and not r["success"]
    with pytest.raises(Exception):
        env.act("move", {"handle": 0, "dx": 0.05})  # no further actions accepted
    assert env.finalize()["score"] == 0.0
