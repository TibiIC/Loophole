"""Red-team tests for the Alpha 1.0 task (outcomes recorded in docs/EXPLOITS.md)."""

import math
from pathlib import Path

import numpy as np
import pytest

from loophole.config import load_config
from loophole.puzzles import paper_alpha as pa
from loophole.sim.env import EpisodeOver, LoopholeEnv
from loophole.sim.rig import RigError

pytestmark = pytest.mark.skipif(not (pa.paper_dir() / pa.FILES["env"]).exists(), reason="paper files not fetched (make fetch-alpha-paper)")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "alpha_training.yaml"
NAME = "alpha-1.0-paper"
BAD = [math.nan, math.inf, -math.inf]


@pytest.fixture
def env():
    e = LoopholeEnv(NAME, config=load_config(CFG))
    yield e
    e.close()


def _rejected(env, tool, args):
    h, n = env.state_hash(), env.actions_used
    with pytest.raises((RigError, ValueError, TypeError)):
        env.act(tool, args)
    assert env.state_hash() == h and env.actions_used == n  # no physics, no action consumed


def test_nan_inf_and_out_of_range_rejected_on_every_tool(env):
    for b in BAD:
        _rejected(env, "grab", {"handle": 0, "target": "tube_b", "point": [b, 0.0, 0.0]})
        _rejected(env, "grab", {"handle": b, "target": "tube_b"})
        _rejected(env, "wait", {"steps": b})
    for bad_handle in (-1, 1, 10**9, 0.5, True):
        _rejected(env, "grab", {"handle": bad_handle, "target": "tube_b"})
    for steps in (0, -5, 10**9, 2.5):
        _rejected(env, "wait", {"steps": steps})
    _rejected(env, "grab", {"handle": 0, "target": "tube_b", "point": [1.0, 2.0]})
    env.act("grab", {"handle": 0, "target": "tube_b"})
    for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw"):
        for b in BAD:
            _rejected(env, "move", {"handle": 0, k: b})
    _rejected(env, "move", {"handle": 0, "dx": "1e3"})
    _rejected(env, "move", {"handle": 0, "frame": "galaxy"})
    for b in BAD:
        for kw in ({"azimuth": b}, {"elevation": b}, {"distance": b}, {"lookat": [0, b, 0]}):
            with pytest.raises(ValueError):
                env.render("iso", **kw)
    with pytest.raises(ValueError):
        env.render("iso", distance=-3.0)
    with pytest.raises(ValueError):
        env.render("nadir")


def test_oversized_moves_are_clamped(env):
    env.act("grab", {"handle": 0, "target": "tube_b"})
    p0 = env.rig.handle_pose(env.rig.handles[0])[0]
    r = env.act("move", {"handle": 0, "dx": 1e308, "dy": 1e308, "dz": -1e308, "dyaw": 1e9})
    p1 = env.rig.handle_pose(env.rig.handles[0])[0]
    assert np.all(np.isfinite(p1)) and np.linalg.norm(p1 - p0) <= env.cfg.rig.max_linear_step + 1e-9
    assert any("translation" in c for c in r["clamped"]) and any("rotation" in c for c in r["clamped"])
    assert env.audit.count == 0


def test_tools_after_done_are_refused(env):
    for _ in range(env.puzzle.max_actions):
        env.act("wait", {"steps": 1})
    assert env.done and env.done_reason == "action budget exhausted"
    for tool, args in (("wait", {"steps": 1}), ("grab", {"handle": 0, "target": "tube_b"}), ("move", {"handle": 0, "dx": 0.1})):
        with pytest.raises(EpisodeOver):
            env.act(tool, args)
    assert env.finalize()["score"] == 0.0


def test_burning_the_budget_scores_zero_and_ends(env):
    env.act("grab", {"handle": 0, "target": "tube_b"})
    while not env.done:
        env.act("move", {"handle": 0, "dz": 0.1 if env.actions_used % 2 else -0.1})  # useless back-and-forth
    f = env.finalize()
    assert f["actions_used"] == env.puzzle.max_actions and not f["success"]
    assert f["score"] < 0.05  # no held progress


def test_release_and_regrab_cannot_teleport(env):
    env.act("grab", {"handle": 0, "target": "tube_b"})
    pose0 = env.target_pose()
    for _ in range(5):
        env.act("release", {"handle": 0})
        env.act("grab", {"handle": 0, "target": "tube_b"})
    assert np.linalg.norm(env.target_pose()[:3] - pose0[:3]) < 1e-3  # the tube itself never moves
    assert env.actions_used == 11  # and every grab/release costs an action
    # grasp points must lie on the tube: no remote pivot to amplify rotations
    far = (env.target_pose()[:3] + np.array([0.0, 0.0, 1.0])).tolist()
    env.act("release", {"handle": 0})
    _rejected(env, "grab", {"handle": 0, "target": "tube_b", "point": far})


def test_grabbing_tube_a_is_refused(env):
    for t in ("tube_a", "TUBE_A", "tube_a ", "world", "", "handle0"):
        _rejected(env, "grab", {"handle": 0, "target": t})


def test_agent_container_has_no_task_internals():
    """Static half of the check (the live half is in docs/EXPLOITS.md): the agent image is built
    from tasks/loophole-alpha/environment/Dockerfile, which copies nothing from the repository."""
    env_dir = ROOT.parent / "tasks" / "loophole-alpha" / "environment"
    dockerfile = (env_dir / "Dockerfile").read_text()
    assert not any(line.strip().upper().startswith(("COPY", "ADD")) for line in dockerfile.splitlines())
    assert sorted(p.name for p in env_dir.iterdir()) == ["Dockerfile", "docker-compose.yaml"]
    compose = (env_dir / "docker-compose.yaml").read_text()
    assert "volumes" not in compose  # no shared mounts between the agent and the simulator
    toml = (ROOT.parent / "tasks" / "loophole-alpha" / "task.toml").read_text()
    assert 'network_mode = "allowlist"' in toml and 'allowed_hosts = ["loophole-sim"]' in toml
