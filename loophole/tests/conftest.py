from pathlib import Path

import pytest

from loophole.config import load_config
from loophole.sim.env import LoopholeEnv

FIXTURES = Path(__file__).parent / "fixtures"

# Scripted solution of the placeholder easy puzzle (also the reference trajectory).
EASY_SOLUTION = (
    [("grab", {"handle": 0, "target": "rope", "segment": 20})]
    + [("move", {"handle": 0, "dz": 0.1})] * 4
    + [("grab", {"handle": 1, "target": "ring"})]
    + [("move", {"handle": 1, "dx": 0.1})] * 9
)


@pytest.fixture
def easy_env():
    env = LoopholeEnv("easy")
    yield env
    env.close()


@pytest.fixture
def hollow_env():
    env = LoopholeEnv(FIXTURES / "hollow.yaml")
    yield env
    env.close()


def make_env(puzzle="easy", **overrides):
    return LoopholeEnv(puzzle, config=load_config(overrides=overrides or None))


def run(env, seq):
    out = []
    for tool, args in seq:
        out.append(env.act(tool, dict(args)))
    return out


def move_to(env, handle, target, max_moves=20):
    """Move a handle towards a world position in clamped steps; returns last result."""
    import numpy as np

    r = None
    for _ in range(max_moves):
        p = np.array(env.rig.handle_pose(env.rig.handles[handle])[0])
        d = np.asarray(target, dtype=float) - p
        if np.linalg.norm(d) < 1e-3:
            break
        r = env.act("move", {"handle": handle, "dx": d[0], "dy": d[1], "dz": d[2]})
        if r["blocked"]:
            break
    return r
