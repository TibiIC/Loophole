"""Alpha 1.0 training environment: score/shaping split and the literature components."""

import json
from pathlib import Path

import numpy as np
import pytest

from loophole.config import load_config
from loophole.puzzles import paper_alpha as pa
from loophole.reference.recorder import load_trajectory
from loophole.reference.replay import replay
from loophole.reward.literature_rewards import episode_score
from loophole.sim.env import LoopholeEnv

pytestmark = pytest.mark.skipif(not (pa.paper_dir() / pa.FILES["env"]).exists(), reason="paper files not fetched (make fetch-alpha-paper)")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "alpha_training.yaml"
NAME = "alpha-1.0-paper"


def test_centerlines_identical_and_inside():
    pytest.importorskip("trimesh")
    lengths = []
    for key, f in (("env", "derived_env.obj"), ("robot", "derived_robot.obj")):
        P = np.load(pa.paper_dir() / f"derived_{key}_centerline.npy")
        m = pa.load_obj(pa.paper_dir() / f)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        assert m.contains(P).all()
        lengths.append(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    assert abs(lengths[0] - lengths[1]) / lengths[1] < 0.02  # congruent tubes


def test_oracle_score_is_episode_score_and_signals_are_logged(tmp_path):
    log = tmp_path / "ep.jsonl"
    env = LoopholeEnv(NAME, config=load_config(CFG), log_path=log)
    final = replay(env, load_trajectory(ROOT / "reference" / "trajectories" / f"{NAME}.jsonl"))
    b = final["breakdown"]
    assert final["success"]
    # one formula: the verifier score == cspace_tunnel.terminal == episode_score(final state)
    assert final["score"] == pytest.approx(b["cspace_tunnel"]["terminal"])
    assert final["score"] == pytest.approx(episode_score(env._reward_state()))
    assert final["score"] == pytest.approx(1 - 0.5 * final["actions_used"] / final["max_actions"])
    assert not b["clearance_shaping"]["in_score"] and not b["gauss_linking_diagnostic"]["in_score"]
    assert b["clearance_shaping"]["weight"] == 0 and b["gauss_linking_diagnostic"]["weight"] == 0
    recs = [json.loads(l) for l in log.read_text().splitlines() if '"type":"action"' in l]
    assert set(recs[0]["rewards"]) == {"cspace_tunnel", "clearance_shaping", "gauss_linking_diagnostic"}
    assert all(r["rewards"]["gauss_linking_diagnostic"] == 0.0 for r in recs)  # never a reward
    assert all("linking_open" in r["diagnostics"] for r in recs)  # logged per step
    assert all("cost_to_go" in r for r in recs)


def test_training_noop_scores_zero():
    assert LoopholeEnv(NAME, config=load_config(CFG)).finalize()["score"] == 0.0
