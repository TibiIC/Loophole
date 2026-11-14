"""Alpha 1.0 training environment: the three literature rewards as logged training signals."""

import json
from pathlib import Path

import numpy as np
import pytest

from loophole.config import load_config
from loophole.puzzles import paper_alpha as pa
from loophole.reference.recorder import load_trajectory
from loophole.reference.replay import replay
from loophole.sim.env import LoopholeEnv

pytestmark = pytest.mark.skipif(not (pa.paper_dir() / pa.FILES["env"]).exists(), reason="paper files not fetched (make fetch-alpha-paper)")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "alpha_training.yaml"
NAME = "alpha-1.0-paper"


def test_centerlines_identical_and_inside():
    trimesh = pytest.importorskip("trimesh")
    lengths = []
    for key, f in (("env", "derived_env.obj"), ("robot", "derived_robot.obj")):
        P = np.load(pa.paper_dir() / f"derived_{key}_centerline.npy")
        m = pa.load_obj(pa.paper_dir() / f)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        assert m.contains(P).all()
        lengths.append(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    assert abs(lengths[0] - lengths[1]) / lengths[1] < 0.02  # congruent tubes


def test_training_rewards_on_oracle(tmp_path):
    log = tmp_path / "ep.jsonl"
    env = LoopholeEnv(NAME, config=load_config(CFG), log_path=log)
    assert env.diagnostics()["linking_closed"] == pytest.approx(-1.0, abs=0.02)  # interlocked
    final = replay(env, load_trajectory(ROOT / "reference" / "trajectories" / f"{NAME}.jsonl"))
    b = final["breakdown"]
    # the official score is only the base reward
    assert final["success"] and final["score"] == pytest.approx(b["base"]["terminal"])
    assert not any(b[k]["in_score"] for k in ("cspace_tunnel", "industreal_sdf", "gauss_linking"))
    # topological shaping telescopes from |Lk| = 1 to 0 on the published solution
    assert b["gauss_linking"]["step_sum"] == pytest.approx(1.0, abs=0.02)
    assert b["industreal_sdf"]["step_sum"] > 0
    recs = [json.loads(l) for l in log.read_text().splitlines() if '"type":"action"' in l]
    assert set(recs[0]["rewards"]) == {"base", "cspace_tunnel", "industreal_sdf", "gauss_linking"}
    assert recs[-1]["diagnostics"]["linking_closed"] == pytest.approx(0.0, abs=0.02)


def test_training_noop_scores_zero():
    assert LoopholeEnv(NAME, config=load_config(CFG)).finalize()["score"] == 0.0
