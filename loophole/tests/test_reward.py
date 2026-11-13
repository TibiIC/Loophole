"""Milestone 4: base reward, composer and the independent replay verifier."""

import json

import pytest

from loophole.baselines import noop, stop_early
from loophole.reference.replay import replay
from loophole.reference.scripted import SCRIPTS
from loophole.reward.components import BaseReward, RewardComposer, RewardState
from loophole.reward.verify import verify_episode
from loophole.config import RewardConfig
from loophole.sim.env import LoopholeEnv


def _rs(**kw):
    base = dict(puzzle="easy", actions_used=10, max_actions=60, success=True, violations=0, done=True)
    return RewardState(**{**base, **kw})


def test_base_reward_formula():
    b = BaseReward(alpha=0.5)
    assert b.terminal(_rs(), {}) == pytest.approx(1 - 0.5 * 10 / 60)
    assert b.terminal(_rs(success=False), {}) == 0.0
    assert b.terminal(_rs(violations=1), {}) == 0.0  # tunnelling forces failure
    # any solve (even using the whole budget) beats any non-solve
    assert b.terminal(_rs(actions_used=60), {}) == pytest.approx(0.5)


def test_noop_scores_zero():
    assert noop.run("easy")["score"] == 0.0


def test_stop_early_scores_zero():
    final = stop_early.run("easy")
    assert final["actions_used"] == stop_early.N_STEPS and final["score"] == 0.0


def test_reference_solves_with_good_score():
    final = replay(LoopholeEnv("easy"), SCRIPTS["easy"])
    assert final["success"] and final["score"] > 0.5
    assert final["violations"]["count"] == 0


def test_injected_violation_scores_zero():
    env = LoopholeEnv("easy")
    steps = SCRIPTS["easy"]
    for tool, args in steps[:6]:
        env.act(tool, dict(args))
    env.audit.inject(env.substeps)
    for tool, args in steps[6:]:
        if env.done:
            break
        env.act(tool, dict(args))
    final = env.finalize()
    assert not final["success"] and final["score"] == 0.0


def test_composer_keeps_solve_above_non_solve():
    cfg = RewardConfig.model_validate(
        {"components": [{"type": "base", "weight": 1.0, "params": {"alpha": 0.5}}, {"type": "topology_shaping", "weight": 0.2, "params": {"gamma": 1.0}}]}
    )
    comp = RewardComposer.from_config(cfg)
    comp.reset(_rs(actions_used=0, success=False, done=False, topology_word="ab"))
    comp.step(_rs(actions_used=1, success=False, done=False, topology_word="a"))
    comp.step(_rs(actions_used=2, success=False, done=False, topology_word=""))
    comp.terminal(_rs(actions_used=2, success=False))
    non_solve = comp.score()
    assert 0 < non_solve <= 0.2 < 0.5


# ---------------------------------------------------------------- verifier
@pytest.fixture
def solved_log(tmp_path):
    p = tmp_path / "episode.jsonl"
    replay(LoopholeEnv("easy", log_path=p), SCRIPTS["easy"])
    return p


def test_verifier_replays_and_matches(solved_log):
    res = verify_episode(solved_log, "easy")
    assert res["log_consistent"] == 1 and res["sidecar_agrees"] == 1
    assert res["success"] == 1 and res["reward"] == pytest.approx(0.875)


def test_verifier_works_without_final_record(solved_log):
    lines = [l for l in solved_log.read_text().splitlines() if json.loads(l)["type"] != "final"]
    solved_log.write_text("\n".join(lines) + "\n")
    assert verify_episode(solved_log, "easy")["reward"] == pytest.approx(0.875)


def test_verifier_rejects_tampered_actions(solved_log):
    recs = [json.loads(l) for l in solved_log.read_text().splitlines()]
    # pretend the ring was carried away in bigger steps than really executed
    for r in recs:
        if r.get("type") == "action" and r["tool"] == "move" and r["args"]["handle"] == 1:
            r["args"]["dx"] = 0.05
    solved_log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    res = verify_episode(solved_log, "easy")
    assert res["reward"] == 0.0 and res["log_consistent"] == 0


def test_verifier_rejects_forged_success(solved_log):
    recs = [json.loads(l) for l in solved_log.read_text().splitlines()]
    hdr, acts = recs[0], [r for r in recs if r.get("type") == "action"][:3]
    forged = [hdr, *acts, {"type": "final", "score": 1.0, "success": True}]
    solved_log.write_text("\n".join(json.dumps(r) for r in forged) + "\n")
    res = verify_episode(solved_log, "easy")
    assert res["reward"] == 0.0 and res["sidecar_agrees"] == 0


def test_verifier_rejects_other_puzzle_or_config(solved_log, tmp_path):
    recs = [json.loads(l) for l in solved_log.read_text().splitlines()]
    recs[0]["puzzle_sha256"] = "0" * 64
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert verify_episode(bad, "easy")["reward"] == 0.0
    recs[0] = json.loads(solved_log.read_text().splitlines()[0])
    recs[0]["config"]["rig"]["max_linear_step"] = 1.0
    bad.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert verify_episode(bad, "easy")["reward"] == 0.0


def test_verifier_missing_log(tmp_path):
    assert verify_episode(tmp_path / "nope.jsonl", "easy")["reward"] == 0.0
