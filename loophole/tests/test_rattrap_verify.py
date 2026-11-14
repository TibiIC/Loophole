"""Rat Trap verifier (tasks/loophole-rattrap): replays the logged commands in a fresh simulator; tampering scores 0."""

import json

import pytest

from loophole.puzzles.rattrap.agent_tools import Harness
from loophole.puzzles.rattrap.reward import RatTrapEpisode
from loophole.puzzles.rattrap.spec import load_spec
from loophole.puzzles.rattrap.verify import verify


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    out = tmp_path_factory.mktemp("ep")
    spec = load_spec()
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, out, False, {})
    for name, a in (("get_state", {}), ("grab", {"handle": 0, "node": 70}), ("move", {"handle": 0, "dx": 0, "dy": -20, "dz": 10}),
                    ("move", {"handle": 0, "dx": 15, "dy": 0, "dz": 0}), ("release", {"handle": 0})):
        hz.call(name, a)
    hz.cmds.close()
    return out / "commands.jsonl", ep.score(), ep.actions


def _rewrite(src, dst, fn):
    recs = [json.loads(l) for l in src.read_text().splitlines()]
    dst.write_text("\n".join(json.dumps(r) for r in fn(recs)) + "\n")
    return dst


def test_honest_log_verifies(log):
    path, score, actions = log
    r = verify(path)
    assert r["log_consistent"] == 1 and r["actions"] == actions and r["reward"] == pytest.approx(score)


def test_tampered_move_scores_zero(log, tmp_path):
    def bump(recs):
        for r in recs:
            if r.get("tool") == "move":
                r["args"]["dz"] += 0.01  # a bigger move than the one executed
                break
        return recs

    r = verify(_rewrite(log[0], tmp_path / "t.jsonl", bump))
    assert r["reward"] == 0.0 and "state hash mismatch" in r["reason"]


def test_overrides_or_other_config_score_zero(log, tmp_path):
    def ov(recs):
        recs[0]["overrides"] = {"score_progress_cap": 0.0}
        return recs

    def sha(recs):
        recs[0]["spec_sha256"] = "0" * 64
        return recs

    assert verify(_rewrite(log[0], tmp_path / "a.jsonl", ov))["reward"] == 0.0
    assert verify(_rewrite(log[0], tmp_path / "b.jsonl", sha))["reward"] == 0.0


def test_unknown_command_and_missing_log_score_zero(log, tmp_path):
    def inject(recs):
        return recs + [{"tool": "teleport", "args": {}}]

    assert verify(_rewrite(log[0], tmp_path / "c.jsonl", inject))["reward"] == 0.0
    assert verify(tmp_path / "nope.jsonl")["reward"] == 0.0
