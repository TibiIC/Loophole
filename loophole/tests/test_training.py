"""Training support (loophole.gym, start sampling, MCP training mode). Evaluation must be unaffected."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

from loophole.puzzles import paper_alpha as pa

pytestmark = pytest.mark.skipif(not (pa.paper_dir() / pa.FILES["env"]).exists(), reason="paper files not fetched (make fetch-alpha-paper)")
gym = pytest.importorskip("gymnasium")
CFG = "loophole/configs/alpha_training.yaml"


@pytest.fixture(scope="module")
def train_env():
    from loophole.gym import AlphaEnv

    e = AlphaEnv(start="perturb")
    yield e
    e.close()


def test_training_never_uses_the_canonical_start(train_env):
    from loophole.gym import AlphaEnv
    from loophole.sim.collision import pose_metric

    with pytest.raises(ValueError):
        AlphaEnv(start="canonical")
    with pytest.raises(ValueError):
        train_env.reset(seed=0, options={"start": "canonical"})
    s = train_env.sampler
    for seed in range(3):
        train_env.reset(seed=seed)
        q = train_env.env.start_pose
        assert pose_metric(q, s.q0[None], s.w)[0] >= s.exclude_radius
        assert train_env.env.checker.is_valid(q)  # collision-free
        assert train_env.env.escape_direction(q) is None  # still interlocked
        assert train_env.start_info["start_ctg"] >= s.min_ctg_frac * s.c0


def test_reset_is_seeded(train_env):
    a = train_env.reset(seed=7)[0]["pose"]
    b = train_env.reset(seed=7)[0]["pose"]
    c = train_env.reset(seed=8)[0]["pose"]
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_reverse_curriculum_distance_orders_cost_to_go(train_env):
    rng = np.random.default_rng(0)
    near = [train_env.sampler.sample(rng, mode="curriculum", curriculum_distance=0.1)[1]["start_ctg"] for _ in range(3)]
    far = [train_env.sampler.sample(rng, mode="curriculum", curriculum_distance=0.8)[1]["start_ctg"] for _ in range(3)]
    road = train_env.sampler.sample(rng, mode="curriculum", curriculum_distance=0.4, curriculum_source="roadmap")[1]
    assert max(near) < min(far)
    assert abs(road["start_ctg"] / road["canonical_ctg"] - 0.4) <= train_env.sampler.band + 1e-9


def test_eval_mode_score_equals_verifier(tmp_path):
    from loophole.gym import AlphaEnv
    from loophole.reward.verify import verify_episode

    env = AlphaEnv(mode="eval")
    env.reset(seed=0, options={"log_path": tmp_path / "ep.jsonl"})
    for a in ([1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [-1, 0, 0, 0, 0, 0.5]):
        env.step(np.array(a, float) * 0.5)
    score = env.env.finalize()["score"]
    v = verify_episode(tmp_path / "ep.jsonl", "alpha-1.0-paper", 0, CFG)
    assert v["log_consistent"] and v["reward"] == pytest.approx(score)
    env.close()


def test_training_logs_are_not_evaluation_logs(train_env, tmp_path):
    from loophole.reward.verify import verify_episode

    train_env.reset(seed=1, options={"log_path": tmp_path / "t.jsonl"})
    train_env.step(np.zeros(6))
    train_env.env.finalize()
    v = verify_episode(tmp_path / "t.jsonl", "alpha-1.0-paper", 1, CFG)
    assert v["reward"] == 0.0 and "training episode" in v["reason"]


def test_shaping_telescopes_and_score_at_end(train_env):
    train_env.reset(seed=2)
    rs = []
    for a in ([1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]):
        _, r, *_ = train_env.step(np.array(a, float) * 0.5)
        rs.append(r)
    assert sum(rs) == pytest.approx(0.0, abs=1e-6)  # potential-based: there and back nets zero
    info = None
    while not train_env.env.done:
        _, _, term, trunc, info = train_env.step(np.zeros(6))
    assert "score" in info and 0.0 <= info["score"] <= 0.3


def test_vector_env():
    from loophole.gym import make_vec_env

    v = make_vec_env(2, asynchronous=False, start="perturb")
    obs, _ = v.reset(seed=[0, 1])
    assert obs["pose"].shape == (2, 7) and not np.allclose(obs["pose"][0], obs["pose"][1])
    obs, r, term, trunc, info = v.step(np.zeros((2, 6), np.float32))
    assert r.shape == (2,)
    v.close()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(tmp_path, training: bool):
    port = _free_port()
    args = [sys.executable, "-m", "loophole.mcp_server", "--puzzle", "alpha-1.0-paper", "--config", CFG, "--port", str(port),
            "--host", "127.0.0.1", "--log-dir", str(tmp_path)] + (["--training"] if training else [])
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ})
    url = f"http://127.0.0.1:{port}"
    for _ in range(240):
        try:
            urllib.request.urlopen(url + "/healthz", timeout=1)
            return proc, url
        except Exception:
            time.sleep(0.25)
    proc.kill()
    raise RuntimeError(proc.stdout.read().decode())


async def _calls(url, calls):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    out = []
    async with streamable_http_client(url + "/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            out.append(sorted(t.name for t in (await s.list_tools()).tools))
            for name, args in calls:
                out.append(await s.call_tool(name, args))
    return out


def test_mcp_eval_mode_has_no_reset_and_no_shaping(tmp_path):
    proc, url = _serve(tmp_path, training=False)
    try:
        res = asyncio.run(_calls(url, [("grab", {"handle": 0, "target": "tube_b"})]))
        assert "reset" not in res[0]
        out = json.loads(res[1].content[0].text)
        assert not ({"shaping_reward", "rewards", "score", "goal", "progress"} & set(out))
    finally:
        proc.terminate()
        proc.wait(10)


def test_mcp_training_mode_resets_and_returns_shaping(tmp_path):
    proc, url = _serve(tmp_path, training=True)
    try:
        calls = [("grab", {"handle": 0, "target": "tube_b"}),  # refused: canonical start held out
                 ("reset", {"seed": 3}), ("grab", {"handle": 0, "target": "tube_b"}), ("move", {"handle": 0, "dx": 0.05}),
                 ("reset", {"seed": 4, "start": "curriculum", "curriculum_distance": 0.5})]
        res = asyncio.run(_calls(url, calls))
        assert "reset" in res[0]
        assert res[1].isError and "reset" in res[1].content[0].text
        r1 = json.loads(res[2].content[0].text)
        assert r1["episode"] == 1 and r1["start"]["start_mode"] == "perturb"
        mv = json.loads(res[4].content[0].text)
        assert "shaping_reward" in mv and "cspace_tunnel" in mv["rewards"]
        assert json.loads(res[5].content[0].text)["episode"] == 2
        logs = sorted(p.name for p in tmp_path.glob("episode-*.jsonl"))
        assert len(logs) == 2 and not (tmp_path / "episode.jsonl").exists()
        hdr = json.loads((tmp_path / logs[0]).open().readline())
        assert hdr["start_pose"] and hdr["training"]["mcp_training"]
    finally:
        proc.terminate()
        proc.wait(10)


def test_rollout_logs_export_and_replay(tmp_path):
    """collect_rollouts -> export_traces: randomised training episodes replay bit-identically."""
    import importlib.util
    from pathlib import Path

    from loophole.gym import AlphaEnv

    env = AlphaEnv(start="curriculum", curriculum_distance=0.5, randomize_friction=True, randomize_stiffness=True,
                   randomize_orientation=True)
    env.reset(seed=5, options={"log_path": tmp_path / "ep.jsonl"})
    rng = np.random.default_rng(5)
    for _ in range(3):
        env.step(rng.uniform(-1, 1, 6))
    score = env.env.finalize()["score"]
    env.close()
    spec = importlib.util.spec_from_file_location("export_traces", Path(__file__).resolve().parents[2] / "scripts" / "export_traces.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ep = mod.export(tmp_path / "ep.jsonl", tmp_path / "out", "instruction", inline=False)
    assert ep["replay_consistent"] and ep["score"] == pytest.approx(score) and ep["training"]["physics"]
    roles = [m["role"] for m in ep["messages"]]
    assert roles[:2] == ["system", "user"] and roles.count("tool") == roles.count("assistant") == 4  # grab + 3 moves
