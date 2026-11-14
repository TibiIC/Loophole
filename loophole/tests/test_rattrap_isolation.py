"""Rat Trap task isolation (static half; the live half is loophole/harbor_probe.py, recorded in docs/EXPLOITS.md):
the agent container holds nothing of the reference (solution/, the recorded trajectory, the checkpoint file or
anything derived from them), and the MCP surface leaks none of it."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks" / "loophole-rattrap"


def test_agent_image_copies_nothing_and_shares_nothing():
    env = TASK / "environment"
    assert sorted(p.name for p in env.iterdir()) == ["Dockerfile", "docker-compose.yaml"]
    dockerfile = (env / "Dockerfile").read_text()
    assert not any(l.strip().upper().startswith(("COPY", "ADD")) for l in dockerfile.splitlines())
    compose = (env / "docker-compose.yaml").read_text()
    assert "volumes" not in compose  # no shared mounts between the agent and the simulator
    main = compose.split("loophole-sim:\n    #")[0]  # the "main" (agent) service block
    assert "build:" not in main and "context" not in main  # the agent image is not built from the repository


def test_network_and_artifacts():
    toml = (TASK / "task.toml").read_text()
    assert 'network_mode = "allowlist"' in toml and 'allowed_hosts = ["loophole-sim"]' in toml
    assert re.search(r'artifacts = \[\{ source = "/var/loophole/episode", destination = "episode", service = "loophole-sim" \}\]', toml)
    assert 'network_mode = "no-network"' in toml  # the verifier


def test_instruction_contains_no_part_of_the_reference():
    text = (TASK / "instruction.md").read_text().lower()
    for word in ("checkpoint", "rattrap-solve", "reference solve", "solution/", "teleop", ".npz"):
        assert word not in text
    ref = [json.loads(l) for l in (TASK / "solution" / "reference.jsonl").read_text().splitlines() if l.strip()]
    moves = [r["args"] for r in ref if r.get("tool") == "move" and abs(r["args"]["dz"]) >= 1.0][:20]
    assert moves and not any(f"{m['dz']:.3f}" in text for m in moves)  # no move of the reference is quoted


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from loophole.puzzles.rattrap.mcp_server import build

    return build(tmp_path_factory.mktemp("ep"))


def test_mcp_surface_is_the_five_game_tools(server):
    import anyio

    tools = anyio.run(server.list_tools)
    assert sorted(t.name for t in tools) == ["get_state", "give_up", "grab", "move", "release"]


def test_observations_and_results_carry_no_score_or_progress(tmp_path):
    from loophole.puzzles.rattrap.agent_tools import Harness
    from loophole.puzzles.rattrap.reward import RatTrapEpisode
    from loophole.puzzles.rattrap.spec import load_spec

    spec = load_spec()
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, tmp_path, False, {})
    outs = [hz.call("get_state", {})[0], hz.call("grab", {"handle": 0, "node": 70})[0],
            hz.call("move", {"handle": 0, "dx": 0, "dy": -10, "dz": 5})[0], hz.call("release", {"handle": 0})[0]]
    leaked = {"score", "progress", "checkpoints", "checkpoints_reached", "p", "reward", "shaping", "norm", "word", "eps"}
    for o in outs:
        assert not (leaked & set(o)), set(o) & leaked
        assert "checkpoint" not in json.dumps(o)


def test_admin_finalize_refuses_non_loopback(server):
    from starlette.testclient import TestClient

    app = server.streamable_http_app()
    with TestClient(app) as c:  # the test client's address is "testclient", not loopback
        r = c.post("/admin/finalize")
    assert r.status_code == 403 and r.json() == {"error": "forbidden"}
