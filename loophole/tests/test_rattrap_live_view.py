"""The live viewer (docs/watching.md) is off by default, observes without changing anything, and never reaches the agent."""

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

from loophole.puzzles.rattrap.agent_tools import Harness
from loophole.puzzles.rattrap.reward import RatTrapEpisode
from loophole.puzzles.rattrap.spec import load_spec

CALLS = [("get_state", {}), ("grab", {"handle": 0, "node": 70}), ("move", {"handle": 0, "dx": 0, "dy": -20, "dz": 10}),
         ("move", {"handle": 0, "dx": 15, "dy": 0, "dz": 0}), ("release", {"handle": 0})]


def _run(port=None):
    out = Path(tempfile.mkdtemp())
    spec = load_spec()
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, out, False, {})
    lv = None
    if port:
        from loophole.puzzles.rattrap.live_view import LiveView

        lv = LiveView(out, port).attach(hz)
    results = [hz.call(t, a)[0] for t, a in CALLS]
    hz.cmds.close()
    return out, results, lv


def test_off_by_default():
    _, _, _ = _run()
    spec = load_spec()
    assert Harness(spec, RatTrapEpisode(spec), Path(tempfile.mkdtemp()), False, {}).live is None


def test_on_and_off_give_byte_identical_episode_logs_and_tool_results():
    off, r_off, _ = _run()
    on, r_on, lv = _run(port=8790)
    try:
        assert hashlib.sha256((off / "commands.jsonl").read_bytes()).digest() == hashlib.sha256((on / "commands.jsonl").read_bytes()).digest()
        assert r_off == r_on  # the agent sees exactly the same results
        assert lv.server.server_address[0] == "127.0.0.1"  # localhost only
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:8790/status.json", timeout=5).read())
        assert st["action"] == 4 and st["last_call"].startswith("release(")
        assert len(list((on / "frames").glob("*.png"))) == 4  # one frame per action, written as the run goes
    finally:
        lv.close()


def test_mcp_tools_unchanged_by_the_viewer():
    import anyio

    from loophole.puzzles.rattrap.mcp_server import build

    names = lambda m: sorted(t.name for t in anyio.run(m.list_tools))  # noqa: E731
    assert names(build(Path(tempfile.mkdtemp()))) == names(build(Path(tempfile.mkdtemp()), live_view=8791)) == ["get_state", "give_up", "grab", "move", "release"]
