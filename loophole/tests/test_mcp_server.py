"""Milestone 3: the MCP server end to end over streamable-http (real subprocess + client)."""

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .conftest import EASY_SOLUTION


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("episode")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "loophole.mcp_server", "--puzzle", "easy", "--port", str(port), "--host", "127.0.0.1", "--log-dir", str(log_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.kill()
        raise RuntimeError(proc.stdout.read().decode())
    yield url, log_dir
    proc.terminate()
    proc.wait(10)


async def _session_calls(url, calls):
    out = []
    async with streamable_http_client(url + "/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            out.append(sorted(t.name for t in tools.tools))
            for name, args in calls:
                out.append(await s.call_tool(name, args))
    return out


def test_tools_state_render_and_actions(server):
    url, log_dir = server
    calls = [("get_state", {}), ("render", {"view": "front"}), ("grab", {"handle": 7, "target": "ring"})]
    calls += [(t, a) for t, a in EASY_SOLUTION[:3]]
    res = asyncio.run(_session_calls(url, calls))
    assert res[0] == ["get_state", "grab", "move", "release", "render", "wait"]  # nothing else exposed
    state = json.loads(res[1].content[0].text)
    assert state["actions_used"] == 0 and state["rope"]["num_segments"] == 40
    img, cap = res[2].content
    assert img.type == "image" and img.mimeType == "image/png"
    assert base64.b64decode(img.data)[:8] == b"\x89PNG\r\n\x1a\n"
    assert "ring" in cap.text
    assert res[3].isError and "handle" in res[3].content[0].text  # invalid: not counted
    moved = json.loads(res[6].content[0].text)  # res[4..6] = grab, move, move
    assert moved["action"] == 3 and not moved["done"]


def test_admin_finalize_is_loopback_only_and_logged(server):
    url, log_dir = server
    req = urllib.request.Request(url + "/admin/finalize", data=b"{}", method="POST")
    final = json.loads(urllib.request.urlopen(req, timeout=30).read())
    assert final["type"] == "final" and final["score"] == 0.0 and final["actions_used"] == 3
    recs = [json.loads(l) for l in (log_dir / "episode.jsonl").read_text().splitlines()]
    kinds = [r["type"] for r in recs]
    assert kinds[0] == "header" and kinds[-1] == "final"
    assert kinds.count("action") == 3 and "rejected" in kinds and "observe" in kinds
    # after finalisation the episode accepts no further actions
    res = asyncio.run(_session_calls(url, [("wait", {"steps": 10})]))
    assert res[1].isError
