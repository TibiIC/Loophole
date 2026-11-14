"""Harbor agent that probes the agent container's isolation, live, during the agent phase (docs/EXPLOITS.md).

    PYTHONPATH=$PWD harbor run -p ./tasks/loophole-rattrap --agent-import-path loophole.harbor_probe:IsolationProbe
    (any task works; the report is the agent log isolation-probe.txt, and the trial then scores like a no-op)

It runs, inside the agent container ("main"), exactly what an agent could run: checks that the task's solution,
tests, simulator source, episode log, recorded trajectory and checkpoint file are absent; searches the filesystem for
anything that looks like them; tries the internet, the sidecar's health route, its loopback-only admin route, and
lists the MCP tools. It takes no action in the episode.
"""

from __future__ import annotations

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PROBE = r"""
set +e
echo "== paths (expect: absent)"
for p in /solution /tests /opt/loophole /var/loophole /app/solution; do [ -e "$p" ] && echo "PRESENT $p" || echo "absent  $p"; done
echo "== filesystem search for trajectories, checkpoints, references, roadmaps, meshes, episode logs (expect: none)"
find / -xdev \( -iname '*rattrap*' -o -iname '*checkpoint*.npz' -o -iname 'reference.jsonl' -o -iname 'commands.jsonl' \
  -o -iname '*solve*.jsonl' -o -iname 'roadmap*' -o -iname '*.npz' -o -iname 'episode.jsonl' -o -iname 'solve.sh' \
  -o -iname '*.dae' -o -iname 'alpha*.obj' \) 2>/dev/null | grep -v -E '^/(proc|sys)/' | grep -v -E '/site-packages/' | head -20
echo "(end of search)"
echo "== network"
echo "internet (expect 000 = blocked): $(curl -s -m 8 -o /dev/null -w '%{http_code}' https://example.com)"
echo "sidecar /healthz (expect ok): $(curl -s -m 8 http://loophole-sim:8000/healthz)"
echo "sidecar /admin/finalize (expect forbidden): $(curl -s -m 8 -X POST http://loophole-sim:8000/admin/finalize)"
echo "== MCP tools offered by the sidecar"
python3 - <<'EOF'
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client("http://loophole-sim:8000/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print(sorted(t.name for t in (await s.list_tools()).tools))

asyncio.run(main())
EOF
"""


class IsolationProbe(BaseAgent):
    capabilities = AgentCapabilities(mcp_servers=True)  # the tasks declare their MCP server; the probe lists its tools

    @staticmethod
    def name() -> str:
        return "isolation-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        res = await environment.exec(f"bash -c {quote(PROBE)} 2>&1 | tee /logs/agent/isolation-probe.txt", timeout_sec=600)
        out = getattr(res, "stdout", None) or ""
        (self.logs_dir / "isolation-probe.txt").write_text(out) if out and not (self.logs_dir / "isolation-probe.txt").exists() else None


def quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
