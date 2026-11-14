#!/bin/bash
# Oracle: replay a stored trajectory through the MCP server, exactly like an agent.
#   LOOPHOLE_SOLUTION=reference   (default) scripted reference solution
#   LOOPHOLE_SOLUTION=stop_early  stop-early baseline: a few plausible moves, then stop
#   LOOPHOLE_SOLUTION=noop        do nothing (same as `-a nop`)
set -euo pipefail
case "${LOOPHOLE_SOLUTION:-reference}" in
  reference)  TRAJ=/solution/reference.jsonl ;;
  stop_early) TRAJ=/solution/stop_early.jsonl ;;
  noop)       echo "noop baseline: no actions"; exit 0 ;;
  *) echo "unknown LOOPHOLE_SOLUTION=${LOOPHOLE_SOLUTION}" >&2; exit 2 ;;
esac
python3 /solution/replay_mcp.py "$TRAJ" --url "${LOOPHOLE_MCP_URL:-http://loophole-sim:8000/mcp}"
