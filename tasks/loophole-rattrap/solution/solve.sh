#!/bin/bash
# Oracle for the Rat Trap. The reference is Tibi's recorded human solve: fixed, committed waypoints
# (solution/waypoints.jsonl), followed closed-loop from the gripper's actual position, the same correction the offline
# conversion made, done live so cross-platform numerical differences do not accumulate. No search, no planning.
#   LOOPHOLE_SOLUTION=reference   (default) the recorded human solve
#   LOOPHOLE_SOLUTION=stop_early  stop-early baseline: grab + five random moves, then stop
#   LOOPHOLE_SOLUTION=noop        do nothing (same as `-a nop`)
set -euo pipefail
URL="${LOOPHOLE_MCP_URL:-http://loophole-sim:8000/mcp}"
case "${LOOPHOLE_SOLUTION:-reference}" in
  reference)  exec python3 /solution/follow_waypoints.py /solution/waypoints.jsonl --url "$URL" ;;
  stop_early) exec python3 /solution/replay_mcp.py /solution/stop_early.jsonl --url "$URL" ;;
  noop)       echo "noop baseline: no actions"; exit 0 ;;
  *) echo "unknown LOOPHOLE_SOLUTION=${LOOPHOLE_SOLUTION}" >&2; exit 2 ;;
esac
