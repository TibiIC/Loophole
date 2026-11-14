#!/bin/bash
# Harbor verifier entrypoint, baked into the loophole-sim image at /tests/test.sh.
# Runs in a separate container that receives only the sim sidecar's episode
# directory (as an artifact). It replays the logged actions in a fresh simulator
# and recomputes the reward; nothing the agent wrote is read. Output: one float in
# [0, 1] in /logs/verifier/reward.txt (details in /logs/verifier/report.json).
set -uo pipefail
mkdir -p /logs/verifier
EPISODE_DIR="${LOOPHOLE_LOG_DIR:-/var/loophole/episode}"

# The Rat Trap (tasks/loophole-rattrap) logs discrete commands with state hashes; its own verifier replays them.
if [ "${LOOPHOLE_TASK:-}" = "rattrap" ]; then
  python -m loophole.puzzles.rattrap.verify \
    --log "${EPISODE_DIR}/commands.jsonl" \
    --out /logs/verifier/reward.txt \
    --report /logs/verifier/report.json \
    || echo 0 > /logs/verifier/reward.txt
  cp "${EPISODE_DIR}"/commands.jsonl "${EPISODE_DIR}"/tools.jsonl /logs/verifier/ 2>/dev/null || true
  cat /logs/verifier/reward.txt
  exit 0
fi

python -m loophole.reward.verify \
  --log "${EPISODE_DIR}/episode.jsonl" \
  --puzzle "${LOOPHOLE_PUZZLE:?LOOPHOLE_PUZZLE must be set in [verifier.env]}" \
  --seed "${LOOPHOLE_SEED:-0}" \
  --out /logs/verifier/reward.txt \
  --report /logs/verifier/report.json \
  || echo 0 > /logs/verifier/reward.txt

# keep the raw episode next to the reward (training data / debugging)
cp "${EPISODE_DIR}/episode.jsonl" /logs/verifier/episode.jsonl 2>/dev/null || true
cat /logs/verifier/reward.txt
