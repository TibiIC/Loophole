#!/bin/bash
# Harbor verifier entrypoint, baked into the loophole-sim image at /tests/test.sh.
# Runs in a separate container that receives only the sim sidecar's episode
# directory (as an artifact). It replays the logged actions in a fresh simulator
# and recomputes the reward; nothing the agent wrote is read.
set -uo pipefail
mkdir -p /logs/verifier
EPISODE_DIR="${LOOPHOLE_LOG_DIR:-/var/loophole/episode}"

python -m loophole.reward.verify \
  --log "${EPISODE_DIR}/episode.jsonl" \
  --puzzle "${LOOPHOLE_PUZZLE:?LOOPHOLE_PUZZLE must be set in [verifier.env]}" \
  --seed "${LOOPHOLE_SEED:-0}" \
  --out /logs/verifier/reward.json \
  --report /logs/verifier/report.json \
  || echo '{"reward": 0}' > /logs/verifier/reward.json

# keep the raw episode next to the reward (training data / debugging)
cp "${EPISODE_DIR}/episode.jsonl" /logs/verifier/episode.jsonl 2>/dev/null || true
cat /logs/verifier/reward.json
