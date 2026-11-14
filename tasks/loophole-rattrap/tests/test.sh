#!/bin/bash
# Reference copy only. The verifier image (tests/docker-compose.yaml) is the
# loophole-sim image, which bakes in /tests/test.sh from verifier/test.sh at the repository root and
# overrides this file. That script replays the episode log collected from the sim
# sidecar in a fresh simulator and writes /logs/verifier/reward.json.
exec python -m loophole.puzzles.rattrap.verify --log /var/loophole/episode/commands.jsonl --out /logs/verifier/reward.txt
