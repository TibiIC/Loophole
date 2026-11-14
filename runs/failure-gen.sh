#!/bin/zsh
# Failure-generation runs on tasks/loophole-alpha: three cheaper Gemini models, 3 runs each, same settings as the
# Gemini 4 Carbon runs; one model at a time.
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
for m in gemini/gemini-3.8-flash gemini/gemini-3.5-flash-lite gemini/gemini-2.5-flash; do
  echo "=== $m $(date)"
  HARBOR=$HOME/.local/bin/harbor .venv/bin/python scripts/eval.py --model "$m" --runs 3 --agents-only --harbor-args "--agent-setup-timeout-multiplier 4"
done
echo "=== done $(date)"
# DeepSeek and Kimi through Verda (appended 2026-11-14 after the Verda probe made MCP tool calls)
zsh runs/failure-gen-verda.sh
