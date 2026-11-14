#!/bin/zsh
# Failure-generation runs, part 2: DeepSeek and Kimi through Verda's OpenAI-compatible endpoint, 3 runs each,
# same settings as the Gemini runs. Runs once (mkdir lock), whichever of failure-gen.sh / the waiter gets here first.
cd "$(dirname "$0")/.."
mkdir runs/.failure-gen-verda.lock 2>/dev/null || exit 0
set -a; . ./.env; set +a
for spec in verda/deepseek-4-1-flash-1017:DEEPSEEK_AI_API_KEY verda/kimi-k3-1-pro:KIMI_AI_API_KEY; do
  m=${spec%%:*}; key=${spec##*:}
  echo "=== $m $(date)"
  HARBOR=$HOME/.local/bin/harbor .venv/bin/python scripts/eval.py --model "$m" --runs 3 --agents-only \
    --harbor-args "--agent-setup-timeout-multiplier 4 --base-url https://inference.verda.com/v1 --api-key-name $key --api-format openai_chat_completions"
done
echo "=== verda done $(date)"
