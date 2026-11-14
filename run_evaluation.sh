#!/usr/bin/env bash
# One command: install what is needed, build, and run the full evaluation of one task through Harbor.
#
#   ./run_evaluation.sh [alpha|rattrap] [provider/model]        (default: alpha, gemini/gemini-4-carbon)
#   make judge TASK=rattrap MODEL=gemini/gemini-4-carbon
#
# Runs, in order: the reference (Harbor's `oracle` agent replaying solution/), the stop-early baseline, the do-nothing
# baseline (`nop`), and three agent runs (OpenCode with the given model). Prints one score per run and writes
# jobs/judge-<task>-<time>/summary.md. Needs Docker (running) and network for the first build (it fetches the Alpha
# meshes and Python packages). Model keys come from the environment or ./.env (e.g. GOOGLE_GENERATIVE_AI_API_KEY for
# gemini/...; see the README). Env knobs: RUNS (agent runs, default 3), N_CONCURRENT (parallel agent runs, default 3).
#
# Expected wall-clock (one machine, measured; see the README): alpha: build 10-15 min, oracle ~3 min, baselines < 1 min
# each, an agent run ~1 h. rattrap: oracle ~1.5 h (284 actions replayed, then re-verified), baselines a few min, an
# agent run up to 4 h plus up to ~2 h of verification.
set -euo pipefail
cd "$(dirname "$0")"

TASK="${1:-alpha}"
MODEL="${2:-${MODEL:-gemini/gemini-4-carbon}}"
RUNS="${RUNS:-3}"
N_CONCURRENT="${N_CONCURRENT:-3}"
HARBOR_VERSION="0.25.0"
case "$TASK" in
  alpha|loophole-alpha)     TASK_DIR=tasks/loophole-alpha; TASK=alpha ;;
  rattrap|loophole-rattrap) TASK_DIR=tasks/loophole-rattrap; TASK=rattrap ;;
  *) echo "usage: $0 [alpha|rattrap] [provider/model]" >&2; exit 2 ;;
esac
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="jobs/judge-${TASK}-${STAMP}"
mkdir -p "$OUT"

say() { printf '\n==> %s\n' "$*"; }

# 1. prerequisites -------------------------------------------------------------------------------
say "checking prerequisites"
command -v docker >/dev/null || { echo "Docker is required (https://docs.docker.com/get-docker/)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker is installed but not running" >&2; exit 1; }
if ! command -v harbor >/dev/null; then
  if ! command -v uv >/dev/null; then
    say "installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  say "installing harbor==$HARBOR_VERSION"
  uv tool install "harbor==$HARBOR_VERSION"
  export PATH="$HOME/.local/bin:$PATH"
fi
HARBOR="$(command -v harbor)"
echo "harbor: $("$HARBOR" --version 2>/dev/null || echo unknown)"
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# 2. build ---------------------------------------------------------------------------------------
say "building the simulator / verifier image (first build fetches the Alpha meshes; later builds are cached)"
docker build -f docker/sim.Dockerfile -t loophole-sim:local . | tail -2

# 3. runs ----------------------------------------------------------------------------------------
run() {  # name, then harbor arguments
  local name="$1"; shift
  say "$name"
  "$HARBOR" run -p "$TASK_DIR" -o "$OUT" --job-name "$name" "$@" || echo "(harbor exited non-zero for $name; see $OUT/$name)"
}
run reference  -a oracle
run stop-early -a oracle --ae LOOPHOLE_SOLUTION=stop_early
run noop       -a nop
run agent      -a opencode -m "$MODEL" -k "$RUNS" -n "$N_CONCURRENT" --agent-setup-timeout-multiplier 4

# 4. summary -------------------------------------------------------------------------------------
say "scores ($TASK_DIR, $OUT)"
{
  echo "| run | trial | score |"
  echo "|---|---|---|"
  for job in reference stop-early noop agent; do
    for t in "$OUT/$job"/*/; do
      [ -d "$t" ] || continue
      r="$(cat "$t/verifier/reward.txt" 2>/dev/null || echo 'no reward (see trial.log)')"
      echo "| $job | $(basename "$t") | $r |"
    done
  done
} | tee "$OUT/summary.md"
