# Loophole: one-command workflows. Requires Docker and Harbor (`uv tool install harbor`).
HARBOR ?= harbor
AGENT  ?= opencode
MODEL  ?= anthropic/claude-sonnet-5
RUNS   ?= 3
TASKS  ?= tasks
JOBS   ?= jobs
PY     ?= python3.11
STAMP  := $(shell date +%Y%m%d-%H%M%S)

.PHONY: help build dev test tasks alpha fetch-alpha-paper view-alpha reference agent noop stop-early eval view clean rattrap-oracle rattrap-noop rattrap-random judge watch watch-last

help:
	@echo "make build                         build the simulator / verifier image"
	@echo "make reference | noop | stop-early run the oracle / baselines through Harbor"
	@echo "make agent MODEL=<provider/model> [AGENT=opencode] [RUNS=3]"
	@echo "make rattrap-oracle | rattrap-noop | rattrap-random   Rat Trap baselines (in-process)"
	@echo "make eval MODEL=<provider/model>   oracle + RUNS opencode runs + noop + stop-early -> runs/<ts>/summary.md"
	@echo "make dev && make test              local venv + pytest"

build:
	docker build -f docker/sim.Dockerfile -t loophole-sim:local .

dev:
	$(PY) -m venv .venv && .venv/bin/pip install -e ".[mesh,dev]"

test:
	.venv/bin/python -m pytest -q

# regenerate the Alpha-style puzzles, plans and roadmaps (offline, ~10 min; outputs are committed)
alpha:
	.venv/bin/python -m loophole.puzzles.alpha.generate
	.venv/bin/python scripts/build_alpha.py

# Alpha 1.0 as in Zhang et al. 2020 (meshes + .cfg + solution; fetched, never committed)
fetch-alpha-paper:
	scripts/fetch_alpha_paper.sh

# watch the oracle (the paper's published solution) in the MuJoCo viewer
# first run simulates once (~80 s) into .view_cache/; playback speed: SPEED=10, the
# `playback_speed` slider (right panel > Control) or the up/down arrows; space pauses
SPEED ?= 5
view-alpha: fetch-alpha-paper
	.venv/bin/mjpython -m loophole.viewer --trajectory loophole/reference/trajectories/alpha-1.0-paper.jsonl \
		--config loophole/configs/alpha_training.yaml --speed $(SPEED)

# regenerate trajectories + Harbor task dirs from puzzles/ and the scripted references
tasks:
	.venv/bin/python scripts/make_reference.py
	.venv/bin/python scripts/sync_tasks.py

reference: build
	$(HARBOR) run -p $(TASKS) -a oracle -o $(JOBS) --job-name reference-$(STAMP)

agent: build
	$(HARBOR) run -p $(TASKS) -a $(AGENT) -m $(MODEL) -k $(RUNS) -o $(JOBS) --job-name agent-$(AGENT)-$(STAMP)

noop: build
	$(HARBOR) run -p $(TASKS) -a nop -o $(JOBS) --job-name noop-$(STAMP)

stop-early: build
	$(HARBOR) run -p $(TASKS) -a oracle --ae LOOPHOLE_SOLUTION=stop_early -o $(JOBS) --job-name stop-early-$(STAMP)

# oracle + RUNS OpenCode runs + noop + stop-early on tasks/loophole-alpha -> runs/<timestamp>/summary.md
# API keys from the environment; a local .env (gitignored) is sourced if present.
eval: build
	set -a; [ -f .env ] && . ./.env; set +a; HARBOR=$(HARBOR) .venv/bin/python scripts/eval.py --model $(MODEL) --runs $(RUNS)

# One command for the whole evaluation of one task: reference, stop-early, noop, three agent runs (run_evaluation.sh)
judge:
	./run_evaluation.sh $(or $(TASK),alpha) $(if $(filter command line environment,$(origin MODEL)),$(MODEL),gemini/gemini-4-carbon)

# Live viewer (docs/watching.md): off for scored runs; WHO=oracle|random|deepseek|kimi|nemotron, PORT=8770
watch:
	PYTHONPATH=$(CURDIR) .venv/bin/python scripts/watch_rattrap.py --who $(or $(WHO),oracle) --port $(or $(PORT),8770)
# encode the most recently watched run's frames into an MP4 (no re-simulation) and open it
watch-last:
	@d=$$(ls -dt runs/watch/*/frames 2>/dev/null | head -1); [ -n "$$d" ] || { echo "no watched run in runs/watch/"; exit 1; }; \
	ffmpeg -y -loglevel error -framerate 4 -i "$$d/%06d.png" -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "$$d/../run.mp4" && \
	echo "$$d/../run.mp4" && (open "$$d/../run.mp4" 2>/dev/null || xdg-open "$$d/../run.mp4" 2>/dev/null || true)

# Rat Trap (in-process, no Docker): the oracle (reference solve), noop and a random agent, scored by the real reward
rattrap-oracle:
	PYTHONPATH=$(CURDIR) .venv/bin/python scripts/rattrap_baselines.py oracle
rattrap-noop:
	PYTHONPATH=$(CURDIR) .venv/bin/python scripts/rattrap_baselines.py noop
rattrap-random:
	PYTHONPATH=$(CURDIR) .venv/bin/python scripts/rattrap_baselines.py random --seed $(or $(SEED),0)

view:
	$(HARBOR) view $(JOBS)

clean:
	rm -rf $(JOBS) .pytest_cache MUJOCO_LOG.TXT
