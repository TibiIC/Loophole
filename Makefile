# Loophole: one-command workflows. Requires Docker and Harbor (`uv tool install harbor`).
HARBOR ?= harbor
AGENT  ?= opencode
MODEL  ?= anthropic/claude-sonnet-5
RUNS   ?= 3
TASKS  ?= tasks
JOBS   ?= jobs
PY     ?= python3.11
STAMP  := $(shell date +%Y%m%d-%H%M%S)

.PHONY: help build dev test tasks alpha fetch-alpha fetch-alpha-paper view-alpha reference agent noop stop-early eval view clean

help:
	@echo "make build                         build the simulator / verifier image"
	@echo "make reference | noop | stop-early run the oracle / baselines through Harbor"
	@echo "make agent MODEL=<provider/model> [AGENT=opencode] [RUNS=3]"
	@echo "make eval MODEL=<provider/model>   build + reference + agent x RUNS + noop + stop-early"
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

# LOCAL ONLY: canonical OMPL.app Alpha meshes for comparison (Rice licence: never committed or shipped)
fetch-alpha:
	scripts/fetch_alpha.sh

# Alpha 1.0 as in Zhang et al. 2020 (meshes + .cfg + solution; fetched, never committed)
fetch-alpha-paper:
	scripts/fetch_alpha_paper.sh

# watch the oracle (the paper's published solution) in the MuJoCo viewer
view-alpha: fetch-alpha-paper
	.venv/bin/mjpython -m loophole.viewer --trajectory loophole/reference/trajectories/alpha-1.0-paper.jsonl

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

eval: build reference agent noop stop-early

view:
	$(HARBOR) view $(JOBS)

clean:
	rm -rf $(JOBS) .pytest_cache MUJOCO_LOG.TXT
