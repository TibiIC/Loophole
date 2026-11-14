# Quickstart

Three ways in, from least to most setup: evaluate a model through the harness, drive the
simulator with your own agent, or train on the Alpha puzzle. Commands run from the repo root of
[TibiIC/Loophole](https://github.com/TibiIC/Loophole). Full details: the
[repo README](https://github.com/TibiIC/Loophole#readme) and [Training](TRAINING.md).

## 1. Evaluate a model (one command)

You need Docker running and a model key in the environment or in `./.env` (`.env*` is
gitignored). For Gemini through OpenCode the key must be named `GOOGLE_GENERATIVE_AI_API_KEY`;
for OpenAI, `OPENAI_API_KEY`.

```bash
git clone https://github.com/TibiIC/Loophole && cd Loophole
./run_evaluation.sh alpha   gemini/gemini-4-carbon     # or: make judge TASK=alpha MODEL=...
./run_evaluation.sh rattrap gemini/gemini-4-carbon     # or: make judge TASK=rattrap MODEL=...
```

The script installs Harbor (0.25.0) and `uv` if missing, builds the simulator image (the first
build fetches the Alpha meshes), then runs through Harbor:

1. the reference (Harbor's `oracle` agent replaying the task's `solution/`),
2. the stop-early baseline and the do-nothing baseline (`nop`),
3. three agent runs (OpenCode with your model; `RUNS=` and `N_CONCURRENT=` change that).

It prints one score per run and writes `jobs/judge-<task>-<time>/summary.md`. Every score comes
from a separate verifier container that replays the logged actions in a fresh simulator.

**Budget time.** The first image build takes 10–15 min. On Alpha the reference takes about 3 min
and an agent run about an hour. The Rat Trap is much slower: its reference replay is about
1.5 h, and an agent run can take up to 4 h plus verification.

### Single runs

```bash
harbor run -p ./tasks/loophole-alpha -a oracle                                  # reference solution
harbor run -p ./tasks/loophole-alpha -a nop                                     # do-nothing baseline
harbor run -p ./tasks/loophole-alpha -a opencode -m gemini/gemini-4-carbon     # one agent run
harbor run -p ./tasks/loophole-alpha-planner -a opencode -m gemini/gemini-4-carbon  # + planning queries
```

`tasks/loophole-alpha` gives the agent vision, proprioception and touch (`render`, `get_state`,
`get_contacts`); `tasks/loophole-alpha-planner` adds collision queries and the meshes
(`check_motion`, `check_poses`, `get_geometry`). Actions are `grab`, `move`, `release` and
`wait`. The agent's container holds no simulator code, meshes, solver or reference; it reaches
the simulator only over MCP.

## 2. Drive the simulator with your own agent

Any MCP client can act on the puzzle. For development, run a **training-mode** MCP server
locally; it adds a `reset` tool, so one server can serve many episodes:

```bash
make dev                  # python 3.11 venv + editable install
make fetch-alpha-paper    # Alpha 1.0 meshes, sha256-checked, into .alpha_paper/ (never committed)
.venv/bin/python -m loophole.mcp_server --puzzle alpha-1.0-paper \
    --config loophole/configs/alpha_training.yaml --training --log-dir runs/train-mcp
```

Or let the rollout collector start servers and call your agent once per episode (`{url}` and
`{seed}` are substituted):

```bash
.venv/bin/python scripts/collect_rollouts.py --policy 'mcp:python my_agent.py --mcp {url}' --episodes 10
```

Training mode is never enabled by the Harbor tasks. To score your agent for real, run it
through the Harbor task, as in section 1.

## 3. Train on the Alpha puzzle

The training tooling is built for Alpha 1.0. It needs the same `make dev` and
`make fetch-alpha-paper` as above.

```python
from loophole.gym import AlphaEnv, make_vec_env

env = AlphaEnv(start="perturb")              # training mode: sampled starts, never the canonical one
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# info["score"] at the end of an episode is the verifier's score for it

venv = make_vec_env(8, start="curriculum", curriculum_distance=0.3)   # reverse curriculum
test = AlphaEnv(mode="eval")                 # the canonical start: evaluation only
```

* **Action:** six numbers in [−1, 1], a translation (× 0.1 m) and a rotation (× 0.2 rad)
  of tube B about the grasp point, stopping early when blocked.
* **Observation:** `pose` by default; add `rgb`, `depth` or `segmentation` with
  `observations=(...)` and `image_size=`.
* **Reward:** per-step potential-based shaping on the cost-to-go, kept separate from the
  score. At the last step, add `info["score"]` to the shaping; otherwise a violation and a
  solve earn the same shaping return.
* **Starts:** `perturb` jitters the canonical start; `curriculum` starts along the published
  path or on the roadmap at distance `curriculum_distance` from the goal. Start near 0.1 and
  raise it as the success rate grows.
* **Domain randomisation** (off by default): orientation, friction, contact stiffness,
  observation noise.

**Rollouts and supervised traces:**

```bash
.venv/bin/python scripts/collect_rollouts.py --policy random --episodes 50 --start perturb
.venv/bin/python scripts/export_traces.py 'runs/rollouts/*/episode-*.jsonl' -o traces/train.jsonl --min-score 0.5
```

`export_traces.py` replays each episode log and writes OpenAI-style chat messages with the
tool calls and their results, from training rollouts or from evaluation runs.

**Keep evaluation held out.** Never train, or select hyper-parameters, on the canonical start:
the env refuses to sample it. Evaluate a final model once, through the Harbor task. The
protocol and leakage checklist are in [Training](TRAINING.md).
