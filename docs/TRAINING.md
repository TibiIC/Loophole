# Training on the Alpha puzzle

Evaluation is the Harbor task (`tasks/loophole-alpha`): one episode from the **canonical
start**, scored by an independent verifier. None of the training machinery below changes
it. Evaluation servers never expose `reset` or shaping rewards. Training logs (non-canonical
start or randomised scene) are rejected by the verifier.

## Protocol

| | training | evaluation (held-out test) |
|---|---|---|
| start state | sampled: perturbed canonical starts, or a reverse curriculum along the oracle path / roadmap | the canonical start (`alpha-1.0-paper.yaml`, from Zhang et al. 2020), only |
| scene | optionally randomised (orientation, friction, contact stiffness, observation noise) | fixed: frictionless, task contact parameters, no noise |
| reward | per-step shaping (`cspace_tunnel`: Φ = −cost-to-go, γ = 1) + the episode score | the episode score only (`/logs/verifier/reward.txt`) |
| episodes | many per process (`AlphaEnv.reset`, or MCP `reset` in training mode) | one per MCP server process |
| who scores | the env itself (`info["score"]`, same `episode_score` code) | the separate verifier replaying the log |

1. **Train on sampled starts only.** The canonical start is never sampled. `AlphaEnv(start="canonical")`
   and `reset(options={"start": "canonical"})` raise. Every sampled start is at least 0.05
   (SE(3) metric) away from it. In MCP training mode, actions are refused until the first
   `reset`.
2. **Select hyper-parameters on sampled starts** (e.g. a fixed set of seeds of
   `start="perturb"`), never on the canonical start.
3. **Evaluate only on the canonical start**, through the Harbor task (`make eval MODEL=…`) or
   `AlphaEnv(mode="eval")`. In eval mode the score equals the verifier's
   (`test_eval_mode_score_equals_verifier`). Report it once per final model.

## Gymnasium API (`loophole.gym`)

```python
from loophole.gym import AlphaEnv, make_vec_env

env = AlphaEnv(start="perturb")                         # mode="train" (default)
obs, info = env.reset(seed=0)                           # samples a start, grabs tube B (1 action)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# at episode end: info["score"] = the verifier's score for this episode's rules

venv = make_vec_env(8, start="curriculum", curriculum_distance=0.3)   # AsyncVectorEnv
test = AlphaEnv(mode="eval")                            # the canonical start, no randomisation
```

* **Action** in [−1, 1]⁶: translation × 0.1 m and rotation vector × 0.2 rad (the rig caps). It is
  applied as one `move` about the grasp point; motion stops early when blocked, as in the task.
* **Observation** (`observations=`): `pose` (tube B, x y z qw qx qy qz) + `proprio` (blocked,
  fraction of actions left, contact force magnitude, grip lag): proprioception and touch.
  Images: `rgb`, `depth` (metres), `segmentation` (0 background, 1 tube A, 2 tube B, 3
  handle) from an iso camera at `image_size`.
* **Reward** = the config's per-step training signal (`cspace_tunnel`, potential-based, so
  there-and-back nets zero; `clearance_shaping` at weight 0). `terminated` = free or
  violation; `truncated` = action budget (198).
* **Shaping scaffold** (`loophole.reward.components.PotentialShaping`, every potential component):
  F = λ(γ·Φ̄(s′) − Φ(s)) with Φ̄ = 0 once the episode has *terminated* (free or a violation:
  absorbing) and Φ kept at *truncation* (the budget: bootstrap V(s′) there). The configs use
  γ = 1, λ = 1. With γ < 1 (set it to the learner's discount, e.g. 0.99) only the *discounted*
  shaping return is path-independent (γᵀΦ̄(s_T) − Φ(s₀)); an undiscounted sum then pays
  (1 − γ)|Φ| per step for standing still. **The step reward is shaping only, and the score is
  kept separate** (`info["score"]`, the verifier's value). Because Φ̄ = 0 at every terminal state,
  the shaping return of *any* terminated episode is −Φ(s₀), crash and solve alike: add the task
  reward (`info["score"]` at the last step) to the shaping when training, or a violation ties a solve.
* `reset(options=…)`: `start`, `curriculum_distance`, `curriculum_source`, `log_path`.

### Start-state distribution (`loophole/training/starts.py`)

* `start="perturb"`: canonical start + N(0, σ_t = 5 cm) translation and a random rotation
  (σ_r = 0.1 rad). A sample is accepted only if it is collision-free, **still interlocked**
  (no free escape by translation), has roadmap cost-to-go ≥ 0.5 × the canonical start's,
  and is outside the canonical exclusion radius.
* `start="curriculum"` (reverse curriculum): `curriculum_distance` d ∈ [0, 1] from the goal.
  * `curriculum_source="oracle"`: along the authors' path, at path length (1 − d) × s\*.
    s\* is the path length up to the first free pose on the path (found once by bisection), so
    d = 0 is "just before free". A small jitter is added.
  * `"roadmap"`: roadmap nodes with cost-to-go / c₀ within ±0.05 of d.

  Same acceptance tests (collision-free, not yet free, not canonical). Start at d ≈ 0.1 and
  raise it as the success rate grows.

### Domain randomisation (flags, all off by default, training mode only)

| flag | effect |
|---|---|
| `randomize_orientation` | a random rotation per episode for the observation/action frame; images use a random camera azimuth |
| `randomize_friction` | Coulomb friction μ ~ U(0, 0.5) on the tubes (the task is frictionless) |
| `randomize_stiffness` | contact time constant × U(0.5, 2) |
| `obs_noise=σ` | Gaussian noise on pose (m / rad), pixels (σ × 255) and depth (m) |

The sampled parameters are written to the episode log (`header.training`). The exporter
re-applies them, so randomised episodes still replay bit-identically.

## MCP training mode (admin only)

```bash
.venv/bin/python -m loophole.mcp_server --puzzle alpha-1.0-paper \
    --config loophole/configs/alpha_training.yaml --training --log-dir runs/train-mcp
```

* `--training` (or `LOOPHOLE_TRAINING=1`) is never set by the Harbor tasks.
* It adds the tool `reset(seed, start, curriculum_distance, curriculum_source)`. Each call
  starts a new episode from a sampled training start and writes its own log
  (`episode-00001-seed3.jsonl`, …).
* Action results gain `shaping_reward`, per-component `rewards`, and `score` at the end.
* Evaluation mode (the default) is unchanged: one episode per process, no `reset`, no
  shaping (`test_mcp_eval_mode_has_no_reset_and_no_shaping`).

## Rollouts and traces

```bash
# scripted / Gym policies (in-process)
.venv/bin/python scripts/collect_rollouts.py --policy random --episodes 50 --start perturb
.venv/bin/python scripts/collect_rollouts.py --policy gym:my_pkg.policies:Policy --episodes 50 \
    --start curriculum --curriculum-distance 0.2 --randomize-friction
# any MCP client, e.g. an LLM agent, against a training-mode server ({url}, {seed} substituted)
.venv/bin/python scripts/collect_rollouts.py --policy 'mcp:python my_agent.py --mcp {url}' --episodes 10

# chat-SFT traces from any episode logs (training rollouts or evaluation runs)
.venv/bin/python scripts/export_traces.py 'runs/rollouts/*/episode-*.jsonl' -o traces/train.jsonl --min-score 0.5
```

`collect_rollouts.py` writes one simulator episode log per episode (the evaluation server's
format, plus `header.start_pose` / `header.training`) and `rollouts.jsonl` (seed, score,
success, actions, start). `export_traces.py` replays each log and re-runs every logged tool
call, observations included, to produce OpenAI-style `messages` with tool calls and tool
results (images as PNG files or data URLs) plus the exact tool schemas.

## Leakage checklist

* The canonical start is not in any training distribution (exclusion radius + explicit errors).
* The oracle path is used for *training starts* (reverse curriculum). It never appears in
  observations, and evaluation does not use it.
* Evaluation observations (`sensors` profile) show no goal test, distance, cost-to-go or
  escape direction. Those exist only as training rewards.
