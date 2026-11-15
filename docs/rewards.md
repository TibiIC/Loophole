# Reward design

This page defines every reward component of the Loophole suite, in both environments: `tasks/loophole-alpha`
(Alpha 1.0, Zhang et al. 2020; also `tasks/loophole-alpha-planner`) and `tasks/loophole-rattrap` (a closed string
loop with a mouse bead on a wooden frame). Every formula was checked against the code. Where the code and an older
document disagreed, the code is what is written here, and the disagreement is listed under
[Discrepancies found while writing this page](#discrepancies-found-while-writing-this-page).

**How the suite is built.** One score shape, one shaping rule and one violation rule are shared. Each environment
plugs in its own success test, its own partial-credit measure and its own shaping potential:

| | shared (S) | Alpha (A) | Rat Trap (R) |
|---|---|---|---|
| success | – | A1 free escape by translation | R1 frozen-shape escape |
| partial credit | S1 composite: $0.3\,p$, below every solve | A4 held cost-to-go progress (on A3) | R3 checkpoints in order |
| efficiency | S2 $1 - 0.5\,a/T$ | T = 198 | T = 852 |
| violations | S3 any violation scores 0 | A2 penetration 4 mm, speed, instability | R2 penetration 1 mm, release exception |
| shaping | S4 potential-based, $\bar\Phi = 0$ on termination | A5 tunnel potential (weight 1); A6 clearance (weight 0) | R4 checkpoint potential (+ violation penalty); R5 aperture alignment (weight 0) |
| diagnostics | S5 swept-motion certificate | A7 Gauss linking | R6 disc word and P&S norm (unsound) |

Every component also sits in one of three tiers, marked in its heading, and nothing crosses between tiers:

| tier | what it is | where it goes | standard |
|---|---|---|---|
| **scored** | the episode score, one float in [0, 1] | `/logs/verifier/reward.txt`, written by the verifier after replaying the episode | must be unexploitable |
| **shaping** | dense per-action training signals | the episode log (Alpha always; Rat Trap with `training=True`) | potential-based, never in the score |
| **diagnostic** | measured quantities for analysis | the episode log only | never rewarded, never scored |

> **⚠ Warning for anyone training on these environments: do not add undiscounted shaping to the score.**
> * Potential-based shaping telescopes. On *any* termination, solved or failed by a violation, the potential after
>   the last step is 0. So the shaping return is $-\lambda\,\Phi(s_0)$ either way: +21 for the Rat Trap, +1 for
>   Alpha (S4).
> * A learner that adds that undiscounted sum to the score, and treats the action-budget cut-off as terminal, would
>   prefer an early violation to most non-solves. A truncated non-solve collects only
>   $\lambda(\Phi(s_T) - \Phi(s_0))$, which lies between 0 and 21 (Rat Trap) and is at most 1 (Alpha).
> * Shaping never reaches the verifier, so scored runs are unaffected. This is about training.
> * **Rat Trap:** with `training=True`, a violation also pays `violation_penalty` $= -(\lambda(K+1) + 1) = -22$ on the
>   step it happens. A violation's shaping return is then −1, below every truncated non-solve's (0 or more).
>   It is one line in the training-only block of `RatTrapEpisode.act`, with no YAML key, so `rattrap.yaml`'s hash and
>   every committed log are unchanged. Tested by `test_rattrap_reward.py::test_violation_shaping_return_is_below_every_non_solve`.
> * **Alpha:** not changed. The switch would have to live in `loophole/configs/alpha_training.yaml`, and the verifier
>   compares that file against the configuration in every Alpha log's header, so changing it would make the
>   committed Alpha logs unverifiable. Bootstrap at the budget cut-off, or subtract your own penalty of more than
>   $\lambda\,\max_t c(q_t)/c_0$ on a violation (the cost-to-go can exceed $c_0$).

## Contents

* [The components at a glance](#the-components-at-a-glance), [How a score is produced](#how-a-score-is-produced)
* **Shared by both environments:** [S1 composite score](#s1-the-composite-score-and-why-every-solve-beats-every-non-solve-scored),
  [S2 efficiency](#s2-the-efficiency-term-scored), [S3 violation rule](#s3-the-violation-rule-forcing-scores-0-scored),
  [S4 potential-based shaping](#s4-potential-based-shaping-generally-shaping),
  [S5 swept-motion certificate](#s5-the-swept-motion-certificate-diagnostic)
* **Alpha:** [A1 free escape](#a1-success-free-escape-by-translation-scored), [A2 violation checks](#a2-alphas-violation-checks-scored),
  [A3 cost-to-go](#a3-c-space-cost-to-go-input-to-a4-and-a5), [A4 held progress](#a4-partial-credit-held-cost-to-go-progress-scored),
  [A5 tunnel potential](#a5-the-c-space-tunnel-potential-cspace_tunnel-shaping-weight-1),
  [A6 clearance shaping](#a6-clearance-shaping-clearance_shaping-shaping-weight-0),
  [A7 Gauss linking](#a7-the-gauss-linking-diagnostic-gauss_linking_diagnostic-diagnostic),
  [A8 planner task](#a8-the-planner-task-tasksloophole-alpha-planner-adds-no-reward)
* **Rat Trap:** [R1 frozen-shape escape](#r1-success-frozen-shape-escape-scored),
  [R2 violation check](#r2-the-rat-traps-violation-check-and-the-release-exception-scored),
  [R3 checkpoint progress](#r3-partial-credit-checkpoint-progress-scored),
  [R4 checkpoint potential](#r4-the-checkpoint-shaping-potential-and-the-training-violation-penalty-shaping),
  [R5 aperture alignment](#r5-aperture-alignment-aperture_alignment-shaping-weight-0),
  [R6 disc word and P&S norm](#r6-the-disc-word-and-the-przytyckisikora-norm-unsound-diagnostic)
* [Inactive and legacy components](#inactive-and-legacy-components) ·
  [Discrepancies found while writing this page](#discrepancies-found-while-writing-this-page) ·
  [Writing your own reward](#writing-your-own-reward)

## The components at a glance

| id | component | config key / code | Alpha | Rat Trap | tier |
|---|---|---|---|---|---|
| S1 | composite score | `episode_score` (`reward/literature_rewards.py`); `RatTrapEpisode.score` | terminal of `cspace_tunnel` | on | scored |
| S2 | efficiency, 1 − α·a/T | same | T = 198 | T = 852 | scored |
| S3 | violation rule | `sim/audit.py`; `puzzles/rattrap/sim.py` | A2 | R2 | scored |
| S4 | potential-based shaping | `reward/components.py: PotentialShaping` | base class | base class | shaping |
| S5 | swept-motion certificate | `sim/certify.py`; `puzzles/rattrap/certify.py` | rigid, off | cord, `log` | diagnostic |
| A1 | free escape by translation | `goal.criterion: escape`; `sim/collision.py: escape_direction` | on | – | scored |
| A2 | penetration, speed, instability | `audit:` in `default_config.yaml` | on | – | scored |
| A3 | C-space cost-to-go | `reward.roadmap: true`; `reward/roadmap.py` | on | – | input |
| A4 | held cost-to-go progress | `progress_cap` of `cspace_tunnel`; `StableMin` | on | – | scored |
| A5 | C-space tunnel potential | `cspace_tunnel` | weight 1 | – | shaping |
| A6 | clearance shaping | `clearance_shaping` (alias `IndustRealSDFReward`; `industreal_sdf` is not a key) | weight 0 | – | shaping |
| A7 | Gauss linking | `gauss_linking_diagnostic` (alias `GaussLinkingPotentialReward`; `gauss_linking` is not a key) | logged | – | diagnostic |
| A8 | planner task | `alpha_planner.yaml`: same reward block, more observation tools | – | – | – |
| R1 | frozen-shape escape | `escape_directions`; `FrozenEscape` | – | on | scored |
| R2 | penetration and release exception | `penetration_tolerance`, `release_*` | – | on | scored |
| R3 | checkpoint progress | `checkpoints_file`, `score_progress_cap`; `CheckpointTracker` | – | on | scored |
| R4 | checkpoint potential, violation penalty | `shaping_gamma`, `shaping_lambda`; `CheckpointPotential` (`rattrap_checkpoints`) | – | λ = 1 (`training=True`) | shaping |
| R5 | aperture alignment | `shaping_alignment` (registered as `aperture_alignment`) | not used | weight 0 | shaping |
| R6 | disc word, P&S norm | `topology_alphabet`; `topology.py`, `norm.py` | – | logged | diagnostic |

Paths are relative to `loophole/`. "Weight 0" means computed and logged but contributing nothing. Everything passed
to `register_component` in the codebase is here: `cspace_tunnel`, `clearance_shaping`, `gauss_linking_diagnostic` and
`aperture_alignment`; the five built-in registry entries are under [Inactive and legacy
components](#inactive-and-legacy-components).

## How a score is produced

The score is never read from anything the agent or the simulator server wrote. After the agent's container stops, a
separate verifier container with no network replays every logged action in a fresh simulator. It recomputes the
score from the replayed physics:

* **Alpha:** `python -m loophole.reward.verify` (`reward/verify.py`). The log header must name the same puzzle file
  (SHA-256), seed, effective configuration and roadmap (SHA-256). The initial state hash and every per-action state
  hash must match the replay. Then `LoopholeEnv.finalize()` computes the score through the `RewardComposer`.
* **Rat Trap:** `python -m loophole.puzzles.rattrap.verify`. It needs the same `rattrap.yaml` (SHA-256), the same
  checkpoint file (SHA-256), no overrides, the same start state, only `grab_node` / `move` / `release`, and matching
  state hashes. Then `RatTrapEpisode.score()` computes the score of the replay.

Anything else scores 0, with the reason in the report. The worked examples below are taken from committed runs. The
inputs are the fields logged by the verifier's replay, and the arithmetic is redone by hand.

---

## Shared by both environments

Shaping is a training signal. It never enters the score: for Alpha it is the per-action `rewards` in the episode
log, and for the Rat Trap it is `shaping` / `shaping_parts` when the episode is created with `training=True`. Scored
Rat Trap runs, including every Harbor run, do not compute it.

### S1. The composite score, and why every solve beats every non-solve (scored)

Putting the violation rule (S3), success (A1 / R1), efficiency (S2) and partial credit (A4 / R3) together, the
score of an episode is

$$
S =
\begin{cases}
0 & V > 0 \\
1 - \alpha\,\min(a, T)/T & \text{solved} \\
\kappa\, p & \text{otherwise}
\end{cases}
\qquad \alpha = 0.5,\ \kappa = 0.3,
$$

with $p$ from A4 (Alpha) or R3 (Rat Trap). In code: `episode_score` (Alpha, returned as the
`cspace_tunnel` terminal value and clipped to [0, 1] by `RewardComposer.score`) and `RatTrapEpisode.score`.

**Every solve beats every non-solve.** $\min S_{\text{solved}} = 1 - \alpha = 0.5 > 0.3 = \kappa \ge \max
S_{\text{unsolved}}$. `CSpaceTunnelPotentialReward` and `ScaledProgressScore` refuse parameters with
$\kappa \ge 1 - \alpha$. The tests cover it:
* `test_rattrap_checkpoints.py::test_every_solve_beats_every_non_solve_on_a_grid`;
* `test_reward.py::test_composer_keeps_solve_above_non_solve`.

| run | outcome | S |
|---|---|---|
| Alpha reference (59 actions) | solved | 0.851 |
| Alpha, Gemini 4 Carbon, run 1 (`sbMy6Ld`, 45) | solved | 0.886 |
| Alpha, Gemini 4 Carbon, original interface, run 3 (`Kt5DJ3h`, 157) | unsolved, p = 0.465 | 0.139 |
| Rat Trap reference (284) | solved | 0.833 |
| Rat Trap, fixed commands on Linux (284) | unsolved, 18 / 20 | 0.27 |
| do nothing, stop early (both) | unsolved, p = 0 | 0 |
| any violation (both) | – | 0 |

### S2. The efficiency term (scored)

**Config.** Both environments.

```yaml
# Alpha: loophole/configs/alpha_training.yaml (params of the scored component)
- type: cspace_tunnel
  params: {alpha: 0.5, progress_cap: 0.3}
# Alpha's budget: loophole/puzzles/alpha_paper/alpha-1.0-paper.yaml
max_actions: 198

# Rat Trap: loophole/puzzles/rattrap/rattrap.yaml
max_actions: 852
score_alpha: 0.5
```

**Definition.** For a solved episode without violations,

$$
S_{\text{solved}} = 1 - \alpha\,\frac{\min(a, T)}{T}, \qquad S_{\text{solved}} \in [1-\alpha,\ 1],
$$

* $a$: actions used (every physics tool call counts one; observations are free);
* $T$ = `max_actions`, the action budget: 198 for Alpha, 852 for the Rat Trap. Both are about 3 × the
  reference solve (59 and 284 actions);
* $\alpha$ = 0.5.

**Intuition.** A solve is worth between 0.5 and 1: the faster, the more. Time is counted in actions, never in
seconds, so a slow model or a slow machine is not penalised. The penalty is linear and capped at half the score, so
even a solve on the last action is worth more than the best possible non-solve (0.3).

**Precisely.**
* **What it measures:** the number of physics actions.
* **What it ignores:**
  * observations (`render`, `get_state`, contacts);
  * wall-clock time and tokens;
  * how close the path was to the reference;
  * how the actions were spent.

**Worked examples.**
* Alpha, Gemini 4 Carbon trial `runs/20261114-111735/opencode/loophole-alpha__sbMy6Ld` (`final` record):
  solved, $a$ = 45, $T$ = 198. $S = 1 - 0.5 \cdot 45/198 = 1 - 0.113636 = 0.886364$; logged `score` 0.886364.
* Alpha reference: $a$ = 59. $S = 1 - 0.5 \cdot 59/198 = 0.851010$; logged 0.85101.
* Rat Trap reference: $a$ = 284, $T$ = 852. $S = 1 - 0.5 \cdot 284/852 = 1 - 0.166667 = 0.833333$.

**Provenance.** Ours.

### S3. The violation rule: forcing scores 0 (scored)

**Config.** Both environments; it cannot be switched off. The checks themselves differ per environment: A2 (Alpha) and
R2 (Rat Trap).

**Definition.** Let $V$ be the number of violations recorded up to now. The episode ends at the first violation, and

$$
V > 0 \;\Longrightarrow\; S = 0 .
$$

Every check runs after every physics substep (1 ms in both environments), never sampled more sparsely. A violation
is the absorbing failure state of S4.

**Intuition.** You may not push one solid through another. The simulator's contacts are soft: two bodies pressed
together overlap a little, and the overlap grows the harder they are pressed. Past a small overlap, the only way to
get there is by forcing, so it counts as cheating and the episode is over with nothing. The rig itself stops a move
when the held piece is obstructed (Alpha's rig stops pushing at 2 mm, half the threshold; the Rat Trap's halts
the move on contact), so honest play does not get there.

**Precisely.**
* **What it measures:** the depth of the deepest contact after every substep, against a per-environment threshold;
  Alpha adds speed and integrator checks (A2).
* **What it ignores:**
  * which body is at fault;
  * how long the penetration lasted;
  * anything between two substeps. The swept-motion certificate (S5) covers that, in log mode.

**Worked examples:** A2 (Alpha: every committed run's deepest contact against its threshold) and R2 (Rat Trap: the
release exception).

**Provenance.** Ours. The thresholds are design choices: 4 mm is the Alpha escape sweep's step, and 1 mm is below the
1.2 mm cord radius.

### S4. Potential-based shaping, generally (shaping)

**Config.** The base class of every shaping component; its parameters are those of each subclass.

```yaml
# Alpha: on each component
params: {gamma: 1.0, lam: 1.0}
# Rat Trap: rattrap.yaml
shaping_gamma: 1.0     # must equal the learner's discount for policy invariance
shaping_lambda: 1.0
```

**Definition** (`reward/components.py: PotentialShaping.step`).

$$
F(s, s') = \lambda\,\bigl(\gamma\,\bar\Phi(s') - \Phi(s)\bigr), \qquad
\bar\Phi(s') =
\begin{cases}
0 & s' \text{ terminated (solved, or a violation)} \\
\Phi(s') & \text{otherwise, including truncation at the budget}
\end{cases}
$$

* $\Phi$: the subclass's potential, a function of the measured state only;
* $\gamma \in (0, 1]$: the learner's discount;
* $\lambda \ge 0$: a scale.

If $\Phi(s)$ or $\Phi(s')$ is unavailable (`None`), $F$ = 0 for that step. The terminal value of every shaping
component is 0.

Over an episode $s_0, \dots, s_T$ the discounted shaping return telescopes:

$$
\sum_{t=0}^{T-1} \gamma^{t}\, F(s_t, s_{t+1}) = \lambda\bigl(\gamma^{T}\,\bar\Phi(s_T) - \Phi(s_0)\bigr) .
$$

**Intuition.** A potential is a score for "how good is this state", and the shaping reward is its change. Climbing
earns what falling gives back. However an episode wanders, the shaping it collects depends only on where it
started and where it ended, so going round in circles earns nothing. Ng, Harada and Russell proved that rewards of
exactly this form never change which policy is optimal; they only make it easier to find.

**Precisely.**
* **What it measures:** the change of $\Phi$.
* **What it ignores:** everything $\Phi$ ignores.
* **Termination versus truncation:**
  * Solving and failing by a violation are absorbing, so their potential is 0.
  * Running out of actions is a cut-off from outside, so $\Phi$ is kept and a bootstrapping learner sees the
    same returns as without the limit.
* **With $\gamma < 1$ only the discounted sum is invariant.** Standing still at $\Phi < 0$ earns
  $(1-\gamma)\lvert\Phi\rvert$ per step in an undiscounted sum.
* **A caveat for learners that do not bootstrap.** Because $\bar\Phi$ = 0 for *both* absorbing outcomes, a
  violation collects the same terminal shaping, $-\lambda\,\Phi(s_{T-1})$, as a solve from the same state.
  * Policy invariance holds for a learner that bootstraps at truncation; the score still separates the two
    outcomes (0 against at least 0.5).
  * A learner that simply adds undiscounted shaping to the score, treating truncation as termination, would rank
    an early violation above a long truncated non-solve. See the warning at the top of this page. The Rat Trap's
    training mode now adds an explicit violation penalty, so a violation's shaping return there is −1. Alpha's
    is unchanged.

**Worked example** (`loophole/tests/test_shaping_scaffold.py::test_terminal_potential_is_zero_and_truncation_keeps_it`).
Take $\Phi(s) = -(\text{number of } a\text{'s in the word})$, $\gamma$ = 0.99, $\lambda$ = 1, $s$ = `aaa`
($\Phi = -3$) and $s'$ = `aa` ($\Phi = -2$):

| $s'$ is | $\bar\Phi(s')$ | $F$ |
|---|---|---|
| an ordinary state | −2 | 0.99 × (−2) + 3 = 1.02 |
| solved | 0 | 0 + 3 = 3 |
| a violation | 0 | 0 + 3 = 3 |
| truncated at the budget | −2 | 1.02 |

The oscillation `aa → a → aaa → a → aa` with $\gamma$ = 1 sums to exactly 0
(`test_discounted_shaping_depends_only_on_the_endpoints`). For the real-environment examples see A5 and R4.

**Provenance.**
* [NHR99] (policy invariance).
* The $\gamma$ / $\lambda$ scaffold and the terminal-versus-truncation rule are adapted from a colleague's
  ring-on-wire environment [HO26] (`DisentanglementPuzzle.step` / `potential`; their defaults $\lambda$ = 1,
  $\gamma$ = 0.99).

### S5. The swept-motion certificate (diagnostic)

**Config.** Rat Trap, `log` mode: uncertified substeps are counted and reported (`uncertified_substeps`), not
violations. Alpha has it available (`LoopholeEnv.enable_motion_certificate`) and off.

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
certify_motion: log    # off | log | enforce (enforce: an uncertified substep is a violation, S3)
```

**Definition** (`sim/certify.py`). For the interval between two consecutive substep states, with midpoint
configuration $m$ and sweep bound $b$ (an upper bound on how far any point of the moving body travels between $m$
and either end):

$$
\mathrm{clear}(m) > b - \text{slack} \;\Longrightarrow\; \text{certified}; \quad\text{otherwise bisect, to depth } 12 ,
$$

* cord (nodes moving linearly): $b = \max_i \lVert\Delta x_i\rVert / 2$. A point of a segment is a convex
  combination of its end nodes, so it moves no more than they do;
* rigid body: $b = \lVert\Delta p\rVert/2 + r_{\text{out}}\lVert\Delta\theta\rVert/2$;
* slack = `penetration_tolerance` (1 mm).

A midpoint in contact, or the depth limit, leaves the interval uncertified.

**Intuition.** The penetration check looks at snapshots. Between two snapshots, a thin strand could in principle
hop over a thin wall: the cord is 1.2 mm in radius and the hole walls are 1.5 mm thick. The certificate proves,
interval by interval, that nothing came closer to the structure than the tolerance anywhere in between.

**Precisely.**
* **What it measures:** a conservative proof of clearance over each substep interval. It may fail to certify a
  safe motion; it never certifies an unsafe one.
* **What it ignores:** cord against cord (self-passage), which stays with the physics. Every pair of non-adjacent
  segments collides (EXPLOITS.md R9).

**Worked example** (the reference replay): every substep interval of the 284 actions was certified: `uncertified_substeps` = 0
at the end.

**Provenance.** The recursive-bisection certificate and its rigid sweep bound are adapted from `motion_safe` in
[HO26]. Our changes:
* a clearance predicate instead of a clearance value;
* the slack;
* the polyline variant for the cord.

---

## Alpha (`tasks/loophole-alpha`, `tasks/loophole-alpha-planner`)

### A1. Success: free escape by translation (scored)

**Config.** Alpha only.

```yaml
# loophole/configs/alpha_training.yaml
goal:
  criterion: escape
  escape_directions: 64
```

**Definition.** Let $q = (t, R)$ be tube B's pose relative to tube A. Then

$$
\text{solved}(q) \iff \exists\, u \in U_{64} :\ \forall s \ge 0,\ \ B(t + s\,u,\ R) \cap A = \varnothing ,
$$

* $U_{64}$: 64 near-uniform unit vectors, the Fibonacci lattice
  $z_i = 1 - 2(i+\tfrac12)/64$, $\phi_i = \pi(1+\sqrt5)(i+\tfrac12)$ (`sim/collision.py: fibonacci_directions`);
* $B(\cdot)$, $A$: the two tubes as solids; the orientation $R$ is held fixed.

The sweep along $u$ (`CollisionChecker.escape_direction`) runs until the bounding spheres of the two pieces are
disjoint, past which no contact is possible:
* far from contact it advances by $d - \tau$, with $d$ the separation distance and $\tau$ = 1 mm (conservative
  advancement [Mir96]: a translation shorter than the distance cannot touch);
* within $4h$ of contact ($h$ = 4 mm) it switches to the simulator's exact validity test every $h$;
* it gives up on $u$ after 50 such steps (20 cm of sliding contact).

The test runs after every action. The first action at which it succeeds ends the episode as solved.

**Intuition.** The puzzle is solved when tube B could simply slide away along some straight line without touching
tube A. There is no target position and no "far enough": a single free direction to infinity is the definition of
"apart".

**Precisely.**
* **What it measures:** whether a pure translation to infinity exists at the current orientation.
* **Sound:** a reported direction is free. The far-field steps use a lower-bound distance; near contact the test is
  exact at every 4 mm.
* **What it ignores:**
  * escapes that need a rotation on the way out. Such a state is not yet solved; one more action usually makes it so;
  * directions between the 64 sampled ones;
  * how far apart the tubes currently are.
* **Missing an escape only delays success**, by an action or more; it never grants a false one. EXPLOITS.md #19
  checks a reported direction with a dense 1 mm sweep.

**Worked example** (`runs/20261114-111735/oracle/loophole-alpha__fVDVriY/verifier/episode.jsonl`, the Alpha
reference solve):
* After action 58 no direction was free.
* After action 59 the test found $u = (0.9533, 0.1436, 0.2656)$, logged as `goal.escape_direction`, so `success`
  became true and the episode ended.
* The cost-to-go at that pose was still 5.72 (A3): the tube had a straight way out long before the
  roadmap's goal set, which counts only poses more than 0.4 m apart.

**Provenance.** Removability of a part by a translation to infinity is the classic assembly-planning test
[WL94]. Conservative advancement is from [Mir96].

### A2. Alpha's violation checks (scored)

**Config.**

```yaml
# loophole/default_config.yaml (not overridden by loophole/configs/alpha_training.yaml)
audit:
  penetration_threshold: 0.004       # m
  anchor_drift_threshold: 0.005      # m (rope puzzles; Alpha has no anchored rope)
  max_substep_displacement: 0.003    # m per physics substep
rig:
  stall_penetration: 0.002           # m: the rig stops pushing here, before the audit threshold
```

**Definition.** (`sim/audit.py: Auditor.check`, after every physics substep). One violation is recorded per failed check:

$$
\max_{c}\,\bigl(-\mathrm{dist}_c\bigr) > p_{\max}, \qquad
\max_{b}\,\lVert x_b(t) - x_b(t-\Delta t)\rVert > v_{\max}, \qquad
N_{\text{warn}}(t) > N_{\text{warn}}(t - \Delta t),
$$

* $\mathrm{dist}_c$: the signed distance of MuJoCo contact $c$ (negative when penetrating);
* $p_{\max}$ = `penetration_threshold` = 4 mm;
* $x_b$: the position of each moving body, after each substep of length $\Delta t$;
* $v_{\max}$ = `max_substep_displacement` = 3 mm;
* $N_{\text{warn}}$: MuJoCo's counters of bad accelerations, velocities and positions (integrator divergence, after
  which MuJoCo silently resets the state).

**Intuition.** Three ways to cheat a contact check, three checks: press through (penetration), move so fast that
two consecutive checks straddle a thin part (speed), or blow up the integrator so MuJoCo resets the state
(instability). The rig stops pushing at 2 mm, half the threshold, so honest pressing against a wall never reaches it.

**Precisely.**
* **What it measures:** contact depth, the per-substep displacement of every moving body, and MuJoCo's divergence
  counters.
* **What it ignores:** which body is at fault, and how long a penetration lasted.

**Worked example** (all 24 committed Alpha runs with at least one action; the `violations` field of each `final`
record in `verifier/episode.jsonl`):
* 941 blocked moves in total, and 0 violations.
* The deepest contact of any run is 2.085 mm, in DeepSeek V4.1 Flash, run 1 (`runs/20261114-200129/opencode/...`,
  148 blocked moves): $2.085 \le p_{\max} = 4$, so not a violation. It sits just above the rig's 2 mm stall
  threshold, because the rig checks after each substep.
* The next deepest are 1.082 mm (Kimi K3 Pro, run 3) and 0.899 mm (Gemini 3.8 Flash, run 2).

**Provenance.** Ours.

### A3. C-space cost-to-go (input to A4 and A5)

The navigation function that A4 and A5 are built on. It is computed after every Alpha action.

**Config.** Alpha only.

```yaml
# loophole/configs/alpha_training.yaml
reward:
  roadmap: true                # load loophole/puzzles/alpha_paper/roadmap-alpha-1.0-paper.npz
  auto_roadmap_components: false
```

**Definition.**

$$
d(q_1, q_2) = \lVert t_1 - t_2 \rVert + w \cdot 2\arccos\lvert \langle \mathbf q_1, \mathbf q_2 \rangle \rvert ,
$$

$$
c(q) = \min_{n \,\in\, N_{10}(q),\ \overline{q\,n} \subset \mathcal C_{\text{free}}} \bigl[\, d(q, n) + \mathrm{ctg}(n) \,\bigr] ,
$$

* $q = (t, \mathbf q)$: tube B's pose relative to tube A, translation $t$ and unit quaternion $\mathbf q$. The second
  term of $d$ is $w$ times the rotation angle between the two orientations;
* $w$ = 1.5408645980226425 m, tube B's characteristic radius. A pure rotation by $\theta$ moves no point of B by
  more than $w\theta$, so $d$ bounds every point's motion;
* the roadmap: 8,044 collision-free poses and 60,260 collision-checked edges;
* $\mathrm{ctg}(n)$: the Dijkstra distance along roadmap edges from node $n$ to the nearest of the 847 goal nodes,
  the nodes whose tubes are more than $d_{\text{sep}}$ = 0.4 m apart;
* $N_{10}(q)$: the 10 roadmap nodes nearest to $q$ under $d$;
* $\overline{q\,n} \subset \mathcal C_{\text{free}}$: the straight segment from $q$ to $n$ is collision-free, checked
  at resolution 0.004 (`CollisionChecker.segment_valid`).

If no candidate is visible, $c$ is unavailable: the previous value is kept and logged as a fallback.

**Intuition.** How far is tube B, measured along the shortest way out that does not go through tube A? Straight-line
distance is useless here: the tubes start interlocked, and the exit is a long, narrow tunnel through the space
of positions and orientations. So we laid a road network over that space offline, with every road checked
collision-free, and measured each junction's distance to the exit along the roads. At play time, the pose is
connected to the best nearby junction it can reach in a straight line without hitting anything.

**Precisely.**
* **What it measures:** an upper-bound estimate of the shortest collision-free path length to the separated set,
  through the roadmap.
* **The visibility test is what stops "the nearest node is through the wall".**
* **What it ignores:**
  * paths the roadmap does not contain. It can overestimate the true distance;
  * dynamics and the rig's limits;
  * how the agent would actually execute the motion.
* **It need not reach 0 at a solve.** The goal set is "0.4 m apart", while success (A1) is "a straight way
  out". The reference solves with $c$ = 5.72.

**Worked example** (`runs/20261114-111735/oracle/loophole-alpha__fVDVriY/verifier/episode.jsonl`, action 2):
* The start pose is roadmap node 0: $c_0 = 0 + 21.606657 = 21.606657$.
* After action 2, among the 10 nearest nodes, sorted by $d + \mathrm{ctg}$:

  | node | $d(q, n)$ | $\mathrm{ctg}(n)$ | sum |
  |---|---|---|---|
  | 26 | 0.000253 | 21.294721 | 21.294974 |
  | 27 | 0.013575 | 21.282724 | 21.296298 |
  | 28 | 0.025666 | 21.270726 | 21.296392 |
  | 29 | 0.037770 | 21.258729 | 21.296499 |

* From the replayed pose the straight segments to nodes 26–28 fail the visibility test, and the one to node 29
  passes. So $c = 0.037770 + 21.258729 = 21.296499$, logged `cost_to_go` 21.29649859 (`ctg_fallback` false).
* The $d$ values are computed from the logged `rel_pose`, which is rounded to $10^{-6}$; node 29's $d$ is the
  logged $c$ minus its ctg.

**Provenance.**
* The tunnel structure of the Alpha puzzle's C-space is the subject of [ZBKV20].
* A navigation function rather than distance to one reference path, because the puzzle family has more than one
  homotopy class of solutions [JSCP15, §6].
* The roadmap and its visibility-guarded query are ours (`reward/roadmap.py: build_roadmap`, `Roadmap.cost_to_go`).
  It is a PRM: the densified reference path, Gaussian samples in a tube around it and uniform samples, with
  k-nearest edges validated densely.

### A4. Partial credit: held cost-to-go progress (scored)

**Config.** Alpha only. It is the `progress_cap` of the scored `cspace_tunnel` component, together with the hold
length of `StableMin` (fixed at 3 in `sim/env.py`).

```yaml
- type: cspace_tunnel
  weight: 1.0
  in_score: true
  params: {gamma: 1.0, alpha: 0.5, progress_cap: 0.3}
```

**Definition.** For an unsolved episode without violations,

$$
S_{\text{unsolved}} = \kappa\, p, \qquad
p = \mathrm{clip}\!\left(1 - \frac{c^{*}}{c_0},\ 0,\ 1\right), \qquad
c^{*} = \min\!\Bigl(c_0,\ \min_{t}\ \max\bigl(c_{t-2},\, c_{t-1},\, c_t\bigr)\Bigr),
$$

* $\kappa$ = `progress_cap` = 0.3;
* $c_t$: the cost-to-go (A3) after action $t$;
* the inner minimum runs over every window of 3 consecutive actions that all had a visible roadmap node. An action
  with no visible node (a fallback) breaks the window (`StableMin.update(None)` clears it);
* $c_0$: the cost-to-go of the start pose, 21.606657.

**Intuition.** A non-solve earns up to 0.3 for how far along the way out it got. Two refinements:
* **The level must be held.** It counts only if three actions in a row stayed at or below it, so a single lucky
  reading, or a nearest-node glitch, earns nothing.
* **Only the best held level counts.** Wiggling back and forth cannot add to it.

Doing nothing scores 0, because $c^* = c_0$ and $p = 0$.

**Precisely.**
* **What it measures:** the best sustained progress along the roadmap's way out, as a fraction of the start's
  distance.
* **What it ignores:**
  * where the episode ended: backing off afterwards costs nothing;
  * how many actions it took;
  * any route the roadmap does not cover.
* **Monotone.** It can never decrease during an episode.

**Worked example** (`runs/20261114-080705/opencode/loophole-alpha__Kt5DJ3h/verifier/episode.jsonl`, `final` record):
* Gemini 4 Carbon, on the original harness, which still exposed `cost_to_go` to the agent (since removed; rule 3
  under [Writing your own reward](#the-rules)). It ran finite-difference descent on it (`docs/RESULTS.md`).
* The agent stopped after 157 actions, unsolved, with no violation and 42 fallbacks.
* Its best single reading was 11.253124, but the best level held for three consecutive actions was
  $c^*$ = 11.566060.
* $p = 1 - 11.566060/21.606657 = 1 - 0.535305 = 0.464695$, so $S = 0.3 \times 0.464695 = 0.139409$; logged `score`
  0.13941.
* Without the hold the score would have been $0.3\,(1 - 11.253124/21.606657) = 0.143754$. The hold removed
  0.0043 of credit for a level the agent did not sustain.

**Provenance.** Ours. The cost-to-go is from A3. The hold is EXPLOITS.md #16
(`test_single_glitch_cannot_spike_progress`).

### A5. The C-space tunnel potential: `cspace_tunnel` (shaping, weight 1)

**Config.** Alpha, the primary component. Its terminal value is the score (A4); its step is the shaping.

```yaml
# loophole/configs/alpha_training.yaml
- type: cspace_tunnel
  weight: 1.0
  in_score: true
  params: {gamma: 1.0, alpha: 0.5, progress_cap: 0.3}
```

**Definition.**

$$
\Phi(q) = -\frac{c(q)}{c_0}, \qquad
F_t = \lambda\bigl(\gamma\,\bar\Phi(q_t) - \Phi(q_{t-1})\bigr) \overset{\gamma=\lambda=1}{=} \frac{c(q_{t-1}) - c(q_t)}{c_0}
\quad\text{(non-terminal steps),}
$$

* $c$: the cost-to-go of A3;
* $c_0 = c(q_0)$ = 21.606657;
* $\bar\Phi$ as in S4;
* when $c$ is unavailable (a fallback), the previous value is kept, so $F$ = 0.

**Intuition.** Every action is rewarded by how much closer it brought tube B to the exit, measured along the
collision-free way out, as a fraction of the starting distance. Retreating costs exactly what advancing earned.

**Precisely.**
* **What it measures:** the normalised change of the roadmap cost-to-go.
* **What it ignores:** everything the cost-to-go ignores (A3).
* **Bounded returns.** With $\gamma$ = 1, the shaping return of an unsolved, truncated episode is
  $(c_0 - c_T)/c_0 \le 1$. That of a terminated episode is $-\Phi(q_0)$ = 1, whatever the path.

**Worked example** (Alpha reference, `runs/20261114-111735/oracle/.../episode.jsonl`, action 2):
* $c$ went from 21.606657 to 21.296499.
* $F = (21.606657 - 21.296499)/21.606657 = 0.310158/21.606657 = 0.014355$; logged `rewards.cspace_tunnel`
  0.014355.
* Action 1 (a grab) did not move the tube: $F$ = 0, logged 0.0.

**Provenance.** The potential is the navigation function of A3 [ZBKV20, JSCP15]; potential-based shaping is
[NHR99]; the component is ours.

### A6. Clearance shaping: `clearance_shaping` (shaping, weight 0)

**Config.** Alpha, weight 0 (logged, contributes nothing). The key is `clearance_shaping`. The older name
`industreal_sdf` survives only as a Python alias (`IndustRealSDFReward`), not as a registry key: a config with
`type: industreal_sdf` fails to load with `KeyError: 'industreal_sdf'`.

```yaml
# loophole/configs/alpha_training.yaml (and alpha_planner.yaml, identical reward block)
- type: clearance_shaping
  weight: 0.0
  in_score: false
  params: {d_sep: 0.4, sigma_dist: 0.08, gamma: 1.0}
  # defaults: w_penetration: 50.0, penetration_threshold: 0.002, w_action: 0.05, lam: 1.0
```

**Definition.**

$$
\Phi(s) = \exp\!\Bigl(-\frac{\max(0,\ d_{\text{sep}} - d(s))}{\sigma}\Bigr),
$$

$$
r = \lambda\bigl(\gamma\,\bar\Phi(s') - \Phi(s)\bigr)
\;-\; w_{\text{pen}}\,\max(0,\ p - p_{\text{thr}})
\;-\; w_{\text{act}}\,\bigl(\Delta_t^2 + 0.1\,\Delta_\theta^2\bigr),
$$

* $d(s)$: the minimum distance between the tubes (m); $\Phi$ = 0 when it is unavailable;
* $d_{\text{sep}}$ = 0.4 m; $\sigma$ = 0.08 m;
* $p$: the largest penetration during the action (`audit.window_penetration`); $p_{\text{thr}}$ = 2 mm;
  $w_{\text{pen}}$ = 50;
* $\Delta_t$, $\Delta_\theta$: the largest translation (m) and rotation angle (rad) of a handle during the action, as
  executed by the rig (`env._executed_motion`), not as requested; $w_{\text{act}}$ = 0.05.

**Intuition.** "More room between the tubes is better, don't press, don't flail." The kernel rises from almost 0
when the tubes touch to 1 at 0.4 m apart, and the penalties charge pressing into contact and large motions.
Because the room between the tubes can grow while they stay interlocked, this is a weak signal for this puzzle,
which is why it has weight 0.

**Precisely.**
* **What it measures:** the change in a clearance kernel, minus contact pressure and handle motion.
* **What it ignores:** topology and the way out. Clearance can increase in a dead end.
* **Only the kernel is potential-based.** The two penalties are not, so they do not telescope.
* **The action term charges the handle, not the tube.** A `grab` moves the handle from its parking spot to the
  grasp point, and that is charged even though the tube does not move.

**Worked example** (Alpha reference, `runs/20261114-111735/oracle/.../episode.jsonl`):
* **Action 1, a grab.** $d$ stays 0.04445 m, so the kernel term is 0.
  * The handle travelled 2.0794 m from (−0.15, −1.2, 0.02) to the grasp point:
    $r = -0.05 \times 2.0794^2 = -0.216185$; logged −0.216185.
* **Action 2, a move.** $d$: 0.04445 → 0.01983 m.
  * $\Phi(s) = e^{-(0.4-0.04445)/0.08} = e^{-4.444375} = 0.011744$, and
    $\Phi(s') = e^{-(0.4-0.01983)/0.08} = e^{-4.752125} = 0.008633$. Kernel term −0.003111.
  * Penalty: $p$ is below 2 mm, so 0.
  * Action term: with the requested motion as an approximation ($\lVert\Delta t\rVert$ = 0.04424 m, rotation about
    0.1702 rad), $-0.05\,(0.001957 + 0.002897) = -0.000243$.
  * Total −0.003354; logged −0.003358. The residual 4e-6 is the difference between executed and requested motion,
    which the log does not record separately.

**Provenance.** Adapted from IndustReal's SDF-based dense reward [TLA+23], built on Factory [NSA+22]. IndustReal
measures alignment with a goal pose (−log of the mean SDF of plug points at the target pose); we measure the
clearance between the two pieces instead, since a separation task has no goal pose.

### A7. The Gauss linking diagnostic: `gauss_linking_diagnostic` (diagnostic)

**Config.** Alpha, logged only. The key is `gauss_linking_diagnostic`. The older name `gauss_linking` survives only as
a Python alias (`GaussLinkingPotentialReward`), not as a registry key: a config with `type: gauss_linking` fails to
load with `KeyError: 'gauss_linking'`.

```yaml
# loophole/configs/alpha_training.yaml (and alpha_planner.yaml, identical reward block)
- type: gauss_linking_diagnostic
  weight: 0.0
  in_score: false
```

**Definition.** For two polylines $P$, $Q$, the Gauss linking integral is

$$
\mathrm{Lk}(P, Q) = \frac{1}{4\pi} \int_{P}\!\int_{Q}
\frac{(r_P - r_Q)\cdot(\mathrm d r_P \times \mathrm d r_Q)}{\lVert r_P - r_Q\rVert^{3}} ,
$$

* $r_P$, $r_Q$: points on $P$ and $Q$;
* it is evaluated exactly, segment pair by segment pair: the solid angle each pair of segments subtends, with its sign
  [KL00] (`puzzles/alpha/geometry.py: gauss_linking`).

The environment logs it, after every action, under `diagnostics`, for two choices of curve (`LoopholeEnv.diagnostics`):

$$
\texttt{linking\_open} = \mathrm{Lk}(\gamma_A, \gamma_B), \qquad
\texttt{linking\_closed} = \mathrm{Lk}(\bar\gamma_A, \bar\gamma_B), \qquad
\bar\gamma = \gamma[i^*{:}j^*] \cup \overline{\gamma(j^*)\,\gamma(i^*)} ,
$$

* $\gamma_A$, $\gamma_B$: the two tubes' centre-lines, open curves, in world coordinates;
* $\bar\gamma$: a tube's loop closed across its own gap. $(i^*, j^*)$ is the closest pair of centre-line points
  at least 15 points apart along the curve (the alpha's crossing). $\bar\gamma$ is the centre-line from $i^*$ to
  $j^*$ plus the straight closing segment back to $i^*$. It is fixed in the tube's body frame, computed once at
  build (`env._closed_loop`);
* `linking_integral` is `linking_closed` when both loops exist, else `linking_open`.

The registered component records `linking_open` (`GaussLinkingDiagnostic.last`); its step and terminal are 0.

**Intuition.**
* **The open-curve value has no topological meaning on its own.** Two open arcs can always be pulled apart if
  they are allowed to pass through each other, so nothing about them is invariant. `linking_open` is a real number
  that drifts as the tubes move and changes when a tube merely turns in place. What traps the open tubes is their
  rigidity and thickness, which is geometry, not topology.
* **Closing each alpha across its gap gives two loops whose linking number is a true integer invariant.**
  * It is −1 at the start: each tube threads the other's loop.
  * The solid part of a loop cannot be crossed without a violation. So it can change only when a tube passes
    through the other's gap, across the closing segment, which is the puzzle's key move.
  * It is 0 once the tubes are topologically apart.

**Precisely.**
* **`linking_closed` measures** whether the two closed-up loops are linked (−1, 0 or +1, logged rounded to $10^{-5}$).
  **It ignores**:
  * how far through the separation the tubes are;
  * everything after the unlinking: the tubes still have to be slid apart;
  * the tubes' thickness.
* **`linking_open` measures** the open-curve integral, a geometric winding number. **It ignores** topology
  entirely.
* **Neither is rewarded or scored.** `linking_closed` is binary, a single cliff with no gradient on either side,
  and turning it into credit would change the scored rule.

**Worked example** (every committed Alpha run with at least one action, 24 trials, from the `diagnostics` field of
each `verifier/episode.jsonl`):
* **Start of every run:** `linking_closed` −1.0 and `linking_open` −1.29204.
* **Reference** (`runs/20261114-111735/oracle/...`):
  * `linking_closed` flips to 0 at action 47, while `linking_open` is −1.79761;
  * the escape test succeeds at action 59, with `linking_open` −0.8118.

  So the topological event came 12 actions before the geometric one. `linking_open` was further from 0 at the
  flip than at the start, which is why it cannot serve as progress.
* **All 24 runs:**

  | | runs | `linking_closed` flips to 0 | flips back |
  |---|---|---|---|
  | solved | 8 | all 8, 12–47 actions before the solve (Gemini 4 Carbon: 18 → 45, 50 → 70, 17 → 46, 17 → 64; references: 47 → 59, 47 → 79) | never |
  | unsolved | 16 | none | – |

* In this sample it separates solves from non-solves perfectly. That is 24 runs of one puzzle, so it is evidence
  for a milestone, not a proof that unlinking is necessary or sufficient. It is listed under future work as a
  candidate stage marker.

**Provenance.**
* The Gauss integral and its exact segment-pair evaluation are [KL00]; background [RN11].
* Closing each alpha across its gap is ours.
* Topological models of string puzzles, for contrast with R6: [PS02].

### A8. The planner task (`tasks/loophole-alpha-planner`) adds no reward

**Config.** `loophole/configs/alpha_planner.yaml`. Its `reward:` block is byte-identical to `alpha_training.yaml`
(`cspace_tunnel` weight 1 in the score, `clearance_shaping` weight 0, `gauss_linking_diagnostic` logged). The only
difference is `observation.profile: planner`, which adds three query tools to the sensors profile: `check_motion`,
`check_poses` and `get_geometry`.

**So every Alpha section applies to it unchanged**, including the score, the checks and the shaping.

**Worked example** (`runs/20261114-093603-v2`): the planner-task reference (`oracle-planner`) and the plain
reference (`oracle`) replay the same 59 commands and both score 0.85101; their `linking_closed` flips at the same
action, 47.

---

## Rat Trap (`tasks/loophole-rattrap`)

### R1. Success: frozen-shape escape (scored)

**Config.** Rat Trap only.

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
escape_directions: 64
```

**Definition.** Let $x_0, \dots, x_n$ be the cord's node positions after an action. $C(x)$ is the rigid assembly of
one capsule of radius $r_c$ = 1.2 mm per segment $[x_i, x_{i+1}]$, plus the mouse sphere at its node. $F$ is the
frame with its posts, dowels, ring and sphere. Then

$$
\text{solved}(x) \iff \exists\, u \in U_{64} :\ \forall s \ge 0,\ \ \bigl(C(x) + s\,u\bigr) \cap F = \varnothing .
$$

* $U_{64}$ is as in A1.
* The sweep (`FrozenEscape.escape_direction`) runs to the bounding sphere of the structure:
  * far from contact it advances by $d - \tau$ ($\tau$ = 0.5 mm; distances queried at margin 30 mm);
  * near contact it checks exact non-penetration ($\mathrm{dist} \ge -10^{-5}$ m) every 1 mm;
  * it gives up after 150 such steps.

**Intuition.** Freeze the string in whatever shape it is in, mouse and all, and ask whether that frozen shape
could slide off the frame in a straight line. If it could, the string is free, whatever its shape: no
reference, no target pose.

**Precisely.**
* **What it measures:** whether the current shape, held rigid, has a straight way out. It ignores the cord's
  flexibility.
* **Sound but strict.** It never calls a trapped cord free. A cord that is topologically free but still draped
  over a post does not count until it is pulled clear.
* **What it ignores:**
  * how close to free the cord is (that is R3's job);
  * any deformation that would be needed on the way out.

**Worked example** (the reference: Tibi's recorded human solve, replayed closed-loop from its fixed waypoints).
Replayed in-process with `scripts/rattrap_shaping_trace.py` (output in `runs/rewards-examples/`):
* After each of the first 283 actions, every one of the 64 directions was blocked.
* After action 284 one was free: `solved` became true, and the episode ended.
* The score is 0.833333 (S2). The Harbor oracle replays the same waypoints in the Linux container.

**Provenance.** The same assembly-planning test as A1 [WL94], applied to the frozen cord: ours.

### R2. The Rat Trap's violation check and the release exception (scored)

**Config.**

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
penetration_tolerance: 0.001         # m
release_settle_audit: true
release_transient_limit: 0.002       # m
certify_motion: log                  # off | log | enforce (S5)
```

**Definition.** (`puzzles/rattrap/sim.py`). The check is on penetration only, over every contact, including the cord
against itself:

$$
\max_{t}\, \max_{c}\, \bigl(-\mathrm{dist}_c(t)\bigr) > p_{\text{tol}} = 1\ \text{mm}
$$

or, with `certify_motion: enforce`, an uncertified substep (S5).

* The maximum is over every physics substep of every action, with one exception. During the settle that follows a
  `release`, the depth is tracked separately as $p_{\text{peak}}$ (the largest transient) and $p_{\text{end}}$ (the
  depth once settled).
* That release counts only if
  $p_{\text{end}} > p_{\text{tol}}$ or $p_{\text{peak}} > p_{\text{rel}}$, with $p_{\text{rel}}$ =
  `release_transient_limit` = 2 mm.

**Intuition.** The cord may not be pressed into the frame by more than 1 mm, a little under its 1.2 mm radius. One
exception: letting go of a strand that was strained against the frame makes it overshoot for a few milliseconds.
That overshoot is not the agent's doing (`docs/HARNESS_NOTES.md`), so during the settle after a release only a
penetration that persists, or a transient above 2 mm, counts.

**Precisely.**
* **What it measures:** contact depth over every contact, including the cord against itself, with the release
  exception.
* **What it ignores:** per-substep speed and integrator divergence. Alpha checks both; the Rat Trap does not
  (discrepancy 7, and `docs/EXPLOITS.md`).
* **The release limit is generous.** 2 mm is above the 1.95 mm depth of a full wall crossing (discrepancy 6).

**Worked example** (`loophole/tests/fixtures/rattrap_release_after_block.jsonl`: the first 89 actions of a committed
Gemini 4 Carbon run, excluded from the results for this reason; replayed with
`loophole/tests/test_rattrap_release.py`):

* **The sequence.** A `move` is blocked with the cord strained against the frame, then, at action 89, `release`.
  While the cord settles, the deepest contact reaches $p_{\text{peak}}$ = 1.006 mm for a few substeps. Once settled
  it is $p_{\text{end}}$ = 0.472 mm.
* **The old rule** (every substep audited, `release_settle_audit: false`): $1.006 > p_{\text{tol}} = 1.000$, so a
  violation; the episode ends with $S = 0$.
* **The current rule:**
  * $p_{\text{end}} = 0.472 \le 1$ and $p_{\text{peak}} = 1.006 \le p_{\text{rel}} = 2$, so no violation;
  * the episode's deepest counted contact is 0.472 mm;
  * the run continues (here unsolved, $S$ = 0, for lack of progress, not for cheating).
* **The exception is not a blind spot.** `test_a_large_release_transient_still_counts` sets $p_{\text{rel}}$ = 0.5 mm,
  and the same 1.006 mm transient is a violation again. But see discrepancy 6 for why 2 mm is too generous.

**Provenance.** Ours.

### R3. Partial credit: checkpoint progress (scored)

**Config.** Rat Trap only.

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
score_progress_cap: 0.3
checkpoints_file: rattrap-checkpoints.npz   # loophole/reference/; built by scripts/build_rattrap_reference.py
```

**Definition.** The checkpoints $c_1, \dots, c_K$ are full cord states (all node positions) from the reference
solve. Checkpoint $k$ is reached at evaluation $t$ (after action $t$) if

$$
\delta\bigl(x(t),\, c_k\bigr) = \max_i \lVert x_i(t) - c_{k,i} \rVert < \varepsilon
\quad\text{and}\quad c_1, \dots, c_{k-1} \text{ were reached at evaluations } < t ,
$$

and at most one checkpoint advances per evaluation. With $k$ checkpoints reached,

$$
S_{\text{unsolved}} = \kappa\,\frac{k}{K} .
$$

* $x_i(t)$: the position of cord node $i$ after action $t$. The mouse is welded at node 71, so the nodes carry it;
* $\delta$: the max-norm deviation over nodes, not the mean;
* $K$ = 20 checkpoints, thinned from the reference's 75 grab/release boundaries. Each kept checkpoint is at least
  $2\varepsilon$ = 33.2 mm from the previous one, and $c_1$ from the start;
* $\varepsilon$ = 16.6 mm = 1.5 × $\varepsilon_{\min}$, where $\varepsilon_{\min}$ = 11.08 mm is the smallest radius
  at which the reference, as agent commands, registers every checkpoint in order;
* $\kappa$ = `score_progress_cap` = 0.3.

**Intuition.** The reference is Tibi's human solve of this exact puzzle. We marked 20 moments along it. A run that
does not solve earns credit for how many of those moments it reproduced, in order. A moment counts as
reproduced when the whole string is close to how it was then: every point within 1.66 cm, not just most of the
string.

**Precisely.**
* **What it measures:** progress along *this* route. A run gets credit for the checkpoints it reproduces in
  order, and keeps it.
* **The order is strict:** none can be skipped, and only one counts per action.
* **What it ignores:**
  * progress along a different route. A different solution earns partial credit only where it passes near Tibi's
    checkpoints; a solve is still a solve;
  * time;
  * distance to a checkpoint not yet reached. That is only in the shaping (R4).
* **The start cannot match $c_1$:** it is 110.8 mm away.

**Worked examples.**
* Rat Trap oracle replayed as a fixed command list on Linux, `runs/harbor-rattrap/rattrap-oracle/` (the
  macOS/Linux divergence that led to the closed-loop reference): unsolved after 284 actions, $k$ = 18.
  $S = 0.3 \times 18/20 = 0.27$.
* Kimi K3 Pro, round 2, `runs/rattrap-agents/kimi-20261114-205209/rescore.json`: unsolved at the full budget of 852
  actions. Its closest approach to $c_1$ was 63.4 mm, against $\varepsilon$ = 16.6 mm, so $k$ = 0 and $S$ = 0.

**Provenance.**
* Staged geometric milestones with simple success tests follow cable-routing RL practice [TXR+26, §4.2]: SILO
  splits routing into stages, each with an $\varepsilon$-ball test.
* Topological signatures are used to guide planning, not as reward scalars [BLK11, MB24].
* The ordered max-norm matcher and the calibration are ours.

### R4. The checkpoint shaping potential and the training violation penalty (shaping)

**Config.** Rat Trap, computed when the episode is created with `training=True`.

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
shaping_gamma: 1.0
shaping_lambda: 1.0
checkpoints_file: rattrap-checkpoints.npz
```

**Definition** (`checkpoints.py: CheckpointTracker.potential`, through `PotentialShaping`).

$$
\Phi(x, k) =
\begin{cases}
-\Bigl[(K - k) + \min\!\Bigl(1,\ \dfrac{\delta(x, c_{k+1})}{D_{k+1}}\Bigr)\Bigr] & k < K \\[1ex]
0 & k = K
\end{cases}
\qquad D_{k+1} = \delta(c_k, c_{k+1}),\ c_0 = \text{the start},
$$

* $x$: the cord's node positions;
* $k$: checkpoints reached so far (R3), part of the episode state;
* $\delta$: the max-norm deviation;
* $D_{k+1}$: the length of the stage from $c_k$ to $c_{k+1}$. $D_1$ = 110.8 mm, $D_2$ = 36.6 mm,
  $D_3$ = 35.7 mm; the shortest is 35.7 mm, all above $2\varepsilon$ = 33.2 mm;
* $\Phi \in [-(K+1), 0]$, and $\Phi(\text{start}) = -(20 + 110.8/110.8) = -21$.

**Intuition.** Each checkpoint reached is worth one point, and inside a stage the string earns the fraction of the
way it has come towards the next checkpoint. The potential is continuous across a checkpoint:
* just before reaching $c_{k+1}$, the string is close to it, so $\Phi \approx -(K-k)$;
* just after, it is a full stage away from $c_{k+2}$, so $\Phi \approx -(K-k-1) - 1$.

So reaching a checkpoint is not itself a jackpot; getting close to it is what pays, and wandering back gives it up.

**Precisely.**
* **What it measures:** distance along Tibi's route, in stages, with the current stage measured in the max-norm.
* **What it ignores:**
  * other routes;
  * the distance to anything but the next checkpoint;
  * time.
* **Beyond a full stage length there is no gradient:** the $\min(1, \cdot)$ caps the in-stage term.
* **Oscillation:** with $\gamma$ = 1 the return telescopes to $\Phi(\text{end}) - \Phi(\text{start})$; an
  oscillation earns only that.

**Worked examples.**
* **Kimi K3 Pro, round 2** (`runs/rattrap-agents/kimi-20261114-205209/rescore.json`): closest approach to $c_1$ was
  63.4 mm, with $k$ = 0.
  * $\Phi = -(20 + 63.4/110.8) = -20.572$.
  * The shaping return from the start to that state was $-20.572 - (-21) = +0.428$. The score was 0, because no
    checkpoint was reached.
* **The reference**, replayed with `training=True` (`runs/rewards-examples/rattrap-reference-shaping.jsonl`, written
  for this page; per-action $\Phi$, $F$ and alignment terms):

| action | event | $k$ | $\Phi(s')$ | $F$ | arithmetic |
|---|---|---|---|---|---|
| 0 | start | 0 | −21 | – | $-(20 + 110.8/110.8)$ |
| 2 | first move | 0 | −20.842827 | +0.157173 | $-20.842827 - (-21)$ |
| 20 | approaching $c_1$ | 0 | −20.338394 | +0.075045 | in-stage term $0.338394$, i.e. $\delta = 0.338394 \times 110.8 = 37.5$ mm from $c_1$ |
| 21 | **$c_1$ reached** | 1 | −20.000000 | +0.338394 | $\delta(x, c_1) < 16.6$ mm; now a full stage or more from $c_2$: $-(19 + 1)$ |
| 28 | **$c_2$ reached** | 2 | −19.000000 | +0.936708 | from $-(19 + 0.936708)$, i.e. 34.3 mm from $c_2$, to $-(18 + 1)$ |
| 268 | **$c_{20}$ reached** | 20 | 0 | +1.447552 | from $-(1 + 0.447552)$ to $\Phi = 0$ at $k = K$ |
| 269–284 | to the solve | 20 | 0 | 0 | already 0; the solve's $\bar\Phi = 0$ adds nothing |

* **Sum of $F$ over the 284 actions: 21.000000** $= 0 - \Phi(\text{start})$, exactly the telescoped value.
* **The checkpoints were reached at actions** 21, 28, 32, 55, 65, 84, 95, 99, 108, 117, 122, 141, 168, 196, 201,
  210, 228, 239, 258 and 268.
* **The potential is continuous, but individual steps need not be small.** A 30 mm move can cross most of a short
  stage, as at action 28.

**The training violation penalty** (`RatTrapEpisode.act`, training-only block; no YAML key). On the step a violation
happens, the shaping also gets

$$
F_{\text{viol}} = -\bigl(\lambda\,(K + 1) + 1\bigr) = -22 \quad (\lambda = 1,\ K = 20),
$$

logged as `shaping_parts.violation_penalty` (0 on every other step).
* **Why.** Under S4's rule a violation is absorbing, so $\bar\Phi = 0$ and the episode's shaping return is
  $-\lambda\Phi(s_0) = +21$, as much as a solve. The penalty brings a violation's shaping return to
  $21 - 22 = -1$, below every truncated non-solve's $\lambda(\Phi(s_T) + 21) \ge 0$.
* **What it changes.** Only training signals. It is not a potential difference, which is intended: it is an explicit
  cost for an outcome the score already sets to 0. Scored runs never execute the block. `rattrap.yaml`, its hash and
  every committed log are unchanged.
* **Worked example** (`test_rattrap_reward.py::test_violation_shaping_return_is_below_every_non_solve`):
  * a grab, then a move with an injected 1 m penetration;
  * the penetration ends the episode as a violation; `violation_penalty` = −22 and the shaping total is −1.000;
  * the score is 0.

**Provenance.** Staged decomposition with dense in-stage proximity follows [TXR+26, §4.2]; the potential, its
continuity across stages and its bounds are ours; potential-based shaping [NHR99].

### R5. Aperture alignment: `aperture_alignment` (shaping, weight 0)

**Config.** Rat Trap, weight 0 for all three apertures (computed with `training=True` and logged under
`alignment`, contributes nothing).

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
shaping_alignment:
  - {disc: ring, weight: 0.0, gate_distance: 0.02}
  - {disc: hole_upper, weight: 0.0, gate_distance: 0.01}
  - {disc: hole_lower, weight: 0.0, gate_distance: 0.01}
```

**Definition** (`reward/alignment.py`).

$$
\Phi(s) = -w \cdot \exp\!\Bigl(-\bigl(d/\sigma\bigr)^2\Bigr) \cdot \max\bigl(0,\ 1 - (\hat n \cdot \hat\tau)^2\bigr) ,
$$

* $\hat n$: the aperture's unit axis (the disc normal of the ring or a hole);
* $\hat\tau$, $d$: the cord's local unit tangent near the aperture centre, and the distance from the centre to the
  cord (`local_tangent`). $\hat\tau$ is a distance-weighted average of segment directions, weights
  $\exp(-(d_i^2 - d_{\min}^2)/\ell^2)$ with $\ell$ = 1 cm. Each direction is first sign-aligned with the nearest
  segment, so the two antiparallel strands of a doubled cord do not cancel;
* $\sigma$ = `gate_distance` (2 cm for the ring, 1 cm for the holes);
* $w$ = `weight`.

Only $(\hat n\cdot\hat\tau)^2$ enters. A disc normal has no sign, a closed cord has no direction, and spinning the
aperture about its axis changes nothing.

**Intuition.** Thread a needle along its eye, not across it. Near a hole, the cord should run along the hole's axis
if it is going to pass. The term is a penalty when the cord lies across the hole next to it, and it fades with
distance, where orientation means nothing.

**Precisely.**
* **What it measures:** the misalignment of the nearest stretch of cord with each aperture axis, gated by distance.
* **What it ignores:**
  * whether passing through that aperture is the right move at all (the mouse cannot pass either hole);
  * which side of the aperture the cord is on.
* **As a potential it only ever enters through `PotentialShaping`**, so it is shaping, never score.

**Worked example** (the reference replay with `training=True`, `runs/rewards-examples/rattrap-reference-shaping.jsonl`):

* **After action 21** (the step that reaches $c_1$), hole_lower ($\sigma$ = 10 mm):
  * $d$ = 4.411 mm and $\lvert\hat n\cdot\hat\tau\rvert$ = 0.951429;
  * penalty $= e^{-(4.411/10)^2}\,(1 - 0.951429^2) = 0.82319 \times 0.094783 = 0.078024$ (logged 0.078026,
    from unrounded inputs);
  * at $w$ = 0, $\Phi$ = 0. At the class default $w$ = 0.25 it would be −0.0195.
* **After action 1** the cord is $d$ = 42.0 mm from that hole, so the gate is $e^{-17.7} \approx 2 \times 10^{-8}$
  and the penalty is 0.
* **The largest penalty of the whole solve** is the ring's at action 277: $d$ = 12.7 mm, $\lvert\hat n\cdot\hat\tau\rvert$
  = 0.168, penalty 0.648 (ring $\sigma$ = 20 mm). The cord there lies nearly across the ring, close to it, a few
  actions before it comes free.
* **This is why it is weight 0.** It penalises a configuration the human solve passes through on its way out.

**Provenance.** The orientation term and `local_tangent` are adapted from a colleague's ring-on-wire environment
[HO26] (`potential_terms`, `local_tangent`; there $\hat n$ is the ring's rotation-matrix third column and
$\hat\tau$ the wire's tangent). Our changes:
* the aperture and the passer are inputs;
* segment directions are sign-aligned before averaging.

Their straight-line distance-to-goal potential was not ported.

### R6. The disc word and the Przytycki–Sikora norm: unsound (diagnostic)

This was the original Rat Trap progress measure. It is mathematically the most elegant thing in the repository,
and it does not work here. It is kept as a logged diagnostic so that the failure stays visible.

**Config.** Rat Trap, always computed and logged in every action's result (`word`, `norm`); never scored or shaped.

```yaml
# loophole/puzzles/rattrap/rattrap.yaml
topology_alphabet:
  A: {disc: frame_main, positive: "front to back (+y)", aperture: null, passable: true}
  B: {disc: frame_upper_right, ...}
  C: {disc: frame_lower_left, ...}
  U: {disc: hole_upper, positive: "upwards (+z)", aperture: post_upper.hole, passable: false}
  L: {disc: hole_lower, ...}
  R: {disc: ring, positive: "left to right (+x)", aperture: ring.inside, passable: false}
```

**The construction.**
1. **Discs span the holes.** Each independent passage of the rigid structure is spanned by a disc with a
   normal: the three regions A, B, C of the frame opening, the two hole annuli U, L and the ring's interior R.
2. **The cord reads a word.** Walk around the closed cord and write down each signed crossing: the letter for a
   crossing along the normal, its inverse for a crossing against it. Cancel adjacent inverse pairs and reduce
   cyclically. The result $w$ is an element of the free group on the six generators, up to conjugacy, and
   moving the cord without crossing a solid cannot change it.
3. **Measure the word.** [PS02] define, for a set $P$ of *passable* generators (holes the cord with everything it
   carries can pass through),

$$
\lVert w \rVert = \min\Bigl\{ k :\ w = \prod_{i=1}^{k} p_i\, g_{a_i}^{\pm1}\, p_i^{-1},\ g_{a_i} \in P \Bigr\}
\quad(\infty \text{ if there is none}),
$$

   the minimal number of single passes through passable holes that undo the word. Their Theorem 3.2 turns it into a
   combinatorial quantity, $\lVert w\rVert = \lvert w\rvert_c$: the minimum, over all non-crossing pairings of letters
   $g$ with $g^{-1}$, of the number of letters left unpaired, where only letters of $P$ may be left unpaired.
   `norm.py` computes it with an $O(L^3)$ interval dynamic programme:

$$
f(i, j) = \min\Bigl( \underbrace{1 + f(i{+}1, j)}_{w_i \in P \text{ left unpaired}},\ \
\min_{k:\ w_k = w_i^{-1}} f(i{+}1, k{-}1) + f(k{+}1, j) \Bigr), \qquad f(i, j) = 0 \text{ for } i > j .
$$

**Why it is elegant.**
* It is exact, discrete and invariant: no thresholds and no reference solution.
* It is a theorem about precisely this kind of puzzle. The paper's subject is the Chinese Rings, a rope-and-rings
  disentanglement puzzle.
* The free reduction makes threading through a hole and straight back cancel, for free.
* The norm is a true distance to "unlinked" in the number of passes: free is $\lVert w\rVert$ = 0.
* The start word is `Ac`, with $\lVert \texttt{Ac} \rVert$ = 2: two passes from free.

**Why it fails: the measurement.** Along the reference solve (Tibi's, 284 actions as agent commands):
* the start word is `Ac` and $\lVert w\rVert$ = 2;
* **the word is empty, $\lVert w\rVert$ = 0, after action 15, while the cord is still trapped;**
* the cord stays trapped for the remaining 269 actions.

`test_rattrap_reward.py::test_recorded_solve_scores_highest` prints it. Under the original scheme, where progress
was the norm's reduction, an agent could have earned the full 0.3 partial credit within about 15 actions with no
real progress.

**Why it fails: the cause.**
* **The disc system is not a complete meridian system.** The six discs do not cut the complement of the structure
  into a simply connected region, because parts thread through each other without touching: each dowel passes
  through its post's hole with an annular gap, and the upper post passes through the ring.
  * These add independent loops that no disc spans. An annulus is not a disc, and nor is the ring's interior minus
    the post.
  * So the cord can be wrapped around a dowel inside a hole, for example, and read as the empty word.
* **The mouse is not in the word.** The bare cord fits through every hole; only the stretch of cord carrying the
  mouse does not. A cyclic word forgets where the mouse sits.
* **Homotopy is not isotopy.** The cord cannot pass through itself, so an empty word is necessary for freedom but
  not sufficient.

The word is still logged, and still useful: it flags crossings, and an episode that ends with a non-empty word is
certainly not free.

**A correction to the original specification, kept for the record.** The literal $\lVert a_n \rVert$ in [PS02]
Prop. 2.2′ is not $2^{n-1}$; it is $\infty$ or 1. The value $2^{n-1}$ belongs to the distance between $a_n$ and
$a_{n-1}$ with only $g_n$ passable. `test_rattrap_norm.py` tests the paper's statement in that form.

**Worked example** (the reference replay, `runs/rewards-examples/rattrap-reference-shaping.jsonl`):

* Start: $w$ = `Ac`, $\lVert w\rVert$ = 2 (A and C are frame regions, passable; the pairing of `A` with `c` is
  impossible, so both stay unpaired).
* After action 15: $w$ = `` (empty), $\lVert w\rVert$ = 0, with $k$ = 0 checkpoints and the cord nowhere near free.
* Then it becomes non-empty again: from action 29 the word runs through `ALraRl` (65 actions), `AUau` (44),
  `ALrbRl` (38) and `ALbAlc` (24), with $\lVert w\rVert$ up to 6.
* At action 282 it is `ALal` ($\lVert w\rVert$ = 2: the `A…a` and `L…l` arcs would cross, and L, a hole the
  mouse cannot pass, must be paired, so A and a stay unpaired). It is empty again from action 283, and the cord is
  free at 284.
* **As a progress measure it reads "free" 269 actions early, and then "further from free than the start".**

**Provenance.**
* The norm, Theorem 3.2 and the Chinese Rings model are [PS02].
* The disc system, the passability rule and the finding that it is unsound here are ours.
* See `docs/RATTRAP_DESIGN.md` for the design history.

---

## Inactive and legacy components

Registered in `reward/components.py`, enabled in neither active task. They remain for the procedural rope puzzles
(`loophole/default_config.yaml`) and the deprecated Alpha variants.

| component | definition | status |
|---|---|---|
| `base` | $S = 1 - \alpha\,a/T$ if solved without violations, else 0 | rope-puzzle default (weight 1) |
| `topology_shaping` | $\Phi = -g(w)/d_0$, $g$ = `goal_distance(word)` (default: reduced word length), $d_0 = \max(1, g(w_0))$; terminal $(g(w_0) - g(w_T))/d_0$, 0 after a violation | rope default, `enabled: false` (weight 0.2) |
| `cost_to_go_shaping` | $\Phi = -c(q)/c_0$ (as A5); terminal 0 | added by `auto_roadmap_components` |
| `progress` | $\kappa\,(1 - c_{\text{best}}/c_0)$ with the unheld minimum $c_{\text{best}}$; 0 if solved or on a violation | added by `auto_roadmap_components` |
| `scaled_progress` | A4's score as a stand-alone component | not used by any config |
| `reward/progress.py: progress_potential` | $-c(q)$, unnormalised | unused helper |

## Discrepancies found while writing this page

Code against documents and logs; the code is what this page states.

1. **The committed Alpha oracle log predates the terminal rule.**
   * In `runs/20261114-111735/oracle`, the final step's `cspace_tunnel` reward is 0.009462, which is
     $(5.925163 - 5.720725)/21.606657$: $\Phi(s')$ kept at the solve.
   * The current code sets $\bar\Phi$ = 0 there, giving $5.925163/21.606657 = 0.274231$.
   * The score is unaffected (0.851010); only the logged shaping is.
2. **Committed Alpha logs do not re-verify on macOS.**
   * Replayed with the current code on macOS, `runs/20261114-111735` (oracle, `sbMy6Ld`) fails with "state hash
     mismatch after action 2": the pose differs by 5e-7 after action 2. This is consistent with the known
     macOS/Linux divergence (the Rat Trap's 18 / 20); they were verified in the Linux image.
   * `runs/20261114-080705` (`Kt5DJ3h`) fails earlier, with "episode was played on a different puzzle file": the
     puzzle YAML changed after that run.
   * The worked examples above therefore use the verifier's logged fields, not a fresh macOS replay. They were not
     re-run in the Linux image for this page.
3. **Clearance shaping charges the handle's grab relocation** (A6), 0.216 on the reference's first action,
   though the tube does not move. The docstring calls it "the executed handle motion", which is literally true but
   not the intent ("don't flail"). Weight 0, so nothing is affected.
4. **Clearance shaping cannot be recomputed exactly from the log.** The executed motion is not logged separately;
   requested motion reproduces it to about 4e-6.
5. **Violations and solves receive the same terminal shaping** (S4). This is correct for policy invariance
   with bootstrapping. A learner that adds undiscounted shaping to the score and treats the budget cut-off as
   terminal would prefer an early violation to a long non-solve:
   * Rat Trap: $+21\lambda$ for violating at the start, against $\lambda(\Phi(s_T) + 21) \le 21\lambda$ for a truncated run;
   * Alpha: $+1$ against $1 - c_T/c_0$.

   The rule itself is unchanged: it is the textbook rule, and the score is what Harbor reads. The Rat Trap's training
   mode now adds `violation_penalty` = −22 on a violation, so a violation's shaping return there is −1. Alpha is
   documented, not changed (see the warning at the top).
6. **The Rat Trap release transient limit is not below the depth of a wall crossing.**
   * `rattrap.yaml` and `sim.py: release` justify `release_transient_limit` = 2 mm as "below the cord's diameter,
     so a release cannot spring the cord through a wall".
   * The bound that matters is the deepest contact on the way through a wall. For the 1.2 mm-radius cord and a
     1.5 mm hole wall that is $r_c + h/2$ = 1.95 mm, measured with MuJoCo: a capsule of radius 1.2 mm swept
     through a 1.5 mm box peaks at 1.950 mm.
   * So a release that sprang the cord through a hole wall, and settled clear, would not register as a violation.
     Only the certificate (log mode) would record it.
   * A limit below 1.95 mm (for example 1.5 mm, still above the 1.006 mm transient that motivated the exception)
     would close this. It is a change to a scored rule, so it is left for a decision; the reference would have to
     be re-validated under it.
7. **The Rat Trap audit has no speed or instability check.** Alpha's has both (S3). An integrator
   divergence, which MuJoCo resolves by silently resetting the state, is not a violation in the Rat Trap.
8. **`RatTrapEpisode.score` does not validate its parameters.** `CSpaceTunnelPotentialReward` and
   `ScaledProgressScore` refuse $\kappa \ge 1 - \alpha$; the Rat Trap reads `score_alpha` / `score_progress_cap`
   from the YAML without that check. With the committed values (0.5, 0.3) the ordering holds and is tested.
9. **`select_checkpoints` does not guarantee the $2\varepsilon$ separation for the last checkpoint.** When the final
   state replaces the last kept one, it is not re-checked against its new predecessor. In the committed file every
   consecutive separation is at least 35.7 mm > 33.2 mm, so the property holds for the data in use.

---

## Writing your own reward

### The interface

Every component subclasses `RewardComponent` (`loophole/reward/components.py`) and sees only `RewardState`
snapshots taken by the environment after each action, never anything the agent reports:

```python
@dataclass(frozen=True)
class RewardState:
    puzzle: str
    actions_used: int
    max_actions: int
    success: bool
    violations: int
    done: bool
    topology_word: str = ""
    state_hash: str = ""
    cost_to_go: float | None = None
    start_cost_to_go: float | None = None
    best_cost_to_go: float | None = None
    stable_best_cost_to_go: float | None = None
    min_distance: float | None = None
    geometry: Mapping[str, Any] | None = None     # named measured geometry (Rat Trap: "checkpoint_potential",
                                                 # and, with alignment terms, "cord" and "disc:<name>")
    terminated: bool   # property: success or violations > 0 (absorbing)
    truncated: bool    # property: done and not terminated (budget or external stop)

class RewardComponent(ABC):
    name = "component"
    def __init__(self, **params): ...
    def reset(self, initial: RewardState) -> None: ...
    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float: ...   # dense, per action
    def terminal(self, final: RewardState, info: dict) -> float: ...                # once, at the end
```

`info` (Alpha) carries what the simulator measured during the action: `goal`, `min_distance`, `last_penetration`,
`args` (as requested), `executed` (as executed), `tool`, `diagnostics`, `linking_integral`.

For shaping, subclass `PotentialShaping` and define only the potential:

```python
class PotentialShaping(RewardComponent):
    def __init__(self, gamma: float = 0.99, lam: float = 1.0, **kw): ...
    def phi(self, s: RewardState) -> float | None: ...   # None: unavailable here, no shaping this step
    # step() = lam * (gamma * Phi_bar(cur) - Phi(prev)), Phi_bar = 0 if cur.terminated; terminal() = 0
```

### Register and enable

**Alpha (and any `LoopholeEnv` puzzle).** Register the class under a type name, then list it in the config:

```python
from loophole.reward.components import register_component
register_component("my_component", MyComponent)
```

```yaml
# loophole/configs/alpha_training.yaml (served to both the sim server and the verifier via LOOPHOLE_CONFIG)
reward:
  components:
    - type: my_component
      weight: 0.0          # shaping weight in the per-step sum
      in_score: false      # true only for a component whose terminal() is (part of) the score
      enabled: true
      params: {...}        # passed to __init__
```

`RewardComposer.from_config` imports `reward/literature_rewards.py` and `reward/alignment.py` to register their
components. A new module must be imported before the composer is built: either import it there, or register from a
module that is already imported.

**Rat Trap.** `RatTrapEpisode` does not use the composer. Its score is `RatTrapEpisode.score()`, and its shaping is
the list built by `shaping_components(spec)` in `puzzles/rattrap/reward.py`. Add a `PotentialShaping` there (reading
`RewardState.geometry`, which `RatTrapEpisode._state` fills), with any parameters as `rattrap.yaml` keys.

### How the composer combines components

For entries $(w_j, C_j)$, with $m_j$ = 1 if the entry has `in_score: true`:

$$
r_t = \sum_j w_j\, C_j.\mathrm{step}(s_{t-1}, s_t, \text{info}_t), \qquad
S = \mathrm{clip}\Bigl(\sum_{j} m_j\, w_j\, C_j.\mathrm{terminal}(s_T),\ 0,\ 1\Bigr).
$$

* $r_t$ is logged as the action's `reward`, with each component's unweighted step value under `rewards`.
* $S$ is the score `finalize()` writes and the verifier recomputes.
* `breakdown()` logs each component's weight, step sum and terminal value.
* With `auto_roadmap_components: true`, the composer also appends `cost_to_go_shaping` and `progress` when they
  are missing. The Alpha config turns this off.

### The rules

1. **Shaping must be potential-based.** Write a `PotentialShaping` subclass and let its `step` compute
   $\lambda(\gamma\bar\Phi(s') - \Phi(s))$. Penalties that are not potential differences, such as clearance
   shaping's, must stay at weight 0 or be justified separately: they do not telescope, so they can be farmed or can
   change the optimal policy.
2. **Shaping never enters the score.** `in_score: false`, `terminal()` = 0. The score is computed from the final
   state by a scoring component, never by summing step rewards.
3. **No privileged state to the agent.** A component may read the simulator's measurements, the roadmap, the
   checkpoints, anything. None of it may reach an MCP tool result, `get_state`, or anything else the agent
   receives. Rat Trap tool results carry no score, progress, checkpoint index, distance, word or norm
   (`test_observations_and_results_carry_no_score_or_progress`).
4. **The verifier recomputes it from the logged actions.** A scored quantity must be a deterministic function of
   the replayed physics and committed files (puzzle, config, roadmap, checkpoints, each pinned by hash in the log
   header). Never read a value the server or the agent wrote. If the component needs a new data file, add its hash
   to the header and to the verifier's checks.
5. **Keep every solve above every non-solve.** Any partial credit must be capped below $1 - \alpha$.

### Required tests

Each has a template in the existing suites; copy it for your component.

| test | requirement | templates |
|---|---|---|
| **no-op scores zero** | an episode with no actions scores exactly 0, and its shaping return is 0 | `test_reward.py::test_noop_scores_zero`, `test_rattrap_checkpoints.py::test_noop_scores_zero` |
| **stop-early near zero** | the reference's first few actions, then stop: score 0 (or within the progress actually made, which must be tiny) | `test_reward.py::test_stop_early_scores_zero`, `test_rattrap_checkpoints.py::test_stop_early_scores_zero` |
| **oscillation nets nothing** | going there and back returns the same shaping as standing still ($\gamma$ = 1: exactly 0; $\gamma$ < 1: equal discounted sums) | `test_shaping_scaffold.py::test_discounted_shaping_depends_only_on_the_endpoints`, `test_rattrap_reward.py::test_oscillation_through_a_hole_earns_nothing`, `test_rattrap_checkpoints.py::test_jiggling_near_the_start_earns_nothing` |
| **any solve beats any non-solve** | on a grid of actions used and progress, the worst solve outscores the best non-solve | `test_rattrap_checkpoints.py::test_every_solve_beats_every_non_solve_on_a_grid`, `test_reward.py::test_composer_keeps_solve_above_non_solve` |

Also run the verifier tests (`test_reward.py::test_verifier_*`, `test_rattrap_verify.py`): a new scored quantity
must survive replay and reject a tampered log.

### A short working example

A potential on the escape test's cheapest signal for Alpha: the fraction of the separation distance still to go.
It is shaping only, weight 0 by default, and reads only `min_distance`, which the environment measures.

```python
# loophole/reward/separation.py
from loophole.reward.components import PotentialShaping, RewardState, register_component


class SeparationPotential(PotentialShaping):
    """Phi = -max(0, d_sep - d) / d_sep: -1 in contact, 0 once the pieces are d_sep apart. Training shaping only."""

    name = "separation_potential"

    def __init__(self, d_sep: float = 0.4, gamma: float = 1.0, lam: float = 1.0, **kw):
        super().__init__(gamma=gamma, lam=lam, d_sep=d_sep, **kw)
        self.d_sep = float(d_sep)

    def phi(self, s: RewardState) -> float | None:
        if s.min_distance is None:
            return None  # unavailable: no shaping this step
        return -max(0.0, self.d_sep - float(s.min_distance)) / self.d_sep


register_component("separation_potential", SeparationPotential)
```

```yaml
- type: separation_potential
  weight: 0.0
  in_score: false
  params: {d_sep: 0.4, gamma: 1.0}
```

```python
# loophole/tests/test_separation_potential.py
import pytest
from loophole.reward.components import RewardState
from loophole.reward.separation import SeparationPotential


def _rs(d, **kw):
    return RewardState(**{**dict(puzzle="x", actions_used=1, max_actions=10, success=False, violations=0, done=False,
                                 min_distance=d), **kw})


def test_noop_and_oscillation_net_nothing():
    c = SeparationPotential()
    c.reset(_rs(0.04))
    states = [_rs(d) for d in (0.04, 0.04, 0.10, 0.02, 0.04)]
    assert sum(c.step(a, b, {}) for a, b in zip(states, states[1:])) == pytest.approx(0.0)


def test_terminal_and_truncation():
    c = SeparationPotential()
    prev, after = _rs(0.04), _rs(0.10)
    assert c.step(prev, after, {}) == pytest.approx(0.06 / 0.4)                       # 0.15
    assert c.step(prev, _rs(0.10, success=True, done=True), {}) == pytest.approx(0.9)  # Phi_bar = 0
    assert c.step(prev, _rs(0.10, actions_used=10, done=True), {}) == pytest.approx(0.15)  # truncation keeps Phi
    assert c.terminal(after, {}) == 0.0                                                # never part of the score


def test_unavailable_gives_no_shaping():
    assert SeparationPotential().step(_rs(None), _rs(0.1), {}) == 0.0
```

A real episode also needs the composer-level checks: the no-op and stop-early scores unchanged, and the
solve-above-non-solve grid. Those hold automatically for a component with `in_score: false`, because the score
does not read it. Confirm them with the existing tests above after enabling it.

**Why this example is only a template.** Straight-line separation has the same weakness as clearance shaping
(A6): it can grow in a dead end. For the Alpha puzzle the C-space tunnel potential (A5) is the
signal to use.

## References

Full entries, with DOIs and arXiv identifiers checked, are on the [References](references.md) page: [ZBKV20], [JSCP15],
[NHR99], [WL94], [Mir96], [TLA+23], [NSA+22], [KL00], [RN11], [PS02], [MB24], [BLK11], [TXR+26], [HO26].
