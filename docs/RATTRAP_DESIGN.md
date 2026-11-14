# Rat Trap: reward design

Code: `loophole/puzzles/rattrap/` (`reward.py`, `checkpoints.py`). Reference:
`loophole/reference/rattrap-solve.jsonl` (your teleoperated solve),
`rattrap-solve.commands.jsonl` (its faithful command conversion) and
`rattrap-checkpoints.npz`. Built by `scripts/build_rattrap_reference.py`. Every dimension in
`rattrap.yaml` is still provisional.

## The reference is a human solve, recorded and replayed

* **What it is.** `tasks/loophole-rattrap/solution/reference.jsonl` is a fixed, committed list of 284 commands.
  * It is derived from Tibi's own teleoperated solve, recorded in this simulator (`loophole/reference/rattrap-solve.jsonl`).
  * It was converted once into agent commands (`loophole/reference/rattrap-solve.commands.jsonl`), as described
    under "Reference solve and conversion fidelity".
* **No solver.** There is no solver, planner or search anywhere in the Rat Trap pipeline: not at runtime, not
  offline, not as a fallback. Harbor's "oracle" agent only replays `solution/`; it searches for nothing.
* **What depends on it.** The checkpoints (`rattrap-checkpoints.npz`) are states of that same recorded run, and ε
  and T were calibrated on it.
* **What the agent can see of it:** nothing. The agent container holds none of `solution/`, the recording, the
  checkpoint file or anything derived from them; the MCP tools expose none of it. This is tested statically
  (`test_rattrap_isolation.py`) and live (`loophole/harbor_probe.py`; docs/EXPLOITS.md).
* **The contrast with Alpha.** Alpha's reference has a different provenance: the solution path published by
  Zhang et al. 2020, produced by their C-space planning pipeline, converted once and committed. Neither task runs a
  solver, but the Rat Trap's reference shows a *human* can solve the environment as built, while Alpha's shows that
  a *planner's* path is feasible in our simulator.

## Score (the verifier's one float)

| Outcome | Score |
|---|---|
| any contact pressed deeper than `penetration_tolerance` (1 mm) | **0**, and the episode ends |
| solved | **1 − α·a/T**, α = 0.5, a = actions used, T = `max_actions` |
| not solved | **0.3·p**, p = checkpoints reached in order / K |

Solves land in [0.5, 1] and non-solves in [0, 0.3], so every solve beats every non-solve
(tested on a grid of (p, a)).

**Solved** means the frozen cord can slide free (`FrozenEscape`):
* The cord is frozen in its current shape, with the mouse on it, as one rigid body.
* It is solved if that body can translate to infinity along one of 64 directions without
  touching the structure.
* The sweep is conservative advancement far from contact, with exact checks near contact.
* Sound (it never calls a trapped cord free) but strict: a cord that is unlinked but draped
  over a post doesn't count until it's pulled clear.

## Progress: ordered checkpoint matching

**Checkpoints c₁…c_K** come from the reference run (the exact replay of your solve):
* each is the full cord state, all node positions, at one of its grab/release boundaries,
  taken up to the moment the converted solve comes free;
* the mouse is welded at node 71, so the nodes carry it;
* a checkpoint is kept only if it is at least 2ε (max-norm) from the previous kept one and
  from the start, so neighbouring ε-balls are disjoint and the start cannot match c₁.

**Reached.** Checkpoint k is reached at evaluation time t (after each action) if both hold:
* max_i ‖x_i(t) − c_k,i‖ < ε;
* checkpoints 1…k−1 were each reached at strictly earlier times.

One evaluation advances at most one checkpoint, so none can be skipped. The max-norm, not the
mean, means it can't be bought by getting most of the cord roughly right.

**Progress** p = k/K. It is monotone: reached checkpoints stay reached.

**Calibration of ε.**
* Your solve, as the agent-interface commands, runs through the matcher.
* ε_min is the smallest ε at which every checkpoint registers in order (bisection; greedy
  earliest matching is optimal for ordered matching).
* ε = 1.5·ε_min. The thinning depends on ε, so the two are iterated to a fixed point.

**Values:** K = 20 (thinned from the 75 grab/release boundaries), ε_min = 11.1 mm, **ε = 16.6 mm**, minimum separation 2ε = 33.2 mm,
start-to-c₁ distance 110.8 mm.

## Training shaping (never scored)

All shaping goes through `loophole.reward.components.PotentialShaping`, as for Alpha, and is
kept separate from the score:

    F = λ(γ·Φ̄(s′) − Φ(s)),   Φ̄ = 0 once terminated (solved or a violation), Φ kept at the budget cut-off
    Φ = −[(K − k) + min(1, d(x, c_{k+1}) / D_{k+1})],   Φ = 0 once k = K

* d is the max-norm distance, and D_{k+1} = d(c_k, c_{k+1}) with c₀ the start.
* Φ is continuous across a checkpoint: just before reaching c_{k+1}, Φ ≈ −(K−k). Just after,
  the distance to c_{k+2} is about D_{k+2}, so Φ ≈ −(K−k−1) − 1.
* Φ is bounded in [−(K+1), 0]. The old −1e6 branch for words the norm cannot reduce is gone:
  with Φ̄ = 0 on termination it paid +1e6 to an episode ending in such a state.
* γ = λ = 1 in the YAML. With γ = 1 it telescopes, so an oscillation earns only
  Φ(end) − Φ(start).
* Optional AperturePassageAlignment terms (ring and both holes) are computed at weight 0.

## Diagnostics only: the word and its norm (UNSOUND as progress)

The cord's word w (signed crossings through six declared discs, cyclically reduced) and its
Przytycki–Sikora norm ‖w‖ (arXiv:math/0007134, Thm 3.2) are still computed and logged after
every action. They are never scored or shaped, because they are not a sound progress measure here.

* **Evidence.** Along your solve the start word is `Ac`, and the word becomes empty while the
  cord is still trapped:
  * at about command 10 of 221 of the earlier conversion;
  * **after action 15 of 284** of the faithful one (`test_recorded_solve_scores_highest`
    prints it).
  Under the old scheme an agent could earn the full 0.3 partial credit in about 10 actions,
  with no real progress.
* **Cause: the disc system is not a complete meridian system.** The three frame regions
  (A, B, C), the two hole annuli (U, L) and the ring's interior (R) do not cut the complement
  of the structure into a simply connected region, because parts thread through each other
  without touching:
  * each dowel through its post's hole (an annular gap);
  * the upper post through the ring.

  These add independent loops not spanned by the discs. An annulus is not a disc, and nor is
  the ring's interior minus the post. Not fixed here.
* **The mouse is not in the word.** The bare cord passes every hole; only the stretch of cord
  carrying the mouse cannot. A cyclic word forgets where the mouse sits.
* **Homotopy versus isotopy.** The cord cannot pass through itself, so an empty word is
  necessary for freedom but not sufficient.
* **A correction to the original specification.** The literal ‖a_n‖ in P&S Prop. 2.2′ is
  not 2^{n−1}: it is ∞, or 1. The value 2^{n−1} belongs to the pair (a_n, a_{n−1}), the
  distance between them, with only g_n passable. `test_rattrap_norm.py` tests the paper's
  statement in that form.

## Reference solve and conversion fidelity

**We have a reference run.** Your teleoperated solve replays exactly: the recorded handle pose
before every physics step ends free, within 2.8 mm of the live session. The problem was
conversion fidelity: turning that path into the discrete agent commands (`grab_node`, capped
`move`, `release`) without losing the solve.

**Diagnosis** (`scripts/rattrap_conversion.py`, `scripts/rattrap_divergence.py`; plot in
[docs/media/rattrap-divergence.png](media/rattrap-divergence.png)). The physics is unchanged: no added force,
tension or gravity.

* **(a) First divergence.** Measured against the exact replay at every waypoint, the first
  deviation over 2 mm comes 802 (0.1 mm RDP) or 2,939 (0.5 mm RDP) physics steps after the
  first grab. That is mid-stretch, not at a grab or release, so the bug is not grab/release
  timing or node selection. It sits within 200 steps of a blocked motion in the recording.
* **(b) Granularity.**

  | Move cap | RDP tolerance | Moves | Free? | Max deviation at a grab/release boundary |
  |---|---|---|---|---|
  | 30 mm | 0.5 mm | 140 | no (`Ab`) | 77.9 mm |
  | 10 mm | 0.5 mm | 227 | no (`ArbR`) | 74.0 mm |
  | 3 mm | 0.5 mm | 535 | no (`Ab`) | 84.3 mm |
  | 30 mm | 0.1 mm | 262 | **yes** | 12.5 mm |
  | 10 mm | 0.1 mm | 330 | **yes** | 12.9 mm |

  Final divergence does not fall with move size. It falls with path-simplification
  (Ramer–Douglas–Peucker) tolerance: following the recorded path more closely is what matters.
* **(c) Blocked moves.** The recording has 19 blocked-and-snapped motions.
  * At 0.1 mm RDP all 13 converted blocks coincide with recorded ones.
  * At 0.5 mm the conversion produces 19–51 blocks, of which only 15 coincide.

  A move halting on contact at a different point is where the runs part.
* **(d)** Not needed.

**The faithful conversion:** closed-loop (each waypoint approached from where the handle
actually is), 30 mm cap, RDP 0.1 mm.
* **Oracle:** 284 actions to free, under the agent interface.
* **T = 852** (about 3× the oracle, the ratio Alpha uses: 198 for a 59-action oracle), so the
  oracle scores **0.833**.
* The runs logged before this change used T = 400.

## Why staged geometric milestones, not a topological signature, as the reward

* **Topological signatures guide planning.**
  * The h-signature (Bhattacharya, Likhachev & Kumar, RSS 2011) identifies homotopy classes
    of trajectories for search-based planning.
  * The grasp loop signature (Mitrano & Berenson, ICRA 2024, arXiv:2403.01611) categorises
    rope / cable states to guide manipulation planning.
  * Neither is used as a reward scalar.
* **Cable-routing RL decomposes into stages.** SILO (Tao et al., arXiv:2607.04616, §4.2) splits
  multi-harness routing into single-harness subtasks.
  * Each subtask has a simple geometric success test: the two cable links nearest the gripper
    within a radius ε of the harness centre.
  * Within a stage the reward is dense proximity to that stage's target plus success
    indicators.
* **That is our scheme:** stages (checkpoints) with a geometric test (max-norm within ε), and a
  dense term (normalised distance to the next checkpoint) inside the shaping.
* **The cost:** progress is measured along *your* route. A different solution route earns
  partial credit only where it passes near your checkpoints, though a solve is still a solve.

## Evidence from the agent runs

* **Round 1** (sparse reward, T = 400): DeepSeek V4.1 Flash, Kimi K3 Pro and Nemotron 3.5 Super VL used 322, 357
  and 343 actions. Their command logs were replayed offline under the checkpoint reward
  (`scripts/rescore_rattrap.py`; same physics, no API calls).
  * **All three land at p = 0.**
  * None came within ε of the first checkpoint: the closest approaches were 97.7, 70.1 and 110.8 mm, against
    ε = 16.6 mm and a start that is itself 110.8 mm from it.
  * This is difficulty evidence: hundreds of plausible-looking moves don't reproduce even the first stage of a solve.
  * It is also reward-integrity evidence: the partial credit isn't earned by activity, blocked pushing or wandering.
* **Random agent** (`make rattrap-random`, and `test_random_agent_to_the_full_budget_earns_almost_nothing`): random
  grabs, moves and releases to the full budget (see the README table for results).
* Per-model results: the README table.
