# Rat Trap reward: open issues (2026-11-14)

> **Superseded.** The ‖w‖ partial credit has been replaced by ordered checkpoint matching, and the
> conversion is fixed. Current design: [RATTRAP_DESIGN.md](RATTRAP_DESIGN.md). Kept as the record
> of why.

A brief for ideation. Branch `rattrap`; code in `loophole/puzzles/rattrap/`
(geometry `rattrap.yaml`, topology `topology.py`, norm `norm.py`, reward `reward.py`).

## What is solid

* **Violation → 0.** Any contact pressed deeper than `penetration_tolerance` (1 mm) ends the
  episode with score 0. A swept-motion certificate now also checks every substep interval of
  the cord against the structure (`certify.py`, mode `log`). Your recorded solve: 115,336 of
  115,336 intervals certified.
* **Success test.** `FrozenEscape`: freeze the cord's current shape (with the mouse on it) as
  one rigid body; it is free if it can translate to infinity along one of 64 directions
  without touching anything. Conservative advancement plus exact checks near contact.
  * Sound: it never calls a trapped cord free.
  * Strict: a cord that is unlinked but draped over a post doesn't count until it's pulled clear.
* **Solved score:** 1 − 0.5·a/T. This is what the first agent runs use; unsolved scores 0.

## What is not solid: partial credit and dense shaping

The design was: unsolved → 0.3·p, with p = 1 − min_t ‖w_t‖ / ‖w₀‖, and training shaping from
the potential Φ = −‖w‖. Here w is the cyclically reduced word of the closed cord's signed
crossings through declared discs, and ‖·‖ is the Przytycki–Sikora norm (arXiv:math/0007134,
Thm 3.2): the minimal number of hole passes, with the passable generators declared in the YAML.

### 1. The disc system is incomplete (the main problem)

Six discs are declared:
* A, B, C: three regions of the frame opening, split by chains of the structure;
* U, L: the annuli of the two post holes around the dowels;
* R: the ring's interior minus the upper post.

The word is only a faithful invariant if these discs cut the complement of the rigid
structure into a simply connected region. That is, if the structure is a handlebody and
the discs are a complete meridian system. They are not, because parts thread through each
other without touching:

* the right dowel passes through the lower post's hole (an annular gap all round);
* the left dowel passes through the upper post's hole;
* the upper post passes through the ring.

Each of these makes extra independent loops in the free space. U and L are annuli, not
discs, and R minus the post is not a disc either.

**Evidence:**
* Your recorded solve (221 commands, 37 grab/release cycles) reads start word `Ac`.
* The word becomes **empty at about command 10**, while the cord is still trapped.
* So p = 1 (score 0.3) is reachable in about 10 actions without real progress. It's
  gameable and carries no signal.

**Open questions:**
* What is the genus of the structure? By Alexander duality, b₁(complement) = b₁(structure).
* Is the complement's π₁ free at all? Everything rigid is welded into one connected body,
  so it should be a handlebody unless something is knotted. That needs checking.
* What is a complete disc system? A candidate: split each annulus with a radial cut, and
  the ring interior into the parts above and below the post.
* How do we verify completeness? A candidate: voxelise the free space, remove the disc
  voxels, then check that it's connected and that b₁ = 0, using the Euler characteristic
  of the cubical complex plus counts of components and cavities.

### 2. The mouse is not in the word

The mouse is a bead tied to node 71. It cannot pass U, L, R or the gaps beside the sphere;
it can pass the end gaps of the posts. We encode it by declaring U, L and R not passable in
the norm.

This is too coarse. The bare cord passes every hole; only the stretch of cord carrying the
mouse cannot. A cyclic word forgets where the mouse sits between the crossings. A faithful
state is probably a word with a marked point for the mouse, i.e. a based loop. Then the only
forbidden moves are those that carry the marked point through a non-passable disc. The norm
would need re-deriving for that setting, and I haven't done that.

### 3. Homotopy versus isotopy, and the norm's scope

* The word sees homotopy, but the cord can't pass through itself. A trivial word is necessary
  for freedom but not sufficient: self-entanglement around a post is invisible to it.
* P&S state the norm for string words in their Chinese Rings setting; we apply it to the
  cyclic word of a closed loop. That transfer is assumed, not proven.
* A related check: the literal ‖a_n‖ (Prop. 2.2′) is not 2^{n−1}; it's ∞ or 1. The
  2^{n−1} value belongs to the pair (a_n, a_{n−1}) with only g_n passable. I wrote this up
  earlier when a test was specified the other way.

### 4. Conversion fidelity (resolved)

* We have a reference run: your teleoperated solve replays exactly and ends free (within 2.8 mm).
* The problem was **conversion fidelity**: the discrete-command conversion (0.5 mm path
  simplification) ended at word `Ab`, not free.
* **Cause:** coarse path following makes moves halt on contact at different points.
* **Fix:** closed-loop conversion at 0.1 mm RDP reproduces the solve. See RATTRAP_DESIGN.md.

### 5. Shaping mechanics, after the PotentialShaping port

* (Fixed: the −1e6 branch is gone.) Words the norm cannot reduce got potential −1e6. With Φ̄ = 0 on termination, an episode
  ending from such a state would be paid +1e6.
* Because Φ̄ = 0 at every terminal state, the shaping return of every terminated episode is
  −Φ(s₀). A crash ties a solve unless the learner adds the task reward (`info["score"]`)
  at the last step. Shaping and score are deliberately kept apart, as for Alpha.

### 6. Provisional inputs

* Every dimension in `rattrap.yaml` is provisional, read off a photo.
* The clearance table, which sets the passability flags and the segment count, depends on
  those dimensions.

## Ruled out, or not available

* **Euclidean distance to an exit:** ignores that the cord must travel along the
  constraint. Our colleague's ring-on-wire environment, with a distance potential plus
  orientation alignment, got 0/300 successes.
* **Roadmap cost-to-go (as for Alpha):** the configuration space is about 180-D and
  deformable; no roadmap is feasible.
* **Gauss linking numbers of the cord with the structure's cycles:** a sound detector of
  "not free" (non-zero ⇒ linked), but zero doesn't imply free. It's abelian, so it misses
  commutator entanglement. At most a diagnostic or a lower bound.

## Candidate directions

1. **A complete, verified disc system** (issue 1), then a mouse-marked word (issue 2), then
   a norm for based words. This is the principled fix, if the complement's π₁ is free.
2. **Milestones from the recorded solve:** progress = the furthest checkpoint of your solve
   whose topological state has been matched. Needs a faithful state (1) to define a match.
3. **Abelian invariants as a lower bound:** linking numbers with the H₁ generators;
   progress only when a non-zero invariant reaches zero. Sound but coarse.
4. **Sparse only** (what the first agent runs use): solved-or-0, plus violation → 0.
   Sound, no partial credit, so no gradient for weaker agents.

## First agent results (trivial reward)

* DeepSeek V4.1 Flash (via Verda): 322 actions, not solved, score 0.
* Kimi K3 Pro and Nemotron 3.5 Super VL: still running.
* Gemini 4 Carbon: being added.

None has solved it so far.
