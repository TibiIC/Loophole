# Anomaly: agents out-score the oracle

**Observation.** On Alpha 1.0, Gemini 4 Carbon (OpenCode) runs score above our oracle (the
published solution of Zhang et al. 2020, replayed through the rig):

* redesigned task (`sensors` profile, free-escape goal):
  * oracle **0.851** (59 actions);
  * `sbMy6Ld` **0.886** (45 actions);
  * `AbJs9Px` **0.884** (46 actions).
* original interface (min distance > 0.4 m goal, progress exposed):
  * oracle **0.801** (79 actions);
  * `5oFTZb8` **0.838** (64 actions).

Physics validity of the winning runs has been checked visually and by replay. This note
diagnoses *how* they scored higher. Reproduce with:

```
python scripts/diagnose_run.py <trial dirs> --json docs/anomaly-diagnosis.json
```

The raw output is in `docs/anomaly-diagnosis.json`.

## 1. Did the winner use progress or cost-to-go as a search signal?

**No.** Evidence for `sbMy6Ld`, the best run:

* **Not available to it.** It ran with the `sensors` observation profile. `get_state`, action
  results and render captions contain no progress, cost-to-go, distance or goal test (tested
  in `test_sensors_profile_hides_privileged_information`). The only "progress" token in its
  125 k characters of code is a comment inside its own C++ planner ("ensure retracted node
  actually made progress").
* **No probing.**
  * **0 probe-then-reverse patterns.** A pattern here is a move, then ≥ 1 observation, then a
    move undoing ≥ 75 % of it.
  * **0 blocked moves.**
* **44 of 44 moves committed, 0 exploratory.**
* **A scripted burst.** All 45 actions ran within 110 s (median 0.86 s apart), from
  `execute_plan.py`. The `get_state` / `get_contacts` calls between them (43 of each)
  monitored execution: the script stopped if a move was blocked.
* **`AbJs9Px` is the same:** 0 reversals, 0 blocked, 45 / 45 committed, a 199 s burst.

For comparison, on the original interface, which did expose progress:

* **`5oFTZb8`** (0.838) *printed* `progress` and `cost_to_go` while executing its waypoints
  (`prog = info["progress"]["progress"]`). The plan itself was computed beforehand from its
  own reconstruction: 0 reversals, 0 blocked, 63 / 63 committed. That is monitoring, not search.
* **`Kt5DJ3h`** (0.139, unsolved) **did** use the signal for search. Its code estimates
  finite-difference gradients of `cost_to_go` ("Trans grad cost_to_go", "Rot grad
  cost_to_go") and steps down them. It made 36 blocked moves and 2 strict probe-then-reverse
  pairs in 156 moves, and failed. So the exposed progress signal was exploitable as a search
  heuristic, but it did not produce the winning runs.

## 2. Did it execute code, read repository files, or reconstruct the geometry?

| | `sbMy6Ld` | `5oFTZb8` |
|---|---|---|
| executed code | yes: 26 bash calls, 13 files written (Python + C++/OpenMP) | yes: Python + C++ |
| read repository / task files | **no**. It read `/app` (empty), its own OpenCode config and `/proc` info. The agent image holds nothing of the task (EXPLOITS.md #7, #8) | no |
| reconstructed geometry | **yes**. 31 renders, mostly `depth` + `segmentation` at 1280 px with the camera model → back-projected point clouds of both tubes → ICP of B onto A (the pieces are congruent) → centre-line extraction → capsule / distance-field model | yes: space carving from 329 RGB renders (images decoded by hand) |
| planned | own sampling planner in C++ (`solve_alpha.cpp`, OpenMP) on the reconstruction; plan verified offline (`verify_plan.py`), then executed | own planner in C++ on the carved model |

The mechanism is **perception → offline motion planning → open-loop execution**. This is what
the `sensors` interface is meant to allow (genuine vision; the agent may write a planner).

## 3. Path length, measured over physics substeps

Path length of tube B relative to tube A is integrated over every physics substep (1 ms) by
replaying the log (`scripts/diagnose_run.py`), not summed over logged action endpoints.
Translation and rotation are reported separately; L = translation + w · rotation, with
w = 1.54 m. The substep values match the endpoint sums to within 0.1 %, because each move is a
straight SE(3) segment executed without bounce. The exception is the gradient-search run: its
36 blocked moves add 0.4 %.

| run | interface | score | actions a | a/T | translation (m) | rotation (rad) | L |
|---|---|---|---|---|---|---|---|
| oracle `fVDVriY` | redesigned | 0.851 | 59 | 0.298 | 3.24 | **8.23** | **15.92** |
| `sbMy6Ld` | redesigned | 0.886 | 45 | 0.227 | 1.39 | **2.86** | **5.79** |
| `AbJs9Px` | redesigned | 0.884 | 46 | 0.232 | 1.38 | 3.49 | 6.76 |
| `qxQuETk` | redesigned | 0.823 | 70 | 0.354 | 3.44 | 8.18 | 16.05 |
| oracle `Xx78e6N` | original | 0.801 | 79 | 0.399 | 4.52 | 11.70 | 22.54 |
| `5oFTZb8` | original | 0.838 | 64 | 0.323 | 2.83 | 5.03 | 10.57 |
| `Kt5DJ3h` | original | 0.139 | 157 | 0.793 | 5.19 | 10.91 | 21.99 |

The two fastest runs move tube B along paths 2.4–2.7× shorter than the published solution, and
most of the difference is rotation: 2.9–3.5 rad against 8.2. `qxQuETk` followed essentially the
published route (L 16.05 vs 15.92) and scored below the oracle.

## 4. Strategy classes (all runs: same model, same harness, same settings)

Every agent run used `gemini/gemini-4-carbon` through OpenCode with default settings
(`kwargs: None`) and the same timeout. The two interfaces are the only difference between the
batches.

| run | interface | strategy | score | actions | blocked | reversals |
|---|---|---|---|---|---|---|
| `sbMy6Ld` | redesigned | reconstruct-and-plan: depth+seg → point clouds → ICP (B onto A) → own C++ RRT → execute | **0.886** | 45 | 0 | 0 |
| `AbJs9Px` | redesigned | reconstruct-and-plan: same pipeline, PCA + trimmed ICP | **0.884** | 46 | 0 | 0 |
| `qxQuETk` | redesigned | reconstruct-and-plan: same pipeline, planner found the published-like route | 0.823 | 70 | 0 | 0 |
| `5oFTZb8` | original | reconstruct-and-plan: RGB space carving of both tubes → own C++ planner → execute | **0.838** | 64 | 0 | 0 |
| `Kt5DJ3h` | original | **gradient search**: finite differences of the exposed `cost_to_go`, step downhill | 0.139 | 157 | 36 | 2 |
| `wRU3ykj` | original | reconstruct (PCA + ICP alignment of voxel clouds), aborted by a Gemini API error before any action | 0.000 | 0 | 0 | 0 |

No run used pure direct interaction, i.e. moving and reading state without either a model or
a gradient. `5oFTZb8` and `Kt5DJ3h` are the cleanest contrast: same model, same environment
version, same eval batch. Tool-building wins; direct search on the cost-to-go signal loses.

## 5. Congruence made reconstruction easy

Alpha 1.0's two pieces are congruent: tube B is an exact rigid copy of tube A. All three
redesigned-task winners and the aborted `wRU3ykj` exploited this. They reconstructed point
clouds of both tubes from depth + segmentation, then **registered B onto A with ICP**
(PCA initialisation, 4–24 sign or rotation candidates, then point-to-point refinement). The
result was B's exact pose in A's frame and one shared shape model, rather than two independent,
noisy reconstructions.

A non-congruent variant would remove this shortcut: each piece would have to be reconstructed
and modelled separately, with no cross-check between the two clouds. That is the stated
reason to prefer a non-congruent puzzle as a harder variant (see the escalation note).

## Refuted hypotheses

| hypothesis | test | result |
|---|---|---|
| H1. The winners used progress / cost-to-go as a search signal | probe-then-reverse count, blocked moves, code for reads of progress / cost-to-go | **Refuted.** The redesigned task exposes no such value; the winners made 0 reversals and 0 blocked moves, executing a precomputed plan. Only the losing `Kt5DJ3h` searched on it. |
| H2. The oracle is penalised for tool-call granularity; a path-length efficiency term would make it best | path length over physics substeps, normalised by the oracle's | **Refuted.** The agents' paths are 2.4–2.7× *shorter*. A path-length metric widens their lead (illustrative scores, solved → 1 − 0.5 · min(1, L / (2.5 · L_oracle)): 0.927 / 0.915 vs oracle 0.800). The oracle already uses the same per-call caps (0.95 safety factor). |
| H3. A tighter action budget (~1.5× the oracle) would penalise trial and error and create a gap | actions used by every run | **Refuted.** The winners used 45–46 of 198 actions. A cap of ≈ 1.5 × 59 ≈ 89 would not bind them; it would only truncate the gradient-search run (157), which already scores 0.139. |

Steps 2 (path-length efficiency term) and 4 (action cap) of the brief were therefore dropped.

## Conclusion

* The agents did not game the reward. They solved the puzzle with their own perception and
  planning, and found **shorter** collision-free paths than the published one.
* The oracle is a *validity* reference (it proves solvability and anchors partial credit),
  not an optimal one.
* The fix for the reference is to implement it properly: shortcut-smooth the published path
  in our own simulator, rotation first (section "Smoothed oracle" below, once it lands).
* The progress signal was only exposed on the original interface. There, one agent used it
  for gradient-style search, and that run did not win.
