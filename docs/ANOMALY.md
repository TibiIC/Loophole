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

## 3. Action counts, a/T, and a path-length efficiency metric

Path length = SE(3) path of tube B relative to tube A over the logged poses:
Σ ‖Δt‖ + w·∠, with w = 1.54 m.

The path-length score column uses solved → 1 − 0.5 · min(1, L / (2.5 · L_oracle)). Here 2.5
is the same budget ratio as T = 198 ≈ 2.5 × the oracle's original 79 actions. This is an
illustration ahead of step 2; the constants are not tuned.

| run | goal | actions a | a/T | score (actions) | path length L | L / L_oracle | score (path length) |
|---|---|---|---|---|---|---|---|
| oracle `fVDVriY` | free escape | 59 | 0.298 | 0.851 | **15.92** | 1.00 | 0.800 |
| `sbMy6Ld` | free escape | 45 | 0.227 | 0.886 | **5.79** | 0.36 | **0.927** |
| `AbJs9Px` | free escape | 46 | 0.232 | 0.884 | **6.76** | 0.42 | **0.915** |
| oracle `Xx78e6N` | distance 0.4 m | 79 | 0.399 | 0.801 | **22.54** | 1.00 | 0.800 |
| `5oFTZb8` | distance 0.4 m | 64 | 0.323 | 0.838 | **10.57** | 0.47 | **0.906** |

**The agents' paths are 2.1–2.7× shorter than the oracle's.** A path-length metric
**widens** the agents' lead instead of removing it. The oracle loses on motion quality, not
on tool-call granularity:
* It is the authors' *simplified* tunnel-discovery path (22 poses). That path was planned for
  validity in their C-space, not optimised for length in our SE(3) metric.
* It rotates 11.6 rad in total; `5oFTZb8` rotated 5.0 rad.
* Its granularity cost is small. It uses the same per-call caps (0.1 m, 0.2 rad) at a 0.95
  safety factor, plus per-segment rounding. That costs a few actions, not the 14–20 gap.

## Conclusion

* The agents did not game the reward. They solved the puzzle with their own perception and
  planning, and found **shorter** collision-free paths than the published one.
* The oracle is a *validity* reference (it proves solvability and anchors partial credit),
  not an optimal one.
* Changing the efficiency term to path length (step 2) will not make the oracle the best
  score as it stands. Only a shorter oracle path would, for example the published path
  shortened in our own simulator.
* The progress signal was only exposed on the original interface. There, one agent used it
  for gradient-style search, and that run did not win.
