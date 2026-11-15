# Docs-site clips

Short illustrative clips, not full runs. Each has a full-quality `.mp4` (for the site: autoplay,
muted, loop) and a palette-optimised `.gif` (≤ 600 px wide, 12 fps) for places a video player
won't render, such as GitHub markdown. Made by `scripts/make_doc_clips.py`, which replays the
logged actions in a fresh simulator and renders them. The replay reproduced each run's logged
outcome; per-clip timing is in `<clip>.timing.json`.

## `clip-alpha-reference`: Alpha 1.0, reference solve

The published solution path (Zhang et al. 2020), replayed through the agent's rig: Harbor
oracle trial `loophole-alpha__fVDVriY` (run `20261114-111735`), solved in 59 actions. Left:
iso view; right: top view. The whole solve is compressed into about 11 s, but not uniformly:
the approach and the exit run at 0.1 s per action, and the narrow passage, actions 43–56 (the
longest stretch where tube B's clearance to tube A is under 3 mm), slows to 0.35 s per action
so the twist that does the work stays readable. The label marks the slowed stretch.

## `clip-rattrap-reference`: Rat Trap, reference solve

The human teleoperated Rat Trap solve, replayed exactly in the simulator, at 20× sim time.
Left: iso view; right: front view (the source is already a 2-up of two cameras). Used whole
(5.8 s); it is the existing `runs/rattrap-solve.mp4` from the rattrap work, not re-timed or
re-encoded. The GIF is made from it.

## `clip-rattrap-failure`: Rat Trap, an agent failing

Kimi K3 Pro (`kimi-k3-1-pro`), run `kimi-20261114-185131` (`runs/rattrap-agents/` on the
rattrap branch): 357 actions, 47 blocked moves, 0 / 20 checkpoints, score 0, ended at the
time / turn limit. The clip is actions 45–72, the 28-action stretch with the most blocked
moves (11): the agent grabs a cord node, pushes, is blocked, releases, grabs another node and
pushes again (four blocked in a row at 66–69). Every action gets 0.25 s; blocked moves are
labelled. Left: iso view; right: front view. Replayed from the run's `commands.jsonl` through
`RatTrapEpisode`; the replay reproduces the committed re-score (357 actions, 47 blocked, 0).

Not the run with the most blocked moves overall: that is `kimi-20261114-205209` (141), whose
command log is not committed.

## `clip-rattrap-contrast`: Rat Trap, human reference beside an agent (lead clip)

The lead clip. The only text in the frames is "human reference — solved" and "agent — 0/20 checkpoints";
counts live in the results table, not in the clip. Left: the human teleoperated solve. Right: Kimi K3 Pro,
run `kimi-20261114-185131` (round 1), illustrating the failure mode. Iso view on both sides.

* **Reference half.** Re-rendered from the committed recording (`loophole/reference/rattrap-solve.jsonl`)
  by exact replay, one frame per 20 ms of sim time: 5,766 frames, all 75 grab / release events.
  `make_doc_clips.py rattrap-reference-capture`.
* **Agent half.** The run's `commands.jsonl` replayed through `RatTrapEpisode` at 640×480; the replay
  reproduces the committed re-score (357 actions, 47 blocked). `make_doc_clips.py rattrap-capture`.
* **Alignment by action index.** Every action gets 1/15 s on both sides. The reference's frames are mapped
  to the 284 commands of the converted solve (`loophole/reference/rattrap-solve.commands.jsonl`): its 75
  grab / release events match the recording's 75 events one to one, and the moves between two events are
  spread evenly between them. No frame repeats while the reference is moving. Once it is solved it holds
  its final frame while the agent continues.
* Inputs checked identical on `main` at `b8b82ad`.

## Files

| clip | GIF | MP4 | length |
|---|---|---|---|
| `clip-rattrap-contrast` | 600×224, 1.1 MB | 1284×480, CRF 16, 1.7 MB | 26.3 s |
| `clip-alpha-reference` | 600×226, 1.2 MB | 960×360, 0.51 MB | 10.7 s |
| `clip-rattrap-reference` | 600×226, 0.47 MB | 1280×480, 0.25 MB | 5.8 s |
| `clip-rattrap-failure` | 600×226, 0.39 MB | 960×360, 0.23 MB | 7.5 s |
| `clip-alpha-contrast` | 600×224, 1.3 MB | 964×360, 0.51 MB | 17.8 s |

## `clip-alpha-contrast`: Alpha 1.0, a solve beside a failure

Illustrative, not a controlled comparison (see the last paragraph).

Gemini 4 Carbon (OpenCode), two runs side by side, iso view, aligned by action index: every
action gets 0.1 s on both sides, the same rate as the fast parts of `clip-alpha-reference`, with
no slowdown for the twist.
* Left, `sbMy6Ld` (trial `loophole-alpha__sbMy6Ld`, run `20261114-111735`): reconstructed
  the geometry from renders, planned, and executed the plan; solved in 45 actions, then held
  on its final frame.
* Right, `Kt5DJ3h` (trial `loophole-alpha__Kt5DJ3h`, run `20261114-080705`): finite-difference
  descent on the exposed cost-to-go; it keeps pushing until it stops at 157 of its 198 actions,
  unsolved. It was not cut off by the budget.

The two runs come from different interface rounds: `sbMy6Ld` from the redesigned task
(free-escape goal, no progress signal shown), `Kt5DJ3h` from the original one (cost-to-go
exposed in `get_state`). The puzzle and start pose are identical (same initial-state hash); the
rig differs only in grasp tolerance (5 cm for `sbMy6Ld`, 3 m for `Kt5DJ3h`; tightened later, see
`docs/EXPLOITS.md` #11). Each side is replayed under its own logged config. See `docs/RESULTS.md`.
