# Loophole

An RL environment (Harbor + MCP) where an LLM agent solves 3D disentanglement puzzles in a
MuJoCo simulation. The agent talks to the simulator only over MCP, from a separate container,
so it never sees the simulator, the solver or the reference. A separate verifier container
replays every logged action in a fresh simulator to score the episode.

> **PLACEHOLDER: 2-up video.** Not produced yet. It goes here as
> `<video src="media/<file>.mp4" autoplay muted loop playsinline></video>`, with the GIF version
> as the fallback for places a player won't render.

## Results

> **PLACEHOLDER: final results pending.** No numbers yet; this table is filled in when they exist.

| Task | Agent / model | Runs | Solved | Score (mean ± sd) | Notes |
|---|---|---|---|---|---|
| Alpha 1.0 | pending | pending | pending | pending | |
| Rat Trap | pending | pending | pending | pending | |

## The Alpha finding

Same model, same environment, opposite outcomes. The Gemini 4 Carbon runs that solved Alpha 1.0
did not reason about the 3-D shapes directly: they rendered depth and segmentation from many
viewpoints, reconstructed both tubes, wrote their own motion planner, and executed the plan with
no blocked moves. A run of the same model that instead did finite-difference descent on the
exposed cost-to-go stalled against the other tube and failed. The fastest solves even out-scored
the published reference path, because their planners found shorter routes out; the diagnosis is
in [Anomaly diagnosis](ANOMALY.md).

## Read more

* [Results](RESULTS.md): every evaluation run, strategy classes, behaviour axes and plots.
* [Anomaly diagnosis](ANOMALY.md): how agent runs scored above the oracle.
* [Reward design (Rat Trap)](RATTRAP_DESIGN.md): reference solve, score and checkpoint progress.
* [Exploits](EXPLOITS.md): reward and harness red-team attempts and what stopped each one.
* [Harness notes](HARNESS_NOTES.md): task scope, geometry provenance and licence.
* [Variants](VARIANTS.md): harder puzzles prepared from the published archive.
* [References](https://github.com/TibiIC/Loophole/blob/HEAD/REFERENCES.md): every citation, checked.
* [Running the code](https://github.com/TibiIC/Loophole#readme): the repo README, for setup and reproduction.
