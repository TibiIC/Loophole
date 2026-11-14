# Loophole: Alpha-style tubes (medium)

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is a 3D *disentanglement puzzle* in the spirit of the classic Alpha puzzle:

- two identical rigid **tubes**, each bent into an "alpha" (a loop whose two ends cross
  with a small gap between them, then continue as tails),
- **`tube_a`** (grey) is fixed in space; **`tube_b`** (blue) is free. There is no gravity.
- The tubes are interlocked: each loop threads the other.

## Goal

Separate the tubes: move `tube_b` so that its minimum distance to `tube_a` exceeds
**0.06 m** (`get_state` reports it under `goal`). The episode ends as soon as this holds.

Nothing can pass through anything else: the simulator detects and fails any attempt to
force the tubes through each other. The way out is narrow (a few millimetres of clearance
for the harder variants), so you have to find it, not force it.

## Tools (MCP server `loophole`)

Observation tools are free:
- `get_state()`: JSON with the pose of `tube_b`, both tube centre-lines (world frame),
  the tube radius, the loop discs, goal distance, action budget, and `progress`: a
  normalised estimate of how far along the shortest collision-free way out you are
  (0 = start, 1 = separated). It tells you whether you are getting closer, not how.
- `render(view, azimuth, elevation, distance, lookat)`: PNG image + caption.

Each of these costs **one action** from a budget of **150**:
- `grab(handle=0, target="tube_b")`: grasp the tube (the handle then sits at the tube's
  own origin, which is the point between its two crossing strands).
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion of the grasped
  tube about the handle, at most 0.05 m / 0.4 rad per call (larger requests are
  clamped). Motion stops early (`blocked`) if the tube is obstructed.
- `release(handle)`, `wait(steps)`.

There is one handle (0).

## Scoring

Separate the tubes and you score 1 - 0.5 × (actions used / 150).
If you don't, you get at most 0.3 × the best `progress` you reached.
Plan with the free observation tools before acting. The simulator's source is not
available to you; the score is computed by independently replaying your actions.
When you are done (or out of ideas), simply stop.
