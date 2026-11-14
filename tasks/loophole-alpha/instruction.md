# Loophole: the Alpha puzzle

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is the classic **Alpha puzzle** (a motion-planning benchmark): two identical
rigid tubes, each bent into the shape of the Greek letter alpha, interlocked so that each
one threads the other.

- **`tube_a`** (grey) is fixed in space; **`tube_b`** (blue) is free. There is no gravity
  and contacts are frictionless.
- Units are metres and radians, world frame. The tubes are about 2 m across.

## Goal

Separate the tubes: move `tube_b` until its **minimum distance to `tube_a` exceeds
0.4 m**. `get_state` reports the current value under
`goal.target_min_distance_to_frame`. The episode ends as soon as this holds.

Nothing can pass through anything else. The simulator audits every physics step and
an episode in which the tubes are forced into each other scores 0.

## Tools (MCP server `loophole`)

Free (no action cost):
- `get_state()`: JSON with the pose of `tube_b` (position + quaternion w,x,y,z), the
  goal distance, handle status, the action budget, and the rig limits.
- `render(view, azimuth, elevation, distance, lookat)`: PNG image + caption
  (presets: iso, front, back, side, top).

Each costs one action from a budget of **348**:
- `grab(handle=0, target="tube_b")`: grasp the tube. The handle sits at the tube's
  reference point, and motions are applied about that point.
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion of the grasped
  tube. At most **0.05 m** of translation and **0.1 rad** of rotation per call;
  larger requests are clamped. Execution is velocity-limited (0.1 m/s,
  0.1 rad/s) and stops early (`blocked: true`) if the tube is obstructed.
  `frame` is `"world"` or `"local"` (the tube's own axes).
- `release(handle)`, `wait(steps)`.

There is one handle (0). The workspace is bounded to the box reported in `get_state`.

## Scoring

If you separate the tubes: score = 1 - 0.5 × (actions used / 348).
Otherwise: score = 0. Observation tools are free, so look before you move. The
simulator's source is not available to you; the score is computed by independently
replaying your actions. When you are done (or out of ideas), simply stop.
