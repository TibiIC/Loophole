# Loophole: easy disentanglement puzzle

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is a 3D *disentanglement puzzle*:

- a rigid **frame** (grey) fixed to the table,
- a **rope** (orange) whose two ends are anchored to the frame. It is flexible, cannot
  stretch, and slides without friction,
- a free piece, the **`ring`** (blue), which is entangled with the rope and the frame.

A rope bight hangs through a horizontal loop; free the ring trapped around both strands above the loop.

## Goal

Free the `ring`: move it farther from **every rope anchor than the rope could
ever reach** and clear of the frame (`get_state` reports the exact distances under
`goal`). The episode ends as soon as this holds.

Nothing can pass through anything else: the simulator detects and fails any
attempt to force pieces through each other. You have to find the real way out.

## Tools (MCP server `loophole`)

Observation tools are free:
- `get_state()`: JSON with all poses, rope node positions, anchors, holes, handles,
  action budget, goal distances and a description of the frame.
- `render(view, azimuth, elevation, distance, lookat)`: PNG image + caption
  (presets: iso, front, back, side, top).

Each of these costs **one action** from a budget of **60**:
- `grab(handle, target, point?, segment?)`: grasp `ring` or a rope segment (`target="rope"`).
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion, at most
  0.1 m / 0.6 rad per call (larger requests are clamped). Motion stops early
  (`blocked`) if the held body is obstructed.
- `release(handle)`, `wait(steps)`.

You have two handles (0 and 1) that can hold things at the same time.

## Scoring

Score = 1 - 0.5 × (actions used / 60) if you free the `ring`, otherwise 0.
Any solve beats any non-solve, and faster solves score higher. Plan with the free
observation tools before acting. The simulator's source is not available to you;
the score is computed by independently replaying your actions. When you are done
(or out of ideas), simply stop.
