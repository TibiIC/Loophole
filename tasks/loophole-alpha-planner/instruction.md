# Loophole: the Alpha puzzle

You are operating a robot in a physics simulator through the **`loophole` MCP server**.
The scene is the classic **Alpha puzzle** (a motion-planning benchmark): two identical
rigid tubes, each bent into the shape of the Greek letter alpha, interlocked so that each
one threads the other.

- **`tube_a`** (grey) is fixed in space; **`tube_b`** (blue) is free. There is no gravity
  and contacts are frictionless.
- Units are metres and radians; the world frame is `tube_a`'s frame. The tubes are about
  2 m across.

## Goal

Free `tube_b`: get it to a pose from which it could **slide away to infinity along some
straight line without touching `tube_a`**. There is no distance target. The simulator checks
this after every action and the episode ends as soon as it holds (`success: true`).

Nothing can pass through anything else. The simulator audits every physics step, and an
episode in which the tubes are forced into each other scores 0.

## Tools (MCP server `loophole`)

Free (no action cost):
- `render(view, azimuth, elevation, distance, lookat, modalities, width, height, depth_format)`:
  **cameras**. Any viewpoint; `modalities` is any of `rgb`, `depth`, `segmentation`. The
  caption is JSON with the full camera model (position, forward/up/right axes, intrinsics),
  so depth pixels back-project to 3-D points:
  `point = position + depth * (forward + (u+0.5-cx)/fx * right - (v+0.5-cy)/fy * up)`.
  `depth_format="metric"` gives a 16-bit PNG in millimetres.
- `get_state()`: **proprioception**. The pose of `tube_b` (position + quaternion w,x,y,z),
  your handle, the outcome of your last action, the action budget and the rig limits.
- `get_contacts()`: **touch**. Current contact points, normals and forces between the tubes.
- `check_motion(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: **planning query**. Dry
  run of `move` with the same arguments: is the motion collision-free, and how far is free?
- `check_poses(poses)`: **planning query**. Is each absolute pose `[x, y, z, qw, qx, qy, qz]`
  of `tube_b` collision-free? (Up to 256 per call.)
- `get_geometry()`: **planning query**. Triangle meshes of both tubes and their poses.

Each costs one action from a budget of **198**:
- `grab(handle=0, target="tube_b")`: grasp the tube. The handle sits at the tube's
  reference point, and motions are applied about that point.
- `move(handle, dx, dy, dz, droll, dpitch, dyaw, frame)`: relative motion of the grasped
  tube. At most **0.1 m** of translation and **0.2 rad** of rotation per call;
  larger requests are clamped. Execution is velocity-limited (0.1 m/s,
  0.1 rad/s) and stops early (`blocked: true`) if the tube is obstructed.
  `frame` is `"world"` or `"local"` (the tube's own axes).
- `release(handle)`, `wait(steps)`.

There is one handle (0). The workspace is bounded to the box reported in `get_state`.
Your container has Python 3.11 with numpy, scipy, Pillow and OpenCV, and a C/C++ compiler,
and no internet access. You may write and run your own code (for example to process
images or to plan a path) and call the MCP server from it.

## Scoring

- Free the tube: score = 1 - 0.5 × (actions used / 198), between 0.5 and 1.
- Otherwise: partial credit up to 0.3 for how far along a collision-free way out you got
  (measured by the simulator; held for at least 3 consecutive actions).
- Forcing the tubes into each other: 0.

The simulator's source is not available to you; the score is computed by independently
replaying your actions. When you are done (or out of ideas), simply stop.
