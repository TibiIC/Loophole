# Watching a Rat Trap run live

A live viewer for us, not for the agent: a browser page showing the cord after every action, with the run's progress
beside it. **It is off for scored runs.** The Harbor tasks, the oracle and the verifier never start it, and nothing of it
reaches the agent: it has its own HTTP port, never MCP, and adds nothing to `get_state` or any tool result.

## Start a run with the viewer

In-process (no Docker): the viewer binds to `127.0.0.1`.

```bash
make watch                 # the reference (Tibi's recorded solve, replayed closed-loop), viewer at http://127.0.0.1:8770/
make watch WHO=random      # a random agent
make watch WHO=kimi        # a model through scripts/agent_rattrap.py (key in .env)
make watch PORT=8771       # another port
```

The command prints the URL. Open it in a browser. The page polls once a second.

From the Docker simulator image, with no agent container: both ports are published on the host's `127.0.0.1` only.

```bash
docker compose -f docker/live-view.compose.yaml up --build
# drive it from the host, e.g. with the reference:
python tasks/loophole-rattrap/solution/follow_waypoints.py tasks/loophole-rattrap/solution/waypoints.jsonl --url http://127.0.0.1:8000/mcp
# open http://127.0.0.1:8770/
```

Inside a Harbor run the viewer is not available, and that is deliberate. The agent container's network allowlist is per
host, so any viewer port on the `loophole-sim` sidecar would be within the agent's reach.

The simulator server takes `--live-view PORT` (and `--live-host`, default `127.0.0.1`). `scripts/agent_rattrap.py` takes
`--live-view PORT`.

## What the page shows

* The latest frame: iso and front views of the frame, the cord and the mouse, rendered after the last action.
* **actions**: actions used out of the budget (T = 852).
* **checkpoints**: reference checkpoints reached in order, out of 20.
* **distance to next**: the max per-node distance from the cord to the next checkpoint, in mm. A checkpoint is reached
  below ε = 16.6 mm.
* **last move blocked**: whether the last `move` halted on contact.
* **last call**: the last tool call, with its arguments as the agent gave them.
* **state**: running, solved, violation or ended.
* Below: the last ten tool calls.

## Where the frames land

Every frame is appended as the run goes, so the run's video exists when it ends, with no re-simulation:

* in-process: `runs/watch/<time>-<who>/frames/NNNNNN.png`, plus `frames.jsonl` (the overlay data for each frame);
* Docker: `runs/watch/docker/frames/`.

```bash
make watch-last            # encodes the most recent watched run's frames into run.mp4 (ffmpeg, 4 frames/s) and opens it
```

## What it costs

Rendering and writing one frame takes 0.05 s per action, measured over 40 actions of the reference replay: about 3 % of
a Rat Trap action (1.5 s in-process).

With the viewer on, off, and before the viewer existed, the episode log of those 40 actions (every action's state hash)
was byte-identical: SHA-256 `baef23f4…`. `loophole/tests/test_rattrap_live_view.py` keeps it that way. It checks the
viewer is off by default, that tool results and logs are identical with it on, that it binds to localhost, and that the
MCP tool list is unchanged.
