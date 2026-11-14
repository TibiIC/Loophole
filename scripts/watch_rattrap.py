#!/usr/bin/env python3
"""Watch a Rat Trap run live (docs/watching.md). In-process; the viewer binds to 127.0.0.1.

    .venv/bin/python scripts/watch_rattrap.py [--who oracle|random|deepseek|kimi|nemotron] [--port 8770]
    make watch [WHO=oracle] [PORT=8770]        make watch-last    (encode the last watched run's frames; no re-simulation)

Frames: runs/watch/<time>-<who>/frames/NNNNNN.png and frames.jsonl. Not a scored run: nothing is verified or archived.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", default="oracle", help="oracle (the recorded reference), random, or a model of scripts/agent_rattrap.py")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "runs" / "watch" / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.who}"
    out.mkdir(parents=True, exist_ok=True)
    if a.who not in ("oracle", "random"):  # a model: the function-calling harness, with the viewer on
        return subprocess.call([sys.executable, str(ROOT / "scripts" / "agent_rattrap.py"), "--model", a.who, "--out", str(out),
                                "--live-view", str(a.port)])
    from loophole.puzzles.rattrap.agent_tools import Harness
    from loophole.puzzles.rattrap.live_view import LiveView
    from loophole.puzzles.rattrap.reward import RatTrapEpisode
    from loophole.puzzles.rattrap.spec import load_spec

    spec = load_spec()
    ep = RatTrapEpisode(spec)
    hz = Harness(spec, ep, out, False, {})
    lv = LiveView(out, a.port).attach(hz)
    print(f"live view: {lv.url}   (frames: {lv.dir})", flush=True)
    call = lambda tool, args: hz.call(tool, args)[0]  # noqa: E731
    if a.who == "oracle":
        from loophole.reference.follow_waypoints import follow, load

        follow(load(str(ROOT / "tasks" / "loophole-rattrap" / "solution" / "waypoints.jsonl")), call)
    else:
        import numpy as np

        rng, held = np.random.default_rng(a.seed), {0: None, 1: None}
        while not ep.done:
            holding = [h for h, k in held.items() if k is not None]
            if not holding:
                h = int(rng.integers(2)); k = int(rng.integers(ep.sim.n_nodes)); call("grab", {"handle": h, "node": k}); held[h] = k
            elif rng.random() < 0.1:
                h = holding[0]; call("release", {"handle": h}); held[h] = None
            else:
                d = rng.uniform(-30, 30, 3); call("move", {"handle": holding[0], "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2])})
    hz.cmds.close()
    print(f"done: {ep.actions} actions, solved={ep.solved}, score={ep.score():.3f}; viewer overhead {lv.seconds / max(1, lv.n):.2f} s per action. "
          f"Ctrl-C to stop the viewer; `make watch-last` encodes the video.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        lv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
