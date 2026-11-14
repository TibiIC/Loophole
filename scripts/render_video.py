#!/usr/bin/env python3
"""Render an episode as a 2-up H.264 video (two cameras side by side).

    .venv/bin/python scripts/render_video.py runs/<ts>/oracle/<trial>/verifier/episode.jsonl -o docs/media/oracle.mp4
    .venv/bin/python scripts/render_video.py loophole/reference/trajectories/alpha-1.0-paper.jsonl \
        --puzzle alpha-1.0-paper --config loophole/configs/alpha_training.yaml -o docs/media/oracle.mp4

Input: a simulator episode log (header with puzzle + config, then action records) or a
trajectory file (then --puzzle / --config are needed). The actions are replayed in a fresh
simulator; frames are captured every --every physics substeps (1 ms each) during execution,
so the video shows the motion itself, at sim time x --speed. Encoding: ffmpeg libx264,
preset veryfast, yuv420p.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.config import EnvConfig, load_config  # noqa: E402
from loophole.reference.recorder import load_trajectory  # noqa: E402
from loophole.sim.env import EpisodeOver, LoopholeEnv  # noqa: E402
from loophole.sim.rig import RigError  # noqa: E402

VIEWS = {"iso": {"view": "iso"}, "front": {"view": "front"}, "top": {"view": "top"}, "side": {"view": "side"}}


def load_episode(path: Path, puzzle: str | None, config: str | None) -> tuple[str, EnvConfig, int, list[tuple[str, dict]]]:
    first = json.loads(path.open().readline())
    if first.get("type") == "header":  # simulator episode log
        steps = [(r["tool"], r["args"]) for r in map(json.loads, path.open()) if r.get("type") == "action"]
        return first["puzzle"], EnvConfig.model_validate(first["config"]), int(first.get("seed", 0)), steps
    if not puzzle:
        sys.exit("a trajectory file needs --puzzle (and usually --config)")
    return puzzle, load_config(config), 0, load_trajectory(path)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(255, 255, 255))
    d.text((4, 3), text, fill=(0, 0, 0))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--puzzle")
    ap.add_argument("--config")
    ap.add_argument("--views", default="iso,top", help="two of: " + ", ".join(VIEWS))
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--every", type=int, default=40, help="capture a frame every N substeps (1 ms each)")
    ap.add_argument("--speed", type=float, default=1.0, help="video time = sim time / speed")
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    views = a.views.split(",")
    if len(views) != 2 or not set(views) <= set(VIEWS):
        sys.exit(f"--views: two of {sorted(VIEWS)}")
    puzzle, cfg, seed, steps = load_episode(a.episode, a.puzzle, a.config)
    env = LoopholeEnv(puzzle, config=cfg, seed=seed)
    fps = 1000.0 / a.every * a.speed
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * a.width}x{a.height}",
         "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(a.out)],
        stdin=subprocess.PIPE,
    )
    state = {"i": 0, "frames": 0, "tool": "start"}

    def frame() -> None:
        tiles = []
        for v in views:
            imgs, _ = env.render_views(**VIEWS[v], width=a.width, height=a.height)
            from PIL import Image
            import io

            tiles.append(np.asarray(Image.open(io.BytesIO(imgs[0][1])).convert("RGB")))
        img = np.concatenate(tiles, axis=1)
        img = label(img, f"action {state['i']}/{env.puzzle.max_actions}  {state['tool']}  t={env.data.time:.2f}s")
        ff.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        state["frames"] += 1

    inner = env._substep

    def substep() -> None:  # capture during execution, not only between actions
        inner()
        if env.substeps % a.every == 0:
            frame()

    env._substep = substep
    env.rig.substep = substep  # the rig holds its own reference
    for _ in range(int(fps)):  # one second of the start pose
        frame()
    for tool, args in steps:
        if env.done:
            break
        state["i"], state["tool"] = env.actions_used + 1, tool
        try:
            env.act(tool, args)
        except (RigError, EpisodeOver, ValueError, TypeError):
            continue
    state["tool"] = f"done: {env.done_reason or 'stopped'}"
    for _ in range(int(2 * fps)):  # two seconds of the final pose
        frame()
    final = env.finalize()
    ff.stdin.close()
    ff.wait()
    env.close()
    print(json.dumps({"out": str(a.out), "frames": state["frames"], "fps": fps, "score": final["score"],
                      "success": final["success"], "actions": final["actions_used"]}))


if __name__ == "__main__":
    main()
