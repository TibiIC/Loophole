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


def load_episode(path: Path, puzzle: str | None, config: str | None):
    """(puzzle, config, seed, steps, logged action records or None, header or None)."""
    first = json.loads(path.open().readline())
    if first.get("type") == "header":  # simulator episode log
        acts = [r for r in map(json.loads, path.open()) if r.get("type") == "action"]
        return first["puzzle"], EnvConfig.model_validate(first["config"]), int(first.get("seed", 0)), [(r["tool"], r["args"]) for r in acts], acts, first
    if not puzzle:
        sys.exit("a trajectory file needs --puzzle (and usually --config)")
    return puzzle, load_config(config), 0, load_trajectory(path), None, None


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
    ap.add_argument("--sheet", type=int, default=0, help="also write a contact sheet of N keyframes (<out>.sheet.png)")
    ap.add_argument("--title", default="", help="text shown in every frame (e.g. the run name)")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    views = a.views.split(",")
    if len(views) != 2 or not set(views) <= set(VIEWS):
        sys.exit(f"--views: two of {sorted(VIEWS)}")
    puzzle, cfg, seed, steps, logged, header = load_episode(a.episode, a.puzzle, a.config)
    env = LoopholeEnv(puzzle, config=cfg, seed=seed, start_pose=(header or {}).get("start_pose"))
    check = {"initial_state_matches_log": None if header is None else env.initial_hash == header.get("initial_state_hash"),
             "max_pose_deviation_from_log_m": 0.0, "outcome_mismatches": [], "max_penetration_m": 0.0, "blocked_moves": 0,
             "min_distance_m": None}
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
        img = label(img, f"{a.title}  action {state['i']}/{env.puzzle.max_actions}  {state['tool']}  t={env.data.time:.2f}s")
        ff.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        state["frames"] += 1
        if a.sheet:
            keep.append((state["frames"], img))

    inner = env._substep

    keep: list = []

    def substep() -> None:  # capture during execution, not only between actions
        inner()
        check["max_penetration_m"] = max(check["max_penetration_m"], float(env.audit.last_penetration))
        if env.substeps % a.every == 0:
            frame()

    env._substep = substep
    env.rig.substep = substep  # the rig holds its own reference
    for _ in range(int(fps)):  # one second of the start pose
        frame()
    for k, (tool, args) in enumerate(steps):
        if env.done:
            break
        state["i"], state["tool"] = env.actions_used + 1, tool
        try:
            res = env.act(tool, args)
        except (RigError, EpisodeOver, ValueError, TypeError):
            continue
        check["blocked_moves"] += bool(res.get("blocked"))
        md = env._last_min_distance
        if md is not None:
            check["min_distance_m"] = md if check["min_distance_m"] is None else min(check["min_distance_m"], md)
        if logged is not None:  # the replay must reproduce what the simulator logged during the run
            r = logged[k]
            dev = float(np.abs(env.target_pose() - np.asarray(r["rel_pose"])).max())
            check["max_pose_deviation_from_log_m"] = max(check["max_pose_deviation_from_log_m"], dev)
            if bool(res.get("blocked")) != bool((r.get("result") or {}).get("blocked")) or env.success != r["success"] or env.audit.count != r["violations"]:
                check["outcome_mismatches"].append(k + 1)
        state["tool"] = f"{tool}{' BLOCKED' if res.get('blocked') else ''}"
    state["tool"] = f"done: {env.done_reason or 'stopped'}"
    for _ in range(int(2 * fps)):  # two seconds of the final pose
        frame()
    final = env.finalize()
    ff.stdin.close()
    ff.wait()
    env.close()
    if a.sheet and keep:
        from PIL import Image

        idx = np.linspace(0, len(keep) - 1, a.sheet).round().astype(int)
        tiles = [keep[i][1] for i in idx]
        cols = 2
        h, w = tiles[0].shape[:2]
        sheet = np.full(((len(tiles) + cols - 1) // cols * h, cols * w, 3), 255, np.uint8)
        for j, t in enumerate(tiles):
            sheet[(j // cols) * h:(j // cols + 1) * h, (j % cols) * w:(j % cols + 1) * w] = t
        Image.fromarray(sheet).save(a.out.with_suffix(".sheet.png"))
    check.update(audit_violations=final["violations"]["count"], score=final["score"], success=final["success"],
                 actions=final["actions_used"], done_reason=final["done_reason"],
                 goal=final.get("goal"), logged_final_score=None)
    if logged is not None:
        fin = [json.loads(l) for l in a.episode.open() if '"type":"final"' in l.replace(" ", "")]
        check["logged_final_score"] = fin[-1]["score"] if fin else None
    a.out.with_suffix(".check.json").write_text(json.dumps(check, indent=1, default=str))
    print(json.dumps({"out": str(a.out), "frames": state["frames"], "fps": fps, **check}, default=str))


if __name__ == "__main__":
    main()
