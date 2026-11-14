#!/usr/bin/env python3
"""Watch a recorded Rat Trap solve: exact step-for-step replay, cached, then played back at any speed.

    PYTHONPATH=$PWD .venv/bin/mjpython scripts/view_rattrap_solve.py [--speed 10] [loophole/reference/rattrap-solve.jsonl]
    PYTHONPATH=$PWD .venv/bin/python scripts/view_rattrap_solve.py --video runs/rattrap-solve.mp4 [--speed 20]

The first run re-simulates the recording (replay_exact: the recorded handle pose before every physics step; ~7 min
for a 100k-step solve) and caches the frames in .view_cache/; later runs start immediately.
Viewer controls: `playback_speed` slider (right panel > Control, 0.1x-100x), up / down arrows (x1.5 / /1.5),
space pauses, backspace restarts. The HUD shows sim time, the action (grab / release count) and the cord's word.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.sim import RatTrapSim  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import load_records, replay_exact  # noqa: E402
from loophole.puzzles.rattrap.topology import build_discs, read_word  # noqa: E402

CACHE = ROOT / ".view_cache"
EVERY = 20  # physics steps per recorded frame (20 ms of sim time)


def record(path: Path, spec) -> dict:
    key = hashlib.sha256(path.read_bytes() + json.dumps(spec.raw, sort_keys=True, default=str).encode()).hexdigest()[:16]
    out = CACHE / f"rattrap-{key}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=False)
        return {k: z[k] for k in z.files}
    recs = load_records(path)
    sim = RatTrapSim(spec)
    discs = build_discs(spec)
    frames = {"qpos": [], "mocap_pos": [], "mocap_quat": [], "time": [], "events": [], "word": []}
    count = {"n": 0, "events": 0, "word": read_word(sim.node_positions(), discs, True)}
    inner = sim._substep

    def substep():
        inner()
        count["n"] += 1
        if count["n"] % EVERY == 0:
            if count["n"] % (EVERY * 25) == 0:
                count["word"] = read_word(sim.node_positions(), discs, True)
            frames["qpos"].append(sim.data.qpos.copy())
            frames["mocap_pos"].append(sim.data.mocap_pos.copy())
            frames["mocap_quat"].append(sim.data.mocap_quat.copy())
            frames["time"].append(sim.data.time)
            frames["events"].append(count["events"])
            frames["word"].append(count["word"] or "(empty)")
            if count["n"] % 10000 == 0:
                print(f"\rrecording: {count['n']} steps", end="", flush=True)

    sim._substep = substep
    sim.rig.substep = substep
    grab, release = sim.grab_node, sim.release

    def grab_node(h, k):
        count["events"] += 1
        return grab(h, k)

    def rel(h):
        count["events"] += 1
        return release(h)

    sim.grab_node, sim.release = grab_node, rel
    replay_exact(recs, spec, sim)
    print(f"\nrecorded {len(frames['time'])} frames, {count['events']} grabs / releases")
    data = {k: np.array(v) for k, v in frames.items()}
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(out, **data)
    return data


def video(rec: dict, spec, out: Path, speed: float, w=640, h=480) -> None:
    from PIL import Image, ImageDraw

    sim = RatTrapSim(spec)
    m, d = sim.model, sim.data
    fps = 30.0
    T = rec["time"]
    times = np.arange(0.0, T[-1], speed / fps)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}", "-r", f"{fps}",
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for t in times:
        i = int(np.clip(np.searchsorted(T, t), 0, len(T) - 1))
        d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = rec["qpos"][i], rec["mocap_pos"][i], rec["mocap_quat"][i]
        mujoco.mj_forward(m, d)
        tiles = [sim.render("iso", distance=0.45, width=w, height=h), sim.render("front", distance=0.4, width=w, height=h)]
        im = Image.fromarray(np.concatenate(tiles, axis=1))
        ImageDraw.Draw(im).text((8, 8), f"t={T[i]:.1f}s  ({speed:g}x)  grabs/releases {int(rec['events'][i])}  word {rec['word'][i]}", fill=(0, 0, 0))
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}")


def play(rec: dict, spec, speed: float) -> None:
    import mujoco.viewer

    sim = RatTrapSim(spec)
    jn = "cord_free" if any(sim.model.joint(j).name == "cord_free" for j in range(sim.model.njnt)) else sim.model.joint(0).name
    xml = sim.built.xml.replace("</mujoco>", f'<actuator><general name="playback_speed" joint="{jn}" gear="0 0 0 0 0 0" '
                                              'ctrllimited="true" ctrlrange="0.1 100"/></actuator></mujoco>', 1)
    vm = mujoco.MjModel.from_xml_string(xml)
    vd = mujoco.MjData(vm)
    aid = vm.actuator("playback_speed").id
    vd.ctrl[aid] = speed
    state = {"paused": False, "restart": False, "scale": 1.0}

    def keys(c: int) -> None:
        if c == 32:
            state["paused"] = not state["paused"]
        elif c == 259:
            state["restart"] = True
        elif c == 265:
            state["scale"] *= 1.5
        elif c == 264:
            state["scale"] /= 1.5

    T = rec["time"]
    with mujoco.viewer.launch_passive(vm, vd, key_callback=keys) as v:
        v.cam.lookat[:] = [0.0, -0.05, 0.12]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.55, 120, -25
        t_sim, last = 0.0, time.time()
        while v.is_running():
            now = time.time()
            with v.lock():
                if state["scale"] != 1.0:
                    vd.ctrl[aid] = float(np.clip(vd.ctrl[aid] * state["scale"], 0.1, 100.0))
                    state["scale"] = 1.0
                spd = float(np.clip(vd.ctrl[aid], 0.1, 100.0))
                if state["restart"]:
                    t_sim, state["restart"] = 0.0, False
                if not state["paused"]:
                    t_sim = min(float(T[-1]), t_sim + (now - last) * spd)
                i = int(np.clip(np.searchsorted(T, t_sim), 0, len(T) - 1))
                vd.qpos[:], vd.mocap_pos[:], vd.mocap_quat[:] = rec["qpos"][i], rec["mocap_pos"][i], rec["mocap_quat"][i]
                mujoco.mj_forward(vm, vd)
            try:  # outside the lock: set_texts takes it itself
                v.set_texts((int(mujoco.mjtFontScale.mjFONTSCALE_150), int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                             "speed\nsim time\ngrabs/releases\nword\n",
                             f"{spd:.1f}x {'paused' if state['paused'] else ''}\n{T[i]:.1f} / {T[-1]:.1f} s\n{int(rec['events'][i])}\n{rec['word'][i]}\n"))
            except Exception:
                pass
            v.sync()
            last = now
            time.sleep(1 / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path, nargs="?", default=ROOT / "loophole" / "reference" / "rattrap-solve.jsonl")
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--video", type=Path, default=None, help="write an MP4 (2-up: iso + front) instead of opening the viewer")
    a = ap.parse_args()
    spec = load_spec()
    rec = record(a.recording, spec)
    if a.video:
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found")
        video(rec, spec, a.video, a.speed)
    else:
        play(rec, spec, a.speed)


if __name__ == "__main__":
    main()
