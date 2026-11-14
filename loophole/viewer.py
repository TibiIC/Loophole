"""Local dev viewer (outside Docker): watch scripted episodes or replay episode logs.

    mjpython -m loophole.viewer --trajectory loophole/reference/trajectories/easy.jsonl
    mjpython -m loophole.viewer --episode runs/episode.jsonl        # an MCP/Harbor episode log
    mjpython -m loophole.viewer --puzzle easy                        # just look at a puzzle

Episodes are simulated once into a recording (cached in .view_cache/), then played back,
so playback speed is not limited by the physics. Controls during playback:
  * right panel > Control > `playback_speed`: slider, 0.1x to 50x
  * up / down arrows: speed x1.5 / /1.5
  * space: pause / resume;  backspace: restart
`--live` instead steps the physics in the viewer (about real time at best for the Alpha
meshes; `--speed` only slows it down).

On macOS the passive viewer needs `mjpython` (ships with the mujoco wheel);
on Linux plain `python` works.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .config import load_config
from .reference.recorder import load_trajectory
from .sim.env import PHYSICS_TOOLS, LoopholeEnv
from .sim.rig import RigError


def view_replay(env: LoopholeEnv, steps: list[tuple[str, dict]], speed: float = 1.0, sync_every: int = 8) -> None:
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        lo, hi = env.frame_aabb()
        v.cam.lookat[:] = (lo + hi) / 2
        v.cam.distance = max(1.0, 1.5 * float(np.max(hi - lo)))  # fit the scene (Alpha tubes are ~2.5 m)
        v.cam.azimuth, v.cam.elevation = 135, -20
        inner = env._substep
        counter = {"n": 0}
        dt = env.model.opt.timestep

        def substep():
            inner()
            counter["n"] += 1
            if counter["n"] % sync_every == 0:
                v.sync()
                time.sleep(sync_every * dt / speed)

        env._substep = substep
        env.rig.substep = substep
        time.sleep(1.0)
        for tool, args in steps:
            if not v.is_running():
                return
            if tool not in PHYSICS_TOOLS or env.done:
                continue
            try:
                r = env.act(tool, dict(args))
                print(tool, args, "->", {k: r.get(k) for k in ("action", "blocked", "done", "success")})
            except (RigError, ValueError) as e:
                print(tool, args, "-> rejected:", e)
        print("final:", {k: env.finalize()[k] for k in ("score", "success", "actions_used")})
        while v.is_running():
            v.sync()
            time.sleep(0.05)


CACHE = Path(__file__).resolve().parents[1] / ".view_cache"
RECORD_EVERY = 10  # substeps (10 ms of sim time) per recorded frame


def record(env: LoopholeEnv, steps: list[tuple[str, dict]], key: str) -> dict:
    """Simulate the episode once and keep the kinematic state every RECORD_EVERY substeps."""
    path = CACHE / f"{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    m, d = env.model, env.data
    qpos, mpos, mquat, t, act, labels = [], [], [], [], [], []
    cur = {"i": 0, "label": "start"}

    def grab_frame():
        qpos.append(d.qpos.copy())
        mpos.append(d.mocap_pos.copy())
        mquat.append(d.mocap_quat.copy())
        t.append(d.time)
        act.append(cur["i"])
        labels.append(cur["label"])

    inner = env._substep

    def substep():
        inner()
        if env.substeps % RECORD_EVERY == 0:
            grab_frame()

    env._substep = substep
    env.rig.substep = substep
    grab_frame()
    t0 = time.time()
    for k, (tool, args) in enumerate(steps):
        if tool not in PHYSICS_TOOLS or env.done:
            continue
        cur["i"], cur["label"] = env.actions_used + 1, tool
        try:
            env.act(tool, dict(args))
        except (RigError, ValueError):
            pass
        print(f"\rrecording: action {env.actions_used}/{len(steps)}  ({time.time() - t0:.0f} s)", end="", flush=True)
    f = env.finalize()
    cur["label"] = f"done: {f['done_reason']}  score {f['score']:.3f}"
    grab_frame()
    print(f"\nrecorded {len(t)} frames; final: score {f['score']:.3f}, success {f['success']}, actions {f['actions_used']}")
    rec = {"qpos": np.array(qpos), "mocap_pos": np.array(mpos), "mocap_quat": np.array(mquat), "time": np.array(t),
           "action": np.array(act), "label": np.array(labels), "max_actions": np.array(env.puzzle.max_actions)}
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **rec)
    return rec


def _viewing_model(env: LoopholeEnv) -> tuple[mujoco.MjModel, int]:
    """The episode's model plus a dummy `playback_speed` actuator: the viewer's Control panel
    shows it as a slider. Zero gear and the model is never stepped, so it changes nothing."""
    m = env.model
    jnt = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    xml = env.built.xml.replace("</mujoco>", f'<actuator><general name="playback_speed" joint="{m.joint(jnt).name}" '
                                'gear="0 0 0 0 0 0" ctrllimited="true" ctrlrange="0.1 50"/></actuator></mujoco>', 1)
    vm = mujoco.MjModel.from_xml_string(xml)
    return vm, vm.actuator("playback_speed").id


def play(env: LoopholeEnv, rec: dict, speed: float = 1.0) -> None:
    vm, act_id = _viewing_model(env)
    vd = mujoco.MjData(vm)
    vd.ctrl[act_id] = speed
    state = {"paused": False, "restart": False}

    def keys(code: int) -> None:
        if code == 32:  # space
            state["paused"] = not state["paused"]
        elif code == 259:  # backspace
            state["restart"] = True
        elif code == 265:  # up
            vd.ctrl[act_id] = min(50.0, vd.ctrl[act_id] * 1.5)
        elif code == 264:  # down
            vd.ctrl[act_id] = max(0.1, vd.ctrl[act_id] / 1.5)

    T = rec["time"]
    with mujoco.viewer.launch_passive(vm, vd, key_callback=keys) as v:
        lo, hi = env.frame_aabb()
        v.cam.lookat[:] = (lo + hi) / 2
        v.cam.distance = max(1.0, 1.5 * float(np.max(hi - lo)))
        v.cam.azimuth, v.cam.elevation = 135, -20
        sim_t, last, i = 0.0, time.time(), 0
        while v.is_running():
            now = time.time()
            with v.lock():
                spd = float(vd.ctrl[act_id])
                if state["restart"]:
                    sim_t, state["restart"] = 0.0, False
                if not state["paused"]:
                    sim_t = min(float(T[-1]), sim_t + (now - last) * spd)
                i = int(np.clip(np.searchsorted(T, sim_t), 0, len(T) - 1))
                vd.qpos[:] = rec["qpos"][i]
                vd.mocap_pos[:] = rec["mocap_pos"][i]
                vd.mocap_quat[:] = rec["mocap_quat"][i]
                mujoco.mj_forward(vm, vd)
                status = "paused" if state["paused"] else ("end" if sim_t >= T[-1] else "")
                try:
                    v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                 "speed\naction\nsim time\n",
                                 f"{spd:.1f}x {status}\n{int(rec['action'][i])}/{int(rec['max_actions'])}  {rec['label'][i]}\n"
                                 f"{T[i]:.2f} s / {T[-1]:.1f} s\n"))
                except Exception:
                    pass
            v.sync()
            last = now
            time.sleep(1 / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--episode", default=None, help="episode.jsonl written by the MCP server / Harbor")
    ap.add_argument("--speed", type=float, default=1.0, help="initial playback speed (0.1 to 50)")
    ap.add_argument("--live", action="store_true", help="step the physics in the viewer instead of playing a recording")
    ap.add_argument("--config", default=None, help="sim config (default: the episode log's, else the default config)")
    a = ap.parse_args()
    steps: list[tuple[str, dict]] = []
    puzzle, seed, cfg = a.puzzle, 0, load_config(a.config)
    if a.episode:
        recs = [json.loads(l) for l in open(a.episode) if l.strip()]
        hdr = recs[0]
        puzzle = puzzle or hdr["puzzle"]
        seed = hdr.get("seed", 0)
        cfg = cfg.model_validate(hdr["config"])
        steps = [(r["tool"], r["args"]) for r in recs if r.get("type") == "action"]
    elif a.trajectory:
        steps = load_trajectory(a.trajectory)
        if puzzle is None:
            puzzle = json.loads(open(a.trajectory).readline()).get("puzzle")
    env = LoopholeEnv(puzzle or "easy", config=cfg, seed=seed)
    if a.live or not steps:
        view_replay(env, steps, speed=a.speed)
        return
    key = hashlib.sha256(json.dumps([env.puzzle.source_sha256, cfg.model_dump(), seed, steps, RECORD_EVERY],
                                    sort_keys=True, default=str).encode()).hexdigest()[:16]
    play(env, record(env, steps, key), speed=float(np.clip(a.speed, 0.1, 50.0)))


if __name__ == "__main__":
    main()
