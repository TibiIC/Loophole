#!/usr/bin/env python3
"""Drive the Rat Trap cord by hand, through the agent's rig, and record a solve.

    .venv/bin/mjpython scripts/teleop_rattrap.py [--out loophole/reference/rattrap-solve.jsonl]

Controls (the viewer's native camera controls are unchanged: left-drag orbits, right-drag pans, scroll zooms):
  double-click a cord segment   grab the nearest node with the ACTIVE handle (re-grabs if it holds something)
  Ctrl + right-drag             drag the active handle's target in the camera plane (MuJoCo's native drag)
  Ctrl + Shift + right-drag     drag along the camera's depth axis (MuJoCo's native drag)
  arrows                        nudge the target in the camera plane (left/right, up/down), 2 mm
  PgUp / PgDn                   nudge the target along the camera's depth axis (away / towards you), 2 mm
  1 / 2                         make handle 1 / handle 2 active (the other holds still)
  [ / ]                         select the previous / next node;  G grabs the selected node with the active handle
  R                             release the active handle
  S / U                         snapshot / restore the last snapshot (undone motion is cut from the recording)
  Esc or closing the window     save: raw recording + converted commands
Every motion goes through the rig: velocity-capped per physics step, halts when blocked, penetration audited.
The HUD shows the active handle and node, whether the last motion was blocked, the reduced topological word,
and the number of discrete actions the recording converts to so far.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.sim import RatTrapSim  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import Teleop, save_commands, to_commands  # noqa: E402

NUDGE = 0.002
KEY = {"LEFT": 263, "RIGHT": 262, "UP": 265, "DOWN": 264, "PGUP": 266, "PGDN": 267, "LBR": 91, "RBR": 93,
       "1": 49, "2": 50, "G": 71, "R": 82, "S": 83, "U": 85}


def camera_axes(cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([np.sin(az), -np.cos(az), 0.0])
    up = np.cross(right, fwd)
    return right, up, fwd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "loophole" / "reference" / "rattrap-solve.jsonl")
    ap.add_argument("--yaml", type=Path, default=None)
    ap.add_argument("--steps-per-frame", type=int, default=16, help="physics steps (1 ms) per rendered frame")
    ap.add_argument("--frames", type=int, default=0, help="smoke test: close after this many frames (0 = run until closed)")
    a = ap.parse_args()
    sim = RatTrapSim(load_spec(a.yaml))
    tp = Teleop(sim)
    m, d = sim.model, sim.data
    cord_bodies = {m.body(b).id: k for k, b in enumerate(sim.built.rope.body_names)}
    pending: list = []  # key presses, applied in the main loop (the callback runs on the viewer's thread)

    def keys(code: int) -> None:
        pending.append(code)

    status = {"msg": "double-click a cord segment to grab it"}
    drag = {"active": False, "ref0": None, "t0": None}
    last_select = -1
    with mujoco.viewer.launch_passive(m, d, key_callback=keys) as v:
        v.cam.lookat[:] = [0.0, -0.05, 0.12]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.55, 120, -25
        frame_no = 0
        if a.frames:  # smoke test: grab the loop's far end and pull it a little, through the same code paths
            far = int(np.argmin(sim.node_positions()[:, 1]))
        while v.is_running() and not (a.frames and frame_no >= a.frames):
            frame_no += 1
            if a.frames and frame_no == 2:
                tp.grab(far)
            if a.frames and 3 <= frame_no <= 20:
                pending.append(KEY["RIGHT"])
            t_frame = time.time()
            with v.lock():
                right, up, fwd = camera_axes(v.cam)
                # keys
                while pending:
                    c = pending.pop(0)
                    try:
                        if c == KEY["1"]:
                            tp.set_active(0)
                        elif c == KEY["2"]:
                            tp.set_active(1)
                        elif c == KEY["LBR"]:
                            tp.selected = (tp.selected - 1) % sim.n_nodes
                        elif c == KEY["RBR"]:
                            tp.selected = (tp.selected + 1) % sim.n_nodes
                        elif c == KEY["G"]:
                            tp.grab(tp.selected)
                        elif c == KEY["R"]:
                            tp.release()
                        elif c == KEY["S"]:
                            tp.save()
                            status["msg"] = f"snapshot saved at step {tp.steps}"
                        elif c == KEY["U"]:
                            status["msg"] = "restored the snapshot" if tp.undo() else "no snapshot to restore"
                        elif c in (KEY["LEFT"], KEY["RIGHT"], KEY["UP"], KEY["DOWN"], KEY["PGUP"], KEY["PGDN"]):
                            vec = {KEY["LEFT"]: -right, KEY["RIGHT"]: right, KEY["UP"]: up, KEY["DOWN"]: -up,
                                   KEY["PGUP"]: fwd, KEY["PGDN"]: -fwd}[c]
                            tp.nudge(NUDGE * vec)
                    except Exception as e:  # rig errors (e.g. grabbing out of range) are shown, never fatal
                        status["msg"] = f"refused: {e}"
                # double-click selection -> grab the nearest node of that segment with the active handle
                sel = int(v.perturb.select)
                if sel != last_select:
                    last_select = sel
                    if sel in cord_bodies:
                        k = cord_bodies[sel]
                        local = np.asarray(v.perturb.localpos)
                        node = k + (1 if local[0] > sim.built.rope.seg_lengths[k] / 2 else 0)
                        try:
                            tp.grab(node)
                            status["msg"] = f"handle {tp.active + 1} grabbed node {node}"
                        except Exception as e:
                            status["msg"] = f"refused: {e}"
                # native Ctrl+drag moves the perturbation reference; use it as the active handle's target
                pert = int(v.perturb.active) & int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
                if pert and sel in cord_bodies and tp.rig.handles[tp.active].body is not None:
                    ref = np.asarray(v.perturb.refpos, float).copy()
                    if not drag["active"]:
                        drag.update(active=True, ref0=ref, t0=tp.handle_pos().copy())
                    tp.set_target(drag["t0"] + (ref - drag["ref0"]))
                elif drag["active"]:
                    drag["active"] = False
                # physics, only through the rig (Teleop.step also zeroes any applied forces every step)
                tp.step(a.steps_per_frame)
                if tp.steps % 64 < a.steps_per_frame:
                    tp.update_topology()
                if tp.steps % 512 < a.steps_per_frame:
                    status["actions"] = tp.estimated_actions()
                mocap = d.mocap_pos.copy()
            v.sync()
            with v.lock():  # the viewer's own drag may teleport a mocap handle: revert it, only the rig moves handles
                if not np.array_equal(d.mocap_pos, mocap):
                    d.mocap_pos[:] = mocap
            h = tp.active
            held = tp.held_node[h]
            text_l = "handle\nnode\nlast motion\nword\nactions\nsnapshot\n"
            text_r = (f"{h + 1} (holding node {held})\n" if held is not None else f"{h + 1} (empty)\n") + \
                     f"selected {tp.selected}\n" + ("BLOCKED\n" if tp.last_blocked[h] else "free\n") + \
                     f"{tp.tracker.word or '(empty)'}\n{status.get('actions', 0)}\n" + ("yes\n" if tp.snapshot else "none\n")
            try:
                v.set_texts([(int(mujoco.mjtFontScale.mjFONTSCALE_150), int(mujoco.mjtGridPos.mjGRID_TOPLEFT), text_l, text_r),
                             (int(mujoco.mjtFontScale.mjFONTSCALE_150), int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT), status["msg"], "")])
            except Exception:
                pass
            time.sleep(max(0.0, a.steps_per_frame * tp.dt - (time.time() - t_frame)))
    # save
    tp.save_recording(a.out)
    cmds = to_commands(tp.records, sim.spec)
    cmd_path = a.out.with_name(a.out.stem + ".commands.jsonl")
    final = {"word": tp.update_topology()["word"], "nodes": sim.node_positions().round(7).tolist(), "max_penetration": sim.max_penetration}
    save_commands(cmds, cmd_path, meta={"live_final": final, "steps": tp.steps})
    print(json.dumps({"recording": str(a.out), "commands": str(cmd_path), "n_commands": len(cmds), "final_word": final["word"],
                      "max_penetration_m": sim.max_penetration}))


if __name__ == "__main__":
    main()
