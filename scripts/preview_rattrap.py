#!/usr/bin/env python3
"""Look at the Rat Trap scaffold and poke its physics.

    .venv/bin/python scripts/preview_rattrap.py [--out runs/rattrap-preview] [--yaml loophole/puzzles/rattrap/rattrap.yaml]

Writes:
  rigid_{front,side,top,iso}.png     the rigid structure alone (no cord)
  start_{front,side,top,iso}.png     the full assembly in its start configuration
  pull.mp4                           grab a mid-cord node, translate it, the rest of the cord follows
  blocked.mp4                        drive a node straight into a bar: the move halts instead of passing through
  preview.json                       segment derivation, node count, and the outcome of both scripted moves
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

from loophole.puzzles.rattrap.sim import RatTrapSim  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.sim.render import encode_png  # noqa: E402

VIEWS = ("front", "side", "top", "iso")


def snapshots(sim: RatTrapSim, out: Path, prefix: str) -> None:
    for v in VIEWS:
        (out / f"{prefix}_{v}.png").write_bytes(encode_png(sim.render(v, distance=0.6)))


class Recorder:
    """Pipe frames to ffmpeg (H.264, veryfast) while the physics runs: two views side by side."""

    def __init__(self, sim: RatTrapSim, path: Path, every: int = 20, views=("iso", "front"), w=480, h=360, label=""):
        self.sim, self.every, self.views, self.w, self.h, self.label = sim, every, views, w, h, label
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}",
                                    "-r", "50", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
                                   stdin=subprocess.PIPE)
        self.inner = sim._substep
        self.n = 0
        sim._substep = self._substep
        sim.rig.substep = self._substep
        self.text = ""

    def frame(self):
        tiles = [self.sim.render(v, distance=0.45 if v == "front" else 0.55, width=self.w, height=self.h) for v in self.views]
        img = np.concatenate(tiles, axis=1)
        try:
            from PIL import Image, ImageDraw

            im = Image.fromarray(img)
            ImageDraw.Draw(im).text((6, 6), f"{self.label}  {self.text}", fill=(0, 0, 0))
            img = np.asarray(im)
        except ImportError:
            pass
        self.ff.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    def _substep(self):
        self.inner()
        self.n += 1
        if self.n % self.every == 0:
            self.frame()

    def hold(self, frames: int):
        for _ in range(frames):
            self.frame()

    def close(self):
        self.sim._substep = self.inner
        self.sim.rig.substep = self.inner
        self.ff.stdin.close()
        self.ff.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rattrap-preview")
    ap.add_argument("--yaml", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(a.yaml)
    report = {"segment_derivation": spec.segment_derivation}

    rigid = RatTrapSim(spec, include_cord=False)
    snapshots(rigid, a.out, "rigid")
    rigid.close()
    sim = RatTrapSim(spec)
    report["nodes"] = sim.n_nodes
    report["ball_node"] = sim.built.ball_node
    snapshots(sim, a.out, "start")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found (needed for the videos)")

    # 1. pull: grab a mid-cord node (on the strand between the two bars) and translate it
    P = sim.node_positions()
    zmid = spec.z0 + (spec["bar_upper_z"] + spec["bar_lower_z"]) / 2
    inside = np.abs(P[:, 1]) < 0.005  # the strand inside the frame plane, between the two bars
    mid = int(np.argmin(np.where(inside, np.abs(P[:, 2] - zmid), np.inf)))
    rec = Recorder(sim, a.out / "pull.mp4", label=f"pull: grab node {mid}, translate")
    rec.hold(25)
    sim.grab_node(0, mid)
    P0 = sim.node_positions()
    pulls = []
    for d in ([0, -0.03, 0], [0, -0.03, 0], [-0.03, 0, 0], [0, 0, -0.02]):
        rec.text = f"move {d}"
        r = sim.move(0, d)
        pulls.append({"move": d, "blocked": r["blocked"], "fraction": r["requested_fraction_executed"]})
    rec.hold(40)
    rec.close()
    P1 = sim.node_positions()
    seg0 = np.array(sim.built.rope.seg_lengths)
    report["pull"] = {"node": mid, "moves": pulls, "node_displacement_m": round(float(np.linalg.norm(P1[mid] - P0[mid])), 4),
                      "neighbours_moved_m": {k: round(float(np.linalg.norm(P1[k] - P0[k])), 4) for k in (mid - 10, mid - 3, mid + 3, mid + 10) if 0 <= k < sim.n_nodes},
                      "max_segment_stretch_m": float(np.max(np.linalg.norm(np.diff(P1, axis=0), axis=1) - seg0)),
                      "max_penetration_m": sim.max_penetration}
    sim.close()

    # 2. blocked: drive the free end (slack: nothing else resists it) straight back into the right vertical rail
    sim = RatTrapSim(spec)
    end = sim.n_nodes - 1
    sim.grab_node(0, end)
    rec = Recorder(sim, a.out / "blocked.mp4", views=("iso", "top"), label=f"blocked: drive the free end (node {end}) into the right rail")
    rec.hold(25)
    P = sim.node_positions()
    rail_x = spec["frame_inner_width"] / 2 + spec["rail_thickness"] / 2
    rec.text = "align with the rail"
    sim.move(0, [rail_x - P[end][0], 0, 0])
    target_y = -0.05  # behind the rail: impossible without passing through it
    last = None
    for _ in range(10):
        rec.text = "move -y into the rail"
        last = sim.move(0, [0, target_y - sim.rig.handle_pose(sim.rig.handles[0])[0][1], 0])
        if last["blocked"]:
            break
    rec.text = f"BLOCKED at fraction {last['requested_fraction_executed']}" if last["blocked"] else "NOT blocked"
    rec.hold(60)
    rec.close()
    y_node = float(sim.node_positions()[end][1])
    report["blocked"] = {"node": end, "blocked": last["blocked"], "reason": last["blocked_reason"], "fraction": last["requested_fraction_executed"],
                         "node_y_m": round(y_node, 4), "rail_front_face_y_m": spec["rail_thickness"] / 2,
                         "cord_surface_to_rail_m": round(y_node - spec["cord_radius"] - spec["rail_thickness"] / 2, 5),
                         "max_penetration_m": sim.max_penetration}
    sim.close()
    (a.out / "preview.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
