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

VIEWS = ("front", "side", "iso")
GEOMETRY_KEYS = ["base_length", "base_depth", "base_thickness", "frame_inner_width", "frame_inner_height", "rail_thickness",
                 "post_thickness", "post_upper_z_frac", "post_upper_length", "post_lower_z_frac", "post_lower_length",
                 "hole_radius", "hole_sides", "hole_wall", "dowel_radius", "dowel_left_x_frac", "dowel_right_x_frac",
                 "sphere_radius", "sphere_z_frac", "ring_major_radius", "ring_tube_radius", "ring_segments",
                 "cord_radius", "mouse_enabled", "mouse_radius", "cord_ball_enabled"]


def annotate(img: np.ndarray, spec, title: str, with_cord: bool) -> np.ndarray:
    """The render with the current YAML values (and derived landmarks) printed beside it."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=14)
        small = ImageFont.load_default(size=12)
    except TypeError:
        font = small = ImageFont.load_default()
    h = img.shape[0]
    panel = Image.new("RGB", (430, h), (250, 250, 247))
    d = ImageDraw.Draw(panel)
    y = 8
    d.text((10, y), title, fill=(0, 0, 0), font=font)
    y += 22
    d.text((10, y), "rattrap.yaml (metres; fractions of the inner opening)", fill=(90, 90, 90), font=small)
    y += 18
    for k in GEOMETRY_KEYS:
        d.text((10, y), f"{k}: {spec[k]}", fill=(0, 0, 0), font=small)
        y += 15
    y += 4
    d.text((10, y), "derived:", fill=(90, 90, 90), font=small)
    y += 15
    for k, v in spec.landmarks.items():
        d.text((10, y), f"{k}: {v}", fill=(40, 40, 120), font=small)
        y += 15
    if with_cord:
        sd = spec.segment_derivation
        d.text((10, y), f"cord: {sd['segments']} segments of {sd['segment_length_m'] * 1000:.2f} mm "
                        f"(binding: {sd['binding_aperture']})", fill=(40, 40, 120), font=small)
    return np.concatenate([img, np.asarray(panel)], axis=1)


def snapshots(sim: RatTrapSim, out: Path, prefix: str, title: str) -> None:
    for v in VIEWS:
        img = sim.render(v, distance=0.5, width=640, height=720)
        (out / f"{prefix}_{v}.png").write_bytes(encode_png(annotate(img, sim.spec, f"{title}: {v}", sim.built.rope is not None)))


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
    ap.add_argument("--stills-only", action="store_true", help="only the annotated renders, no videos")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(a.yaml)
    report = {"segment_derivation": spec.segment_derivation}

    for old in a.out.glob("*_top.png"):
        old.unlink()
    rigid = RatTrapSim(spec, include_cord=False)
    snapshots(rigid, a.out, "rigid", "rigid structure (no cord)")
    rigid.close()
    sim = RatTrapSim(spec)
    report["nodes"] = sim.n_nodes
    report["ball_node"] = sim.built.ball_node
    snapshots(sim, a.out, "start", "start configuration (placeholder route)")
    if a.stills_only:
        (a.out / "preview.json").write_text(json.dumps({**report, "landmarks": spec.landmarks}, indent=1, default=float))
        print(json.dumps(report, indent=1, default=float))
        return
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found (needed for the videos)")

    # 1. pull: grab a mid-cord node (on the strand between the two bars) and translate it
    P = sim.node_positions()
    lm = spec.landmarks
    # a node on one strand of the loop, out in front of the frame
    target = np.array([lm["dowel_right_x"] + 0.01, -0.08, spec["base_thickness"] + 0.0022])
    mid = int(np.argmin(np.linalg.norm(P - target, axis=1)))
    rec = Recorder(sim, a.out / "pull.mp4", label=f"pull: grab node {mid}, translate")
    rec.hold(25)
    sim.grab_node(0, mid)
    P0 = sim.node_positions()
    pulls = []
    for d in ([0.02, 0, 0], [0, 0, 0.02], [0.02, 0, 0], [0, -0.02, 0]):
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
    P = sim.node_positions()
    end = int(np.argmin(P[:, 1]))  # the loop's far end, out in front
    sim.grab_node(0, end)
    rec = Recorder(sim, a.out / "blocked.mp4", views=("iso", "top"), label=f"blocked: drive the loop's far end (node {end}) into the right rail")
    rec.hold(25)
    P = sim.node_positions()
    rail_x = spec["frame_inner_width"] / 2 + spec["rail_thickness"] / 2
    rec.text = "align with the rail"
    sim.move(0, [rail_x - P[end][0], 0, 0])
    side = np.sign(P[end][1]) or -1.0
    target_y = -side * 0.05  # the other side of the rail: impossible without passing through it
    last = None
    for _ in range(10):
        rec.text = "move into the rail"
        last = sim.move(0, [0, target_y - sim.rig.handle_pose(sim.rig.handles[0])[0][1], 0])
        if last["blocked"]:
            break
    rec.text = f"BLOCKED at fraction {last['requested_fraction_executed']}" if last["blocked"] else "NOT blocked"
    rec.hold(60)
    rec.close()
    y_node = float(sim.node_positions()[end][1])
    report["blocked"] = {"node": end, "blocked": last["blocked"], "reason": last["blocked_reason"], "fraction": last["requested_fraction_executed"],
                         "node_y_m": round(y_node, 4), "rail_face_y_m": side * spec["rail_thickness"] / 2,
                         "cord_surface_to_rail_m": round(abs(y_node) - spec["cord_radius"] - spec["rail_thickness"] / 2, 5),
                         "max_penetration_m": sim.max_penetration}
    sim.close()
    (a.out / "preview.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
