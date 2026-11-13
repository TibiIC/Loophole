"""Offscreen rendering (MUJOCO_GL=osmesa in Docker; egl optional; native locally)."""

from __future__ import annotations

import os
import struct
import zlib

import mujoco
import numpy as np

# MuJoCo picks its GL backend from MUJOCO_GL at import of the rendering module.
# PyOpenGL needs PYOPENGL_PLATFORM to match for osmesa / egl.
if os.environ.get("MUJOCO_GL") in ("osmesa", "egl") and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]

PRESETS = {
    "front": {"azimuth": 90.0, "elevation": -10.0},
    "back": {"azimuth": -90.0, "elevation": -10.0},
    "side": {"azimuth": 0.0, "elevation": -10.0},
    "top": {"azimuth": 90.0, "elevation": -89.0},
    "iso": {"azimuth": 135.0, "elevation": -25.0},
}


def encode_png(rgb: np.ndarray) -> bytes:
    """Minimal dependency-free PNG encoder for HxWx3 uint8 arrays."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


class SceneRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int, default_lookat, default_distance: float):
        self.model = model
        self.width = width
        self.height = height
        self.default_lookat = np.asarray(default_lookat, dtype=float)
        self.default_distance = float(default_distance)
        self._renderer: mujoco.Renderer | None = None

    def _get(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        return self._renderer

    def camera(self, view: str | None, azimuth=None, elevation=None, distance=None, lookat=None) -> mujoco.MjvCamera:
        view = view or "iso"
        if view not in PRESETS:
            raise ValueError(f"unknown view {view!r}; presets: {sorted(PRESETS)}")
        p = PRESETS[view]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = float(p["azimuth"] if azimuth is None else azimuth)
        cam.elevation = float(np.clip(p["elevation"] if elevation is None else elevation, -89.9, 89.9))
        cam.distance = float(np.clip(self.default_distance if distance is None else distance, 0.05, 5.0))
        la = self.default_lookat if lookat is None else np.asarray(lookat, dtype=float)
        if la.shape != (3,) or not np.all(np.isfinite(la)):
            raise ValueError("lookat must be [x, y, z]")
        cam.lookat[:] = la
        return cam

    def render(self, data: mujoco.MjData, cam: mujoco.MjvCamera) -> bytes:
        r = self._get()
        opt = mujoco.MjvOption()
        r.update_scene(data, cam, scene_option=opt)
        return encode_png(r.render())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
