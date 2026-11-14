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


def _png(raw_rows: list[bytes], w: int, h: int, bit_depth: int, colour_type: int) -> bytes:
    raw = b"".join(b"\x00" + r for r in raw_rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth, colour_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def encode_png(rgb: np.ndarray) -> bytes:
    """Minimal dependency-free PNG encoder for HxWx3 uint8 arrays."""
    h, w, _ = rgb.shape
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return _png([rgb[y].tobytes() for y in range(h)], w, h, 8, 2)


def encode_png16(gray: np.ndarray) -> bytes:
    """16-bit greyscale PNG (big-endian samples) for HxW uint16 arrays."""
    h, w = gray.shape
    g = np.ascontiguousarray(gray, dtype=">u2")
    return _png([g[y].tobytes() for y in range(h)], w, h, 16, 0)


# segmentation labels and their display colours (label 0 = background)
SEG_COLOURS = np.array([[0, 0, 0], [150, 150, 150], [40, 110, 230], [230, 40, 40], [240, 150, 40]], np.uint8)


def depth_visual(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Depth as an 8-bit greyscale RGB image: white = near, black = far / background."""
    t = np.clip((far - depth) / max(far - near, 1e-9), 0.0, 1.0)
    t[~np.isfinite(depth)] = 0.0
    g = (t * 255).round().astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


class SceneRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int, default_lookat, default_distance: float):
        self.model = model
        self.width = width
        self.height = height
        self.default_lookat = np.asarray(default_lookat, dtype=float)
        self.default_distance = float(default_distance)
        self._renderer: mujoco.Renderer | None = None

    def _get(self, width: int | None = None, height: int | None = None) -> mujoco.Renderer:
        size = (height or self.height, width or self.width)
        if self._renderer is None or (self._renderer.height, self._renderer.width) != size:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, *size)
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
        cam.distance = float(np.clip(self.default_distance if distance is None else distance, 0.05, 20.0))
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

    def render_modalities(self, data: mujoco.MjData, cam: mujoco.MjvCamera, modalities: list[str],
                          width: int, height: int) -> tuple[dict[str, np.ndarray], dict]:
        """Render any of rgb (HxWx3 uint8), depth (HxW float32, metres along the optical axis)
        and segmentation (HxW int32 geom ids, -1 = background); plus the camera model."""
        r = self._get(width, height)
        r.update_scene(data, cam, scene_option=mujoco.MjvOption())
        out = {}
        if "rgb" in modalities:
            out["rgb"] = r.render().copy()
        if "depth" in modalities:
            r.enable_depth_rendering()
            try:
                out["depth"] = r.render().copy()
            finally:
                r.disable_depth_rendering()
        if "segmentation" in modalities:
            r.enable_segmentation_rendering()
            try:
                seg = r.render().copy()
            finally:
                r.disable_segmentation_rendering()
            ids = seg[..., 0].astype(np.int32)
            ids[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
            out["segmentation"] = ids
        glc = r.scene.camera
        pos = (np.asarray(glc[0].pos) + np.asarray(glc[1].pos)) / 2
        fwd = np.asarray(glc[0].forward, float)
        up = np.asarray(glc[0].up, float)
        right = np.cross(fwd, up)
        fovy = float(self.model.vis.global_.fovy)
        f = (height / 2) / np.tan(np.radians(fovy) / 2)
        camera = {
            "position": pos.round(5).tolist(),
            "forward": fwd.round(6).tolist(),
            "up": up.round(6).tolist(),
            "right": right.round(6).tolist(),
            "fovy_deg": fovy,
            "width": width,
            "height": height,
            "intrinsics": {"fx": round(float(f), 4), "fy": round(float(f), 4), "cx": width / 2, "cy": height / 2},
            "znear": float(self.model.vis.map.znear * self.model.stat.extent),
            "zfar": float(self.model.vis.map.zfar * self.model.stat.extent),
            "pixel_to_ray": "pixel (u, v) = (column, row) from the top-left, centre at +0.5: direction = forward + (u + 0.5 - cx)/fx * right - (v + 0.5 - cy)/fy * up; "
                            "point = position + depth * direction",
        }
        return out, camera

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
