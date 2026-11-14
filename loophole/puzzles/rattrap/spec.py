"""Rat Trap spec: load the YAML, validate it, derive the rigid parts, apertures and the cord discretisation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
DEFAULT = HERE / "rattrap.yaml"


class RatTrapError(ValueError):
    pass


@dataclass
class Box:
    name: str
    pos: np.ndarray
    half: np.ndarray  # half sizes (x, y, z)
    part: str  # which named part it belongs to


@dataclass
class Cylinder:
    name: str
    pos: np.ndarray
    radius: float
    half_length: float  # along z
    part: str


@dataclass
class Aperture:
    name: str
    center: np.ndarray
    axis: np.ndarray  # unit, the passage direction
    half_sizes: tuple[float, float]  # clear half-sizes across the axis
    depth: float  # extent along the axis to clear on each side

    @property
    def radius(self) -> float:
        """Clear radius of the tightest cross-section (half the smaller window side)."""
        return float(min(self.half_sizes))


@dataclass
class RatTrapSpec:
    raw: dict
    boxes: list[Box] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)
    apertures: dict[str, Aperture] = field(default_factory=dict)
    route: np.ndarray | None = None
    cord_length: float = 0.0
    cord_segments: int = 0
    segment_derivation: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.raw[k]


def load_spec(path: str | Path | None = None, overrides: dict | None = None) -> RatTrapSpec:
    raw = yaml.safe_load(Path(path or DEFAULT).read_text())
    raw.update(overrides or {})
    s = RatTrapSpec(raw=raw)
    _rigid(s)
    _route(s)
    _derive_segments(s)
    return s


def _rigid(s: RatTrapSpec) -> None:
    r = s.raw
    T, t = r["base_thickness"], r["rail_thickness"]
    W, H = r["frame_inner_width"], r["frame_inner_height"]
    s.boxes.append(Box("base", np.array([0, 0, T / 2]), np.array([r["base_length"] / 2, r["base_depth"] / 2, T / 2]), "base"))
    z_bot = T + t / 2  # bottom rail centre
    z_top = T + t + H + t / 2
    s.boxes += [
        Box("frame_bottom", np.array([0, 0, z_bot]), np.array([W / 2 + t, t / 2, t / 2]), "frame"),
        Box("frame_top", np.array([0, 0, z_top]), np.array([W / 2 + t, t / 2, t / 2]), "frame"),
        Box("frame_left", np.array([-(W / 2 + t / 2), 0, (z_bot + z_top) / 2]), np.array([t / 2, t / 2, (z_top - z_bot) / 2 - t / 2]), "frame"),
        Box("frame_right", np.array([W / 2 + t / 2, 0, (z_bot + z_top) / 2]), np.array([t / 2, t / 2, (z_top - z_bot) / 2 - t / 2]), "frame"),
    ]
    z0 = T + t  # top face of the bottom rail
    s.z0 = z0
    bt = r["bar_thickness"]
    for bar in ("bar_upper", "bar_lower"):
        zc = z0 + r[f"{bar}_z"]
        holes = sorted(r.get(f"{bar}_holes") or [], key=lambda h: h["x"])
        _holed_bar(s, bar, zc, W, bt, holes)
    # post: from the top rail's underside down to the upper bar's top face
    ztop = z_top - t / 2
    zbot = z0 + r["bar_upper_z"] + bt / 2
    if zbot >= ztop:
        raise RatTrapError("post: upper bar is above the top rail")
    px, pr = r["post_x"], r["post_radius"]
    if r.get("post_hole"):
        hh, hw = r["post_hole_height"], r["post_hole_width"]
        if hw >= 2 * pr:
            raise RatTrapError("post_hole_width must be < 2 * post_radius")
        zh = zbot + r["post_hole_frac"] * (ztop - zbot)
        lo_top, hi_bot = zh - hh / 2, zh + hh / 2
        s.cylinders.append(Cylinder("post_lower", np.array([px, 0, (zbot + lo_top) / 2]), pr, (lo_top - zbot) / 2, "post"))
        s.cylinders.append(Cylinder("post_upper", np.array([px, 0, (hi_bot + ztop) / 2]), pr, (ztop - hi_bot) / 2, "post"))
        side = (pr - hw / 2) / 2  # half width of each side cheek (x)
        for sgn, nm in ((-1, "post_cheek_left"), (1, "post_cheek_right")):
            s.boxes.append(Box(nm, np.array([px + sgn * (hw / 2 + side), 0, zh]), np.array([side, pr * 0.8, hh / 2]), "post"))
        s.apertures["post.hole"] = Aperture("post.hole", np.array([px, 0, zh]), np.array([0.0, 1.0, 0.0]), (hw / 2, hh / 2), 2 * pr)
    else:
        s.cylinders.append(Cylinder("post", np.array([px, 0, (zbot + ztop) / 2]), pr, (ztop - zbot) / 2, "post"))
    # ring annulus (between post and ring tube), along z; windows of the frame, along y
    R, rt = r["ring_major_radius"], r["ring_tube_radius"]
    gap = (R - rt - pr) / 2
    s.apertures["ring.annulus"] = Aperture("ring.annulus", np.array([px + pr + gap, 0, z0 + r["ring_z"]]), np.array([0.0, 0.0, 1.0]),
                                           (gap, gap), 4 * rt)
    zl = z0 + r["bar_lower_z"]
    zu = z0 + r["bar_upper_z"]
    s.apertures["window.lower"] = Aperture("window.lower", np.array([-W / 4, 0, (z0 + zl - bt / 2) / 2]), np.array([0.0, 1.0, 0.0]),
                                           (W / 4, (zl - bt / 2 - z0) / 2), t)
    s.apertures["window.middle"] = Aperture("window.middle", np.array([-W / 4, 0, (zl + zu) / 2]), np.array([0.0, 1.0, 0.0]),
                                            (W / 4, (zu - zl - bt) / 2), t)


def _holed_bar(s: RatTrapSpec, bar: str, zc: float, W: float, bt: float, holes: list[dict]) -> None:
    """A bar spanning the frame with vertical square through-holes: solid boxes between holes, and front /
    back bridges beside each hole (the window is the gap between them). No meshes."""
    x_edges = [-W / 2]
    for i, h in enumerate(holes):
        a = h["size"] / 2
        if a >= bt / 2:
            raise RatTrapError(f"{bar}: hole {i} ({h['size']} m) does not fit the bar ({bt} m)")
        x_edges += [h["x"] - a, h["x"] + a]
    x_edges.append(W / 2)
    for k in range(0, len(x_edges), 2):  # solid spans
        x0, x1 = x_edges[k], x_edges[k + 1]
        if x1 - x0 > 1e-9:
            s.boxes.append(Box(f"{bar}_span{k // 2}", np.array([(x0 + x1) / 2, 0, zc]), np.array([(x1 - x0) / 2, bt / 2, bt / 2]), bar))
    for i, h in enumerate(holes):
        a = h["size"] / 2
        bridge = (bt / 2 - a) / 2  # half depth (y) of each bridge
        for sgn, nm in ((-1, "front"), (1, "back")):
            s.boxes.append(Box(f"{bar}_hole{i}_{nm}", np.array([h["x"], sgn * (a + bridge), zc]), np.array([a, bridge, bt / 2]), bar))
        s.apertures[f"{bar}.hole{i}"] = Aperture(f"{bar}.hole{i}", np.array([h["x"], 0.0, zc]), np.array([0.0, 0.0, 1.0]), (a, a), bt)


def _route(s: RatTrapSpec) -> None:
    pts = np.asarray(s.raw["cord_route"], float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise RatTrapError("cord_route: a list of at least two [x, y, z] waypoints")
    L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    want = s.raw.get("cord_length", "auto")
    if want != "auto" and abs(float(want) - L) > 0.01 * L:
        raise RatTrapError(f"cord_length {want} m does not match the route's length {L:.4f} m (author the route to the cord's length)")
    s.route, s.cord_length = pts, L


def _derive_segments(s: RatTrapSpec) -> None:
    """Segment length L such that a straight segment's chord deviation from the bent cord stays inside the
    tightest aperture: the sagitta of a chord of length L on a bend of radius R is L^2 / (8 R); require
    L^2 / (8 R) <= c, with c = (aperture radius - cord radius) the tightest clearance and R the tightest bend
    (the cord bending around that aperture's edge: R = aperture radius). So L <= sqrt(8 R c)."""
    rc = s.raw["cord_radius"]
    apert = [(a.radius, n) for n, a in s.apertures.items()]
    r_min, name = min(apert)
    c = r_min - rc
    if c <= 0:
        raise RatTrapError(f"the cord ({rc} m radius) does not fit aperture {name} (radius {r_min} m)")
    L_max = math.sqrt(8 * r_min * c)
    n_auto = max(2, math.ceil(s.cord_length / L_max))
    n = n_auto if s.raw.get("cord_segments", "auto") == "auto" else int(s.raw["cord_segments"])
    if n < n_auto:
        raise RatTrapError(f"cord_segments {n} < derived minimum {n_auto}")
    s.cord_segments = n
    s.segment_derivation = {"tightest_aperture": name, "aperture_radius_m": r_min, "cord_radius_m": rc, "clearance_m": round(c, 6),
                            "bend_radius_m": r_min, "max_segment_length_m": round(L_max, 6), "cord_length_m": round(s.cord_length, 6),
                            "segments": n, "segment_length_m": round(s.cord_length / n, 6),
                            "sagitta_at_segment_length_m": round((s.cord_length / n) ** 2 / (8 * r_min), 6)}
