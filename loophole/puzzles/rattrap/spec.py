"""Rat Trap spec: load the YAML, validate it, derive the rigid parts, apertures and the cord discretisation.

Topology (from the physical puzzle):
* frame (top, bottom, left, right rails) on a base plank;
* upper post: cantilevered from the right rail at 2/3 height, ~2/3 of the width long, with a vertical hole
  near its free end;
* lower post: cantilevered from the left rail at 1/3 height, with a vertical hole near its free end;
* left dowel: hangs from the top rail at 1/3 width from the left, passes through the upper post's hole
  without touching it, ends in a sphere at mid height;
* right dowel: rises from the bottom rail at 1/3 width from the right, passes through the lower post's hole
  without touching it, ends in a ring (axis left-right) at 2/3 height; the upper post passes through the
  ring without touching it.
The holes are regular hexagons (hole_sides) of thin wall boxes; their clearances are measured at the
inscribed radius (the narrowest point), so they are conservative. All rigid parts are static. The string
passes through each hole's annulus, between the sphere and each post, and inside the ring around the upper post.
"""

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
    half: np.ndarray  # half sizes (x, y, z) in the box's own frame
    part: str
    yaw: float = 0.0  # rotation about z (rad)


@dataclass
class Cylinder:
    name: str
    pos: np.ndarray
    radius: float
    half_length: float  # along z
    part: str


@dataclass
class Sphere:
    name: str
    pos: np.ndarray
    radius: float
    part: str


@dataclass
class Capsule:
    name: str
    p0: np.ndarray
    p1: np.ndarray
    radius: float
    part: str


@dataclass
class Aperture:
    name: str
    center: np.ndarray
    axis: np.ndarray  # unit, the passage direction
    half_sizes: tuple[float, float]  # clear half-sizes across the axis
    depth: float  # extent along the axis to clear
    bend_radius: float | None = None  # tightest radius the cord can bend around here (default: half-width)

    @property
    def radius(self) -> float:
        """Clear half-width of the passage."""
        return float(min(self.half_sizes))


@dataclass
class RatTrapSpec:
    raw: dict
    boxes: list[Box] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)
    spheres: list[Sphere] = field(default_factory=list)
    capsules: list[Capsule] = field(default_factory=list)
    apertures: dict[str, Aperture] = field(default_factory=dict)
    landmarks: dict = field(default_factory=dict)
    route: np.ndarray | None = None
    cord_length: float = 0.0
    cord_segments: int = 0
    segment_derivation: dict = field(default_factory=dict)
    z0: float = 0.0

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


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RatTrapError(msg)


def _rigid(s: RatTrapSpec) -> None:
    r = s.raw
    T, t = r["base_thickness"], r["rail_thickness"]
    W, H = r["frame_inner_width"], r["frame_inner_height"]
    s.boxes.append(Box("base", np.array([0, 0, T / 2]), np.array([r["base_length"] / 2, r["base_depth"] / 2, T / 2]), "base"))
    z0 = s.z0 = T + t  # top face of the bottom rail = floor of the opening
    ztop = z0 + H  # underside of the top rail
    s.boxes += [
        Box("rail_bottom", np.array([0, 0, T + t / 2]), np.array([W / 2 + t, t / 2, t / 2]), "frame"),
        Box("rail_top", np.array([0, 0, ztop + t / 2]), np.array([W / 2 + t, t / 2, t / 2]), "frame"),
        Box("rail_left", np.array([-(W / 2 + t / 2), 0, z0 + H / 2]), np.array([t / 2, t / 2, H / 2]), "frame"),
        Box("rail_right", np.array([W / 2 + t / 2, 0, z0 + H / 2]), np.array([t / 2, t / 2, H / 2]), "frame"),
    ]
    ps = r["post_thickness"] / 2  # half side of the posts' square cross-section
    rh, rd, rw, K = r["hole_radius"], r["dowel_radius"], r["hole_wall"], int(r["hole_sides"])
    xL = -W / 2 + r["dowel_left_x_frac"] * W
    xR = W / 2 - r["dowel_right_x_frac"] * W
    z_up = z0 + r["post_upper_z_frac"] * H
    z_lo = z0 + r["post_lower_z_frac"] * H
    _need(K >= 3, "hole_sides must be >= 3")
    _need(rd < rh, "dowel_radius must be smaller than hole_radius (the dowel must not touch its hole)")
    _need((rh + rw) / math.cos(math.pi / K) <= ps * math.sqrt(2) and rh + rw < ps,
          "hole_radius + hole_wall must fit inside the post's square cross-section")

    # upper post: from the right rail leftwards, hole around the left dowel
    x_end = W / 2 - r["post_upper_length"]
    _need(x_end <= xL - ps, "post_upper_length too short: the upper post must extend past the left dowel's hole")
    _need(x_end > -W / 2, "the upper post must not reach the left rail")
    _post_with_hole(s, "post_upper", x_end, W / 2, z_up, ps, xL, rh, rw, K)
    # lower post: from the left rail rightwards, hole around the right dowel
    x_end = -W / 2 + r["post_lower_length"]
    _need(x_end >= xR + ps, "post_lower_length too short: the lower post must extend past the right dowel's hole")
    _need(x_end < W / 2, "the lower post must not reach the right rail")
    _post_with_hole(s, "post_lower", -W / 2, x_end, z_lo, ps, xR, rh, rw, K)

    # left dowel + sphere
    rs = r["sphere_radius"]
    zs = z0 + r["sphere_z_frac"] * H
    _need(zs + rs < z_up - ps, "the sphere touches the upper post")
    _need(zs - rs > z_lo + ps, "the sphere touches the lower post")
    s.cylinders.append(Cylinder("dowel_left", np.array([xL, 0, (zs + ztop) / 2]), rd, (ztop - zs) / 2, "dowel_left"))
    s.spheres.append(Sphere("sphere", np.array([xL, 0, zs]), rs, "dowel_left"))

    # right dowel + ring (axis x, centre on the upper post's axis above the right dowel)
    R, rt, n = r["ring_major_radius"], r["ring_tube_radius"], int(r["ring_segments"])
    _need(R - rt > ps * math.sqrt(2), "the upper post (half-diagonal) touches the ring: ring_major_radius - ring_tube_radius too small")
    _need(z_up + R + rt < ztop, "the ring touches the top rail")
    zr_bottom = z_up - R  # where the dowel meets the ring's tube centre-line
    _need(zr_bottom - rt > z_lo + ps, "the ring touches the lower post")
    s.cylinders.append(Cylinder("dowel_right", np.array([xR, 0, (z0 + zr_bottom) / 2]), rd, (zr_bottom - z0) / 2, "dowel_right"))
    c = np.array([xR, 0.0, z_up])
    th = np.linspace(0, 2 * np.pi, n + 1)
    for i in range(n):
        p0 = c + R * np.array([0.0, np.cos(th[i]), np.sin(th[i])])
        p1 = c + R * np.array([0.0, np.cos(th[i + 1]), np.sin(th[i + 1])])
        s.capsules.append(Capsule(f"ring_c{i}", p0, p1, rt, "dowel_right"))

    # apertures (for the clearance table and the segment derivation)
    rc = r["cord_radius"]
    ann = (rh - rd) / 2  # annulus half-width at the hexagon's inscribed radius (narrowest)
    for nm, x, z in (("post_upper.hole", xL, z_up), ("post_lower.hole", xR, z_lo)):
        s.apertures[nm] = Aperture(nm, np.array([x + rd + ann, 0.0, z]), np.array([0.0, 0.0, 1.0]), (ann, ann), 2 * ps, bend_radius=rd + rc)
    gap = (R - rt - ps) / 2
    s.apertures["ring.inside"] = Aperture("ring.inside", np.array([xR, 0.0, z_up + ps + gap]), np.array([1.0, 0.0, 0.0]), (gap, gap), 2 * rt)
    side = rd + 0.003  # beside the dowel
    gu = ((z_up - ps) - (zs + rs)) / 2
    gl = ((zs - rs) - (z_lo + ps)) / 2
    s.apertures["gap.sphere_upper"] = Aperture("gap.sphere_upper", np.array([xL + side, 0.0, zs + rs + gu]), np.array([0.0, 1.0, 0.0]), (gu, gu), 2 * ps)
    s.apertures["gap.sphere_lower"] = Aperture("gap.sphere_lower", np.array([xL + side, 0.0, z_lo + ps + gl]), np.array([0.0, 1.0, 0.0]), (gl, gl), 2 * ps)
    gx_u = ((W / 2 - r["post_upper_length"]) - (-W / 2)) / 2
    gx_l = (W / 2 - (-W / 2 + r["post_lower_length"])) / 2
    s.apertures["gap.upper_left"] = Aperture("gap.upper_left", np.array([-W / 2 + gx_u, 0.0, z_up]), np.array([0.0, 0.0, 1.0]), (gx_u, t / 2), 2 * ps)
    s.apertures["gap.lower_right"] = Aperture("gap.lower_right", np.array([W / 2 - gx_l, 0.0, z_lo]), np.array([0.0, 0.0, 1.0]), (gx_l, t / 2), 2 * ps)
    s.landmarks = {"inner_floor_z": round(z0, 4), "inner_ceiling_z": round(ztop, 4), "dowel_left_x": round(xL, 4), "dowel_right_x": round(xR, 4),
                   "post_upper_z": round(z_up, 4), "post_lower_z": round(z_lo, 4), "sphere_center_z": round(zs, 4),
                   "ring_center": [round(x, 4) for x in c], "hole_annulus_mid_offset": round(rd + ann, 4)}


def _post_with_hole(s: RatTrapSpec, part: str, x0: float, x1: float, z: float, ps: float, xh: float, rh: float, rw: float, K: int) -> None:
    """A square post from x0 to x1 with a vertical hole at xh: solid boxes either side of the hole block; the hole
    block = a regular K-gon (hexagon by default) of thin wall boxes tangent to the hole (inscribed radius rh),
    plus four side boxes closing it to the post's square outline. Enclosed voids between them are unreachable."""
    for nm, a, b in (("a", x0, xh - ps), ("b", xh + ps, x1)):
        if b - a > 1e-9:
            s.boxes.append(Box(f"{part}_solid_{nm}", np.array([(a + b) / 2, 0, z]), np.array([(b - a) / 2, ps, ps]), part))
    ro = rh + rw  # outer apothem of the polygonal tube
    for sgn in (-1, 1):  # side boxes: x-sides span the full depth, y-sides span between them
        s.boxes.append(Box(f"{part}_side_x{sgn:+d}", np.array([xh + sgn * (ro + ps) / 2, 0, z]), np.array([(ps - ro) / 2, ps, ps]), part))
        s.boxes.append(Box(f"{part}_side_y{sgn:+d}", np.array([xh, sgn * (ro + ps) / 2, z]), np.array([ro, (ps - ro) / 2, ps]), part))
    half_side = ro * math.tan(math.pi / K)  # covers each polygon side out to the outer apothem (closes the corners)
    for k in range(K):
        a = 2 * math.pi * k / K
        c = np.array([xh + (rh + rw / 2) * math.cos(a), (rh + rw / 2) * math.sin(a), z])
        s.boxes.append(Box(f"{part}_hole_w{k}", c, np.array([rw / 2, half_side, ps]), part, yaw=a))


def _route(s: RatTrapSpec) -> None:
    pts = np.asarray(s.raw["cord_route"], float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise RatTrapError("cord_route: a list of at least two [x, y, z] waypoints")
    if s.raw.get("cord_closed"):
        if s.raw.get("cord_anchor_start"):
            raise RatTrapError("a closed cord cannot be anchored (cord_anchor_start must be false)")
        if np.linalg.norm(pts[-1] - pts[0]) > 1e-9:
            pts = np.vstack([pts, pts[:1]])  # close the loop
    L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    want = s.raw.get("cord_length", "auto")
    if want != "auto" and abs(float(want) - L) > 0.01 * L:
        raise RatTrapError(f"cord_length {want} m does not match the route's length {L:.4f} m (author the route to the cord's length)")
    s.route, s.cord_length = pts, L


def _derive_segments(s: RatTrapSpec) -> None:
    """Segment length L such that a straight segment's chord deviation from the bent cord stays inside every
    aperture: the sagitta of a chord of length L on a bend of radius R is L^2 / (8 R); require
    L^2 / (8 R) <= c, with c = (aperture half-width - cord radius) the clearance and R the tightest bend there
    (around a dowel: dowel radius + cord radius). So L <= sqrt(8 R c); the binding aperture gives the minimum."""
    rc = s.raw["cord_radius"]
    best = None
    for n, a in s.apertures.items():
        c = a.radius - rc
        if c <= 0:
            raise RatTrapError(f"the cord ({rc} m radius) does not fit aperture {n} (half-width {a.radius} m)")
        R = a.bend_radius if a.bend_radius is not None else a.radius
        L = math.sqrt(8 * R * c)
        if best is None or L < best[0]:
            best = (L, n, a.radius, c, R)
    L_max, name, r_min, c, R = best
    n_auto = max(2, math.ceil(s.cord_length / L_max))
    n = n_auto if s.raw.get("cord_segments", "auto") == "auto" else int(s.raw["cord_segments"])
    if n < n_auto:
        raise RatTrapError(f"cord_segments {n} < derived minimum {n_auto}")
    s.cord_segments = n
    s.segment_derivation = {"binding_aperture": name, "aperture_half_width_m": round(r_min, 6), "cord_radius_m": rc,
                            "clearance_m": round(c, 6), "bend_radius_m": round(R, 6), "max_segment_length_m": round(L_max, 6),
                            "cord_length_m": round(s.cord_length, 6), "segments": n, "segment_length_m": round(s.cord_length / n, 6),
                            "sagitta_at_segment_length_m": round((s.cord_length / n) ** 2 / (8 * R), 6)}
