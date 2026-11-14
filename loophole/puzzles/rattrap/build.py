"""Rat Trap spec -> MJCF (generated, never hand-edited).

Rigid parts are primitives only (boxes, cylinders, capsules, spheres): no meshes, no convex
decomposition. Every rigid part except the ring is static (a body without joints = welded to the
world). The ring is a free body made of N capsules around a circle (never a convex mesh), captive
around the post. The cord is a chain of capsules connected by ball joints, a kinematic tree in joint
coordinates: segment lengths are exact, so it is inextensible by construction (no springs, no
breaking, no elasticity). Gravity off, friction off (condim 1), light damping on every joint.

The returned `RatTrapBuilt` carries the fields the Alpha rig (loophole.sim.rig.Rig) reads, so the
handle mechanism and its blocked-move semantics are reused unchanged.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from ...sim.builder import RopeInfo
from ...sim.geometry import basis_perp, frame_from_x, quat_conj, quat_mul, resample_arclength
from .spec import RatTrapSpec

RGBA = {"base": (0.55, 0.42, 0.30, 1), "frame": (0.62, 0.48, 0.34, 1), "bar_upper": (0.70, 0.55, 0.38, 1),
        "bar_lower": (0.70, 0.55, 0.38, 1), "post": (0.5, 0.5, 0.55, 1), "ring": (0.85, 0.7, 0.2, 1),
        "cord": (0.9, 0.35, 0.15, 1), "ball": (0.2, 0.45, 0.85, 1), "mouse": (0.6, 0.6, 0.6, 1), "probe": (0.2, 0.75, 0.3, 1)}
HANDLE_RGBA = [(0.9, 0.2, 0.2, 0.6), (0.2, 0.8, 0.3, 0.6)]


def _f(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.atleast_1d(v))


@dataclass
class RatTrapBuilt:
    xml: str
    num_handles: int
    rope: RopeInfo | None
    graspable: list[str]
    frame_bodies: list[str]
    object_bodies: list[str]
    weld_names: dict = field(default_factory=dict)
    geom_part: dict = field(default_factory=dict)  # geom name -> part name (rigid parts, ring, ball, mouse, probe)
    ball_node: int | None = None
    probe: str | None = None


def build(spec: RatTrapSpec, include_cord: bool = True, ring_free: bool = True, probe: dict | None = None,
          num_handles: int | None = None) -> RatTrapBuilt:
    """probe (clearance tests): {"kind": "cord"|"ball", "pos": [...], "axis": [...]} adds a free, graspable probe
    body: a cord-radius capsule along `axis`, or a ball-radius sphere."""
    r = spec.raw
    nh = int(num_handles if num_handles is not None else r["rig_num_handles"])
    root = ET.Element("mujoco", model="rattrap")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    # a light viscous medium damps whole-body motion (a freely swinging tail, the ring), which per-joint damping
    # cannot: joint damping only resists relative rotation between neighbouring segments
    ET.SubElement(root, "option", timestep=f"{r['timestep']}", gravity="0 0 0", integrator="implicitfast",
                  viscosity=f"{r.get('medium_viscosity', 0.0)}")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    ET.SubElement(vis, "quality", shadowsize="2048")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", condim="1", solref=_f(r["contact_solref"]), solimp=_f(r["contact_solimp"]), margin="0")
    hdef = ET.SubElement(default, "default", **{"class": "handle"})
    ET.SubElement(hdef, "geom", contype="0", conaffinity="0", group="3", type="sphere", size="0.004")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.95 0.96 0.98", rgb2="0.75 0.8 0.88", width="256", height="256")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.2 -0.6 0.8", dir="-0.2 0.6 -0.8", diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "light", pos="-0.3 0.4 0.6", dir="0.3 -0.4 -0.6", diffuse="0.3 0.3 0.3")
    geom_part: dict = {}

    # static rigid parts (bodies without joints = welded to the world)
    parts: dict[str, ET.Element] = {}

    def part_body(name: str) -> ET.Element:
        if name not in parts:
            parts[name] = ET.SubElement(world, "body", name=name)
        return parts[name]

    for b in spec.boxes:
        body = part_body(b.part)
        ET.SubElement(body, "geom", name=b.name, type="box", pos=_f(b.pos), size=_f(b.half), rgba=_f(RGBA[b.part]))
        geom_part[b.name] = b.part
    for c in spec.cylinders:
        body = part_body(c.part)
        ET.SubElement(body, "geom", name=c.name, type="cylinder", pos=_f(c.pos), size=_f([c.radius, c.half_length]), rgba=_f(RGBA[c.part]))
        geom_part[c.name] = c.part

    # ring: N capsules around a circle (axis z, around the post), a free body (captive)
    R, rt, n = r["ring_major_radius"], r["ring_tube_radius"], int(r["ring_segments"])
    rc = np.array([r["post_x"], 0.0, spec.z0 + r["ring_z"]])
    ring = ET.SubElement(world, "body", name="ring", pos=_f(rc))
    if ring_free:
        ET.SubElement(ring, "joint", name="ring_free", type="free", damping=f"{r['ring_damping']}")
    u, v = basis_perp(np.array([0.0, 0.0, 1.0]))
    th = np.linspace(0, 2 * np.pi, n + 1)
    for i in range(n):
        p0 = R * (np.cos(th[i]) * u + np.sin(th[i]) * v)
        p1 = R * (np.cos(th[i + 1]) * u + np.sin(th[i + 1]) * v)
        ET.SubElement(ring, "geom", name=f"ring_c{i}", type="capsule", fromto=_f(np.concatenate([p0, p1])), size=_f(rt),
                      mass=f"{r['ring_mass'] / n:.9g}", rgba=_f(RGBA["ring"]))
        geom_part[f"ring_c{i}"] = "ring"

    rope, graspable, ball_node = None, [], None
    contact = ET.SubElement(root, "contact")
    if include_cord:
        rope, ball_node = _cord(spec, world, contact, geom_part)
        graspable = list(rope.body_names)
    probe_name = None
    if probe is not None:
        probe_name = "probe"
        pb = ET.SubElement(world, "body", name="probe", pos=_f(probe["pos"]), quat=_f(frame_from_x(np.asarray(probe["axis"], float))))
        ET.SubElement(pb, "freejoint", name="probe_free")
        if probe["kind"] == "ball":
            ET.SubElement(pb, "geom", name="probe_g", type="sphere", size=_f(r["ball_radius"]), mass="0.002", rgba=_f(RGBA["probe"]))
        else:  # a short piece of cord along the passage axis
            h = 2 * r["cord_radius"]
            ET.SubElement(pb, "geom", name="probe_g", type="capsule", fromto=_f([-h, 0, 0, h, 0, 0]), size=_f(r["cord_radius"]), mass="0.0002",
                          rgba=_f(RGBA["probe"]))
        geom_part["probe_g"] = "probe"
        graspable.append("probe")

    for h in range(nh):
        hb = ET.SubElement(world, "body", name=f"handle{h}", mocap="true", pos=_f([0.2 + 0.03 * h, -0.15, 0.02]))
        ET.SubElement(hb, "geom", name=f"handle{h}_marker", rgba=_f(HANDLE_RGBA[h]), **{"class": "handle"})
    eq = ET.SubElement(root, "equality")
    weld_names = {}
    for h in range(nh):
        for g in graspable:
            nm = f"grip{h}__{g}"
            ET.SubElement(eq, "weld", name=nm, body1=f"handle{h}", body2=g, active="false", solref=_f(r["rig_weld_solref"]),
                          solimp="0.9 0.95 0.001 0.5 2")
            weld_names[(h, g)] = nm
    xml = ET.tostring(root, encoding="unicode")
    return RatTrapBuilt(xml=xml, num_handles=nh, rope=rope, graspable=graspable, frame_bodies=sorted(parts),
                        object_bodies=["ring"] + (["probe"] if probe_name else []), weld_names=weld_names, geom_part=geom_part,
                        ball_node=ball_node, probe=probe_name)


def _cord(spec: RatTrapSpec, world: ET.Element, contact: ET.Element, geom_part: dict) -> tuple[RopeInfo, int]:
    r = spec.raw
    n = spec.cord_segments
    nodes = resample_arclength(spec.route, n)  # n + 1 nodes, equal arc length
    seg = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    quats = [frame_from_x(d) for d in seg]
    m = r["cord_mass"] / n
    parent, names, elems = world, [], []
    for i in range(n):
        name = f"cord_{i}"
        if i == 0:
            pos, quat = nodes[0], quats[0]
        else:
            pos, quat = np.array([lengths[i - 1], 0.0, 0.0]), quat_mul(quat_conj(quats[i - 1]), quats[i])
        b = ET.SubElement(parent, "body", name=name, pos=_f(pos), quat=_f(quat))
        if i == 0 and not r["cord_anchor_start"]:
            ET.SubElement(b, "freejoint", name="cord_free")
        else:  # i == 0 with an anchored start: a ball joint to the world pins node 0
            ET.SubElement(b, "joint", name=f"{name}_ball", type="ball", damping=f"{r['cord_damping']}", armature=f"{r['cord_armature']}")
        ET.SubElement(b, "geom", name=f"{name}_g", type="capsule", size=_f(r["cord_radius"]), fromto=_f([0, 0, 0, lengths[i], 0, 0]),
                      mass=f"{m:.9g}", rgba=_f(RGBA["cord"]))
        names.append(name)
        elems.append(b)
        parent = b
    bodies = {}  # node k -> (body index, local position)
    for k in range(n + 1):
        bodies[k] = (k, np.zeros(3)) if k < n else (n - 1, np.array([lengths[-1], 0, 0]))
    ball_node = None
    if r.get("ball_mode", "fixed") != "fixed":
        raise NotImplementedError(f"ball_mode {r['ball_mode']!r}: only 'fixed' is implemented")
    k = int(r["ball_node"])
    ball_node = k % (n + 1)
    bi, local = bodies[ball_node]
    _attach_sphere(names, elems, bi, local, "ball", r["ball_radius"], r, contact, lengths, geom_part)
    if r.get("mouse_enabled"):
        bi, local = bodies[n]
        _attach_sphere(names, elems, bi, local, "mouse", r["mouse_radius"], r, contact, lengths, geom_part)
    # the anchored node sits on the base: don't collide the first segment with the base
    if r["cord_anchor_start"]:
        ET.SubElement(contact, "exclude", body1="base", body2=names[0])
    rope = RopeInfo(name="cord", body_names=names, seg_lengths=[float(x) for x in lengths], radius=r["cord_radius"],
                    length=float(lengths.sum()), start=nodes[0], end=None, connect_eq=None)
    return rope, ball_node


def _attach_sphere(names, elems, bi, local, nm, radius, r, contact, lengths, geom_part) -> None:
    """A sphere welded to a cord node: a geom of that segment's body at the node. It never collides with the
    cord segments it overlaps by construction (those within its radius along the chain)."""
    ET.SubElement(elems[bi], "geom", name=nm, type="sphere", pos=_f(local), size=_f(radius), mass=f"{r['cord_mass'] / 20:.9g}",
                  rgba=_f(RGBA[nm]))
    geom_part[nm] = nm
    reach = int(math.ceil((radius + r["cord_radius"]) / float(np.min(lengths)))) + 1
    for j in range(max(0, bi - reach), min(len(names), bi + reach + 1)):
        if j != bi:
            ET.SubElement(contact, "exclude", body1=names[bi], body2=names[j])
