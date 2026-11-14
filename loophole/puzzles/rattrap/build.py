"""Rat Trap spec -> MJCF (generated, never hand-edited).

Rigid parts are primitives only (boxes, cylinders, capsules, spheres): no meshes, no convex decomposition.
All rigid parts are static (bodies without joints = welded to the world): the frame and base, the two
cantilevered posts (each with a hexagonal hole of thin wall boxes), the left dowel with its sphere, and the
right dowel with its ring (N capsules around a circle). The cord is a chain of capsules connected by ball
joints, a kinematic tree in joint coordinates: segment lengths are exact, so it is inextensible by
construction (no springs, no breaking, no elasticity). Gravity off, friction off (condim 1), light damping on
every joint plus a light viscous medium.

The returned `RatTrapBuilt` carries the fields the Alpha rig (loophole.sim.rig.Rig) reads, so the handle
mechanism and its blocked-move semantics are reused unchanged.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from ...sim.builder import RopeInfo
from ...sim.geometry import frame_from_x, quat_conj, quat_from_axis_angle, quat_mul, resample_arclength
from .spec import RatTrapSpec

WOOD, WOOD_LIGHT, DARK = (0.55, 0.38, 0.22, 1), (0.82, 0.68, 0.48, 1), (0.25, 0.16, 0.10, 1)
RGBA = {"base": WOOD, "frame": WOOD, "post_upper": WOOD, "post_lower": WOOD, "dowel_left": DARK, "dowel_right": DARK,
        "sphere": WOOD_LIGHT, "ring": WOOD, "cord": (0.95, 0.80, 0.15, 1), "cord_ball": (0.2, 0.45, 0.85, 1),
        "mouse": (0.45, 0.30, 0.20, 1), "probe": (0.2, 0.75, 0.3, 1)}
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
    geom_part: dict = field(default_factory=dict)  # geom name -> part name
    ball_node: int | None = None
    probe: str | None = None
    seg_reversed: list = field(default_factory=list)  # per segment: body frame origin at its END node (built backwards)


def build(spec: RatTrapSpec, include_cord: bool = True, probe: dict | None = None, num_handles: int | None = None,
          ring_free: bool | None = None) -> RatTrapBuilt:
    """probe (clearance tests): {"kind": "cord"|"mouse"|"cord_ball", "pos": [...], "axis": [...]} adds a free,
    graspable probe body: a cord-radius capsule along `axis`, or a sphere of the mouse / cord-ball radius.
    (`ring_free` is accepted for backwards compatibility and ignored: the ring is part of the right dowel.)"""
    r = spec.raw
    nh = int(num_handles if num_handles is not None else r["rig_num_handles"])
    root = ET.Element("mujoco", model="rattrap")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    # explicit Euler: the implicit integrators differentiate the viscous medium densely (~6x slower here)
    ET.SubElement(root, "option", timestep=f"{r['timestep']}", gravity="0 0 0", integrator="Euler",
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
    parts: dict[str, ET.Element] = {}

    def part_body(name: str) -> ET.Element:
        if name not in parts:
            parts[name] = ET.SubElement(world, "body", name=name)
        return parts[name]

    for b in spec.boxes:
        kw = {"quat": _f(quat_from_axis_angle([0, 0, 1], b.yaw))} if b.yaw else {}
        ET.SubElement(part_body(b.part), "geom", name=b.name, type="box", pos=_f(b.pos), size=_f(b.half), rgba=_f(RGBA[b.part]), **kw)
        geom_part[b.name] = b.part
    for c in spec.cylinders:
        ET.SubElement(part_body(c.part), "geom", name=c.name, type="cylinder", pos=_f(c.pos), size=_f([c.radius, c.half_length]),
                      rgba=_f(RGBA[c.part]))
        geom_part[c.name] = c.part
    for sp in spec.spheres:
        ET.SubElement(part_body(sp.part), "geom", name=sp.name, type="sphere", pos=_f(sp.pos), size=_f(sp.radius), rgba=_f(RGBA["sphere"]))
        geom_part[sp.name] = "sphere"
    for cp in spec.capsules:
        ET.SubElement(part_body(cp.part), "geom", name=cp.name, type="capsule", fromto=_f(np.concatenate([cp.p0, cp.p1])),
                      size=_f(cp.radius), rgba=_f(RGBA["ring"]))
        geom_part[cp.name] = "ring"

    rope, graspable, ball_node = None, [], None
    contact = ET.SubElement(root, "contact")
    eq = ET.SubElement(root, "equality")
    if include_cord:
        rope, ball_node = _cord(spec, world, contact, eq, geom_part)
        graspable = list(rope.body_names)
    probe_name = None
    if probe is not None:
        probe_name = "probe"
        pb = ET.SubElement(world, "body", name="probe", pos=_f(probe["pos"]), quat=_f(frame_from_x(np.asarray(probe["axis"], float))))
        ET.SubElement(pb, "freejoint", name="probe_free")
        if probe["kind"] in ("mouse", "cord_ball"):
            rad = r["mouse_radius"] if probe["kind"] == "mouse" else r["cord_ball_radius"]
            ET.SubElement(pb, "geom", name="probe_g", type="sphere", size=_f(rad), mass="0.002", rgba=_f(RGBA["probe"]))
        else:  # a short piece of cord along the passage axis
            h = 2 * r["cord_radius"]
            ET.SubElement(pb, "geom", name="probe_g", type="capsule", fromto=_f([-h, 0, 0, h, 0, 0]), size=_f(r["cord_radius"]), mass="0.0002",
                          rgba=_f(RGBA["probe"]))
        geom_part["probe_g"] = "probe"
        graspable.append("probe")

    for h in range(nh):
        hb = ET.SubElement(world, "body", name=f"handle{h}", mocap="true", pos=_f([0.15 + 0.03 * h, -0.12, 0.03]))
        ET.SubElement(hb, "geom", name=f"handle{h}_marker", rgba=_f(HANDLE_RGBA[h]), **{"class": "handle"})
    weld_names = {}
    for h in range(nh):
        for g in graspable:
            nm = f"grip{h}__{g}"
            ET.SubElement(eq, "weld", name=nm, body1=f"handle{h}", body2=g, active="false", solref=_f(r["rig_weld_solref"]),
                          solimp="0.9 0.95 0.001 0.5 2")
            weld_names[(h, g)] = nm
    xml = ET.tostring(root, encoding="unicode")
    return RatTrapBuilt(xml=xml, num_handles=nh, rope=rope, graspable=graspable, frame_bodies=sorted(parts),
                        object_bodies=(["probe"] if probe_name else []), weld_names=weld_names, geom_part=geom_part,
                        ball_node=ball_node, probe=probe_name, seg_reversed=list(getattr(rope, "_reversed", [])) if rope else [])


def _cord(spec: RatTrapSpec, world: ET.Element, contact: ET.Element, eq: ET.Element, geom_part: dict) -> tuple[RopeInfo, int | None]:
    r = spec.raw
    n = spec.cord_segments
    nodes = resample_arclength(spec.route, n)  # n + 1 nodes, equal arc length
    seg = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    quats = [frame_from_x(d) for d in seg]
    m = r["cord_mass"] / n
    # Kinematic tree. Pinned open cord: rooted at node 0 (ball joint to the world). Otherwise rooted at the middle
    # segment with two branches: in joint coordinates the cost grows with the chain's depth, so a 2-branch tree
    # is ~3x faster than one long chain. Segment i always spans node i -> node i+1; segments of the backward
    # branch are built with their frame origin at node i+1 and x pointing to node i (`reversed`).
    names = [f"cord_{i}" for i in range(n)]
    elems: list = [None] * n
    rev = [False] * n
    anchored = bool(r["cord_anchor_start"])
    root = 0 if anchored else n // 2

    def add(i: int, parent, pos, quat, backward: bool):
        b = ET.SubElement(parent, "body", name=names[i], pos=_f(pos), quat=_f(quat))
        if parent is world and not anchored:
            ET.SubElement(b, "freejoint", name="cord_free")
        else:
            ET.SubElement(b, "joint", name=f"{names[i]}_ball", type="ball", damping=f"{r['cord_damping']}", armature=f"{r['cord_armature']}")
        ET.SubElement(b, "geom", name=f"{names[i]}_g", type="capsule", size=_f(r["cord_radius"]), fromto=_f([0, 0, 0, lengths[i], 0, 0]),
                      mass=f"{m:.9g}", rgba=_f(RGBA["cord"]))
        elems[i], rev[i] = b, backward
        return b

    qf = quats  # frame of segment i pointing node i -> node i+1
    qb = [frame_from_x(-d) for d in seg]  # pointing node i+1 -> node i
    parent = add(root, world, nodes[root], qf[root], False)
    for i in range(root + 1, n):  # forward branch: each child starts at its parent's end node
        parent = add(i, parent, [lengths[i - 1], 0.0, 0.0], quat_mul(quat_conj(qf[i - 1]), qf[i]), False)
    if root > 0:  # backward branch: segment root-1 hangs off node `root` (the root's origin)
        parent = add(root - 1, elems[root], [0.0, 0.0, 0.0], quat_mul(quat_conj(qf[root]), qb[root - 1]), True)
        for i in range(root - 2, -1, -1):  # each child starts at its parent's far end (node i+1)
            parent = add(i, parent, [lengths[i + 1], 0.0, 0.0], quat_mul(quat_conj(qb[i + 1]), qb[i]), True)
    # node k -> (segment body, local position of the node in that body's frame)
    bodies = {}
    for k in range(n + 1):
        i = min(k, n - 1)
        at_start = k < n
        bodies[k] = (i, np.zeros(3) if at_start != rev[i] else np.array([lengths[i], 0, 0]))
    ball_node = None
    if r.get("cord_ball_enabled"):
        if r.get("ball_mode", "fixed") != "fixed":
            raise NotImplementedError(f"ball_mode {r['ball_mode']!r}: only 'fixed' is implemented")
        ball_node = int(r["cord_ball_node"]) % (n + 1)
        bi, local = bodies[ball_node]
        _attach_sphere(names, elems, bi, local, "cord_ball", r["cord_ball_radius"], r, contact, lengths, geom_part)
    if r.get("mouse_enabled"):
        mn = r.get("mouse_node", "auto")
        if mn == "auto":  # open cord: the free end; closed loop: the node farthest from node 0 along the cord
            mn = n if not r.get("cord_closed") else n // 2
        bi, local = bodies[int(mn) % (n + 1)]
        _attach_sphere(names, elems, bi, local, "mouse", r["mouse_radius"], r, contact, lengths, geom_part)
    if r["cord_anchor_start"]:  # the anchored node sits on the base: don't collide the first segment with it
        ET.SubElement(contact, "exclude", body1="base", body2=names[0])
    if r.get("cord_closed"):
        # close the loop: the end of the last segment is held on the start of the first (MuJoCo's stiffest stable
        # constraint, time constant 2 dt); the two meet at that node, so they must not collide with each other
        ET.SubElement(eq, "connect", name="cord_closure", body1=names[-1], body2=names[0], anchor=_f(bodies[n][1]),
                      solref=_f([2 * r["timestep"], 1.0]), solimp="0.95 0.99 0.0001 0.5 2")
        ET.SubElement(contact, "exclude", body1=names[-1], body2=names[0])
    rope = RopeInfo(name="cord", body_names=names, seg_lengths=[float(x) for x in lengths], radius=r["cord_radius"],
                    length=float(lengths.sum()), start=nodes[0], end=None, connect_eq=None)
    rope._reversed = rev
    return rope, ball_node


def _attach_sphere(names, elems, bi, local, nm, radius, r, contact, lengths, geom_part) -> None:
    """A sphere welded to a cord node: a geom of that segment's body at the node. It never collides with the
    cord segments it overlaps by construction (those within its radius along the chain)."""
    ET.SubElement(elems[bi], "geom", name=nm, type="sphere", pos=_f(local), size=_f(radius), mass=f"{r['cord_mass'] / 20:.9g}",
                  rgba=_f(RGBA[nm]))
    geom_part[nm] = nm
    reach = int(math.ceil((radius + r["cord_radius"]) / float(np.min(lengths)))) + 1
    n = len(names)
    if r.get("cord_closed"):  # a closed loop wraps around: the neighbours across the closure are also overlapped
        js = {(bi + o) % n for o in range(-reach, reach + 1)}
    else:
        js = set(range(max(0, bi - reach), min(n, bi + reach + 1)))
    for j in sorted(js - {bi}):
        ET.SubElement(contact, "exclude", body1=names[bi], body2=names[j])
