"""Build an MJCF model programmatically from a :class:`Puzzle` spec.

Never hand-edit the generated XML: change the YAML or this builder.

Structure of the generated model
--------------------------------
* ``floor`` plane at z = 0.
* every frame body is a static child of the world (no joint).
* every free object gets a free joint (with damping).
* the rope is a chain of capsule bodies linked by ball joints; the first one is
  attached to the world at ``rope.start``, the last one is pinned to
  ``rope.end`` with a ``connect`` equality. Joint coordinates make stretching
  impossible; only the closing ``connect`` is soft and it is audited.
* ``num_handles`` mocap bodies (``handle0``, ``handle1``) and, for every
  (handle, graspable body) pair, an inactive ``weld`` equality.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import basis_perp, frame_from_x, polyline_length, quat_conj, quat_mul, resample_arclength
from .spec import Body, Geom, Puzzle

FRAME_RGBA = (0.55, 0.6, 0.68, 1.0)
OBJECT_RGBA = (0.2, 0.55, 0.85, 1.0)
HANDLE_RGBA = [(0.9, 0.2, 0.2, 0.6), (0.2, 0.8, 0.3, 0.6)]


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.ravel(v))


@dataclass
class RopeInfo:
    name: str
    body_names: list[str]
    seg_lengths: list[float]
    radius: float
    length: float
    start: np.ndarray
    end: np.ndarray | None
    connect_eq: str | None


@dataclass
class BuiltModel:
    xml: str
    puzzle: Puzzle
    num_handles: int
    rope: RopeInfo
    graspable: list[str]  # body names
    frame_bodies: list[str]
    object_bodies: list[str]
    weld_names: dict[tuple[int, str], str] = field(default_factory=dict)
    frame_aabb: tuple[np.ndarray, np.ndarray] | None = None


def _ring_capsules(center, normal, radius, tube_radius, segments):
    u, v = basis_perp(normal)
    c = np.asarray(center, dtype=float)
    th = np.linspace(0, 2 * np.pi, segments + 1)
    pts = [c + radius * (np.cos(t) * u + np.sin(t) * v) for t in th]
    return [(pts[i], pts[i + 1]) for i in range(segments)]


def _geom_aabb_points(g: Geom) -> list[np.ndarray]:
    """Rough points bounding a primitive in its body frame (used for the frame AABB)."""
    p = np.asarray(g.pos, dtype=float)
    if g.type == "box":
        r = float(np.linalg.norm(g.size))
        return [p - r, p + r]
    if g.type == "sphere":
        return [p - g.size[0], p + g.size[0]]
    if g.type in ("capsule", "cylinder"):
        a, b = np.asarray(g.fromto[:3]), np.asarray(g.fromto[3:])
        r = g.size[0]
        return [np.minimum(a, b) - r, np.maximum(a, b) + r]
    if g.type == "capsule_ring":
        r = g.radius + g.tube_radius
        return [p - r, p + r]
    return [p]


class MJCFBuilder:
    def __init__(self, puzzle: Puzzle, num_handles: int = 2, weld_solref=(0.02, 1.0), weld_solimp=(0.9, 0.95, 0.001, 0.5, 2.0)):
        self.p = puzzle
        self.num_handles = num_handles
        self.weld_solref = weld_solref
        self.weld_solimp = weld_solimp
        self.root = ET.Element("mujoco", model=puzzle.name)
        self.asset = None
        self._mesh_count = 0
        self._base_dir = Path(puzzle.source_path).parent if puzzle.source_path else Path.cwd()

    # ------------------------------------------------------------------ geoms
    def _add_geoms(self, parent: ET.Element, body: Body, default_rgba, mass_total: float | None, cls: str):
        elems: list[ET.Element] = []
        for gi, g in enumerate(body.geoms):
            rgba = g.rgba or body.rgba or default_rgba
            base = f"{body.name}_{g.name or gi}"
            attrs = {"class": cls, "rgba": _fmt(rgba)}
            if g.type == "capsule_ring":
                for si, (a, b) in enumerate(_ring_capsules(g.pos, g.normal, g.radius, g.tube_radius, g.segments)):
                    elems.append(
                        ET.SubElement(parent, "geom", name=f"{base}_s{si}", type="capsule", size=_fmt([g.tube_radius]), fromto=_fmt([*a, *b]), **attrs)
                    )
            elif g.type == "mesh":
                from .mesh_import import decompose_mesh

                mpath = Path(g.path)
                if not mpath.is_absolute():
                    mpath = (self._base_dir / mpath).resolve()
                parts = decompose_mesh(mpath, g.scale, g.decompose, g.coacd_threshold)
                for pi, part in enumerate(parts):
                    mname = f"mesh{self._mesh_count}"
                    self._mesh_count += 1
                    ET.SubElement(self.asset, "mesh", name=mname, file=str(part))
                    elems.append(
                        ET.SubElement(parent, "geom", name=f"{base}_p{pi}", type="mesh", mesh=mname, pos=_fmt(g.pos), euler=_fmt(g.euler), **attrs)
                    )
            else:
                kw = dict(attrs)
                if g.type in ("capsule", "cylinder"):
                    kw["fromto"] = _fmt(g.fromto)
                else:
                    kw["pos"] = _fmt(g.pos)
                    kw["euler"] = _fmt(g.euler)
                elems.append(ET.SubElement(parent, "geom", name=base, type=g.type, size=_fmt(g.size), **kw))
        if mass_total is not None and elems:
            for e in elems:
                e.set("mass", f"{mass_total / len(elems):.9g}")
        return elems

    # ------------------------------------------------------------------ build
    def build(self) -> BuiltModel:
        p = self.p
        root = self.root
        ET.SubElement(root, "compiler", angle="degree", autolimits="true")
        ET.SubElement(
            root,
            "option",
            timestep=f"{p.timestep}",
            gravity=_fmt(p.gravity),
            integrator="implicitfast",
            cone="pyramidal",
        )
        ET.SubElement(root, "size", memory="64M")
        vis = ET.SubElement(root, "visual")
        ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6", specular="0.1 0.1 0.1")
        ET.SubElement(vis, "global", offwidth="1280", offheight="960")
        ET.SubElement(vis, "quality", shadowsize="2048")

        default = ET.SubElement(root, "default")
        ET.SubElement(
            default,
            "geom",
            condim="1",  # frictionless contacts everywhere (rope must slide freely)
            solref=_fmt(p.contact_solref),
            solimp=_fmt(p.contact_solimp),
            margin="0",
        )
        for cls in ("frame", "object", "rope"):
            ET.SubElement(ET.SubElement(default, "default", **{"class": cls}), "geom", group={"frame": "0", "object": "1", "rope": "2"}[cls])
        hdef = ET.SubElement(default, "default", **{"class": "handle"})
        ET.SubElement(hdef, "geom", contype="0", conaffinity="0", group="3", type="sphere", size="0.012")

        self.asset = ET.SubElement(root, "asset")
        ET.SubElement(self.asset, "texture", name="grid", type="2d", builtin="checker", rgb1=".85 .85 .85", rgb2=".75 .75 .78", width="512", height="512")
        ET.SubElement(self.asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.05")
        ET.SubElement(self.asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.95 0.96 0.98", rgb2="0.75 0.8 0.88", width="256", height="256")

        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", name="key", pos="0.5 -0.8 1.8", dir="-0.3 0.5 -1", diffuse="0.6 0.6 0.6", castshadow="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.05", material="grid", condim="3", friction="0.8 0.01 0.001", group="0")

        # named fixed cameras (presets for render)
        frame_pts = []
        frame_names = []
        for b in p.frame:
            be = ET.SubElement(world, "body", name=b.name, pos=_fmt(b.pos), euler=_fmt(b.euler))
            self._add_geoms(be, b, FRAME_RGBA, None, "frame")
            frame_names.append(b.name)
            for g in b.geoms:
                for q in _geom_aabb_points(g):
                    frame_pts.append(np.asarray(b.pos) + q)  # euler ignored: coarse bound

        obj_names = []
        graspable = []
        for b in p.objects:
            be = ET.SubElement(world, "body", name=b.name, pos=_fmt(b.pos), euler=_fmt(b.euler))
            ET.SubElement(be, "joint", name=f"{b.name}_free", type="free", damping=f"{b.damping}")
            self._add_geoms(be, b, OBJECT_RGBA, b.mass, "object")
            obj_names.append(b.name)
            if b.graspable:
                graspable.append(b.name)

        rope = self._build_rope(world)
        if p.rope.graspable:
            graspable.extend(rope.body_names)

        # handles
        for h in range(self.num_handles):
            hb = ET.SubElement(world, "body", name=f"handle{h}", mocap="true", pos=f"{0.3 * h - 0.15} -1.2 0.02")
            ET.SubElement(hb, "geom", name=f"handle{h}_marker", rgba=_fmt(HANDLE_RGBA[h]), **{"class": "handle"})
            ET.SubElement(hb, "site", name=f"handle{h}_site", size="0.004")

        eq = ET.SubElement(root, "equality")
        if rope.connect_eq:
            last = rope.body_names[-1]
            ET.SubElement(eq, "connect", name=rope.connect_eq, body1=last, anchor=_fmt([rope.seg_lengths[-1], 0, 0]), solref="0.004 1")
        weld_names = {}
        for h in range(self.num_handles):
            for g in graspable:
                n = f"grip{h}__{g}"
                ET.SubElement(
                    eq, "weld", name=n, body1=f"handle{h}", body2=g, active="false",
                    solref=_fmt(self.weld_solref), solimp=_fmt(self.weld_solimp),
                )
                weld_names[(h, g)] = n

        # the first / last rope segments sit on their anchors: don't collide them with the anchoring body
        contact = ET.SubElement(root, "contact")
        for anchor, bname in ((p.rope.start, rope.body_names[0]), (p.rope.end, rope.body_names[-1])):
            if anchor is None:
                continue
            targets = frame_names if anchor.body == "world" else [anchor.body]
            for t in targets:
                ET.SubElement(contact, "exclude", body1=t, body2=bname)

        aabb = None
        if frame_pts:
            arr = np.array(frame_pts)
            aabb = (arr.min(axis=0), arr.max(axis=0))
        ET.indent(root)
        xml = ET.tostring(root, encoding="unicode")
        return BuiltModel(
            xml=xml,
            puzzle=p,
            num_handles=self.num_handles,
            rope=rope,
            graspable=graspable,
            frame_bodies=frame_names,
            object_bodies=obj_names,
            weld_names=weld_names,
            frame_aabb=aabb,
        )

    def _build_rope(self, world: ET.Element) -> RopeInfo:
        r = self.p.rope
        pts = [np.asarray(r.start.pos, dtype=float)] + [np.asarray(w, dtype=float) for w in r.path]
        if r.end is not None:
            pts.append(np.asarray(r.end.pos, dtype=float))
        nodes = resample_arclength(np.array(pts), r.segments)
        seg = nodes[1:] - nodes[:-1]
        lengths = np.linalg.norm(seg, axis=1)
        quats = [frame_from_x(d) for d in seg]
        seg_mass = r.mass / r.segments
        parent = world
        names = []
        for i in range(r.segments):
            name = f"{r.name}_{i}"
            if i == 0:
                pos, quat = nodes[0], quats[0]
            else:
                pos = np.array([lengths[i - 1], 0.0, 0.0])
                quat = quat_mul(quat_conj(quats[i - 1]), quats[i])
            b = ET.SubElement(parent, "body", name=name, pos=_fmt(pos), quat=_fmt(quat))
            ET.SubElement(b, "joint", name=f"{name}_ball", type="ball", damping=f"{r.damping}", armature=f"{r.armature}")
            ET.SubElement(
                b, "geom", name=f"{name}_geom", type="capsule", size=_fmt([r.radius]),
                fromto=_fmt([0, 0, 0, lengths[i], 0, 0]), mass=f"{seg_mass:.9g}", rgba=_fmt(r.rgba), **{"class": "rope"},
            )
            names.append(name)
            parent = b
        return RopeInfo(
            name=r.name,
            body_names=names,
            seg_lengths=[float(x) for x in lengths],
            radius=r.radius,
            length=float(np.sum(lengths)),
            start=nodes[0],
            end=nodes[-1] if r.end is not None else None,
            connect_eq=f"{r.name}_end" if r.end is not None else None,
        )


def build_model(puzzle: Puzzle, num_handles: int = 2, **kw) -> BuiltModel:
    return MJCFBuilder(puzzle, num_handles=num_handles, **kw).build()


__all__ = ["BuiltModel", "RopeInfo", "build_model", "polyline_length"]
