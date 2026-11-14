"""Pydantic schema for puzzle YAML files.

A puzzle is a static *frame* (fixed to the world), zero or more free rigid
*objects*, an optional *rope* and a list of *holes* (discs used by the topology
tracker). Geometry is described with primitives; hollow parts use the
``capsule_ring`` / ``tube`` primitives or a convex-decomposed ``mesh``.

Units: every length in the YAML is in *native* units and multiplied by
``unit_scale`` in exactly one place (:func:`apply_unit_scale`, at load time)
to get simulator units (metres). ``Puzzle.native`` keeps the untouched values
and :func:`to_native` converts simulator lengths back. Values under ``config``
(environment overrides) are already in simulator units.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Vec3 = tuple[float, float, float]


class Geom(BaseModel):
    """One primitive. Coordinates are in the owning body's frame."""

    type: Literal["box", "sphere", "capsule", "cylinder", "capsule_ring", "tube", "mesh"]
    name: str | None = None
    pos: Vec3 = (0.0, 0.0, 0.0)
    euler: Vec3 = (0.0, 0.0, 0.0)  # degrees, XYZ
    size: list[float] | None = None  # box half-sizes / sphere radius / capsule radius
    fromto: list[float] | None = None  # capsule / cylinder endpoints (6 numbers)
    # capsule_ring: a hollow ring made of `segments` capsules around `normal`.
    normal: Vec3 = (0.0, 0.0, 1.0)
    radius: float | None = None  # ring centreline radius
    tube_radius: float | None = None
    segments: int = 16
    # tube: a swept sphere along a polyline centre-line = a chain of capsules (an exact
    # convex decomposition of the swept volume; error vs the smooth curve = polyline sagitta).
    points: list[Vec3] | None = None
    # mesh: imported and (by default) convex-decomposed with CoACD.
    path: str | None = None
    scale: Vec3 = (1.0, 1.0, 1.0)
    decompose: Literal["coacd", "facewedge", "chunkhull", "convex", "none"] = "coacd"
    coacd_threshold: float = 0.05
    rgba: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Geom":
        t = self.type
        if t == "box" and (not self.size or len(self.size) != 3):
            raise ValueError("box needs size: [hx, hy, hz]")
        if t == "sphere" and (not self.size or len(self.size) != 1):
            raise ValueError("sphere needs size: [r]")
        if t in ("capsule", "cylinder") and (not self.size or not self.fromto or len(self.fromto) != 6):
            raise ValueError(f"{t} needs size: [r] and fromto: [x1 y1 z1 x2 y2 z2]")
        if t == "capsule_ring" and (self.radius is None or self.tube_radius is None):
            raise ValueError("capsule_ring needs radius and tube_radius")
        if t == "tube" and (not self.points or len(self.points) < 2 or self.tube_radius is None):
            raise ValueError("tube needs points (>= 2) and tube_radius")
        if t == "mesh" and not self.path:
            raise ValueError("mesh needs path")
        return self


class Body(BaseModel):
    name: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    euler: Vec3 = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] | None = None  # (w, x, y, z); overrides euler
    geoms: list[Geom]
    graspable: bool = False
    mass: float | None = None  # total mass for free objects
    damping: float = 0.05  # free-joint damping (objects only): keeps drops quasi-static (terminal speed ~ m g / damping)
    rgba: tuple[float, float, float, float] | None = None


class Anchor(BaseModel):
    pos: Vec3
    body: str = "world"  # frame bodies are static, so world coordinates are used


class Rope(BaseModel):
    name: str = "rope"
    start: Anchor
    end: Anchor | None = None  # None => free end
    path: list[Vec3] = Field(..., description="waypoints between the anchors (start/end included automatically)")
    segments: int = 30
    radius: float = 0.006
    mass: float = 0.03  # total
    damping: float = 0.005
    armature: float = 1e-5
    graspable: bool = True
    rgba: tuple[float, float, float, float] = (0.85, 0.55, 0.15, 1.0)


class Hole(BaseModel):
    """A disc spanning a hole. Crossing it in the +normal direction appends the
    lowercase symbol to topology words, crossing it in -normal the uppercase one."""

    name: str  # single lowercase letter used in topology words
    center: Vec3
    normal: Vec3
    radius: float
    body: str = "world"  # holes on free objects move with them (center/normal in body frame)

    @model_validator(mode="after")
    def _check(self) -> "Hole":
        if len(self.name) != 1 or not self.name.islower():
            raise ValueError("hole names must be single lowercase letters (used as word symbols)")
        return self


class Goal(BaseModel):
    # target_free: target farther from every rope anchor than the rope can reach
    # separation:  minimum distance between target and frame geometry above d_sep
    type: Literal["target_free", "separation"] = "target_free"
    margin: float = 0.05  # extra distance beyond the rope's reach
    d_sep: float = 0.05


class Puzzle(BaseModel):
    name: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    description: str = ""
    max_actions: int = 60
    timestep: float = 0.001
    gravity: Vec3 = (0.0, 0.0, -9.81)
    floor: bool = True  # ground plane at z = 0 (off for floating, gravity-free puzzles)
    contact_solref: list[float] = [0.002, 1.0]  # stiffest stable: timeconst = 2 * timestep
    contact_solimp: list[float] = [0.95, 0.99, 0.001, 0.5, 2.0]
    unit_scale: float = 1.0  # native YAML units -> simulator metres (applied once, at load)
    frame: list[Body]
    objects: list[Body] = []
    rope: Rope | None = None
    holes: list[Hole] = []
    target: str
    goal: Goal = Goal()
    init_noise: float = 0.0  # std of seeded jitter on free-object start positions (m)
    roadmap: str | None = None  # cost-to-go roadmap (.npz) next to the YAML, if any
    config: dict[str, Any] = {}  # EnvConfig overrides for this puzzle (simulator units)
    meta: dict[str, Any] = {}  # free-form generator metadata (parameters, clearances, ...)
    native: dict[str, Any] | None = None  # the YAML exactly as written (native units)
    source_path: str | None = None
    source_sha256: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "Puzzle":
        names = [b.name for b in self.frame] + [b.name for b in self.objects]
        if len(set(names)) != len(names):
            raise ValueError("body names must be unique")
        if self.target not in [o.name for o in self.objects]:
            raise ValueError("target must be one of the free objects")
        return self


def _scale_vecs(v, s):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v * s
    return [_scale_vecs(x, s) for x in v]


def apply_unit_scale(data: dict) -> dict:
    """Return a copy of a raw puzzle dict with every *length* multiplied by unit_scale.

    The single place where native units become simulator units. Angles,
    normals, quaternions, masses, damping and `config` are left unchanged.
    """
    s = float(data.get("unit_scale", 1.0))
    out = copy.deepcopy(data)
    if s == 1.0:
        return out

    def body(b):
        if "pos" in b:
            b["pos"] = _scale_vecs(b["pos"], s)
        for g in b.get("geoms", []):
            for k in ("pos", "fromto", "points", "radius", "tube_radius"):
                if k in g:
                    g[k] = _scale_vecs(g[k], s)
            if "size" in g:
                g["size"] = _scale_vecs(g["size"], s)
            if g.get("type") == "mesh":
                g["scale"] = _scale_vecs(g.get("scale", [1.0, 1.0, 1.0]), s)

    for b in out.get("frame", []) + out.get("objects", []):
        body(b)
    if out.get("rope"):
        r = out["rope"]
        for k in ("start", "end"):
            if r.get(k):
                r[k]["pos"] = _scale_vecs(r[k]["pos"], s)
        r["path"] = _scale_vecs(r.get("path", []), s)
        if "radius" in r:
            r["radius"] = r["radius"] * s
    for h in out.get("holes", []):
        h["center"] = _scale_vecs(h["center"], s)
        h["radius"] = h["radius"] * s
    if out.get("goal"):
        for k in ("margin", "d_sep"):
            if k in out["goal"]:
                out["goal"][k] = out["goal"][k] * s
    if "init_noise" in out:
        out["init_noise"] = out["init_noise"] * s
    return out


def to_native(value, puzzle: "Puzzle", ndigits: int = 9):
    """Convert simulator lengths (scalar or nested lists) back to native YAML units."""
    s = puzzle.unit_scale
    if isinstance(value, (int, float)):
        return round(value / s, ndigits)
    return [to_native(v, puzzle, ndigits) for v in value]


def load_puzzle(path: str | Path) -> Puzzle:
    path = Path(path)
    raw = path.read_bytes()
    native = yaml.safe_load(raw)
    data = apply_unit_scale(native)
    data["native"] = native
    data["source_path"] = str(path.resolve())
    data["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return Puzzle.model_validate(data)


PUZZLE_DIR = Path(__file__).resolve().parent.parent / "puzzles"


def resolve_puzzle(name_or_path: str) -> Path:
    """Accept a puzzle name (``easy``) or a path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    for cand in (PUZZLE_DIR / f"{name_or_path}.yaml", PUZZLE_DIR / "alpha" / f"{name_or_path}.yaml", PUZZLE_DIR / "alpha_ompl" / f"{name_or_path}.yaml", PUZZLE_DIR / "alpha_paper" / f"{name_or_path}.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"unknown puzzle {name_or_path!r}")
