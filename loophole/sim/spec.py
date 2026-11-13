"""Pydantic schema for puzzle YAML files.

A puzzle is a static *frame* (fixed to the world), zero or more free rigid
*objects*, one *rope* and a list of *holes* (discs used by the topology
tracker). Geometry is described with primitives; hollow parts use the
``capsule_ring`` primitive or a convex-decomposed ``mesh``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Vec3 = tuple[float, float, float]


class Geom(BaseModel):
    """One primitive. Coordinates are in the owning body's frame."""

    type: Literal["box", "sphere", "capsule", "cylinder", "capsule_ring", "mesh"]
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
    # mesh: imported and (by default) convex-decomposed with CoACD.
    path: str | None = None
    scale: Vec3 = (1.0, 1.0, 1.0)
    decompose: Literal["coacd", "convex", "none"] = "coacd"
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
        if t == "mesh" and not self.path:
            raise ValueError("mesh needs path")
        return self


class Body(BaseModel):
    name: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    euler: Vec3 = (0.0, 0.0, 0.0)
    geoms: list[Geom]
    graspable: bool = False
    mass: float | None = None  # total mass for free objects
    damping: float = 0.01  # free-joint damping (objects only)
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
    damping: float = 0.002
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
    type: Literal["target_free"] = "target_free"
    margin: float = 0.05  # extra distance beyond the rope's reach


class Puzzle(BaseModel):
    name: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    description: str = ""
    max_actions: int = 60
    timestep: float = 0.002
    gravity: Vec3 = (0.0, 0.0, -9.81)
    contact_solref: list[float] = [0.004, 1.0]
    contact_solimp: list[float] = [0.95, 0.99, 0.001, 0.5, 2.0]
    frame: list[Body]
    objects: list[Body] = []
    rope: Rope
    holes: list[Hole] = []
    target: str
    goal: Goal = Goal()
    init_noise: float = 0.0  # std of seeded jitter on free-object start positions (m)
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


def load_puzzle(path: str | Path) -> Puzzle:
    path = Path(path)
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    data["source_path"] = str(path.resolve())
    data["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return Puzzle.model_validate(data)


PUZZLE_DIR = Path(__file__).resolve().parent.parent / "puzzles"


def resolve_puzzle(name_or_path: str) -> Path:
    """Accept a puzzle name (``easy``) or a path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = PUZZLE_DIR / f"{name_or_path}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"unknown puzzle {name_or_path!r}")
