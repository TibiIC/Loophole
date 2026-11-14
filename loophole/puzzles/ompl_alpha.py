"""Canonical Alpha puzzle from OMPL.app (resources/3D), fetched at build time, never committed.

Source: https://github.com/ompl/omplapp @ 958884d6593809ee6a564907ff4e796691d48ffc
Licence: Rice University Software Distribution License (non-commercial, no
redistribution). scripts/fetch_alpha.sh downloads the files with sha256 checks
into LOOPHOLE_OMPL_DIR (default <repo>/.ompl_alpha); `prepare()` writes derived
OBJ meshes next to them; CoACD hulls go to the (gitignored) mesh cache. Puzzle
YAMLs reference meshes as ``ompl:<file>`` and contain only poses and parameters.

Pose convention (from OMPL.app's own source, RigidBodyGeometry.cpp / assimpUtil.cpp /
FCLMethodWrapper.h): meshes are loaded with Assimp, node transforms applied, and
Assimp's Collada importer converts Z_UP files to Y-up, (x, y, z) -> (x, z, -y); the
robot's vertices are then shifted by its "scene centre" (mean of its vertices)
before the state (x, y, z, rotation) is applied. Environment vertices are used as-is.
Verified: with this convention OMPL's own alpha-1.5.path is collision-free under
FCL on the raw triangles (0/103 states), and fails badly without the up-axis step.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COMMIT = "958884d6593809ee6a564907ff4e796691d48ffc"
FILES = {
    "1.5": {"cfg": "alpha-1.5.cfg", "env": "alpha_env-1.5.dae", "robot": "alpha_robot.dae", "path": "alpha-1.5.path"},
    "1.2": {"cfg": "alpha-1.2.cfg", "env": "alpha_env-1.2.dae", "robot": "alpha_robot.dae"},
    "1.0": {"cfg": "alpha-1.0.cfg", "env": "alpha_env-1.0.dae", "robot": "alpha_robot.dae"},
}


def ompl_dir() -> Path:
    return Path(os.environ.get("LOOPHOLE_OMPL_DIR", REPO / ".ompl_alpha"))


def resolve_mesh_path(p: str) -> Path:
    """`ompl:<name>` -> file in the fetched OMPL directory."""
    if p.startswith("ompl:"):
        f = ompl_dir() / p[len("ompl:") :]
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run `make fetch-alpha` (OMPL.app meshes are fetched, never committed)")
        return f
    return Path(p)


def axis_angle_quat(theta: float, axis) -> np.ndarray:
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if theta == 0 or n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / n
    return np.array([np.cos(theta / 2), *(np.sin(theta / 2) * a)])


def parse_cfg(path: str | Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(path)
    pr = cp["problem"]
    out = {"name": pr["name"], "robot": pr["robot"], "world": pr["world"]}
    for k in ("start", "goal"):
        if f"{k}.x" in pr:
            out[k] = {
                "xyz": [float(pr[f"{k}.x"]), float(pr[f"{k}.y"]), float(pr[f"{k}.z"])],
                "theta": float(pr[f"{k}.theta"]),
                "axis": [float(pr[f"{k}.axis.x"]), float(pr[f"{k}.axis.y"]), float(pr[f"{k}.axis.z"])],
            }
    out["volume"] = {
        "min": [float(pr[f"volume.min.{a}"]) for a in "xyz"],
        "max": [float(pr[f"volume.max.{a}"]) for a in "xyz"],
    }
    return out


def parse_path(path: str | Path) -> np.ndarray:
    """OMPL SE3 path file: `x y z qx qy qz qw` per line -> (N, 7) as x y z qw qx qy qz (native units)."""
    rows = [list(map(float, l.split())) for l in Path(path).read_text().splitlines() if l.strip()]
    a = np.array(rows)
    return np.concatenate([a[:, :3], a[:, 6:7], a[:, 3:6]], axis=1)


ZUP_TO_YUP = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])


def load_scene_mesh(path: str | Path, assimp_frame: bool = False):
    """Load a Collada scene as one mesh (node transforms applied). With
    assimp_frame=True also apply Assimp's Z_UP -> Y-up conversion (OMPL's frame)."""
    import trimesh

    s = trimesh.load(path)
    m = s.to_geometry() if hasattr(s, "to_geometry") else s
    if assimp_frame:
        m.apply_transform(ZUP_TO_YUP)
    return m


def robot_center(mesh, convention: str = "mean_vertices") -> np.ndarray:
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if convention == "mean_vertices":  # every loaded vertex (closest to Assimp's per-face-normal vertex lists)
        return V.mean(axis=0)
    if convention == "mean_unique":
        return np.unique(V.round(6), axis=0).mean(axis=0)
    if convention == "mean_corners":
        return V[F].reshape(-1, 3).mean(axis=0)
    raise ValueError(convention)


def prepare(variant: str = "1.5", convention: str = "mean_vertices") -> dict:
    """Write derived OBJs (native units) into the OMPL dir; return metadata."""
    d = ompl_dir()
    f = FILES[variant]
    env = load_scene_mesh(d / f["env"], assimp_frame=True)
    rob = load_scene_mesh(d / f["robot"], assimp_frame=True)
    c = robot_center(rob, convention)
    rob_c = rob.copy()
    rob_c.apply_translation(-c)
    env_out = d / f"derived_env-{variant}.obj"
    rob_out = d / f"derived_robot-{convention}.obj"
    env.export(env_out)
    rob_c.export(rob_out)
    return {"env_obj": env_out.name, "robot_obj": rob_out.name, "robot_center_native": c.tolist(), "convention": convention}


UNIT_SCALE = 0.01  # OMPL native units -> metres (the scene becomes about +-2 m)
PUZZLE_OUT = Path(__file__).resolve().parent / "alpha_ompl"


def write_puzzle(variant: str = "1.5", convention: str = "mean_vertices", decompose: str = "facewedge", coacd_threshold: float = 0.03,
                 d_sep_native: float = 10.0, max_actions: int = 300, config: dict | None = None, name: str | None = None) -> Path:
    """Write the committed puzzle YAML: poses + parameters only; meshes are `ompl:` refs."""
    import yaml

    d = ompl_dir()
    f = FILES[variant]
    cfg = parse_cfg(d / f["cfg"])
    meta = prepare(variant, convention)
    s = cfg["start"]
    q = axis_angle_quat(s["theta"], s["axis"])
    spec = {
        "name": name or f"alpha-{variant}",
        "difficulty": {"1.5": "easy", "1.2": "medium", "1.0": "hard"}[variant],
        "description": "Canonical Alpha puzzle (OMPL.app): two interlocked alpha-shaped tubes; separate them.",
        "max_actions": max_actions,
        "unit_scale": UNIT_SCALE,
        "gravity": [0.0, 0.0, 0.0],
        "floor": False,
        "frame": [{"name": "tube_a", "geoms": [{"type": "mesh", "path": f"ompl:{meta['env_obj']}", "decompose": decompose, "coacd_threshold": coacd_threshold}]}],
        "objects": [{
            "name": "tube_b", "pos": s["xyz"], "quat": [float(v) for v in q], "graspable": True, "mass": 0.5, "damping": 2.0,
            "rgba": [0.2, 0.5, 0.9, 1.0],
            "geoms": [{"type": "mesh", "path": f"ompl:{meta['robot_obj']}", "decompose": decompose, "coacd_threshold": coacd_threshold}],
        }],
        "target": "tube_b",
        "goal": {"type": "separation", "d_sep": d_sep_native},
        "config": config or {},
        "meta": {
            "source": f"https://github.com/ompl/omplapp/tree/{COMMIT}/resources/3D",
            "licence": "Rice University Software Distribution License (non-commercial, no redistribution): files fetched, never committed",
            "files": {k: v for k, v in f.items()},
            "cfg_start": s, "cfg_goal": cfg.get("goal"), "cfg_volume": cfg["volume"],
            "robot_center_convention": convention, "robot_center_native": meta["robot_center_native"],
        },
    }
    PUZZLE_OUT.mkdir(exist_ok=True)
    out = PUZZLE_OUT / f"{spec['name']}.yaml"
    out.write_text("# GENERATED by loophole.puzzles.ompl_alpha.write_puzzle - poses/parameters only, no meshes.\n"
                   + yaml.safe_dump(spec, sort_keys=False, width=200))
    return out
