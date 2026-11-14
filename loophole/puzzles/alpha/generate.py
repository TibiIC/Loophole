"""Generate the Alpha-style puzzle variants (our own procedural geometry).

    python -m loophole.puzzles.alpha.generate            # all variants
    python -m loophole.puzzles.alpha.generate --only hard

Writes loophole/puzzles/alpha/alpha-<variant>.yaml and assets/alpha-<variant>-tube.obj.
Two identical tubes; tube A is fixed, tube B is free. The start pose is found by
a seeded search for the collision-free pose with the largest surface distance
among poses whose loops are linked (Gauss linking number of the closed loops = +-1).
Difficulty = the nominal clearance of the gap at each tube's crossing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ...sim.geometry import quat_from_rotvec, quat_normalize, quat_to_mat
from .geometry import centerline, gauss_linking, polyline_sagitta, resample_by_sagitta, segment_distances, twist_for_clearance

HERE = Path(__file__).resolve().parent

S = 0.15  # loop length scale (m)
R_TUBE = 0.01  # tube radius (m)
T_TAIL = 1.45  # parameter range: tails length
MAX_SAGITTA = 0.0003  # polyline deviation from the smooth centre-line (m)
D_SEP = 0.06  # separation threshold: no linked pose can have min distance > ~0.04 m

VARIANTS = {  # name: (difficulty, nominal gap clearance in m, max_actions)
    "alpha-easy": ("easy", 0.012, 120),
    "alpha-medium": ("medium", 0.006, 150),
    "alpha-hard": ("hard", 0.003, 200),
}

PUZZLE_CONFIG = {
    "reset_settle_steps": 50,
    "rig": {
        "num_handles": 1,
        "max_linear_step": 0.05,
        "max_angular_step": 0.4,
        "linear_speed": 0.1,
        "angular_speed": 0.5,
        "weld_solref": [0.002, 1.0],
        "lag_limit": 0.006,
        "angle_lag_limit": 0.08,
        "workspace_min": [-1.0, -1.0, -1.0],
        "workspace_max": [1.0, 1.0, 1.0],
        "grasp_tolerance": 0.3,
    },
    "topology": {"primary": "tube_b"},
}


def loop_polyline(k: float, n: int = 120) -> np.ndarray:
    L = centerline(S, k, 1.0, n)
    return np.vstack([L, L[:1]])  # closed through the gap


def loop_disc(k: float) -> dict:
    """Approximate spanning disc of the loop: centroid, area normal, inner radius."""
    L = centerline(S, k, 1.0, 400)
    c = L.mean(axis=0)
    area = 0.5 * np.sum(np.cross(L - c, np.roll(L, -1, axis=0) - c), axis=0)
    n = area / np.linalg.norm(area)
    radius = float(np.min(np.linalg.norm(L - c, axis=1))) - R_TUBE
    return {"center": c.round(6).tolist(), "normal": n.round(6).tolist(), "radius": round(radius, 6)}


def place(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Q @ quat_to_mat(quat_from_rotvec(x[:3])).T + x[3:]


def search_start(P: np.ndarray, L: np.ndarray, seed: int = 0, restarts: int = 40, iters: int = 300) -> tuple[np.ndarray, float, float]:
    def score(x):
        lk = gauss_linking(L, place(x, L))
        if abs(abs(lk) - 1) > 0.2:
            return -1.0, lk
        return float(segment_distances(P, place(x, P)).min() - 2 * R_TUBE), lk

    rng = np.random.default_rng(seed)
    best = (-np.inf, None, 0.0)
    for _ in range(restarts):
        x = np.concatenate([rng.normal(0, 1.5, 3), rng.normal([-0.5 * S, 0, 0], 0.06)])
        fx, lk = score(x)
        step = np.array([0.3, 0.3, 0.3, 0.02, 0.02, 0.02])
        for _ in range(iters):
            y = x + rng.normal(0, 1, 6) * step
            fy, lky = score(y)
            if fy > fx:
                x, fx, lk = y, fy, lky
            else:
                step *= 0.995
        if fx > best[0]:
            best = (fx, x.copy(), lk)
    return best[1], best[0], best[2]


def sweep_obj(path: Path, k: float, n_along: int = 400, n_around: int = 24) -> None:
    """Triangle mesh of the swept tube (rotation-minimising frames), for visualisation/export."""
    C = centerline(S, k, T_TAIL, n_along)
    tang = np.gradient(C, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    u = np.cross(tang[0], [0, 0, 1.0])
    u /= np.linalg.norm(u)
    verts = []
    for i in range(n_along):
        if i:
            u = u - np.dot(u, tang[i]) * tang[i]
            u /= np.linalg.norm(u)
        v = np.cross(tang[i], u)
        for j in range(n_around):
            a = 2 * np.pi * j / n_around
            verts.append(C[i] + R_TUBE * (np.cos(a) * u + np.sin(a) * v))
    faces = []
    for i in range(n_along - 1):
        for j in range(n_around):
            a, b = i * n_around + j, i * n_around + (j + 1) % n_around
            c, d = a + n_around, b + n_around
            faces += [(a, c, b), (b, c, d)]
    caps = []
    for end, rng_ in ((0, range(n_around)), (n_along - 1, range(n_around))):
        verts.append(C[end])
        ci = len(verts) - 1
        for j in rng_:
            a, b = end * n_around + j, end * n_around + (j + 1) % n_around
            caps.append((ci, b, a) if end == 0 else (ci, a, b))
    with open(path, "w") as f:
        f.write(f"# Loophole alpha-style tube: S={S} r={R_TUBE} k={k:.6f} T={T_TAIL} (generated)\n")
        for p in verts:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in faces + caps:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def generate(name: str, seed: int = 0) -> dict:
    difficulty, clearance, max_actions = VARIANTS[name]
    k = twist_for_clearance(S, R_TUBE, clearance)
    P = resample_by_sagitta(S, k, T_TAIL, MAX_SAGITTA)
    sag = polyline_sagitta(P, S, k, T_TAIL)
    L = loop_polyline(k)
    x, dmin, lk = search_start(P, L, seed=seed)
    q = quat_normalize(quat_from_rotvec(x[:3]))
    disc = loop_disc(k)
    (HERE / "assets").mkdir(exist_ok=True)
    sweep_obj(HERE / "assets" / f"{name}-tube.obj", k)
    pts = [[round(float(v), 7) for v in p] for p in P]
    spec = {
        "name": name,
        "difficulty": difficulty,
        "description": (
            "Two identical alpha-shaped tubes are interlocked: each loop threads the other. "
            "Tube A is fixed; separate tube B from it."
        ),
        "max_actions": max_actions,
        "unit_scale": 1.0,
        "gravity": [0.0, 0.0, 0.0],
        "floor": False,
        "frame": [{"name": "tube_a", "geoms": [{"type": "tube", "tube_radius": R_TUBE, "points": pts}]}],
        "objects": [
            {
                "name": "tube_b",
                "pos": [round(float(v), 7) for v in x[3:]],
                "quat": [round(float(v), 9) for v in q],
                "graspable": True,
                "mass": 0.2,
                "damping": 0.5,
                "rgba": [0.2, 0.5, 0.9, 1.0],
                "geoms": [{"type": "tube", "tube_radius": R_TUBE, "points": pts}],
            }
        ],
        "holes": [
            {"name": "a", "body": "world", **disc},
            {"name": "b", "body": "tube_b", **disc},
        ],
        "target": "tube_b",
        "goal": {"type": "separation", "d_sep": D_SEP},
        "roadmap": f"roadmap-{name}.npz",
        "config": PUZZLE_CONFIG,
        "meta": {
            "generator": "loophole.puzzles.alpha.generate",
            "centerline": "S*(t^2-1, t^3-t, k*t), t in [-T, T]",
            "S": S,
            "tube_radius": R_TUBE,
            "T": T_TAIL,
            "k": round(k, 9),
            "nominal_gap_clearance": clearance,
            "polyline_points": len(P),
            "decomposition": "capsule chain along the polyline (exact convex pieces of the swept sphere)",
            "decomposition_error_sagitta": round(sag, 7),
            "start_min_surface_distance": round(dmin, 6),
            "start_loop_linking_number": round(lk, 4),
            "start_search_seed": seed,
            "mesh": f"assets/{name}-tube.obj",
        },
    }
    out = HERE / f"{name}.yaml"
    header = (
        f"# GENERATED by `python -m loophole.puzzles.alpha.generate` - do not edit by hand.\n"
        f"# Alpha-style puzzle ({difficulty}): our own procedural geometry, inspired by the Alpha\n"
        f"# puzzle (B. Yamrom; Parasol Lab / N. Amato; OMPL.app benchmark). No third-party meshes.\n"
    )
    out.write_text(header + yaml.safe_dump(spec, sort_keys=False, width=200))
    return spec["meta"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(VARIANTS), default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in [a.only] if a.only else VARIANTS:
        meta = generate(name, a.seed)
        print(name, json.dumps({k: meta[k] for k in ("nominal_gap_clearance", "decomposition_error_sagitta", "start_min_surface_distance", "start_loop_linking_number", "polyline_points")}))


if __name__ == "__main__":
    main()
