"""Mesh import with automatic convex decomposition.

MuJoCo collides every mesh geom through its *convex hull*, so a ring or a frame
with holes imported as one mesh behaves as a solid lump. ``decompose_mesh``
splits a mesh into approximately convex parts with CoACD; each part becomes its
own geom, which keeps the holes open. Results are cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("LOOPHOLE_MESH_CACHE", Path(__file__).resolve().parents[2] / ".mesh_cache"))


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def decompose_mesh(
    path: str | Path,
    scale=(1.0, 1.0, 1.0),
    method: str = "coacd",
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    seed: int = 0,
) -> list[Path]:
    """Return a list of OBJ files (one per convex part) for the given mesh.

    method: ``coacd`` (convex decomposition), ``convex``/``none`` (single part:
    MuJoCo will use its convex hull, only correct for genuinely convex parts).
    """
    import trimesh  # optional dependency (extra "mesh")

    path = Path(path).resolve()
    raw = path.read_bytes()
    key = hashlib.sha256(
        raw + json.dumps([list(scale), method, threshold, max_convex_hull, seed]).encode()
    ).hexdigest()[:16]
    out_dir = CACHE_DIR / f"{path.stem}-{key}"
    manifest = out_dir / "parts.json"
    if manifest.exists():
        return [out_dir / p for p in json.loads(manifest.read_text())]

    mesh = trimesh.load(path, force="mesh")
    mesh.apply_scale(np.asarray(scale, dtype=float))
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[tuple[np.ndarray, np.ndarray]]
    if method == "facewedge":
        parts = facewedge_decomposition(mesh)
    elif method == "coacd":
        import coacd

        coacd.set_log_level("error")
        cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_convex_hull, seed=seed)
    else:
        parts = [(np.asarray(mesh.vertices), np.asarray(mesh.faces))]
    names = []
    for i, (v, f) in enumerate(parts):
        name = f"part_{i:03d}.obj"
        _write_obj(out_dir / name, np.asarray(v), np.asarray(f))
        names.append(name)
    manifest.write_text(json.dumps(names))
    return [out_dir / n for n in names]


def facewedge_decomposition(mesh, depth_frac: float = 0.8) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact-outside convex decomposition of a thin *tube-like* surface (e.g. a triangle soup).

    Each (deduplicated) triangle becomes the convex wedge spanned by the triangle and
    its copy pushed inward by `depth_frac` of the local half-thickness. The outer
    surface of the union is exactly the original triangles (zero outward error), so
    clearances are preserved; pieces are deep enough that passing through one needs
    far more penetration than the audit allows.

    Inward direction and thickness come from ray casting along +-normal: the inward
    ray hits the tube's opposite wall first (about one tube diameter away), which
    needs no consistent face orientation (triangle soups from SketchUp are double-sided).
    """
    import trimesh

    m = mesh.copy()
    m.merge_vertices(digits_vertex=6)
    F = np.unique(np.sort(np.asarray(m.faces), axis=1), axis=0)
    V = np.asarray(m.vertices)
    F = F[np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) > 1e-9]
    clean = trimesh.Trimesh(V, F, process=False)
    tri = V[F]
    cen = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    eps = 1e-4 * float(np.linalg.norm(clean.extents))
    first = {}
    for sgn in (1.0, -1.0):
        origins = cen + sgn * eps * n
        dirs = sgn * n
        loc, ray_idx, _ = clean.ray.intersects_location(origins, dirs, multiple_hits=True)
        dist = np.full(len(F), np.inf)
        if len(loc):
            dd = np.linalg.norm(loc - origins[ray_idx], axis=1)
            for r, d in zip(ray_idx, dd):
                if d > 2 * eps and d < dist[r]:
                    dist[r] = d
        first[sgn] = dist
    inward_pos = first[1.0] < first[-1.0]
    thick = np.where(inward_pos, first[1.0], first[-1.0])
    if not np.all(np.isfinite(thick)):
        bad = ~np.isfinite(thick)
        thick[bad] = np.median(thick[~bad])
    d = np.where(inward_pos[:, None], n, -n)
    faces = np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]])
    parts = []
    for k in range(len(F)):
        depth = depth_frac * 0.5 * thick[k]  # half the wall-to-wall distance ~ to the axis
        parts.append((np.concatenate([tri[k], tri[k] + depth * d[k]]), faces))
    return parts
