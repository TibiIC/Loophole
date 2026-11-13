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
    if method == "coacd":
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
