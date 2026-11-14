#!/usr/bin/env python3
"""Side-by-side numbers: canonical OMPL.app Alpha meshes (local only, see fetch_alpha.sh)
vs our generated Alpha-style tubes. Canonical files are read, never modified; their
native units are converted with a single unit_scale (default 0.01) for comparison."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from loophole.sim.spec import load_puzzle, resolve_puzzle

ROOT = Path(__file__).resolve().parents[1]


def stats(mesh):
    ext = mesh.bounds[1] - mesh.bounds[0]
    return {"extent": np.round(ext, 4).tolist(), "volume": round(float(abs(mesh.volume)), 7), "faces": len(mesh.faces)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-scale", type=float, default=0.01)
    a = ap.parse_args()
    d = ROOT / ".ompl_alpha"
    if not d.exists():
        raise SystemExit("run `make fetch-alpha` first (local only)")
    for f in ["alpha_robot.dae", "alpha_env-1.0.dae", "alpha_env-1.2.dae", "alpha_env-1.5.dae"]:
        m = trimesh.load(d / f, force="mesh")
        m.apply_scale(a.unit_scale)
        print(f"OMPL {f:22s}", stats(m))
    for v in ("alpha-easy", "alpha-medium", "alpha-hard"):
        p = load_puzzle(resolve_puzzle(v))
        m = trimesh.load(Path(p.source_path).parent / p.meta["mesh"], force="mesh")
        print(f"ours {v:22s}", stats(m), "gap clearance", p.meta["nominal_gap_clearance"])


if __name__ == "__main__":
    main()
