#!/usr/bin/env python3
"""Conversion fidelity of the recorded Rat Trap solve (diagnostics; the physics is unchanged).

    PYTHONPATH=$PWD .venv/bin/python scripts/rattrap_conversion.py --mode exact|closed|open [--cap 0.03] [--rdp 0.0005] --out X.npz

exact:  the recorded handle pose before every physics step (the reference run). Stores the cord nodes every
        EVERY physics steps (`ref_steps`, `ref_nodes`) and at every grab / release boundary.
closed: teleop.convert_closed_loop (each RDP waypoint approached from where the handle actually is, moves capped at
        --cap). Stores, after every waypoint, the recorded step it came from, the handle, the number of moves and
        whether the last one blocked, and the cord nodes (`wp_*`); plus every boundary.
open:   teleop.to_commands (capped straight moves, open loop); boundaries and the end state only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.reward import FrozenEscape  # noqa: E402
from loophole.puzzles.rattrap.sim import RatTrapSim  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import convert_closed_loop, load_records, replay_exact, to_commands  # noqa: E402
from loophole.puzzles.rattrap.topology import build_discs, read_word  # noqa: E402

REC = ROOT / "loophole" / "reference" / "rattrap-solve.jsonl"
EVERY = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact", "open", "closed"], required=True)
    ap.add_argument("--rdp", type=float, default=0.0005)
    ap.add_argument("--cap", type=float, default=None, help="rig_max_linear_step for the command split (m); default the YAML's")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = load_spec()
    cspec = load_spec(overrides={"rig_max_linear_step": a.cap}) if a.cap else spec
    t0 = time.time()
    recs = load_records(REC)
    sim = RatTrapSim(cspec)  # the cap only changes how commands are split; everything else is the YAML's
    boundary, ref_steps, ref_nodes = [], [], []
    wp = {"s": [], "h": [], "n": [], "blocked": [], "nodes": []}
    grab, release = sim.grab_node, sim.release

    def g(h, k):  # a boundary is the state at the moment of the grab / release, before it acts
        boundary.append(sim.node_positions())
        return grab(h, k)

    def r(h):
        boundary.append(sim.node_positions())
        return release(h)

    sim.grab_node, sim.release = g, r
    cmds = []
    if a.mode == "exact":
        inner, count = sim._substep, [0]

        def sub():
            inner()
            count[0] += 1
            if count[0] % EVERY == 0:
                ref_steps.append(count[0])
                ref_nodes.append(sim.node_positions().astype(np.float32))

        sim._substep = sub
        sim.rig.substep = sub
        replay_exact(recs, spec, sim)
    elif a.mode == "open":
        cmds = to_commands(recs, cspec, tolerance=a.rdp)
        for tool, args in cmds:
            if tool == "grab_node":
                sim.grab_node(args["handle"], args["node"])
            elif tool == "release":
                sim.release(args["handle"])
            else:
                sim.move(args["handle"], [args["dx"], args["dy"], args["dz"]])
    else:
        def on_wp(s, h, n, blocked):
            for k, v in (("s", s), ("h", h), ("n", n), ("blocked", blocked)):
                wp[k].append(v)
            wp["nodes"].append(sim.node_positions().astype(np.float32))

        cmds, _ = convert_closed_loop(recs, cspec, tolerance=a.rdp, sim=sim, on_waypoint=on_wp)
    nodes = sim.node_positions()
    free = FrozenEscape(spec, sim).escape_direction(nodes) is not None
    out = {"mode": a.mode, "rdp": a.rdp, "cap": a.cap or float(spec["rig_max_linear_step"]), "free": free,
           "word": read_word(nodes, build_discs(spec), closed=True), "commands": len(cmds),
           "moves": sum(1 for t, _ in cmds if t == "move"), "boundaries": len(boundary),
           "blocked_waypoints": int(sum(wp["blocked"])), "max_penetration": sim.max_penetration, "seconds": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def arr(x, shape):
        return np.array(x) if len(x) else np.zeros(shape)

    np.savez_compressed(a.out, final_nodes=nodes, boundary_nodes=np.array(boundary), ref_steps=np.array(ref_steps, int),
                        ref_nodes=arr(ref_nodes, (0, len(nodes), 3)), wp_s=np.array(wp["s"], int), wp_h=np.array(wp["h"], int),
                        wp_n=np.array(wp["n"], int), wp_blocked=np.array(wp["blocked"], bool), wp_nodes=arr(wp["nodes"], (0, len(nodes), 3)),
                        meta=json.dumps(out))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
