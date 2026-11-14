#!/usr/bin/env python3
"""Build the Rat Trap reference: the faithful command conversion of the recorded solve, its checkpoints and eps.

    PYTHONPATH=$PWD .venv/bin/python scripts/build_rattrap_reference.py [--exact runs/rattrap-conversion/exact.npz] [--rdp 0.0001]

1. Exact replay of loophole/reference/rattrap-solve.jsonl (or a saved one, --exact): the cord at every grab / release.
2. Closed-loop conversion (teleop.convert_closed_loop) at --rdp, the faithful setting found by
   scripts/rattrap_conversion.py + rattrap_divergence.py; written to loophole/reference/rattrap-solve.commands.jsonl.
3. The commands run through RatTrapEpisode, the agent interface, until solved: the oracle's action count, and the cord
   after every action (the matcher's evaluation times).
4. Checkpoints: the exact replay's boundary states reached before the oracle is free, thinned to min separation 2 eps
   (and away from the start). eps = safety * (the smallest eps at which the oracle registers every checkpoint in
   order); iterated, since the thinning depends on eps. Written to loophole/reference/rattrap-checkpoints.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.puzzles.rattrap.checkpoints import DEFAULT_FILE, Checkpoints, calibrate_eps, max_dev, select_checkpoints  # noqa: E402
from loophole.puzzles.rattrap.reward import RatTrapEpisode  # noqa: E402
from loophole.puzzles.rattrap.sim import RatTrapSim  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import convert_closed_loop, load_records, replay_exact, save_commands  # noqa: E402

REF = ROOT / "loophole" / "reference"
REC = REF / "rattrap-solve.jsonl"
CMDS = REF / "rattrap-solve.commands.jsonl"


def exact_boundaries(spec) -> tuple[np.ndarray, list[np.ndarray]]:
    sim = RatTrapSim(spec)
    start = sim.node_positions()
    out = []
    grab, release = sim.grab_node, sim.release
    sim.grab_node = lambda h, k: (out.append(sim.node_positions()), grab(h, k))[1]
    sim.release = lambda h: (out.append(sim.node_positions()), release(h))[1]
    replay_exact(load_records(REC), spec, sim)
    return start, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", type=Path, default=None, help="a saved exact replay (rattrap_conversion.py --mode exact)")
    ap.add_argument("--rdp", type=float, default=0.0001)
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--T-factor", type=float, default=3.0, help="T = ceil(factor * oracle actions)")
    a = ap.parse_args()
    spec = load_spec()
    start = RatTrapSim(spec).node_positions()
    if a.exact:
        z = np.load(a.exact)
        bounds = list(z["boundary_nodes"])
    else:
        start, bounds = exact_boundaries(spec)

    cmds, sim = convert_closed_loop(load_records(REC), spec, tolerance=a.rdp)
    ep = RatTrapEpisode(load_spec(overrides={"max_actions": 100000}), checkpoints=Checkpoints(start, np.zeros((0,) + start.shape), 1.0))
    traj, boundaries_done = [], 0
    for tool, args in cmds:
        if tool != "move":
            boundaries_done += 1
        ep.act(tool, args)
        traj.append(ep.sim.node_positions())
        if ep.done:
            break
    if not ep.solved:
        sys.exit(f"the converted solve does not solve under the agent interface (violation={ep.violation})")
    n_oracle = ep.actions
    save_commands(cmds[:n_oracle], CMDS, {"source": REC.name, "conversion": "convert_closed_loop", "rdp_tolerance_m": a.rdp,
                                          "cap_m": float(spec["rig_max_linear_step"]), "oracle_actions": n_oracle})
    cands = bounds[:boundaries_done]
    eps, sep, hist = None, 0.0, []
    for _ in range(6):
        idx = select_checkpoints(start, cands, sep)
        cps = [cands[i] for i in idx]
        e_min = calibrate_eps(traj, cps)
        new = a.safety * e_min
        hist.append({"min_separation_m": sep, "K": len(cps), "eps_min_m": e_min, "eps_m": new})
        if eps is not None and abs(new - eps) < 1e-6 and sep >= 2 * new - 1e-9:
            break
        eps, sep = new, 2 * new
    idx = select_checkpoints(start, cands, 2 * eps)
    cps = np.array([cands[i] for i in idx])
    e_min = calibrate_eps(traj, list(cps))
    T = int(np.ceil(a.T_factor * n_oracle))
    meta = {"source": REC.name, "boundaries_before_free": boundaries_done, "kept_boundary_indices": idx, "K": len(cps),
            "eps_min_m": e_min, "safety": a.safety, "eps_m": eps, "min_separation_m": 2 * eps, "oracle_actions": n_oracle,
            "T_recommended": T, "oracle_score_at_T": 1 - 0.5 * n_oracle / T, "start_to_c1_m": max_dev(start, cps[0]),
            "calibration_history": hist}
    Checkpoints(start=start, states=cps, eps=eps, meta=meta).save(DEFAULT_FILE)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
