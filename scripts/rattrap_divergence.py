#!/usr/bin/env python3
"""Where a converted Rat Trap run leaves the recorded solve: per-waypoint max node displacement from the exact replay,
the first step over a threshold and its nearest grab / release event, and blocked-move agreement with the recording.

    .venv/bin/python scripts/rattrap_divergence.py runs/rattrap-conversion/exact.npz runs/rattrap-conversion/closed-*.npz [--plot out.png]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.puzzles.rattrap.teleop import load_records  # noqa: E402

REC = ROOT / "loophole" / "reference" / "rattrap-solve.jsonl"


def recorded_blocks(recs, vstep: float) -> list[tuple[int, int]]:
    """(step, handle) of every blocked-and-snapped motion in the recording: the handle jumps (back onto the node)
    by more than the rig's per-step speed cap."""
    last, out = {}, []
    for r in recs:
        if "p" not in r:
            if r.get("type") == "grab":
                last.pop(r["handle"], None)
            continue
        p = np.asarray(r["p"], float)
        h = r["h"]
        if h in last and r["s"] == last[h][0] + 1 and np.linalg.norm(p - last[h][1]) > 1.5 * vstep:
            out.append((r["s"], h))
        last[h] = (r["s"], p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exact", type=Path)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.002, 0.005, 0.01])
    ap.add_argument("--plot", type=Path, default=None)
    a = ap.parse_args()
    spec = load_spec()
    recs = load_records(REC)
    events = [(r["s"], r["type"], r["handle"]) for r in recs if r.get("type") in ("grab", "release")]
    ev_steps = np.array([e[0] for e in events])
    blocks = recorded_blocks(recs, float(spec["rig_linear_speed"]) * float(spec["timestep"]))
    E = np.load(a.exact)
    ref_s, ref_n = E["ref_steps"], E["ref_nodes"]
    results, curves = [], []
    for f in a.runs:
        Z = np.load(f)
        meta = json.loads(str(Z["meta"]))
        s, nodes = Z["wp_s"], Z["wp_nodes"]
        if not len(s):
            continue
        idx = np.clip(np.searchsorted(ref_s, s), 0, len(ref_s) - 1)
        dev = np.max(np.linalg.norm(nodes - ref_n[idx], axis=2), axis=1)
        bdev = np.max(np.linalg.norm(Z["boundary_nodes"] - E["boundary_nodes"], axis=2), axis=1)
        first = {}
        for th in a.thresholds:
            k = np.flatnonzero(dev > th)
            if len(k):
                st = int(s[k[0]])
                j = int(np.argmin(np.abs(ev_steps - st)))
                prev_ev = int(np.searchsorted(ev_steps, st)) - 1
                first[f"{th * 1000:g}mm"] = {"waypoint": int(k[0]), "step": st, "nearest_event": {"step": int(ev_steps[j]), "type": events[j][1], "index": j},
                                             "steps_since_last_event": st - int(ev_steps[prev_ev]) if prev_ev >= 0 else None,
                                             "recorded_block_within_200_steps": any(abs(b - st) <= 200 for b, _ in blocks)}
            else:
                first[f"{th * 1000:g}mm"] = None
        conv_blocks = s[Z["wp_blocked"]]
        # blocked agreement: a converted block "matches" a recorded block of the same stretch within 300 steps
        matched = sum(any(abs(int(c) - b) <= 300 for b, _ in blocks) for c in conv_blocks)
        results.append({"run": f.stem, **{k: meta[k] for k in ("cap", "rdp", "free", "word", "commands", "moves")},
                        "final_max_dev_mm": round(float(np.max(np.linalg.norm(Z["final_nodes"] - E["final_nodes"], axis=1))) * 1000, 1),
                        "max_boundary_dev_mm": round(float(bdev.max()) * 1000, 1), "first_over": first,
                        "blocked": {"recorded": len(blocks), "converted": int(len(conv_blocks)), "converted_matching_a_recorded_block": int(matched)}})
        curves.append((f.stem, s, dev))
    print(json.dumps(results, indent=1))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        for name, s, dev in curves:
            ax[0].semilogy(s / 1000, np.maximum(dev, 1e-5) * 1000, label=name, lw=0.8)
        for e in ev_steps:
            ax[0].axvline(e / 1000, color="0.85", lw=0.4, zorder=0)
        ax[0].set_xlabel("recorded time (s); grey: grab / release"); ax[0].set_ylabel("max node deviation (mm)"); ax[0].legend(fontsize=7)
        for r in results:
            ax[1].scatter(r["moves"], max(r["final_max_dev_mm"], 1e-2), marker="o" if r["free"] else "x", s=50)
            ax[1].annotate(f"cap {r['cap'] * 1000:g} mm, rdp {r['rdp'] * 1000:g} mm", (r["moves"], max(r["final_max_dev_mm"], 1e-2)), fontsize=7)
        ax[1].set_yscale("log"); ax[1].set_xlabel("moves in the converted solve"); ax[1].set_ylabel("final max node deviation (mm); o = free, x = not")
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)


if __name__ == "__main__":
    main()
