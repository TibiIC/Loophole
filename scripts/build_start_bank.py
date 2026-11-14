#!/usr/bin/env python3
"""Build the certified start bank and the training roadmap for Alpha 1.0.

    .venv/bin/python scripts/build_start_bank.py --per-tie 200 --budget 90 --workers 8

For each tie (canonical: linking number -1; flipped: +1), sample interlocked starts
(loophole.training.starts, `interlocked` mode: collision-free, |Lk| = 1, not yet free, never
near the canonical start) and the turned-over canonical start, then try to connect each to the
evaluation roadmap with loophole.training.bank.connect_to_roadmap (time budget per start).
Certified starts and their corridors go into
  loophole/puzzles/alpha_paper/start-bank-alpha-1.0-paper.npz    (starts, tie, Lk, ctg)
  loophole/puzzles/alpha_paper/roadmap-alpha-1.0-paper-train.npz (evaluation roadmap + corridors)
  loophole/puzzles/alpha_paper/start-bank-alpha-1.0-paper.json   (per-tie success rates, timings)
Uncertified starts are kept in the report (a planner failure is not a proof of unsolvability).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAME = "alpha-1.0-paper"
_W = {}


def _init():
    from loophole.config import load_config
    from loophole.sim.env import LoopholeEnv
    from loophole.training.starts import StartSampler

    env = LoopholeEnv(NAME, config=load_config(ROOT / "loophole" / "configs" / "alpha_training.yaml"))
    _W["env"] = env
    _W["sampler"] = StartSampler(env, env.target_pose().copy(), mode="interlocked")


def _work(job):
    from loophole.training.bank import connect_to_roadmap, densify, shortcut_path

    k, start, budget, seed = job
    env, s = _W["env"], _W["sampler"]
    rng = np.random.default_rng(seed)
    lo = env.frame_aabb()[0] - 1.0
    hi = env.frame_aabb()[1] + 1.0
    r = connect_to_roadmap(env.checker, env.roadmap, np.asarray(start), rng, lo, hi, time_budget=budget)
    out = {"k": k, "certified": r.path is not None, "seconds": round(r.seconds, 1), "tree": r.tree_size, "node": r.node}
    if r.path is not None:
        w, res = float(env.roadmap.w), float(env.roadmap.resolution)
        p = shortcut_path(env.checker, r.path, w, res, rng)
        out["corridor"] = densify(p, w).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tie", type=int, default=200)
    ap.add_argument("--budget", type=float, default=90.0, help="planner seconds per start")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    from loophole.config import load_config
    from loophole.puzzles import paper_alpha as pa
    from loophole.sim.env import LoopholeEnv
    from loophole.training.bank import BANK_FILE, TRAIN_ROADMAP_FILE, merge_corridors
    from loophole.training.starts import StartSampler

    t0 = time.time()
    env = LoopholeEnv(NAME, config=load_config(ROOT / "loophole" / "configs" / "alpha_training.yaml"))
    s = StartSampler(env, env.target_pose().copy(), mode="interlocked")
    rng = np.random.default_rng(a.seed)
    starts, ties, lks = [], [], []
    for tie in ("canonical", "flipped"):
        n = 0
        while n < a.per_tie:
            q, info = s.sample(rng, mode="interlocked", chirality=tie)
            starts.append(q), ties.append(tie), lks.append(info["linking_number"])
            n += 1
    starts.append(s.flipped_start()), ties.append("flipped"), lks.append(round(s.linking(s.flipped_start()), 4))
    print(f"sampled {len(starts)} starts ({time.time() - t0:.0f}s); planning with {a.workers} workers, {a.budget:.0f}s each", flush=True)
    jobs = [(k, q.tolist(), a.budget, a.seed * 100000 + k) for k, q in enumerate(starts)]
    results = [None] * len(jobs)
    with Pool(a.workers, initializer=_init) as pool:
        for n, r in enumerate(pool.imap_unordered(_work, jobs)):
            results[r["k"]] = r
            if n % 10 == 0:
                done = [x for x in results if x is not None]
                print(f"  {n + 1}/{len(jobs)} done, certified {sum(x['certified'] for x in done)} ({time.time() - t0:.0f}s)", flush=True)
    ok = [k for k, r in enumerate(results) if r["certified"]]
    corridors = [np.asarray(results[k]["corridor"]) for k in ok]
    targets = [int(results[k]["node"]) for k in ok]
    rm = merge_corridors(env.roadmap, corridors, targets, pa.PUZZLE_OUT / TRAIN_ROADMAP_FILE.format(name=NAME),
                         n_corridors=len(ok))
    ctg = np.array([rm.cost_to_go(np.asarray(starts[k]), env.checker)[0] for k in ok], dtype=float)
    np.savez_compressed(pa.PUZZLE_OUT / BANK_FILE.format(name=NAME), starts=np.array([starts[k] for k in ok]),
                        tie=np.array([ties[k] for k in ok]), linking=np.array([lks[k] for k in ok]), ctg=ctg,
                        roadmap_sha256=rm.sha256)
    report = {"seconds": round(time.time() - t0, 1), "budget_s": a.budget, "per_tie": a.per_tie,
              "training_roadmap_nodes": int(len(rm.nodes)), "training_roadmap_sha256": rm.sha256}
    for tie in ("canonical", "flipped"):
        ks = [k for k in range(len(starts)) if ties[k] == tie]
        cert = [k for k in ks if results[k]["certified"]]
        report[tie] = {"sampled": len(ks), "certified": len(cert),
                       "median_seconds_certified": float(np.median([results[k]["seconds"] for k in cert])) if cert else None,
                       "ctg_range": [float(np.nanmin(ctg[[ok.index(k) for k in cert]])), float(np.nanmax(ctg[[ok.index(k) for k in cert]]))] if cert else None}
    k_flip = len(starts) - 1
    report["turned_over_canonical_start_certified"] = results[k_flip]["certified"]
    (pa.PUZZLE_OUT / BANK_FILE.format(name=NAME).replace(".npz", ".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
