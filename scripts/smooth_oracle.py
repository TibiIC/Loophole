#!/usr/bin/env python3
"""Shortcut-smooth the Alpha 1.0 oracle in our own simulator (rotation first), then re-convert to rig moves.

    .venv/bin/python scripts/smooth_oracle.py [--write]

Seed: the authors' 22-pose solution (Zhang et al. 2020; alpha-1.0-paper-reference.npz).
1. Densify the path (0.02 SE(3)-metric spacing).
2. Rotation-first shortcutting: rank all pairs (i, j) by the rotation a straight shortcut would save
   (rotation summed along the path minus the direct rotation from i to j); try the largest savings first;
   accept a shortcut only if the straight SE(3) segment is collision-free against our exact hulls
   (CollisionChecker.segment_valid, <= 2 mm motion of any point between checks, margin 0).
3. The same on the full metric (translation + w * rotation), then repeat both until nothing improves.
4. Final dense validation of the smoothed path at 0.8 mm (as for the published path).
5. Convert to rig moves under the agent's caps (0.1 m / 0.2 rad per move; safety factor reported) and
   replay with the task config (free-escape goal): success, violations, blocked moves, actions, score,
   and the path length integrated over physics substeps.

Reported before / after: path length (translation m, rotation rad, L), moves, actions to free, score.
--write replaces the oracle trajectory (loophole/reference/trajectories/alpha-1.0-paper.jsonl) and the
task solutions; without it nothing is written.
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

from loophole.config import load_config  # noqa: E402
from loophole.planning.commands import path_to_commands  # noqa: E402
from loophole.puzzles import paper_alpha as pa  # noqa: E402
from loophole.reference.replay import replay  # noqa: E402
from loophole.sim.collision import interpolate, pose_metric  # noqa: E402
from loophole.sim.env import LoopholeEnv  # noqa: E402
from loophole.sim.geometry import quat_conj, quat_mul, quat_to_axis_angle  # noqa: E402

NAME = "alpha-1.0-paper"
CFG = ROOT / "loophole" / "configs" / "alpha_training.yaml"


def ang(a, b) -> float:
    return abs(float(quat_to_axis_angle(quat_mul(quat_conj(a[3:]), b[3:]))[1]))


def stats(path, w) -> dict:
    t = sum(float(np.linalg.norm(b[:3] - a[:3])) for a, b in zip(path[:-1], path[1:]))
    r = sum(ang(a, b) for a, b in zip(path[:-1], path[1:]))
    return {"translation_m": round(t, 4), "rotation_rad": round(r, 4), "L": round(t + w * r, 4), "waypoints": len(path)}


def densify(path, w, spacing):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        m = max(1, int(np.ceil(float(pose_metric(a, b[None], w)[0]) / spacing)))
        out += [interpolate(a, b, k / m) for k in range(1, m + 1)]
    return out


def _window_valid(ck, poses, w, res) -> bool:
    return all(ck.segment_valid(x, y, w, res) for x, y in zip(poses[:-1], poses[1:]))


def shortcut_pass(ck, path, w, res, kind: str, tries: int, rng, log) -> tuple[list, int]:
    """Random shortcuts at all scales. kind:
      rotation    partial shortcut: keep the translations of path[i..j], replace the rotations by the direct
                  slerp from q_i to q_j (parameterised by translational arc length): removes rotations that cancel
      translation partial shortcut: keep the rotations, replace the translations by the straight line t_i -> t_j
      full        straight SE(3) segment from path[i] to path[j]
    A candidate is accepted only if every resulting segment is collision-free at `res`."""
    from loophole.sim.geometry import quat_slerp

    P = [np.asarray(p) for p in path]
    accepted, saved = 0, 0.0
    for _ in range(tries):
        n = len(P)
        if n < 3:
            break
        span = int(np.clip(np.exp(rng.uniform(np.log(2), np.log(n - 1))), 2, n - 1))
        i = int(rng.integers(0, n - span))
        j = i + span
        win = P[i:j + 1]
        if kind == "rotation":
            seg = np.array([np.linalg.norm(b[:3] - a[:3]) for a, b in zip(win[:-1], win[1:])])
            s_ = np.concatenate([[0.0], np.cumsum(seg)])
            f = s_ / s_[-1] if s_[-1] > 1e-12 else np.linspace(0, 1, len(win))
            new = [np.concatenate([win[k][:3], quat_slerp(win[0][3:], win[-1][3:], float(f[k]))]) for k in range(len(win))]
            before = sum(ang(a, b) for a, b in zip(win[:-1], win[1:]))
            after = sum(ang(a, b) for a, b in zip(new[:-1], new[1:]))
            gain = w * (before - after)
        elif kind == "translation":
            r = [ang(a, b) for a, b in zip(win[:-1], win[1:])]
            s_ = np.concatenate([[0.0], np.cumsum(r)])
            f = s_ / s_[-1] if s_[-1] > 1e-12 else np.linspace(0, 1, len(win))
            new = [np.concatenate([win[0][:3] + f[k] * (win[-1][:3] - win[0][:3]), win[k][3:]]) for k in range(len(win))]
            before = sum(np.linalg.norm(b[:3] - a[:3]) for a, b in zip(win[:-1], win[1:]))
            after = float(np.linalg.norm(win[-1][:3] - win[0][:3]))
            gain = before - after
        else:
            new = [win[0], win[-1]]
            gain = sum(float(pose_metric(a, b[None], w)[0]) for a, b in zip(win[:-1], win[1:])) - float(pose_metric(win[0], win[-1][None], w)[0])
        if gain < 1e-4 or not _window_valid(ck, new, w, res):
            continue
        P = P[:i] + new + P[j + 1:]
        accepted += 1
        saved += gain
    log(f"  {kind}: {accepted} of {tries} accepted, saving {saved:.3f} (metric)")
    return P, accepted


def evaluate(path, safety: float, label: str) -> dict:
    env = LoopholeEnv(NAME, config=load_config(CFG))
    cmds = path_to_commands([np.asarray(p) for p in path], env.puzzle.target, env.cfg.rig.max_linear_step,
                            env.cfg.rig.max_angular_step, safety=safety)
    acc = {"t": 0.0, "r": 0.0}
    prev = [env.target_pose().copy()]
    inner = env._substep

    def substep():
        inner()
        cur = env.target_pose()
        acc["t"] += float(np.linalg.norm(cur[:3] - prev[0][:3]))
        acc["r"] += ang(prev[0], cur)
        prev[0] = cur.copy()

    env._substep = substep
    env.rig.substep = substep
    blocked = {"n": 0}
    final = replay(env, cmds, on_step=lambda i, t, a, r: blocked.__setitem__("n", blocked["n"] + bool((r or {}).get("blocked"))))
    w = float(env.roadmap.w)
    out = {"label": label, "safety": safety, "moves_in_plan": len(cmds) - 1, "actions_to_free": final["actions_used"],
           "success": final["success"], "violations": final["violations"]["count"], "blocked": blocked["n"],
           "score": final["score"], "executed_translation_m": round(acc["t"], 4), "executed_rotation_rad": round(acc["r"], 4),
           "executed_L": round(acc["t"] + w * acc["r"], 4)}
    env.close()
    return out, cmds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--res", type=float, default=0.001, help="shortcut validation: max point motion between checks (m)")
    ap.add_argument("--final-res", type=float, default=0.0008)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tries", type=int, default=600, help="shortcut attempts per kind per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.time()
    env = LoopholeEnv(NAME, config=load_config(CFG))
    ck, w = env.checker, float(env.roadmap.w)
    ref = np.load(pa.PUZZLE_OUT / f"{NAME}-reference.npz")
    seed = [np.asarray(p) for p in ref["path"]]
    report = {"seed_path": stats(seed, w)}
    log(f"seed path: {report['seed_path']}")

    path = seed
    rng = np.random.default_rng(a.seed)
    for rnd in range(a.rounds):
        dense = densify(path, w, a.spacing)
        changed = 0
        for kind in ("rotation", "translation", "full"):  # rotation first
            dense, c = shortcut_pass(ck, dense, w, a.res, kind, a.tries, rng, log)
            changed += c
        path = dense
        st = stats(path, w)
        log(f"round {rnd + 1}: {st} ({time.time() - t0:.0f}s)")
        if not changed:
            break
    # drop collinear-ish intermediate waypoints that shortcutting left in place (straight segments only)
    simp = [path[0]]
    for k in range(1, len(path) - 1):
        if not ck.segment_valid(simp[-1], path[k + 1], w, a.final_res):
            simp.append(path[k])
    simp.append(path[-1])
    path = simp
    report["smoothed_path"] = stats(path, w)
    log(f"smoothed: {report['smoothed_path']} ({time.time() - t0:.0f}s)")
    bad = 0
    for x, y in zip(path[:-1], path[1:]):  # final dense validation
        if not ck.segment_valid(x, y, w, a.final_res):
            bad += 1
    report["final_validation"] = {"resolution_m": a.final_res, "invalid_segments": bad}
    log(f"final validation at {a.final_res * 1000:.1f} mm: {bad} invalid segments")
    if bad:
        sys.exit("smoothed path failed the dense validation")

    rows = []
    for label, p, safety in (("published path, current conversion", seed, 0.95), ("published path, re-timed to the caps", seed, 0.999),
                             ("smoothed path, re-timed to the caps", path, 0.999)):
        r, cmds = evaluate(p, safety, label)
        rows.append(r)
        log(json.dumps(r))
        if label.startswith("smoothed"):
            smoothed_cmds = cmds
    report["replays"] = rows
    report["seconds"] = round(time.time() - t0, 1)
    out = pa.PUZZLE_OUT / f"{NAME}-smoothed.json"
    out.write_text(json.dumps(report, indent=1))
    np.savez(pa.PUZZLE_OUT / f"{NAME}-smoothed.npz", path=np.asarray(path), seed=np.asarray(seed), w=w)
    log(f"wrote {out}")
    if a.write:
        from loophole.reference.recorder import write_trajectory

        best = rows[-1]
        if not (best["success"] and best["violations"] == 0):
            sys.exit("smoothed oracle did not solve cleanly; not writing")
        hdr = json.loads((ROOT / "loophole/reference/trajectories/alpha-1.0-paper.jsonl").open().readline())
        write_trajectory(ROOT / "loophole/reference/trajectories/alpha-1.0-paper.jsonl", NAME, smoothed_cmds,
                         source="Zhang et al. 2020 supplementary path, shortcut-smoothed in our simulator (scripts/smooth_oracle.py)",
                         meta={"expected_score": best["score"], "expected_actions": best["actions_to_free"],
                               "config": "loophole/configs/alpha_training.yaml", "published_path_score": rows[0]["score"],
                               "supp_sha256": hdr.get("supp_sha256"), "attribution": hdr.get("attribution")})
        log("oracle trajectory replaced; run `python scripts/sync_tasks.py --alpha` to update the task solutions")


if __name__ == "__main__":
    main()
