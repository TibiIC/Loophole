"""Progress potential Phi(state) for reward shaping.

MVP: returns 0 (contributes nothing) unless the work-in-progress cost-to-go roadmap
reward is switched on with `reward.roadmap: true` and the puzzle ships a roadmap
(loophole/puzzles/alpha/roadmap-<variant>.npz, built by scripts/build_alpha.py).
Then Phi = -cost_to_go(relative pose of B w.r.t. A), the shortest collision-free
path length to the separated set estimated through visible roadmap nodes
(loophole/reward/roadmap.py), and shaping is gamma * Phi(s') - Phi(s).
"""

from __future__ import annotations


def progress_potential(env) -> float:
    roadmap = getattr(env, "roadmap", None)
    if roadmap is None:  # flag off, no roadmap for this puzzle, or not a separation puzzle
        return 0.0
    ctg, _ = roadmap.cost_to_go(env.target_pose(), env.checker)
    return 0.0 if ctg is None else -float(ctg)
