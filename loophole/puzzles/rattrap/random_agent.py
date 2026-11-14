"""Random-action Rat Trap agent: the adversarial probe of the partial credit (what a judge would try first).

Each action, with the agent's own interface and limits: if no gripper holds the cord, grab a uniformly random node with
a random gripper; otherwise move a holding gripper by a uniform random step in [-30, 30] mm per axis (80 %),
release it (10 %), or grab a random node with the other gripper (10 %). Runs until the episode ends (solved, a
violation, or the action budget T).
"""

from __future__ import annotations

import numpy as np

from .reward import RatTrapEpisode
from .spec import load_spec


def run(seed: int, spec=None, max_actions: int | None = None) -> dict:
    spec = spec or load_spec()
    ep = RatTrapEpisode(spec)
    rng = np.random.default_rng(seed)
    held: dict[int, int | None] = {0: None, 1: None}
    cap = float(spec["rig_max_linear_step"])
    blocked = 0
    limit = max_actions or ep.T
    while not ep.done and ep.actions < limit:
        holding = [h for h, k in held.items() if k is not None]
        r = rng.random()
        if not holding or (r < 0.1 and len(holding) < 2):
            h = int(rng.integers(2)) if not holding else 1 - holding[0]
            k = int(rng.integers(ep.sim.n_nodes))
            ep.act("grab_node", {"handle": h, "node": k})
            held[h] = k
        elif r < 0.2:
            h = holding[int(rng.integers(len(holding)))]
            ep.act("release", {"handle": h})
            held[h] = None
        else:
            h = holding[int(rng.integers(len(holding)))]
            d = rng.uniform(-cap, cap, 3)
            out = ep.act("move", {"handle": h, "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2])})
            blocked += bool(out.get("blocked"))
    return {"seed": seed, "actions": ep.actions, "T": ep.T, "solved": ep.solved, "violation": ep.violation, "score": ep.score(),
            "p": ep.progress(), "checkpoints_reached": None if ep.tracker is None else ep.tracker.k, "blocked_moves": blocked}
