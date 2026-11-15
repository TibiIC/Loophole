"""Alpha-puzzle score and literature-based training signals.

Roles (strictly separated):

* ``episode_score(final)`` is THE score: one float in [0, 1], the only value the
  Harbor verifier writes to /logs/verifier/reward.txt.
* ``step()`` rewards of the components below are training signals only (RL /
  distillation traces). They never enter the verifier score.

    violation during the episode -> 0.0  (the episode also ends immediately)
    solved                       -> 1 - 0.5 * a / T      a = actions used, T = max actions
    unsolved                     -> 0.3 * p,   p = clip(1 - best_ctg / start_ctg, 0, 1)

``best_ctg`` is the lowest roadmap cost-to-go *held*: a level counts only when the
matched roadmap node is reachable from the current pose by a collision-free straight
segment and the match holds for >= 3 consecutive actions (reward.components.StableMin).
Time is measured in actions, never wall-clock. Solves land in [0.5, 1] and non-solves
in [0, 0.3], so every solve beats every non-solve.

Components
----------
1. CSpaceTunnelPotentialReward (primary): potential-based shaping with
   Phi(q) = -ctg(q), gamma = 1, on the roadmap navigation function; its terminal()
   returns exactly episode_score().
2. ClearanceShapingReward (training only; weight 0 by default): potential-based
   clearance kernel minus penetration and action-norm penalties.
3. GaussLinkingDiagnostic (log only; weight 0, no reward, no score).

References (each checked against the DOI / arXiv / publisher listing; see REFERENCES.md)
------------------------------------------------------------------------------------------
[ZBKV20] X. Zhang, R. Belfer, P. G. Kry, E. Vouga. C-Space Tunnel Discovery for Puzzle
         Path Planning. ACM Trans. Graph. 39(4), Article 104 (SIGGRAPH 2020).
         doi:10.1145/3386569.3392468
[NHR99]  A. Y. Ng, D. Harada, S. J. Russell. Policy Invariance under Reward
         Transformations: Theory and Application to Reward Shaping. ICML 1999, pp. 278-287.
[JSCP15] L. Janson, E. Schmerling, A. Clark, M. Pavone. Fast Marching Tree: a Fast
         Marching Sampling-Based Method for Optimal Motion Planning in Many Dimensions.
         Int. J. Robotics Research 34(7):883-921, 2015. doi:10.1177/0278364915577958
[TLA+23] B. Tang, M. A. Lin, I. Akinola, A. Handa, G. S. Sukhatme, F. Ramos, D. Fox,
         Y. Narang. IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation
         to Reality. Robotics: Science and Systems (RSS) 2023. arXiv:2305.17110
[NSA+22] Y. Narang, K. Storey, I. Akinola, M. Macklin, P. Reist, L. Wawrzyniak, Y. Guo,
         A. Moravanszky, G. State, M. Lu, A. Handa, D. Fox. Factory: Fast Contact for
         Robotic Assembly. RSS 2022. arXiv:2205.03532
[KL00]   K. Klenin, J. Langowski. Computation of writhe in modeling of supercoiled DNA.
         Biopolymers 54(5):307-317, 2000.
         doi:10.1002/1097-0282(20001015)54:5<307::AID-BIP20>3.0.CO;2-Y
[RN11]   R. L. Ricca, B. Nipoti. Gauss' linking number revisited. J. Knot Theory and Its
         Ramifications 20(10):1325-1343, 2011. doi:10.1142/S0218216511009261
[PS02]   J. H. Przytycki, A. S. Sikora. Topological Insights from the Chinese Rings.
         Proc. Amer. Math. Soc. 130(3):893-902, 2002. arXiv:math/0007134
"""

from __future__ import annotations

import math
from typing import Any

from .components import PotentialShaping, RewardComponent, RewardState, register_component

ALPHA = 0.5  # time penalty weight for solves
PROGRESS_CAP = 0.3  # maximum score of a non-solve (must stay below 1 - ALPHA)


def episode_progress(final: RewardState) -> float:
    """p = clip(1 - best_ctg / start_ctg, 0, 1) with the held (StableMin) best cost-to-go."""
    if not final.start_cost_to_go or final.stable_best_cost_to_go is None:
        return 0.0
    return min(1.0, max(0.0, 1.0 - final.stable_best_cost_to_go / final.start_cost_to_go))


def episode_score(final: RewardState, alpha: float = ALPHA, progress_cap: float = PROGRESS_CAP) -> float:
    """The only value the verifier writes to reward.txt; in [0, 1] by construction."""
    if final.violations > 0:
        return 0.0
    if final.success:
        return 1.0 - alpha * min(final.actions_used, final.max_actions) / max(final.max_actions, 1)
    return progress_cap * episode_progress(final)


# ==============================================================================
# 1. C-space tunnel potential (primary component)
# ==============================================================================
class CSpaceTunnelPotentialReward(PotentialShaping):
    """Potential-based shaping on the C-space navigation function, Phi(q) = -ctg(q).

    The Alpha puzzle's way out is a narrow tunnel in the 6-D configuration space
    [ZBKV20]; Euclidean closeness is deceptive across tube walls. ctg(q) is the
    roadmap estimate of the shortest collision-free path length (cost-to-go) from q to
    the separated set, matched only through collision-free-visible roadmap nodes.
    We use cost-to-go rather than distance to one reference path because the Alpha
    1.5 puzzle presents two homotopy classes of solution paths [JSCP15, Sec. 6]; a
    navigation function credits progress along any of them.

    step:     F = lam * (gamma * Phi_bar(q') - Phi(q)), Phi = -ctg / ctg(q0), gamma = 1, lam = 1, through
              PotentialShaping: Phi_bar = 0 once the episode has terminated (solved or a violation), Phi kept at
              truncation. Potential-based shaping preserves optimal policies [NHR99] and telescopes, so
              oscillating earns nothing.
    terminal: exactly episode_score(final) (never the shaping).
    """

    name = "cspace_tunnel"

    def __init__(self, gamma: float = 1.0, alpha: float = ALPHA, progress_cap: float = PROGRESS_CAP, lam: float = 1.0, **kw: Any):
        super().__init__(gamma=gamma, lam=lam, alpha=alpha, progress_cap=progress_cap, **kw)
        if not (0 <= alpha < 1 and 0 <= progress_cap < 1 - alpha):
            raise ValueError("need 0 <= alpha < 1 and progress_cap < 1 - alpha (every solve beats every non-solve)")
        self.alpha = float(alpha)
        self.progress_cap = float(progress_cap)
        self._c0 = 1.0

    def reset(self, initial: RewardState) -> None:
        self._c0 = initial.start_cost_to_go if initial.start_cost_to_go and initial.start_cost_to_go > 1e-9 else 1.0

    def phi(self, s: RewardState) -> float | None:
        return None if s.cost_to_go is None else -float(s.cost_to_go) / self._c0

    def terminal(self, final: RewardState, info: dict) -> float:
        return episode_score(final, self.alpha, self.progress_cap)


# ==============================================================================
# 2. Clearance shaping (training only)
# ==============================================================================
class ClearanceShapingReward(PotentialShaping):
    """Potential-based clearance shaping, adapted from IndustReal's SDF-based dense reward.

    Inspired by [TLA+23] (built on [NSA+22]). Note the difference: IndustReal's SDF
    reward measures alignment of a part with its *goal pose* (points sampled on the
    plug, evaluated in the SDF of the same plug mesh at its target pose; r = -log of
    the mean SDF value). We adapt the idea to a separation task, where the relevant
    distance is the *clearance between the two tubes*:

        Phi(s) = exp(-max(0, d_sep - d(s)) / sigma)        d(s): min distance A-B
        r      = lam * (gamma * Phi_bar(s') - Phi(s)) - w_pen * max(0, pen - pen_thr) - w_act * |a|^2
        (the first term through PotentialShaping: Phi_bar = 0 after termination, Phi kept at truncation)
        (|a|: the executed handle motion - translation^2 + 0.1 * rotation^2 - not the raw request)

    Potential-based, so doing nothing (or returning to the same pose) earns 0 from the
    kernel; stalling no longer accumulates reward. Caveat: clearance can still grow
    while the tubes remain interlocked, so this is a weak signal for this puzzle;
    weight 0 in the default config (logged only).
    """

    name = "clearance_shaping"

    def __init__(self, d_sep: float = 0.4, sigma_dist: float = 0.08, gamma: float = 1.0, w_penetration: float = 50.0,
                 w_action: float = 0.05, penetration_threshold: float = 0.002, lam: float = 1.0, **kw: Any):
        super().__init__(gamma=gamma, lam=lam, d_sep=d_sep, sigma_dist=sigma_dist, w_penetration=w_penetration,
                         w_action=w_action, penetration_threshold=penetration_threshold, **kw)
        self.d_sep = float(d_sep)
        self.sigma = float(sigma_dist)
        self.w_pen = float(w_penetration)
        self.w_act = float(w_action)
        self.pen_thr = float(penetration_threshold)

    def phi(self, s: RewardState) -> float:
        if s.min_distance is None:
            return 0.0
        return math.exp(-max(0.0, self.d_sep - float(s.min_distance)) / self.sigma)

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        shaping = super().step(prev, cur, info)
        pen = max(0.0, float(info.get("last_penetration", 0.0) or 0.0) - self.pen_thr)
        ex = info.get("executed") or {}  # the motion the rig actually executed (clamped), never raw requests
        act = float(ex.get("translation", 0.0)) ** 2 + 0.1 * float(ex.get("rotation", 0.0)) ** 2
        return float(shaping - self.w_pen * pen - self.w_act * act)

    def terminal(self, final: RewardState, info: dict) -> float:
        return 0.0


# ==============================================================================
# 3. Gauss linking diagnostic (log only)
# ==============================================================================
class GaussLinkingDiagnostic(RewardComponent):
    """Open-curve Gauss linking integral of the two tube centre-lines: logged, never rewarded.

    Computed with the exact segment-pair formula of [KL00] (see also [RN11]) and
    logged per step next to the cost-to-go for analysis plots. For open curves the
    integral is real-valued and changes when a piece merely rotates in place: open arcs
    carry no linking invariant, and what traps the open tubes is rigidity and thickness.
    The environment also logs `linking_closed`, the integer linking number of the two
    loops each closed across its own gap: -1 at the start, 0 once a tube has passed the
    other's gap (docs/rewards.md section 13). Neither is rewarded or scored. (For
    topological models of string puzzles see [PS02], which concerns the Chinese Rings,
    not the Alpha puzzle.)
    """

    name = "gauss_linking_diagnostic"

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self.last: float | None = None

    def reset(self, initial: RewardState) -> None:
        self.last = None

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        diag = info.get("diagnostics") or {}
        self.last = diag.get("linking_open")
        return 0.0

    def terminal(self, final: RewardState, info: dict) -> float:
        return 0.0


# Backwards-compatible names
IndustRealSDFReward = ClearanceShapingReward
GaussLinkingPotentialReward = GaussLinkingDiagnostic

register_component("cspace_tunnel", CSpaceTunnelPotentialReward)
register_component("clearance_shaping", ClearanceShapingReward)
register_component("gauss_linking_diagnostic", GaussLinkingDiagnostic)
