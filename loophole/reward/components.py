"""Modular reward.

A :class:`RewardComponent` sees immutable :class:`RewardState` snapshots taken
by the environment after every action (never anything the agent reports).
Components produce a dense per-step reward (`step`, logged for RL training)
and a terminal value; the *episode score* used by the Harbor verifier is the
weighted sum of terminal values, clipped to [0, 1].

Invariants kept by the default weights: any solve scores >= 1 - alpha (0.5);
any non-solve scores at most the summed terminal weight of the progress /
shaping components (0.2 for the rope puzzles, 0.3 for the Alpha puzzles), so a
solve always beats a non-solve; doing nothing scores 0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..config import RewardConfig
from .topology import get_goal_distance


@dataclass(frozen=True)
class RewardState:
    puzzle: str
    actions_used: int
    max_actions: int
    success: bool
    violations: int
    done: bool
    topology_word: str = ""
    state_hash: str = ""
    cost_to_go: float | None = None  # roadmap cost-to-go of the current pose (None: unavailable)
    start_cost_to_go: float | None = None
    best_cost_to_go: float | None = None  # minimum reached so far in the episode
    stable_best_cost_to_go: float | None = None  # minimum level held for `hold` consecutive visible matches
    min_distance: float | None = None  # min distance target-frame (separation puzzles), for clearance shaping
    geometry: Mapping[str, Any] | None = None  # named measured geometry (aperture centres / axes, cord nodes) for shaping

    @property
    def terminated(self) -> bool:
        """The episode ended inside the MDP: solved, or failed by a violation (an absorbing state)."""
        return bool(self.success or self.violations > 0)

    @property
    def truncated(self) -> bool:
        """The episode was cut off from outside (action budget, or the agent / harness stopped): not absorbing."""
        return bool(self.done and not self.terminated)


class RewardComponent(ABC):
    name = "component"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def reset(self, initial: RewardState) -> None: ...

    @abstractmethod
    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float: ...

    @abstractmethod
    def terminal(self, final: RewardState, info: dict) -> float: ...


class PotentialShaping(RewardComponent):
    """Potential-based shaping, the single code path for every potential component:

        F(s, s') = lam * (gamma * Phi_bar(s') - Phi(s)),   Phi_bar(s') = 0 if s' is terminal, else Phi(s')

    Terminal (absorbing) states, i.e. solved or failed by a violation, get potential 0 [NHR99]: their value is 0
    after the transition, whatever Phi would say about them. A time-limit truncation (the action budget, or an
    external stop) is not absorbing, so Phi(s') is kept there and a learner that bootstraps V(s') at the cut-off
    sees the same shaped returns as without the limit. Then, for any gamma, the discounted shaping return of an
    episode is gamma^T Phi_bar(s_T) - Phi(s_0): it depends only on where the episode starts and ends, so loops,
    oscillation and standing still gain nothing over each other.

    `gamma` must equal the learner's discount for the policy-invariance guarantee. With gamma < 1 and Phi < 0,
    standing still earns (1 - gamma) |Phi| per step in an *undiscounted* sum; only the discounted sum is invariant.

    Shaping is a training signal only: `terminal()` is 0, and an episode score never depends on gamma, lam or
    Phi_bar (scores are computed from the final state by the scoring components, as for Alpha).

    Subclasses define `phi(state) -> float | None` (None: unavailable at this state, so no shaping this step).

    Provenance: the gamma / lam scaffold and the terminal-vs-truncation rule are adapted from a colleague's
    ring-on-wire disentanglement environment (puzzle-handoff-v1, `DisentanglementPuzzle.step` / `potential`;
    shared with us, no licence stated; REFERENCES.md [HO26]). Their defaults: lam = 1, gamma = 0.99.
    """

    name = "potential_shaping"

    def __init__(self, gamma: float = 0.99, lam: float = 1.0, **kw):
        super().__init__(gamma=gamma, lam=lam, **kw)
        if not (0.0 < gamma <= 1.0 and lam >= 0.0):
            raise ValueError("need 0 < gamma <= 1 and lam >= 0")
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.last_potentials: tuple[float | None, float | None] = (None, None)

    def phi(self, s: RewardState) -> float | None:
        raise NotImplementedError

    def reset(self, initial: RewardState) -> None:
        pass

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        before = self.phi(prev)
        after = 0.0 if cur.terminated else self.phi(cur)
        self.last_potentials = (before, after)
        if before is None or after is None:
            return 0.0
        return float(self.lam * (self.gamma * after - before))

    def terminal(self, final: RewardState, info: dict) -> float:
        return 0.0


class BaseReward(RewardComponent):
    """score = success * (1 - alpha * actions_used / max_actions).

    Success requires the goal condition *and* zero anti-tunnelling violations.
    Doing nothing, stopping early or failing scores exactly 0.
    """

    name = "base"

    def __init__(self, alpha: float = 0.5, **kw):
        super().__init__(alpha=alpha, **kw)
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = alpha

    def reset(self, initial: RewardState) -> None:
        pass

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        return 0.0

    def terminal(self, final: RewardState, info: dict) -> float:
        if not final.success or final.violations > 0:
            return 0.0
        frac = min(final.actions_used, final.max_actions) / max(final.max_actions, 1)
        return 1.0 - self.alpha * frac


class TopologyShaping(PotentialShaping):
    """Potential-based shaping (PotentialShaping) with Phi = -goal_distance / d0.

    Normalised by the initial goal distance d0, so the shaping return lies in [-1, 1]; with gamma = 1 it
    telescopes, so oscillating across a hole earns nothing. Its terminal value (which counts towards the score
    when the component is in the score) is the topological progress (d0 - d_T) / d0 of the final state, and 0
    after a violation; it does not depend on gamma or on the terminal rule of the step shaping. (Before the
    PotentialShaping refactor it was the undiscounted sum of the step rewards: equal for gamma = 1 except after a
    violation, which used to keep the progress made before it.)
    """

    name = "topology_shaping"

    def __init__(self, gamma: float = 1.0, lam: float = 1.0, **kw):
        super().__init__(gamma=gamma, lam=lam, **kw)
        self._d0 = 1
        self._start = 0.0

    def _dist(self, s: RewardState) -> float:
        return float(get_goal_distance(s.puzzle)(s.topology_word))

    def phi(self, s: RewardState) -> float:
        return -self._dist(s) / self._d0

    def reset(self, initial: RewardState) -> None:
        self._start = self._dist(initial)
        self._d0 = max(1, int(self._start))

    def terminal(self, final: RewardState, info: dict) -> float:
        if final.violations > 0:
            return 0.0
        return (self._start - self._dist(final)) / self._d0  # the gamma = 1 telescoped progress, from the endpoints


class CostToGoShaping(PotentialShaping):
    """Dense potential-based shaping on the roadmap navigation function (via PotentialShaping).

    Phi(q) = -cost_to_go(q) / ctg(start). With gamma = 1 it telescopes, so going back and forth earns nothing.
    Its terminal value is 0: it shapes per-step rewards (training signal) but the episode score's progress
    credit comes from `progress`.
    """

    name = "cost_to_go_shaping"

    def __init__(self, gamma: float = 1.0, lam: float = 1.0, **kw):
        super().__init__(gamma=gamma, lam=lam, **kw)
        self._c0 = 1.0

    def reset(self, initial: RewardState) -> None:
        self._c0 = initial.start_cost_to_go or 1.0

    def phi(self, s: RewardState) -> float | None:
        return None if s.cost_to_go is None else -float(s.cost_to_go) / self._c0


class ProgressCredit(RewardComponent):
    """Partial credit for unsolved episodes: cap * (1 - best_ctg / ctg(start)).

    Uses the minimum cost-to-go *reached* (oscillating cannot raise it further)
    and is 0 for solved episodes (the base reward dominates) or on violations.
    """

    name = "progress"

    def __init__(self, cap: float = 0.3, **kw):
        super().__init__(cap=cap, **kw)
        self.cap = cap

    def reset(self, initial: RewardState) -> None:
        pass

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        return 0.0

    def terminal(self, final: RewardState, info: dict) -> float:
        if final.success or final.violations > 0 or not final.start_cost_to_go or final.best_cost_to_go is None:
            return 0.0
        return self.cap * max(0.0, 1.0 - final.best_cost_to_go / final.start_cost_to_go)


class StableMin:
    """Best (lowest) cost-to-go level that was *held*: a level L counts only once
    `hold` consecutive actions all had a collision-free-visible roadmap match with
    cost-to-go <= L. An action without a visible match breaks the streak, so a single
    nearest-neighbour glitch cannot spike the progress."""

    def __init__(self, start: float | None, hold: int = 3):
        self.hold = hold
        self.window: list[float] = []
        self.best = start

    def update(self, ctg: float | None) -> float | None:
        if ctg is None:
            self.window.clear()
            return self.best
        self.window = (self.window + [float(ctg)])[-self.hold :]
        if len(self.window) == self.hold:
            level = max(self.window)
            self.best = level if self.best is None else min(self.best, level)
        return self.best


class ScaledProgressScore(RewardComponent):
    """Official episode score in [0, 1] with partial credit for unsolved runs.

        solved:   1 - alpha * actions_used / max_actions          in [1 - alpha, 1]
        unsolved: progress_cap * p,  p = 1 - stable_best_ctg / start_ctg   in [0, progress_cap]
        any anti-tunnelling violation: 0

    With alpha = 0.5 and progress_cap = 0.3, solves land in [0.5, 1] and non-solves in
    [0, 0.3], so any solve beats any non-solve. p uses the roadmap's visibility-guarded
    cost-to-go (shortest collision-free route to separation) and only levels held for
    3 consecutive actions (see StableMin). Doing nothing scores 0. Time = actions.
    """

    name = "scaled_progress"

    def __init__(self, alpha: float = 0.5, progress_cap: float = 0.3, **kw):
        super().__init__(alpha=alpha, progress_cap=progress_cap, **kw)
        if not (0 <= alpha < 1 and 0 <= progress_cap < 1 - alpha):
            raise ValueError("need 0 <= alpha < 1 and progress_cap < 1 - alpha (solves must beat non-solves)")
        self.alpha = float(alpha)
        self.cap = float(progress_cap)

    def reset(self, initial: RewardState) -> None:
        pass

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        return 0.0

    @staticmethod
    def progress(s: RewardState) -> float:
        if not s.start_cost_to_go or s.stable_best_cost_to_go is None:
            return 0.0
        return min(1.0, max(0.0, 1.0 - s.stable_best_cost_to_go / s.start_cost_to_go))

    def terminal(self, final: RewardState, info: dict) -> float:
        if final.violations > 0:
            return 0.0
        if final.success:
            return 1.0 - self.alpha * min(final.actions_used, final.max_actions) / max(final.max_actions, 1)
        return self.cap * self.progress(final)


REGISTRY: dict[str, type[RewardComponent]] = {
    "base": BaseReward,
    "topology_shaping": TopologyShaping,
    "cost_to_go_shaping": CostToGoShaping,
    "progress": ProgressCredit,
    "scaled_progress": ScaledProgressScore,
}


def register_component(name: str, cls: type[RewardComponent]) -> None:
    REGISTRY[name] = cls


@dataclass
class RewardComposer:
    entries: list[tuple[float, RewardComponent]]
    in_score: list[bool] | None = None  # per entry; None = all count towards the score
    last_step: dict[str, float] = field(default_factory=dict)  # per-component rewards of the last step
    step_totals: list[float] = field(default_factory=list)
    terminal_values: list[float] | None = None
    _prev: RewardState | None = None

    @classmethod
    def from_config(cls, cfg: RewardConfig) -> "RewardComposer":
        if any(c.enabled and c.type not in REGISTRY for c in cfg.components):
            from . import alignment, literature_rewards  # noqa: F401  (register their components)
        entries, in_score = [], []
        for c in cfg.components:
            if c.enabled:
                entries.append((c.weight, REGISTRY[c.type](**c.params)))
                in_score.append(c.in_score)
        if cfg.roadmap and cfg.auto_roadmap_components:  # roadmap reward: dense cost-to-go shaping + capped progress credit
            names = {comp.name for _, comp in entries}
            if "cost_to_go_shaping" not in names:
                entries.append((1.0, CostToGoShaping(gamma=1.0)))
                in_score.append(True)
            if "progress" not in names:
                entries.append((1.0, ProgressCredit(cap=0.3)))
                in_score.append(True)
        return cls(entries, in_score)

    def reset(self, initial: RewardState) -> None:
        for _, comp in self.entries:
            comp.reset(initial)
        self.step_totals = [0.0] * len(self.entries)
        self.terminal_values = None
        self._prev = initial

    def step(self, cur: RewardState, info: dict | None = None) -> float:
        info = info or {}
        r = 0.0
        self.last_step = {}
        for i, (w, comp) in enumerate(self.entries):
            v = comp.step(self._prev, cur, info)
            self.step_totals[i] += v
            self.last_step[comp.name] = round(float(v), 6)
            r += w * v
        self._prev = cur
        return r

    def terminal(self, final: RewardState, info: dict | None = None) -> float:
        info = info or {}
        self.terminal_values = [comp.terminal(final, info) for _, comp in self.entries]
        return sum(w * v for (w, _), v in zip(self.entries, self.terminal_values))

    def breakdown(self) -> dict:
        out = {}
        for i, (w, comp) in enumerate(self.entries):
            term = self.terminal_values[i] if self.terminal_values is not None else None
            out[comp.name] = {"weight": w, "step_sum": round(self.step_totals[i], 6), "terminal": term,
                              "in_score": True if self.in_score is None else self.in_score[i]}
        return out

    def score(self) -> float:
        """Episode score in [0, 1]: weighted terminal values (requires terminal())."""
        assert self.terminal_values is not None, "call terminal() first"
        mask = self.in_score or [True] * len(self.entries)
        total = sum(w * t for (w, _), t, m in zip(self.entries, self.terminal_values, mask) if m)
        return float(min(1.0, max(0.0, total)))
