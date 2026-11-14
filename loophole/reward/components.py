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


class TopologyShaping(RewardComponent):
    """Potential-based shaping F = gamma * Phi(s') - Phi(s) with Phi = -goal_distance.

    Normalised by the initial goal distance so the shaping total lies in
    [-1, 1]; with gamma = 1 it telescopes to (d0 - d_T) / d0, so oscillating
    across a hole earns nothing. (gamma < 1 with a negative potential would
    *reward* standing still, so keep gamma = 1 unless you shift Phi.)
    """

    name = "topology_shaping"

    def __init__(self, gamma: float = 1.0, **kw):
        super().__init__(gamma=gamma, **kw)
        self.gamma = gamma
        self._d0 = 1
        self._sum = 0.0

    def _phi(self, s: RewardState) -> float:
        return -float(get_goal_distance(s.puzzle)(s.topology_word))

    def reset(self, initial: RewardState) -> None:
        self._d0 = max(1, int(get_goal_distance(initial.puzzle)(initial.topology_word)))
        self._sum = 0.0

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        if cur.violations > 0:
            return 0.0
        r = (self.gamma * self._phi(cur) - self._phi(prev)) / self._d0
        self._sum += r
        return r

    def terminal(self, final: RewardState, info: dict) -> float:
        return self._sum  # the telescoped shaping counts towards the episode score


class CostToGoShaping(RewardComponent):
    """Dense potential-based shaping on the roadmap navigation function.

    Phi(q) = -cost_to_go(q); step reward (gamma * Phi(q') - Phi(q)) / ctg(start).
    With gamma = 1 it telescopes, so going back and forth earns nothing. Its
    terminal value is 0: it shapes per-step rewards (training signal) but the
    episode score's progress credit comes from `progress`.
    """

    name = "cost_to_go_shaping"

    def __init__(self, gamma: float = 1.0, **kw):
        super().__init__(gamma=gamma, **kw)
        self.gamma = gamma

    def reset(self, initial: RewardState) -> None:
        self._c0 = initial.start_cost_to_go or 1.0

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        if cur.violations > 0 or cur.cost_to_go is None or prev.cost_to_go is None:
            return 0.0
        return (prev.cost_to_go - self.gamma * cur.cost_to_go) / self._c0

    def terminal(self, final: RewardState, info: dict) -> float:
        return 0.0


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


REGISTRY: dict[str, type[RewardComponent]] = {
    "base": BaseReward,
    "topology_shaping": TopologyShaping,
    "cost_to_go_shaping": CostToGoShaping,
    "progress": ProgressCredit,
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
            from . import literature_rewards  # noqa: F401  (registers its components)
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
