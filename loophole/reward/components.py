"""Modular reward.

A :class:`RewardComponent` sees immutable :class:`RewardState` snapshots taken
by the environment after every action (never anything the agent reports). The
:class:`RewardComposer` sums weighted components and clips the episode score
to [0, 1] for the Harbor verifier.

Invariant kept by the default weights: any solve scores >= 1 - alpha (0.5) and
any non-solve scores <= the summed weight of shaping components (0.2), so a
solve always beats a non-solve.
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

    def _phi(self, s: RewardState) -> float:
        return -float(get_goal_distance(s.puzzle)(s.topology_word))

    def reset(self, initial: RewardState) -> None:
        self._d0 = max(1, int(get_goal_distance(initial.puzzle)(initial.topology_word)))

    def step(self, prev: RewardState, cur: RewardState, info: dict) -> float:
        if cur.violations > 0:
            return 0.0
        return (self.gamma * self._phi(cur) - self._phi(prev)) / self._d0

    def terminal(self, final: RewardState, info: dict) -> float:
        return 0.0


REGISTRY: dict[str, type[RewardComponent]] = {"base": BaseReward, "topology_shaping": TopologyShaping}


def register_component(name: str, cls: type[RewardComponent]) -> None:
    REGISTRY[name] = cls


@dataclass
class RewardComposer:
    entries: list[tuple[float, RewardComponent]]
    step_totals: list[float] = field(default_factory=list)
    terminal_values: list[float] | None = None
    _prev: RewardState | None = None

    @classmethod
    def from_config(cls, cfg: RewardConfig) -> "RewardComposer":
        entries = []
        for c in cfg.components:
            if c.enabled:
                entries.append((c.weight, REGISTRY[c.type](**c.params)))
        return cls(entries)

    def reset(self, initial: RewardState) -> None:
        for _, comp in self.entries:
            comp.reset(initial)
        self.step_totals = [0.0] * len(self.entries)
        self.terminal_values = None
        self._prev = initial

    def step(self, cur: RewardState, info: dict | None = None) -> float:
        info = info or {}
        r = 0.0
        for i, (w, comp) in enumerate(self.entries):
            v = comp.step(self._prev, cur, info)
            self.step_totals[i] += v
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
            out[comp.name] = {"weight": w, "step_sum": round(self.step_totals[i], 6), "terminal": term}
        return out

    def score(self) -> float:
        """Episode score in [0, 1] (requires terminal() to have been called)."""
        assert self.terminal_values is not None, "call terminal() first"
        total = sum(w * (s + t) for (w, _), s, t in zip(self.entries, self.step_totals, self.terminal_values))
        return float(min(1.0, max(0.0, total)))
