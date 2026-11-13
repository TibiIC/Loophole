"""Anti-tunnelling audit, run after every physics substep.

A violation is recorded when
* any contact penetrates deeper than ``penetration_threshold`` (something is
  being forced through something else), or
* a rope end drifts from its anchor by more than ``anchor_drift_threshold``
  (the closing ``connect`` constraint was overpowered: the rope "broke"), or
* any rope segment / free object moves more than ``max_substep_displacement``
  in one substep (fast enough to skip through a thin part between checks), or
* the integrator went unstable (MuJoCo then silently resets the state, which
  must never be usable as a "teleport").

Any violation makes the episode unsuccessful (see reward/base.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import AuditConfig
from .builder import BuiltModel


@dataclass
class Violation:
    step: int
    kind: str
    detail: str
    value: float

    def to_dict(self) -> dict:
        return {"step": self.step, "kind": self.kind, "detail": self.detail, "value": round(float(self.value), 6)}


@dataclass
class Auditor:
    model: mujoco.MjModel
    built: BuiltModel
    cfg: AuditConfig
    violations: list[Violation] = field(default_factory=list)
    count: int = 0
    max_penetration: float = 0.0
    last_penetration: float = 0.0  # current-substep values, read by the rig's stall check
    last_anchor_drift: float = 0.0

    def __post_init__(self):
        r = self.built.rope
        self._end_body = self.model.body(r.body_names[-1]).id
        self._end_local = np.array([r.seg_lengths[-1], 0.0, 0.0])
        self._end_anchor = None if r.end is None else np.asarray(r.end, dtype=float)
        self._moving = np.array([self.model.body(b).id for b in r.body_names + self.built.object_bodies])
        self._prev_pos = None
        self._bad = None

    def _instability_count(self, data: mujoco.MjData) -> int:
        return int(sum(data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQPOS)))

    def reset(self) -> None:
        self.violations.clear()
        self.count = 0
        self.max_penetration = 0.0
        self._prev_pos = None
        self._bad = None

    def prime(self, data: mujoco.MjData) -> None:
        """Start a fresh audit window from the current state (no gap after reset)."""
        self._prev_pos = data.xpos[self._moving].copy()
        self._bad = self._instability_count(data)

    def _record(self, v: Violation) -> None:
        self.count += 1
        if len(self.violations) < self.cfg.max_records:
            self.violations.append(v)

    def check(self, data: mujoco.MjData, step: int) -> int:
        """Audit the current state; returns the number of new violations."""
        before = self.count
        bad = self._instability_count(data)
        if self._bad is not None and bad > self._bad:
            self._record(Violation(step, "instability", "integrator diverged (state was reset by MuJoCo)", float(bad - self._bad)))
        self._bad = bad
        n = data.ncon
        self.last_penetration = 0.0
        if n:
            dist = data.contact.dist[:n]
            pen = -float(dist.min())
            self.last_penetration = pen
            self.max_penetration = max(self.max_penetration, pen)
            if pen > self.cfg.penetration_threshold:
                i = int(np.argmin(dist))
                c = data.contact[i]
                g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                self._record(Violation(step, "penetration", f"{g1} <-> {g2}", pen))
        if self._end_anchor is not None:
            end = data.xpos[self._end_body] + data.xmat[self._end_body].reshape(3, 3) @ self._end_local
            drift = float(np.linalg.norm(end - self._end_anchor))
            self.last_anchor_drift = drift
            if drift > self.cfg.anchor_drift_threshold:
                self._record(Violation(step, "anchor_drift", "rope end left its anchor", drift))
        pos = data.xpos[self._moving]
        if self._prev_pos is not None:
            disp = np.linalg.norm(pos - self._prev_pos, axis=1)
            k = int(np.argmax(disp))
            if disp[k] > self.cfg.max_substep_displacement:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self._moving[k]))
                self._record(Violation(step, "speed", f"{name} moved {disp[k]:.4f} m in one substep", float(disp[k])))
        self._prev_pos = pos.copy()
        return self.count - before

    def inject(self, step: int, detail: str = "injected for testing") -> None:
        """Test hook (never exposed over MCP)."""
        self._record(Violation(step, "injected", detail, 1.0))

    def summary(self) -> dict:
        return {
            "count": self.count,
            "max_penetration": round(self.max_penetration, 6),
            "records": [v.to_dict() for v in self.violations],
        }
