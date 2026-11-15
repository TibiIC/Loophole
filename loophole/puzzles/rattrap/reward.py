"""Rat Trap reward: ordered checkpoint matching against the reference solve, and the frozen-shape free-escape test.

Score (verifier side only; one float in [0, 1]):
    a violation (any contact pressed deeper than penetration_tolerance) -> 0, the episode ends
    solved (the frozen cord assembly can slide to infinity along one of `escape_directions` directions)
                                                           -> 1 - alpha * a / T          in [1 - alpha, 1]
    unsolved                                               -> cap * p,   p = k_reached / K
with alpha = 0.5 and cap = 0.3 < 1 - alpha, so any solve beats any non-solve. p counts the checkpoints of the reference
solve reached in order (checkpoints.py: max per-node displacement < eps, strictly increasing times).
Training shaping (never scored; kept apart from the score exactly as for Alpha): every potential goes through
loophole.reward.components.PotentialShaping, lam * (gamma * Phi_bar(s') - Phi(s)) with Phi_bar = 0 once the episode has
terminated (solved, or a violation) and Phi kept at the action-budget truncation (potential-based: Ng, Harada &
Russell, ICML 1999; the terminal / truncation rule after REFERENCES.md [HO26]). The potential is the checkpoint one,
Phi = -[(K - k) + min(1, d(x, c_{k+1}) / D_{k+1})]. Optional extra potentials: AperturePassageAlignment terms for the
ring and the two holes (rattrap.yaml `shaping_alignment`; weight 0 = logged only).
Diagnostics only (logged, never scored or shaped; UNSOUND as a progress measure, see docs/RATTRAP_DESIGN.md): the word
w of signed disc crossings (topology.py) and its Przytycki-Sikora norm ||w|| (norm.py).
Anti-tunnelling: the per-substep penetration audit, plus (rattrap.yaml `certify_motion`) the swept-motion
certificate of certify.CordMotionCertifier: "log" records uncertified substeps, "enforce" also makes them violations.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ...reward.alignment import AperturePassageAlignment
from ...reward.components import PotentialShaping, RewardState
from ...sim.collision import fibonacci_directions
from ...sim.geometry import frame_from_x, quat_mul
from .build import build
from .checkpoints import DEFAULT_FILE, Checkpoints, CheckpointTracker
from .norm import INF, norm
from .sim import RatTrapSim
from .spec import RatTrapSpec
from .topology import build_discs, read_word


_Z_TO_X = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90 degrees about y: local z -> x


def passable_generators(spec: RatTrapSpec) -> set[str]:
    return {g for g, d in spec.raw["topology_alphabet"].items() if d.get("passable", True)}


def word_norm(word: str, spec: RatTrapSpec) -> float:
    return norm(list(word), passable_generators(spec))


class FrozenEscape:
    """Can the cord, frozen in its current shape (with any ball / mouse on it), translate to infinity without
    touching the rigid structure? One query model: the static structure plus a free body holding one capsule
    per cord segment (and the spheres); a check only rewrites those geoms' poses."""

    def __init__(self, spec: RatTrapSpec, sim: RatTrapSim):
        r = spec.raw
        self.spec = spec
        self.n = len(sim.built.rope.body_names)
        self.lengths = np.asarray(sim.built.rope.seg_lengths, float)
        self.rc = float(r["cord_radius"])
        self.spheres = []  # (node index, radius)
        if r.get("cord_ball_enabled") and sim.built.ball_node is not None:
            self.spheres.append((sim.built.ball_node, float(r["cord_ball_radius"])))
        if r.get("mouse_enabled"):
            mn = r.get("mouse_node", "auto")
            self.spheres.append(((self.n if not r.get("cord_closed") else self.n // 2) if mn == "auto" else int(mn) % (self.n + 1),
                                 float(r["mouse_radius"])))
        xml = build(spec, include_cord=False, num_handles=0).xml
        body = ['<body name="assembly"><freejoint name="assembly_free"/>']
        for i, L in enumerate(self.lengths):
            body.append(f'<geom name="as_c{i}" type="capsule" size="{self.rc:.9g} {L / 2:.9g}" mass="0.001"/>')
        for k, (_, rad) in enumerate(self.spheres):
            body.append(f'<geom name="as_s{k}" type="sphere" size="{rad:.9g}" mass="0.001"/>')
        body.append("</body>")
        xml = xml.replace("</worldbody>", "".join(body) + "</worldbody>", 1)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        aid = self.m.body("assembly").id
        self.ag = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == aid])
        self.is_a = np.zeros(self.m.ngeom, bool)
        self.is_a[self.ag] = True
        # the compiler marks geoms placed at their body's origin as "same frame" and then ignores runtime pose
        # writes; these geoms are re-posed for every check, so they must use geom_pos / geom_quat
        self.m.geom_sameframe[self.ag] = 0
        # the midphase culls pairs by bounding volumes computed at compile time (all assembly geoms at the origin)
        # and it ignores margins: both are wrong here, so it stays off for this query model
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        self.qadr = self.m.jnt_qposadr[self.m.joint("assembly_free").id]
        mujoco.mj_forward(self.m, self.d)
        static = np.flatnonzero(~self.is_a)
        lo = np.min(self.d.geom_xpos[static] - self.m.geom_rbound[static, None], axis=0)
        hi = np.max(self.d.geom_xpos[static] + self.m.geom_rbound[static, None], axis=0)
        self.c_struct = (lo + hi) / 2
        self.r_struct = float(np.max(np.linalg.norm(self.d.geom_xpos[static] - self.c_struct, axis=1) + self.m.geom_rbound[static]))
        self.dirs = fibonacci_directions(int(r.get("escape_directions", 64)))
        self._margin = None

    def _set_margin(self, mg: float) -> None:
        if mg != self._margin:
            self.m.geom_margin[self.ag] = mg
            self._margin = mg

    def _min_dist(self, offset: np.ndarray, margin: float) -> float:
        self._set_margin(margin)
        self.d.qpos[self.qadr:self.qadr + 3] = offset
        self.d.qpos[self.qadr + 3:self.qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        n = self.d.ncon
        if not n:
            return margin
        g1, g2 = self.d.contact.geom1[:n], self.d.contact.geom2[:n]
        mask = self.is_a[g1] != self.is_a[g2]
        return float(min(margin, self.d.contact.dist[:n][mask].min())) if mask.any() else margin

    def set_shape(self, nodes: np.ndarray) -> None:
        P = np.asarray(nodes, float)
        for i in range(self.n):
            a, b = P[i], P[i + 1]
            g = self.ag[i]
            self.m.geom_pos[g] = (a + b) / 2
            # MuJoCo capsules lie along their local z axis: rotate x -> segment, then x -> z swap via frame_from_x
            # MuJoCo capsules lie along their local z axis: (x -> segment) composed with (z -> x, 90 deg about y)
            self.m.geom_quat[g] = quat_mul(frame_from_x(b - a), _Z_TO_X)
        for k, (node, _) in enumerate(self.spheres):
            self.m.geom_pos[self.ag[self.n + k]] = P[node]
        self.center = P.mean(axis=0)
        self.radius = float(np.max(np.linalg.norm(P - self.center, axis=1)) + max([self.rc] + [rad for _, rad in self.spheres]))

    def escape_direction(self, nodes: np.ndarray, step: float = 0.001, tol: float = 0.0005, max_contact_steps: int = 150,
                         dmax: float = 0.03) -> np.ndarray | None:
        """A direction along which the frozen assembly slides to infinity untouched, or None. Sound: far from the
        structure the sweep advances by the separation distance minus `tol` (conservative advancement); close to it
        by exact non-penetration checks every `step`; it gives up (blocked) after `max_contact_steps` of those."""
        self.set_shape(nodes)
        R = self.r_struct + self.radius + tol
        w = self.center - self.c_struct
        for u in self.dirs:
            uw = float(u @ w)
            disc = uw * uw - float(w @ w) + R * R
            s_end = max(0.0, -uw + math.sqrt(disc)) if disc > 0 else 0.0
            s, k, free = 0.0, 0, True
            while s < s_end and free:
                dd = self._min_dist(s * u, dmax)
                if dd - tol >= 4 * step:
                    s += dd - tol
                    continue
                for _ in range(8):
                    if k >= max_contact_steps or self._min_dist(s * u, 0.0) < -1e-5:
                        free = False
                        break
                    s += step
                    k += 1
            if free:
                return u.copy()
        return None


class CheckpointPotential(PotentialShaping):
    """Phi = -[(K - k) + min(1, d(x, c_{k+1}) / D_{k+1})] (checkpoints.CheckpointTracker.potential), precomputed by the
    episode for each state (it depends on the checkpoints reached so far, part of the episode state)."""

    name = "rattrap_checkpoints"

    def phi(self, s: RewardState) -> float | None:
        return None if s.geometry is None else s.geometry.get("checkpoint_potential")


def shaping_components(spec: RatTrapSpec) -> list[PotentialShaping]:
    g, lam = float(spec["shaping_gamma"]), float(spec.raw.get("shaping_lambda", 1.0))
    comps: list[PotentialShaping] = [CheckpointPotential(gamma=g, lam=lam)]
    for a in spec.raw.get("shaping_alignment") or []:
        comps.append(AperturePassageAlignment(f"disc:{a['disc']}", "cord", gate_distance=float(a["gate_distance"]),
                                              weight=float(a["weight"]), tangent_length_scale=float(a.get("tangent_length_scale", 0.01)),
                                              closed_passer=bool(spec["cord_closed"]), gamma=g, lam=lam, label=a["disc"]))
    return comps


class RatTrapEpisode:
    """One scored episode: discrete commands through the rig; checkpoints, violations and success after each action
    (word and norm as diagnostics)."""

    def __init__(self, spec: RatTrapSpec, training: bool = False, sim: RatTrapSim | None = None, checkpoints: Checkpoints | None = None):
        self.spec = spec
        self.sim = sim or RatTrapSim(spec)
        self.discs = build_discs(spec)
        self.closed = bool(spec["cord_closed"])
        self.escape = FrozenEscape(spec, self.sim)
        self.training = training
        self.T = int(spec["max_actions"])
        self.actions = 0
        self.word0 = self.word()
        self.norm0 = word_norm(self.word0, spec)
        self.norms = [self.norm0]
        self.violation = False
        self.solved = False
        self.done = False
        self.shaping_total = 0.0
        self.cps = checkpoints if checkpoints is not None else load_checkpoints(spec)
        self.tracker = CheckpointTracker(self.cps) if self.cps is not None else None
        self.shaping = shaping_components(spec)
        self._geom = any(isinstance(c, AperturePassageAlignment) for c in self.shaping)
        mode = str(spec.raw.get("certify_motion", "off"))
        if mode not in ("off", "log", "enforce"):
            raise ValueError(f"certify_motion must be off, log or enforce (got {mode!r})")
        self.certify = mode
        if mode != "off":
            from .certify import CordMotionCertifier

            self.sim.certifier = CordMotionCertifier(self.sim, escape=self.escape)
            self.sim.certifier.prime()
        self._rs = self._state(self.word0)

    def _state(self, word: str) -> RewardState:
        nodes = self.sim.node_positions()
        geometry = {"checkpoint_potential": None if self.tracker is None else self.tracker.potential(nodes)}
        if self._geom:
            geometry.update({"cord": nodes, **{f"disc:{d.name}": {"center": d.origin, "normal": d.normal} for d in self.discs}})
        return RewardState(puzzle="rattrap", actions_used=self.actions, max_actions=self.T, success=self.solved,
                           violations=int(self.violation), done=self.done, topology_word=word, geometry=geometry)

    @property
    def uncertified(self) -> int:
        return 0 if self.sim.certifier is None else self.sim.certifier.log.failed

    def word(self) -> str:
        return read_word(self.sim.node_positions(), self.discs, self.closed)

    def act(self, tool: str, a: dict) -> dict:
        if self.done:
            raise RuntimeError("episode is over")
        if tool == "grab_node":
            self.sim.grab_node(a["handle"], a["node"])
        elif tool == "move":
            mv = self.sim.move(a["handle"], [a["dx"], a["dy"], a["dz"]])
        elif tool == "release":
            self.sim.release(a["handle"])
        else:
            raise ValueError(f"unknown command {tool!r}")
        self.actions += 1
        if self.tracker is not None:
            self.tracker.update(self.sim.node_positions())
        w = self.word()  # diagnostics only
        nrm = word_norm(w, self.spec)
        self.norms.append(nrm)
        if self.sim.max_penetration > self.spec["penetration_tolerance"] or (self.certify == "enforce" and self.uncertified):
            self.violation, self.done = True, True
        elif self.escape.escape_direction(self.sim.node_positions()) is not None:
            self.solved, self.done = True, True
        elif self.actions >= self.T:
            self.done = True
        out = {"done": self.done, "solved": self.solved, "violation": self.violation,
               "checkpoints_reached": None if self.tracker is None else self.tracker.k, "word": w, "norm": nrm}
        if tool == "move":
            out.update(blocked=mv.get("blocked"), blocked_reason=mv.get("blocked_reason"), executed_fraction=mv.get("requested_fraction_executed"))
        if self.certify != "off":
            out["uncertified_substeps"] = self.uncertified
        cur = self._state(w)
        if self.training:  # potential-based shaping, training only (never part of the score)
            parts = {c.name: c.step(self._rs, cur, {}) for c in self.shaping}
            parts["violation_penalty"] = -(self.shaping[0].lam * (self.cps.K + 1) + 1.0) if self.violation and self.cps is not None else 0.0  # docs/rewards.md §8
            sh = float(sum(parts.values()))
            self.shaping_total += sh
            out["shaping"] = sh
            out["shaping_parts"] = parts
            if self._geom:
                out["alignment"] = {c.name: c.last_terms for c in self.shaping if isinstance(c, AperturePassageAlignment)}
        self._rs = cur
        return out

    def progress(self) -> float:
        """p = checkpoints reached in order / K (0 without a checkpoint file)."""
        return 0.0 if self.tracker is None else self.tracker.progress()

    def score(self) -> float:
        if self.violation:
            return 0.0
        if self.solved:
            return 1.0 - float(self.spec["score_alpha"]) * min(self.actions, self.T) / self.T
        return float(self.spec["score_progress_cap"]) * self.progress()


def load_checkpoints(spec: RatTrapSpec) -> Checkpoints | None:
    """The reference solve's checkpoints (rattrap.yaml `checkpoints_file`, relative to loophole/reference/), or None."""
    name = spec.raw.get("checkpoints_file")
    path = DEFAULT_FILE if not name else DEFAULT_FILE.parent / name
    if not path.exists():
        return None
    cps = Checkpoints.load(path)
    if cps.states.ndim != 3 or cps.states.shape[1:] != cps.start.shape:
        raise ValueError(f"{path}: malformed checkpoint file")
    return cps


def check_passability(spec: RatTrapSpec) -> list[str]:
    """Generators declared passable whose hole a body carried by the cord cannot pass (per the clearance table)."""
    carried = [o for o, on in (("mouse", spec.raw.get("mouse_enabled")), ("cord_ball", spec.raw.get("cord_ball_enabled"))) if on]
    table = {(r["aperture"], r["object"]): r["expect"] for r in spec.raw["clearance_table"]}
    bad = []
    for g, d in spec.raw["topology_alphabet"].items():
        ap = d.get("aperture")
        if not d.get("passable", True) or ap is None:
            continue
        for obj in carried:
            if table.get((ap, obj)) == "block":
                bad.append(f"{g} ({ap}) is declared passable but the {obj} on the cord cannot pass it")
    return bad
