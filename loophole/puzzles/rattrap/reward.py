"""Rat Trap reward: Przytycki-Sikora norm of the cord's word, and the frozen-shape free-escape success test.

State: the freely (cyclically, for the closed loop) reduced word w of signed disc crossings (topology.py).
Potential: Phi(s) = -||w||, the norm of Przytycki & Sikora (arXiv:math/0007134, Theorem 3.2) with the
generators declared passable in the YAML (norm.py). ||w|| is the minimal number of hole passes that undo w.

Score (verifier side only; one float in [0, 1]):
    a violation (any contact pressed deeper than penetration_tolerance) -> 0, the episode ends
    solved (the frozen cord assembly can slide to infinity along one of `escape_directions` directions)
                                                           -> 1 - alpha * a / T          in [1 - alpha, 1]
    unsolved                                               -> cap * p,  p = clip(1 - min_t ||w_t|| / ||w_0||, 0, 1)
with alpha = 0.5 and cap = 0.3 < 1 - alpha, so any solve beats any non-solve.
Training shaping (never scored): gamma * Phi(s') - Phi(s), potential-based (Ng, Harada & Russell, ICML 1999), so it
telescopes: moving a piece of cord in and out of a hole nets zero.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ...sim.collision import fibonacci_directions
from ...sim.geometry import frame_from_x, quat_mul
from .build import build
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


class RatTrapEpisode:
    """One scored episode: discrete commands through the rig; word, norm, violations and success after each action."""

    def __init__(self, spec: RatTrapSpec, training: bool = False, sim: RatTrapSim | None = None):
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

    def word(self) -> str:
        return read_word(self.sim.node_positions(), self.discs, self.closed)

    def phi(self, nrm: float) -> float:
        return -nrm if nrm < INF else -1e6  # an unreachable state: an (arbitrary) very low potential

    def act(self, tool: str, a: dict) -> dict:
        if self.done:
            raise RuntimeError("episode is over")
        if tool == "grab_node":
            self.sim.grab_node(a["handle"], a["node"])
        elif tool == "move":
            self.sim.move(a["handle"], [a["dx"], a["dy"], a["dz"]])
        elif tool == "release":
            self.sim.release(a["handle"])
        else:
            raise ValueError(f"unknown command {tool!r}")
        self.actions += 1
        prev = self.norms[-1]
        w = self.word()
        nrm = word_norm(w, self.spec)
        self.norms.append(nrm)
        if self.sim.max_penetration > self.spec["penetration_tolerance"]:
            self.violation, self.done = True, True
        elif self.escape.escape_direction(self.sim.node_positions()) is not None:
            self.solved, self.done = True, True
        elif self.actions >= self.T:
            self.done = True
        out = {"word": w, "norm": nrm, "done": self.done, "solved": self.solved, "violation": self.violation}
        if self.training:  # potential-based shaping, training only
            g = float(self.spec["shaping_gamma"])
            sh = g * self.phi(nrm) - self.phi(prev)
            self.shaping_total += sh
            out["shaping"] = sh
        return out

    def progress(self) -> float:
        if not self.norm0 or self.norm0 == INF:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - min(self.norms) / self.norm0)))

    def score(self) -> float:
        if self.violation:
            return 0.0
        if self.solved:
            return 1.0 - float(self.spec["score_alpha"]) * min(self.actions, self.T) / self.T
        return float(self.spec["score_progress_cap"]) * self.progress()


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
