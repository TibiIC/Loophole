"""Manipulation rig: up to two mocap "handles" that grasp bodies through welds.

MuJoCo cannot add constraints at runtime, so the builder predeclares one
inactive ``weld`` per (handle, graspable body). Grabbing places the handle's
mocap body at the grasp point, writes the weld's relative pose into
``model.eq_data`` and activates it; releasing deactivates it.

Moves are clamped to a maximum linear/angular displacement and executed as an
interpolated mocap trajectory at bounded speed. If the grasped body lags the
handle by more than ``lag_limit`` (it is blocked by something), or any contact / the rope's
anchor constraint is strained beyond the "careful" stall levels, the handle
stops advancing and snaps back onto the body: the rig can never push an object
through an obstacle with unbounded force, and an honest agent never trips the
anti-tunnelling audit just by pulling a rope taut or pushing into the frame.

Handles have no collision geometry. Teleporting an *empty* handle to a grasp
point therefore cannot move anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from ..config import RigConfig
from .builder import BuiltModel
from .geometry import quat_conj, quat_from_euler_xyz, quat_mul, quat_normalize, quat_slerp, quat_to_axis_angle, quat_to_mat, quat_to_rotvec, quat_from_rotvec


class RigError(ValueError):
    """Invalid rig request (reported to the agent, costs no physics)."""


@dataclass
class Handle:
    idx: int
    mocap_id: int
    body: str | None = None
    eq_id: int | None = None
    anchor_local: np.ndarray | None = None  # grasp point in the grasped body's frame


def _point_geom_distance(model: mujoco.MjModel, data: mujoco.MjData, gid: int, p: np.ndarray) -> float:
    """Distance from a world point to a geom surface (negative inside). Exact for
    sphere/capsule/box, bounding-sphere approximation otherwise."""
    gtype = model.geom_type[gid]
    size = model.geom_size[gid]
    pos = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)
    local = mat.T @ (p - pos)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(np.linalg.norm(local) - size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(local[2], -size[1], size[1])
        return float(np.linalg.norm(local - np.array([0, 0, z])) - size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(local) - size[:3]
        return float(np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], q[1], q[2]), 0.0))
    return float(np.linalg.norm(local) - model.geom_rbound[gid])


class Rig:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        built: BuiltModel,
        cfg: RigConfig,
        substep: Callable[[], None],
        strain: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0),
    ):
        self.strain = strain  # () -> (current max penetration, current rope anchor drift)
        self.m = model
        self.d = data
        self.built = built
        self.cfg = cfg
        self.substep = substep
        self.handles = [Handle(h, int(model.body_mocapid[model.body(f"handle{h}").id])) for h in range(built.num_handles)]
        self.weld_ids = {k: model.equality(n).id for k, n in built.weld_names.items()}
        self.rope_bodies = built.rope.body_names if built.rope is not None else []
        self._park = [data.mocap_pos[h.mocap_id].copy() for h in self.handles]

    # ---------------------------------------------------------------- helpers
    def reset(self) -> None:
        for h in self.handles:
            h.body = h.eq_id = h.anchor_local = None
            self.d.mocap_pos[h.mocap_id] = self._park[h.idx]
            self.d.mocap_quat[h.mocap_id] = [1, 0, 0, 0]
        for eid in self.weld_ids.values():
            self.d.eq_active[eid] = 0

    def _handle(self, idx: int) -> Handle:
        if not isinstance(idx, int) or not 0 <= idx < len(self.handles):
            raise RigError(f"handle must be an integer in [0, {len(self.handles) - 1}]")
        return self.handles[idx]

    def _body_geoms(self, bid: int) -> list[int]:
        return [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == bid]

    def _distance_to_body(self, bid: int, p: np.ndarray) -> float:
        return min(_point_geom_distance(self.m, self.d, g, p) for g in self._body_geoms(bid))

    def grasp_point_world(self, h: Handle) -> np.ndarray:
        bid = self.m.body(h.body).id
        return self.d.xpos[bid] + self.d.xmat[bid].reshape(3, 3) @ h.anchor_local

    def handle_pose(self, h: Handle) -> tuple[np.ndarray, np.ndarray]:
        return self.d.mocap_pos[h.mocap_id].copy(), self.d.mocap_quat[h.mocap_id].copy()

    def resolve_target(self, target: str, point=None, segment=None) -> tuple[str, np.ndarray | None]:
        """Map a user target (+ optional point/segment) to a graspable body name."""
        rope = self.built.rope.name if self.built.rope is not None else None
        graspable = set(self.built.graspable)
        t = str(target).strip()
        if rope and t.startswith(rope + ":"):
            t, segment = rope, t.split(":", 1)[1]
        if rope and t == rope:
            if segment is not None:
                try:
                    k = int(segment)
                except (TypeError, ValueError):
                    raise RigError("segment must be an integer") from None
                if not 0 <= k < len(self.rope_bodies):
                    raise RigError(f"rope segment must be in [0, {len(self.rope_bodies) - 1}]")
                body = self.rope_bodies[k]
            elif point is not None:
                p = np.asarray(point, dtype=float)
                dists = [self._distance_to_body(self.m.body(b).id, p) for b in self.rope_bodies]
                body = self.rope_bodies[int(np.argmin(dists))]
            else:
                raise RigError("grabbing the rope needs `segment` (index) or `point`")
        else:
            body = t
        if body not in graspable:
            extra = f" or '{rope}' with a segment" if rope else ""
            raise RigError(f"{target!r} is not graspable; graspable: {sorted(set(self.built.object_bodies) & graspable)}{extra}")
        return body, (None if point is None else np.asarray(point, dtype=float))

    # ---------------------------------------------------------------- actions
    def grab(self, idx: int, target: str, point=None, segment=None) -> dict:
        h = self._handle(idx)
        if h.body is not None:
            raise RigError(f"handle {idx} is already holding {h.body}; release it first")
        body, p = self.resolve_target(target, point, segment)
        bid = self.m.body(body).id
        if p is None:
            if body in self.rope_bodies:
                k = self.rope_bodies.index(body)
                p = self.d.xpos[bid] + self.d.xmat[bid].reshape(3, 3) @ np.array([self.built.rope.seg_lengths[k] / 2, 0, 0])
            else:
                p = self.d.xpos[bid].copy()
        else:
            if len(p) != 3 or not np.all(np.isfinite(p)):
                raise RigError("point must be three finite numbers")
            dist = self._distance_to_body(bid, p)
            if dist > self.cfg.grasp_tolerance:
                raise RigError(f"point is {dist:.3f} m from {body}; must be within {self.cfg.grasp_tolerance} m of its surface")
        R = self.d.xmat[bid].reshape(3, 3)
        anchor = R.T @ (p - self.d.xpos[bid])
        eid = self.weld_ids[(idx, body)]
        # weld data: anchor on body2 (grasped body), anchor on body1 (handle), relquat, torquescale
        self.m.eq_data[eid, 0:3] = anchor
        self.m.eq_data[eid, 3:6] = 0.0
        self.m.eq_data[eid, 6:10] = [1, 0, 0, 0]
        self.d.mocap_pos[h.mocap_id] = p
        self.d.mocap_quat[h.mocap_id] = self.d.xquat[bid]
        self.d.eq_active[eid] = 1
        h.body, h.eq_id, h.anchor_local = body, eid, anchor
        for _ in range(self.cfg.settle_steps):
            self.substep()
        return {"handle": idx, "grasped": body, "grasp_point": p.round(5).tolist()}

    def release(self, idx: int) -> dict:
        h = self._handle(idx)
        if h.body is None:
            raise RigError(f"handle {idx} is not holding anything")
        released = h.body
        self.d.eq_active[h.eq_id] = 0
        h.body = h.eq_id = h.anchor_local = None
        for _ in range(self.cfg.settle_steps):
            self.substep()
        return {"handle": idx, "released": released}

    def clamp_request(self, idx: int, dpos, drot, frame: str) -> dict:
        h = self._handle(idx)
        if frame not in ("world", "local"):
            raise RigError("frame must be 'world' or 'local'")
        dpos = np.asarray(dpos, dtype=float)
        drot = np.asarray(drot, dtype=float)
        if dpos.shape != (3,) or drot.shape != (3,) or not (np.all(np.isfinite(dpos)) and np.all(np.isfinite(drot))):
            raise RigError("dx..dyaw must be finite numbers")
        p0, q0 = self.handle_pose(h)
        R0 = quat_to_mat(q0)
        dq = quat_from_euler_xyz(drot)
        clamped = []
        lin = float(np.linalg.norm(dpos))
        if lin > self.cfg.max_linear_step:
            dpos = dpos * (self.cfg.max_linear_step / lin)
            clamped.append(f"translation {lin:.3f} m -> {self.cfg.max_linear_step} m")
        rv = quat_to_rotvec(dq)
        ang = float(np.linalg.norm(rv))
        if ang > self.cfg.max_angular_step:
            rv = rv * (self.cfg.max_angular_step / ang)
            dq = quat_from_rotvec(rv)
            clamped.append(f"rotation {ang:.3f} rad -> {self.cfg.max_angular_step} rad")
        if frame == "local":
            dpos_w = R0 @ dpos
            q1 = quat_normalize(quat_mul(q0, dq))
        else:
            dpos_w = dpos
            q1 = quat_normalize(quat_mul(dq, q0))
        p1 = p0 + dpos_w
        lo, hi = np.asarray(self.cfg.workspace_min), np.asarray(self.cfg.workspace_max)
        p1c = np.clip(p1, lo, hi)
        if not np.allclose(p1c, p1):
            clamped.append("workspace bounds")
        return {"p0": p0, "q0": q0, "p1": p1c, "q1": q1, "clamped": clamped}

    def move(self, idx: int, dpos, drot, frame: str = "world") -> dict:
        req = self.clamp_request(idx, dpos, drot, frame)
        h = self.handles[idx]
        p0, q0, p1, q1 = req["p0"], req["q0"], req["p1"], req["q1"]
        lin = float(np.linalg.norm(p1 - p0))
        _, ang = quat_to_axis_angle(quat_mul(quat_conj(q0), q1))
        dt = self.m.opt.timestep
        n = max(1, math.ceil(max(lin / self.cfg.linear_speed, abs(ang) / self.cfg.angular_speed) / dt))
        blocked = None
        done_frac = 0.0
        for k in range(1, n + 1):
            if h.body is not None and (blocked := self._lagging(h) or self._strained()):
                self._snap_to_body(h)
                break
            f = k / n
            self.d.mocap_pos[h.mocap_id] = p0 + f * (p1 - p0)
            self.d.mocap_quat[h.mocap_id] = quat_slerp(q0, q1, f)
            done_frac = f
            self.substep()
        if not blocked:
            for _ in range(self.cfg.settle_steps):
                self.substep()
        pf, qf = self.handle_pose(h)
        out = {
            "handle": idx,
            "holding": h.body,
            "requested_fraction_executed": round(done_frac, 3),
            "blocked": bool(blocked),
            "blocked_reason": blocked or None,
            "clamped": req["clamped"],
            "handle_pos": pf.round(5).tolist(),
            "handle_quat": qf.round(5).tolist(),
        }
        if h.body is not None:
            out["grasp_point"] = self.grasp_point_world(h).round(5).tolist()
        return out

    def _lagging(self, h: Handle) -> str | None:
        """Reason string if the grasped body lags the handle too much, else None."""
        gp = self.grasp_point_world(h)
        lag = float(np.linalg.norm(self.d.mocap_pos[h.mocap_id] - gp))
        if lag > self.cfg.lag_limit:
            return f"{h.body} lags the handle by {lag:.3f} m (blocked by contact or constraint)"
        bq = self.d.xquat[self.m.body(h.body).id]
        _, ang = quat_to_axis_angle(quat_mul(quat_conj(self.d.mocap_quat[h.mocap_id]), bq))
        if abs(ang) > self.cfg.angle_lag_limit:
            return f"{h.body} orientation lags the handle by {ang:.2f} rad (blocked by contact or constraint)"
        return None

    def _strained(self) -> str | None:
        pen, drift = self.strain()
        if pen > self.cfg.stall_penetration:
            return f"contact pressed {pen * 1000:.1f} mm deep (blocked by contact)"
        if drift > self.cfg.stall_anchor_drift:
            return "rope pulled taut against its anchor (blocked by rope length)"
        return None

    def _snap_to_body(self, h: Handle) -> None:
        """Move the handle onto the body's actual pose so the weld exerts no force."""
        bid = self.m.body(h.body).id
        self.d.mocap_pos[h.mocap_id] = self.grasp_point_world(h)
        self.d.mocap_quat[h.mocap_id] = self.d.xquat[bid]

    def status(self) -> list[dict]:
        out = []
        for h in self.handles:
            p, q = self.handle_pose(h)
            out.append({"id": h.idx, "holding": h.body, "pos": p.round(4).tolist(), "quat": q.round(4).tolist()})
        return out
