"""Rat Trap simulator scaffold: build from YAML, initialise the cord along its route, grab and move cord nodes.

No reward, no MCP tools, no Harbor task. The manipulation rig is loophole.sim.rig.Rig, reused unchanged:
a move advances the handle in substeps at bounded speed and halts the moment the held node lags its handle
(lag_limit) or any contact is pressed deeper than stall_penetration, reporting `blocked` and the fraction
completed. Nothing is ever forced through anything.
"""

from __future__ import annotations

import hashlib

import math

import mujoco
import numpy as np

from ...config import RigConfig
from ...sim.geometry import quat_conj, quat_mul, quat_slerp, quat_to_axis_angle
from ...sim.rig import Rig, RigError
from .build import RatTrapBuilt, build
from .spec import RatTrapError, RatTrapSpec, load_spec

STATE_SIG = mujoco.mjtState.mjSTATE_INTEGRATION


class RatTrapRig(Rig):
    """The shared rig with Rat Trap blocked-move semantics (rattrap.yaml `rig_block_mode`).

    halt:    as for Alpha: on a block the handle snaps onto the grasped node and stops there, against the obstruction.
             The cord keeps the strain it was pulled into, and a later release can spring it into a contact.
    retreat: on a block the handle moves back along the commanded path, at the rig's speed, to the last configuration
             where the node's lag and the contact depth were both under `rig_retreat_relax` of their stall limits
             (the move's start if none), and settles there: no strain is stored."""

    def __init__(self, *a, block_mode: str = "halt", relax: float = 0.5, **kw):
        super().__init__(*a, **kw)
        if block_mode not in ("halt", "retreat"):
            raise ValueError(f"rig_block_mode must be halt or retreat (got {block_mode!r})")
        self.block_mode, self.relax = block_mode, float(relax)

    def move(self, idx: int, dpos, drot, frame: str = "world") -> dict:
        if self.block_mode == "halt":
            return super().move(idx, dpos, drot, frame)
        req = self.clamp_request(idx, dpos, drot, frame)
        h = self.handles[idx]
        p0, q0, p1, q1 = req["p0"], req["q0"], req["p1"], req["q1"]
        lin = float(np.linalg.norm(p1 - p0))
        _, ang = quat_to_axis_angle(quat_mul(quat_conj(q0), q1))
        dt = self.m.opt.timestep
        n = max(1, math.ceil(max(lin / self.cfg.linear_speed, abs(ang) / self.cfg.angular_speed) / dt))
        blocked, done_frac, safe_f = None, 0.0, 0.0
        for k in range(1, n + 1):
            if self.abort():
                break
            if h.body is not None and (blocked := self._lagging(h) or self._strained()):
                done_frac = self._retreat(h, p0, q0, p1, q1, done_frac, safe_f)
                break
            f = k / n
            self.d.mocap_pos[h.mocap_id] = p0 + f * (p1 - p0)
            self.d.mocap_quat[h.mocap_id] = quat_slerp(q0, q1, f)
            done_frac = f
            self.substep()
            if h.body is not None and self._relaxed(h):
                safe_f = f
        if not blocked:
            for _ in range(self.cfg.settle_steps):
                if self.abort():
                    break
                self.substep()
        pf, qf = self.handle_pose(h)
        out = {"handle": idx, "holding": h.body, "requested_fraction_executed": round(done_frac, 3), "blocked": bool(blocked),
               "blocked_reason": blocked or None, "clamped": req["clamped"], "handle_pos": pf.round(5).tolist(), "handle_quat": qf.round(5).tolist()}
        if blocked:
            out["retreated"] = True
        if h.body is not None:
            out["grasp_point"] = self.grasp_point_world(h).round(5).tolist()
        return out

    def _relaxed(self, h) -> bool:
        gp = self.grasp_point_world(h)
        lag = float(np.linalg.norm(self.d.mocap_pos[h.mocap_id] - gp))
        pen, _ = self.strain()
        return lag <= self.relax * self.cfg.lag_limit and pen <= self.relax * self.cfg.stall_penetration

    def _retreat(self, h, p0, q0, p1, q1, f_now: float, f_safe: float) -> float:
        """Move the handle back from fraction f_now to f_safe along the commanded path at the rig's speed, then settle."""
        dt = self.m.opt.timestep
        dist = float(np.linalg.norm((p1 - p0) * (f_now - f_safe)))
        m = max(1, math.ceil(dist / self.cfg.linear_speed / dt))
        for j in range(1, m + 1):
            if self.abort():
                break
            f = f_now + (f_safe - f_now) * j / m
            self.d.mocap_pos[h.mocap_id] = p0 + f * (p1 - p0)
            self.d.mocap_quat[h.mocap_id] = quat_slerp(q0, q1, f)
            self.substep()
        for _ in range(self.cfg.settle_steps):
            if self.abort():
                break
            self.substep()
        return f_safe


def rig_config(spec: RatTrapSpec) -> RigConfig:
    r = spec.raw
    return RigConfig(
        num_handles=r["rig_num_handles"], max_linear_step=r["rig_max_linear_step"], max_angular_step=r["rig_max_angular_step"],
        linear_speed=r["rig_linear_speed"], angular_speed=r["rig_angular_speed"], settle_steps=r["rig_settle_steps"],
        lag_limit=r["rig_lag_limit"], angle_lag_limit=r["rig_angle_lag_limit"], grasp_tolerance=r["rig_grasp_tolerance"],
        stall_penetration=r["rig_stall_penetration"], workspace_min=r["rig_workspace_min"], workspace_max=r["rig_workspace_max"],
        weld_solref=r["rig_weld_solref"],
    )


class RatTrapSim:
    def __init__(self, spec: RatTrapSpec | None = None, include_cord: bool = True, ring_free: bool | None = None,
                 probe: dict | None = None, check_route: bool = True):
        self.spec = spec or load_spec()
        self.built: RatTrapBuilt = build(self.spec, include_cord=include_cord, probe=probe)
        self.model = mujoco.MjModel.from_xml_string(self.built.xml)
        self.data = mujoco.MjData(self.model)
        self.cfg = rig_config(self.spec)
        self.rig = RatTrapRig(self.model, self.data, self.built, self.cfg, self._substep, strain=self._strain,
                              block_mode=str(self.spec.raw.get("rig_block_mode", "halt")), relax=float(self.spec.raw.get("rig_retreat_relax", 0.5)))
        self._quiet = False  # release settling: contact depth is watched separately (see release)
        self._window_peak = 0.0
        m = self.model
        self.part_of_geom = {}
        for g in range(m.ngeom):
            nm = m.geom(g).name
            if nm in self.built.geom_part:
                self.part_of_geom[g] = self.built.geom_part[nm]
            elif nm.startswith("cord_"):
                self.part_of_geom[g] = "cord"
        self.cord_geoms = {g: int(m.geom(g).name.split("_")[1]) for g, p in self.part_of_geom.items() if p == "cord"}
        self.max_penetration = 0.0
        self.certifier = None  # optional swept-motion certificate (certify.CordMotionCertifier), alongside the audit
        self._renderer = None
        self.reset(check_route=check_route)

    # ------------------------------------------------------------------ physics
    def penetrations(self) -> list[tuple[float, str, str]]:
        """(depth, part a, part b) of every contact pressed into its partner (dist < 0)."""
        d, out = self.data, []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0:
                a = self._label(c.geom1)
                b = self._label(c.geom2)
                out.append((-float(c.dist), a, b))
        return out

    def _label(self, g: int) -> str:
        p = self.part_of_geom.get(g, self.model.geom(g).name)
        return f"cord segment {self.cord_geoms[g]}" if p == "cord" else p

    def deepest(self) -> float:
        """Deepest current contact penetration (m), vectorised (used every physics step)."""
        n = self.data.ncon
        return float(max(0.0, -np.min(self.data.contact.dist[:n]))) if n else 0.0

    def _strain(self) -> tuple[float, float]:
        return self.deepest(), 0.0

    def _substep(self) -> None:
        mujoco.mj_step(self.model, self.data)
        if self._quiet:
            self._window_peak = max(self._window_peak, self.deepest())
        else:
            self.max_penetration = max(self.max_penetration, self.deepest())
        if self.certifier is not None:
            self.certifier.after_substep()

    def reset(self, check_route: bool = True) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.rig.reset()
        mujoco.mj_forward(m, d)
        if check_route:
            self.assert_no_penetration("route (before settling)")
        for _ in range(int(self.spec.raw["settle_after_route_steps"])):
            mujoco.mj_step(m, d)
        if check_route:
            self.assert_no_penetration("route (after settling)")
        self.max_penetration = 0.0
        if getattr(self, "certifier", None) is not None:
            self.certifier.prime()

    def assert_no_penetration(self, when: str) -> None:
        """Fail loudly, naming the segment and the part, if the cord penetrates anything (route is wrong)."""
        tol = self.spec.raw["penetration_tolerance"]
        bad = [x for x in self.penetrations() if x[0] > tol and ("cord" in x[1] or "cord" in x[2])]
        if bad:
            depth, a, b = max(bad)
            raise RatTrapError(f"{when}: {a} penetrates {b} by {depth * 1000:.2f} mm (tolerance {tol * 1000:.2f} mm); "
                               f"{len(bad)} contact(s) in total. The cord route is wrong there.")

    def state_hash(self) -> str:
        buf = np.empty(mujoco.mj_stateSize(self.model, STATE_SIG))
        mujoco.mj_getState(self.model, self.data, buf, STATE_SIG)
        h = hashlib.sha256(buf.tobytes())
        h.update(self.model.eq_data.tobytes())
        return h.hexdigest()

    def max_speed(self) -> float:
        return float(np.max(np.abs(self.data.qvel))) if self.model.nv else 0.0

    # ------------------------------------------------------------------ cord nodes
    @property
    def n_nodes(self) -> int:
        return len(self.built.rope.body_names) + 1

    def node_index(self, k: int) -> int:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise RigError("node must be an integer index")
        n = self.n_nodes
        if not -n <= k < n:
            raise RigError(f"node must be in [{-n}, {n - 1}]")
        return int(k) % n

    def node_positions(self) -> np.ndarray:
        """Node k = start of segment k (k < n), node n = end of the last segment. Segments built backwards (from the
        middle-rooted tree) have their frame origin at their end node."""
        d, L = self.data, self.built.rope.seg_lengths
        if not hasattr(self, "_seg_ids"):
            self._seg_ids = np.array([self.model.body(b).id for b in self.built.rope.body_names])
            self._rev = np.array(self.built.seg_reversed or [False] * len(L), bool)
            self._L = np.asarray(L, float)
        pos = d.xpos[self._seg_ids]
        far = pos + d.xmat[self._seg_ids].reshape(-1, 3, 3)[:, :, 0] * self._L[:, None]  # origin + L * local x axis
        start = np.where(self._rev[:, None], far, pos)
        end_last = pos[-1] if self._rev[-1] else far[-1]
        return np.vstack([start, end_last[None]])

    def grab_node(self, handle: int, k: int) -> dict:
        k = self.node_index(k)
        seg = min(k, self.n_nodes - 2)
        p = self.node_positions()[k]
        out = self.rig.grab(handle, "cord", point=p.tolist(), segment=seg)
        out["node"] = k
        return out

    def grab_body(self, handle: int, body: str) -> dict:
        return self.rig.grab(handle, body)

    def move(self, handle: int, dpos, drot=(0.0, 0.0, 0.0), frame: str = "world") -> dict:
        return self.rig.move(handle, list(map(float, dpos)), list(map(float, drot)), frame)

    def move_to(self, handle: int, target, max_moves: int = 200) -> dict:
        """Repeated capped moves toward a world point until reached or blocked."""
        res = {}
        for _ in range(max_moves):
            h = self.rig.handles[handle]
            d = np.asarray(target, float) - self.rig.handle_pose(h)[0]
            if np.linalg.norm(d) < 1e-6:
                break
            res = self.move(handle, d)
            if res["blocked"]:
                break
        return res

    def release(self, handle: int) -> dict:
        """Open a gripper. rattrap.yaml `release_settle_audit`: the rig's settle after a release is watched separately;
        a penetration counts only if it persists after settling (deepest contact at the end), or if a transient during
        the settle exceeds `release_transient_limit` (below the cord's diameter, so a release cannot spring the cord
        through a wall). Letting go of a strained strand otherwise produces a brief overshoot that is not the agent's
        doing (docs/HARNESS_NOTES.md)."""
        if not self.spec.raw.get("release_settle_audit", False):
            return self.rig.release(handle)
        self._quiet, self._window_peak = True, 0.0
        try:
            out = self.rig.release(handle)
        finally:
            self._quiet = False
        end = self.deepest()
        limit = float(self.spec.raw.get("release_transient_limit", 0.002))
        if end > float(self.spec["penetration_tolerance"]) or self._window_peak > limit:
            self.max_penetration = max(self.max_penetration, end, self._window_peak)
        else:
            self.max_penetration = max(self.max_penetration, end)
        out["release_transient_peak_m"] = round(self._window_peak, 6)
        return out

    # ------------------------------------------------------------------ rendering
    def render(self, view: str = "iso", azimuth=None, elevation=None, distance=None, lookat=None, width=640, height=480) -> np.ndarray:
        from ...sim.render import SceneRenderer

        if self._renderer is None:
            r = self.spec.raw
            center = np.array([0.0, 0.0, self.spec.z0 + r["frame_inner_height"] / 2])
            self._renderer = SceneRenderer(self.model, width, height, center, 0.75)
        cam = self._renderer.camera(view, azimuth, elevation, distance, lookat)
        arrays, _ = self._renderer.render_modalities(self.data, cam, ["rgb"], width, height)
        return arrays["rgb"]

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
