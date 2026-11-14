"""Rat Trap simulator scaffold: build from YAML, initialise the cord along its route, grab and move cord nodes.

No reward, no MCP tools, no Harbor task. The manipulation rig is loophole.sim.rig.Rig, reused unchanged:
a move advances the handle in substeps at bounded speed and halts the moment the held node lags its handle
(lag_limit) or any contact is pressed deeper than stall_penetration, reporting `blocked` and the fraction
completed. Nothing is ever forced through anything.
"""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np

from ...config import RigConfig
from ...sim.rig import Rig, RigError
from .build import RatTrapBuilt, build
from .spec import RatTrapError, RatTrapSpec, load_spec

STATE_SIG = mujoco.mjtState.mjSTATE_INTEGRATION


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
    def __init__(self, spec: RatTrapSpec | None = None, include_cord: bool = True, ring_free: bool = True,
                 probe: dict | None = None, check_route: bool = True):
        self.spec = spec or load_spec()
        self.built: RatTrapBuilt = build(self.spec, include_cord=include_cord, ring_free=ring_free, probe=probe)
        self.model = mujoco.MjModel.from_xml_string(self.built.xml)
        self.data = mujoco.MjData(self.model)
        self.cfg = rig_config(self.spec)
        self.rig = Rig(self.model, self.data, self.built, self.cfg, self._substep, strain=self._strain)
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

    def _strain(self) -> tuple[float, float]:
        pen = max((x[0] for x in self.penetrations()), default=0.0)
        return pen, 0.0

    def _substep(self) -> None:
        mujoco.mj_step(self.model, self.data)
        pen = max((x[0] for x in self.penetrations()), default=0.0)
        self.max_penetration = max(self.max_penetration, pen)

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
        d, names, L = self.data, self.built.rope.body_names, self.built.rope.seg_lengths
        pts = [d.xpos[self.model.body(b).id].copy() for b in names]
        last = self.model.body(names[-1]).id
        pts.append(d.xpos[last] + d.xmat[last].reshape(3, 3) @ np.array([L[-1], 0.0, 0.0]))
        return np.array(pts)

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
        return self.rig.release(handle)

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
