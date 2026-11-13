"""LoopholeEnv: one puzzle episode (physics + rig + audit + topology + reward + log).

This is the single source of truth used by the MCP server, the reference
replayer, the baselines, the tests and the verifier. Agents only ever reach it
through the MCP tools; reset/puzzle/seed are constructor arguments (admin only).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..config import EnvConfig, load_config
from ..reward.components import RewardComposer, RewardState
from ..reward.topology import Disc, TopologyTracker
from .audit import Auditor
from .builder import BuiltModel, build_model
from .rig import Rig, RigError
from .spec import Puzzle, load_puzzle, resolve_puzzle

PHYSICS_TOOLS = ("grab", "move", "release", "wait")
STATE_SIG = mujoco.mjtState.mjSTATE_INTEGRATION
LOG_VERSION = 1


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


class EpisodeOver(RuntimeError):
    pass


class EpisodeLog:
    """Append-only JSONL log of every call (also future training data)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._f = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "w", buffering=1)

    def write(self, rec: dict) -> None:
        if self._f:
            self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


class LoopholeEnv:
    def __init__(
        self,
        puzzle: str | Path | Puzzle,
        config: EnvConfig | None = None,
        seed: int = 0,
        log_path: str | Path | None = None,
    ):
        self.cfg = config or load_config()
        self.puzzle = puzzle if isinstance(puzzle, Puzzle) else load_puzzle(resolve_puzzle(str(puzzle)))
        self.seed = int(seed)
        self.built: BuiltModel = build_model(
            self.puzzle,
            num_handles=self.cfg.rig.num_handles,
            weld_solref=self.cfg.rig.weld_solref,
            weld_solimp=self.cfg.rig.weld_solimp,
        )
        self.model = mujoco.MjModel.from_xml_string(self.built.xml)
        self.data = mujoco.MjData(self.model)
        self._eq_data0 = self.model.eq_data.copy()
        self.audit = Auditor(self.model, self.built, self.cfg.audit)
        self.rig = Rig(self.model, self.data, self.built, self.cfg.rig, self._substep)
        self.target_id = self.model.body(self.puzzle.target).id
        self.target_radius = self._body_radius(self.target_id)
        self.rope_node_bodies = [self.model.body(b).id for b in self.built.rope.body_names]
        anchors = [self.built.rope.start] + ([self.built.rope.end] if self.built.rope.end is not None else [])
        self.anchors = [np.asarray(a, dtype=float) for a in anchors]
        self.tracker = TopologyTracker(
            [
                Disc(h.name, h.body, np.asarray(h.center, float), np.asarray(h.normal, float) / np.linalg.norm(h.normal), h.radius)
                for h in self.puzzle.holes
            ],
            target_name=self.puzzle.target,
        )
        self._disc_body = {d.name: (None if d.body == "world" else self.model.body(d.body).id) for d in self.tracker.discs}
        self.composer = RewardComposer.from_config(self.cfg.reward)
        self.log = EpisodeLog(log_path)
        self._renderer = None
        self.reset()

    # ------------------------------------------------------------ geometry
    def _body_radius(self, bid: int) -> float:
        r = 0.0
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == bid:
                r = max(r, float(np.linalg.norm(self.model.geom_pos[g]) + self.model.geom_rbound[g]))
        return r

    def frame_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        frame_ids = {self.model.body(b).id for b in self.built.frame_bodies}
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] in frame_ids:
                c, rb = self.data.geom_xpos[g], self.model.geom_rbound[g]
                lo, hi = np.minimum(lo, c - rb), np.maximum(hi, c + rb)
        return lo, hi

    def rope_nodes(self) -> np.ndarray:
        d = self.data
        pts = [d.xpos[b].copy() for b in self.rope_node_bodies]
        last = self.rope_node_bodies[-1]
        pts.append(d.xpos[last] + d.xmat[last].reshape(3, 3) @ np.array([self.built.rope.seg_lengths[-1], 0, 0]))
        return np.array(pts)

    def _update_discs(self) -> None:
        for disc in self.tracker.discs:
            bid = self._disc_body[disc.name]
            if bid is None:
                self.tracker.set_disc_pose(disc, np.zeros(3), np.eye(3))
            else:
                self.tracker.set_disc_pose(disc, self.data.xpos[bid], self.data.xmat[bid].reshape(3, 3))

    # ------------------------------------------------------------ goal
    def goal_metrics(self) -> dict:
        p = self.data.xpos[self.target_id]
        d_anchor = min(float(np.linalg.norm(p - a)) for a in self.anchors)
        lo, hi = self.frame_aabb()
        d_frame = float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))))
        need_anchor = self.built.rope.length + self.target_radius + self.puzzle.goal.margin
        need_frame = self.target_radius + self.puzzle.goal.margin
        return {
            "target_distance_to_nearest_anchor": round(d_anchor, 4),
            "required_anchor_distance": round(need_anchor, 4),
            "target_distance_to_frame_box": round(d_frame, 4),
            "required_frame_distance": round(need_frame, 4),
            "goal_reached": bool(d_anchor > need_anchor and d_frame > need_frame),
        }

    # ------------------------------------------------------------ physics
    def _substep(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.substeps += 1
        self.audit.check(self.data, self.substeps)
        if self.cfg.topology.enabled and self.tracker.discs:
            self._update_discs()
            self.tracker.update_target(self.data.xpos[self.target_id])

    def state_hash(self) -> str:
        buf = np.empty(mujoco.mj_stateSize(self.model, STATE_SIG))
        mujoco.mj_getState(self.model, self.data, buf, STATE_SIG)
        h = hashlib.sha256(buf.tobytes())
        h.update(self.model.eq_data.tobytes())
        return h.hexdigest()

    def reset(self) -> dict:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.eq_data[:] = self._eq_data0
        self.rig.reset()
        rng = np.random.default_rng(self.seed)
        if self.puzzle.init_noise > 0:
            for name in self.built.object_bodies:
                j = m.body(name).jntadr[0]
                adr = m.jnt_qposadr[j]
                d.qpos[adr : adr + 2] += rng.normal(0.0, self.puzzle.init_noise, size=2)
        mujoco.mj_forward(m, d)
        self.substeps = 0
        self.audit.reset()
        for _ in range(self.cfg.reset_settle_steps):
            mujoco.mj_step(m, d)
        settle_violations = self.audit.check(d, 0)
        self.audit.reset()
        d.time = 0.0
        self.actions_used = 0
        self.done = False
        self.success = False
        self.done_reason = None
        self.finalized = None
        self.tracker.reset()
        if self.cfg.topology.enabled and self.tracker.discs:
            self._update_discs()
            self.tracker.update_target(d.xpos[self.target_id])
            self.tracker.update_rope(self.rope_nodes())
        self._last_rstate = self._reward_state()
        self.composer.reset(self._last_rstate)
        self.initial_hash = self.state_hash()
        self.log.write(
            {
                "type": "header",
                "version": LOG_VERSION,
                "puzzle": self.puzzle.name,
                "puzzle_sha256": self.puzzle.source_sha256,
                "seed": self.seed,
                "config": self.cfg.model_dump(),
                "initial_state_hash": self.initial_hash,
                "max_actions": self.puzzle.max_actions,
                "settle_violations": settle_violations,
                "wall_time": time.time(),
            }
        )
        return self.get_state()

    def _reward_state(self) -> RewardState:
        return RewardState(
            puzzle=self.puzzle.name,
            actions_used=self.actions_used,
            max_actions=self.puzzle.max_actions,
            success=self.success,
            violations=self.audit.count,
            done=self.done,
            topology_word=self.tracker.word if self.cfg.topology.enabled else "",
        )

    # ------------------------------------------------------------ actions
    def act(self, tool: str, args: dict[str, Any] | None = None) -> dict:
        """Execute one physics action.

        Invalid requests raise RigError/EpisodeOver/ValueError *before* any
        physics step; they are logged as rejected and do not consume an action.
        """
        args = dict(args or {})
        try:
            if self.done:
                raise EpisodeOver(f"episode is over ({self.done_reason}); no further actions are accepted")
            if tool not in PHYSICS_TOOLS:
                raise RigError(f"unknown action {tool!r}")
            norm = self._normalise(tool, args)
            result = self._execute(tool, norm)
        except (RigError, EpisodeOver, TypeError, ValueError) as e:
            self.log.write({"type": "rejected", "tool": tool, "args": args, "error": str(e), "wall_time": time.time()})
            raise
        return self._finish_action(tool, norm, result)

    def _normalise(self, tool: str, a: dict) -> dict:
        def need_handle():
            h = a.get("handle")
            if isinstance(h, bool) or not isinstance(h, (int, float)) or int(h) != h:
                raise RigError("handle must be an integer")
            return int(h)

        if tool == "grab":
            norm = {"handle": need_handle(), "target": str(a.get("target", ""))}
            if a.get("point") is not None:
                norm["point"] = [float(x) for x in a["point"]]
            if a.get("segment") is not None:
                norm["segment"] = int(a["segment"])
            return norm
        if tool == "move":
            norm = {"handle": need_handle(), "frame": str(a.get("frame", "world"))}
            for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw"):
                norm[k] = float(a.get(k, 0.0) or 0.0)
            return norm
        if tool == "release":
            return {"handle": need_handle()}
        steps = a.get("steps", 100)
        if isinstance(steps, bool) or int(steps) != steps:
            raise RigError("steps must be an integer")
        return {"steps": int(np.clip(int(steps), 1, self.cfg.rig.max_wait_steps))}

    def _execute(self, tool: str, a: dict) -> dict:
        if tool == "grab":
            return self.rig.grab(a["handle"], a["target"], a.get("point"), a.get("segment"))
        if tool == "move":
            return self.rig.move(a["handle"], [a["dx"], a["dy"], a["dz"]], [a["droll"], a["dpitch"], a["dyaw"]], a["frame"])
        if tool == "release":
            return self.rig.release(a["handle"])
        for _ in range(a["steps"]):
            self._substep()
        return {"waited_steps": a["steps"], "sim_time": round(self.data.time, 4)}

    def _finish_action(self, tool: str, norm: dict, result: dict) -> dict:
        self.actions_used += 1
        if self.cfg.topology.enabled and self.tracker.discs:
            self._update_discs()
            self.tracker.update_rope(self.rope_nodes())
        goal = self.goal_metrics()
        if self.audit.count > 0:
            self.success = False
            self.done, self.done_reason = True, "anti-tunnelling violation (episode failed)"
        elif goal["goal_reached"]:
            self.success = True
            self.done, self.done_reason = True, "solved"
        elif self.actions_used >= self.puzzle.max_actions:
            self.done, self.done_reason = True, "action budget exhausted"
        rstate = self._reward_state()
        step_reward = self.composer.step(rstate)
        self._last_rstate = rstate
        h = self.state_hash()
        self.log.write(
            {
                "type": "action",
                "i": self.actions_used,
                "tool": tool,
                "args": norm,
                "result": result,
                "state_hash": h,
                "reward": round(step_reward, 6),
                "success": self.success,
                "done": self.done,
                "violations": self.audit.count,
                "topology": self.tracker.words if self.cfg.topology.enabled else None,
                "sim_time": round(self.data.time, 4),
                "wall_time": time.time(),
            }
        )
        if self.done:
            self.finalize()
        return {
            **result,
            "action": self.actions_used,
            "actions_remaining": self.puzzle.max_actions - self.actions_used,
            "done": self.done,
            "done_reason": self.done_reason,
            "success": self.success,
            "violations": self.audit.count,
            "target": self._target_summary(),
            "goal": goal,
        }

    def log_observation(self, tool: str, args: dict) -> None:
        self.log.write({"type": "observe", "tool": tool, "args": args, "i": self.actions_used, "wall_time": time.time()})

    def finalize(self) -> dict:
        """Close the episode: terminal reward, final record. Idempotent."""
        if self.finalized is not None:
            return self.finalized
        self.done = True
        self.done_reason = self.done_reason or "stopped by agent/harness"
        final_state = self._reward_state()
        self.composer.terminal(final_state)
        score = self.composer.score()
        self.finalized = {
            "type": "final",
            "puzzle": self.puzzle.name,
            "score": round(score, 6),
            "success": self.success,
            "actions_used": self.actions_used,
            "max_actions": self.puzzle.max_actions,
            "done_reason": self.done_reason,
            "violations": self.audit.summary(),
            "breakdown": self.composer.breakdown(),
            "final_state_hash": self.state_hash(),
            "topology": self.tracker.words if self.cfg.topology.enabled else None,
            "goal": self.goal_metrics(),
        }
        self.log.write({**self.finalized, "wall_time": time.time()})
        self.log.close()
        return self.finalized

    # ------------------------------------------------------------ observation
    def _target_summary(self) -> dict:
        bid = self.target_id
        return {"name": self.puzzle.target, "pos": _r(self.data.xpos[bid]), "quat": _r(self.data.xquat[bid])}

    def scene_description(self) -> dict:
        """Static description of the frame (from the puzzle spec) and the holes."""
        frame = []
        for b in self.puzzle.frame:
            for g in b.geoms:
                e = {"body": b.name, "type": g.type}
                if g.type in ("capsule", "cylinder"):
                    e.update(fromto=list(g.fromto), radius=g.size[0])
                elif g.type == "capsule_ring":
                    e.update(center=list(g.pos), normal=list(g.normal), ring_radius=g.radius, tube_radius=g.tube_radius)
                elif g.type == "mesh":
                    e.update(path=Path(g.path).name, pos=list(g.pos))
                else:
                    e.update(pos=list(g.pos), size=list(g.size))
                frame.append(e)
        return {"frame_parts": frame, "frame_body_offsets": {b.name: list(b.pos) for b in self.puzzle.frame}}

    def get_state(self) -> dict:
        d = self.data
        objects = {}
        for name in self.built.object_bodies:
            bid = self.model.body(name).id
            objects[name] = {
                "pos": _r(d.xpos[bid]),
                "quat": _r(d.xquat[bid]),
                "graspable": name in self.built.graspable,
                "bounding_radius": round(self._body_radius(bid), 4),
            }
        holes = []
        if self.tracker.discs:
            self._update_discs()
            for disc in self.tracker.discs:
                holes.append({"name": disc.name, "attached_to": disc.body, "center": _r(disc.center), "normal": _r(disc.normal), "radius": disc.radius})
        rope = self.built.rope
        state = {
            "puzzle": self.puzzle.name,
            "difficulty": self.puzzle.difficulty,
            "target": self.puzzle.target,
            "sim_time": round(d.time, 4),
            "actions_used": self.actions_used,
            "max_actions": self.puzzle.max_actions,
            "actions_remaining": self.puzzle.max_actions - self.actions_used,
            "done": self.done,
            "done_reason": self.done_reason,
            "success": self.success,
            "violations": self.audit.count,
            "objects": objects,
            "rope": {
                "name": rope.name,
                "num_segments": len(rope.body_names),
                "segment_length": round(float(np.mean(rope.seg_lengths)), 5),
                "radius": rope.radius,
                "length": round(rope.length, 4),
                "anchors": [_r(a) for a in self.anchors],
                "nodes": _r(self.rope_nodes()),
                "note": "nodes[k] is the start of segment k; grab a segment with target='rope', segment=k",
            },
            "handles": self.rig.status(),
            "holes": holes,
            "goal": self.goal_metrics(),
            "scene": self.scene_description(),
            "limits": {
                "max_linear_step_m": self.cfg.rig.max_linear_step,
                "max_angular_step_rad": self.cfg.rig.max_angular_step,
                "workspace_min": self.cfg.rig.workspace_min,
                "workspace_max": self.cfg.rig.workspace_max,
                "grasp_tolerance_m": self.cfg.rig.grasp_tolerance,
            },
        }
        if self.cfg.topology.enabled and self.cfg.topology.expose_in_state:
            state["topology"] = self.tracker.summary()
        return state

    def render(self, view: str | None = None, azimuth=None, elevation=None, distance=None, lookat=None) -> tuple[bytes, str]:
        from .render import SceneRenderer

        if self._renderer is None:
            lo, hi = self.frame_aabb()
            center = (lo + hi) / 2
            span = float(np.max(hi - lo))
            self._renderer = SceneRenderer(self.model, self.cfg.render.width, self.cfg.render.height, center, 1.6 * span + 0.2)
        cam = self._renderer.camera(view, azimuth, elevation, distance, lookat)
        png = self._renderer.render(self.data, cam)
        t = self._target_summary()
        held = ", ".join(f"handle{h['id']}={'holding ' + h['holding'] if h['holding'] else 'empty'}" for h in self.rig.status())
        caption = (
            f"view={view or 'iso'} az={cam.azimuth:.0f} el={cam.elevation:.0f} dist={cam.distance:.2f} "
            f"lookat={_r(cam.lookat, 3)} | {t['name']} at {t['pos']} | {held} | "
            f"actions {self.actions_used}/{self.puzzle.max_actions}"
            f"{' | DONE: ' + str(self.done_reason) if self.done else ''}. "
            "Colours: frame grey, rope orange, target blue, handle0 red, handle1 green."
        )
        return png, caption

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self.log.close()
