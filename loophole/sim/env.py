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

from ..config import EnvConfig, effective_config, load_config
from ..reward.components import RewardComposer, RewardState, StableMin
from ..reward.progress import progress_potential
from ..reward.roadmap import Roadmap
from ..reward.topology import Disc, TopologyTracker
from .audit import Auditor
from .builder import BuiltModel, build_model
from .collision import CollisionChecker, fibonacci_directions, interpolate
from .rig import Rig, RigError
from .spec import Puzzle, load_puzzle, resolve_puzzle, to_native

PHYSICS_TOOLS = ("grab", "move", "release", "wait")
STATE_SIG = mujoco.mjtState.mjSTATE_INTEGRATION
LOG_VERSION = 1


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


class EpisodeOver(RuntimeError):
    pass


def _load_centerline(ref: str, puzzle: Puzzle) -> np.ndarray:
    """Centre-line polyline (native units, body frame) -> simulator units."""
    if ref.startswith("paper:"):
        from ..puzzles.paper_alpha import resolve_mesh_path as resolve
    elif ref.startswith("ompl:"):
        from ..puzzles.ompl_alpha import resolve_mesh_path as resolve
    else:
        def resolve(r):
            return Path(puzzle.source_path).parent / r
    return np.load(resolve(ref)) * puzzle.unit_scale


def _gap_indices(P: np.ndarray, min_sep: int = 15) -> tuple[int, int] | None:
    """Indices of the alpha's crossing: closest pair of centre-line points far apart along it."""
    if len(P) < 2 * min_sep:
        return None
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    idx = np.arange(len(P))
    D[np.abs(idx[:, None] - idx[None]) < min_sep] = np.inf
    i, j = sorted(np.unravel_index(int(np.argmin(D)), D.shape))
    return int(i), int(j)


def _closed_loop(P: np.ndarray, min_sep: int = 15) -> np.ndarray | None:
    """An alpha's loop closed across its gap: the centre-line between its crossing
    (closest pair of points far apart along the curve) plus the closing segment."""
    if len(P) < 2 * min_sep:
        return None
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    idx = np.arange(len(P))
    D[np.abs(idx[:, None] - idx[None]) < min_sep] = np.inf
    i, j = sorted(np.unravel_index(int(np.argmin(D)), D.shape))
    return np.vstack([P[i : j + 1], P[i : i + 1]])


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
        start_pose=None,
    ):
        self.puzzle = puzzle if isinstance(puzzle, Puzzle) else load_puzzle(resolve_puzzle(str(puzzle)))
        self.cfg = effective_config(config or load_config(), self.puzzle)
        self.seed = int(seed)
        # training only: a non-canonical start pose of the free piece (None = the puzzle's own start;
        # evaluation always uses None, and the verifier rejects logs whose header has a start_pose)
        self.start_pose = None if start_pose is None else np.asarray(start_pose, float)
        self.episode_meta: dict | None = None
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
        self.certifier = None  # optional swept-motion certificate (enable_motion_certificate), alongside the audit
        self.rig = Rig(
            self.model, self.data, self.built, self.cfg.rig, self._substep,
            strain=lambda: (self.audit.last_penetration, self.audit.last_anchor_drift),
            abort=lambda: self.audit.count > 0,
        )
        self.target_id = self.model.body(self.puzzle.target).id
        self.target_radius = self._body_radius(self.target_id)
        rope = self.built.rope
        self.rope_node_bodies = [self.model.body(b).id for b in rope.body_names] if rope else []
        anchors = [] if rope is None else [rope.start] + ([rope.end] if rope.end is not None else [])
        self.anchors = [np.asarray(a, dtype=float) for a in anchors]
        self.checker = CollisionChecker(self.puzzle) if self.puzzle.goal.type == "separation" else None
        self._escape_dirs = fibonacci_directions(self.cfg.goal.escape_directions) if self.checker is not None else None
        self._escape_cache: tuple[bytes, np.ndarray | None] | None = None
        self._escape_first: int | None = None
        # planning queries need the rigid-pair collision checker (separation puzzles only)
        planning = {"check_motion", "check_poses", "get_geometry"}
        self.tools = [t for t in self.cfg.observation.tool_names() if self.checker is not None or t not in planning]
        # tube centre-lines (body frame) are tracked as curves by the topology diagnostics
        self.curves = {
            b.name: np.concatenate([np.asarray(g.points, float) for g in b.geoms if g.type == "tube"])
            for b in self.puzzle.frame + self.puzzle.objects
            if any(g.type == "tube" for g in b.geoms)
        }
        for b in self.puzzle.frame + self.puzzle.objects:  # mesh bodies with an extracted centre-line
            if b.centerline:
                self.curves[b.name] = _load_centerline(b.centerline, self.puzzle)
        self.loops = {name: _closed_loop(c) for name, c in self.curves.items()}
        self._tube_radius = self._estimate_tube_radius()
        self.tracker = TopologyTracker(
            [
                Disc(h.name, h.body, np.asarray(h.center, float), np.asarray(h.normal, float) / np.linalg.norm(h.normal), h.radius)
                for h in self.puzzle.holes
            ],
            target_name=self.puzzle.target,
            primary=self.cfg.topology.primary,
        )
        self._disc_body = {d.name: (None if d.body == "world" else self.model.body(d.body).id) for d in self.tracker.discs}
        # WIP cost-to-go roadmap reward, behind reward.roadmap (off by default)
        self.roadmap = Roadmap.for_puzzle(self.puzzle) if (self.checker is not None and self.cfg.reward.roadmap) else None
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
        if not self.rope_node_bodies:
            return np.empty((0, 3))
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
    def target_pose(self) -> np.ndarray:
        """Target pose relative to the fixed frame (which sits at the origin): x y z qw qx qy qz."""
        return np.concatenate([self.data.xpos[self.target_id], self.data.xquat[self.target_id]])

    def curve_points(self, body: str) -> np.ndarray:
        bid = self.model.body(body).id
        return self.data.xpos[bid] + self.curves[body] @ self.data.xmat[bid].reshape(3, 3).T

    def goal_metrics(self) -> dict:
        if self.puzzle.goal.type == "separation":
            pose = self.target_pose()
            if self.cfg.goal.criterion == "escape":
                u = self.escape_direction(pose)
                return {
                    "criterion": "free escape by translation",
                    "goal_reached": u is not None,
                    "escape_direction": None if u is None else _r(u, 4),
                }
            d_sep = self.puzzle.goal.d_sep
            return {
                "criterion": "distance",
                "target_min_distance_to_frame": round(self.checker.clearance(pose, 2 * d_sep), 5),
                "required_separation": d_sep,
                "goal_reached": self.checker.is_separated(pose),
            }
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

    def escape_direction(self, pose: np.ndarray | None = None) -> np.ndarray | None:
        """A direction along which the target slides to infinity untouched, or None (cached per pose)."""
        pose = self.target_pose() if pose is None else np.asarray(pose, float)
        key = pose.round(9).tobytes()
        if self._escape_cache is None or self._escape_cache[0] != key:
            u = self.checker.escape_direction(pose, self._escape_dirs, first=self._escape_first)
            if u is not None:
                self._escape_first = self.checker.escape_index
            self._escape_cache = (key, u)
        return self._escape_cache[1]

    def min_distance(self) -> float | None:
        """Minimum target-frame distance (analysis / training signal only; never shown in `sensors`)."""
        if self.checker is None:
            return None
        return round(self.checker.clearance(self.target_pose(), max(0.8, 2 * self.puzzle.goal.d_sep)), 5)

    def _update_curves(self) -> None:
        if self.rope_node_bodies:
            self.tracker.update_curve("rope", self.rope_nodes())
        for body in self.curves:
            self.tracker.update_curve(body, self.curve_points(body), owner=body)

    # ------------------------------------------------------------ physics
    def _substep(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.substeps += 1
        self.audit.check(self.data, self.substeps)
        if self.certifier is not None:
            self.certifier.after_substep(self.substeps)
        if self.cfg.topology.enabled and self.tracker.discs:
            self._update_discs()
            self.tracker.update_target(self.data.xpos[self.target_id])

    def enable_motion_certificate(self, slack: float | None = None, max_depth: int = 12):
        """Certify every substep interval of the target's rigid motion (sim/certify.py), alongside the penetration
        audit. Off by default and outside the config, so evaluation logs and scores are unchanged; uncertified
        intervals are tallied in `self.certifier.log`, not counted as violations."""
        if self.checker is None:
            raise ValueError("motion certificate needs a separation puzzle with a collision checker")
        from .certify import RigidMotionCertifier

        self.certifier = RigidMotionCertifier(self, slack=slack, max_depth=max_depth)
        self.certifier.prime()
        return self.certifier

    def state_hash(self) -> str:
        buf = np.empty(mujoco.mj_stateSize(self.model, STATE_SIG))
        mujoco.mj_getState(self.model, self.data, buf, STATE_SIG)
        h = hashlib.sha256(buf.tobytes())
        h.update(self.model.eq_data.tobytes())
        return h.hexdigest()

    def reset(self, seed: int | None = None, log_path: str | Path | None = None) -> dict:
        """Start an episode. Evaluation constructs a fresh env and never calls this again; training
        may reset repeatedly (new `seed`, new per-episode `log_path`, `self.start_pose`)."""
        if seed is not None:
            self.seed = int(seed)
        if log_path is not None:
            self.log.close()
            self.log = EpisodeLog(log_path)
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
        if self.start_pose is not None:
            adr = m.jnt_qposadr[m.body(self.puzzle.target).jntadr[0]]
            d.qpos[adr : adr + 3] = self.start_pose[:3]
            d.qpos[adr + 3 : adr + 7] = self.start_pose[3:7] / np.linalg.norm(self.start_pose[3:7])
        mujoco.mj_forward(m, d)
        self.substeps = 0
        self.audit.reset()
        for _ in range(self.cfg.reset_settle_steps):
            mujoco.mj_step(m, d)
        settle_violations = self.audit.check(d, 0)
        self.audit.reset()
        self.audit.prime(d)
        if self.certifier is not None:
            self.certifier.prime()
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
            self._update_curves()
        self.ctg = self.ctg0 = self.best_ctg = None
        self.ctg_fallbacks = 0
        self._last_min_distance = None
        self._stable = StableMin(None)
        if self.roadmap is not None:
            self.ctg, self._ctg_node = self.roadmap.cost_to_go(self.target_pose(), self.checker)
            self.ctg0 = self.best_ctg = self.ctg
        self._stable = StableMin(self.ctg0, hold=3)
        self._last_min_distance = self.min_distance()
        self._escape_cache, self._escape_first = None, None
        self._last_action = None
        self._last_rstate = self._reward_state()
        self.composer.reset(self._last_rstate)
        self.initial_hash = self.state_hash()
        extra = {}
        if self.start_pose is not None:
            extra["start_pose"] = [float(x) for x in self.start_pose]  # full precision: replays must reproduce it
        if self.episode_meta:
            extra["training"] = self.episode_meta
        self.log.write(
            {
                **extra,
                "type": "header",
                "version": LOG_VERSION,
                "puzzle": self.puzzle.name,
                "puzzle_sha256": self.puzzle.source_sha256,
                "seed": self.seed,
                "config": self.cfg.model_dump(),
                "initial_state_hash": self.initial_hash,
                "roadmap_sha256": self.roadmap.sha256 if self.roadmap is not None else None,
                "start_cost_to_go": self.ctg0,
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
            cost_to_go=self.ctg,
            start_cost_to_go=self.ctg0,
            best_cost_to_go=self.best_ctg,
            stable_best_cost_to_go=self._stable.best,
            min_distance=self._last_min_distance,
        )

    def _update_cost_to_go(self) -> bool:
        """Recompute the roadmap cost-to-go; keep the previous value if no node is visible."""
        if self.roadmap is None:
            return False
        c, node = self.roadmap.cost_to_go(self.target_pose(), self.checker)
        self._stable.update(c)  # None (no visible node) breaks the 3-action hold streak
        if c is None:
            self.ctg_fallbacks += 1
            return True
        self.ctg, self._ctg_node = c, node
        self.best_ctg = c if self.best_ctg is None else min(self.best_ctg, c)
        return False

    def progress_summary(self) -> dict | None:
        if self.roadmap is None or not self.ctg0:
            return None
        return {
            "cost_to_go": None if self.ctg is None else round(self.ctg, 5),
            "start_cost_to_go": round(self.ctg0, 5),
            "progress": None if self.ctg is None else round(1.0 - self.ctg / self.ctg0, 5),
            "best_progress": round(1.0 - self.best_ctg / self.ctg0, 5),
            "note": "1 = separated, 0 = as far as the start (shortest collision-free route, estimated)",
        }

    def _world(self, body: str, pts: np.ndarray) -> np.ndarray:
        bid = self.model.body(body).id
        return self.data.xpos[bid] + pts @ self.data.xmat[bid].reshape(3, 3).T

    def diagnostics(self) -> dict:
        """Topological quantities of the two tubes (Gauss linking, Klenin & Langowski 2000).

        linking_open:   integral over the whole open centre-lines (real-valued; geometric)
        linking_closed: linking number of the two alphas' loops, each closed across its own
                        gap (integer: +-1 interlocked, 0 once a tube has passed a gap)
        """
        out = {}
        if len(self.curves) >= 2:
            from ..puzzles.alpha.geometry import gauss_linking

            a, b = list(self.curves)[:2]
            out["linking_open"] = round(gauss_linking(self.curve_points(a), self.curve_points(b)), 5)
            if self.loops.get(a) is not None and self.loops.get(b) is not None:
                out["linking_closed"] = round(gauss_linking(self._world(a, self.loops[a]), self._world(b, self.loops[b])), 5)
            out["linking_integral"] = out.get("linking_closed", out["linking_open"])
        return out

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
            self.audit.window_penetration = 0.0
            self._handles_before = [self.rig.handle_pose(h) for h in self.rig.handles]
            result = self._execute(tool, norm)
        except (RigError, EpisodeOver, TypeError, ValueError) as e:
            self.log.write({"type": "rejected", "tool": tool, "args": args, "error": str(e), "wall_time": time.time()})
            raise
        return self._finish_action(tool, norm, result)

    def _normalise(self, tool: str, a: dict) -> dict:
        def need_handle():
            h = a.get("handle")
            if isinstance(h, bool) or not isinstance(h, (int, float)) or not np.isfinite(h) or int(h) != h:
                raise RigError("handle must be an integer")
            return int(h)

        def finite(x, name):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
                raise RigError(f"{name} must be a finite number")
            return float(x)

        if tool == "grab":
            norm = {"handle": need_handle(), "target": str(a.get("target", ""))[:64]}
            if a.get("point") is not None:
                pt = a["point"]
                if not isinstance(pt, (list, tuple)) or len(pt) != 3:
                    raise RigError("point must be [x, y, z]")
                norm["point"] = [finite(x, "point") for x in pt]
            if a.get("segment") is not None:
                seg = a["segment"]
                if isinstance(seg, bool) or not isinstance(seg, (int, float)) or not np.isfinite(seg) or int(seg) != seg:
                    raise RigError("segment must be a finite integer")
                norm["segment"] = int(seg)
            return norm
        if tool == "move":
            norm = {"handle": need_handle(), "frame": str(a.get("frame", "world"))}
            for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw"):
                v = a.get(k, 0.0)
                norm[k] = finite(0.0 if v is None else v, k)  # huge finite values are clamped by the rig
            return norm
        if tool == "release":
            return {"handle": need_handle()}
        steps = a.get("steps", 100)
        if isinstance(steps, bool) or not isinstance(steps, (int, float)) or not np.isfinite(steps) or int(steps) != steps:
            raise RigError("steps must be a finite integer")
        if not 1 <= int(steps) <= self.cfg.rig.max_wait_steps:
            raise RigError(f"steps must be in [1, {self.cfg.rig.max_wait_steps}]")
        return {"steps": int(steps)}

    def _execute(self, tool: str, a: dict) -> dict:
        if tool == "grab":
            return self.rig.grab(a["handle"], a["target"], a.get("point"), a.get("segment"))
        if tool == "move":
            return self.rig.move(a["handle"], [a["dx"], a["dy"], a["dz"]], [a["droll"], a["dpitch"], a["dyaw"]], a["frame"])
        if tool == "release":
            return self.rig.release(a["handle"])
        for _ in range(a["steps"]):
            self._substep()
            if self.audit.count > 0:  # end immediately on any violation
                break
        return {"waited_steps": a["steps"], "sim_time": round(self.data.time, 4)}

    def _finish_action(self, tool: str, norm: dict, result: dict) -> dict:
        self.actions_used += 1
        if self.cfg.topology.enabled and self.tracker.discs:
            self._update_discs()
            self._update_curves()
        fallback = self._update_cost_to_go()
        goal = self.goal_metrics()
        self._last_min_distance = self.min_distance()
        if self.audit.count > 0:
            self.success = False
            self.done, self.done_reason = True, "anti-tunnelling violation (episode failed)"
        elif goal["goal_reached"]:
            self.success = True
            self.done, self.done_reason = True, "solved"
        elif self.actions_used >= self.puzzle.max_actions:
            self.done, self.done_reason = True, "action budget exhausted"
        rstate = self._reward_state()
        diag = self.diagnostics()
        info = {  # what dense reward components may read (all measured by the simulator)
            "goal": goal,
            "min_distance": self._last_min_distance,
            "last_penetration": self.audit.window_penetration,
            "args": norm,  # as requested (validated, but not clamped)
            "executed": self._executed_motion(),  # what the rig actually did (clamped, maybe blocked)
            "tool": tool,
            "diagnostics": diag,
            "linking_integral": diag.get("linking_integral"),
        }
        step_reward = self.composer.step(rstate, info)
        self.last_step_reward = float(step_reward)  # training signal (never part of the score)
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
                "cost_to_go": self.ctg,
                "ctg_fallback": fallback,
                # analysis only (not scored): relative pose of the target w.r.t. the fixed frame,
                # min distance, and the (flagged-off) progress potential hook
                "rel_pose": _r(self.target_pose(), 6),
                "rel_pose_native": None if self.puzzle.unit_scale == 1.0 else to_native(self.target_pose()[:3].tolist(), self.puzzle, 4) + _r(self.target_pose()[3:], 6),
                "min_distance": self._last_min_distance,
                "escape_direction": goal.get("escape_direction"),
                "progress_potential": progress_potential(self),
                "diagnostics": diag,
                "rewards": self.composer.last_step,  # per-component step rewards (training signal)
                "sim_time": round(self.data.time, 4),
                "wall_time": time.time(),
            }
        )
        executed = info["executed"]
        self._last_action = {"i": self.actions_used, "tool": tool, "blocked": bool(result.get("blocked", False)),
                             "clamped": result.get("clamped", []),
                             "executed_translation_m": round(executed["translation"], 5),
                             "executed_rotation_rad": round(executed["rotation"], 5)}
        if self.done:
            self.finalize()
        out = {
            **result,
            "action": self.actions_used,
            "actions_remaining": self.puzzle.max_actions - self.actions_used,
            "done": self.done,
            "done_reason": self.done_reason,
            "success": self.success,
            "violations": self.audit.count,
            "target": self._target_summary(),
        }
        if self.privileged:
            out.update(goal=goal, progress=self.progress_summary())
        return out

    @property
    def privileged(self) -> bool:
        """`full` observation profile: goal metrics, progress and scene geometry are shown to the agent."""
        return self.cfg.observation.profile == "full"

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
            "cost_to_go": {"start": self.ctg0, "final": self.ctg, "best": self.best_ctg, "stable_best": self._stable.best, "fallbacks": self.ctg_fallbacks},
        }
        self.log.write({**self.finalized, "wall_time": time.time()})
        self.log.close()
        return self.finalized

    # ------------------------------------------------------------ observation
    def _target_summary(self) -> dict:
        bid = self.target_id
        out = {"name": self.puzzle.target, "pos": _r(self.data.xpos[bid], 5), "quat": _r(self.data.xquat[bid], 5)}
        if self.puzzle.unit_scale != 1.0:
            out["pos_native_units"] = to_native(self.data.xpos[bid].tolist(), self.puzzle, 4)
        return out

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
                elif g.type == "tube":
                    e.update(tube_radius=g.tube_radius, centerline=_r(np.asarray(g.points), 4))
                elif g.type == "mesh":
                    e.update(path=Path(g.path).name, pos=list(g.pos))
                else:
                    e.update(pos=list(g.pos), size=list(g.size))
                frame.append(e)
        out = {"frame_parts": frame, "frame_body_offsets": {b.name: list(b.pos) for b in self.puzzle.frame}}
        movers = {}
        for b in self.puzzle.objects:
            tubes = [g for g in b.geoms if g.type == "tube"]
            if tubes:
                movers[b.name] = {
                    "tube_radius": tubes[0].tube_radius,
                    "centerline_body_frame": _r(self.curves[b.name], 4),
                    "centerline_world": _r(self.curve_points(b.name), 4),
                }
        if movers:
            out["objects"] = movers
        if self.puzzle.unit_scale != 1.0:
            out["unit_scale"] = self.puzzle.unit_scale
        return out

    def get_state(self) -> dict:
        """Proprioception (`sensors` / `planner` profiles) or the full privileged state (`full`)."""
        return self.full_state() if self.privileged else self.proprioception()

    def proprioception(self) -> dict:
        """What a robot knows without looking: its own handle, the held piece's pose (from the
        grasp), the action budget, the outcome of its last action, and its limits."""
        d = self.data
        objects = {}
        for name in self.built.object_bodies:
            bid = self.model.body(name).id
            objects[name] = {"pos": _r(d.xpos[bid], 5), "quat": _r(d.xquat[bid], 5), "graspable": name in self.built.graspable}
        return {
            "puzzle": self.puzzle.name,
            "frame": "world = the fixed piece's frame; metres, radians, quaternions w x y z",
            "fixed": [b.name for b in self.puzzle.frame],
            "objects": objects,
            "handles": self.rig.status(),
            "last_action": self._last_action,
            "actions_used": self.actions_used,
            "max_actions": self.puzzle.max_actions,
            "actions_remaining": self.puzzle.max_actions - self.actions_used,
            "done": self.done,
            "done_reason": self.done_reason,
            "success": self.success,
            "violations": self.audit.count,
            "limits": {
                "max_linear_step_m": self.cfg.rig.max_linear_step,
                "max_angular_step_rad": self.cfg.rig.max_angular_step,
                "linear_speed_m_s": self.cfg.rig.linear_speed,
                "angular_speed_rad_s": self.cfg.rig.angular_speed,
                "workspace_min": self.cfg.rig.workspace_min,
                "workspace_max": self.cfg.rig.workspace_max,
                "grasp_tolerance_m": self.cfg.rig.grasp_tolerance,
            },
            "tools": self.tools,
        }

    def full_state(self) -> dict:
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
            "rope": None if rope is None else {
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
            "progress": self.progress_summary(),
            "progress_potential": progress_potential(self),  # 0 unless reward.roadmap is on (see reward/progress.py)
            "geometry": self.geometry_description(),
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

    def _seg_labels(self) -> np.ndarray:
        """geom id -> segmentation label (0 background, 1 fixed piece, 2 free piece, 3 handle, 4 rope)."""
        if getattr(self, "_seg_lut", None) is None:
            m = self.model
            frame = {m.body(b).id for b in self.built.frame_bodies}
            objects = {m.body(b).id for b in self.built.object_bodies}
            rope = set(self.rope_node_bodies)
            lut = np.zeros(m.ngeom + 1, np.uint8)  # last entry: background (-1)
            for g in range(m.ngeom):
                bid = m.geom_bodyid[g]
                name = m.body(bid).name
                lut[g] = 1 if bid in frame else 2 if bid in objects else 4 if bid in rope else 3 if name.startswith("handle") else 0
            self._seg_lut = lut
        return self._seg_lut

    def _camera(self, view, azimuth, elevation, distance, lookat):
        from .render import SceneRenderer

        if self._renderer is None:  # default framing: the fixed piece and the free one at its start pose
            lo, hi = self.frame_aabb()
            p, r = self.data.xpos[self.target_id], self.target_radius
            lo, hi = np.minimum(lo, p - r), np.maximum(hi, p + r)
            center = (lo + hi) / 2
            span = float(np.max(hi - lo))
            self._renderer = SceneRenderer(self.model, self.cfg.render.width, self.cfg.render.height, center, max(0.6, 1.3 * span))
        for name, v in (("azimuth", azimuth), ("elevation", elevation), ("distance", distance)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)):
                raise ValueError(f"{name} must be a finite number")
        if distance is not None and not 0.05 <= distance <= 20.0:
            raise ValueError("distance must be in [0.05, 20] m")
        if lookat is not None:
            if not isinstance(lookat, (list, tuple)) or len(lookat) != 3 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in lookat):
                raise ValueError("lookat must be three finite numbers")
        return self._renderer.camera(view, azimuth, elevation, distance, lookat)

    def render(self, view: str | None = None, azimuth=None, elevation=None, distance=None, lookat=None) -> tuple[bytes, str]:
        """One RGB image + caption (kept for the viewer, scripts and tests)."""
        imgs, caption = self.render_views(view, azimuth, elevation, distance, lookat)
        return imgs[0][1], caption

    def render_views(self, view: str | None = None, azimuth=None, elevation=None, distance=None, lookat=None,
                     modalities=("rgb",), width: int | None = None, height: int | None = None,
                     depth_format: str = "visual") -> tuple[list[tuple[str, bytes]], str]:
        """Camera images: rgb, depth and/or segmentation (PNGs), plus a JSON caption with the full
        camera model (intrinsics + pose) so pixels can be back-projected to 3-D points.

        depth_format: "visual" = 8-bit greyscale, white = near, black = far (range in the caption);
                      "metric" = 16-bit greyscale PNG in millimetres along the optical axis (0 = background).
        segmentation: colour-coded labels (see caption: segmentation_labels).
        """
        from .render import SEG_COLOURS, depth_visual, encode_png, encode_png16

        cam = self._camera(view, azimuth, elevation, distance, lookat)
        mods = list(modalities) if modalities else ["rgb"]
        if not set(mods) <= {"rgb", "depth", "segmentation"} or len(set(mods)) != len(mods):
            raise ValueError("modalities: a list of distinct 'rgb', 'depth', 'segmentation'")
        if depth_format not in ("visual", "metric"):
            raise ValueError("depth_format must be 'visual' or 'metric'")
        W = self.cfg.render.width if width is None else width
        H = self.cfg.render.height if height is None else height
        for name, v, hi in (("width", W, 1280), ("height", H, 960)):
            if isinstance(v, bool) or not isinstance(v, int) or not 64 <= v <= hi:
                raise ValueError(f"{name} must be an integer in [64, {hi}]")
        arrays, camera = self._renderer.render_modalities(self.data, cam, mods, W, H)
        images, info = [], {}
        for mname in mods:
            a = arrays[mname]
            if mname == "rgb":
                images.append(("rgb", encode_png(a)))
            elif mname == "depth":
                bg = a >= 0.999 * camera["zfar"]
                fg = a[~bg]
                if depth_format == "metric":
                    mm = np.where(bg, 0, np.clip(np.round(a * 1000), 1, 65535)).astype(np.uint16)
                    images.append(("depth", encode_png16(mm)))
                    info["depth"] = {"format": "metric", "encoding": "16-bit greyscale PNG, millimetres along the optical axis, 0 = background"}
                else:
                    near = float(fg.min()) if fg.size else 0.0
                    far = float(fg.max()) if fg.size else 1.0
                    d = np.where(bg, np.inf, a)
                    images.append(("depth", encode_png(depth_visual(d, near, far))))
                    info["depth"] = {"format": "visual", "near_m": round(near, 4), "far_m": round(far, 4),
                                     "encoding": "8-bit grey: depth = far - (grey / 255) * (far - near); black = background"}
            else:
                lut = self._seg_labels()
                lab = lut[np.where(a < 0, len(lut) - 1, a)]
                images.append(("segmentation", encode_png(SEG_COLOURS[lab])))
                info["segmentation_labels"] = {"black": "background", "grey": self._piece_name(1), "blue": self._piece_name(2),
                                               "red": "handle", "orange": "rope"}
        t = self._target_summary()
        cap = {
            "view": view or "iso", "azimuth": round(cam.azimuth, 3), "elevation": round(cam.elevation, 3),
            "distance": round(cam.distance, 4), "lookat": _r(cam.lookat, 4), "images": [k for k, _ in images],
            "camera": camera, **info,
            "colours": self._colour_legend(),
            "actions": f"{self.actions_used}/{self.puzzle.max_actions}",
        }
        if self.done:
            cap["done"] = self.done_reason
        if self.privileged:
            cap["target"] = t
            cap["goal"] = self.goal_metrics()
            cap["progress"] = self.progress_summary()
        return images, json.dumps(cap, separators=(",", ":"))

    def _piece_name(self, label: int) -> str:
        if label == 1:
            return ", ".join(self.built.frame_bodies) + " (fixed)"
        return ", ".join(self.built.object_bodies) + " (free)"

    # ------------------------------------------------------------ touch
    def get_contacts(self, max_contacts: int = 64) -> dict:
        """Touch: current contacts between the held/free piece and everything else, with normals,
        penetration (negative distance) and normal force; plus the total contact wrench on the
        free piece and how far the piece lags its handle (a strain gauge in the grip)."""
        m, d = self.model, self.data
        targets = {m.body(b).id for b in self.built.object_bodies}
        out, F, T = [], np.zeros(3), np.zeros(3)
        f6 = np.zeros(6)
        tpos = d.xpos[self.target_id]
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in targets) == (b2 in targets):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            n = np.asarray(c.frame[:3], float)  # from geom1 to geom2
            if b1 in targets:
                n = -n  # report normals pointing from the other body into the free piece
            other = b2 if b1 in targets else b1
            fn = float(f6[0])
            F += fn * n
            T += np.cross(np.asarray(c.pos) - tpos, fn * n)
            out.append({"pos": _r(c.pos, 5), "normal_into_free_piece": _r(n, 4), "distance_m": round(float(c.dist), 6),
                        "normal_force": round(fn, 5), "with": m.body(other).name})
        out.sort(key=lambda e: (e["distance_m"], -e["normal_force"]))
        lags = []
        for h in self.rig.handles:
            if h.body is not None:
                lags.append({"handle": h.idx, "lag_m": round(float(np.linalg.norm(d.mocap_pos[h.mocap_id] - self.rig.grasp_point_world(h))), 6)})
        return {
            "n_contacts": len(out),
            "in_contact": any(e["distance_m"] <= 0 for e in out),
            "contacts": out[:max_contacts],
            "note": "distance_m < 0 = touching (penetration depth); > 0 = within the contact margin, not touching yet",
            "total_contact_force": _r(F, 5),
            "total_contact_torque_about_origin": _r(T, 5),
            "grip_lag": lags,
            "last_action": self._last_action,
        }

    # ------------------------------------------------------------ planner queries (`planner` profile)
    def check_poses(self, poses: list) -> dict:
        """Collision-free test of up to 256 absolute poses [x y z qw qx qy qz] of the free piece."""
        if not isinstance(poses, (list, tuple)) or not 1 <= len(poses) <= 256:
            raise ValueError("poses: a list of 1..256 poses [x, y, z, qw, qx, qy, qz]")
        P = []
        for q in poses:
            if not isinstance(q, (list, tuple)) or len(q) != 7 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in q):
                raise ValueError("each pose must be 7 finite numbers [x, y, z, qw, qx, qy, qz]")
            if np.linalg.norm(q[3:]) < 1e-9:
                raise ValueError("quaternion must be non-zero")
            P.append(np.asarray(q, float))
        return {"valid": [self.checker.is_valid(q) for q in P]}

    def check_motion(self, handle: int = 0, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, frame: str = "world") -> dict:
        """Dry run of `move` with the same clamping and kinematics: is the straight-line motion of
        the held piece collision-free? Does not change the simulation or cost an action."""
        from .geometry import quat_conj, quat_mul, quat_slerp, quat_to_mat

        args = self._normalise("move", {"handle": handle, "dx": dx, "dy": dy, "dz": dz, "droll": droll, "dpitch": dpitch, "dyaw": dyaw, "frame": frame})
        h = self.rig._handle(args["handle"])
        if h.body is None:
            raise RigError(f"handle {h.idx} is not holding anything")
        req = self.rig.clamp_request(h.idx, [args["dx"], args["dy"], args["dz"]], [args["droll"], args["dpitch"], args["dyaw"]], args["frame"])
        bid = self.model.body(h.body).id
        ph, qh = req["p0"], req["q0"]
        Rh = quat_to_mat(qh)
        t_rel = Rh.T @ (self.data.xpos[bid] - ph)
        q_rel = quat_mul(quat_conj(qh), self.data.xquat[bid])

        def body_pose(f):
            p = req["p0"] + f * (req["p1"] - req["p0"])
            q = quat_slerp(req["q0"], req["q1"], f)
            return np.concatenate([p + quat_to_mat(q) @ t_rel, quat_mul(q, q_rel)])

        start, end = body_pose(0.0), body_pose(1.0)
        w = self.target_radius
        from .collision import pose_metric

        n = max(1, int(np.ceil(float(pose_metric(start, end, w)) / 0.002)))
        free_frac = 1.0
        for i in range(1, n + 1):
            if not self.checker.is_valid(body_pose(i / n)):
                free_frac = (i - 1) / n
                break
        return {"free": free_frac == 1.0, "free_fraction": round(free_frac, 4), "clamped": req["clamped"],
                "end_pose": _r(end, 6), "resolution_m": 0.002,
                "note": "exact collision test of the held piece against the fixed one along the move's path"}

    def get_geometry(self) -> dict:
        """Triangle meshes of every piece in its own body frame (simulator units), plus current poses."""
        import trimesh

        if getattr(self, "_geometry", None) is None:
            from .geometry import quat_from_euler_xyz  # noqa: F401
            out = {}
            for b in self.puzzle.frame + self.puzzle.objects:
                verts, faces = [], []
                for g in b.geoms:
                    if g.type != "mesh":
                        continue
                    if g.path.startswith("paper:"):
                        from ..puzzles.paper_alpha import resolve_mesh_path as res
                    elif g.path.startswith("ompl:"):
                        from ..puzzles.ompl_alpha import resolve_mesh_path as res
                    else:
                        def res(r):
                            return Path(self.puzzle.source_path).parent / r
                    mesh = trimesh.load(res(g.path), force="mesh")
                    mesh.apply_scale(np.asarray(g.scale, float))
                    if any(g.euler):
                        mesh.apply_transform(trimesh.transformations.euler_matrix(*np.radians(g.euler), axes="rxyz"))  # MuJoCo eulerseq "xyz" = intrinsic
                    mesh.apply_translation(np.asarray(g.pos, float))
                    faces.append(np.asarray(mesh.faces) + sum(len(v) for v in verts))
                    verts.append(np.asarray(mesh.vertices))
                if verts:
                    out[b.name] = {"vertices": _r(np.vstack(verts), 6), "faces": np.vstack(faces).astype(int).tolist()}
            self._geometry = out
        poses = {name: _r(np.concatenate([self.data.xpos[self.model.body(name).id], self.data.xquat[self.model.body(name).id]]), 6)
                 for name in self._geometry}
        return {"frame": "each mesh in its body frame; world point = pos + R(quat) @ vertex", "meshes": self._geometry, "poses": poses}

    def _estimate_tube_radius(self) -> float | None:
        """Tube radius from mesh volume and centre-line length (V = pi r^2 L), simulator units."""
        for b in self.puzzle.objects:
            if b.centerline and any(g.type == "mesh" for g in b.geoms):
                try:
                    import trimesh

                    from .builder import MJCFBuilder  # noqa: F401  (same mesh resolution as the builder)
                    g = next(g for g in b.geoms if g.type == "mesh")
                    if g.path.startswith("paper:"):
                        from ..puzzles.paper_alpha import resolve_mesh_path as res
                    else:
                        from ..puzzles.ompl_alpha import resolve_mesh_path as res
                    m = trimesh.load(res(g.path), force="mesh")
                    m.merge_vertices()
                    L = float(np.sum(np.linalg.norm(np.diff(self.curves[b.name], axis=0), axis=1)))
                    s = self.puzzle.unit_scale
                    return float(np.sqrt(abs(m.volume) * s**3 / (np.pi * L))) if L > 0 else None
                except Exception:
                    return None
        return None

    def geometry_description(self) -> dict | None:
        """Text description of the tubes for agents (no image needed): centre-lines (world
        frame, decimated), each alpha's gap endpoints, tube radius, and B's pose relative to A."""
        if not self.curves or self.puzzle.goal.type != "separation":
            return None
        out = {"tube_radius_m": None if self._tube_radius is None else round(self._tube_radius, 4)}
        for name, c in self.curves.items():
            world = self.curve_points(name)
            idx = np.linspace(0, len(world) - 1, 24).round().astype(int)
            entry = {"centerline_world_m": _r(world[idx], 3),
                     "note": "polyline along the tube's axis, end to end"}
            gap = _gap_indices(c)
            if gap is not None:
                entry["gap_endpoints_world_m"] = [_r(world[gap[0]], 3), _r(world[gap[1]], 3)]
                entry["gap_note"] = "the two axis points where this alpha's loop closes on itself (its opening)"
            out[name] = entry
        out["relative_pose_target_in_frame"] = _r(self.target_pose(), 5)
        out["relative_pose_note"] = "x y z qw qx qy qz of the target in the fixed tube's frame (the world frame)"
        return out

    def _executed_motion(self) -> dict:
        """Largest handle translation (m) and rotation (rad) actually executed by the last action."""
        from .geometry import quat_conj, quat_mul, quat_to_axis_angle

        dp, dr = 0.0, 0.0
        for h, (p0, q0) in zip(self.rig.handles, getattr(self, "_handles_before", [])):
            p1, q1 = self.rig.handle_pose(h)
            if h.body is None:
                continue
            dp = max(dp, float(np.linalg.norm(p1 - p0)))
            dr = max(dr, abs(float(quat_to_axis_angle(quat_mul(quat_conj(q0), q1))[1])))
        return {"translation": dp, "rotation": dr}

    def _colour_legend(self) -> str:
        parts = [f"Colours: {', '.join(self.built.frame_bodies)} (fixed) grey"]
        if self.built.rope is not None:
            parts.append("rope orange")
        parts.append(f"{', '.join(self.built.object_bodies)} (free) blue")
        parts += [f"handle{h} {c}" for h, c in zip(range(len(self.rig.handles)), ("red", "green"))]
        return ", ".join(parts) + "."

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self.log.close()
