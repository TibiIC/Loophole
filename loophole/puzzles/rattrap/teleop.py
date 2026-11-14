"""Rat Trap teleoperation core (no UI): drive the rig continuously, record, convert to discrete commands.

Every motion goes through the agent's rig (loophole.sim.rig.Rig, unchanged):
* the active handle's mocap advances toward a continuous target at most `rig_linear_speed * dt` per physics
  step (the rig's velocity cap);
* before every advance, the rig's own blocked tests run (`Rig._lagging`: the held node trails its handle by more
  than `rig_lag_limit`; `Rig._strained`: any contact pressed deeper than `rig_stall_penetration`); when blocked
  the handle snaps back onto the node (`Rig._snap_to_body`), exactly as `Rig.move` does, and the target is
  dropped so it stops pushing;
* every substep is the simulator's own (penetration audit included);
* grabs and releases are `Rig.grab` / `Rig.release`.
Only the active handle moves; the other holds still, so the recording converts exactly into sequential moves.

Recording: one record per physics step in which the active handle moved, plus grab / release events.
`to_commands` turns it into discrete capped commands: each continuous stretch of one handle's motion is
simplified (Ramer-Douglas-Peucker, `tolerance`) and split into straight moves no longer than
`rig_max_linear_step`, the same splitting the Alpha oracle uses (loophole.planning.commands.path_to_commands).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .sim import RatTrapSim
from .topology import TopologyTracker, build_discs


@dataclass
class Snapshot:
    state: np.ndarray
    eq_data: np.ndarray
    handles: list
    targets: list
    active: int
    selected: int
    n_records: int
    steps: int


@dataclass
class Teleop:
    sim: RatTrapSim
    records: list = field(default_factory=list)
    active: int = 0
    selected: int = 0
    steps: int = 0

    def __post_init__(self):
        self.rig = self.sim.rig
        self.dt = float(self.sim.model.opt.timestep)
        self.vmax = float(self.sim.cfg.linear_speed)
        self.targets: list[np.ndarray | None] = [None] * len(self.rig.handles)
        self.last_blocked = [False] * len(self.rig.handles)
        self.held_node: list[int | None] = [None] * len(self.rig.handles)
        self.tracker = TopologyTracker(build_discs(self.sim.spec), closed=bool(self.sim.spec["cord_closed"]))
        self.tracker.update(self.sim.node_positions())
        self.snapshot: Snapshot | None = None
        self._sig = mujoco.mjtState.mjSTATE_INTEGRATION
        self.records.append({"type": "header", "yaml": self.sim.spec.raw, "time": time.time(), "nodes": self.sim.n_nodes})

    # ------------------------------------------------------------------ actions
    def grab(self, node: int, handle: int | None = None) -> dict:
        h = self.active if handle is None else handle
        if self.rig.handles[h].body is not None:
            self.release(h)
        out = self.sim.grab_node(h, node)
        self.held_node[h] = out["node"]
        self.selected = out["node"]
        self.targets[h] = None
        self.records.append({"type": "grab", "s": self.steps, "handle": h, "node": out["node"],
                             "pos": [round(float(x), 7) for x in self.handle_pos(h)]})
        return out

    def release(self, handle: int | None = None) -> None:
        h = self.active if handle is None else handle
        if self.rig.handles[h].body is None:
            return
        self.sim.release(h)
        self.held_node[h] = None
        self.targets[h] = None
        self.records.append({"type": "release", "s": self.steps, "handle": h})

    def set_active(self, h: int) -> None:
        if h != self.active:
            self.targets[self.active] = None  # the other handle holds still
            self.active = h

    def handle_pos(self, h: int | None = None) -> np.ndarray:
        h = self.active if h is None else h
        return self.rig.handle_pose(self.rig.handles[h])[0]

    def set_target(self, p) -> None:
        if self.rig.handles[self.active].body is not None:
            self.targets[self.active] = np.asarray(p, float)

    def nudge(self, d) -> None:
        if self.rig.handles[self.active].body is None:
            return
        base = self.targets[self.active] if self.targets[self.active] is not None else self.handle_pos()
        self.targets[self.active] = base + np.asarray(d, float)

    # ------------------------------------------------------------------ physics
    def step(self, n: int = 1) -> None:
        d = self.sim.data
        h = self.rig.handles[self.active]
        for _ in range(n):
            d.xfrc_applied[:] = 0.0  # nothing but the rig may move the cord
            t = self.targets[self.active]
            moved = False
            if h.body is not None and t is not None:
                blocked = self.rig._lagging(h) or self.rig._strained()
                if blocked:
                    self.rig._snap_to_body(h)
                    self.targets[self.active] = None
                    self.last_blocked[self.active] = True
                    moved = True  # the snap moved the handle: record it
                else:
                    p = d.mocap_pos[h.mocap_id]
                    delta = t - p
                    dist = float(np.linalg.norm(delta))
                    if dist > 1e-9:
                        p += delta * min(1.0, self.vmax * self.dt / dist)
                        self.last_blocked[self.active] = False
                        moved = True
                    else:
                        self.targets[self.active] = None
            self.sim._substep()
            self.steps += 1
            if moved:
                self.records.append({"s": self.steps, "h": self.active, "p": [round(float(x), 7) for x in d.mocap_pos[h.mocap_id]],
                                     "q": [round(float(x), 7) for x in d.mocap_quat[h.mocap_id]]})
            if self.steps % 500 == 0:  # cord state snapshot (analysis / restarting a replay mid-way)
                self.records.append({"type": "state", "s": self.steps, "nodes": np.round(self.sim.node_positions(), 6).tolist()})

    def update_topology(self) -> dict:
        return self.tracker.update(self.sim.node_positions())

    # ------------------------------------------------------------------ snapshot / undo
    def save(self) -> None:
        m, d = self.sim.model, self.sim.data
        buf = np.empty(mujoco.mj_stateSize(m, self._sig))
        mujoco.mj_getState(m, d, buf, self._sig)
        hs = [(h.body, h.eq_id, None if h.anchor_local is None else h.anchor_local.copy()) for h in self.rig.handles]
        self.snapshot = Snapshot(buf, m.eq_data.copy(), hs, [None if t is None else t.copy() for t in self.targets], self.active,
                                 self.selected, len(self.records), self.steps)

    def undo(self) -> bool:
        """Restore the last snapshot; the recording is truncated, so undone motion never reaches the commands."""
        s = self.snapshot
        if s is None:
            return False
        m, d = self.sim.model, self.sim.data
        mujoco.mj_setState(m, d, s.state, self._sig)
        m.eq_data[:] = s.eq_data
        for h, (body, eq_id, anchor) in zip(self.rig.handles, s.handles):
            h.body, h.eq_id, h.anchor_local = body, eq_id, anchor
        mujoco.mj_forward(m, d)
        self.targets = [None if t is None else t.copy() for t in s.targets]
        self.active, self.selected, self.steps = s.active, s.selected, s.steps
        self.held_node = [None if h.body is None else self._node_of(h) for h in self.rig.handles]
        del self.records[s.n_records:]
        self.records.append({"type": "undo_marker", "s": self.steps})  # kept for the audit trail; ignored by to_commands
        self.tracker.update(self.sim.node_positions())
        return True

    def _node_of(self, h) -> int:
        P = self.sim.node_positions()
        return int(np.argmin(np.linalg.norm(P - self.rig.grasp_point_world(h), axis=1)))

    # ------------------------------------------------------------------ output
    def estimated_actions(self) -> int:
        return len(to_commands(self.records, self.sim.spec))

    def save_recording(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in self.records:
                f.write(json.dumps(r, separators=(",", ":"), default=float) + "\n")


def _rdp(P: np.ndarray, tol: float) -> list[int]:
    """Ramer-Douglas-Peucker: indices of the points kept (always the first and last)."""
    if len(P) < 3:
        return list(range(len(P)))
    a, b = P[0], P[-1]
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-12:
        d = np.linalg.norm(P - a, axis=1)
    else:
        d = np.linalg.norm(np.cross(P - a, ab / n), axis=1)
    k = int(np.argmax(d))
    if d[k] <= tol:
        return [0, len(P) - 1]
    left = _rdp(P[: k + 1], tol)
    right = _rdp(P[k:], tol)
    return left[:-1] + [i + k for i in right]


def to_commands(records: list, spec, tolerance: float = 0.0005) -> list[tuple[str, dict]]:
    """Recording -> discrete commands: ("grab_node", {handle, node}), ("move", {handle, dx, dy, dz}), ("release", {handle})."""
    cap = float(spec["rig_max_linear_step"])
    cmds: list[tuple[str, dict]] = []
    pos: dict[int, np.ndarray] = {}
    stretch: list[np.ndarray] = []
    stretch_h: int | None = None

    def flush():
        nonlocal stretch, stretch_h
        if stretch_h is not None and len(stretch) >= 2:
            P = np.array(stretch)
            keep = _rdp(P, tolerance)
            for a, b in zip(keep[:-1], keep[1:]):
                d = P[b] - P[a]
                L = float(np.linalg.norm(d))
                if L < 1e-7:
                    continue
                k = max(1, math.ceil(L / cap - 1e-9))
                for _ in range(k):
                    s = d / k
                    cmds.append(("move", {"handle": stretch_h, "dx": float(s[0]), "dy": float(s[1]), "dz": float(s[2])}))
        stretch, stretch_h = [], None

    for r in records:
        t = r.get("type")
        if t in ("header", "undo_marker", "state"):
            continue
        if t == "grab":
            flush()
            cmds.append(("grab_node", {"handle": r["handle"], "node": r["node"]}))
            pos[r["handle"]] = np.asarray(r["pos"], float)  # the handle sits on the node after a grab
            continue
        if t == "release":
            flush()
            cmds.append(("release", {"handle": r["handle"]}))
            continue
        h, p = r["h"], np.asarray(r["p"], float)
        if stretch_h != h:
            flush()
            stretch_h = h
            stretch = [pos[h]] if h in pos else []
        stretch.append(p)
        pos[h] = p
    flush()
    return cmds


def replay(commands: list, spec, sim: RatTrapSim | None = None) -> RatTrapSim:
    """Execute discrete commands headlessly through the rig (Rig.move: straight capped moves at the rig's speed,
    halting when blocked)."""
    sim = sim or RatTrapSim(spec)
    for tool, a in commands:
        if tool == "grab_node":
            sim.grab_node(a["handle"], a["node"])
        elif tool == "move":
            sim.move(a["handle"], [a["dx"], a["dy"], a["dz"]])
        elif tool == "release":
            sim.release(a["handle"])
        else:
            raise ValueError(f"unknown command {tool!r}")
    return sim


def load_records(path: Path) -> list:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def save_commands(commands: list, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"type": "trajectory", "puzzle": "rattrap", **(meta or {})}) + "\n")
        for tool, a in commands:
            f.write(json.dumps({"tool": tool, "args": a}) + "\n")


def load_commands(path: Path) -> list:
    out = []
    for line in Path(path).open():
        r = json.loads(line)
        if r.get("type") == "trajectory":
            continue
        out.append((r["tool"], r["args"]))
    return out


def convert_closed_loop(records: list, spec, tolerance: float = 0.0005, sim: RatTrapSim | None = None,
                        on_waypoint=None) -> tuple[list, RatTrapSim]:
    """Recording -> discrete commands, generated by executing them: grabs / releases as recorded; each continuous stretch
    of one handle's recorded path is simplified (RDP, `tolerance`) and every waypoint is approached from where the
    handle actually is (moves capped at rig_max_linear_step, through Rig.move), so a move that blocks a little earlier or
    later than in the live session does not offset everything after it. Returns (executed commands, final sim).
    on_waypoint(step, handle, n_moves, blocked): called after each waypoint, with the recorded physics step it came from
    (diagnostics: aligning the converted run with the recording)."""
    sim = sim or RatTrapSim(spec)
    cap = float(spec["rig_max_linear_step"])
    cmds: list[tuple[str, dict]] = []
    stretch: list[np.ndarray] = []
    stretch_h: int | None = None

    def go(h: int, wp: np.ndarray) -> tuple[int, bool]:
        n = 0
        for _ in range(64):  # approach the waypoint; stop on a block (the next waypoint is aimed from wherever we are)
            cur = sim.rig.handle_pose(sim.rig.handles[h])[0]
            d = wp - cur
            L = float(np.linalg.norm(d))
            if L < 1e-5:
                return n, False
            step = d if L <= cap else d * (cap / L)
            a = {"handle": h, "dx": float(step[0]), "dy": float(step[1]), "dz": float(step[2])}
            r = sim.move(h, [a["dx"], a["dy"], a["dz"]])
            cmds.append(("move", a))
            n += 1
            if r["blocked"]:
                return n, True
        return n, False

    steps: list[int] = []

    def flush():
        nonlocal stretch, stretch_h, steps
        if stretch_h is not None and len(stretch) >= 2:
            P = np.array(stretch)
            for idx in _rdp(P, tolerance)[1:]:
                n, blocked = go(stretch_h, P[idx])
                if on_waypoint is not None:
                    on_waypoint(steps[idx], stretch_h, n, blocked)
        stretch, stretch_h, steps = [], None, []

    pos: dict[int, np.ndarray] = {}
    grab_step: dict[int, int] = {}
    for r in records:
        t = r.get("type")
        if t in ("header", "undo_marker", "state"):
            continue
        if t in ("grab", "release"):
            flush()
            if t == "grab":
                sim.grab_node(r["handle"], r["node"])
                cmds.append(("grab_node", {"handle": r["handle"], "node": r["node"]}))
                pos[r["handle"]] = np.asarray(r["pos"], float)
                grab_step[r["handle"]] = int(r.get("s", 0))
            else:
                sim.release(r["handle"])
                cmds.append(("release", {"handle": r["handle"]}))
            continue
        h, p = r["h"], np.asarray(r["p"], float)
        if stretch_h != h:
            flush()
            stretch_h = h
            stretch = [pos[h]] if h in pos else []
            steps = [grab_step.get(h, int(r["s"]) - 1)] if h in pos else []
        stretch.append(p)
        steps.append(int(r["s"]))
        pos[h] = p
    flush()
    return cmds, sim


def replay_exact(records: list, spec, sim: RatTrapSim | None = None) -> RatTrapSim:
    """Re-execute a teleop recording step for step: the recorded handle pose before every physics step, grabs / releases
    at their recorded step. Reproduces the live session (determinism check; a mouse or other change to the model can
    be tested against the same hand-made trajectory)."""
    sim = sim or RatTrapSim(spec)
    d = sim.data
    events = [r for r in records if r.get("type") not in ("header", "undo_marker", "state")]
    last = max((r["s"] for r in events), default=0)
    step = k = 0
    while step < last:
        while k < len(events) and events[k].get("type") in ("grab", "release") and events[k]["s"] == step:
            e = events[k]
            sim.grab_node(e["handle"], e["node"]) if e["type"] == "grab" else sim.release(e["handle"])
            k += 1
        if k < len(events) and "p" in events[k] and events[k]["s"] == step + 1:
            e = events[k]
            mid = sim.rig.handles[e["h"]].mocap_id
            d.mocap_pos[mid] = e["p"]
            if "q" in e:
                d.mocap_quat[mid] = e["q"]
            k += 1
        d.xfrc_applied[:] = 0.0
        sim._substep()
        step += 1
    return sim


def extract_waypoints(records: list, tolerance: float = 0.0001) -> list[dict]:
    """The fixed waypoints of a recording, for the closed-loop reference replay (loophole/reference/follow_waypoints.py):
    grabs and releases in order, and for each stretch of one handle's motion the recorded hand positions kept by
    Ramer-Douglas-Peucker at `tolerance` (the points convert_closed_loop approaches). Positions in millimetres."""
    out: list[dict] = []
    pos: dict[int, np.ndarray] = {}
    stretch: list[np.ndarray] = []
    stretch_h: int | None = None

    def flush():
        nonlocal stretch, stretch_h
        if stretch_h is not None and len(stretch) >= 2:
            P = np.array(stretch)
            for idx in _rdp(P, tolerance)[1:]:
                out.append({"kind": "waypoint", "handle": stretch_h, "mm": [round(float(x) * 1000.0, 6) for x in P[idx]]})
        stretch, stretch_h = [], None

    for r in records:
        t = r.get("type")
        if t in ("header", "undo_marker", "state"):
            continue
        if t in ("grab", "release"):
            flush()
            if t == "grab":
                out.append({"kind": "grab", "handle": r["handle"], "node": r["node"]})
                pos[r["handle"]] = np.asarray(r["pos"], float)
            else:
                out.append({"kind": "release", "handle": r["handle"]})
            continue
        h, p = r["h"], np.asarray(r["p"], float)
        if stretch_h != h:
            flush()
            stretch_h = h
            stretch = [pos[h]] if h in pos else []
        stretch.append(p)
        pos[h] = p
    flush()
    return out
