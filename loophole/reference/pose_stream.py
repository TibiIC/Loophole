"""External pose input: drive the handles from a stream of absolute hand poses.

Intended for a digital twin of a human solving the physical puzzle (e.g. ArUco
markers tracked by a webcam). The human path goes through *exactly* the same
handle mechanism as the agent: absolute poses are turned into grab / clamped
move / release calls on the env (or on a Recorder, which also writes a
trajectory that the oracle can replay later).

Message format (one JSON object per UDP datagram or per line of a file)::

    {"t": 1.25, "handle": 0, "pos": [x, y, z], "quat": [w, x, y, z],
     "grip": true, "target": "ring" | "rope" | null, "segment": 12 | null}

* ``grip`` rising edge  -> grab at ``pos`` (target optional: nearest graspable body)
* ``grip`` held         -> move the handle towards (pos, quat) in clamped steps,
                           issued when the pose moved more than ``min_step``
* ``grip`` falling edge -> release

    python -m loophole.reference.pose_stream --puzzle easy --udp 0.0.0.0:9870 --record demo.jsonl
    python -m loophole.reference.pose_stream --puzzle easy --file poses.jsonl --record demo.jsonl
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from ..sim.env import EpisodeOver, LoopholeEnv
from ..sim.geometry import quat_conj, quat_mul, quat_normalize, quat_to_mat
from ..sim.rig import RigError


@dataclass
class PoseMsg:
    t: float
    handle: int
    pos: np.ndarray
    quat: np.ndarray | None
    grip: bool
    target: str | None = None
    segment: int | None = None

    @classmethod
    def from_json(cls, d: dict) -> "PoseMsg":
        return cls(
            t=float(d.get("t", time.monotonic())),
            handle=int(d.get("handle", 0)),
            pos=np.asarray(d["pos"], dtype=float),
            quat=None if d.get("quat") is None else quat_normalize(d["quat"]),
            grip=bool(d.get("grip", False)),
            target=d.get("target"),
            segment=d.get("segment"),
        )


class Actor(Protocol):  # LoopholeEnv or Recorder
    def act(self, tool: str, args: dict) -> dict: ...


def euler_xyz_from_quat(q) -> tuple[float, float, float]:
    """Inverse of geometry.quat_from_euler_xyz (extrinsic X, Y, Z)."""
    R = quat_to_mat(q)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


class PoseStreamDriver:
    def __init__(self, env: LoopholeEnv, actor: Actor | None = None, min_step: float = 0.03, min_angle: float = 0.1, max_moves_per_msg: int = 5):
        self.env = env
        self.actor = actor or env
        self.min_step = min_step
        self.min_angle = min_angle
        self.max_moves_per_msg = max_moves_per_msg
        self.gripping: dict[int, bool] = {}
        self.events: list[dict] = []

    def _act(self, tool: str, args: dict) -> dict | None:
        try:
            r = self.actor.act(tool, args)
        except (RigError, ValueError) as e:
            self.events.append({"tool": tool, "args": args, "error": str(e)})
            return None
        self.events.append({"tool": tool, "args": args, "action": r.get("action")})
        return r

    def _nearest_target(self, pos: np.ndarray) -> tuple[str, dict]:
        rig = self.env.rig
        best, best_d = None, np.inf
        for b in self.env.built.graspable:
            d = rig._distance_to_body(self.env.model.body(b).id, pos)
            if d < best_d:
                best, best_d = b, d
        if best in rig.rope_bodies:
            return self.env.built.rope.name, {"segment": rig.rope_bodies.index(best)}
        return best, {}

    def handle(self, m: PoseMsg) -> None:
        if self.env.done:
            raise EpisodeOver(self.env.done_reason)
        was = self.gripping.get(m.handle, False)
        if m.grip and not was:
            target, extra = (m.target, {}) if m.target else self._nearest_target(m.pos)
            args = {"handle": m.handle, "target": target, **extra}
            if m.segment is not None:
                args["segment"] = int(m.segment)
            elif "segment" not in args:
                args["point"] = m.pos.round(5).tolist()
            if self._act("grab", args) is not None:
                self.gripping[m.handle] = True
            return
        if not m.grip and was:
            self._act("release", {"handle": m.handle})
            self.gripping[m.handle] = False
            return
        if m.grip:
            self._track(m)

    def _track(self, m: PoseMsg) -> None:
        rig = self.env.rig
        for _ in range(self.max_moves_per_msg):
            p, q = rig.handle_pose(rig.handles[m.handle])
            dp = m.pos - p
            rpy = (0.0, 0.0, 0.0)
            if m.quat is not None:
                rpy = euler_xyz_from_quat(quat_mul(m.quat, quat_conj(q)))  # world-frame delta
            if np.linalg.norm(dp) < self.min_step and max(abs(x) for x in rpy) < self.min_angle:
                return
            r = self._act("move", {"handle": m.handle, "dx": float(dp[0]), "dy": float(dp[1]), "dz": float(dp[2]),
                                   "droll": rpy[0], "dpitch": rpy[1], "dyaw": rpy[2], "frame": "world"})
            if r is None or r.get("blocked") or r.get("done"):
                return

    def run(self, msgs: Iterable[PoseMsg]) -> None:
        for m in msgs:
            try:
                self.handle(m)
            except EpisodeOver:
                return


def file_source(path: str | Path, realtime: bool = False) -> Iterator[PoseMsg]:
    """Replay a recorded pose stream (JSONL), optionally at recorded speed."""
    t_prev = None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        m = PoseMsg.from_json(json.loads(line))
        if realtime and t_prev is not None:
            time.sleep(max(0.0, m.t - t_prev))
        t_prev = m.t
        yield m


def udp_source(host: str = "0.0.0.0", port: int = 9870, timeout: float | None = None) -> Iterator[PoseMsg]:
    """Receive pose messages as JSON datagrams. Stops after `timeout` s of silence."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(timeout)
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                return
            try:
                yield PoseMsg.from_json(json.loads(data))
            except (ValueError, KeyError) as e:
                print(f"bad pose message: {e}", file=sys.stderr)
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    from .recorder import Recorder

    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default="easy")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--udp", help="host:port to listen on")
    src.add_argument("--file", help="JSONL pose file to replay")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--record", help="write the resulting handle commands as a trajectory")
    ap.add_argument("--log", help="episode log path")
    ap.add_argument("--min-step", type=float, default=0.03, help="m of hand motion before a move is issued (each move costs an action)")
    a = ap.parse_args(argv)
    env = LoopholeEnv(a.puzzle, log_path=a.log)
    actor = Recorder(env, a.record, source="pose_stream") if a.record else env
    if a.udp:
        host, port = a.udp.rsplit(":", 1)
        msgs = udp_source(host, int(port), timeout=30.0)
    else:
        msgs = file_source(a.file, a.realtime)
    PoseStreamDriver(env, actor, min_step=a.min_step).run(msgs)
    if a.record:
        actor.close()
    final = env.finalize()
    print(json.dumps({k: final[k] for k in ("score", "success", "actions_used", "done_reason")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
