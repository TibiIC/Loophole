"""Milestone 6: recorder / replay / external pose stream."""

import json
import socket
import threading
import time

import numpy as np
import pytest

from loophole.reference.pose_stream import PoseMsg, PoseStreamDriver, euler_xyz_from_quat, file_source, udp_source
from loophole.reference.recorder import Recorder, load_trajectory, write_trajectory
from loophole.reference.replay import replay
from loophole.reference.scripted import SCRIPTS
from loophole.sim.env import LoopholeEnv
from loophole.sim.geometry import quat_from_euler_xyz


def test_recorder_roundtrip_is_deterministic(tmp_path):
    p = tmp_path / "traj.jsonl"
    env = LoopholeEnv("easy")
    with Recorder(env, p, source="test") as rec:
        for tool, args in SCRIPTS["easy"]:
            rec.act(tool, dict(args))
    h = env.state_hash()
    env2 = LoopholeEnv("easy")
    replay(env2, load_trajectory(p))
    assert env2.state_hash() == h and env2.success


def test_write_and_load_trajectory(tmp_path):
    p = tmp_path / "t.jsonl"
    write_trajectory(p, "easy", SCRIPTS["easy"])
    assert load_trajectory(p) == [(t, a) for t, a in SCRIPTS["easy"]]
    assert json.loads(p.read_text().splitlines()[0])["type"] == "trajectory"


@pytest.mark.parametrize("rpy", [(0.1, -0.2, 0.3), (0.0, 0.0, 0.0), (-0.5, 0.4, 2.0)])
def test_euler_roundtrip(rpy):
    assert np.allclose(euler_xyz_from_quat(quat_from_euler_xyz(rpy)), rpy)


def human_demo(env) -> list[dict]:
    """A synthetic 'human' pose stream solving the easy puzzle (pinch tip, lift, carry ring)."""
    tip = (env.rope_nodes()[20] + env.rope_nodes()[21]) / 2
    ring = env.data.xpos[env.model.body("ring").id].copy()
    msgs, t = [], 0.0
    for z in np.linspace(0, 0.4, 41):  # 1 cm per frame, like a 30 Hz tracker
        msgs.append({"t": t, "handle": 0, "pos": (tip + [0, 0, z]).tolist(), "grip": True})
        t += 0.033
    for x in np.linspace(0, 0.92, 93):
        msgs.append({"t": t, "handle": 1, "pos": (ring + [x, 0, 0]).tolist(), "grip": True, "target": "ring"})
        t += 0.033
    return msgs


def test_pose_stream_file_solves_and_records(tmp_path):
    env = LoopholeEnv("easy")
    poses = tmp_path / "poses.jsonl"
    poses.write_text("\n".join(json.dumps(m) for m in human_demo(env)) + "\n")
    traj = tmp_path / "demo.jsonl"
    rec = Recorder(env, traj, source="pose_stream")
    PoseStreamDriver(env, rec, min_step=0.05).run(file_source(poses))
    rec.close()
    assert env.success and env.audit.count == 0
    assert env.actions_used <= env.puzzle.max_actions
    # the recorded human demo is an ordinary trajectory: it replays to the same state
    env2 = LoopholeEnv("easy")
    replay(env2, load_trajectory(traj))
    assert env2.state_hash() == env.state_hash()


def test_pose_stream_udp():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    env = LoopholeEnv("easy")
    msgs = human_demo(env)[:20]

    def send():
        time.sleep(0.3)
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for m in msgs:
            out.sendto(json.dumps(m).encode(), ("127.0.0.1", port))
        out.close()

    th = threading.Thread(target=send)
    th.start()
    received = list(udp_source("127.0.0.1", port, timeout=1.5))
    th.join()
    assert len(received) == len(msgs) and isinstance(received[0], PoseMsg)
    d = PoseStreamDriver(env, min_step=0.05)
    d.run(received)
    assert env.rig.handles[0].body == "rope_20"  # nearest-body grasp picked the tip segment
    assert env.actions_used >= 2
