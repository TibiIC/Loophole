"""Turn an SE(3) path of the target into rig commands (grab / move ... ).

The handle grasps the target at its body origin with the body's orientation,
so the handle pose *is* the target pose. Each path segment is split into moves
whose translation and rotation stay below the rig caps; each move is a world-
frame increment, which the rig executes as lerp(translation) + slerp(rotation)
about the grasp point - exactly the planner's straight-line interpolation.
"""

from __future__ import annotations

import math

import numpy as np

from ..reference.pose_stream import euler_xyz_from_quat
from ..sim.geometry import quat_from_rotvec, quat_to_rotvec
from .se3 import rel_rotation


def path_to_commands(path: list[np.ndarray], target: str, max_lin: float, max_ang: float, safety: float = 0.95, handle: int = 0) -> list[tuple[str, dict]]:
    cmds: list[tuple[str, dict]] = [("grab", {"handle": handle, "target": target})]
    for a, b in zip(path[:-1], path[1:]):
        dt = b[:3] - a[:3]
        rv = quat_to_rotvec(rel_rotation(a[3:], b[3:]))
        n = max(1, math.ceil(max(np.linalg.norm(dt) / (safety * max_lin), np.linalg.norm(rv) / (safety * max_ang))))
        step_t = dt / n
        rpy = euler_xyz_from_quat(quat_from_rotvec(rv / n))
        for _ in range(n):
            cmds.append(
                ("move", {"handle": handle, "dx": float(step_t[0]), "dy": float(step_t[1]), "dz": float(step_t[2]),
                          "droll": float(rpy[0]), "dpitch": float(rpy[1]), "dyaw": float(rpy[2]), "frame": "world"})
            )
    return cmds
