#!/usr/bin/env python3
"""Diagnose how a run achieved its score: search signals, exploration vs commitment, path length.

    .venv/bin/python scripts/diagnose_run.py <trial dir> [<trial dir> ...] [--json out.json]

A trial dir is a Harbor trial (verifier/episode.jsonl; agent/opencode.txt if it was an agent).
Per trial it reports:
  * probe-then-reverse patterns: a move, then >= 1 observation, then a move that approximately
    undoes it (|t1 + t2| <= 25 % of |t1| and the composed rotation <= 25 % of the first's angle);
    both moves of such a pair count as exploratory, every other executed move as committed;
  * blocked moves, observation calls by tool, whether actions ran in a single scripted burst;
  * which state fields the agent's code and reasoning reference (get_state / get_contacts /
    render caption keys), and whether any progress / cost-to-go value was ever available to it;
  * actions a, a/T, and the SE(3) path length of tube B relative to tube A
    (sum over consecutive logged poses of ||dt|| + w * angle, w = the roadmap's characteristic length).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loophole.sim.geometry import quat_conj, quat_from_euler_xyz, quat_mul, quat_to_axis_angle  # noqa: E402

W = 1.5408646  # roadmap characteristic length of Alpha 1.0 (rotation weight in the SE(3) metric)
Q0 = np.array([1.2, 0.36, 0.28, 1.0, 0.0, 0.0, 0.0])  # canonical start of tube B (alpha-1.0-paper)

STATE_FIELDS = {
    "get_state": ["objects", "pos", "quat", "graspable", "handles", "holding", "last_action", "blocked", "clamped",
                  "executed_translation_m", "executed_rotation_rad", "actions_used", "max_actions", "actions_remaining",
                  "done", "success", "violations", "limits", "max_linear_step_m", "max_angular_step_rad",
                  "workspace_min", "workspace_max", "grasp_tolerance_m"],
    "get_contacts": ["contacts", "n_contacts", "in_contact", "normal_into_free_piece", "distance_m", "normal_force",
                     "total_contact_force", "total_contact_torque_about_origin", "grip_lag", "lag_m"],
    "render caption": ["camera", "position", "forward", "up", "right", "fovy_deg", "intrinsics", "fx", "fy", "cx", "cy",
                       "znear", "zfar", "depth", "near_m", "far_m", "segmentation_labels", "images"],
    "solution signals (never in the sensors profile)": ["progress", "cost_to_go", "best_progress", "start_cost_to_go",
                                                        "min_distance", "target_min_distance_to_frame", "goal_reached",
                                                        "escape_direction", "progress_potential"],
}


def pose_metric(a, b) -> float:
    dq = quat_mul(quat_conj(a[3:]), b[3:])
    return float(np.linalg.norm(b[:3] - a[:3]) + W * abs(quat_to_axis_angle(dq)[1]))


def move_delta(args: dict) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([args.get(k, 0.0) for k in ("dx", "dy", "dz")], float)
    q = quat_from_euler_xyz([args.get(k, 0.0) for k in ("droll", "dpitch", "dyaw")])
    return t, q


def agent_text(trial: Path) -> tuple[str, str]:
    """(reasoning + messages, code written / run) from the OpenCode transcript."""
    f = trial / "agent" / "opencode.txt"
    if not f.exists():
        return "", ""
    prose, code = [], []
    for line in f.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = e.get("part") or {}
        if p.get("type") in ("text", "reasoning"):
            prose.append(p.get("text", ""))
        elif p.get("type") == "tool" and (p.get("state") or {}).get("status") == "completed":
            i = p["state"].get("input") or {}
            code.append(str(i.get("command") or i.get("content") or i.get("newString") or ""))
    return "\n".join(prose), "\n".join(code)


def diagnose(trial: Path) -> dict:
    recs = [json.loads(l) for l in (trial / "verifier" / "episode.jsonl").open() if l.strip()]
    hdr = recs[0]
    T = int(hdr["max_actions"])
    events = [r for r in recs if r["type"] in ("action", "observe")]
    acts = [r for r in recs if r["type"] == "action"]
    moves = [(k, r) for k, r in enumerate(events) if r["type"] == "action" and r["tool"] == "move"]
    probes = []
    for (k1, m1), (k2, m2) in zip(moves[:-1], moves[1:]):
        obs_between = [e for e in events[k1 + 1:k2] if e["type"] == "observe"]
        t1, q1 = move_delta(m1["args"])
        t2, q2 = move_delta(m2["args"])
        a1 = abs(quat_to_axis_angle(q1)[1])
        a12 = abs(quat_to_axis_angle(quat_mul(q2, q1))[1])
        n1 = np.linalg.norm(t1)
        if (n1 > 1e-4 or a1 > 1e-3) and np.linalg.norm(t1 + t2) <= 0.25 * max(n1, 1e-9) + 1e-4 and a12 <= 0.25 * max(a1, 1e-9) + 1e-3:
            probes.append({"moves": [m1["i"], m2["i"]], "observations_between": [e["tool"] for e in obs_between]})
    exploratory = sorted({i for p in probes for i in p["moves"]})
    n_moves = len(moves)
    poses = [Q0] + [np.asarray(r["rel_pose"], float) for r in acts]
    L = sum(pose_metric(a, b) for a, b in zip(poses[:-1], poses[1:]))
    wall = [r["wall_time"] for r in acts]
    gaps = np.diff(wall) if len(wall) > 1 else np.array([])
    prose, code = agent_text(trial)
    fields = {}
    for group, keys in STATE_FIELDS.items():
        fields[group] = {k: {"code": len(re.findall(rf"\b{re.escape(k)}\b", code)), "reasoning": len(re.findall(rf"\b{re.escape(k)}\b", prose))}
                         for k in keys}
        fields[group] = {k: v for k, v in fields[group].items() if v["code"] or v["reasoning"]}
    profile = (hdr.get("config") or {}).get("observation", {}).get("profile", "full (legacy interface)")
    final = next((r for r in recs if r["type"] == "final"), {})
    obs = {}
    for r in events:
        if r["type"] == "observe":
            obs[r["tool"]] = obs.get(r["tool"], 0) + 1
    return {
        "trial": trial.name, "observation_profile": profile,
        "solution_signals_exposed": profile not in ("sensors", "planner"),
        "score": final.get("score"), "success": final.get("success"), "actions": len(acts), "max_actions": T,
        "a_over_T": round(len(acts) / T, 4), "moves": n_moves,
        "blocked_moves": sum(bool((r.get("result") or {}).get("blocked")) for r in acts),
        "probe_then_reverse_pairs": len(probes), "probe_examples": probes[:5],
        "exploratory_moves": len(exploratory), "committed_moves": n_moves - len(exploratory),
        "observations": obs,
        "observations_between_actions": sum(1 for a, b in zip(events[:-1], events[1:]) if a["type"] == "action" and b["type"] == "observe"),
        "action_burst": {"median_gap_s": round(float(np.median(gaps)), 2) if gaps.size else None,
                         "max_gap_s": round(float(gaps.max()), 1) if gaps.size else None,
                         "span_s": round(wall[-1] - wall[0], 1) if wall else None},
        "path_length_se3": round(L, 4),
        "fields_referenced": fields,
        "code_chars": len(code), "reasoning_chars": len(prose),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = [diagnose(t) for t in a.trials]
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    for d in out:
        print(json.dumps({k: v for k, v in d.items() if k not in ("fields_referenced", "probe_examples")}))


if __name__ == "__main__":
    main()
